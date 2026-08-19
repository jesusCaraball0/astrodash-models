#!/usr/bin/env python3
"""
True vs predicted spectroscopic redshift for latent MLP and/or Dash 1D CNN
redshift regressors.

Discovers `split_*` run dirs under a root (or `try_*/split_*` one level down).

  python zmodel_training/redshift_pred_plots.py \\
      data/pre_trained_models/daep_latent_redshift/try_5_noz

  python zmodel_training/redshift_pred_plots.py \\
      data/pre_trained_models/daep_latent_logz/try_5_noz

  python zmodel_training/redshift_pred_plots.py \\
      data/pre_trained_models/henna_matched_split_noz_redshift

  python zmodel_training/redshift_pred_plots.py \\
      data/pre_trained_models/henna_matched_split_noz_logz
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
WISEREP_DIR = PROJECT_ROOT / "WiserepData"
for path in (PROJECT_ROOT, SCRIPT_DIR, WISEREP_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import constants as const
import dash_retrain
import dash_retrain_redshift as dash_z
import helpers as helpers
from cache_dash_preprocessed import resolve_flux_cache
from train_latent import DROPOUT, LatentClassifier, load_assignment_indices_from_dir, load_latent_and_meta
from train_latent_redshift import (
    REDSHIFT_COLUMN,
    TARGET_LOGZ,
    LatentRedshiftDataset,
    collate_latent_redshift,
    ensure_redshift_column,
    numpy_train_target_to_z,
    regression_metrics,
    resolve_target,
    split_and_filter_redshift,
)
from TwinsModel_Wiserep import device_from_str
from TwinsTrain_Wiserep import to_device

SPLIT_DIR_RE = re.compile(r"^split_(\d+)$")
LATENT_CKPT = "regressor_best.pt"
CNN_CKPT = "model.pth"


def discover_run_dirs(root: Path) -> list[Path]:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    def _collect(parent: Path) -> list[tuple[int, Path]]:
        found: list[tuple[int, Path]] = []
        for p in parent.iterdir():
            if not p.is_dir():
                continue
            m = SPLIT_DIR_RE.fullmatch(p.name)
            if m and _kind(p) is not None:
                found.append((int(m.group(1)), p))
        found.sort(key=lambda x: x[0])
        return found

    found = _collect(root)
    if not found:
        nested: list[tuple[int, Path]] = []
        for sub in sorted(p for p in root.iterdir() if p.is_dir()):
            nested.extend(_collect(sub))
        found = nested
    if not found:
        raise FileNotFoundError(
            f"No split_* redshift runs with {LATENT_CKPT} or {CNN_CKPT} under {root}"
        )
    return [p for _, p in found]


def _kind(run_dir: Path) -> str | None:
    if (run_dir / LATENT_CKPT).is_file():
        return "latent"
    cfg_path = run_dir / "training_config.json"
    if (run_dir / CNN_CKPT).is_file() and cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        task = str(cfg.get("task") or "")
        if (
            task in ("redshift_mse", "redshift_log_mse")
            or cfg.get("target") in ("z", "logz")
            or int(cfg.get("num_outputs", 0) or 0) == 1
        ):
            return "cnn"
    return None


def _predict_latent(run_dir: Path, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    ckpt = torch.load(run_dir / LATENT_CKPT, map_location=device, weights_only=False)
    cfg = ckpt.get("cfg") or json.loads((run_dir / "cfg_used.json").read_text(encoding="utf-8"))
    target, z_floor = resolve_target({**cfg, "target": ckpt.get("target") or cfg.get("target")})
    latent_dir = Path(cfg["latent_npz"]).resolve().parent
    meta, z, _, _ = load_latent_and_meta(latent_dir)
    meta = ensure_redshift_column(meta)
    raw_tr, raw_va, raw_te, split_src = load_assignment_indices_from_dir(latent_dir, meta)
    _, _, test_idx = split_and_filter_redshift(
        meta, train_idx=raw_tr, val_idx=raw_va, test_idx=raw_te, split_tag=split_src
    )
    embed_dim = int(np.prod(z.shape[1:]))
    head_hidden = int(cfg["ff_dim"]) if cfg.get("ff_dim") is not None else 512
    head_dropout = float(cfg.get("dropout", DROPOUT))
    model = LatentClassifier(
        embed_dim=embed_dim, n_cls=1, head_hidden=head_hidden, head_dropout=head_dropout
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    loader = DataLoader(
        LatentRedshiftDataset(meta, z, test_idx, REDSHIFT_COLUMN),
        batch_size=int(cfg.get("batch_size", 16)),
        shuffle=False,
        collate_fn=collate_latent_redshift,
        num_workers=0,
    )
    ys: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            y = batch.pop("y").to(device)
            pred = model(batch).squeeze(-1)
            ys.append(y.detach().cpu().numpy())
            preds.append(pred.detach().cpu().numpy())
    y_true = np.concatenate(ys)
    y_pred = numpy_train_target_to_z(np.concatenate(preds), target, z_floor=z_floor)
    return y_true, y_pred


def _predict_cnn(run_dir: Path, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    cfg = json.loads((run_dir / "training_config.json").read_text(encoding="utf-8"))
    splits = helpers.load_json(Path(cfg["splits_file"]))
    test_files = list(splits.get("test", []))
    if not test_files:
        raise RuntimeError(f"No test filenames in {cfg['splits_file']}")
    metadata = helpers.load_metadata(const.METADATA_CSV)
    cache_dir = Path(cfg["dash_cache"]) if cfg.get("dash_cache") else None
    flux_cache = resolve_flux_cache(False, cache_dir=cache_dir, disable=False)
    loader = DataLoader(
        dash_z.WISeREPRedshiftDataset(
            test_files, const.SPECTRA_DIR, metadata, flux_cache=flux_cache
        ),
        batch_size=int(cfg.get("batch_size", const.BATCH_SIZE)),
        shuffle=False,
        num_workers=0,
        collate_fn=dash_z.collate_redshift,
    )
    model = dash_retrain.DashCNN1D(input_length=const.TARGET_LENGTH, num_classes=1).to(device)
    state = torch.load(run_dir / CNN_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    ys: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            x, y = batch
            x, y = x.to(device), y.to(device)
            pred = model(x).squeeze(-1)
            ys.append(y.detach().cpu().numpy())
            preds.append(pred.detach().cpu().numpy())
    if not ys:
        raise RuntimeError(f"No CNN test predictions from {run_dir}")
    target, z_floor = resolve_target(cfg)
    y_true = np.concatenate(ys)
    y_pred = numpy_train_target_to_z(np.concatenate(preds), target, z_floor=z_floor)
    return y_true, y_pred


def predict_run(run_dir: Path, device: torch.device) -> tuple[str, np.ndarray, np.ndarray]:
    kind = _kind(run_dir)
    if kind == "latent":
        y_true, y_pred = _predict_latent(run_dir, device)
    elif kind == "cnn":
        y_true, y_pred = _predict_cnn(run_dir, device)
    else:
        raise RuntimeError(f"Not a redshift run: {run_dir}")
    return kind, y_true.reshape(-1), y_pred.reshape(-1)


def _run_target(run_dir: Path, kind: str) -> str:
    if kind == "latent":
        cfg_path = run_dir / "cfg_used.json"
        if cfg_path.is_file():
            return resolve_target(json.loads(cfg_path.read_text(encoding="utf-8")))[0]
        return "z"
    cfg = json.loads((run_dir / "training_config.json").read_text(encoding="utf-8"))
    return resolve_target(cfg)[0]


def _scatter(
    ax,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    *,
    log_axes: bool = False,
) -> None:
    if log_axes:
        y_true_p = np.log(np.clip(y_true, 1e-8, None))
        y_pred_p = np.log(np.clip(y_pred, 1e-8, None))
        xlabel, ylabel = "True ln z", "Predicted ln z"
        metrics = regression_metrics(y_true_p, y_pred_p)
    else:
        y_true_p, y_pred_p = y_true, y_pred
        xlabel, ylabel = "True redshift", "Predicted redshift"
        metrics = regression_metrics(y_true, y_pred)
    ax.scatter(y_true_p, y_pred_p, s=8, alpha=0.35, linewidths=0, c="#0072B2")
    lo = float(min(y_true_p.min(), y_pred_p.min(), 0.0 if not log_axes else y_true_p.min()))
    hi = float(max(y_true_p.max(), y_pred_p.max()))
    pad = 0.02 * (hi - lo + 1e-6)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"{title}\n"
        f"MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  "
        f"R²={metrics['r2']:.3f}  n={metrics['n']}"
    )
    ax.grid(True, alpha=0.3)


def main() -> None:
    parser = argparse.ArgumentParser(description="True vs predicted redshift plots (latent MLP or Dash CNN).")
    parser.add_argument("root", type=Path, help="Folder with split_* redshift runs.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Plot directory (default: <root>/plots).",
    )
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    run_dirs = discover_run_dirs(root)
    out_dir = (args.out_dir or (root / "plots")).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = device_from_str("auto")
    series: list[tuple[str, np.ndarray, np.ndarray]] = []
    kinds: set[str] = set()
    print(f"Plotting {len(run_dirs)} run(s) from {root}")
    any_logz = False
    for run_dir in run_dirs:
        kind, y_true, y_pred = predict_run(run_dir, device)
        kinds.add(kind)
        target = _run_target(run_dir, kind)
        any_logz = any_logz or target == TARGET_LOGZ
        metrics = regression_metrics(y_true, y_pred)
        extra = ""
        if target == TARGET_LOGZ:
            log_m = regression_metrics(
                np.log(np.clip(y_true, 1e-8, None)),
                np.log(np.clip(y_pred, 1e-8, None)),
            )
            extra = f"  MAE(ln z)={log_m['mae']:.4f}  R2(ln z)={log_m['r2']:.4f}"
        print(
            f"  {run_dir.name} ({kind}, target={target}): n={metrics['n']}  "
            f"MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  R2={metrics['r2']:.4f}"
            f"{extra}"
        )
        series.append((run_dir.name, y_true, y_pred))
        split_png = out_dir / f"{run_dir.name}_z_true_vs_pred.png"
        if split_png.exists() and not args.allow_overwrite:
            raise FileExistsError(f"Refusing to overwrite {split_png} (pass --allow-overwrite)")
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        _scatter(ax, y_true, y_pred, f"{kind} {run_dir.name}")
        fig.tight_layout()
        fig.savefig(split_png, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {split_png}")
        if target == TARGET_LOGZ:
            log_png = out_dir / f"{run_dir.name}_logz_true_vs_pred.png"
            if log_png.exists() and not args.allow_overwrite:
                raise FileExistsError(f"Refusing to overwrite {log_png} (pass --allow-overwrite)")
            fig, ax = plt.subplots(figsize=(6.5, 6.5))
            _scatter(ax, y_true, y_pred, f"{kind} {run_dir.name} (ln z)", log_axes=True)
            fig.tight_layout()
            fig.savefig(log_png, dpi=160, bbox_inches="tight")
            plt.close(fig)
            print(f"  saved {log_png}")

    if len(series) > 1:
        n = len(series)
        cols = min(2, n)
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 6.2 * rows), squeeze=False)
        for i, (name, y_true, y_pred) in enumerate(series):
            _scatter(axes[i // cols][i % cols], y_true, y_pred, name)
        for j in range(len(series), rows * cols):
            axes[j // cols][j % cols].set_visible(False)
        grid_png = out_dir / "z_true_vs_pred_grid.png"
        if grid_png.exists() and not args.allow_overwrite:
            raise FileExistsError(f"Refusing to overwrite {grid_png} (pass --allow-overwrite)")
        fig.tight_layout()
        fig.savefig(grid_png, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {grid_png}")

        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        y_true = np.concatenate([s[1] for s in series])
        y_pred = np.concatenate([s[2] for s in series])
        label = "latent MLP" if kinds == {"latent"} else ("Dash CNN" if kinds == {"cnn"} else "mixed")
        _scatter(ax, y_true, y_pred, f"{label} (all test splits pooled)")
        pooled_png = out_dir / "z_true_vs_pred_pooled.png"
        if pooled_png.exists() and not args.allow_overwrite:
            raise FileExistsError(f"Refusing to overwrite {pooled_png} (pass --allow-overwrite)")
        fig.tight_layout()
        fig.savefig(pooled_png, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {pooled_png}")
        if any_logz:
            log_pooled = out_dir / "logz_true_vs_pred_pooled.png"
            if log_pooled.exists() and not args.allow_overwrite:
                raise FileExistsError(f"Refusing to overwrite {log_pooled} (pass --allow-overwrite)")
            fig, ax = plt.subplots(figsize=(6.5, 6.5))
            _scatter(ax, y_true, y_pred, f"{label} ln z (all test splits pooled)", log_axes=True)
            fig.tight_layout()
            fig.savefig(log_pooled, dpi=160, bbox_inches="tight")
            plt.close(fig)
            print(f"saved {log_pooled}")


if __name__ == "__main__":
    main()
