#!/usr/bin/env python3
"""
Train a Dash 1D CNN to regress spectroscopic redshift (MSE).

Observed-frame preprocessing (no z in the spectrum pipeline). Same ASCII
splits JSON as the matching classifier (`--splits-json`). Does not modify
`dash_retrain.py`.

Usage:
  python zmodel_training/dash_retrain_redshift.py \\
      --splits-json data/wiserep/henna_matched_split_noz_seed36.json \\
      --seed 0 \\
      --out-dir data/pre_trained_models/henna_matched_split_noz_redshift/split_36

  python zmodel_training/dash_retrain_redshift.py \\
      --target logz \\
      --splits-json data/wiserep/henna_matched_split_noz_seed36.json \\
      --seed 0 \\
      --out-dir data/pre_trained_models/henna_matched_split_noz_logz/split_36
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

import constants as const
import dash_retrain
import helpers as helpers
from cache_dash_preprocessed import model_input_from_cache, resolve_flux_cache

for _name in ("spectrum_io", "dash_preprocess"):
    logging.getLogger(_name).setLevel(logging.CRITICAL)

Z_MAX = 6.0
Z_FLOOR = 1e-4
TARGET_Z = "z"
TARGET_LOGZ = "logz"


def task_name_for_target(target: str) -> str:
    return "redshift_log_mse" if target == TARGET_LOGZ else "redshift_mse"


def to_train_target(z: torch.Tensor, target: str, z_floor: float = Z_FLOOR) -> torch.Tensor:
    if target == TARGET_LOGZ:
        return torch.log(z.clamp(min=float(z_floor)))
    return z


def train_target_to_z(
    pred: torch.Tensor,
    target: str,
    *,
    z_floor: float = Z_FLOOR,
    z_max: float = Z_MAX,
) -> torch.Tensor:
    if target == TARGET_LOGZ:
        lo = math.log(float(z_floor))
        hi = math.log(float(z_max))
        return torch.exp(pred.clamp(min=lo, max=hi))
    return pred


def _jsonify_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, val in metrics.items():
        if isinstance(val, dict):
            out[key] = _jsonify_metrics(val)
        elif isinstance(val, float) and not np.isfinite(val):
            out[key] = None
        else:
            out[key] = val
    return out


def _parse_redshift(raw: object) -> Optional[float]:
    try:
        z = float(str(raw).strip())
    except (TypeError, ValueError):
        return None
    if not np.isfinite(z) or z <= 0.0 or z > Z_MAX:
        return None
    return float(z)


class WISeREPRedshiftDataset(Dataset):
    """ASCII spectra → observed-frame DASH vector; target is spectroscopic z."""

    def __init__(
        self,
        filenames: List[str],
        spectra_dir: Path,
        metadata: Dict[str, Dict[str, str]],
        target_length: int = const.TARGET_LENGTH,
        flux_cache=None,
    ):
        self.spectra_dir = spectra_dir
        self.target_length = target_length
        self.flux_cache = flux_cache
        self.samples: List[Tuple[str, float]] = []
        skipped_cache = 0
        for fname in filenames:
            meta = metadata.get(fname)
            if meta is None:
                continue
            z_val = _parse_redshift(meta.get("redshift", ""))
            if z_val is None:
                continue
            if flux_cache is not None and fname not in flux_cache:
                skipped_cache += 1
                continue
            self.samples.append((fname, z_val))
        if flux_cache is not None:
            print(
                f"[cache] redshift dataset kept {len(self.samples)} / "
                f"{len(self.samples) + skipped_cache} ascii "
                f"(dropped {skipped_cache} not in cache)",
                flush=True,
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, float]]:
        fname, z_val = self.samples[idx]
        if self.flux_cache is not None:
            processed = model_input_from_cache(
                self.flux_cache.flux_row(fname),
                z_val,
                has_redshift=False,
            )
            return torch.from_numpy(processed), z_val
        result = helpers.load_spectrum(self.spectra_dir / fname)
        if result is None:
            return None
        wave, flux = result
        processed = helpers.preprocess_spectrum(wave, flux, None, self.target_length)
        if processed is None:
            return None
        processed = np.concatenate([processed, [0.0]])
        return torch.from_numpy(processed.astype(np.float32)), z_val


def collate_redshift(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.tensor(ys, dtype=torch.float32)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    resid = y_pred - y_true
    mse = float(np.mean(resid ** 2)) if y_true.size else float("nan")
    mae = float(np.mean(np.abs(resid))) if y_true.size else float("nan")
    rmse = float(np.sqrt(mse)) if np.isfinite(mse) else float("nan")
    if y_true.size:
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        r2 = float("nan") if ss_tot <= 0.0 else 1.0 - float(np.sum(resid ** 2)) / ss_tot
    else:
        r2 = float("nan")
    return {"n": int(y_true.size), "mse": mse, "mae": mae, "rmse": rmse, "r2": r2}


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    target: str = TARGET_Z,
    z_floor: float = Z_FLOOR,
) -> Dict[str, Any]:
    model.eval()
    y_tgt_all: List[np.ndarray] = []
    pred_tgt_all: List[np.ndarray] = []
    y_lin_all: List[np.ndarray] = []
    pred_lin_all: List[np.ndarray] = []
    total_loss = 0.0
    total = 0
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            x, y_lin = batch
            x, y_lin = x.to(device), y_lin.to(device)
            pred_raw = model(x).squeeze(-1)
            y_tgt = to_train_target(y_lin, target, z_floor)
            loss = criterion(pred_raw, y_tgt)
            pred_lin = train_target_to_z(pred_raw, target, z_floor=z_floor, z_max=Z_MAX)
            bs = int(y_lin.size(0))
            total_loss += loss.item() * bs
            total += bs
            y_tgt_all.append(y_tgt.detach().cpu().numpy())
            pred_tgt_all.append(pred_raw.detach().cpu().numpy())
            y_lin_all.append(y_lin.detach().cpu().numpy())
            pred_lin_all.append(pred_lin.detach().cpu().numpy())
    y_tgt_np = np.concatenate(y_tgt_all) if y_tgt_all else np.zeros((0,), dtype=np.float64)
    pred_tgt_np = np.concatenate(pred_tgt_all) if pred_tgt_all else np.zeros((0,), dtype=np.float64)
    y_lin_np = np.concatenate(y_lin_all) if y_lin_all else np.zeros((0,), dtype=np.float64)
    pred_lin_np = np.concatenate(pred_lin_all) if pred_lin_all else np.zeros((0,), dtype=np.float64)
    metrics: Dict[str, Any] = regression_metrics(y_tgt_np, pred_tgt_np)
    metrics["loss"] = total_loss / max(total, 1)
    if target == TARGET_LOGZ:
        metrics["linear_z"] = regression_metrics(y_lin_np, pred_lin_np)
    return metrics


def make_loader(
    filenames: List[str],
    metadata,
    device: torch.device,
    *,
    shuffle: bool,
    flux_cache=None,
) -> DataLoader:
    ds = WISeREPRedshiftDataset(
        filenames, const.SPECTRA_DIR, metadata, flux_cache=flux_cache
    )
    return DataLoader(
        ds,
        batch_size=const.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_redshift,
        pin_memory=(device.type == "cuda"),
    )


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    out_dir: Path,
    *,
    target: str = TARGET_Z,
    z_floor: float = Z_FLOOR,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "model.pth"
    optimizer = torch.optim.Adam(model.parameters(), lr=const.LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    criterion = nn.MSELoss()
    best_val = float("inf")
    best_epoch = -1
    no_improve = 0
    loss_history: List[List[float]] = []
    loss_label = "MSE(ln z)" if target == TARGET_LOGZ else "MSE"

    for epoch in range(1, const.EPOCHS + 1):
        model.train()
        epoch_loss, epoch_n = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{const.EPOCHS}", leave=True, ncols=100)
        for batch in pbar:
            if batch is None:
                continue
            x, y = batch
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x).squeeze(-1)
            y_tgt = to_train_target(y, target, z_floor)
            loss = criterion(pred, y_tgt)
            loss.backward()
            optimizer.step()
            bs = int(y.size(0))
            epoch_loss += loss.item() * bs
            epoch_n += bs
            pbar.set_postfix(mse=f"{epoch_loss / max(epoch_n, 1):.6f}")

        train_mse = epoch_loss / max(epoch_n, 1)
        val_m = evaluate(model, val_loader, criterion, device, target=target, z_floor=z_floor)
        scheduler.step(val_m["mse"])
        loss_history.append([float(epoch), round(train_mse, 6), round(val_m["mse"], 6)])
        if target == TARGET_LOGZ:
            lin = val_m["linear_z"]
            print(
                f"val {loss_label} {val_m['mse']:.6f}  linear-z MAE {lin['mae']:.6f} "
                f"RMSE {lin['rmse']:.6f} R2 {lin['r2']:.4f} n={val_m['n']}"
            )
        else:
            print(
                f"val MSE {val_m['mse']:.6f} MAE {val_m['mae']:.6f} "
                f"R2 {val_m['r2']:.4f} n={val_m['n']}"
            )
        if val_m["mse"] < best_val:
            best_val = val_m["mse"]
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), best_path)
            perf = {
                "best_epoch": best_epoch,
                "task": task_name_for_target(target),
                "target": target,
                "val": _jsonify_metrics(val_m),
                "loss_by_epoch": loss_history,
            }
            if target == TARGET_LOGZ:
                perf["z_floor"] = z_floor
                perf["log"] = "ln"
            (out_dir / "model_performance.json").write_text(json.dumps(perf, indent=2))
            print(f"New best model saved (val_mse={best_val:.6f})")
        else:
            no_improve += 1
            print(f"No improvement for {no_improve}/{const.EARLY_STOP_PATIENCE} epochs")
            if no_improve >= const.EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch}; best epoch {best_epoch}")
                break
    return best_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Dash 1D CNN redshift regressor (MSE, observed-frame).")
    parser.add_argument("--splits-json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0, help="Training RNG seed (not the data-split seed).")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--target",
        choices=(TARGET_Z, TARGET_LOGZ),
        default=TARGET_Z,
        help="Regression target: linear z (default) or ln(max(z, --z-floor)).",
    )
    parser.add_argument(
        "--z-floor",
        type=float,
        default=Z_FLOOR,
        help=f"Lower clamp on z before log (logz only; default: {Z_FLOOR}).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Observed-frame DASH cache dir (default: data/wiserep/dash_preprocessed/noz).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Read ASCII and preprocess on the fly even if a cache exists.",
    )
    args = parser.parse_args()

    helpers.set_seed(args.seed)
    out_dir = args.out_dir.resolve()
    splits_path = args.splits_json.resolve()
    splits = helpers.load_json(splits_path)
    metadata = helpers.load_metadata(const.METADATA_CSV)
    device = helpers.get_device()
    target = str(args.target)
    z_floor = float(args.z_floor)
    print(
        f"task={task_name_for_target(target)}  target={target}  z_floor={z_floor}  "
        f"has_redshift=False  train_rng_seed={args.seed}  out_dir={out_dir}"
    )
    print(
        f"Splits ({splits_path.name}): train={len(splits.get('train', []))}  "
        f"val={len(splits.get('val', []))}  test={len(splits.get('test', []))}"
    )

    flux_cache = resolve_flux_cache(
        False,
        cache_dir=args.cache_dir,
        disable=bool(args.no_cache),
    )
    train_loader = make_loader(
        list(splits["train"]), metadata, device, shuffle=True, flux_cache=flux_cache
    )
    val_loader = make_loader(
        list(splits["val"]), metadata, device, shuffle=False, flux_cache=flux_cache
    )
    print(f"Effective sizes: train={len(train_loader.dataset)}  val={len(val_loader.dataset)}")

    model = dash_retrain.DashCNN1D(input_length=const.TARGET_LENGTH, num_classes=1).to(device)
    best_path = train(
        model, train_loader, val_loader, device, out_dir, target=target, z_floor=z_floor
    )

    config = {
        "run_id": f"split_{splits.get('data_split_seed', splits.get('seed'))}",
        "task": task_name_for_target(target),
        "target": target,
        "has_redshift": False,
        "target_length": const.TARGET_LENGTH,
        "num_outputs": 1,
        "epochs": const.EPOCHS,
        "batch_size": const.BATCH_SIZE,
        "lr": const.LEARNING_RATE,
        "patience": const.EARLY_STOP_PATIENCE,
        "splits_file": str(splits_path),
        "seed": args.seed,
        "data_split_seed": splits.get("data_split_seed", splits.get("seed")),
        "data_mode": "ascii_redshift_log_mse" if target == TARGET_LOGZ else "ascii_redshift_mse",
        "z_max": Z_MAX,
        "z_floor": z_floor,
        "dash_cache": str(flux_cache.cache_dir) if flux_cache is not None else None,
    }
    if target == TARGET_LOGZ:
        config["log"] = "ln"
    (out_dir / "training_config.json").write_text(json.dumps(config, indent=2))
    print(f"Saved: {best_path}")
    print(f"  Config: {out_dir / 'training_config.json'}")


if __name__ == "__main__":
    main()
