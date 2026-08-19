"""Train a latent MLP to regress spectroscopic redshift (MSE).

Parallel to ``train_latent.py``: same frozen Universal embeddings, assignment
splits, flatten-MLP architecture, and training hparams. The head outputs a
scalar and is trained with unweighted MSE on ``redshift_used`` instead of
weighted CE on the 5-class taxonomy.

``--target z`` (default) writes under ``daep_latent_redshift``. ``--target logz``
trains on ``ln(max(z, z_floor))`` and writes under ``daep_latent_logz``.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import time
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from train_latent import (
    BATCH_SIZE,
    DEVICE,
    DROPOUT,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    ITER10_RECIPE,
    LABEL_COLUMN,
    LR,
    TRAIN_RNG_SEED,
    WEIGHT_DECAY,
    LatentClassifier,
    build_cfg,
    load_assignment_indices_from_dir,
    load_latent_and_meta,
    split_and_filter,
    split_run_name,
    try_run_name,
)
from TwinsModel_Wiserep import device_from_str
from TwinsTrain_Wiserep import set_seeds, to_device


WISEREP_DIR = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = WISEREP_DIR.parent
DEFAULT_LATENT_DIRS_OUT_ROOT = (
    _PROJECT_ROOT / "data" / "pre_trained_models" / "daep_latent_redshift"
)
DEFAULT_LOGZ_OUT_ROOT = _PROJECT_ROOT / "data" / "pre_trained_models" / "daep_latent_logz"

REDSHIFT_COLUMN = "Redshift"
REDSHIFT_ALIASES = ("Redshift", "redshift_used", "redshift")
Z_MAX = 6.0
Z_FLOOR = 1e-4
TARGET_Z = "z"
TARGET_LOGZ = "logz"


def default_out_root_for_target(target: str) -> pathlib.Path:
    return DEFAULT_LOGZ_OUT_ROOT if target == TARGET_LOGZ else DEFAULT_LATENT_DIRS_OUT_ROOT


def task_name_for_target(target: str) -> str:
    return "redshift_log_mse" if target == TARGET_LOGZ else "redshift_mse"


def resolve_target(cfg: Dict[str, Any] | None) -> tuple[str, float]:
    cfg = cfg or {}
    raw = cfg.get("target")
    if raw in (TARGET_Z, TARGET_LOGZ):
        target = str(raw)
    elif cfg.get("task") == "redshift_log_mse" or cfg.get("log") == "ln":
        target = TARGET_LOGZ
    else:
        target = TARGET_Z
    return target, float(cfg.get("z_floor", Z_FLOOR))


def to_train_target(
    z: torch.Tensor,
    target: str,
    z_floor: float = Z_FLOOR,
) -> torch.Tensor:
    """Map linear spectroscopic z to the regression target (identity or ln z)."""
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
    """Invert the training target to linear z; clamp log-space preds before exp."""
    if target == TARGET_LOGZ:
        lo = math.log(float(z_floor))
        hi = math.log(float(z_max))
        return torch.exp(pred.clamp(min=lo, max=hi))
    return pred


def numpy_train_target_to_z(
    pred: np.ndarray,
    target: str,
    *,
    z_floor: float = Z_FLOOR,
    z_max: float = Z_MAX,
) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float64)
    if target == TARGET_LOGZ:
        lo = math.log(float(z_floor))
        hi = math.log(float(z_max))
        return np.exp(np.clip(pred, lo, hi))
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


def latent_dirs_out_dir(latent_dir: pathlib.Path, out_root: pathlib.Path) -> pathlib.Path:
    """<out-root>/<try_N>/split_<seed>/."""
    return pathlib.Path(out_root) / try_run_name(latent_dir) / split_run_name(latent_dir)


def ensure_redshift_column(meta: pd.DataFrame) -> pd.DataFrame:
    """Alias Henna ``redshift_used`` onto ``Redshift`` if needed."""
    out = meta
    if REDSHIFT_COLUMN not in out.columns:
        for alias in REDSHIFT_ALIASES[1:]:
            if alias in out.columns:
                out = out.copy()
                out[REDSHIFT_COLUMN] = out[alias]
                break
        else:
            raise KeyError(
                f"Need a redshift column among {REDSHIFT_ALIASES}; have {list(out.columns)[:20]}"
            )
    return out


def redshift_values(meta: pd.DataFrame, col: str = REDSHIFT_COLUMN) -> np.ndarray:
    return pd.to_numeric(meta[col], errors="coerce").to_numpy(dtype=np.float64)


def filter_indices_finite_redshift(
    meta: pd.DataFrame,
    indices: np.ndarray,
    col: str = REDSHIFT_COLUMN,
    *,
    z_max: float = Z_MAX,
) -> np.ndarray:
    """Keep rows with finite spectroscopic z in (0, z_max]."""
    z = redshift_values(meta, col)
    keep: List[int] = []
    for i in np.asarray(indices, dtype=np.int64):
        zi = z[int(i)]
        if np.isfinite(zi) and zi > 0.0 and zi <= z_max:
            keep.append(int(i))
    return np.asarray(keep, dtype=np.int64)


def report_split_redshift_stats(
    meta: pd.DataFrame,
    idx: np.ndarray,
    tag: str,
    col: str = REDSHIFT_COLUMN,
) -> None:
    z = redshift_values(meta, col)[np.asarray(idx, dtype=np.int64)]
    if z.size == 0:
        print(f"[redshift] {tag}  n=0", flush=True)
        return
    print(
        f"[redshift] {tag}  n={z.size}  min={z.min():.5f}  "
        f"median={np.median(z):.5f}  mean={z.mean():.5f}  max={z.max():.5f}",
        flush=True,
    )


def split_and_filter_redshift(
    meta: pd.DataFrame,
    *,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    split_tag: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same class-mapped membership as ``train_latent``, then drop invalid z."""
    train_idx, val_idx, test_idx = split_and_filter(
        meta,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        split_tag=split_tag,
    )
    out: List[np.ndarray] = []
    for tag, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        kept = filter_indices_finite_redshift(meta, idx)
        dropped = int(len(idx) - len(kept))
        if dropped:
            print(
                f"[redshift] {tag} dropped {dropped} / {len(idx)} rows without finite z in (0, {Z_MAX}]",
                flush=True,
            )
        report_split_redshift_stats(meta, kept, tag)
        out.append(kept)
    return out[0], out[1], out[2]


class LatentRedshiftDataset(Dataset):
    def __init__(self, meta: pd.DataFrame, z: np.ndarray, indices: np.ndarray, redshift_col: str):
        self.meta = meta.reset_index(drop=True)
        self.z = z.astype(np.float32, copy=False)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.redshift_col = redshift_col
        self._redshift = redshift_values(self.meta, redshift_col).astype(np.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        row = int(self.indices[i])
        return {
            "idx": torch.tensor(row, dtype=torch.long),
            "z": torch.from_numpy(self.z[row]),
            "y": torch.tensor(float(self._redshift[row]), dtype=torch.float32),
        }


def collate_latent_redshift(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "idx": torch.stack([b["idx"] for b in batch], dim=0),
        "z": torch.stack([b["z"] for b in batch], dim=0),
        "y": torch.stack([b["y"] for b in batch], dim=0),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    resid = y_pred - y_true
    mse = float(np.mean(resid ** 2)) if y_true.size else float("nan")
    mae = float(np.mean(np.abs(resid))) if y_true.size else float("nan")
    rmse = float(np.sqrt(mse)) if np.isfinite(mse) else float("nan")
    if y_true.size:
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        r2 = float("nan") if ss_tot <= 0.0 else 1.0 - ss_res / ss_tot
    else:
        r2 = float("nan")
    return {
        "n": int(y_true.size),
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


def evaluate_redshift(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    metrics_title: str = "Validation metrics",
    print_summary: bool = True,
    target: str = TARGET_Z,
    z_floor: float = Z_FLOOR,
    z_max: float = Z_MAX,
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
            batch = to_device(batch, device)
            y_lin = batch.pop("y").to(device)
            pred_raw = model(batch).squeeze(-1)
            y_tgt = to_train_target(y_lin, target, z_floor)
            loss = criterion(pred_raw, y_tgt)
            pred_lin = train_target_to_z(pred_raw, target, z_floor=z_floor, z_max=z_max)
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
    if print_summary:
        print(f"\n{metrics_title}:", flush=True)
        if target == TARGET_LOGZ:
            lin = metrics["linear_z"]
            print(
                f"  MSE(ln z): {metrics['mse']:.6f}  RMSE(ln z): {metrics['rmse']:.6f}  "
                f"MAE(ln z): {metrics['mae']:.6f}  R2(ln z): {metrics['r2']:.4f}  n={metrics['n']}",
                flush=True,
            )
            print(
                f"  linear-z MAE: {lin['mae']:.6f}  RMSE: {lin['rmse']:.6f}  "
                f"R2: {lin['r2']:.4f}",
                flush=True,
            )
        else:
            print(
                f"  MSE: {metrics['mse']:.6f}  RMSE: {metrics['rmse']:.6f}  "
                f"MAE: {metrics['mae']:.6f}  R2: {metrics['r2']:.4f}  n={metrics['n']}",
                flush=True,
            )
    return metrics


def latent_checkpoint_payload(model: LatentClassifier, cfg: Dict[str, Any]) -> Dict[str, Any]:
    target, z_floor = resolve_target(cfg)
    payload: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "head_state_dict": model.head.state_dict(),
        "cfg": cfg,
        "task": task_name_for_target(target),
        "target": target,
        "redshift_column": REDSHIFT_COLUMN,
        "label_column": LABEL_COLUMN,
    }
    if target == TARGET_LOGZ:
        payload["z_floor"] = z_floor
        payload["log"] = "ln"
    return payload


def build_performance_json(
    *,
    best_epoch: int,
    val: Dict[str, Any],
    test: Dict[str, Any] | None = None,
    loss_by_epoch: List[List[Any]] | None = None,
    target: str = TARGET_Z,
    z_floor: float = Z_FLOOR,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "best_epoch": best_epoch,
        "task": task_name_for_target(target),
        "target": target,
        "val": _jsonify_metrics(val),
    }
    if target == TARGET_LOGZ:
        out["z_floor"] = z_floor
        out["log"] = "ln"
    if test is not None:
        out["test"] = _jsonify_metrics(test)
    if loss_by_epoch is not None:
        out["loss_by_epoch"] = loss_by_epoch
    return out


def train_latent_regressor(
    model: LatentClassifier,
    tr_load: DataLoader,
    va_load: DataLoader,
    te_load: DataLoader,
    device: torch.device,
    out_dir: pathlib.Path,
    cfg: Dict[str, Any],
    *,
    epochs: int = EPOCHS,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE,
    target: str = TARGET_Z,
    z_floor: float = Z_FLOOR,
) -> pathlib.Path:
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    target, z_floor = resolve_target({"target": target, "z_floor": z_floor, **cfg})

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_ckpt_path = out_dir / "regressor_best.pt"
    final_ckpt_path = out_dir / "regressor.pt"
    perf_path = out_dir / "model_performance.json"

    loss_label = "MSE(ln z)" if target == TARGET_LOGZ else "MSE"
    print(
        f"[train] epochs={epochs}  early_stop_patience={early_stopping_patience}  "
        f"lr={lr}  wd={weight_decay}  loss={loss_label}  target={target}  "
        f"z_floor={z_floor}  out_dir={out_dir}",
        flush=True,
    )

    def _mean(xs: List[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    loss_by_epoch: List[List[Any]] = []
    best_val_mse = float("inf")
    best_epoch = 0
    best_val: Dict[str, Any] | None = None
    best_test: Dict[str, Any] | None = None
    epochs_no_improve = 0

    eval_kw = dict(target=target, z_floor=z_floor, z_max=float(cfg.get("z_max", Z_MAX)))

    for ep in range(1, epochs + 1):
        t_ep = time.perf_counter()
        model.train()
        tl: List[float] = []
        pbar = tqdm(tr_load, desc=f"Epoch {ep}/{epochs} train", leave=False, dynamic_ncols=True)
        for batch in pbar:
            batch = to_device(batch, device)
            y = batch.pop("y").to(device)
            pred = model(batch).squeeze(-1)
            y_tgt = to_train_target(y, target, z_floor)
            loss = loss_fn(pred, y_tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tl.append(loss.item())
            pbar.set_postfix(mse=f"{np.mean(tl):.6f}")

        val_m = evaluate_redshift(
            model,
            va_load,
            loss_fn,
            device,
            metrics_title="Validation metrics",
            print_summary=True,
            **eval_kw,
        )
        test_m = evaluate_redshift(
            model,
            te_load,
            loss_fn,
            device,
            metrics_title="Test metrics",
            print_summary=False,
            **eval_kw,
        )

        train_mse = _mean(tl)
        loss_by_epoch.append(
            [
                ep,
                float(train_mse) if np.isfinite(train_mse) else None,
                float(val_m["mse"]) if np.isfinite(val_m["mse"]) else None,
            ]
        )

        dt = time.perf_counter() - t_ep
        if target == TARGET_LOGZ:
            val_lin = val_m["linear_z"]
            test_lin = test_m["linear_z"]
            print(
                f"epoch {ep}/{epochs} done in {dt:.1f}s  "
                f"train MSE(ln z) {train_mse:.6f}  |  "
                f"val MSE(ln z) {val_m['mse']:.6f} MAE_z {val_lin['mae']:.6f} R2_z {val_lin['r2']:.4f}  |  "
                f"test MSE(ln z) {test_m['mse']:.6f} MAE_z {test_lin['mae']:.6f} R2_z {test_lin['r2']:.4f}",
                flush=True,
            )
        else:
            print(
                f"epoch {ep}/{epochs} done in {dt:.1f}s  "
                f"train MSE {train_mse:.6f}  |  "
                f"val MSE {val_m['mse']:.6f} MAE {val_m['mae']:.6f} R2 {val_m['r2']:.4f}  |  "
                f"test MSE {test_m['mse']:.6f} MAE {test_m['mae']:.6f} R2 {test_m['r2']:.4f}",
                flush=True,
            )

        if val_m["mse"] < best_val_mse:
            epochs_no_improve = 0
            best_val_mse = float(val_m["mse"])
            best_epoch = ep
            best_val = dict(val_m)
            best_test = dict(test_m)
            perf = build_performance_json(
                best_epoch=best_epoch,
                val=best_val,
                test=best_test,
                loss_by_epoch=[list(r) for r in loss_by_epoch],
                target=target,
                z_floor=z_floor,
            )
            perf_path.write_text(json.dumps(perf, indent=2), encoding="utf-8")

            ckpt = latent_checkpoint_payload(model, cfg)
            ckpt["best_epoch"] = best_epoch
            ckpt["val_mse"] = best_val_mse
            ckpt["val_mae"] = float(val_m["mae"])
            ckpt["val_rmse"] = float(val_m["rmse"])
            ckpt["val_r2"] = float(val_m["r2"])
            if target == TARGET_LOGZ:
                ckpt["val_mae_z"] = float(val_m["linear_z"]["mae"])
                ckpt["val_rmse_z"] = float(val_m["linear_z"]["rmse"])
                ckpt["val_r2_z"] = float(val_m["linear_z"]["r2"])
            torch.save(ckpt, best_ckpt_path)
            best_extra = (
                f" val_mae_z={val_m['linear_z']['mae']:.6f}"
                if target == TARGET_LOGZ
                else f" val_mae={val_m['mae']:.6f}"
            )
            print(
                f"[best] epoch={best_epoch} val_mse={best_val_mse:.6f}{best_extra} -> "
                f"{perf_path.name} + {best_ckpt_path.name}",
                flush=True,
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(
                    f"[early stop] no val MSE improvement for {early_stopping_patience} epochs "
                    f"(best epoch {best_epoch})",
                    flush=True,
                )
                break

    if best_val is not None:
        perf = build_performance_json(
            best_epoch=best_epoch,
            val=best_val,
            test=best_test,
            loss_by_epoch=[list(r) for r in loss_by_epoch],
            target=target,
            z_floor=z_floor,
        )
        perf_path.write_text(json.dumps(perf, indent=2), encoding="utf-8")
        print(
            f"[done] refreshed {perf_path.name} with {len(loss_by_epoch)} epochs of loss history",
            flush=True,
        )

    print("\n[saving] last-epoch checkpoint ...", flush=True)
    torch.save(latent_checkpoint_payload(model, cfg), final_ckpt_path)
    cfg_path = out_dir / "cfg_used.json"
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print("[done] wrote", final_ckpt_path.resolve(), flush=True)
    print("[done] wrote", cfg_path.resolve(), flush=True)
    return best_ckpt_path


def run_one(
    seed: int,
    *,
    meta: pd.DataFrame,
    z: np.ndarray,
    latent_npz: pathlib.Path,
    meta_csv: pathlib.Path,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    cfg_json: pathlib.Path,
    out_dir: pathlib.Path,
    run_id: str | None = None,
    split_source: str | None = None,
    train_overrides: Dict[str, Any] | None = None,
    target: str = TARGET_Z,
    z_floor: float = Z_FLOOR,
) -> None:
    print(f"=== train_latent_redshift start | SEED={seed} target={target} ===", flush=True)
    set_seeds(seed)
    resolved_out = out_dir.resolve()

    cfg = build_cfg(
        z,
        latent_npz=latent_npz,
        meta_csv=meta_csv,
        cfg_json=cfg_json,
        seed=seed,
        run_id=run_id,
        split_source=split_source,
        train_overrides=train_overrides,
    )
    cfg["classifier_kind"] = (
        "latent_flatten_mlp_redshift_log_mse"
        if target == TARGET_LOGZ
        else "latent_flatten_mlp_redshift_mse"
    )
    cfg["task"] = task_name_for_target(target)
    cfg["loss"] = "mse"
    cfg["target"] = target
    cfg["redshift_column"] = REDSHIFT_COLUMN
    cfg["z_max"] = Z_MAX
    cfg["z_floor"] = float(z_floor)
    if target == TARGET_LOGZ:
        cfg["log"] = "ln"

    batch_size = int(cfg.get("batch_size", BATCH_SIZE))
    lr = float(cfg.get("lr", LR))
    weight_decay = float(cfg.get("weight_decay", WEIGHT_DECAY))
    epochs = int(cfg.get("epochs", EPOCHS))
    early_stopping_patience = int(cfg.get("early_stopping_patience", EARLY_STOPPING_PATIENCE))
    embed_dim = int(np.prod(z.shape[1:]))
    head_hidden = int(cfg["ff_dim"]) if cfg.get("ff_dim") is not None else 512
    head_dropout = float(cfg.get("dropout", DROPOUT))

    train_ds = LatentRedshiftDataset(meta, z, train_idx, REDSHIFT_COLUMN)
    val_ds = LatentRedshiftDataset(meta, z, val_idx, REDSHIFT_COLUMN)
    test_ds = LatentRedshiftDataset(meta, z, test_idx, REDSHIFT_COLUMN)

    train_load = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_latent_redshift,
        num_workers=0,
    )
    val_load = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_latent_redshift,
        num_workers=0,
    )
    test_load = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_latent_redshift,
        num_workers=0,
    )

    print(f"dataset rows: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}", flush=True)
    print(f"[out] {resolved_out}", flush=True)

    device = device_from_str(DEVICE)
    model = LatentClassifier(
        embed_dim=embed_dim,
        n_cls=1,
        head_hidden=head_hidden,
        head_dropout=head_dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[model] device={device} embed_dim={embed_dim} hidden={head_hidden} dropout={head_dropout} out=1",
        flush=True,
    )
    print(f"[model] parameters total={n_params:,}", flush=True)

    best_path = train_latent_regressor(
        model,
        train_load,
        val_load,
        test_load,
        device,
        resolved_out,
        cfg,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        early_stopping_patience=early_stopping_patience,
        target=target,
        z_floor=z_floor,
    )
    print("[done] best checkpoint:", best_path.resolve(), flush=True)
    print(f"=== train_latent_redshift finished | SEED={seed} target={target} ===", flush=True)


def run_latent_dirs(
    latent_dirs: Sequence[pathlib.Path],
    *,
    out_root: pathlib.Path,
    train_seed: int,
    train_overrides: Dict[str, Any] | None = None,
    target: str = TARGET_Z,
    z_floor: float = Z_FLOOR,
) -> None:
    """Train one redshift regressor per directory; write under out_root/<try_N>/split_<seed>/."""
    out_root = pathlib.Path(out_root).resolve()
    for latent_dir in latent_dirs:
        latent_dir = pathlib.Path(latent_dir).resolve()
        run_name = split_run_name(latent_dir)
        out_dir = latent_dirs_out_dir(latent_dir, out_root)
        cfg_json = latent_dir / "cfg_used.json"
        meta, z, latent_npz, meta_csv = load_latent_and_meta(latent_dir)
        meta = ensure_redshift_column(meta)
        raw_tr, raw_va, raw_te, split_source = load_assignment_indices_from_dir(latent_dir, meta)
        train_idx, val_idx, test_idx = split_and_filter_redshift(
            meta,
            train_idx=raw_tr,
            val_idx=raw_va,
            test_idx=raw_te,
            split_tag=split_source,
        )
        print(
            f"=== latent-dir redshift job | dir={latent_dir} out={out_dir} "
            f"train_seed={train_seed} target={target} ===",
            flush=True,
        )
        run_one(
            train_seed,
            meta=meta,
            z=z,
            latent_npz=latent_npz,
            meta_csv=meta_csv,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            cfg_json=cfg_json,
            out_dir=out_dir,
            run_id=run_name,
            split_source=split_source,
            train_overrides=train_overrides,
            target=target,
            z_floor=z_floor,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train latent MLP redshift regressors (MSE) on Henna Universal latents."
    )
    parser.add_argument(
        "--latent-dirs",
        nargs="+",
        type=pathlib.Path,
        required=True,
        help=(
            "One or more latent directories (each with latent_raw_z_best.npz + meta + assignment). "
            "Trains exactly one model per directory."
        ),
    )
    parser.add_argument(
        "--target",
        choices=(TARGET_Z, TARGET_LOGZ),
        default=TARGET_Z,
        help=(
            "Regression target. 'z' is linear spectroscopic redshift (default). "
            "'logz' is ln(max(z, --z-floor)) and writes under daep_latent_logz."
        ),
    )
    parser.add_argument(
        "--z-floor",
        type=float,
        default=Z_FLOOR,
        help=f"Lower clamp on z before log (logz only; default: {Z_FLOOR}).",
    )
    parser.add_argument(
        "--out-root",
        type=pathlib.Path,
        default=None,
        help=(
            "Parent output directory. Default: daep_latent_redshift for --target z, "
            "daep_latent_logz for --target logz. Each job writes to "
            "<out-root>/<try_N>/split_<seed>/."
        ),
    )
    parser.add_argument(
        "--train-seed",
        type=int,
        default=TRAIN_RNG_SEED,
        help=f"Training RNG seed (default: {TRAIN_RNG_SEED}).",
    )
    parser.add_argument(
        "--recipe",
        choices=("iter10",),
        default=None,
        help=(
            "Named hyperparameter recipe. "
            "'iter10' matches legacy_unique iter10..iter18 "
            "(ff_dim=384, lr=2e-5, batch=16, epochs=50, patience=10)."
        ),
    )
    parser.add_argument("--ff-dim", type=int, default=None, help="MLP hidden width override.")
    parser.add_argument("--lr", type=float, default=None, help="AdamW learning-rate override.")
    parser.add_argument("--batch-size", type=int, default=None, help="Dataloader batch size.")
    parser.add_argument("--epochs", type=int, default=None, help="Max training epochs.")
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="Early-stopping patience (validation MSE).",
    )
    args = parser.parse_args()

    train_overrides: Dict[str, Any] = {}
    if args.recipe == "iter10":
        train_overrides.update(ITER10_RECIPE)
    for key, val in (
        ("ff_dim", args.ff_dim),
        ("lr", args.lr),
        ("batch_size", args.batch_size),
        ("epochs", args.epochs),
        ("early_stopping_patience", args.early_stopping_patience),
    ):
        if val is not None:
            train_overrides[key] = val
    overrides_arg = train_overrides or None
    out_root = args.out_root if args.out_root is not None else default_out_root_for_target(args.target)

    run_latent_dirs(
        args.latent_dirs,
        out_root=out_root,
        train_seed=args.train_seed,
        train_overrides=overrides_arg,
        target=args.target,
        z_floor=float(args.z_floor),
    )


if __name__ == "__main__":
    main()
