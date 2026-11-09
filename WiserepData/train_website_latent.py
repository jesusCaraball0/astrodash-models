#!/usr/bin/env python3
"""Train the website latent MLP (+z embeddings) on a 90/10 IAU train/val split.

Frozen Universal embeddings from a try_* dir; same architecture and default
hparams as ``train_latent.py``. No held-out test set.

Usage (from repo root):
  PYTHONPATH=WiserepData python WiserepData/train_website_latent.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

WISEREP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WISEREP_DIR.parent
ZMODEL_DIR = PROJECT_ROOT / "zmodel_training"
for path in (WISEREP_DIR, ZMODEL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dash_ensemble_plots import save_precision_recall_cm
from iau_train_val_test_split import IAU_SPLIT_SEED, split_row_indices_by_iau_train_val_test
from train_latent import (
    BATCH_SIZE,
    CLASS_NAMES,
    DEVICE,
    DROPOUT,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    IAU_COLUMN,
    ITER10_RECIPE,
    LABEL_COLUMN,
    LR,
    TRAIN_RNG_SEED,
    WEIGHT_DECAY,
    LatentClassifier,
    LatentDataset,
    build_cfg,
    collate_latent,
    inverse_frequency_class_weights,
    load_latent_and_meta,
    split_and_filter,
    train_latent_classifier,
)
from TwinsModel_Wiserep import device_from_str
from TwinsTrain_Wiserep import set_seeds
from torch.utils.data import DataLoader

USE_REDSHIFT = True
TRAIN_FRAC = 0.9
VAL_FRAC = 0.1
TEST_FRAC = 0.0
DEFAULT_LATENT_DIR = PROJECT_ROOT / "data" / "wiserep_henna" / "try_5" / "Dered36_5"
DEFAULT_OUT_DIR = (
    PROJECT_ROOT / "data" / "pre_trained_models" / "website_final" / "latent_z"
)


def website_90_10_indices(meta) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_idx, val_idx, test_idx = split_row_indices_by_iau_train_val_test(
        meta,
        IAU_COLUMN,
        TRAIN_FRAC,
        VAL_FRAC,
        TEST_FRAC,
        IAU_SPLIT_SEED,
    )
    return split_and_filter(
        meta,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        split_tag=f"iau_90_10_seed={IAU_SPLIT_SEED}",
    )


def save_val_cm_from_performance(out_dir: Path, title: str) -> Path:
    perf = json.loads((out_dir / "model_performance.json").read_text(encoding="utf-8"))
    cm = perf["confusion_matrix_raw"]
    labels = list(perf.get("confusion_matrix_labels") or CLASS_NAMES)
    cm_path = out_dir / "confusion_matrices" / "confusion_val.png"
    save_precision_recall_cm(cm, labels, title, cm_path)
    return cm_path


def train_website_latent(
    *,
    latent_dir: Path,
    out_dir: Path,
    train_seed: int,
    train_overrides: Dict[str, Any] | None = None,
    use_redshift: bool = True,
) -> Path:
    latent_dir = latent_dir.resolve()
    out_dir = out_dir.resolve()
    set_seeds(train_seed)

    cfg_json = latent_dir / "cfg_used.json"
    meta, z, latent_npz, meta_csv = load_latent_and_meta(latent_dir)
    train_idx, val_idx, test_idx = website_90_10_indices(meta)

    cfg = build_cfg(
        z,
        latent_npz=latent_npz,
        meta_csv=meta_csv,
        cfg_json=cfg_json,
        seed=train_seed,
        run_id="website_final",
        split_source=f"iau_90_10_seed={IAU_SPLIT_SEED}",
        train_overrides=train_overrides,
    )
    cfg["train_frac"] = TRAIN_FRAC
    cfg["val_frac"] = VAL_FRAC
    cfg["test_frac"] = TEST_FRAC
    cfg["use_redshift"] = bool(use_redshift)

    batch_size = int(cfg.get("batch_size", BATCH_SIZE))
    lr = float(cfg.get("lr", LR))
    weight_decay = float(cfg.get("weight_decay", WEIGHT_DECAY))
    epochs = int(cfg.get("epochs", EPOCHS))
    early_stopping_patience = int(cfg.get("early_stopping_patience", EARLY_STOPPING_PATIENCE))
    embed_dim = int(np.prod(z.shape[1:]))
    head_hidden = int(cfg["ff_dim"]) if cfg.get("ff_dim") is not None else 512
    head_dropout = float(cfg.get("dropout", DROPOUT))

    train_ds = LatentDataset(meta, z, train_idx, LABEL_COLUMN)
    val_ds = LatentDataset(meta, z, val_idx, LABEL_COLUMN)
    test_ds = LatentDataset(meta, z, test_idx, LABEL_COLUMN)
    train_load = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_latent, num_workers=0
    )
    val_load = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_latent, num_workers=0
    )
    test_load = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_latent, num_workers=0
    )

    device = device_from_str(DEVICE)
    model = LatentClassifier(
        embed_dim=embed_dim,
        n_cls=len(CLASS_NAMES),
        head_hidden=head_hidden,
        head_dropout=head_dropout,
    ).to(device)

    print(
        f"[website latent] redshift={use_redshift} dir={latent_dir} out={out_dir} "
        f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}",
        flush=True,
    )
    ce_weights = inverse_frequency_class_weights(meta, train_idx, LABEL_COLUMN).to(device)
    best_path = train_latent_classifier(
        model,
        train_load,
        val_load,
        test_load,
        device,
        out_dir,
        cfg,
        class_weights=ce_weights,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        early_stopping_patience=early_stopping_patience,
    )
    z_tag = "with redshift embeddings" if use_redshift else "without redshift embeddings"
    cm_path = save_val_cm_from_performance(
        out_dir,
        f"Latent MLP {z_tag} (val, 90/10 website final)",
    )
    print(f"[done] val confusion matrix: {cm_path}", flush=True)
    return best_path


def _overrides_from_args(args: argparse.Namespace) -> Dict[str, Any] | None:
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
    return train_overrides or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Website-final latent MLP (+z), 90% train / 10% val by IAU."
    )
    parser.add_argument("--latent-dir", type=Path, default=DEFAULT_LATENT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--train-seed", type=int, default=TRAIN_RNG_SEED)
    parser.add_argument("--recipe", choices=("iter10",), default=None)
    parser.add_argument("--ff-dim", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_website_latent(
        latent_dir=args.latent_dir,
        out_dir=args.out_dir,
        train_seed=args.train_seed,
        train_overrides=_overrides_from_args(args),
        use_redshift=USE_REDSHIFT,
    )


if __name__ == "__main__":
    main()
