#!/usr/bin/env python3
"""Train the website DASH 1D CNN (+redshift) on a 90/10 IAU train/val split.

Same architecture and hparams as ``dash_retrain.py``. No held-out test set:
the remaining 10% is validation for early stopping; production eval is future
spectra.

Usage (from repo root):
  PYTHONPATH=zmodel_training python zmodel_training/train_website_dash.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
WISEREP_DATA = PROJECT_ROOT / "WiserepData"
for path in (PROJECT_ROOT, SCRIPT_DIR, WISEREP_DATA):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import constants as const
import dash_retrain
import helpers as helpers
from cache_dash_preprocessed import resolve_flux_cache
from create_henna_matched_dash_split import (
    ASCII_COLUMN,
    IAU_COLUMN,
    LABEL_COLUMN,
    enrich_henna_meta,
    filter_indices_mapped,
    load_spec_id_to_ascii,
    split_payload_for_indices,
)
from dash_ensemble_plots import save_precision_recall_cm
from iau_train_val_test_split import IAU_SPLIT_SEED, split_row_indices_by_iau_train_val_test

HAS_REDSHIFT = True
TRAIN_FRAC = 0.9
VAL_FRAC = 0.1
TEST_FRAC = 0.0
HENNA_META = const.WISEREP_HENNA_Z / "meta.csv"
DEFAULT_SPLITS_JSON = const.WISEREP_DIR / "website_final_split_z.json"
DEFAULT_OUT_DIR = (
    PROJECT_ROOT / "data" / "pre_trained_models" / "website_final" / "dash_z"
)


def build_website_90_10_splits(
    henna_meta_csv: Path,
    out_json: Path,
    processed_meta_out: Path,
    *,
    require_file: bool = True,
) -> dict[str, Any]:
    raw = pd.read_csv(henna_meta_csv, low_memory=False).reset_index(drop=True)
    meta = enrich_henna_meta(raw, load_spec_id_to_ascii(const.METADATA_CSV))
    processed_meta_out.parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(processed_meta_out, index=False)

    tr_idx, va_idx, te_idx = split_row_indices_by_iau_train_val_test(
        meta,
        IAU_COLUMN,
        TRAIN_FRAC,
        VAL_FRAC,
        TEST_FRAC,
        IAU_SPLIT_SEED,
    )
    tr_idx = filter_indices_mapped(meta, tr_idx)
    va_idx = filter_indices_mapped(meta, va_idx)

    train_part = split_payload_for_indices(meta, tr_idx, const.SPECTRA_DIR, require_file)
    val_part = split_payload_for_indices(meta, va_idx, const.SPECTRA_DIR, require_file)

    iau = meta[IAU_COLUMN].astype(str).str.strip()
    payload: dict[str, Any] = {
        "split_method": "henna_deduped_meta_iau_90_10_label_filtered",
        "seed": IAU_SPLIT_SEED,
        "train_frac": TRAIN_FRAC,
        "val_frac": VAL_FRAC,
        "test_frac": TEST_FRAC,
        "iau_col": IAU_COLUMN,
        "label_col": LABEL_COLUMN,
        "ascii_col": ASCII_COLUMN,
        "henna_meta_csv": str(henna_meta_csv.resolve()),
        "processed_meta_csv": str(processed_meta_out.resolve()),
        "train": train_part["ascii_files"],
        "val": val_part["ascii_files"],
        "test": [],
        "counts": {
            "train": len(train_part["ascii_files"]),
            "val": len(val_part["ascii_files"]),
            "test": 0,
        },
        "unique_iau": {
            "train": int(pd.unique(iau.iloc[tr_idx]).size) if tr_idx.size else 0,
            "val": int(pd.unique(iau.iloc[va_idx]).size) if va_idx.size else 0,
            "test": 0,
        },
        "train_row_indices": train_part["row_indices"],
        "val_row_indices": val_part["row_indices"],
        "test_row_indices": [],
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def save_val_cm_from_performance(out_dir: Path, title: str) -> Path:
    perf_path = out_dir / "model_performance.json"
    perf = json.loads(perf_path.read_text(encoding="utf-8"))
    cm = perf["confusion_matrix_raw"]
    labels = list(perf.get("confusion_matrix_labels") or const.CLASS_NAMES)
    cm_path = out_dir / "confusion_matrices" / "confusion_val.png"
    save_precision_recall_cm(cm, labels, title, cm_path)
    return cm_path


def train_website_dash(
    *,
    has_redshift: bool,
    henna_meta_csv: Path,
    splits_json: Path,
    out_dir: Path,
    seed: int,
    cache_dir: Path | None = None,
    no_cache: bool = False,
) -> Path:
    helpers.set_seed(seed)
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    processed_meta = out_dir / "wiserep_metadata_processed.csv"

    payload = build_website_90_10_splits(
        henna_meta_csv,
        splits_json,
        processed_meta,
    )
    print(
        f"Split {splits_json.name}: train={payload['counts']['train']}  "
        f"val={payload['counts']['val']}  "
        f"IAU train/val={payload['unique_iau']['train']}/{payload['unique_iau']['val']}"
    )

    device = helpers.get_device()
    print(f"has_redshift={has_redshift}  seed={seed}  device={device}  out_dir={out_dir}")

    with open(out_dir / "class_mapping.json", "w") as f:
        json.dump({n: const.CLASS_TO_IDX[n] for n in const.CLASS_NAMES}, f, indent=2)

    metadata = helpers.load_metadata_from_processed_csv(processed_meta)
    train_filenames = list(payload["train"])
    val_filenames = list(payload["val"])
    class_weights = helpers.compute_class_weights_from_filenames(train_filenames, metadata)
    flux_cache = resolve_flux_cache(
        has_redshift,
        cache_dir=cache_dir,
        disable=bool(no_cache),
    )
    train_loader = helpers.make_loader(
        train_filenames,
        metadata,
        has_redshift,
        device,
        shuffle=True,
        batch_size=const.BATCH_SIZE,
        flux_cache=flux_cache,
    )
    val_loader = helpers.make_loader(
        val_filenames,
        metadata,
        has_redshift,
        device,
        batch_size=const.BATCH_SIZE,
        flux_cache=flux_cache,
    )
    print(
        f"Effective dataset sizes: train={len(train_loader.dataset)}  "
        f"val={len(val_loader.dataset)}"
    )

    model = dash_retrain.DashCNN1D(
        input_length=const.TARGET_LENGTH,
        num_classes=const.NUM_CLASSES,
    ).to(device)
    best_path = dash_retrain.train(
        model,
        train_loader,
        val_loader,
        device,
        class_weights,
        epochs=const.EPOCHS,
        lr=const.LEARNING_RATE,
        patience=const.EARLY_STOP_PATIENCE,
        val_every=const.VAL_EVERY,
        out_dir=out_dir,
    )

    ds_cache = getattr(train_loader.dataset, "flux_cache", None)
    training_config = {
        "run_id": "website_final",
        "has_redshift": has_redshift,
        "target_length": const.TARGET_LENGTH,
        "wave_min": const.WAVE_MIN,
        "wave_max": const.WAVE_MAX,
        "num_classes": const.NUM_CLASSES,
        "class_names": const.CLASS_NAMES,
        "epochs": const.EPOCHS,
        "batch_size": const.BATCH_SIZE,
        "lr": const.LEARNING_RATE,
        "patience": const.EARLY_STOP_PATIENCE,
        "class_weights": class_weights.tolist(),
        "splits_file": str(splits_json.resolve()),
        "processed_meta_csv": str(processed_meta.resolve()),
        "k_fold": 1,
        "seed": seed,
        "data_mode": "website_final_ascii_90_10",
        "dash_cache": str(ds_cache.cache_dir) if ds_cache is not None else None,
    }
    with open(out_dir / "training_config.json", "w") as f:
        json.dump(training_config, f, indent=2)

    z_tag = "with redshift" if has_redshift else "without redshift"
    cm_path = save_val_cm_from_performance(
        out_dir,
        f"Dash 1D CNN {z_tag} (val, 90/10 website final)",
    )
    print(f"\nSaved: {best_path}")
    print(f"  Config: {out_dir / 'training_config.json'}")
    print(f"  Performance: {out_dir / 'model_performance.json'}")
    print(f"  Val confusion matrix: {cm_path}")
    return best_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Website-final DASH 1D CNN (+redshift), 90% train / 10% val by IAU."
    )
    parser.add_argument("--seed", type=int, default=const.SEED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--splits-json", type=Path, default=DEFAULT_SPLITS_JSON)
    parser.add_argument("--henna-meta", type=Path, default=HENNA_META)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_website_dash(
        has_redshift=HAS_REDSHIFT,
        henna_meta_csv=args.henna_meta,
        splits_json=args.splits_json,
        out_dir=args.out_dir,
        seed=args.seed,
        cache_dir=args.cache_dir,
        no_cache=args.no_cache,
    )


if __name__ == "__main__":
    main()
