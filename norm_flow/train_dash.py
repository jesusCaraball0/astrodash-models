#!/usr/bin/env python3
"""Train DASH 1D CNN redshift models with MSE / Gaussian / spline / MoE heads.

Observed-frame preprocessing only (has_redshift=False). Same splits JSON as
dash_retrain_redshift.py.

  python norm_flow/train_dash.py --head flow \\
      --splits-json data/wiserep/henna_matched_split_noz_seed36.json
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from paths import PROJECT_ROOT, setup_imports

setup_imports()

import constants as const
import dash_retrain_redshift as dash_z
import helpers as helpers
from cache_dash_preprocessed import resolve_flux_cache

from heads import FLOW_LIKE_HEADS, HEADS
from leakage import assert_disjoint_ids, assert_observed_frame_splits, document_z_handling
from models import TARGETS, DashCNNTrunk, RedshiftPredictor
from train_utils import Z_FLOOR, Z_MAX, train_and_eval

for _name in ("spectrum_io", "dash_preprocess"):
    logging.getLogger(_name).setLevel(logging.CRITICAL)


def default_out_dir(head: str, split_seed: Any, target: str = "logz") -> Path:
    tag = "logz" if target == "logz" else "z"
    return PROJECT_ROOT / "data" / "pre_trained_models" / f"dash_{tag}_{head}" / f"split_{split_seed}"


class DictDashDataset(Dataset):
    def __init__(self, inner: dash_z.WISeREPRedshiftDataset):
        self.inner = inner

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, i: int) -> Optional[Dict[str, Any]]:
        item = self.inner[i]
        if item is None:
            return None
        x, z_val = item
        fname = self.inner.samples[i][0]
        return {"x": x.float(), "y": torch.tensor(float(z_val), dtype=torch.float32), "id": fname}


def collate_dict(batch: List[Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return {
        "x": torch.stack([b["x"] for b in batch], dim=0),
        "y": torch.stack([b["y"] for b in batch], dim=0),
        "id": [b["id"] for b in batch],
    }


def iau_by_ascii(metadata_csv: Path) -> Dict[str, str]:
    import csv

    out: Dict[str, str] = {}
    with metadata_csv.open("r", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            fname = (row.get("Ascii file") or "").strip()
            iau = (row.get("IAU name") or "").strip()
            if fname and fname not in out:
                out[fname] = iau
    return out


def make_loader(ds: Dataset, shuffle: bool, device: torch.device) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=const.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_dict,
        pin_memory=(device.type == "cuda"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DASH 1D CNN redshift density (MSE / Gaussian / NSF / MoE)."
    )
    parser.add_argument("--head", choices=HEADS, required=True)
    parser.add_argument(
        "--target",
        choices=TARGETS,
        default="logz",
        help="Training-space label: ln z (default) or physical z.",
    )
    parser.add_argument("--splits-json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0, help="Training RNG seed (not the data-split seed).")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--z-floor", type=float, default=Z_FLOOR)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--bins", type=int, default=8)
    parser.add_argument("--transforms", type=int, default=2)
    parser.add_argument("--flow-hidden", type=int, default=256)
    parser.add_argument("--clip-norm", type=float, default=None)
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training; write metrics/plots from model_best.pt.",
    )
    args = parser.parse_args()

    helpers.set_seed(int(args.seed))
    head = str(args.head)
    target = str(args.target)
    clip_norm = args.clip_norm
    if clip_norm is None and head in FLOW_LIKE_HEADS:
        clip_norm = 1.0
    splits_path = args.splits_json.resolve()
    assert_observed_frame_splits(splits_path)
    splits = helpers.load_json(splits_path)
    split_seed = splits.get("data_split_seed", splits.get("seed", splits_path.stem))
    out_dir = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else default_out_dir(head, split_seed, target)
    )

    train_files = list(splits.get("train", []))
    val_files = list(splits.get("val", []))
    test_files = list(splits.get("test", []))
    assert_disjoint_ids(
        {"train": train_files, "val": val_files, "test": test_files},
        kind="ascii file",
    )
    iau_map = iau_by_ascii(const.METADATA_CSV)

    def _iaus(files: List[str]) -> List[str]:
        return [iau_map[f] for f in files if iau_map.get(f, "").strip()]

    assert_disjoint_ids(
        {"train": _iaus(train_files), "val": _iaus(val_files), "test": _iaus(test_files)},
        kind="IAU name",
    )
    print(document_z_handling(z_floor=float(args.z_floor), z_max=Z_MAX, target=target), flush=True)
    print(f"[leakage] observed-frame splits OK: {splits_path.name}", flush=True)
    task_tag = "logz" if target == "logz" else "z"
    print(
        f"task=redshift_{task_tag}_{head}  has_redshift=False  train_rng_seed={args.seed}  out_dir={out_dir}",
        flush=True,
    )
    print(
        f"Splits: train={len(train_files)}  val={len(val_files)}  test={len(test_files)}",
        flush=True,
    )

    metadata = helpers.load_metadata(const.METADATA_CSV)
    device = helpers.get_device()
    flux_cache = resolve_flux_cache(False, cache_dir=args.cache_dir, disable=bool(args.no_cache))
    if flux_cache is not None:
        cache_s = str(flux_cache.cache_dir).lower()
        if "noz" not in cache_s:
            raise RuntimeError(f"DASH cache must be observed-frame / noz, got {flux_cache.cache_dir}")

    def _ds(files: List[str]) -> DictDashDataset:
        return DictDashDataset(
            dash_z.WISeREPRedshiftDataset(files, const.SPECTRA_DIR, metadata, flux_cache=flux_cache)
        )

    train_ds, val_ds, test_ds = _ds(train_files), _ds(val_files), _ds(test_files)
    train_load = make_loader(train_ds, True, device)
    val_load = make_loader(val_ds, False, device)
    test_load = make_loader(test_ds, False, device)
    print(
        f"Effective sizes: train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}",
        flush=True,
    )

    model = RedshiftPredictor(
        DashCNNTrunk(const.TARGET_LENGTH),
        head,
        bins=int(args.bins),
        transforms=int(args.transforms),
        flow_hidden=int(args.flow_hidden),
        target=target,
    ).to(device)
    print(
        f"[model] device={device} params={sum(p.numel() for p in model.parameters()):,} head={head}",
        flush=True,
    )
    opt = torch.optim.Adam(model.parameters(), lr=const.LEARNING_RATE, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
    cfg = {
        "run_id": f"split_{split_seed}",
        "task": f"redshift_{task_tag}_{head}",
        "head": head,
        "target": target,
        "log": "ln" if target == "logz" else None,
        "has_redshift": False,
        "observed_frame": True,
        "target_length": const.TARGET_LENGTH,
        "epochs": const.EPOCHS,
        "batch_size": const.BATCH_SIZE,
        "lr": const.LEARNING_RATE,
        "patience": const.EARLY_STOP_PATIENCE,
        "splits_file": str(splits_path),
        "seed": int(args.seed),
        "data_split_seed": split_seed,
        "z_max": Z_MAX,
        "z_floor": float(args.z_floor),
        "dash_cache": str(flux_cache.cache_dir) if flux_cache is not None else None,
        "flow_bins": int(args.bins),
        "flow_transforms": int(args.transforms),
        "flow_hidden": int(args.flow_hidden),
        "clip_norm": clip_norm,
        "moe_experts": 2 if head == "moe" else None,
        "z_handling": document_z_handling(z_floor=float(args.z_floor), z_max=Z_MAX, target=target),
    }
    train_and_eval(
        model,
        train_load,
        val_load,
        test_load,
        device=device,
        out_dir=out_dir,
        cfg=cfg,
        optimizer=opt,
        scheduler=sched,
        epochs=const.EPOCHS,
        patience=const.EARLY_STOP_PATIENCE,
        clip_norm=clip_norm,
        z_floor=float(args.z_floor),
        z_max=Z_MAX,
        title=f"dash {head} {target} split_{split_seed}",
        eval_only=bool(args.eval_only),
        target=target,
    )


if __name__ == "__main__":
    main()
