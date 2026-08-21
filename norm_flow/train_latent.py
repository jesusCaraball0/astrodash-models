#!/usr/bin/env python3
"""Train latent MLP redshift models with MSE / Gaussian / spline / MoE heads.

Observed-frame Universal embeddings only (try_*_noz / Nodered*). Frozen encoder.
Reuses LatentRedshiftDataset and assignment JSON from train_latent_redshift.py.

  python norm_flow/train_latent.py --head flow \\
      --latent-dirs data/wiserep_henna/try_5_noz/Nodered36_5
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from paths import PROJECT_ROOT, setup_imports

setup_imports()

from TwinsModel_Wiserep import device_from_str
from TwinsTrain_Wiserep import set_seeds
from train_latent import (
    BATCH_SIZE,
    DEVICE,
    DROPOUT,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    ITER10_RECIPE,
    LR,
    TRAIN_RNG_SEED,
    WEIGHT_DECAY,
    build_cfg,
    load_assignment_indices_from_dir,
    load_latent_and_meta,
    split_run_name,
    try_run_name,
)
from train_latent_redshift import (
    REDSHIFT_COLUMN,
    LatentRedshiftDataset,
    ensure_redshift_column,
    split_and_filter_redshift,
)

from heads import FLOW_LIKE_HEADS, HEADS
from leakage import (
    assert_iau_indices_disjoint,
    assert_observed_frame_latent_dir,
    document_z_handling,
)
from models import TARGETS, LatentMLPTrunk, RedshiftPredictor
from train_utils import Z_FLOOR, Z_MAX, train_and_eval


def default_out_root(head: str, target: str = "logz") -> Path:
    tag = "logz" if target == "logz" else "z"
    return PROJECT_ROOT / "data" / "pre_trained_models" / f"daep_latent_{tag}_{head}"


class DictLatentDataset(Dataset):
    def __init__(self, inner: LatentRedshiftDataset):
        self.inner = inner

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        row = self.inner[i]
        return {
            "x": row["z"].reshape(-1).float(),
            "y": row["y"].float(),
            "id": str(int(row["idx"].item())),
        }


def collate_dict(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "x": torch.stack([b["x"] for b in batch], dim=0),
        "y": torch.stack([b["y"] for b in batch], dim=0),
        "id": [b["id"] for b in batch],
    }


def make_loader(ds: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=shuffle,
        collate_fn=collate_dict,
        num_workers=0,
    )


def run_one(
    *,
    head: str,
    latent_dir: Path,
    out_dir: Path,
    train_seed: int,
    train_overrides: Dict[str, Any] | None,
    z_floor: float,
    bins: int,
    transforms: int,
    flow_hidden: int,
    clip_norm: float | None,
    eval_only: bool = False,
    target: str = "logz",
) -> None:
    latent_dir = Path(latent_dir).resolve()
    assert_observed_frame_latent_dir(latent_dir)
    set_seeds(int(train_seed))
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
    assert_iau_indices_disjoint(meta, train_idx, val_idx, test_idx)
    print(document_z_handling(z_floor=z_floor, z_max=Z_MAX, target=target), flush=True)
    print(f"[leakage] observed-frame latent dir OK: {latent_dir}", flush=True)

    cfg = build_cfg(
        z,
        latent_npz=latent_npz,
        meta_csv=meta_csv,
        cfg_json=latent_dir / "cfg_used.json",
        seed=int(train_seed),
        run_id=split_run_name(latent_dir),
        split_source=split_source,
        train_overrides=train_overrides,
    )
    task_tag = "logz" if target == "logz" else "z"
    cfg.update(
        {
            "classifier_kind": f"latent_flatten_mlp_redshift_{task_tag}_{head}",
            "task": f"redshift_{task_tag}_{head}",
            "head": head,
            "target": target,
            "log": "ln" if target == "logz" else None,
            "redshift_column": REDSHIFT_COLUMN,
            "z_max": Z_MAX,
            "z_floor": float(z_floor),
            "has_redshift": False,
            "observed_frame": True,
            "flow_bins": int(bins),
            "flow_transforms": int(transforms),
            "flow_hidden": int(flow_hidden),
            "clip_norm": clip_norm,
            "moe_experts": 2 if head == "moe" else None,
            "z_handling": document_z_handling(z_floor=z_floor, z_max=Z_MAX, target=target),
        }
    )
    batch_size = int(cfg.get("batch_size", BATCH_SIZE))
    lr = float(cfg.get("lr", LR))
    weight_decay = float(cfg.get("weight_decay", WEIGHT_DECAY))
    epochs = int(cfg.get("epochs", EPOCHS))
    patience = int(cfg.get("early_stopping_patience", EARLY_STOPPING_PATIENCE))
    embed_dim = int(np.prod(z.shape[1:]))
    hidden = int(cfg["ff_dim"]) if cfg.get("ff_dim") is not None else 512
    dropout = float(cfg.get("dropout", DROPOUT))

    train_ds = DictLatentDataset(LatentRedshiftDataset(meta, z, train_idx, REDSHIFT_COLUMN))
    val_ds = DictLatentDataset(LatentRedshiftDataset(meta, z, val_idx, REDSHIFT_COLUMN))
    test_ds = DictLatentDataset(LatentRedshiftDataset(meta, z, test_idx, REDSHIFT_COLUMN))
    train_load = make_loader(train_ds, batch_size, True)
    val_load = make_loader(val_ds, batch_size, False)
    test_load = make_loader(test_ds, batch_size, False)
    print(
        f"dataset rows: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}  "
        f"embed_dim={embed_dim}  context={hidden}",
        flush=True,
    )

    device = device_from_str(DEVICE)
    model = RedshiftPredictor(
        LatentMLPTrunk(embed_dim, hidden, dropout),
        head,
        bins=bins,
        transforms=transforms,
        flow_hidden=flow_hidden,
        target=target,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] device={device} params={n_params:,} head={head}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_and_eval(
        model,
        train_load,
        val_load,
        test_load,
        device=device,
        out_dir=out_dir,
        cfg=cfg,
        optimizer=opt,
        epochs=epochs,
        patience=patience,
        clip_norm=clip_norm,
        z_floor=z_floor,
        z_max=Z_MAX,
        title=f"latent {head} {target} {out_dir.name}",
        eval_only=eval_only,
        target=target,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Latent MLP redshift density (MSE / Gaussian / NSF / MoE)."
    )
    parser.add_argument("--head", choices=HEADS, required=True)
    parser.add_argument(
        "--target",
        choices=TARGETS,
        default="logz",
        help="Training-space label: ln z (default) or physical z.",
    )
    parser.add_argument(
        "--latent-dirs",
        nargs="+",
        type=Path,
        required=True,
        help="Observed-frame latent dirs (try_*_noz / Nodered*). One model per dir.",
    )
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--train-seed", type=int, default=TRAIN_RNG_SEED)
    parser.add_argument("--z-floor", type=float, default=Z_FLOOR)
    parser.add_argument("--recipe", choices=("iter10",), default=None)
    parser.add_argument("--ff-dim", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--bins", type=int, default=8, help="Spline bins (flow / moe).")
    parser.add_argument("--transforms", type=int, default=2, help="Stacked spline transforms.")
    parser.add_argument("--flow-hidden", type=int, default=256)
    parser.add_argument(
        "--clip-norm",
        type=float,
        default=None,
        help="Gradient clip. Default 1.0 for flow/moe, disabled otherwise.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training; write metrics/plots from model_best.pt.",
    )
    args = parser.parse_args()
    head = str(args.head)
    target = str(args.target)
    clip_norm = args.clip_norm
    if clip_norm is None and head in FLOW_LIKE_HEADS:
        clip_norm = 1.0
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
    out_root = args.out_root if args.out_root is not None else default_out_root(head, target)

    for latent_dir in args.latent_dirs:
        latent_dir = Path(latent_dir).resolve()
        out_dir = Path(out_root) / try_run_name(latent_dir) / split_run_name(latent_dir)
        print(
            f"=== latent redshift | head={head} target={target} dir={latent_dir} out={out_dir} ===",
            flush=True,
        )
        run_one(
            head=head,
            latent_dir=latent_dir,
            out_dir=out_dir,
            train_seed=int(args.train_seed),
            train_overrides=train_overrides or None,
            z_floor=float(args.z_floor),
            bins=int(args.bins),
            transforms=int(args.transforms),
            flow_hidden=int(args.flow_hidden),
            clip_norm=clip_norm,
            eval_only=bool(args.eval_only),
            target=target,
        )


if __name__ == "__main__":
    main()
