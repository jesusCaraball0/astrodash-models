#!/usr/bin/env python3
"""Train latent MLP redshift models with MSE / Gaussian / spline / MoE heads.

Observed-frame Universal embeddings only (try_*_noz / Nodered*). Frozen encoder.
Reuses LatentRedshiftDataset and assignment JSON from train_latent_redshift.py.

  python norm_flow/train_latent.py --head flow \\
      --latent-dirs data/wiserep_henna/try_5_noz/Nodered36_5

Wavelength-stretch train-only augmentation (5 copies, noise on ln(1+z)):

  python norm_flow/train_latent.py --head flow --aug \\
      --latent-dirs data/wiserep_henna/try_5_noz/Nodered36_5
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

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
    redshift_values,
    split_and_filter_redshift,
)
from augment_logz import (
    AUG_ENCODE_BATCH,
    AUG_EPS_HI,
    AUG_EPS_LO,
    AUG_K,
    dataset_from_cache,
    ensure_aug_latents,
)

from class_cond import (
    N_CLASSES,
    default_latent_cls_dir,
    filter_mapped_indices,
    latent_class_onehot,
    latent_class_probs,
    load_latent_classifier,
)
from heads import FLOW_LIKE_HEADS, HEADS
from leakage import (
    assert_iau_indices_disjoint,
    assert_observed_frame_latent_dir,
    document_z_handling,
)
from models import TARGETS, LatentMLPTrunk, RedshiftPredictor
from train_utils import Z_FLOOR, Z_MAX, output_tag, resolve_better_hps, resolve_deeper, train_and_eval


def default_out_root(
    head: str,
    target: str = "logz",
    cls_cond: bool = False,
    cls_true: bool = False,
    better_hps: bool = False,
    deeper: bool = False,
    reweight: bool = False,
    aug: bool = False,
) -> Path:
    infix = output_tag(
        target,
        cls_cond=cls_cond,
        cls_true=cls_true,
        better_hps=better_hps,
        deeper=deeper,
        reweight=reweight,
        aug=aug,
    )
    return PROJECT_ROOT / "data" / "pre_trained_models" / f"daep_latent_{infix}_{head}"


class DictLatentDataset(Dataset):
    def __init__(
        self,
        inner: LatentRedshiftDataset,
        cls_probs: Optional[torch.Tensor] = None,
    ):
        self.inner = inner
        self.cls_probs = cls_probs

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        row = self.inner[i]
        x = row["z"].reshape(-1).float()
        if self.cls_probs is not None:
            x = torch.cat([x, self.cls_probs[i].float()], dim=0)
        return {
            "x": x,
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
    flow_hidden_layers: int = 2,
    trunk_layers: int = 1,
    eval_only: bool = False,
    target: str = "logz",
    cls_cond: bool = False,
    cls_true: bool = False,
    cls_ckpt: Path | None = None,
    cover_z: float | None = None,
    better_hps: bool = False,
    deeper: bool = False,
    reweight: bool = False,
    reweight_alpha: float = 0.5,
    aug: bool = False,
    aug_k: int = AUG_K,
    aug_eps_lo: float = AUG_EPS_LO,
    aug_eps_hi: float = AUG_EPS_HI,
    aug_force: bool = False,
    aug_encode_batch: int = AUG_ENCODE_BATCH,
    aug_ckpt: Path | None = None,
    aug_flux_dir: Path | None = None,
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
    if cls_cond or cls_true:
        train_idx = filter_mapped_indices(meta, train_idx)
        val_idx = filter_mapped_indices(meta, val_idx)
        test_idx = filter_mapped_indices(meta, test_idx)
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
            "flow_hidden_layers": int(flow_hidden_layers),
            "trunk_layers": int(trunk_layers),
            "clip_norm": clip_norm,
            "moe_experts": 2 if head == "moe" else None,
            "z_handling": document_z_handling(z_floor=z_floor, z_max=Z_MAX, target=target),
            "cls_cond": bool(cls_cond),
            "cls_true": bool(cls_true),
            "cls_source": "onehot" if cls_true else ("softmax" if cls_cond else None),
            "better_hps": bool(better_hps),
            "deeper": bool(deeper),
            "reweight": bool(reweight),
            "reweight_alpha": float(reweight_alpha) if reweight else None,
            "aug": bool(aug),
            "aug_k": int(aug_k) if aug else None,
            "aug_eps_lo": float(aug_eps_lo) if aug else None,
            "aug_eps_hi": float(aug_eps_hi) if aug else None,
            "cover_z": cover_z,
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

    device = device_from_str(DEVICE)
    train_cls = val_cls = test_cls = None
    if cls_true:
        train_cls = latent_class_onehot(meta, train_idx)
        val_cls = latent_class_onehot(meta, val_idx)
        test_cls = latent_class_onehot(meta, test_idx)
        embed_dim = embed_dim + N_CLASSES
        print(
            f"[cls] concat true one-hot at MLP input  embed_dim={embed_dim}  "
            f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}",
            flush=True,
        )
    elif cls_cond:
        ckpt_dir = Path(cls_ckpt).resolve() if cls_ckpt is not None else default_latent_cls_dir(latent_dir)
        cfg["cls_ckpt"] = str(ckpt_dir)
        clf = load_latent_classifier(ckpt_dir, embed_dim, device)
        train_cls = latent_class_probs(clf, z, train_idx, device)
        val_cls = latent_class_probs(clf, z, val_idx, device)
        test_cls = latent_class_probs(clf, z, test_idx, device)
        embed_dim = embed_dim + N_CLASSES
        print(
            f"[cls] concat p_cls at MLP input  embed_dim={embed_dim}  "
            f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}",
            flush=True,
        )

    train_ds = DictLatentDataset(
        LatentRedshiftDataset(meta, z, train_idx, REDSHIFT_COLUMN), train_cls
    )
    val_ds = DictLatentDataset(
        LatentRedshiftDataset(meta, z, val_idx, REDSHIFT_COLUMN), val_cls
    )
    test_ds = DictLatentDataset(
        LatentRedshiftDataset(meta, z, test_idx, REDSHIFT_COLUMN), test_cls
    )
    if aug and eval_only:
        print(
            "[aug] eval-only: skipping stretch/encode; val/test stay unaugmented",
            flush=True,
        )
    if aug and not eval_only:
        if train_cls is not None:
            raise RuntimeError("--aug cannot be combined with --cls-cond / --cls-true")
        z_phys = redshift_values(meta)
        cache = ensure_aug_latents(
            latent_dir=latent_dir,
            meta=meta,
            z_old=z_phys,
            train_idx=train_idx,
            embed_shape=tuple(int(x) for x in z.shape[1:]),
            device=device,
            k=int(aug_k),
            eps_lo=float(aug_eps_lo),
            eps_hi=float(aug_eps_hi),
            z_floor=float(z_floor),
            z_max=Z_MAX,
            seed=int(train_seed),
            flux_dir=aug_flux_dir,
            ckpt_path=aug_ckpt,
            encode_batch=int(aug_encode_batch),
            force=bool(aug_force),
        )
        aug_ds = dataset_from_cache(cache)
        n_orig = len(train_ds)
        train_ds = ConcatDataset([train_ds, aug_ds])
        cfg["aug_n"] = int(len(aug_ds))
        ckpt_saved = cache.get("ckpt")
        if ckpt_saved is not None:
            cfg["aug_ckpt"] = str(np.asarray(ckpt_saved).reshape(-1)[0])
        print(
            f"[aug] train rows {n_orig} original + {len(aug_ds)} stretched = {len(train_ds)}",
            flush=True,
        )
    train_load = make_loader(train_ds, batch_size, True)
    val_load = make_loader(val_ds, batch_size, False)
    test_load = make_loader(test_ds, batch_size, False)
    print(
        f"dataset rows: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}  "
        f"embed_dim={embed_dim}  context={hidden}  trunk_layers={trunk_layers}",
        flush=True,
    )

    model = RedshiftPredictor(
        LatentMLPTrunk(embed_dim, hidden, dropout, n_layers=trunk_layers),
        head,
        bins=bins,
        transforms=transforms,
        flow_hidden=flow_hidden,
        flow_hidden_layers=flow_hidden_layers,
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
        cover_z=cover_z,
        reweight=reweight,
        reweight_alpha=reweight_alpha,
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
    parser.add_argument(
        "--bins",
        type=int,
        default=None,
        help="Spline bins (flow / moe). Default 8, 16 with --deeper, or 64 with --better-hps.",
    )
    parser.add_argument("--transforms", type=int, default=2, help="Stacked spline transforms.")
    parser.add_argument("--flow-hidden", type=int, default=256)
    parser.add_argument(
        "--better-hps",
        action="store_true",
        help="Linear-z NSF: put z<=2 on the spline and use 64 bins. "
        "Writes to daep_latent_z_betterhps_{head}/ and does not overwrite the original z run.",
    )
    parser.add_argument(
        "--deeper",
        action="store_true",
        help="Bigger NSF on the same embeddings: one extra trunk layer, "
        "3-layer conditioner (was 2), 16 spline bins (was 8). "
        "Writes to daep_latent_{target}_deeper_{head}/ and does not overwrite the default run.",
    )
    parser.add_argument(
        "--reweight",
        action="store_true",
        help="Train with inverse-frequency weighted NLL: w_b ∝ N_b^{-alpha} on physical-z "
        "bins (same edges as the MAE-by-z table, alpha=0.5). Eval NLL/MAE stay unweighted. "
        "Writes to daep_latent_{target}_reweight_{head}/ and does not overwrite the default run.",
    )
    parser.add_argument(
        "--reweight-alpha",
        type=float,
        default=None,
        help="Exponent for --reweight (default 0.5). Ignored unless --reweight is set.",
    )
    parser.add_argument(
        "--aug",
        action="store_true",
        help="Train-only wavelength-stretch augmentation: 5 copies per train spectrum, "
        "ε ~ Unif(-0.08, 0.1) on ln(1+z), encoded with the dir's best_ckpt.pt. "
        "Writes to daep_latent_{target}_aug_{head}/ and does not overwrite the default run.",
    )
    parser.add_argument(
        "--aug-k",
        type=int,
        default=None,
        help=f"Augmented copies per train spectrum (default {AUG_K}). Requires --aug.",
    )
    parser.add_argument(
        "--aug-eps-lo",
        type=float,
        default=None,
        help=f"Lower bound for ε on ln(1+z) (default {AUG_EPS_LO}). Requires --aug.",
    )
    parser.add_argument(
        "--aug-eps-hi",
        type=float,
        default=None,
        help=f"Upper bound for ε on ln(1+z) (default {AUG_EPS_HI}). Requires --aug.",
    )
    parser.add_argument(
        "--aug-force",
        action="store_true",
        help="Rebuild the stretched-latent cache even if one exists.",
    )
    parser.add_argument(
        "--aug-encode-batch",
        type=int,
        default=None,
        help=f"Encoder batch size while building the aug cache (default {AUG_ENCODE_BATCH}).",
    )
    parser.add_argument(
        "--aug-ckpt",
        type=Path,
        default=None,
        help="Universal encoder checkpoint (default: <latent-dir>/best_ckpt.pt).",
    )
    parser.add_argument(
        "--aug-flux-dir",
        type=Path,
        default=None,
        help="Observed-frame flux bundle (default: data/wiserep_henna/noderedshift).",
    )
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
    cls_src = parser.add_mutually_exclusive_group()
    cls_src.add_argument(
        "--cls-cond",
        action="store_true",
        help="Concat frozen 5-class softmax onto the latent embedding.",
    )
    cls_src.add_argument(
        "--cls-true",
        action="store_true",
        help="Concat true 5-class one-hot onto the latent embedding (oracle).",
    )
    parser.add_argument(
        "--cls-ckpt",
        type=Path,
        default=None,
        help="Classifier dir (default: data/pre_trained_models/daep_latent/<try>/<split>).",
    )
    args = parser.parse_args()
    if args.cls_true and args.cls_ckpt is not None:
        parser.error("--cls-ckpt is only used with --cls-cond, not --cls-true")
    if args.better_hps and args.deeper:
        parser.error("--deeper and --better-hps are mutually exclusive")
    if args.reweight_alpha is not None and not args.reweight:
        parser.error("--reweight-alpha requires --reweight")
    reweight_alpha = 0.5 if args.reweight_alpha is None else float(args.reweight_alpha)
    if args.reweight and reweight_alpha < 0:
        parser.error("--reweight-alpha must be >= 0")
    aug_extras = (
        args.aug_k,
        args.aug_eps_lo,
        args.aug_eps_hi,
        args.aug_encode_batch,
        args.aug_ckpt,
        args.aug_flux_dir,
    )
    if any(v is not None for v in aug_extras) or bool(args.aug_force):
        if not args.aug:
            parser.error("--aug-* flags require --aug")
    if args.aug and (args.cls_cond or args.cls_true):
        parser.error("--aug cannot be combined with --cls-cond / --cls-true")
    if args.aug_k is not None and int(args.aug_k) < 1:
        parser.error("--aug-k must be >= 1")
    aug_k = AUG_K if args.aug_k is None else int(args.aug_k)
    aug_eps_lo = AUG_EPS_LO if args.aug_eps_lo is None else float(args.aug_eps_lo)
    aug_eps_hi = AUG_EPS_HI if args.aug_eps_hi is None else float(args.aug_eps_hi)
    aug_encode_batch = AUG_ENCODE_BATCH if args.aug_encode_batch is None else int(args.aug_encode_batch)
    if args.aug and aug_eps_lo >= aug_eps_hi:
        parser.error("--aug-eps-lo must be < --aug-eps-hi")
    if aug_encode_batch < 1:
        parser.error("--aug-encode-batch must be >= 1")
    head = str(args.head)
    target = str(args.target)
    try:
        bins_arg, flow_hidden_layers, trunk_layers = resolve_deeper(
            deeper=bool(args.deeper),
            head=head,
            bins=args.bins,
        )
        bins, transforms, cover_z = resolve_better_hps(
            better_hps=bool(args.better_hps),
            target=target,
            head=head,
            bins=bins_arg,
            transforms=int(args.transforms),
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.better_hps:
        print(
            f"[better-hps] bins={bins} transforms={transforms} cover_z={cover_z} "
            f"(z<=2 on NSF u in [-5, 5]; 64 bins)",
            flush=True,
        )
    if args.deeper:
        print(
            f"[deeper] trunk_layers={trunk_layers}  flow_hidden_layers={flow_hidden_layers}  "
            f"bins={bins}  (default is 1 / 2 / 8)",
            flush=True,
        )
    if args.reweight:
        print(
            f"[reweight] alpha={reweight_alpha:g}  inverse-frequency NLL on physical-z bins; "
            f"eval metrics stay unweighted",
            flush=True,
        )
    if args.aug:
        print(
            f"[aug] k={aug_k}  ε~Unif({aug_eps_lo:g}, {aug_eps_hi:g}) on ln(1+z)  "
            f"train only; val/test stay real embeddings",
            flush=True,
        )
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
    out_root = (
        args.out_root
        if args.out_root is not None
        else default_out_root(
            head,
            target,
            cls_cond=bool(args.cls_cond),
            cls_true=bool(args.cls_true),
            better_hps=bool(args.better_hps),
            deeper=bool(args.deeper),
            reweight=bool(args.reweight),
            aug=bool(args.aug),
        )
    )

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
            bins=int(bins),
            transforms=int(transforms),
            flow_hidden=int(args.flow_hidden),
            flow_hidden_layers=int(flow_hidden_layers),
            trunk_layers=int(trunk_layers),
            clip_norm=clip_norm,
            eval_only=bool(args.eval_only),
            target=target,
            cls_cond=bool(args.cls_cond),
            cls_true=bool(args.cls_true),
            cls_ckpt=args.cls_ckpt,
            cover_z=cover_z,
            better_hps=bool(args.better_hps),
            deeper=bool(args.deeper),
            reweight=bool(args.reweight),
            reweight_alpha=reweight_alpha,
            aug=bool(args.aug),
            aug_k=aug_k,
            aug_eps_lo=aug_eps_lo,
            aug_eps_hi=aug_eps_hi,
            aug_force=bool(args.aug_force),
            aug_encode_batch=aug_encode_batch,
            aug_ckpt=args.aug_ckpt,
            aug_flux_dir=args.aug_flux_dir,
        )


if __name__ == "__main__":
    main()
