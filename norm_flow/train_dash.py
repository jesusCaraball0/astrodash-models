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

from class_cond import (
    N_CLASSES,
    dash_class_onehot_by_id,
    dash_class_probs_by_id,
    default_dash_cls_dir,
    filter_mapped_files,
    load_dash_classifier,
)
from heads import FLOW_LIKE_HEADS, HEADS
from leakage import assert_disjoint_ids, assert_observed_frame_splits, document_z_handling
from models import TARGETS, DashCNNTrunk, RedshiftPredictor
from train_utils import Z_FLOOR, Z_MAX, output_tag, resolve_better_hps, resolve_deeper, train_and_eval

for _name in ("spectrum_io", "dash_preprocess"):
    logging.getLogger(_name).setLevel(logging.CRITICAL)


def default_out_dir(
    head: str,
    split_seed: Any,
    target: str = "logz",
    cls_cond: bool = False,
    cls_true: bool = False,
    better_hps: bool = False,
    deeper: bool = False,
    reweight: bool = False,
) -> Path:
    infix = output_tag(
        target,
        cls_cond=cls_cond,
        cls_true=cls_true,
        better_hps=better_hps,
        deeper=deeper,
        reweight=reweight,
    )
    return PROJECT_ROOT / "data" / "pre_trained_models" / f"dash_{infix}_{head}" / f"split_{split_seed}"


class DictDashDataset(Dataset):
    def __init__(
        self,
        inner: dash_z.WISeREPRedshiftDataset,
        cls_by_id: Optional[Dict[str, torch.Tensor]] = None,
    ):
        self.inner = inner
        self.cls_by_id = cls_by_id

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, i: int) -> Optional[Dict[str, Any]]:
        item = self.inner[i]
        if item is None:
            return None
        x, z_val = item
        fname = self.inner.samples[i][0]
        x = x.float()
        if self.cls_by_id is not None:
            p = self.cls_by_id.get(fname)
            if p is None:
                return None
            x = torch.cat([x, p.float()], dim=0)
        return {"x": x, "y": torch.tensor(float(z_val), dtype=torch.float32), "id": fname}


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
    parser.add_argument(
        "--bins",
        type=int,
        default=None,
        help="Spline bins (flow / moe). Default 8, 16 with --deeper, or 64 with --better-hps.",
    )
    parser.add_argument("--transforms", type=int, default=2)
    parser.add_argument("--flow-hidden", type=int, default=256)
    parser.add_argument(
        "--better-hps",
        action="store_true",
        help="Linear-z NSF: put z<=2 on the spline and use 64 bins. "
        "Writes to dash_z_betterhps_{head}/split_<seed>/ and does not overwrite the original z run.",
    )
    parser.add_argument(
        "--deeper",
        action="store_true",
        help="Bigger NSF: one extra FC after the CNN trunk, 3-layer conditioner, 16 bins. "
        "Writes to dash_{target}_deeper_{head}/ and does not overwrite the default run.",
    )
    parser.add_argument(
        "--reweight",
        action="store_true",
        help="Train with inverse-frequency weighted NLL: w_b ∝ N_b^{-alpha} on physical-z "
        "bins (alpha=0.5). Eval NLL/MAE stay unweighted. "
        "Writes to dash_{target}_reweight_{head}/ and does not overwrite the default run.",
    )
    parser.add_argument(
        "--reweight-alpha",
        type=float,
        default=None,
        help="Exponent for --reweight (default 0.5). Ignored unless --reweight is set.",
    )
    parser.add_argument("--clip-norm", type=float, default=None)
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training; write metrics/plots from model_best.pt.",
    )
    cls_src = parser.add_mutually_exclusive_group()
    cls_src.add_argument(
        "--cls-cond",
        action="store_true",
        help="Concat frozen 5-class softmax after the CNN trunk.",
    )
    cls_src.add_argument(
        "--cls-true",
        action="store_true",
        help="Concat true 5-class one-hot after the CNN trunk (oracle).",
    )
    parser.add_argument(
        "--cls-ckpt",
        type=Path,
        default=None,
        help="Classifier dir (default: data/pre_trained_models/henna_matched_split_noz/split_<seed>).",
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

    helpers.set_seed(int(args.seed))
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
            f"[deeper] extra_fc={trunk_layers > 1}  flow_hidden_layers={flow_hidden_layers}  "
            f"bins={bins}  (default is 1 / 2 / 8)",
            flush=True,
        )
    if args.reweight:
        print(
            f"[reweight] alpha={reweight_alpha:g}  inverse-frequency NLL on physical-z bins; "
            f"eval metrics stay unweighted",
            flush=True,
        )
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
        else default_out_dir(
            head,
            split_seed,
            target,
            cls_cond=bool(args.cls_cond),
            cls_true=bool(args.cls_true),
            better_hps=bool(args.better_hps),
            deeper=bool(args.deeper),
            reweight=bool(args.reweight),
        )
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

    cls_by_id: Optional[Dict[str, torch.Tensor]] = None
    cls_dim = 0
    use_cls = bool(args.cls_cond or args.cls_true)
    if use_cls:
        n_tr, n_va, n_te = len(train_files), len(val_files), len(test_files)
        train_files = filter_mapped_files(train_files, metadata)
        val_files = filter_mapped_files(val_files, metadata)
        test_files = filter_mapped_files(test_files, metadata)
        print(
            f"[cls] labeled ascii  train={len(train_files)}/{n_tr}  "
            f"val={len(val_files)}/{n_va}  test={len(test_files)}/{n_te}",
            flush=True,
        )

    def _inner(files: List[str]):
        return dash_z.WISeREPRedshiftDataset(
            files, const.SPECTRA_DIR, metadata, flux_cache=flux_cache
        )

    train_inner, val_inner, test_inner = _inner(train_files), _inner(val_files), _inner(test_files)
    if args.cls_true:
        cls_by_id = {}
        cls_by_id.update(dash_class_onehot_by_id(train_files, metadata))
        cls_by_id.update(dash_class_onehot_by_id(val_files, metadata))
        cls_by_id.update(dash_class_onehot_by_id(test_files, metadata))
        cls_dim = N_CLASSES
        print(f"[cls] concat true one-hot after CNN trunk  class_dim={cls_dim}", flush=True)
    elif args.cls_cond:
        ckpt_dir = (
            args.cls_ckpt.resolve()
            if args.cls_ckpt is not None
            else default_dash_cls_dir(split_seed)
        )
        clf = load_dash_classifier(ckpt_dir, device)
        cls_by_id = {}
        cls_by_id.update(dash_class_probs_by_id(clf, train_inner, device))
        cls_by_id.update(dash_class_probs_by_id(clf, val_inner, device))
        cls_by_id.update(dash_class_probs_by_id(clf, test_inner, device))
        cls_dim = N_CLASSES
        print(f"[cls] concat p_cls after CNN trunk  class_dim={cls_dim}", flush=True)

    train_ds = DictDashDataset(train_inner, cls_by_id)
    val_ds = DictDashDataset(val_inner, cls_by_id)
    test_ds = DictDashDataset(test_inner, cls_by_id)
    train_load = make_loader(train_ds, True, device)
    val_load = make_loader(val_ds, False, device)
    test_load = make_loader(test_ds, False, device)
    print(
        f"Effective sizes: train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}",
        flush=True,
    )

    model = RedshiftPredictor(
        DashCNNTrunk(const.TARGET_LENGTH, extra_fc=bool(args.deeper)),
        head,
        bins=int(bins),
        transforms=int(transforms),
        flow_hidden=int(args.flow_hidden),
        flow_hidden_layers=int(flow_hidden_layers),
        target=target,
        class_dim=cls_dim,
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
        "flow_bins": int(bins),
        "flow_transforms": int(transforms),
        "flow_hidden": int(args.flow_hidden),
        "flow_hidden_layers": int(flow_hidden_layers),
        "trunk_layers": int(trunk_layers),
        "clip_norm": clip_norm,
        "moe_experts": 2 if head == "moe" else None,
        "z_handling": document_z_handling(z_floor=float(args.z_floor), z_max=Z_MAX, target=target),
        "cls_cond": bool(args.cls_cond),
        "cls_true": bool(args.cls_true),
        "cls_source": "onehot" if args.cls_true else ("softmax" if args.cls_cond else None),
        "better_hps": bool(args.better_hps),
        "deeper": bool(args.deeper),
        "reweight": bool(args.reweight),
        "reweight_alpha": reweight_alpha if args.reweight else None,
        "cover_z": cover_z,
        "cls_ckpt": str(args.cls_ckpt.resolve()) if args.cls_ckpt is not None else (
            str(default_dash_cls_dir(split_seed)) if args.cls_cond else None
        ),
        "class_dim": cls_dim,
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
        cover_z=cover_z,
        reweight=bool(args.reweight),
        reweight_alpha=reweight_alpha,
    )


if __name__ == "__main__":
    main()
