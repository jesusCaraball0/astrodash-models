#!/usr/bin/env python3
"""
Train Dash 1D CNN redshift regressors (MSE, observed-frame preprocessing)
on the same Henna assignment splits as the no-z latent redshift models.

Reuses create_henna_matched_dash_split.py helpers to turn row indices into
ascii lists, then calls dash_retrain_redshift.py:

  python zmodel_training/dash_retrain_redshift.py \\
      --splits-json ... --seed 0 --out-dir ...

Default source is try_5_noz (try_6_noz 36/73/149 are the same membership;
Nodered257_6 has empty val/test and is skipped).

Usage:
  python zmodel_training/run_henna_noz_assignment_dash.py
  python zmodel_training/run_henna_noz_assignment_dash.py --seeds 36 73
  python zmodel_training/run_henna_noz_assignment_dash.py --target logz --skip-split-create
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import constants as const
from create_henna_matched_dash_split import (
    IAU_COLUMN,
    enrich_henna_meta,
    filter_indices_mapped,
    load_spec_id_to_ascii,
    split_payload_for_indices,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DASH_RETRAIN = SCRIPT_DIR / "dash_retrain_redshift.py"

TRY_DIR_DEFAULT = PROJECT_ROOT / "data" / "wiserep_henna" / "try_5_noz"
SPLITS_DIR = const.WISEREP_DIR
OUT_ROOT_Z = PROJECT_ROOT / "data" / "pre_trained_models" / "henna_matched_split_noz_redshift"
OUT_ROOT_LOGZ = PROJECT_ROOT / "data" / "pre_trained_models" / "henna_matched_split_noz_logz"
OUT_ROOT = OUT_ROOT_Z
SEEDS_DEFAULT = (36, 73, 149, 257)
TRAIN_RNG_SEED = 0
_SEED_DIR_RE = re.compile(r"(?:dered|nodered)(\d+)", re.IGNORECASE)


def splits_json_for_seed(seed: int) -> Path:
    return SPLITS_DIR / f"henna_matched_split_noz_seed{int(seed)}.json"


def out_root_for_target(target: str) -> Path:
    return OUT_ROOT_LOGZ if target == "logz" else OUT_ROOT_Z


def out_dir_for_seed(seed: int, target: str = "z") -> Path:
    return out_root_for_target(target) / f"split_{int(seed)}"


def load_assignment_indices(
    assignment_json: Path,
    n_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    payload = json.loads(assignment_json.read_text(encoding="utf-8"))
    train_idx = np.asarray(payload["train_idx"], dtype=np.int64)
    val_idx = np.asarray(payload["val_idx"], dtype=np.int64)
    test_idx = np.asarray(payload["test_idx"], dtype=np.int64)
    assignment_n = payload.get("n_rows")
    if assignment_n is not None and int(assignment_n) != n_rows:
        raise ValueError(
            f"Assignment n_rows={assignment_n} != meta rows={n_rows} ({assignment_json})"
        )
    if val_idx.size == 0 or test_idx.size == 0:
        return train_idx, val_idx, test_idx
    for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        if idx.size and (int(idx.min()) < 0 or int(idx.max()) >= n_rows):
            raise ValueError(f"{name} indices out of range for n_rows={n_rows} ({assignment_json})")
    return train_idx, val_idx, test_idx


def _require_disjoint(parts: dict[str, set[str]], kind: str) -> None:
    leaks = {
        f"{a}&{b}": sorted(parts[a] & parts[b])
        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
        if parts[a] & parts[b]
    }
    if leaks:
        preview = {
            k: v[:5] + ([f"...({len(v)} total)"] if len(v) > 5 else [])
            for k, v in leaks.items()
        }
        raise RuntimeError(f"{kind} leakage across splits: {preview}")


def write_assignment_split_json(
    *,
    meta_csv: Path,
    assignment_json: Path,
    out_path: Path,
    seed: int,
) -> dict:
    raw = pd.read_csv(meta_csv, low_memory=False).reset_index(drop=True)
    meta = enrich_henna_meta(raw, load_spec_id_to_ascii(const.METADATA_CSV))
    tr_idx, va_idx, te_idx = load_assignment_indices(assignment_json, n_rows=len(meta))
    if va_idx.size == 0 or te_idx.size == 0:
        print(
            f"Skipping {assignment_json}: empty val/test "
            f"(val={va_idx.size} test={te_idx.size})",
            flush=True,
        )
        return {}
    tr_idx = filter_indices_mapped(meta, tr_idx)
    va_idx = filter_indices_mapped(meta, va_idx)
    te_idx = filter_indices_mapped(meta, te_idx)
    train_part = split_payload_for_indices(meta, tr_idx, const.SPECTRA_DIR, True)
    val_part = split_payload_for_indices(meta, va_idx, const.SPECTRA_DIR, True)
    test_part = split_payload_for_indices(meta, te_idx, const.SPECTRA_DIR, True)
    iau = meta[IAU_COLUMN].astype(str).str.strip()
    _require_disjoint(
        {
            "train": set(iau.iloc[tr_idx].astype(str)),
            "val": set(iau.iloc[va_idx].astype(str)),
            "test": set(iau.iloc[te_idx].astype(str)),
        },
        "IAU name",
    )
    _require_disjoint(
        {
            "train": set(train_part["ascii_files"]),
            "val": set(val_part["ascii_files"]),
            "test": set(test_part["ascii_files"]),
        },
        "ascii file",
    )
    payload = {
        "split_method": "henna_universal_assignment_label_filtered",
        "seed": int(seed),
        "data_split_seed": int(seed),
        "henna_meta_csv": str(meta_csv.resolve()),
        "assignment_json": str(assignment_json.resolve()),
        "wiserep_metadata_csv": str(const.METADATA_CSV.resolve()),
        "spectra_dir": str(const.SPECTRA_DIR.resolve()),
        "counts": {
            "meta_rows": int(len(meta)),
            "mappable_train": int(tr_idx.size),
            "mappable_val": int(va_idx.size),
            "mappable_test": int(te_idx.size),
            "train": len(train_part["ascii_files"]),
            "val": len(val_part["ascii_files"]),
            "test": len(test_part["ascii_files"]),
        },
        "unique_iau": {
            "train": int(pd.unique(iau.iloc[tr_idx]).size) if tr_idx.size else 0,
            "val": int(pd.unique(iau.iloc[va_idx]).size) if va_idx.size else 0,
            "test": int(pd.unique(iau.iloc[te_idx]).size) if te_idx.size else 0,
        },
        "train": train_part["ascii_files"],
        "val": val_part["ascii_files"],
        "test": test_part["ascii_files"],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"wrote {out_path.name}: train={payload['counts']['train']} "
        f"val={payload['counts']['val']} test={payload['counts']['test']}",
        flush=True,
    )
    return payload


def discover_seed_dirs(try_dir: Path, seeds: list[int]) -> list[tuple[int, Path, Path]]:
    wanted = set(int(s) for s in seeds)
    found: list[tuple[int, Path, Path]] = []
    for sub in sorted(p for p in try_dir.iterdir() if p.is_dir()):
        match = _SEED_DIR_RE.search(sub.name)
        if match is None:
            continue
        seed = int(match.group(1))
        if seed not in wanted:
            continue
        meta_csv = sub / "meta_universal.csv"
        assign = sub / f"split_assignment{seed}.json"
        if not assign.is_file():
            matches = sorted(sub.glob("split_assignment*.json"))
            assign = matches[0] if matches else assign
        if meta_csv.is_file() and assign.is_file():
            if assign.name != f"split_assignment{seed}.json":
                raise RuntimeError(
                    f"{sub.name}: expected split_assignment{seed}.json, found {assign.name}"
                )
            found.append((seed, assign, meta_csv))
    missing = sorted(wanted - {seed for seed, _, _ in found})
    if missing:
        raise FileNotFoundError(f"No assignment+meta dirs for seeds {missing} under {try_dir}")
    return found


def train_one(
    split_seed: int,
    splits_json: Path,
    *,
    target: str = "z",
    z_floor: float = 1e-4,
) -> None:
    if not splits_json.is_file():
        print(f"Skipping seed={split_seed}: missing {splits_json}")
        return
    payload = json.loads(splits_json.read_text(encoding="utf-8"))
    json_split_seed = payload.get("data_split_seed", payload.get("seed"))
    if json_split_seed is not None and int(json_split_seed) != int(split_seed):
        raise RuntimeError(
            f"{splits_json.name} has data_split_seed={json_split_seed}, expected {split_seed}"
        )
    if not payload.get("val") or not payload.get("test"):
        print(f"Skipping seed={split_seed}: empty val/test in {splits_json}")
        return
    print(
        f"[split] data_split_seed={split_seed} from {splits_json.name}  "
        f"train_rng_seed={TRAIN_RNG_SEED} target={target} "
        f"(CNN RNG, not the split)",
        flush=True,
    )
    cmd = [
        sys.executable,
        str(DASH_RETRAIN),
        "--splits-json",
        str(splits_json),
        "--seed",
        str(TRAIN_RNG_SEED),
        "--out-dir",
        str(out_dir_for_seed(split_seed, target)),
        "--target",
        str(target),
        "--z-floor",
        str(z_floor),
    ]
    print("\n" + "=" * 72)
    print(" ".join(cmd))
    print("=" * 72)
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dash 1D CNN redshift regressor on Henna no-z assignment splits."
    )
    parser.add_argument(
        "--try-dir",
        type=Path,
        default=TRY_DIR_DEFAULT,
        help="Assignment try dir (default: data/wiserep_henna/try_5_noz).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help=f"Assignment seeds (default: {list(SEEDS_DEFAULT)}).",
    )
    parser.add_argument(
        "--skip-split-create",
        action="store_true",
        help="Reuse existing henna_matched_split_noz_seed*.json files.",
    )
    parser.add_argument(
        "--target",
        choices=("z", "logz"),
        default="z",
        help="Regression target. 'logz' writes under henna_matched_split_noz_logz.",
    )
    parser.add_argument(
        "--z-floor",
        type=float,
        default=1e-4,
        help="Lower clamp on z before log (logz only; default: 1e-4).",
    )
    args = parser.parse_args()
    try_dir = args.try_dir.expanduser().resolve()
    seeds = list(args.seeds) if args.seeds is not None else list(SEEDS_DEFAULT)
    target = str(args.target)
    z_floor = float(args.z_floor)
    out_root = out_root_for_target(target)

    if not args.skip_split_create:
        for seed, assignment_json, meta_csv in discover_seed_dirs(try_dir, seeds):
            write_assignment_split_json(
                meta_csv=meta_csv,
                assignment_json=assignment_json,
                out_path=splits_json_for_seed(seed),
                seed=seed,
            )

    for seed in seeds:
        train_one(seed, splits_json_for_seed(seed), target=target, z_floor=z_floor)

    print("\nDone. Models under:")
    print(f"  {out_root}/split_{{seed}}/")
    print("True vs predicted redshift:")
    print(f"  python zmodel_training/redshift_pred_plots.py {out_root}")


if __name__ == "__main__":
    main()
