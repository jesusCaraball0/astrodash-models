#!/usr/bin/env python3
"""Train the website DASH 1D CNN (no redshift) on a 90/10 IAU train/val split.

Observed-frame preprocessing; same hparams as ``dash_retrain.py --no-redshift``.

Usage (from repo root):
  PYTHONPATH=zmodel_training python zmodel_training/train_website_dash_noz.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import constants as const
from train_website_dash import train_website_dash

HAS_REDSHIFT = False
HENNA_META = const.WISEREP_HENNA_NOZ / "meta.csv"
DEFAULT_SPLITS_JSON = const.WISEREP_DIR / "website_final_split_noz.json"
DEFAULT_OUT_DIR = (
    PROJECT_ROOT / "data" / "pre_trained_models" / "website_final" / "dash_noz"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Website-final DASH 1D CNN (no redshift), 90% train / 10% val by IAU."
    )
    parser.add_argument("--seed", type=int, default=const.SEED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--splits-json", type=Path, default=DEFAULT_SPLITS_JSON)
    parser.add_argument("--henna-meta", type=Path, default=HENNA_META)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
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
