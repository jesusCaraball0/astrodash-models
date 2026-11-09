#!/usr/bin/env python3
"""Train the website latent MLP (no-z embeddings) on a 90/10 IAU train/val split.

Frozen Universal embeddings from a try_*_noz dir; same hparams as ``train_latent.py``.

Usage (from repo root):
  PYTHONPATH=WiserepData python WiserepData/train_website_latent_noz.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

WISEREP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WISEREP_DIR.parent
if str(WISEREP_DIR) not in sys.path:
    sys.path.insert(0, str(WISEREP_DIR))

from train_latent import ITER10_RECIPE, TRAIN_RNG_SEED
from train_website_latent import train_website_latent

USE_REDSHIFT = False
DEFAULT_LATENT_DIR = PROJECT_ROOT / "data" / "wiserep_henna" / "try_5_noz" / "Nodered36_5"
DEFAULT_OUT_DIR = (
    PROJECT_ROOT / "data" / "pre_trained_models" / "website_final" / "latent_noz"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Website-final latent MLP (no-z), 90% train / 10% val by IAU."
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
    args = parser.parse_args()

    train_overrides = {}
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

    train_website_latent(
        latent_dir=args.latent_dir,
        out_dir=args.out_dir,
        train_seed=args.train_seed,
        train_overrides=train_overrides or None,
        use_redshift=USE_REDSHIFT,
    )


if __name__ == "__main__":
    main()
