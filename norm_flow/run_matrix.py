#!/usr/bin/env python3
"""Launch the redshift head matrix (latent × DASH).

Default is MSE / Gaussian / flow on split 36, training in ln z. Pass
``--heads moe`` for the soft two-expert mixture, ``--target z`` to train in
physical redshift (writes to ``*_z_*`` dirs, does not overwrite logz runs).

  python norm_flow/run_matrix.py --dry-run
  python norm_flow/run_matrix.py
  python norm_flow/run_matrix.py --heads moe
  python norm_flow/run_matrix.py --heads mse gaussian flow moe --target z
  python norm_flow/run_matrix.py --seeds 36 73 149 257
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from paths import PROJECT_ROOT, setup_imports

setup_imports()

from heads import HEADS
from models import TARGETS

LATENT_SCRIPT = Path(__file__).resolve().parent / "train_latent.py"
DASH_SCRIPT = Path(__file__).resolve().parent / "train_dash.py"
DEFAULT_TRY = PROJECT_ROOT / "data" / "wiserep_henna" / "try_5_noz"
DEFAULT_SEEDS = (36,)
DEFAULT_HEADS = ("mse", "gaussian", "flow")


def latent_dir_for(try_dir: Path, seed: int) -> Path:
    matches = sorted(try_dir.glob(f"Nodered{int(seed)}_*"))
    if not matches:
        raise FileNotFoundError(f"No Nodered{seed}_* under {try_dir}")
    return matches[0]


def splits_json_for(seed: int) -> Path:
    return PROJECT_ROOT / "data" / "wiserep" / f"henna_matched_split_noz_seed{int(seed)}.json"


def commands_for_seed(
    seed: int, try_dir: Path, python: str, heads: list[str], target: str
) -> list[list[str]]:
    latent = str(latent_dir_for(try_dir, seed))
    splits = str(splits_json_for(seed))
    cmds: list[list[str]] = []
    for head in heads:
        cmds.append(
            [
                python,
                str(LATENT_SCRIPT),
                "--head",
                head,
                "--target",
                target,
                "--latent-dirs",
                latent,
            ]
        )
    for head in heads:
        cmds.append(
            [
                python,
                str(DASH_SCRIPT),
                "--head",
                head,
                "--target",
                target,
                "--splits-json",
                splits,
            ]
        )
    return cmds


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the redshift head comparison.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--heads", nargs="+", choices=HEADS, default=list(DEFAULT_HEADS))
    parser.add_argument(
        "--target",
        choices=TARGETS,
        default="logz",
        help="Training-space label: ln z (default) or physical z.",
    )
    parser.add_argument("--try-dir", type=Path, default=DEFAULT_TRY)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-latent", action="store_true")
    parser.add_argument("--skip-dash", action="store_true")
    args = parser.parse_args()
    try_dir = args.try_dir.expanduser().resolve()
    py = sys.executable
    heads = [str(h) for h in args.heads]
    target = str(args.target)
    all_cmds: list[list[str]] = []
    for seed in args.seeds:
        cmds = commands_for_seed(int(seed), try_dir, py, heads, target)
        if args.skip_latent:
            cmds = [c for c in cmds if "train_latent.py" not in c[1]]
        if args.skip_dash:
            cmds = [c for c in cmds if "train_dash.py" not in c[1]]
        all_cmds.extend(cmds)

    print(f"{len(all_cmds)} job(s). Independent; can be parallelized across terminals.\n")
    for cmd in all_cmds:
        print(" ".join(cmd))
    if args.dry_run:
        return
    for i, cmd in enumerate(all_cmds, 1):
        print(f"\n=== job {i}/{len(all_cmds)} ===")
        subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
    tag = "logz" if target == "logz" else "z"
    print(f"\nDone. Outputs under data/pre_trained_models/daep_latent_{tag}_{{mse,gaussian,flow,moe}}/")
    print(f"and data/pre_trained_models/dash_{tag}_{{mse,gaussian,flow,moe}}/")


if __name__ == "__main__":
    main()
