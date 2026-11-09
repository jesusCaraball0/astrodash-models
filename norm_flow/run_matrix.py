#!/usr/bin/env python3
"""Launch the redshift head matrix (latent × DASH).

Default is MSE / Gaussian / flow on split 36, training in ln z. Pass
``--heads moe`` for the soft two-expert mixture, ``--target z`` to train in
physical redshift (writes to ``*_z_*`` dirs, does not overwrite logz runs).

  python norm_flow/run_matrix.py --dry-run
  python norm_flow/run_matrix.py
  python norm_flow/run_matrix.py --heads moe
  python norm_flow/run_matrix.py --heads flow --deeper
  python norm_flow/run_matrix.py --heads flow --reweight
  python norm_flow/run_matrix.py --heads flow --aug
  python norm_flow/run_matrix.py --heads flow --target z --better-hps
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
from train_utils import output_tag

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
    seed: int,
    try_dir: Path,
    python: str,
    heads: list[str],
    target: str,
    *,
    cls_cond: bool = False,
    cls_true: bool = False,
    better_hps: bool = False,
    deeper: bool = False,
    reweight: bool = False,
    aug: bool = False,
) -> list[list[str]]:
    latent = str(latent_dir_for(try_dir, seed))
    splits = str(splits_json_for(seed))
    extra: list[str] = []
    if cls_cond:
        extra.append("--cls-cond")
    if cls_true:
        extra.append("--cls-true")
    if better_hps:
        extra.append("--better-hps")
    if deeper:
        extra.append("--deeper")
    if reweight:
        extra.append("--reweight")
    latent_extra = list(extra)
    if aug:
        latent_extra.append("--aug")
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
                *latent_extra,
            ]
        )
    if not aug:
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
                    *extra,
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
    cls_src = parser.add_mutually_exclusive_group()
    cls_src.add_argument(
        "--cls-cond",
        action="store_true",
        help="Condition on frozen 5-class softmax from the matching split classifier.",
    )
    cls_src.add_argument(
        "--cls-true",
        action="store_true",
        help="Condition on true 5-class one-hot labels (oracle; writes to *_truecls_*).",
    )
    parser.add_argument(
        "--better-hps",
        action="store_true",
        help="Linear-z NSF covering (z<=2 on the spline, 64 bins). Requires --target z. "
        "Writes to *_z_betterhps_{head}/ and does not overwrite original z runs.",
    )
    parser.add_argument(
        "--deeper",
        action="store_true",
        help="One extra trunk layer, 3-layer NSF conditioner, 16 bins. "
        "Writes to *_deeper_{head}/ and does not overwrite the default run.",
    )
    parser.add_argument(
        "--reweight",
        action="store_true",
        help="Inverse-frequency weighted NLL (alpha=0.5). Writes to *_reweight_{head}/.",
    )
    parser.add_argument(
        "--aug",
        action="store_true",
        help="Latent-only: 5 wavelength-stretched train copies (noise on ln(1+z)). "
        "Skips DASH. Writes to *_aug_{head}/.",
    )
    args = parser.parse_args()
    if args.aug and (args.cls_cond or args.cls_true):
        parser.error("--aug cannot be combined with --cls-cond / --cls-true")
    if args.better_hps and str(args.target) != "z":
        parser.error("--better-hps requires --target z")
    if args.deeper and args.better_hps:
        parser.error("--deeper and --better-hps are mutually exclusive")
    if args.deeper:
        for h in args.heads:
            if str(h) not in ("flow", "moe"):
                parser.error("--deeper only applies to flow / moe")
    try_dir = args.try_dir.expanduser().resolve()
    py = sys.executable
    heads = [str(h) for h in args.heads]
    target = str(args.target)
    all_cmds: list[list[str]] = []
    for seed in args.seeds:
        cmds = commands_for_seed(
            int(seed),
            try_dir,
            py,
            heads,
            target,
            cls_cond=bool(args.cls_cond),
            cls_true=bool(args.cls_true),
            better_hps=bool(args.better_hps),
            deeper=bool(args.deeper),
            reweight=bool(args.reweight),
            aug=bool(args.aug),
        )
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
    infix = output_tag(
        target,
        cls_cond=bool(args.cls_cond),
        cls_true=bool(args.cls_true),
        better_hps=bool(args.better_hps),
        deeper=bool(args.deeper),
        reweight=bool(args.reweight),
        aug=bool(args.aug),
    )
    print(f"\nDone. Outputs under data/pre_trained_models/daep_latent_{infix}_{{mse,gaussian,flow,moe}}/")
    if not args.aug:
        print(f"and data/pre_trained_models/dash_{infix}_{{mse,gaussian,flow,moe}}/")


if __name__ == "__main__":
    main()
