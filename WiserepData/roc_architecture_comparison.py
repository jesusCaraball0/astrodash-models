"""
Micro-average ROC across the six controlled five-class WISeREP variants.

Each ensemble is the four data splits (seeds 36, 73, 149, 257). Every run is
scored on that split's own test set. Original DASH and the transformer are
not included.

  DAEP diffusion latent ±z  → data/pre_trained_models/daep_latent/try_5(_noz)
  DAEP no diffusion ±z      → data/pre_trained_models/daep_init/try_5(_noz)
  1D CNN ±z                 → data/pre_trained_models/henna_matched_split_{z,noz}

Usage:
  python WiserepData/roc_architecture_comparison.py
  python WiserepData/roc_architecture_comparison.py path/to/out.png
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, f1_score, roc_curve
from sklearn.preprocessing import label_binarize

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ZMODEL_DIR = PROJECT_ROOT / "zmodel_training"
PRETRAINED = PROJECT_ROOT / "data" / "pre_trained_models"

for path in (SCRIPT_DIR, PROJECT_ROOT, ZMODEL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import roc_ensemble_daep_comparison as dash_roc
import roc_ensemble_wiserep as wiserep_roc

FPR_GRID = np.linspace(0.0, 1.0, 101)
DEFAULT_OUTPUT = SCRIPT_DIR / "Test" / "architecture_comparison_micro.png"


@dataclass(frozen=True)
class EnsembleSpec:
    label: str
    root: Path
    kind: str
    has_redshift: bool


@dataclass
class EnsembleResult:
    label: str
    root: Path
    n_total: int
    n_unique: int
    f1_mean: float
    f1_std: float
    auc_mean: float
    auc_std: float
    tpr_mean: np.ndarray
    tpr_std: np.ndarray


# Legend order matches mean test accuracy.
SPECS = [
    EnsembleSpec(
        "DAEP diffusion latent with redshift",
        PRETRAINED / "daep_latent" / "try_5",
        "wiserep",
        True,
    ),
    EnsembleSpec(
        "1D CNN with redshift",
        PRETRAINED / "henna_matched_split_z",
        "dash",
        True,
    ),
    EnsembleSpec(
        "DAEP diffusion latent without redshift",
        PRETRAINED / "daep_latent" / "try_5_noz",
        "wiserep",
        False,
    ),
    EnsembleSpec(
        "DAEP no diffusion with redshift",
        PRETRAINED / "daep_init" / "try_5",
        "wiserep",
        True,
    ),
    EnsembleSpec(
        "1D CNN without redshift",
        PRETRAINED / "henna_matched_split_noz",
        "dash",
        False,
    ),
    EnsembleSpec(
        "DAEP no diffusion without redshift",
        PRETRAINED / "daep_init" / "try_5_noz",
        "wiserep",
        False,
    ),
]


def std_ddof1(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size <= 1:
        return 0.0
    return float(np.std(x, ddof=1))


def _prepare_wiserep_globals(has_redshift: bool) -> None:
    wiserep_roc.HAS_REDSHIFT = has_redshift
    wiserep_roc.DATA_DIR = wiserep_roc.data_dir_for(has_redshift)
    wiserep_roc.LATENT_NPZ_FALLBACK = wiserep_roc.DATA_DIR / "latent_raw_z.npz"


def summarize_micro_ensemble(
    all_y_true: list[np.ndarray],
    all_scores: list[np.ndarray],
) -> tuple[float, float, float, float, np.ndarray, np.ndarray]:
    """Mean micro ROC on a shared FPR grid. Each run keeps its own test labels."""
    n_classes = int(all_scores[0].shape[1])
    f1_runs: list[float] = []
    auc_runs: list[float] = []
    tpr_rows: list[np.ndarray] = []

    for y_true, y_score in zip(all_y_true, all_scores):
        y_bin = label_binarize(y_true, classes=np.arange(n_classes))
        if y_bin.ndim == 1:
            y_bin = np.column_stack([1 - y_bin, y_bin])
        f1_runs.append(
            float(f1_score(y_true, np.argmax(y_score, axis=1), average="micro", zero_division=0))
        )
        fpr, tpr, _ = roc_curve(y_bin.ravel(), y_score.ravel())
        tpr_interp = np.interp(FPR_GRID, fpr, tpr)
        tpr_interp[0] = 0.0
        tpr_interp[-1] = 1.0
        tpr_rows.append(tpr_interp)
        auc_runs.append(float(auc(fpr, tpr)))

    tpr_stack = np.stack(tpr_rows, axis=0)
    tpr_mean = np.mean(tpr_stack, axis=0)
    tpr_std = np.std(tpr_stack, axis=0, ddof=1) if len(tpr_rows) > 1 else np.zeros_like(tpr_mean)
    return (
        float(np.mean(f1_runs)),
        std_ddof1(np.array(f1_runs)),
        float(np.mean(auc_runs)),
        std_ddof1(np.array(auc_runs)),
        tpr_mean,
        tpr_std,
    )


def load_ensemble_result(spec: EnsembleSpec, device) -> EnsembleResult:
    print(f"\n{spec.label}")
    print(f"  {spec.root}")
    if spec.kind == "wiserep":
        _prepare_wiserep_globals(spec.has_redshift)
        all_dirs = wiserep_roc.discover_all_ckpt_dirs(spec.root, run_style="split")
        if not all_dirs:
            raise FileNotFoundError(
                f"No split_* dirs with {wiserep_roc.CKPT_NAME} under {spec.root}"
            )
        run_dirs, n_total = wiserep_roc.dedupe_dirs_by_checkpoint(all_dirs)
        all_y_true, all_scores = wiserep_roc.run_models(run_dirs, device)
    elif spec.kind == "dash":
        all_dirs = [
            run_dir
            for run_dir in dash_roc.discover_run_dirs(spec.root)
            if (run_dir / dash_roc.CKPT_NAME).is_file()
        ]
        if not all_dirs:
            raise FileNotFoundError(
                f"No split_* dirs with {dash_roc.CKPT_NAME} under {spec.root}"
            )
        run_dirs, n_total = all_dirs, len(all_dirs)
        all_y_true, all_scores, class_names = dash_roc.collect_run_predictions(run_dirs, device)
        if class_names != wiserep_roc.CLASS_NAMES:
            raise RuntimeError(
                f"Dash class names differ from WISeREP class names: "
                f"{class_names} vs {wiserep_roc.CLASS_NAMES}"
            )
    else:
        raise ValueError(f"Unknown ensemble kind: {spec.kind}")

    f1_mean, f1_std, auc_mean, auc_std, tpr_mean, tpr_std = summarize_micro_ensemble(
        all_y_true,
        all_scores,
    )
    return EnsembleResult(
        label=spec.label,
        root=spec.root,
        n_total=n_total,
        n_unique=len(run_dirs),
        f1_mean=f1_mean,
        f1_std=f1_std,
        auc_mean=auc_mean,
        auc_std=auc_std,
        tpr_mean=tpr_mean,
        tpr_std=tpr_std,
    )


def fmt_sd(x: float) -> str:
    return f"{x:.4f}" if x >= 1e-4 else f"{x:.2e}"


def main(output_path: Path | None = None) -> None:
    output = (output_path or DEFAULT_OUTPUT).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    device = dash_roc.helpers.get_device()
    results = [load_ensemble_result(spec, device) for spec in SPECS]

    fig, ax = plt.subplots(figsize=(10, 8))
    for result in results:
        (line,) = ax.plot(
            FPR_GRID,
            result.tpr_mean,
            lw=2,
            label=(
                f"{result.label} (AUC={result.auc_mean:.3f}±{fmt_sd(result.auc_std)}, "
                f"micro F1={result.f1_mean:.3f}±{fmt_sd(result.f1_std)})"
            ),
        )
        ax.fill_between(
            FPR_GRID,
            np.clip(result.tpr_mean - result.tpr_std, 0, 1),
            np.clip(result.tpr_mean + result.tpr_std, 0, 1),
            color=line.get_color(),
            alpha=0.15,
        )

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Micro-average ROC — controlled five-class comparison")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved {output}")
    print("Micro F1 summary:")
    for result in results:
        skipped = result.n_total - result.n_unique
        print(
            f"  {result.label}: micro F1 = {result.f1_mean:.4f} ± {fmt_sd(result.f1_std)}"
            f" | micro AUC = {result.auc_mean:.4f} ± {fmt_sd(result.auc_std)}"
            f" | unique runs = {result.n_unique}/{result.n_total}"
            f" | duplicates skipped = {skipped}"
        )


if __name__ == "__main__":
    arg_output = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    main(arg_output)
