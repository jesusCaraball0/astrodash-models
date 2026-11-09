#!/usr/bin/env python3
"""Test redshift MAE by class for unconditioned / predicted-class / true-class.

Latent logz flow on split 36. Predicted class is the frozen 5-class softmax;
true class is the mapped one-hot (oracle).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paths import PROJECT_ROOT, setup_imports

setup_imports()

from train_latent import (  # noqa: E402
    CLASS_NAMES,
    LABEL_COLUMN,
    normalize_latent_meta,
    resolve_latent_meta_csv,
)
from TwinsClassifier_Wiserep import row_class_idx  # noqa: E402

CONDITIONS = (
    ("none", "No class", "#0072B2"),
    ("pred", "Predicted class", "#E69F00"),
    ("true", "True class", "#009E73"),
)
RUN_DIRS = {
    "none": "daep_latent_logz_flow",
    "pred": "daep_latent_logz_cls_flow",
    "true": "daep_latent_logz_truecls_flow",
}
DEFAULT_LATENT = PROJECT_ROOT / "data" / "wiserep_henna" / "try_5_noz" / "Nodered36_5"
DEFAULT_OUT = PROJECT_ROOT / "plans" / "class_cond_mae.png"
MODELS_ROOT = PROJECT_ROOT / "data" / "pre_trained_models"
SPLIT = Path("try_5_noz") / "split_36"


def _csv(kind: str) -> Path:
    return MODELS_ROOT / RUN_DIRS[kind] / SPLIT / "test_predictions.csv"


def load_mae_by_class(latent_dir: Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    meta = normalize_latent_meta(pd.read_csv(resolve_latent_meta_csv(latent_dir), low_memory=False))
    frames = {}
    id_sets = []
    for kind, _label, _color in CONDITIONS:
        df = pd.read_csv(_csv(kind))
        df["id"] = df["id"].astype(str)
        frames[kind] = df.set_index("id")
        id_sets.append(set(df["id"]))
    shared = sorted(set.intersection(*id_sets), key=lambda s: int(s))
    y = np.array([row_class_idx(meta, int(i), LABEL_COLUMN) for i in shared], dtype=np.int64)
    if (y < 0).any():
        keep = y >= 0
        shared = [s for s, k in zip(shared, keep) if k]
        y = y[keep]
    mae = {kind: np.full(len(CLASS_NAMES), np.nan) for kind, _l, _c in CONDITIONS}
    counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    for c, _name in enumerate(CLASS_NAMES):
        ids = [shared[i] for i in range(len(shared)) if y[i] == c]
        counts[c] = len(ids)
        if not ids:
            continue
        z_true = frames["none"].loc[ids, "z_true"].to_numpy(dtype=np.float64)
        for kind, _l, _c in CONDITIONS:
            z_pred = frames[kind].loc[ids, "z_median"].to_numpy(dtype=np.float64)
            mae[kind][c] = float(np.mean(np.abs(z_pred - z_true)))
    return mae, counts


def save_figure(mae: dict[str, np.ndarray], counts: np.ndarray, out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(CLASS_NAMES), figsize=(11.4, 4.0), layout="constrained")
    x = np.arange(len(CONDITIONS))
    handles = []
    for ax, c, name in zip(axes, range(len(CLASS_NAMES)), CLASS_NAMES):
        vals = [mae[kind][c] for kind, _l, _col in CONDITIONS]
        ymax = max(vals) if vals else 0.0
        ax.set_ylim(0.0, ymax * 1.22 if ymax > 0 else 1.0)
        for i, (kind, label, color) in enumerate(CONDITIONS):
            bar = ax.bar(i, mae[kind][c], color=color, width=0.72, linewidth=0, label=label)
            if c == 0:
                handles.append(bar)
            ax.text(i, mae[kind][c], f"{mae[kind][c]:.4f}", ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks([])
        ax.set_title(f"{name}\nn={int(counts[c])}", fontsize=10)
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        if c == 0:
            ax.set_ylabel("Test MAE ($z$)")

    fig.suptitle("Redshift MAE by class", fontsize=11)
    fig.legend(
        handles,
        [lab for _k, lab, _c in CONDITIONS],
        loc="outside lower center",
        frameon=False,
        ncols=3,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-dir", type=Path, default=DEFAULT_LATENT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    mae, counts = load_mae_by_class(args.latent_dir.resolve())
    print("class        n   no-class  predicted  true-class")
    for c, name in enumerate(CLASS_NAMES):
        print(
            f"{name:10s} {int(counts[c]):4d}  "
            f"{mae['none'][c]:8.4f}  {mae['pred'][c]:9.4f}  {mae['true'][c]:10.4f}"
        )
    save_figure(mae, counts, args.out.resolve())


if __name__ == "__main__":
    main()
