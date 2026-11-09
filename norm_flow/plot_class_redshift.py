#!/usr/bin/env python3
"""Class vs spectroscopic redshift: talk figure for the cls-cond story.

One shared log-z axis (overlaid densities) plus p(class | z-bin). Do not use
five independent histograms — the claim is about overlap.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, kruskal

from paths import PROJECT_ROOT, setup_imports

setup_imports()

from train_latent import (  # noqa: E402
    CLASS_NAMES,
    LABEL_COLUMN,
    load_assignment_indices_from_dir,
    normalize_latent_meta,
    resolve_latent_meta_csv,
)
from train_latent_redshift import (  # noqa: E402
    ensure_redshift_column,
    filter_indices_finite_redshift,
    redshift_values,
)
from TwinsClassifier_Wiserep import filter_indices_mapped, row_class_idx  # noqa: E402

CLASS_COLORS = {
    "SN Ia": "#0072B2",
    "SN Ib/c": "#E69F00",
    "SN II": "#009E73",
    "SN IIn": "#D55E00",
    "SLSN-I": "#CC79A7",
}

Z_BINS = (0.0, 0.05, 0.10, 0.20, 0.40, 0.80, 6.0)
Z_BIN_LABELS = (
    "[0, 0.05)",
    "[0.05, 0.10)",
    "[0.10, 0.20)",
    "[0.20, 0.40)",
    "[0.40, 0.80)",
    "[0.80, 6)",
)
DEFAULT_LATENT = PROJECT_ROOT / "data" / "wiserep_henna" / "try_5_noz" / "Nodered36_5"
DEFAULT_OUT = PROJECT_ROOT / "plans" / "class_vs_redshift.png"
DEFAULT_SPLITS_OUT = PROJECT_ROOT / "plans" / "class_vs_redshift_splits.png"


def _prep_split(meta: pd.DataFrame, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    idx = filter_indices_finite_redshift(meta, idx)
    idx = filter_indices_mapped(meta, idx, LABEL_COLUMN)
    z = redshift_values(meta)[idx]
    y = np.array([row_class_idx(meta, int(i), LABEL_COLUMN) for i in idx], dtype=np.int64)
    return z, y


def load_class_z_splits(latent_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    meta = normalize_latent_meta(pd.read_csv(resolve_latent_meta_csv(latent_dir), low_memory=False))
    meta = ensure_redshift_column(meta)
    train_idx, val_idx, test_idx, _src = load_assignment_indices_from_dir(latent_dir, meta)
    return {
        "Train": _prep_split(meta, train_idx),
        "Val": _prep_split(meta, val_idx),
        "Test": _prep_split(meta, test_idx),
    }


def class_stats(z: np.ndarray, y: np.ndarray) -> None:
    logz = np.log(np.clip(z, 1e-4, None))
    print(f"n={len(z)}  median z={np.median(z):.4f}  frac z<0.05={(z < 0.05).mean():.3f}")
    pred_logz = np.zeros_like(logz)
    pred_z = np.zeros_like(z)
    groups = []
    for c, name in enumerate(CLASS_NAMES):
        m = y == c
        n = int(m.sum())
        zz = z[m]
        print(
            f"  {name:8s} n={n:5d} ({100 * n / len(z):4.1f}%)  "
            f"med={np.median(zz):.4f}  p90={np.percentile(zz, 90):.4f}  "
            f"z>=0.2={(zz >= 0.2).mean():.3f}"
        )
        pred_logz[m] = logz[m].mean()
        pred_z[m] = np.median(zz)
        groups.append(logz[m])
    h, p = kruskal(*groups)
    r2 = 1.0 - float(np.sum((logz - pred_logz) ** 2) / np.sum((logz - logz.mean()) ** 2))
    mae = float(np.mean(np.abs(z - pred_z)))
    mae0 = float(np.mean(np.abs(z - np.median(z))))
    print(f"Kruskal-Wallis H={h:.0f} p={p:.1e}")
    print(f"class-median predictor  R2(ln z)={r2:.3f}  MAE={mae:.4f}  (global-median MAE={mae0:.4f})")


def _kde_line(logz: np.ndarray, grid: np.ndarray) -> np.ndarray:
    kde = gaussian_kde(logz)
    dens_log = kde(grid)
    # p_z(z) = p_lnz(ln z) / z , but we plot density in ln z so each class is comparable
    return dens_log


def save_combined_figure(z: np.ndarray, y: np.ndarray, out_path: Path, title: str) -> None:
    logz = np.log(np.clip(z, 1e-4, None))
    grid = np.linspace(np.log(0.002), np.log(2.0), 512)
    z_grid = np.exp(grid)

    fig, (ax_den, ax_mix) = plt.subplots(
        1,
        2,
        figsize=(11.4, 4.6),
        gridspec_kw={"width_ratios": [1.15, 1.0]},
        layout="constrained",
    )

    for c, name in enumerate(CLASS_NAMES):
        m = y == c
        n = int(m.sum())
        med = float(np.median(z[m]))
        color = CLASS_COLORS[name]
        dens = _kde_line(logz[m], grid)
        ax_den.plot(
            z_grid,
            dens,
            color=color,
            lw=2.0,
            label=f"{name}  n={n:,}  med={med:.3f}",
        )
        ax_den.fill_between(z_grid, dens, color=color, alpha=0.12)

    ax_den.axvline(0.05, color="0.45", ls="--", lw=0.9)
    ax_den.axvline(0.20, color="0.45", ls=":", lw=0.9)
    ax_den.set_xscale("log")
    ax_den.set_xlim(0.002, 2.0)
    ax_den.set_ylim(bottom=0)
    ax_den.set_xlabel("$z$")
    ax_den.set_ylabel(r"Density of $\ln z$")
    ax_den.set_title(r"$p(\ln z\,|\,\mathrm{class})$")
    ax_den.legend(frameon=False, fontsize=8, loc="upper right")
    ax_den.grid(True, alpha=0.25, which="both")

    bin_idx = np.digitize(z, Z_BINS) - 1
    x = np.arange(len(Z_BIN_LABELS))
    bottom = np.zeros(len(Z_BIN_LABELS))
    counts = np.array([(bin_idx == b).sum() for b in range(len(Z_BIN_LABELS))], dtype=np.int64)
    for c, name in enumerate(CLASS_NAMES):
        fracs = np.array(
            [
                float(np.mean(y[bin_idx == b] == c)) if counts[b] else 0.0
                for b in range(len(Z_BIN_LABELS))
            ]
        )
        ax_mix.bar(
            x,
            fracs,
            bottom=bottom,
            color=CLASS_COLORS[name],
            width=0.78,
            label=name,
            linewidth=0,
        )
        bottom += fracs

    ax_mix.set_xticks(x)
    ax_mix.set_xticklabels(
        [f"{lab}\nn={n}" for lab, n in zip(Z_BIN_LABELS, counts)],
        rotation=30,
        ha="right",
        fontsize=8,
    )
    ax_mix.set_ylim(0.0, 1.0)
    ax_mix.set_ylabel("Fraction")
    ax_mix.set_xlabel("z bin")
    ax_mix.set_title(r"$p(\mathrm{class}\,|\,z\mathrm{-bin})$")

    fig.suptitle(title, fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def save_splits_figure(
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
    out_path: Path,
    title: str,
) -> None:
    grid = np.linspace(np.log(0.002), np.log(2.0), 512)
    z_grid = np.exp(grid)
    names = list(splits)
    fig, axes = plt.subplots(
        len(names),
        2,
        figsize=(11.4, 11.2),
        gridspec_kw={"width_ratios": [1.15, 1.0]},
        layout="constrained",
    )
    dens_max = 0.0
    handles = []
    for row, split_name in enumerate(names):
        z, y = splits[split_name]
        logz = np.log(np.clip(z, 1e-4, None))
        ax_den = axes[row, 0]
        ax_mix = axes[row, 1]
        last = row == len(names) - 1

        for c, name in enumerate(CLASS_NAMES):
            m = y == c
            n = int(m.sum())
            color = CLASS_COLORS[name]
            dens = _kde_line(logz[m], grid)
            dens_max = max(dens_max, float(np.max(dens)))
            line = ax_den.plot(z_grid, dens, color=color, lw=2.0, label=name)[0]
            ax_den.fill_between(z_grid, dens, color=color, alpha=0.12)
            if row == 0:
                handles.append(line)

        ax_den.axvline(0.05, color="0.45", ls="--", lw=0.9)
        ax_den.axvline(0.20, color="0.45", ls=":", lw=0.9)
        ax_den.set_xscale("log")
        ax_den.set_xlim(0.002, 2.0)
        ax_den.set_ylabel(f"{split_name} (n={len(z):,})\n" + r"Density of $\ln z$")
        ax_den.grid(True, alpha=0.25, which="both")
        if row == 0:
            ax_den.set_title(r"$p(\ln z\,|\,\mathrm{class})$")
        if last:
            ax_den.set_xlabel("$z$")

        bin_idx = np.digitize(z, Z_BINS) - 1
        x = np.arange(len(Z_BIN_LABELS))
        bottom = np.zeros(len(Z_BIN_LABELS))
        counts = np.array([(bin_idx == b).sum() for b in range(len(Z_BIN_LABELS))], dtype=np.int64)
        for c, name in enumerate(CLASS_NAMES):
            fracs = np.array(
                [
                    float(np.mean(y[bin_idx == b] == c)) if counts[b] else 0.0
                    for b in range(len(Z_BIN_LABELS))
                ]
            )
            ax_mix.bar(
                x,
                fracs,
                bottom=bottom,
                color=CLASS_COLORS[name],
                width=0.78,
                linewidth=0,
            )
            bottom += fracs
        ax_mix.set_xticks(x)
        ax_mix.set_xticklabels(
            [f"{lab}\nn={n}" for lab, n in zip(Z_BIN_LABELS, counts)],
            rotation=30,
            ha="right",
            fontsize=8,
        )
        ax_mix.set_ylim(0.0, 1.0)
        ax_mix.set_ylabel("Fraction")
        if row == 0:
            ax_mix.set_title(r"$p(\mathrm{class}\,|\,z\mathrm{-bin})$")
        if last:
            ax_mix.set_xlabel("z bin")

    for row in range(len(names)):
        axes[row, 0].set_ylim(0.0, dens_max * 1.05)

    fig.suptitle(title, fontsize=11)
    fig.legend(
        handles,
        CLASS_NAMES,
        loc="outside lower center",
        frameon=False,
        ncols=len(CLASS_NAMES),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-dir", type=Path, default=DEFAULT_LATENT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--splits-out", type=Path, default=DEFAULT_SPLITS_OUT)
    args = parser.parse_args()
    latent_dir = args.latent_dir.resolve()
    splits = load_class_z_splits(latent_dir)
    z_all = np.concatenate([z for z, _y in splits.values()])
    y_all = np.concatenate([y for _z, y in splits.values()])
    print("=== All ===")
    class_stats(z_all, y_all)
    save_combined_figure(z_all, y_all, args.out.resolve(), "Class vs redshift")
    for name, (z, y) in splits.items():
        print(f"=== {name} ===")
        class_stats(z, y)
    save_splits_figure(splits, args.splits_out.resolve(), "Class vs redshift")


if __name__ == "__main__":
    main()
