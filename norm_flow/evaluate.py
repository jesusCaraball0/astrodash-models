"""Point metrics, NLL, PIT, coverage, and diagnostic plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from models import RedshiftPredictor, encode_y, y_to_z

COVERAGE_LEVELS = (0.68, 0.90, 0.95)
QUANTILE_PROBS = (0.025, 0.05, 0.16, 0.5, 0.84, 0.95, 0.975)
Z_GRID_N = 256


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    resid = y_pred - y_true
    mse = float(np.mean(resid**2)) if y_true.size else float("nan")
    mae = float(np.mean(np.abs(resid))) if y_true.size else float("nan")
    rmse = float(np.sqrt(mse)) if np.isfinite(mse) else float("nan")
    if y_true.size:
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        r2 = float("nan") if ss_tot <= 0.0 else 1.0 - float(np.sum(resid**2)) / ss_tot
    else:
        r2 = float("nan")
    return {"n": int(y_true.size), "mse": mse, "mae": mae, "rmse": rmse, "r2": r2}


def jsonify(metrics: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, val in metrics.items():
        if isinstance(val, dict):
            out[key] = jsonify(val)
        elif isinstance(val, (np.floating, np.integer)):
            out[key] = val.item()
        elif isinstance(val, float) and not np.isfinite(val):
            out[key] = None
        elif isinstance(val, np.ndarray):
            out[key] = val.tolist()
        else:
            out[key] = val
    return out


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


@torch.no_grad()
def collect_predictions(
    model: RedshiftPredictor,
    loader,
    device: torch.device,
    *,
    z_floor: float,
    z_max: float,
    target: str | None = None,
) -> Dict[str, np.ndarray]:
    target = str(target if target is not None else getattr(model, "target", "logz")).lower()
    model.eval()
    ids: List[str] = []
    z_true: List[np.ndarray] = []
    y_true: List[np.ndarray] = []
    y_med: List[np.ndarray] = []
    z_med: List[np.ndarray] = []
    nll_y: List[np.ndarray] = []
    nll_z: List[np.ndarray] = []
    pit: List[np.ndarray] = []
    crps: List[np.ndarray] = []
    q_all: List[np.ndarray] = []
    z_q_all: List[np.ndarray] = []
    gate_pi: List[np.ndarray] = []
    probs = torch.tensor(QUANTILE_PROBS, device=device, dtype=torch.float32)

    for batch in loader:
        if batch is None:
            continue
        x = batch["x"].to(device)
        z = batch["y"].to(device).reshape(-1)
        y = encode_y(z, target, z_floor)
        y_hat = model.point_y(x)
        q_y = model.quantiles_y(x, probs)
        ids.extend(str(s) for s in batch["id"])
        z_true.append(_to_numpy(z))
        y_true.append(_to_numpy(y))
        y_med.append(_to_numpy(y_hat))
        z_med.append(_to_numpy(y_to_z(y_hat, target, z_floor, z_max)))
        q_all.append(_to_numpy(q_y))
        z_q_all.append(_to_numpy(y_to_z(q_y, target, z_floor, z_max)))
        if model.probabilistic:
            nll_y.append(_to_numpy(-model.log_prob_y(x, y)))
            nll_z.append(_to_numpy(-model.log_prob_z(x, z, z_floor=z_floor)))
            pit.append(_to_numpy(model.pit(x, y)))
            crps.append(_to_numpy(model.crps_y(x, y)))
        if hasattr(model.head, "gate_pi_low"):
            gate_pi.append(_to_numpy(model.head.gate_pi_low(model.context(x))))

    def _cat(xs: List[np.ndarray], cols: int | None = None) -> np.ndarray:
        if not xs:
            if cols:
                return np.zeros((0, cols), dtype=np.float64)
            return np.zeros((0,), dtype=np.float64)
        return np.concatenate(xs, axis=0)

    z_true_np = _cat(z_true)
    z_med_np = _cat(z_med)
    q = _cat(q_all, cols=len(QUANTILE_PROBS))
    z_q = _cat(z_q_all, cols=len(QUANTILE_PROBS))
    logz_true = np.log(np.clip(z_true_np, float(z_floor), None))
    logz_med = np.log(np.clip(z_med_np, float(z_floor), None))
    out: Dict[str, np.ndarray] = {
        "id": np.asarray(ids, dtype=object),
        "z_true": z_true_np,
        "y_true": _cat(y_true),
        "y_median": _cat(y_med),
        "z_median": z_med_np,
        "logz_true": logz_true,
        "logz_median": logz_med,
        "y_quantiles": q,
        "z_quantiles": z_q,
        "quantile_probs": np.asarray(QUANTILE_PROBS, dtype=np.float64),
        "target": np.asarray(target),
    }
    if model.probabilistic and nll_y:
        out["nll_y"] = _cat(nll_y)
        out["nll_z"] = _cat(nll_z)
        out["pit"] = _cat(pit)
        out["crps_y"] = _cat(crps)
    if gate_pi:
        out["gate_pi_low"] = _cat(gate_pi)
    return out


def math_log(x: float) -> float:
    return float(np.log(x))


def coverage_table(pred: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    y_true = pred["y_true"]
    z_true = pred["z_true"]
    probs = pred["quantile_probs"]
    y_q = pred["y_quantiles"]
    z_q = pred["z_quantiles"]
    rows: Dict[str, Dict[str, float]] = {}
    for level in COVERAGE_LEVELS:
        lo_p = (1.0 - level) / 2.0
        hi_p = 1.0 - lo_p
        lo_i = int(np.argmin(np.abs(probs - lo_p)))
        hi_i = int(np.argmin(np.abs(probs - hi_p)))
        inside_y = (y_true >= y_q[:, lo_i]) & (y_true <= y_q[:, hi_i])
        inside_z = (z_true >= z_q[:, lo_i]) & (z_true <= z_q[:, hi_i])
        width_z = z_q[:, hi_i] - z_q[:, lo_i]
        rows[f"{int(round(level * 100))}%"] = {
            "nominal": float(level),
            "empirical": float(np.mean(inside_y)) if y_true.size else float("nan"),
            "empirical_z": float(np.mean(inside_z)) if z_true.size else float("nan"),
            "mean_width_z": float(np.mean(width_z)) if z_true.size else float("nan"),
            "median_width_z": float(np.median(width_z)) if z_true.size else float("nan"),
        }
    return rows


def summarize_predictions(
    pred: Dict[str, np.ndarray],
    *,
    loss_value: float,
    probabilistic: bool,
) -> Dict[str, Any]:
    z_m = regression_metrics(pred["z_true"], pred["z_median"])
    logz_true = pred.get("logz_true")
    logz_med = pred.get("logz_median")
    if logz_true is None:
        logz_true = pred["y_true"]
        logz_med = pred["y_median"]
    y_m = regression_metrics(logz_true, logz_med)
    y_train = regression_metrics(pred["y_true"], pred["y_median"])
    out: Dict[str, Any] = {
        "loss": float(loss_value),
        "linear_z": z_m,
        "logz": y_m,
        "train_y": y_train,
        "point_estimate": "posterior_median" if probabilistic else "mse_scalar",
    }
    if probabilistic:
        out["nll_y"] = float(np.mean(pred["nll_y"])) if pred["nll_y"].size else float("nan")
        out["nll_z"] = float(np.mean(pred["nll_z"])) if pred["nll_z"].size else float("nan")
        out["crps_y"] = float(np.mean(pred["crps_y"])) if pred["crps_y"].size else float("nan")
        out["coverage"] = coverage_table(pred)
        pit = pred["pit"]
        out["pit"] = {
            "mean": float(np.mean(pit)) if pit.size else float("nan"),
            "std": float(np.std(pit)) if pit.size else float("nan"),
        }
        if "gate_pi_low" in pred:
            g = pred["gate_pi_low"]
            out["gate_pi_low"] = {
                "mean": float(np.mean(g)) if g.size else float("nan"),
                "std": float(np.std(g)) if g.size else float("nan"),
            }
    return out


def write_predictions_csv(pred: Dict[str, np.ndarray], path: Path) -> None:
    probs = pred["quantile_probs"]
    data: Dict[str, Any] = {
        "id": pred["id"],
        "z_true": pred["z_true"],
        "z_median": pred["z_median"],
        "y_true": pred["y_true"],
        "y_median": pred["y_median"],
    }
    for i, q in enumerate(probs):
        tag = f"{q:.3f}".rstrip("0").rstrip(".")
        data[f"z_q{tag}"] = pred["z_quantiles"][:, i]
        data[f"y_q{tag}"] = pred["y_quantiles"][:, i]
    for key in ("nll_y", "nll_z", "pit", "crps_y", "gate_pi_low"):
        if key in pred:
            data[key] = pred[key]
    pd.DataFrame(data).to_csv(path, index=False)


def _scatter(ax, y_true: np.ndarray, y_pred: np.ndarray, title: str, xlabel: str, ylabel: str) -> None:
    metrics = regression_metrics(y_true, y_pred)
    ax.scatter(y_true, y_pred, s=8, alpha=0.35, linewidths=0, c="#0072B2")
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    pad = 0.02 * (hi - lo + 1e-6)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"{title}\nMAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  "
        f"R²={metrics['r2']:.3f}  n={metrics['n']}"
    )
    ax.grid(True, alpha=0.3)


def save_true_vs_pred(pred: Dict[str, np.ndarray], out_dir: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    _scatter(ax, pred["z_true"], pred["z_median"], title, "True redshift", "Predicted redshift")
    fig.tight_layout()
    fig.savefig(out_dir / "z_true_vs_pred.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    _scatter(
        ax,
        pred.get("logz_true", pred["y_true"]),
        pred.get("logz_median", pred["y_median"]),
        f"{title} (ln z)",
        "True ln z",
        "Predicted ln z",
    )
    fig.tight_layout()
    fig.savefig(out_dir / "logz_true_vs_pred.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_pit_histogram(pred: Dict[str, np.ndarray], out_dir: Path, title: str) -> None:
    if "pit" not in pred:
        return
    pit = pred["pit"]
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.hist(pit, bins=20, range=(0.0, 1.0), density=True, color="#0072B2", alpha=0.85, edgecolor="white")
    ax.axhline(1.0, color="k", ls="--", lw=1, label="Uniform[0,1]")
    ax.set_xlabel(r"PIT $F(y_{\mathrm{true}}\mid x)$")
    ax.set_ylabel("Density")
    ax.set_title(f"{title}\nPIT mean={pit.mean():.3f}  std={pit.std():.3f}")
    ax.legend(frameon=False)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(out_dir / "pit_histogram.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_coverage_plot(pred: Dict[str, np.ndarray], out_dir: Path, title: str) -> None:
    if "y_quantiles" not in pred or pred["y_true"].size == 0:
        return
    table = coverage_table(pred)
    nom = [table[k]["nominal"] for k in table]
    emp = [table[k]["empirical"] for k in table]
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.plot(nom, emp, "o-", color="#0072B2")
    ax.set_xlim(0.5, 1.02)
    ax.set_ylim(0.5, 1.02)
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_dir / "coverage.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _count_modes(logp: np.ndarray) -> int:
    p = np.exp(logp - np.max(logp))
    peaks = 0
    for i in range(1, len(p) - 1):
        if p[i] >= p[i - 1] and p[i] > p[i + 1] and p[i] > 0.05:
            peaks += 1
    return peaks


def _second_peak_ratio(logp: np.ndarray) -> float:
    p = np.exp(logp - np.max(logp))
    heights = []
    for i in range(1, len(p) - 1):
        if p[i] >= p[i - 1] and p[i] > p[i + 1]:
            heights.append(float(p[i]))
    heights.sort(reverse=True)
    if len(heights) < 2:
        return 0.0
    return heights[1] / max(heights[0], 1e-12)


@torch.no_grad()
def posterior_grid(
    model: RedshiftPredictor,
    x: torch.Tensor,
    z_grid: torch.Tensor,
    *,
    z_floor: float,
) -> np.ndarray:
    """Return log p(z|x) on a shared physical-z grid; shape (B, G)."""
    b = int(x.shape[0])
    g = int(z_grid.numel())
    x_rep = x.unsqueeze(1).expand(b, g, *x.shape[1:]).reshape(b * g, *x.shape[1:])
    z_rep = z_grid.reshape(1, g).expand(b, g).reshape(-1)
    lp = model.log_prob_z(x_rep, z_rep, z_floor=z_floor)
    return _to_numpy(lp.reshape(b, g))


def _pick_example_indices(pred: Dict[str, np.ndarray], logp: np.ndarray | None) -> Dict[str, int]:
    z_true = pred["z_true"]
    z_med = pred["z_median"]
    err = np.abs(z_med - z_true)
    n = int(z_true.size)
    if n == 0:
        return {}
    chosen: Dict[str, int] = {}
    if "z_quantiles" in pred and pred["z_quantiles"].size:
        probs = pred["quantile_probs"]
        lo_i = int(np.argmin(np.abs(probs - 0.16)))
        hi_i = int(np.argmin(np.abs(probs - 0.84)))
        width = pred["z_quantiles"][:, hi_i] - pred["z_quantiles"][:, lo_i]
        chosen["broad"] = int(np.argmax(width))
        narrow = width <= np.quantile(width, 0.3)
        if np.any(narrow):
            idx = np.where(narrow)[0]
            chosen["narrow_accurate"] = int(idx[np.argmin(err[idx])])
        else:
            chosen["narrow_accurate"] = int(np.argmin(err))
    else:
        chosen["narrow_accurate"] = int(np.argmin(err))
        chosen["broad"] = int(np.argmax(err))
    chosen["bad"] = int(np.argmax(err))
    if logp is not None:
        ratios = np.array([_second_peak_ratio(logp[i]) for i in range(n)])
        modes = np.array([_count_modes(logp[i]) for i in range(n)])
        if np.any(modes >= 2):
            chosen["multimodal"] = int(np.argmax(ratios))
    # unique-ify while keeping labels
    used = set()
    unique: Dict[str, int] = {}
    for key, idx in chosen.items():
        if idx in used:
            continue
        unique[key] = idx
        used.add(idx)
    return unique


def _gather_x_by_index(loader, indices: List[int], n_total: int) -> torch.Tensor:
    wanted = set(int(i) for i in indices)
    found: Dict[int, torch.Tensor] = {}
    seen = 0
    for batch in loader:
        if batch is None:
            continue
        bsz = int(batch["x"].shape[0])
        for j in range(bsz):
            gi = seen + j
            if gi in wanted:
                found[gi] = batch["x"][j].detach().cpu()
        seen += bsz
        if len(found) >= len(wanted) or seen >= n_total:
            break
    missing = [i for i in indices if i not in found]
    if missing:
        raise RuntimeError(f"Could not reload features for example indices {missing}")
    return torch.stack([found[i] for i in indices], dim=0)


@torch.no_grad()
def save_posterior_examples(
    model: RedshiftPredictor,
    loader,
    pred: Dict[str, np.ndarray],
    device: torch.device,
    out_dir: Path,
    *,
    z_floor: float,
    z_max: float,
    multimodal_scan: int = 256,
) -> None:
    if not model.probabilistic:
        return
    picks = _pick_example_indices(pred, None)
    n = int(pred["z_true"].size)
    scan_n = min(int(multimodal_scan), n)
    z_hi = float(min(z_max, max(float(pred["z_true"].max()) * 1.5, 1.0)))
    # Apple MPS does not implement aten::logspace. Build the 1D z grid with
    # NumPy on CPU, then move it to the model device for density evaluation.
    z_grid = torch.from_numpy(
        np.logspace(np.log10(z_floor), np.log10(z_hi), Z_GRID_N).astype(np.float32)
    ).to(device)
    if scan_n > 0:
        scan_idx = list(range(scan_n))
        x_scan = _gather_x_by_index(loader, scan_idx, n).to(device)
        logp_scan = posterior_grid(model, x_scan, z_grid, z_floor=z_floor)
        ratios = np.array([_second_peak_ratio(logp_scan[i]) for i in range(scan_n)])
        modes = np.array([_count_modes(logp_scan[i]) for i in range(scan_n)])
        if np.any(modes >= 2):
            best = int(np.argmax(ratios))
            if best not in picks.values():
                picks["multimodal"] = best

    order = list(picks.items())
    indices = [idx for _, idx in order]
    x_ex = _gather_x_by_index(loader, indices, n).to(device)
    logp = posterior_grid(model, x_ex, z_grid, z_floor=z_floor)
    plot_dir = out_dir / "posteriors"
    plot_dir.mkdir(parents=True, exist_ok=True)
    z_np = _to_numpy(z_grid)
    probs = pred["quantile_probs"]
    lo_i = int(np.argmin(np.abs(probs - 0.16)))
    hi_i = int(np.argmin(np.abs(probs - 0.84)))
    for plot_i, (tag, idx) in enumerate(order):
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        p = np.exp(logp[plot_i] - np.max(logp[plot_i]))
        area = float(np.trapezoid(p, z_np))
        p = p / max(area, 1e-12)
        ax.plot(z_np, p, color="#0072B2", lw=2)
        z_t = float(pred["z_true"][idx])
        z_m = float(pred["z_median"][idx])
        ax.axvline(z_t, color="k", ls="--", lw=1.2, label=f"true z={z_t:.4f}")
        ax.axvline(z_m, color="#D55E00", ls="-", lw=1.2, label=f"median={z_m:.4f}")
        lo = float(pred["z_quantiles"][idx, lo_i])
        hi = float(pred["z_quantiles"][idx, hi_i])
        ax.axvspan(lo, hi, color="#0072B2", alpha=0.15, label="68% interval")
        ax.set_xlabel("Redshift")
        ax.set_ylabel(r"$p(z\mid x)$")
        ax.set_title(f"{tag}  id={pred['id'][idx]}")
        ax.legend(frameon=False, fontsize=8)
        ax.set_xlim(max(z_floor, min(z_t, z_m, lo) * 0.4), max(z_t, z_m, hi) * 1.8)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{tag}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
    np.savez_compressed(
        plot_dir / "example_grids.npz",
        z_grid=z_np,
        logp=logp,
        ids=np.asarray([pred["id"][i] for i in indices], dtype=object),
        tags=np.asarray([t for t, _ in order], dtype=object),
        indices=np.asarray(indices, dtype=np.int64),
    )


def write_all_plots(
    model: RedshiftPredictor,
    loader,
    pred: Dict[str, np.ndarray],
    device: torch.device,
    out_dir: Path,
    *,
    title: str,
    z_floor: float,
    z_max: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    save_true_vs_pred(pred, out_dir, title)
    if model.probabilistic:
        save_pit_histogram(pred, out_dir, title)
        save_coverage_plot(pred, out_dir, f"{title} coverage")
        save_posterior_examples(
            model, loader, pred, device, out_dir, z_floor=z_floor, z_max=z_max
        )
