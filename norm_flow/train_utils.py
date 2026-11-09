"""Shared training loop for latent and DASH redshift density models."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from evaluate import (
    collect_predictions,
    jsonify,
    summarize_predictions,
    write_all_plots,
    write_predictions_csv,
)
from heads import FLOW_LIKE_HEADS
from models import TARGETS, RedshiftPredictor, encode_y

Z_FLOOR = 1e-4
Z_MAX = 6.0
NSF_BOUND = 5.0
BETTER_HPS_Z_COVER = 2.0
BETTER_HPS_BINS = 64
DEFAULT_FLOW_BINS = 8
DEFAULT_FLOW_TRANSFORMS = 2
DEFAULT_FLOW_HIDDEN_LAYERS = 2
DEFAULT_TRUNK_LAYERS = 1
DEEPER_BINS = 16
DEEPER_FLOW_HIDDEN_LAYERS = 3
DEEPER_TRUNK_LAYERS = 2
REWEIGHT_ALPHA = 0.5
# Same physical-z bins as the MAE-by-z tables / class-vs-redshift figure.
REWEIGHT_Z_BINS = (0.0, 0.05, 0.10, 0.20, 0.40, 0.80, 6.0)
REWEIGHT_Z_BIN_LABELS = (
    "[0, 0.05)",
    "[0.05, 0.10)",
    "[0.10, 0.20)",
    "[0.20, 0.40)",
    "[0.40, 0.80)",
    "[0.80, 6)",
)


def to_logz(z: torch.Tensor, z_floor: float = Z_FLOOR) -> torch.Tensor:
    return encode_y(z, "logz", z_floor)


def to_y(z: torch.Tensor, target: str, z_floor: float = Z_FLOOR) -> torch.Tensor:
    return encode_y(z, target, z_floor)


def _target_of(model: RedshiftPredictor, target: str | None) -> str:
    t = str(target if target is not None else getattr(model, "target", "logz")).lower()
    if t not in TARGETS:
        raise ValueError(f"Unknown target {t!r}; expected one of {TARGETS}")
    return t


def jacobian_note(target: str) -> str:
    if target == "logz":
        return "log p_z(z|x) = log p_y(ln z|x) - ln z"
    return "log p_z(z|x) = log p_y(z|x) (identity; model trained in physical z)"


def covering_std(
    mean: float,
    empirical_std: float,
    *,
    z_lo: float,
    z_hi: float,
    bound: float = NSF_BOUND,
) -> float:
    """Smallest std so [z_lo, z_hi] lands in the NSF interval [-bound, bound].

    u = (z - mean) / std. Keep the train mean (bulk near u=0) and widen std
    until z_hi maps to +bound. Does not change the training label; y is still z.
    """
    bound = float(bound)
    if bound <= 0.0:
        raise ValueError(f"NSF bound must be positive, got {bound}")
    need_hi = (float(z_hi) - float(mean)) / bound
    need_lo = (float(mean) - float(z_lo)) / bound
    return max(float(empirical_std), need_hi, need_lo, 1e-3)


def resolve_better_hps(
    *,
    better_hps: bool,
    target: str,
    head: str,
    bins: int | None,
    transforms: int | None,
) -> tuple[int, int, float | None]:
    """Return (bins, transforms, cover_z). cover_z is None unless --better-hps."""
    head = str(head).lower()
    target = str(target).lower()
    if not better_hps:
        return (
            DEFAULT_FLOW_BINS if bins is None else int(bins),
            DEFAULT_FLOW_TRANSFORMS if transforms is None else int(transforms),
            None,
        )
    if target != "z":
        raise ValueError("--better-hps requires --target z (linear-z NSF covering)")
    if head not in FLOW_LIKE_HEADS:
        raise ValueError("--better-hps only applies to flow / moe (NSF domain + bins)")
    return (
        BETTER_HPS_BINS if bins is None else int(bins),
        DEFAULT_FLOW_TRANSFORMS if transforms is None else int(transforms),
        BETTER_HPS_Z_COVER,
    )


def resolve_deeper(
    *,
    deeper: bool,
    head: str,
    bins: int | None,
) -> tuple[int | None, int, int]:
    """Return (bins_override, flow_hidden_layers, trunk_layers).

    ``bins_override`` is None when ``--deeper`` is off so ``resolve_better_hps``
    still owns the default-8 / better-hps-64 choice.
    """
    head = str(head).lower()
    if not deeper:
        return (
            None if bins is None else int(bins),
            DEFAULT_FLOW_HIDDEN_LAYERS,
            DEFAULT_TRUNK_LAYERS,
        )
    if head not in FLOW_LIKE_HEADS:
        raise ValueError("--deeper only applies to flow / moe (extra trunk layer + 16 bins)")
    return (
        DEEPER_BINS if bins is None else int(bins),
        DEEPER_FLOW_HIDDEN_LAYERS,
        DEEPER_TRUNK_LAYERS,
    )


def output_tag(
    target: str,
    *,
    cls_cond: bool = False,
    cls_true: bool = False,
    better_hps: bool = False,
    deeper: bool = False,
    reweight: bool = False,
    aug: bool = False,
) -> str:
    """Infix used in daep_latent_{tag}_{head} / dash_{tag}_{head} folder names."""
    if cls_cond and cls_true:
        raise ValueError("cls_cond and cls_true are mutually exclusive")
    if better_hps and deeper:
        raise ValueError("better_hps and deeper are mutually exclusive")
    if aug and (cls_cond or cls_true):
        raise ValueError("aug cannot be combined with cls_cond / cls_true")
    tag = "logz" if target == "logz" else "z"
    if cls_true:
        tag = f"{tag}_truecls"
    elif cls_cond:
        tag = f"{tag}_cls"
    if better_hps:
        tag = f"{tag}_betterhps"
    if deeper:
        tag = f"{tag}_deeper"
    if reweight:
        tag = f"{tag}_reweight"
    if aug:
        tag = f"{tag}_aug"
    return tag


def assert_finite(t: torch.Tensor, name: str) -> None:
    if not torch.isfinite(t).all():
        raise RuntimeError(f"Non-finite {name}: min={t.min().item() if t.numel() else 'empty'} max={t.max().item() if t.numel() else 'empty'}")


class InverseFreqWeights:
    """Per-sample weights w_b ∝ N_b^{-α} on physical-z bins (train counts only).

    Mean-normalized so the training set has mean weight 1. The training loss is
    L = sum(w ell) / sum(w) with ell = -log p(y|x) (or squared error for MSE).
    For logz models, weighting -log p_y vs -log p_z differs by a z-only constant,
    so gradients match the image formula on p(z|x).
    """

    def __init__(
        self,
        edges: Sequence[float],
        counts: Sequence[int],
        alpha: float,
        w_bin: Sequence[float],
    ):
        self.edges = [float(e) for e in edges]
        self.counts = [int(c) for c in counts]
        self.alpha = float(alpha)
        self.w_bin = [float(w) for w in w_bin]

    @classmethod
    def from_z(
        cls,
        z: np.ndarray | torch.Tensor,
        *,
        edges: Sequence[float] = REWEIGHT_Z_BINS,
        alpha: float = REWEIGHT_ALPHA,
    ) -> "InverseFreqWeights":
        if torch.is_tensor(z):
            z_np = z.detach().cpu().numpy().reshape(-1).astype(np.float64)
        else:
            z_np = np.asarray(z, dtype=np.float64).reshape(-1)
        if z_np.size == 0:
            raise RuntimeError("Cannot build reweight table from empty train redshifts")
        edges_np = np.asarray(list(edges), dtype=np.float64)
        if edges_np.size < 2:
            raise ValueError("Need at least two bin edges")
        idx = np.digitize(z_np, edges_np) - 1
        idx = np.clip(idx, 0, int(edges_np.size) - 2)
        n_bins = int(edges_np.size) - 1
        counts = np.bincount(idx, minlength=n_bins).astype(np.float64)
        w_bin = np.maximum(counts, 1.0) ** (-float(alpha))
        w_i = w_bin[idx]
        w_bin = w_bin * (float(z_np.size) / float(w_i.sum()))
        return cls(edges_np.tolist(), counts.astype(int).tolist(), float(alpha), w_bin.tolist())

    def lookup(self, z: torch.Tensor) -> torch.Tensor:
        z1 = z.reshape(-1)
        edges = torch.as_tensor(self.edges, device=z1.device, dtype=z1.dtype)
        w_bin = torch.as_tensor(self.w_bin, device=z1.device, dtype=z1.dtype)
        idx = torch.bucketize(z1, edges) - 1
        return w_bin[idx.clamp(0, int(w_bin.numel()) - 1)]

    def as_cfg(self) -> Dict[str, Any]:
        labels = list(REWEIGHT_Z_BIN_LABELS)
        if len(labels) != len(self.counts):
            labels = [
                f"[{self.edges[i]:g}, {self.edges[i + 1]:g})"
                for i in range(len(self.counts))
            ]
        return {
            "reweight": True,
            "reweight_alpha": self.alpha,
            "reweight_bin_edges": list(self.edges),
            "reweight_bin_counts": list(self.counts),
            "reweight_bin_weights": list(self.w_bin),
            "reweight_bin_labels": labels,
            "early_stop_loss": "weighted_nll",
        }

    def describe(self) -> str:
        labels = self.as_cfg()["reweight_bin_labels"]
        rows = [
            f"  {lab:16s}  n={n:5d}  w={w:.4f}"
            for lab, n, w in zip(labels, self.counts, self.w_bin)
        ]
        return (
            f"[reweight] alpha={self.alpha:g}  w_b ∝ N_b^{{-alpha}}  "
            f"(train mean w=1; eval NLL stays unweighted)\n" + "\n".join(rows)
        )


def collect_physical_z(loader: DataLoader) -> torch.Tensor:
    zs: List[torch.Tensor] = []
    for batch in loader:
        if batch is None:
            continue
        zs.append(batch["y"].reshape(-1).detach().cpu())
    if not zs:
        raise RuntimeError("Empty loader; cannot collect redshifts for reweighting")
    return torch.cat(zs)


@torch.no_grad()
def mean_loss(
    model: RedshiftPredictor,
    loader: DataLoader,
    device: torch.device,
    *,
    z_floor: float,
    target: str | None = None,
    weights: InverseFreqWeights | None = None,
) -> float:
    target = _target_of(model, target)
    model.eval()
    total = 0.0
    denom = 0.0
    for batch in loader:
        if batch is None:
            continue
        x = batch["x"].to(device)
        z_phys = batch["y"].to(device)
        y = to_y(z_phys, target, z_floor)
        nll = model.per_sample_loss(x, y)
        if weights is None:
            total += float(nll.sum().item())
            denom += float(nll.numel())
        else:
            w = weights.lookup(z_phys)
            total += float((w * nll).sum().item())
            denom += float(w.sum().item())
    return total / max(denom, 1e-8)


def fit_flow_y_stats(
    model: RedshiftPredictor,
    loader: DataLoader,
    device: torch.device,
    *,
    z_floor: float,
    target: str | None = None,
    cover_z: float | None = None,
) -> tuple[float, float]:
    target = _target_of(model, target)
    ys: List[torch.Tensor] = []
    for batch in loader:
        if batch is None:
            continue
        ys.append(to_y(batch["y"].to(device), target, z_floor))
    if not ys:
        raise RuntimeError(f"Empty train loader; cannot standardize y={target} for the flow")
    y = torch.cat(ys)
    mean = float(y.mean().item())
    std = float(y.std(unbiased=False).clamp(min=1e-3).item())
    y_name = "ln z" if target == "logz" else "z"
    if cover_z is not None and target == "z":
        empir = std
        std = covering_std(
            mean,
            empir,
            z_lo=float(z_floor),
            z_hi=float(cover_z),
            bound=NSF_BOUND,
        )
        u_lo = (float(z_floor) - mean) / std
        u_hi = (float(cover_z) - mean) / std
        print(
            f"[{model.head_kind}] covering z in [{z_floor:g}, {float(cover_z):g}] "
            f"inside NSF u in [{-NSF_BOUND:g}, {NSF_BOUND:g}]: "
            f"train mean={mean:.4f} empir_std={empir:.4f} std={std:.4f} "
            f"u(z_floor)={u_lo:.2f} u(z={float(cover_z):g})={u_hi:.2f}",
            flush=True,
        )
    else:
        print(
            f"[{model.head_kind}] standardizing y={y_name} with train mean={mean:.4f} std={std:.4f} "
            "(zuko NSF domain is [-5, 5] after this affine map)",
            flush=True,
        )
    model.set_flow_y_stats(mean, std)
    return mean, std


def train_and_eval(
    model: RedshiftPredictor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    *,
    device: torch.device,
    out_dir: Path,
    cfg: Dict[str, Any],
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau] = None,
    epochs: int,
    patience: int,
    clip_norm: float | None,
    z_floor: float = Z_FLOOR,
    z_max: float = Z_MAX,
    title: str,
    eval_only: bool = False,
    target: str | None = None,
    cover_z: float | None = None,
    reweight: bool = False,
    reweight_alpha: float = REWEIGHT_ALPHA,
) -> Path:
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "model_best.pt"
    last_path = out_dir / "model.pt"
    plots_dir = out_dir / "plots"
    z_floor = float(z_floor)
    z_max = float(z_max)
    target = _target_of(model, target if target is not None else cfg.get("target"))
    cfg["target"] = target
    cfg.setdefault("jacobian", jacobian_note(target))

    if eval_only:
        if not best_path.is_file():
            raise FileNotFoundError(f"--eval-only requires {best_path}")
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        best_epoch = int(ckpt.get("best_epoch", -1))
        history = []
        perf_path = out_dir / "model_performance.json"
        if perf_path.is_file():
            try:
                history = list(json.loads(perf_path.read_text(encoding="utf-8")).get("loss_by_epoch") or [])
            except json.JSONDecodeError:
                history = []
        print(f"[eval] loaded {best_path.name} from epoch {best_epoch} (skipped training)", flush=True)
        _write_eval(
            model,
            val_loader,
            test_loader,
            device=device,
            out_dir=out_dir,
            cfg=cfg,
            plots_dir=plots_dir,
            z_floor=z_floor,
            z_max=z_max,
            title=title,
            best_epoch=best_epoch,
            history=history,
            target=target,
        )
        return best_path

    if hasattr(model.head, "set_y_stats"):
        if cover_z is None:
            cover_z = cfg.get("cover_z")
        y_mean, y_std = fit_flow_y_stats(
            model,
            train_loader,
            device,
            z_floor=z_floor,
            target=target,
            cover_z=cover_z,
        )
        cfg["flow_y_mean"] = y_mean
        cfg["flow_y_std"] = y_std
        cfg["cover_z"] = cover_z

    weights: InverseFreqWeights | None = None
    if reweight or bool(cfg.get("reweight")):
        alpha = float(cfg.get("reweight_alpha", reweight_alpha) or REWEIGHT_ALPHA)
        weights = InverseFreqWeights.from_z(
            collect_physical_z(train_loader),
            edges=list(cfg.get("reweight_bin_edges") or REWEIGHT_Z_BINS),
            alpha=alpha,
        )
        cfg.update(weights.as_cfg())
        print(weights.describe(), flush=True)
    else:
        cfg.setdefault("reweight", False)

    best_val = float("inf")
    best_epoch = -1
    no_improve = 0
    history: List[List[Any]] = []
    nan_streak = 0

    print(
        f"[train] head={model.head_kind}  target={target}  epochs={epochs}  "
        f"patience={patience}  clip_norm={clip_norm}  reweight={weights is not None}  "
        f"out_dir={out_dir}",
        flush=True,
    )

    for epoch in range(1, int(epochs) + 1):
        t0 = time.perf_counter()
        model.train()
        running_num = 0.0
        running_den = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False, dynamic_ncols=True)
        for batch in pbar:
            if batch is None:
                continue
            x = batch["x"].to(device)
            z_phys = batch["y"].to(device)
            y = to_y(z_phys, target, z_floor)
            w = None if weights is None else weights.lookup(z_phys)
            optimizer.zero_grad(set_to_none=True)
            nll = model.per_sample_loss(x, y)
            if w is None:
                loss = nll.mean()
            else:
                loss = (w * nll).sum() / w.sum().clamp(min=1e-8)
            if not torch.isfinite(loss):
                nan_streak += 1
                print(f"[warn] non-finite train loss at epoch {epoch} ({nan_streak} in a row)", flush=True)
                if nan_streak >= 10:
                    raise RuntimeError("Too many non-finite train losses")
                continue
            nan_streak = 0
            loss.backward()
            if clip_norm is not None and clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_norm))
            optimizer.step()
            if w is None:
                running_num += float(nll.detach().sum().item())
                running_den += float(nll.numel())
            else:
                running_num += float((w.detach() * nll.detach()).sum().item())
                running_den += float(w.detach().sum().item())
            pbar.set_postfix(loss=f"{running_num / max(running_den, 1e-8):.5f}")

        train_loss = running_num / max(running_den, 1e-8)
        val_loss = mean_loss(
            model, val_loader, device, z_floor=z_floor, target=target, weights=weights
        )
        if scheduler is not None:
            scheduler.step(val_loss)
        history.append([epoch, train_loss, val_loss])
        dt = time.perf_counter() - t0
        extra = ""
        if weights is not None:
            val_unw = mean_loss(model, val_loader, device, z_floor=z_floor, target=target)
            extra = f"  val_unweighted {val_unw:.6f}"
        print(
            f"epoch {epoch}/{epochs} {dt:.1f}s  train {train_loss:.6f}  val {val_loss:.6f}{extra}",
            flush=True,
        )
        if val_loss < best_val and np.isfinite(val_loss):
            best_val = float(val_loss)
            best_epoch = epoch
            no_improve = 0
            ckpt = {
                "model_state_dict": model.state_dict(),
                "cfg": cfg,
                "head_kind": model.head_kind,
                "best_epoch": best_epoch,
                "val_loss": best_val,
                **model.extra_state(),
            }
            torch.save(ckpt, best_path)
            print(f"[best] epoch={best_epoch} val_loss={best_val:.6f} -> {best_path.name}", flush=True)
        else:
            no_improve += 1
            if no_improve >= int(patience):
                print(f"[early stop] no val improvement for {patience} epochs (best {best_epoch})", flush=True)
                break

    torch.save(
        {"model_state_dict": model.state_dict(), "cfg": cfg, "head_kind": model.head_kind, **model.extra_state()},
        last_path,
    )
    if best_path.is_file():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[eval] loaded best checkpoint from epoch {ckpt.get('best_epoch')}", flush=True)
    else:
        print("[eval] no best checkpoint; using last epoch", flush=True)

    _write_eval(
        model,
        val_loader,
        test_loader,
        device=device,
        out_dir=out_dir,
        cfg=cfg,
        plots_dir=plots_dir,
        z_floor=z_floor,
        z_max=z_max,
        title=title,
        best_epoch=best_epoch,
        history=history,
        target=target,
    )
    return best_path


def _write_eval(
    model: RedshiftPredictor,
    val_loader: DataLoader,
    test_loader: DataLoader,
    *,
    device: torch.device,
    out_dir: Path,
    cfg: Dict[str, Any],
    plots_dir: Path,
    z_floor: float,
    z_max: float,
    title: str,
    best_epoch: int,
    history: List[List[Any]],
    target: str,
) -> None:
    val_pred = collect_predictions(
        model, val_loader, device, z_floor=z_floor, z_max=z_max, target=target
    )
    test_pred = collect_predictions(
        model, test_loader, device, z_floor=z_floor, z_max=z_max, target=target
    )
    val_loss = mean_loss(model, val_loader, device, z_floor=z_floor, target=target)
    test_loss = mean_loss(model, test_loader, device, z_floor=z_floor, target=target)
    val_m = summarize_predictions(val_pred, loss_value=val_loss, probabilistic=model.probabilistic)
    test_m = summarize_predictions(test_pred, loss_value=test_loss, probabilistic=model.probabilistic)
    write_predictions_csv(test_pred, out_dir / "test_predictions.csv")
    write_all_plots(
        model,
        test_loader,
        test_pred,
        device,
        plots_dir,
        title=title,
        z_floor=z_floor,
        z_max=z_max,
    )
    task_tag = "logz" if target == "logz" else "z"
    perf = {
        "best_epoch": best_epoch,
        "task": f"redshift_{task_tag}_{model.head_kind}",
        "target": target,
        "head": model.head_kind,
        "val": jsonify(val_m),
        "test": jsonify(test_m),
        "loss_by_epoch": history,
        "z_floor": z_floor,
        "z_max": z_max,
        "log": "ln" if target == "logz" else None,
        "jacobian": jacobian_note(target),
        "point_estimate": "posterior_median" if model.probabilistic else "mse_scalar",
    }
    (out_dir / "model_performance.json").write_text(json.dumps(perf, indent=2), encoding="utf-8")
    (out_dir / "cfg_used.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(
        f"[done] val loss={val_loss:.6f}  test loss={test_loss:.6f}  "
        f"test MAE_z={test_m['linear_z']['mae']:.5f}  RMSE_z={test_m['linear_z']['rmse']:.5f}",
        flush=True,
    )
    if model.probabilistic:
        print(
            f"       test NLL_y={test_m['nll_y']:.5f}  NLL_z={test_m['nll_z']:.5f}  "
            f"CRPS_y={test_m['crps_y']:.5f}",
            flush=True,
        )
        cov = test_m.get("coverage") or {}
        for name, row in cov.items():
            print(
                f"       {name} nominal={row['nominal']:.2f}  empirical={row['empirical']:.3f}  "
                f"mean_width_z={row['mean_width_z']:.4f}",
                flush=True,
            )
        gate = test_m.get("gate_pi_low")
        if gate:
            print(
                f"       gate_pi_low mean={gate['mean']:.3f} std={gate['std']:.3f}",
                flush=True,
            )
    print(f"[done] plots -> {plots_dir}", flush=True)
