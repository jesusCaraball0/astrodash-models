"""Shared training loop for latent and DASH redshift density models."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from models import TARGETS, RedshiftPredictor, encode_y

Z_FLOOR = 1e-4
Z_MAX = 6.0


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


def assert_finite(t: torch.Tensor, name: str) -> None:
    if not torch.isfinite(t).all():
        raise RuntimeError(f"Non-finite {name}: min={t.min().item() if t.numel() else 'empty'} max={t.max().item() if t.numel() else 'empty'}")


@torch.no_grad()
def mean_loss(
    model: RedshiftPredictor,
    loader: DataLoader,
    device: torch.device,
    *,
    z_floor: float,
    target: str | None = None,
) -> float:
    target = _target_of(model, target)
    model.eval()
    total = 0.0
    n = 0
    for batch in loader:
        if batch is None:
            continue
        x = batch["x"].to(device)
        y = to_y(batch["y"].to(device), target, z_floor)
        loss = model.loss(x, y)
        bs = int(x.shape[0])
        total += float(loss.item()) * bs
        n += bs
    return total / max(n, 1)


def fit_flow_y_stats(
    model: RedshiftPredictor,
    loader: DataLoader,
    device: torch.device,
    *,
    z_floor: float,
    target: str | None = None,
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
    model.set_flow_y_stats(mean, std)
    y_name = "ln z" if target == "logz" else "z"
    print(
        f"[{model.head_kind}] standardizing y={y_name} with train mean={mean:.4f} std={std:.4f} "
        "(zuko NSF domain is [-5, 5] after this affine map)",
        flush=True,
    )
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
        y_mean, y_std = fit_flow_y_stats(
            model, train_loader, device, z_floor=z_floor, target=target
        )
        cfg["flow_y_mean"] = y_mean
        cfg["flow_y_std"] = y_std

    best_val = float("inf")
    best_epoch = -1
    no_improve = 0
    history: List[List[Any]] = []
    nan_streak = 0

    print(
        f"[train] head={model.head_kind}  target={target}  epochs={epochs}  "
        f"patience={patience}  clip_norm={clip_norm}  out_dir={out_dir}",
        flush=True,
    )

    for epoch in range(1, int(epochs) + 1):
        t0 = time.perf_counter()
        model.train()
        running = 0.0
        n = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False, dynamic_ncols=True)
        for batch in pbar:
            if batch is None:
                continue
            x = batch["x"].to(device)
            y = to_y(batch["y"].to(device), target, z_floor)
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(x, y)
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
            bs = int(x.shape[0])
            running += float(loss.item()) * bs
            n += bs
            pbar.set_postfix(loss=f"{running / max(n, 1):.5f}")

        train_loss = running / max(n, 1)
        val_loss = mean_loss(model, val_loader, device, z_floor=z_floor, target=target)
        if scheduler is not None:
            scheduler.step(val_loss)
        history.append([epoch, train_loss, val_loss])
        dt = time.perf_counter() - t0
        print(
            f"epoch {epoch}/{epochs} {dt:.1f}s  train {train_loss:.6f}  val {val_loss:.6f}",
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
