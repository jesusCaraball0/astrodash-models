#!/usr/bin/env python3
"""Lightweight checks: shapes, finite log_prob, inverse, grads, leakage guards."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from paths import PROJECT_ROOT, setup_imports

setup_imports()

from heads import SplineFlowHead, build_head
from leakage import (
    assert_observed_frame_latent_dir,
    assert_observed_frame_splits,
    is_observed_frame_latent_dir,
)
from models import DashCNNTrunk, LatentMLPTrunk, RedshiftPredictor
from train_utils import to_logz


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def test_heads() -> None:
    torch.manual_seed(0)
    b, c = 8, 32
    h = torch.randn(b, c)
    y = torch.randn(b)
    for kind in ("mse", "gaussian", "flow", "moe"):
        head = build_head(kind, c, bins=8, transforms=2, hidden=32)
        if kind in ("flow", "moe"):
            head.set_y_stats(0.1, 1.2)
        loss = head.loss(h, y)
        _check(torch.isfinite(loss), f"{kind} loss not finite: {loss}")
        loss.backward()
        grads = [p.grad.abs().sum() for p in head.parameters() if p.grad is not None]
        _check(bool(grads) and sum(g.item() for g in grads) > 0, f"{kind} has no grads")
        pred = head.point_y(h)
        _check(pred.shape == (b,), f"{kind} point shape {pred.shape}")
        q = head.quantiles_y(h, torch.tensor([0.16, 0.5, 0.84]))
        _check(q.shape == (b, 3), f"{kind} quantile shape {q.shape}")
        if kind != "mse":
            _check((q[:, 0] <= q[:, 1]).all() and (q[:, 1] <= q[:, 2]).all(), f"{kind} quantiles unordered")
        print(f"  head {kind}: loss={loss.item():.4f} point={tuple(pred.shape)} q={tuple(q.shape)}")


def test_flow_inverse_and_jacobian() -> None:
    torch.manual_seed(1)
    head = SplineFlowHead(16, bins=8, transforms=2, hidden=32)
    head.set_y_stats(-2.0, 1.5)
    context = torch.randn(5, 16)
    y = torch.linspace(-3.0, 1.0, 5)
    y_std = head.standardize(y.unsqueeze(-1))
    dist = head._dist(context)
    z = dist.transform(y_std)
    y_back = dist.transform.inv(z)
    err = (y_back - y_std).abs().max().item()
    _check(err < 1e-4, f"flow inverse error {err}")
    samples = head.sample_y(context, n=4)
    _check(samples.shape[0] == 4 and samples.shape[-1] == 5, f"sample shape {samples.shape}")
    z_phys = torch.exp(y)
    lp_y = head.log_prob_y(context, y)
    lp_z = lp_y - torch.log(z_phys)
    _check(torch.isfinite(lp_y).all() and torch.isfinite(lp_z).all(), "non-finite log_prob")
    pit = head.pit(context, y)
    _check(((pit >= 0) & (pit <= 1)).all(), f"PIT out of range {pit}")
    print(f"  flow inverse max err={err:.2e}  pit={pit.detach().numpy().round(3)}")


def test_moe_mixture() -> None:
    torch.manual_seed(3)
    head = build_head("moe", 16, bins=8, transforms=1, hidden=32)
    head.set_y_stats(-2.0, 1.0)
    with torch.no_grad():
        head.gate.weight.zero_()
        head.gate.bias.fill_(20.0)
    h = torch.randn(6, 16)
    y = torch.linspace(-3.0, 0.0, 6)
    lp_mix = head.log_prob_y(h, y)
    lp_low = head.low.log_prob_y(h, y)
    err = (lp_mix - lp_low).abs().max().item()
    _check(err < 1e-3, f"moe pi=1 log_prob mismatch {err}")
    pit = head.pit(h, y)
    _check(((pit >= 0) & (pit <= 1)).all(), f"moe PIT out of range {pit}")
    q = head.quantiles_y(h, torch.tensor([0.5]))
    cdf_at_med = head.pit(h, q.squeeze(-1))
    med_err = (cdf_at_med - 0.5).abs().max().item()
    _check(med_err < 0.05, f"moe median CDF not ~0.5 ({med_err})")
    print(f"  moe pi=1 log_prob err={err:.2e}  median CDF err={med_err:.3f}")


def test_models() -> None:
    torch.manual_seed(2)
    latent = RedshiftPredictor(LatentMLPTrunk(1024, 64, 0.1), "gaussian")
    x = torch.randn(4, 1024)
    y = torch.randn(4)
    loss = latent.loss(x, y)
    loss.backward()
    _check(torch.isfinite(loss), "latent gaussian loss")
    dash = RedshiftPredictor(DashCNNTrunk(1025), "mse")
    spec = torch.randn(3, 1025)
    loss2 = dash.loss(spec, torch.randn(3))
    loss2.backward()
    _check(dash.point_y(spec).shape == (3,), "dash mse point")
    flow = RedshiftPredictor(LatentMLPTrunk(128, 32, 0.0), "flow", bins=8, transforms=1, flow_hidden=32)
    flow.set_flow_y_stats(0.0, 1.0)
    z = torch.tensor([0.01, 0.05, 0.2, 0.4])
    y = to_logz(z)
    lp_y = flow.log_prob_y(torch.randn(4, 128), y)
    lp_z = flow.log_prob_z(torch.randn(4, 128), z, z_floor=1e-4)
    # different random x so only check finite + jacobian identity on a shared forward
    x = torch.randn(4, 128)
    lp_y = flow.log_prob_y(x, y)
    lp_z = flow.log_prob_z(x, z, z_floor=1e-4)
    jac = (lp_y - torch.log(z) - lp_z).abs().max().item()
    _check(jac < 1e-5, f"z-space jacobian mismatch {jac}")
    flow_lin = RedshiftPredictor(
        LatentMLPTrunk(128, 32, 0.0), "flow", bins=8, transforms=1, flow_hidden=32, target="z"
    )
    flow_lin.set_flow_y_stats(0.1, 0.2)
    lp_y_lin = flow_lin.log_prob_y(x, z)
    lp_z_lin = flow_lin.log_prob_z(x, z, z_floor=1e-4)
    jac_lin = (lp_y_lin - lp_z_lin).abs().max().item()
    _check(jac_lin < 1e-5, f"linear-z jacobian should be identity, got {jac_lin}")
    print(f"  models ok  jacobian err={jac:.2e}  linear-z identity={jac_lin:.2e}")


def test_leakage_guards() -> None:
    ok = PROJECT_ROOT / "data" / "wiserep_henna" / "try_5_noz" / "Nodered36_5"
    _check(is_observed_frame_latent_dir(ok), f"should accept {ok}")
    assert_observed_frame_latent_dir(ok)
    bad = PROJECT_ROOT / "data" / "wiserep_henna" / "try_5" / "Dered36_5"
    try:
        assert_observed_frame_latent_dir(bad)
    except RuntimeError:
        pass
    else:
        raise AssertionError("deredshifted latent dir was not rejected")
    try:
        assert_observed_frame_splits(PROJECT_ROOT / "data" / "wiserep" / "henna_matched_split_z_seed36.json")
    except RuntimeError:
        pass
    else:
        raise AssertionError("rest-frame splits JSON was not rejected")
    assert_observed_frame_splits(PROJECT_ROOT / "data" / "wiserep" / "henna_matched_split_noz_seed36.json")
    print("  leakage guards ok")


def test_saved_outputs(tmp: Path) -> None:
    from torch.utils.data import DataLoader

    from train_utils import train_and_eval

    class _DS(torch.utils.data.Dataset):
        def __init__(self, n: int, seed: int):
            g = torch.Generator().manual_seed(seed)
            self.x = torch.randn(n, 32, generator=g)
            z = torch.exp(torch.randn(n, generator=g) * 0.4 - 2.0).clamp(1e-3, 2.0)
            self.y = z

        def __len__(self) -> int:
            return int(self.y.numel())

        def __getitem__(self, i: int):
            return {"x": self.x[i], "y": self.y[i], "id": str(i)}

    def _collate(batch):
        return {
            "x": torch.stack([b["x"] for b in batch]),
            "y": torch.stack([b["y"] for b in batch]),
            "id": [b["id"] for b in batch],
        }

    device = torch.device("cpu")
    for kind in ("mse", "gaussian", "flow", "moe"):
        model = RedshiftPredictor(
            LatentMLPTrunk(32, 16, 0.0),
            kind,
            bins=8,
            transforms=1,
            flow_hidden=16,
        ).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loaders = [
            DataLoader(_DS(48, 0), batch_size=16, collate_fn=_collate),
            DataLoader(_DS(24, 1), batch_size=16, collate_fn=_collate),
            DataLoader(_DS(24, 2), batch_size=16, collate_fn=_collate),
        ]
        out = tmp / kind
        train_and_eval(
            model,
            *loaders,
            device=device,
            out_dir=out,
            cfg={"head": kind, "target": "logz"},
            optimizer=opt,
            epochs=1,
            patience=1,
            clip_norm=1.0 if kind in ("flow", "moe") else None,
            title=f"smoke {kind}",
        )
        _check((out / "model_best.pt").is_file(), f"{kind} missing checkpoint")
        _check((out / "model_performance.json").is_file(), f"{kind} missing metrics")
        _check((out / "test_predictions.csv").is_file(), f"{kind} missing csv")
        _check((out / "plots" / "z_true_vs_pred.png").is_file(), f"{kind} missing scatter")
        if kind != "mse":
            _check((out / "plots" / "pit_histogram.png").is_file(), f"{kind} missing PIT")
            _check((out / "plots" / "coverage.png").is_file(), f"{kind} missing coverage")
            _check((out / "plots" / "posteriors").is_dir(), f"{kind} missing posterior dir")
        if kind == "moe":
            import pandas as pd

            cols = pd.read_csv(out / "test_predictions.csv").columns
            _check("gate_pi_low" in cols, "moe CSV missing gate_pi_low")
        print(f"  saved outputs ok for {kind} -> {out}")

    for kind in ("gaussian", "flow"):
        model = RedshiftPredictor(
            LatentMLPTrunk(32, 16, 0.0),
            kind,
            bins=8,
            transforms=1,
            flow_hidden=16,
            target="z",
        ).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loaders = [
            DataLoader(_DS(48, 0), batch_size=16, collate_fn=_collate),
            DataLoader(_DS(24, 1), batch_size=16, collate_fn=_collate),
            DataLoader(_DS(24, 2), batch_size=16, collate_fn=_collate),
        ]
        out = tmp / f"{kind}_z"
        train_and_eval(
            model,
            *loaders,
            device=device,
            out_dir=out,
            cfg={"head": kind, "target": "z"},
            optimizer=opt,
            epochs=1,
            patience=1,
            clip_norm=1.0 if kind == "flow" else None,
            title=f"smoke {kind} z",
            target="z",
        )
        _check((out / "model_best.pt").is_file(), f"{kind} z missing checkpoint")
        _check((out / "model_performance.json").is_file(), f"{kind} z missing metrics")
        print(f"  saved outputs ok for {kind} target=z -> {out}")


def main() -> None:
    print("norm_flow smoke tests")
    test_heads()
    test_flow_inverse_and_jacobian()
    test_moe_mixture()
    test_models()
    test_leakage_guards()
    tmp = Path(tempfile.mkdtemp(prefix="norm_flow_smoke_"))
    test_saved_outputs(tmp)
    print("all smoke tests passed")


if __name__ == "__main__":
    main()
