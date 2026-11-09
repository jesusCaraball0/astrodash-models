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


def test_covering_affine() -> None:
    from train_utils import NSF_BOUND, covering_std, output_tag, resolve_better_hps

    mean, empir = 0.0525, 0.0909
    std = covering_std(mean, empir, z_lo=1e-4, z_hi=2.0, bound=NSF_BOUND)
    u_hi = (2.0 - mean) / std
    u_lo = (1e-4 - mean) / std
    _check(abs(u_hi - NSF_BOUND) < 1e-6, f"z=2 should sit at u=+{NSF_BOUND}, got {u_hi}")
    _check(u_lo >= -NSF_BOUND - 1e-6, f"z_floor should stay in the box, u={u_lo}")
    _check(std > empir, "covering std should exceed the empirical train std")
    bins, transforms, cover_z = resolve_better_hps(
        better_hps=True, target="z", head="flow", bins=None, transforms=None
    )
    _check(bins == 64 and transforms == 2 and cover_z == 2.0, "better-hps defaults")
    _check(output_tag("z", better_hps=True) == "z_betterhps", output_tag("z", better_hps=True))
    _check(output_tag("z") == "z", "original z tag must stay z")
    try:
        resolve_better_hps(better_hps=True, target="logz", head="flow", bins=None, transforms=None)
        _check(False, "better-hps + logz should error")
    except ValueError:
        pass
    print(f"  covering affine: mean={mean} std={std:.4f} u(z=2)={u_hi:.2f} bins={bins}")


def test_deeper_recipe() -> None:
    from train_utils import output_tag, resolve_deeper

    bins, layers, trunk = resolve_deeper(deeper=True, head="flow", bins=None)
    _check(bins == 16 and layers == 3 and trunk == 2, f"deeper defaults {bins, layers, trunk}")
    bins, layers, trunk = resolve_deeper(deeper=False, head="flow", bins=None)
    _check(bins is None and layers == 2 and trunk == 1, f"default recipe {bins, layers, trunk}")
    _check(output_tag("logz", deeper=True) == "logz_deeper", output_tag("logz", deeper=True))
    _check(output_tag("logz") == "logz", "original logz tag must stay logz")
    try:
        resolve_deeper(deeper=True, head="mse", bins=None)
        _check(False, "deeper + mse should error")
    except ValueError:
        pass
    try:
        output_tag("z", better_hps=True, deeper=True)
        _check(False, "better_hps + deeper tag should error")
    except ValueError:
        pass
    head = SplineFlowHead(16, bins=16, transforms=2, hidden=32, hidden_layers=3)
    head.set_y_stats(0.0, 1.0)
    h = torch.randn(4, 16)
    y = torch.randn(4)
    loss = head.loss(h, y)
    _check(torch.isfinite(loss), f"deeper NSF loss {loss}")
    trunk = LatentMLPTrunk(64, 16, 0.0, n_layers=2)
    _check(trunk(torch.randn(3, 64)).shape == (3, 16), "deeper trunk context")
    dash = DashCNNTrunk(1025, extra_fc=True)
    _check(dash(torch.randn(2, 1025)).shape == (2, 256), "deeper DASH extra fc")
    print(f"  deeper recipe: bins={16} flow_layers={3} trunk_layers={2} loss={loss.item():.4f}")


def test_reweight_nll() -> None:
    import numpy as np

    from train_utils import InverseFreqWeights, output_tag

    z = np.array([0.01] * 16 + [0.08] * 4 + [0.15] * 1, dtype=np.float64)
    tab = InverseFreqWeights.from_z(z, edges=(0.0, 0.05, 0.10, 0.20), alpha=0.5)
    _check(output_tag("logz", reweight=True) == "logz_reweight", output_tag("logz", reweight=True))
    _check(output_tag("logz") == "logz", "original logz tag must stay logz")
    # 16 / 4 / 1 → raw w ∝ 1/4, 1/2, 1; mean-normalized on the 21 samples
    raw = np.array([16.0, 4.0, 1.0]) ** (-0.5)
    w_i = np.concatenate([np.full(16, raw[0]), np.full(4, raw[1]), np.full(1, raw[2])])
    scale = 21.0 / w_i.sum()
    expect = raw * scale
    _check(abs(tab.w_bin[0] - expect[0]) < 1e-6, f"bin0 weight {tab.w_bin[0]} vs {expect[0]}")
    _check(abs(tab.w_bin[2] - expect[2]) < 1e-6, f"bin2 weight {tab.w_bin[2]} vs {expect[2]}")
    w = tab.lookup(torch.tensor([0.01, 0.15]))
    _check(abs(float(w[0]) - expect[0]) < 1e-5, f"lookup low-z {w[0]}")
    _check(abs(float(w[1]) - expect[2]) < 1e-5, f"lookup high-z {w[1]}")
    mean_w = float(tab.lookup(torch.from_numpy(z)).mean())
    _check(abs(mean_w - 1.0) < 1e-5, f"train mean weight should be 1, got {mean_w}")

    torch.manual_seed(0)
    head = SplineFlowHead(8, bins=4, transforms=1, hidden=16)
    head.set_y_stats(0.0, 1.0)
    h = torch.randn(5, 8)
    y = torch.randn(5)
    w = torch.tensor([1.0, 1.0, 4.0, 1.0, 1.0])
    nll = -head.log_prob_y(h, y)
    want = (w * nll).sum() / w.sum()
    got = head.loss(h, y, sample_weight=w)
    _check((got - want).abs().item() < 1e-6, f"weighted NLL {got} vs {want}")
    _check((head.loss(h, y) - nll.mean()).abs().item() < 1e-6, "unweighted still mean NLL")
    print(
        f"  reweight: n={tab.counts} w={[round(x, 3) for x in tab.w_bin]}  "
        f"weighted NLL={got.item():.4f}"
    )


def test_aug_ln1pz() -> None:
    import numpy as np

    from augment_logz import _npz_temp_path, _recover_partial_cache, sample_z_new, warp_spectrum
    from train_utils import Z_FLOOR, Z_MAX, output_tag

    _check(output_tag("logz", aug=True) == "logz_aug", output_tag("logz", aug=True))
    _check(output_tag("logz") == "logz", "original logz tag must stay logz")
    _check(output_tag("logz", reweight=True, aug=True) == "logz_reweight_aug", "aug after reweight")
    try:
        output_tag("logz", cls_cond=True, aug=True)
        _check(False, "aug + cls_cond should error")
    except ValueError:
        pass

    rng = np.random.default_rng(0)
    z_old = np.array([0.005, 0.03, 0.15, 0.5, 1.0], dtype=np.float64)
    z_new, eps = sample_z_new(z_old, rng, k=5, eps_lo=-0.08, eps_hi=0.1)
    _check(z_new.shape == (5, 5), f"z_new shape {z_new.shape}")
    _check(np.all(z_new >= Z_FLOOR - 1e-15), f"z_new below floor {z_new.min()}")
    _check(np.all(z_new <= Z_MAX), f"z_new above max {z_new.max()}")
    _check(np.all(eps >= -0.08 - 1e-12) and np.all(eps <= 0.1 + 1e-12), "eps outside Unif box")
    recon = (1.0 + z_old)[:, None] * np.exp(eps) - 1.0
    _check(np.max(np.abs(recon - z_new)) < 1e-12, "z_new != (1+z_old) exp(ε) - 1")
    _check(float(z_new[1].max()) > 0.03, "z=0.03 should get some redshift")
    _check(float(z_new[3].min()) > 0.3, "z=0.5 blueshift should stay well above the floor")

    wave = 3202.5 + 5.0 * np.arange(1320)
    flux = np.zeros_like(wave)
    peak = 400
    flux[peak] = 1.0
    mask = np.ones_like(wave, dtype=bool)
    z0, z1 = 0.03, 0.13
    f_new, m_new = warp_spectrum(wave, flux, mask, z0, z1)
    s = (1.0 + z1) / (1.0 + z0)
    want_lam = wave[peak] * s
    got_lam = wave[int(np.argmax(f_new))]
    _check(abs(got_lam - want_lam) < 6.0, f"peak moved to {got_lam}, expected {want_lam}")
    _check(bool(m_new[int(np.argmax(f_new))]), "peak bin should stay valid")
    _check(int(m_new.sum()) < mask.sum(), "stretch should pad some bins")
    print(
        f"  aug ln(1+z): z_new[{z_old[1]:.3f}] "
        f"[{z_new[1].min():.4f}, {z_new[1].max():.4f}]  "
        f"peak {wave[peak]:.1f} -> {got_lam:.1f} (want {want_lam:.1f})"
    )

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "latent_aug.npz"
        tmp = _npz_temp_path(out)
        _check(str(tmp).endswith(".tmp.npz"), f"temp must end in .npz, got {tmp}")
        np.savez_compressed(tmp, z=np.arange(4, dtype=np.float32))
        _check(tmp.is_file(), f"savez should write {tmp}")
        _check(not (Path(td) / "latent_aug.tmp.npz.npz").is_file(), "must not double-append .npz")
        tmp.replace(out)
        _check(out.is_file(), "replace onto cache path")
        buggy = Path(td) / "latent_aug.npz.tmp.npz"
        np.savez_compressed(buggy, z=np.array([1.0], dtype=np.float32))
        out.unlink()
        _recover_partial_cache(out)
        _check(out.is_file() and not buggy.is_file(), "recover mangled numpy temp")
    print("  aug cache temp naming ok")


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
    cls_dash = RedshiftPredictor(DashCNNTrunk(1025), "mse", class_dim=5)
    spec_cls = torch.randn(3, 1030)
    loss_cls = cls_dash.loss(spec_cls, torch.randn(3))
    loss_cls.backward()
    _check(torch.isfinite(loss_cls), "dash class_dim mse loss")
    _check(cls_dash.point_y(spec_cls).shape == (3,), "dash class_dim point")
    cls_lat = RedshiftPredictor(LatentMLPTrunk(1029, 32, 0.0), "gaussian")
    x_cls = torch.randn(4, 1029)
    loss_lat = cls_lat.loss(x_cls, torch.randn(4))
    loss_lat.backward()
    _check(torch.isfinite(loss_lat), "latent concat-at-input gaussian loss")
    print(f"  models ok  jacobian err={jac:.2e}  linear-z identity={jac_lin:.2e}")


def test_true_onehot() -> None:
    import numpy as np
    import pandas as pd

    from class_cond import (
        N_CLASSES,
        dash_class_onehot_by_id,
        filter_mapped_indices,
        latent_class_onehot,
    )
    from train_latent import LABEL_COLUMN, normalize_latent_meta
    from train_utils import output_tag

    _check(output_tag("logz", cls_true=True) == "logz_truecls", "truecls output tag")
    _check(output_tag("logz", cls_cond=True) == "logz_cls", "cls output tag")
    by_id = dash_class_onehot_by_id(
        ["ia.txt", "ii.txt"],
        {"ia.txt": {"type": "SN Ia"}, "ii.txt": {"type": "SN II"}},
    )
    _check(tuple(by_id["ia.txt"].shape) == (N_CLASSES,), "dash onehot shape")
    _check(int(by_id["ia.txt"].argmax()) == 0, "SN Ia onehot index")
    _check(int(by_id["ii.txt"].argmax()) == 2, "SN II onehot index")
    _check(float(by_id["ia.txt"].sum()) == 1.0, "dash onehot sums to 1")
    _check(int((by_id["ia.txt"] > 0).sum()) == 1, "dash onehot is exclusive")

    meta_csv = PROJECT_ROOT / "data" / "wiserep_henna" / "try_5_noz" / "Nodered36_5" / "meta_universal.csv"
    meta = normalize_latent_meta(pd.read_csv(meta_csv, nrows=64, low_memory=False))
    idx = filter_mapped_indices(meta, np.arange(len(meta)))
    _check(len(idx) > 0, "expected mapped latent rows")
    oh = latent_class_onehot(meta, idx[: min(8, len(idx))])
    _check(oh.shape[1] == N_CLASSES, f"latent onehot width {oh.shape}")
    _check(torch.allclose(oh.sum(dim=-1), torch.ones(oh.shape[0])), "latent onehot rows sum to 1")
    _check(bool(((oh == 0) | (oh == 1)).all()), "latent onehot is 0/1")
    print(f"  true one-hot ok  n={oh.shape[0]}  counts={oh.sum(dim=0).tolist()}")


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
    test_covering_affine()
    test_deeper_recipe()
    test_reweight_nll()
    test_aug_ln1pz()
    test_flow_inverse_and_jacobian()
    test_moe_mixture()
    test_models()
    test_true_onehot()
    test_leakage_guards()
    tmp = Path(tempfile.mkdtemp(prefix="norm_flow_smoke_"))
    test_saved_outputs(tmp)
    print("all smoke tests passed")


if __name__ == "__main__":
    main()
