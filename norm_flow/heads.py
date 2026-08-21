"""Predictive heads: MSE, heteroscedastic Gaussian, 1D NSF, soft NSF mixture."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG2PI = math.log(2.0 * math.pi)
SIGMA_EPS = 1e-4
HEADS = ("mse", "gaussian", "flow", "moe")
FLOW_LIKE_HEADS = ("flow", "moe")
PI_EPS = 1e-4
MOE_BISECT_ITERS = 24
MOE_YSTD_BOUND = 4.9


def _as_1d(y: torch.Tensor) -> torch.Tensor:
    return y.reshape(-1)


def _ndtri(probs: torch.Tensor) -> torch.Tensor:
    """Inverse standard-normal CDF (quantile function of N(0,1))."""
    p = probs.clamp(1e-6, 1.0 - 1e-6)
    # Apple MPS does not implement aten::special_ndtri. Quantile eval is a
    # tiny 1D vector, so run on CPU and copy the result back to the model device.
    return torch.special.ndtri(p.detach().cpu()).to(device=probs.device, dtype=probs.dtype)


class MSEHead(nn.Module):
    kind = "mse"

    def __init__(self, context_dim: int):
        super().__init__()
        self.lin = nn.Linear(context_dim, 1)

    def loss(self, context: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(self.point_y(context), _as_1d(y))

    def point_y(self, context: torch.Tensor) -> torch.Tensor:
        return self.lin(context).squeeze(-1)

    def log_prob_y(self, context: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.full_like(_as_1d(y), float("nan"))

    def quantiles_y(self, context: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        pred = self.point_y(context)
        return pred.unsqueeze(-1).expand(-1, int(probs.numel()))

    def pit(self, context: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.full_like(_as_1d(y), float("nan"))

    def crps_y(self, context: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (self.point_y(context) - _as_1d(y)).abs()


class GaussianHead(nn.Module):
    kind = "gaussian"

    def __init__(self, context_dim: int, sigma_eps: float = SIGMA_EPS):
        super().__init__()
        self.lin = nn.Linear(context_dim, 2)
        self.sigma_eps = float(sigma_eps)

    def params(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.lin(context)
        mu = raw[..., 0]
        sigma = F.softplus(raw[..., 1]) + self.sigma_eps
        return mu, sigma

    def loss(self, context: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return -self.log_prob_y(context, y).mean()

    def point_y(self, context: torch.Tensor) -> torch.Tensor:
        mu, _ = self.params(context)
        return mu

    def log_prob_y(self, context: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mu, sigma = self.params(context)
        y = _as_1d(y)
        return -0.5 * ((y - mu) / sigma) ** 2 - torch.log(sigma) - 0.5 * LOG2PI

    def quantiles_y(self, context: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        mu, sigma = self.params(context)
        z = _ndtri(probs.to(device=context.device, dtype=context.dtype))
        return mu.unsqueeze(-1) + sigma.unsqueeze(-1) * z.reshape(1, -1)

    def pit(self, context: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mu, sigma = self.params(context)
        y = _as_1d(y)
        return 0.5 * (1.0 + torch.erf((y - mu) / (sigma * math.sqrt(2.0))))

    def crps_y(self, context: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mu, sigma = self.params(context)
        y = _as_1d(y)
        z = (y - mu) / sigma
        phi = torch.exp(-0.5 * z**2) / math.sqrt(2.0 * math.pi)
        cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
        return sigma * (z * (2.0 * cdf - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))


class SplineFlowHead(nn.Module):
    """1D conditional rational-quadratic spline flow on standardized log-z."""

    kind = "flow"

    def __init__(
        self,
        context_dim: int,
        *,
        bins: int = 8,
        transforms: int = 2,
        hidden: int = 256,
        slope: float = 1e-3,
        n_crps_samples: int = 128,
    ):
        super().__init__()
        try:
            from zuko.flows import NSF
        except ImportError as exc:
            raise ImportError(
                "Conditional spline flow requires zuko. Install with: pip install 'zuko>=1.4'"
            ) from exc
        self.flow = NSF(
            features=1,
            context=int(context_dim),
            bins=int(bins),
            transforms=int(transforms),
            hidden_features=(int(hidden), int(hidden)),
            slope=float(slope),
        )
        self.n_crps_samples = int(n_crps_samples)
        self.register_buffer("y_mean", torch.zeros(1))
        self.register_buffer("y_std", torch.ones(1))

    def set_y_stats(self, mean: float, std: float) -> None:
        std = max(float(std), 1e-3)
        self.y_mean.fill_(float(mean))
        self.y_std.fill_(std)

    def standardize(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.y_mean) / self.y_std.clamp(min=1e-6)

    def unstandardize(self, y_std: torch.Tensor) -> torch.Tensor:
        return y_std * self.y_std + self.y_mean

    def _dist(self, context: torch.Tensor):
        return self.flow(context)

    def loss(self, context: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return -self.log_prob_y(context, y).mean()

    def log_prob_y(self, context: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_std = self.standardize(_as_1d(y).unsqueeze(-1))
        lp_std = self._dist(context).log_prob(y_std)
        return lp_std - torch.log(self.y_std.clamp(min=1e-6)).reshape(())

    def point_y(self, context: torch.Tensor) -> torch.Tensor:
        probs = torch.tensor([0.5], device=context.device, dtype=context.dtype)
        return self.quantiles_y(context, probs).squeeze(-1)

    def quantiles_y(self, context: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        dist = self._dist(context)
        batch = int(context.shape[0])
        out = []
        for q in probs.reshape(-1).tolist():
            z = _ndtri(torch.tensor(q, device=context.device, dtype=context.dtype)).expand(batch, 1)
            y_std = dist.transform.inv(z)
            out.append(self.unstandardize(y_std).squeeze(-1))
        return torch.stack(out, dim=-1)

    def pit(self, context: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_std = self.standardize(_as_1d(y).unsqueeze(-1))
        z = self._dist(context).transform(y_std)
        return 0.5 * (1.0 + torch.erf(z.squeeze(-1) / math.sqrt(2.0)))

    def crps_y(self, context: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dist = self._dist(context)
        samples = dist.sample((self.n_crps_samples,)).squeeze(-1)
        samples = self.unstandardize(samples)
        return _crps_from_samples(samples, y)

    def sample_y(self, context: torch.Tensor, n: int = 1) -> torch.Tensor:
        samples = self._dist(context).sample((int(n),)).squeeze(-1)
        return self.unstandardize(samples)


def _crps_from_samples(samples: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Energy-form CRPS from samples with shape (n_samples, batch)."""
    y = _as_1d(y)
    term1 = (samples - y.unsqueeze(0)).abs().mean(0)
    s = samples.sort(dim=0).values
    n = s.shape[0]
    idx = torch.arange(n, device=s.device, dtype=s.dtype).unsqueeze(1)
    pairwise = 2.0 * ((2.0 * idx - n + 1.0) * s).sum(0) / float(n * n)
    return term1 - 0.5 * pairwise


class MixtureSplineHead(nn.Module):
    """Soft two-expert mixture of conditional spline flows on the same context.

    p(y | x) = π(x) p_low(y | x) + (1 − π(x)) p_high(y | x)

    π(x) is a learned gate; experts are not hard-routed and are not given
    the true redshift. Both NSFs share y = ln z standardization.
    """

    kind = "moe"

    def __init__(
        self,
        context_dim: int,
        *,
        bins: int = 8,
        transforms: int = 2,
        hidden: int = 256,
        slope: float = 1e-3,
        n_crps_samples: int = 128,
    ):
        super().__init__()
        kw = dict(bins=bins, transforms=transforms, hidden=hidden, slope=slope)
        self.low = SplineFlowHead(context_dim, **kw)
        self.high = SplineFlowHead(context_dim, **kw)
        self.gate = nn.Linear(int(context_dim), 1)
        self.n_crps_samples = int(n_crps_samples)
        self.register_buffer("y_mean", torch.zeros(1))
        self.register_buffer("y_std", torch.ones(1))

    def set_y_stats(self, mean: float, std: float) -> None:
        self.low.set_y_stats(mean, std)
        self.high.set_y_stats(mean, std)
        std = max(float(std), 1e-3)
        self.y_mean.fill_(float(mean))
        self.y_std.fill_(std)

    def gate_pi_low(self, context: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.gate(context).squeeze(-1)).clamp(PI_EPS, 1.0 - PI_EPS)

    def loss(self, context: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return -self.log_prob_y(context, y).mean()

    def log_prob_y(self, context: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pi = self.gate_pi_low(context)
        stacked = torch.stack(
            [
                torch.log(pi) + self.low.log_prob_y(context, y),
                torch.log(1.0 - pi) + self.high.log_prob_y(context, y),
            ],
            dim=0,
        )
        return torch.logsumexp(stacked, dim=0)

    def pit(self, context: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pi = self.gate_pi_low(context)
        return pi * self.low.pit(context, y) + (1.0 - pi) * self.high.pit(context, y)

    def point_y(self, context: torch.Tensor) -> torch.Tensor:
        probs = torch.tensor([0.5], device=context.device, dtype=context.dtype)
        return self.quantiles_y(context, probs).squeeze(-1)

    def quantiles_y(self, context: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        """Invert the mixture CDF by batched bisection on the NSF y-domain."""
        batch = int(context.shape[0])
        q = probs.to(device=context.device, dtype=context.dtype).reshape(-1)
        n_q = int(q.numel())
        target = q.reshape(1, n_q).expand(batch, n_q)
        ctx = context.unsqueeze(1).expand(batch, n_q, -1).reshape(batch * n_q, -1)
        lo = self.low.unstandardize(
            torch.full((batch, n_q), -MOE_YSTD_BOUND, device=context.device, dtype=context.dtype)
        )
        hi = self.low.unstandardize(
            torch.full((batch, n_q), MOE_YSTD_BOUND, device=context.device, dtype=context.dtype)
        )
        for _ in range(MOE_BISECT_ITERS):
            mid = 0.5 * (lo + hi)
            cdf = self.pit(ctx, mid.reshape(-1)).reshape(batch, n_q)
            go_right = cdf < target
            lo = torch.where(go_right, mid, lo)
            hi = torch.where(go_right, hi, mid)
        return 0.5 * (lo + hi)

    def sample_y(self, context: torch.Tensor, n: int = 1) -> torch.Tensor:
        n = int(n)
        s_low = self.low.sample_y(context, n)
        s_high = self.high.sample_y(context, n)
        pi = self.gate_pi_low(context).unsqueeze(0)
        take_low = torch.rand(n, int(context.shape[0]), device=context.device, dtype=context.dtype) < pi
        return torch.where(take_low, s_low, s_high)

    def crps_y(self, context: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        samples = self.sample_y(context, n=self.n_crps_samples)
        return _crps_from_samples(samples, y)


def build_head(
    kind: str,
    context_dim: int,
    *,
    bins: int = 8,
    transforms: int = 2,
    hidden: int = 256,
    slope: float = 1e-3,
) -> nn.Module:
    kind = str(kind).lower()
    if kind == "mse":
        return MSEHead(context_dim)
    if kind == "gaussian":
        return GaussianHead(context_dim)
    if kind == "flow":
        return SplineFlowHead(
            context_dim, bins=bins, transforms=transforms, hidden=hidden, slope=slope
        )
    if kind == "moe":
        return MixtureSplineHead(
            context_dim, bins=bins, transforms=transforms, hidden=hidden, slope=slope
        )
    raise ValueError(f"Unknown head {kind!r}; expected one of {HEADS}")


def is_probabilistic(head: nn.Module) -> bool:
    return getattr(head, "kind", None) in {"gaussian", "flow", "moe"}
