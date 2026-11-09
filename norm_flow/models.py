"""Encoder trunks + redshift predictor. Trunks match the existing MSE models.

The MoE head is two spline experts on the same trunk context, not two encoders.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from heads import build_head, is_probabilistic

TARGETS = ("logz", "z")


def encode_y(z: torch.Tensor, target: str, z_floor: float) -> torch.Tensor:
    """Map physical redshift to the training-space random variable y."""
    z = z.reshape(-1).clamp(min=float(z_floor))
    if target == "logz":
        return torch.log(z)
    if target == "z":
        return z
    raise ValueError(f"Unknown target {target!r}; expected one of {TARGETS}")


def y_to_z(y: torch.Tensor, target: str, z_floor: float, z_max: float) -> torch.Tensor:
    """Map a training-space prediction back to physical redshift."""
    if target == "logz":
        lo, hi = math.log(float(z_floor)), math.log(float(z_max))
        return torch.exp(y.clamp(lo, hi))
    if target == "z":
        return y.clamp(float(z_floor), float(z_max))
    raise ValueError(f"Unknown target {target!r}; expected one of {TARGETS}")


class LatentMLPTrunk(nn.Module):
    """MLP on a frozen embedding: Linear → GELU → Dropout, repeated ``n_layers`` times.

    Default ``n_layers=1`` matches ``LatentClassifier``. ``--deeper`` uses 2.
    """

    def __init__(self, embed_dim: int, hidden: int, dropout: float, n_layers: int = 1):
        super().__init__()
        n_layers = int(n_layers)
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        self.context_dim = int(hidden)
        layers: list[nn.Module] = [
            nn.Linear(int(embed_dim), int(hidden)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        ]
        for _ in range(n_layers - 1):
            layers.extend(
                [
                    nn.Linear(int(hidden), int(hidden)),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                ]
            )
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.reshape(x.shape[0], -1))


class DashCNNTrunk(nn.Module):
    """Dash 1D CNN up to the 256-d feature used by ``fc2`` in ``DashCNN1D``."""

    def __init__(self, input_length: int, extra_fc: bool = False):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
        )
        conv_out_length = int(input_length) // (4 * 4 * 4)
        self.flat_size = 128 * conv_out_length
        self.fc1 = nn.Linear(self.flat_size, 256)
        self.fc_extra = nn.Linear(256, 256) if extra_fc else None
        self.dropout = nn.Dropout(0.5)
        self.context_dim = 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        if self.fc_extra is not None:
            x = self.dropout(F.relu(self.fc_extra(x)))
        return x


class RedshiftPredictor(nn.Module):
    """Trunk that emits context h(x), plus an MSE / Gaussian / flow / MoE head.

    ``target`` is the training-space random variable: ``logz`` (y = ln z) or
    ``z`` (y is physical redshift). Heads are unchanged; only the label transform
    and the z-space Jacobian differ.
    """

    def __init__(
        self,
        trunk: nn.Module,
        head_kind: str,
        *,
        bins: int = 8,
        transforms: int = 2,
        flow_hidden: int = 256,
        flow_hidden_layers: int = 2,
        slope: float = 1e-3,
        target: str = "logz",
        class_dim: int = 0,
    ):
        super().__init__()
        target = str(target).lower()
        if target not in TARGETS:
            raise ValueError(f"Unknown target {target!r}; expected one of {TARGETS}")
        self.trunk = trunk
        self.head_kind = str(head_kind).lower()
        self.target = target
        self.class_dim = int(class_dim)
        self.head = build_head(
            self.head_kind,
            int(trunk.context_dim) + self.class_dim,
            bins=bins,
            transforms=transforms,
            hidden=flow_hidden,
            hidden_layers=flow_hidden_layers,
            slope=slope,
        )

    @property
    def probabilistic(self) -> bool:
        return is_probabilistic(self.head)

    def context(self, x: torch.Tensor) -> torch.Tensor:
        if self.class_dim <= 0:
            return self.trunk(x)
        feat = x[:, : -self.class_dim]
        cls = x[:, -self.class_dim :]
        return torch.cat([self.trunk(feat), cls], dim=-1)

    def loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.head.loss(self.context(x), y, sample_weight=sample_weight)

    def per_sample_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.head.per_sample_loss(self.context(x), y)

    def point_y(self, x: torch.Tensor) -> torch.Tensor:
        return self.head.point_y(self.context(x))

    def log_prob_y(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.head.log_prob_y(self.context(x), y)

    def quantiles_y(self, x: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        return self.head.quantiles_y(self.context(x), probs)

    def pit(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.head.pit(self.context(x), y)

    def crps_y(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.head.crps_y(self.context(x), y)

    def log_prob_z(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        *,
        z_floor: float,
    ) -> torch.Tensor:
        z = z.reshape(-1).clamp(min=float(z_floor))
        y = encode_y(z, self.target, z_floor)
        lp_y = self.log_prob_y(x, y)
        if self.target == "logz":
            return lp_y - torch.log(z)
        return lp_y

    def set_flow_y_stats(self, mean: float, std: float) -> None:
        if hasattr(self.head, "set_y_stats"):
            self.head.set_y_stats(mean, std)

    def extra_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "head_kind": self.head_kind,
            "target": self.target,
            "class_dim": self.class_dim,
        }
        if hasattr(self.head, "y_mean"):
            state["y_mean"] = float(self.head.y_mean.reshape(-1)[0].item())
            state["y_std"] = float(self.head.y_std.reshape(-1)[0].item())
        return state
