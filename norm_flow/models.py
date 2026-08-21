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
    """Same first layer as ``LatentClassifier``: Linear → GELU → Dropout."""

    def __init__(self, embed_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.context_dim = int(hidden)
        self.net = nn.Sequential(
            nn.Linear(int(embed_dim), int(hidden)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.reshape(x.shape[0], -1))


class DashCNNTrunk(nn.Module):
    """Dash 1D CNN up to the 256-d feature used by ``fc2`` in ``DashCNN1D``."""

    def __init__(self, input_length: int):
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
        self.dropout = nn.Dropout(0.5)
        self.context_dim = 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.dropout(x)


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
        slope: float = 1e-3,
        target: str = "logz",
    ):
        super().__init__()
        target = str(target).lower()
        if target not in TARGETS:
            raise ValueError(f"Unknown target {target!r}; expected one of {TARGETS}")
        self.trunk = trunk
        self.head_kind = str(head_kind).lower()
        self.target = target
        self.head = build_head(
            self.head_kind,
            int(trunk.context_dim),
            bins=bins,
            transforms=transforms,
            hidden=flow_hidden,
            slope=slope,
        )

    @property
    def probabilistic(self) -> bool:
        return is_probabilistic(self.head)

    def context(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.head.loss(self.context(x), y)

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
        state: dict[str, Any] = {"head_kind": self.head_kind, "target": self.target}
        if hasattr(self.head, "y_mean"):
            state["y_mean"] = float(self.head.y_mean.reshape(-1)[0].item())
            state["y_std"] = float(self.head.y_std.reshape(-1)[0].item())
        return state
