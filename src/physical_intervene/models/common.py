"""Shared batch format and helpers for the learned models.

Every model sees the same inputs (DESIGN.md, Models): s_t, the last HISTORY_LEN (state, action)
pairs with their mask, and a_t. Inputs come both standardised (by the training split's InputNorm)
and in physical units (contact_wm uses the latter). Every model outputs the standardised state
change z = (delta_s - mean) / scale, with the delta scaler fitted on the training split.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from physical_intervene.data import Examples, InputNorm, TargetScaler

WIDTH_NAMES = ("small", "medium", "large")


@dataclass
class Batch:
    s_n: torch.Tensor  # (B, 11)
    hs_n: torch.Tensor  # (B, L, 11)
    ha_n: torch.Tensor  # (B, L, 2)
    mask: torch.Tensor  # (B, L)
    a_n: torch.Tensor  # (B, 2)
    s: torch.Tensor  # (B, 11), physical units
    a: torch.Tensor  # (B, 2), physical units

    def __len__(self) -> int:
        return self.s.shape[0]

    def with_action(self, a: torch.Tensor, a_n: torch.Tensor) -> Batch:
        return Batch(self.s_n, self.hs_n, self.ha_n, self.mask, a_n, self.s, a)

    def repeat(self, n: int) -> Batch:
        return Batch(*(x.repeat_interleave(n, dim=0) for x in self.__dict__.values()))


def make_batch(ex: Examples, norm: InputNorm, idx: np.ndarray | None = None, dtype=torch.float32) -> Batch:
    if idx is not None:
        ex = ex.subset(idx)

    def t(x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.ascontiguousarray(x), dtype=dtype)

    hs_n = (ex.hist_s - norm.s_mean) / norm.s_std
    ha_n = (ex.hist_a - norm.a_mean) / norm.a_std * ex.hist_mask[..., None]  # padding stays 0
    return Batch(
        s_n=t((ex.s - norm.s_mean) / norm.s_std),
        hs_n=t(hs_n),
        ha_n=t(ha_n),
        mask=t(ex.hist_mask),
        a_n=t((ex.a - norm.a_mean) / norm.a_std),
        s=t(ex.s),
        a=t(ex.a),
    )


def norm_action(a: np.ndarray, norm: InputNorm) -> np.ndarray:
    return (a - norm.a_mean) / norm.a_std


class WorldModel(nn.Module):
    """Base class: one-step prediction of the standardised state change."""

    arch = "base"
    stochastic = False

    def __init__(self, norm: InputNorm, dscaler: TargetScaler) -> None:
        super().__init__()
        self.register_buffer("d_mean", torch.as_tensor(dscaler.mean, dtype=torch.float32))
        self.register_buffer("d_scale", torch.as_tensor(dscaler.scale, dtype=torch.float32))
        self.norm = norm

    def forward(self, b: Batch) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def predict_z(self, b: Batch, generator: torch.Generator | None = None) -> torch.Tensor:
        """The model's prediction for scoring (deterministic models: forward)."""
        return self.forward(b)

    def n_params(self) -> int:
        """Parameters that count towards the width (for contact_wm, its correction network)."""
        return sum(p.numel() for p in self.parameters())


def mlp_stack(sizes: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:], strict=True)):
        layers.append(nn.Linear(a, b))
        if i < len(sizes) - 2:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)
