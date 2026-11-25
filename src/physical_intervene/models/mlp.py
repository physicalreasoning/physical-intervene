"""mlp (main), blind (sanity control: a_t removed) and the input side of mlp-shortcut.

A residual MLP on (s_t, history, mask, a_t) to the standardised state change. mlp-shortcut is
this model; the shadow-action swap happens in training (train.py), never in the model.
"""

from __future__ import annotations

import torch
from torch import nn

from physical_intervene import constants as c
from physical_intervene.models.common import Batch, WorldModel
from physical_intervene.sim import OBS_DIM


class ResidualBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.l1, self.l2 = nn.Linear(width, width), nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.l2(nn.functional.silu(self.l1(nn.functional.silu(x))))


def mlp_input_dim(use_action: bool) -> int:
    return OBS_DIM + c.HISTORY_LEN * (OBS_DIM + 2) + c.HISTORY_LEN + (2 if use_action else 0)


class MLP(WorldModel):
    arch = "mlp"

    def __init__(self, norm, dscaler, hidden: int, use_action: bool = True) -> None:
        super().__init__(norm, dscaler)
        self.use_action = use_action
        self.inp = nn.Linear(mlp_input_dim(use_action), hidden)
        self.blocks = nn.Sequential(*[ResidualBlock(hidden) for _ in range(c.MLP_RESIDUAL_BLOCKS)])
        self.out = nn.Linear(hidden, OBS_DIM)

    def inputs(self, b: Batch) -> torch.Tensor:
        parts = [b.s_n, b.hs_n.flatten(1), b.ha_n.flatten(1), b.mask]
        if self.use_action:
            parts.append(b.a_n)
        return torch.cat(parts, dim=1)

    def forward(self, b: Batch) -> torch.Tensor:
        return self.out(nn.functional.silu(self.blocks(self.inp(self.inputs(b)))))


class Blind(MLP):
    """As mlp with a_t removed from the input (DESIGN.md, Models). The history keeps its actions."""

    arch = "blind"

    def __init__(self, norm, dscaler, hidden: int) -> None:
        super().__init__(norm, dscaler, hidden, use_action=False)


class MLPShortcut(MLP):
    arch = "mlp-shortcut"
