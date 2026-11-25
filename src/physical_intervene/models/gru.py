"""gru (main): a GRU over the history and the current state, then a head on (h_t, a_t).

The GRU reads states s_{t-8}, ..., s_t, each with the action that led into it (a_{k-1}; zero for
the oldest, whose previous action is not in the history) and the mask (1 for s_t).
"""

from __future__ import annotations

import torch
from torch import nn

from physical_intervene import constants as c
from physical_intervene.models.common import Batch, WorldModel, mlp_stack
from physical_intervene.sim import OBS_DIM


def sequence_inputs(b: Batch) -> torch.Tensor:
    """(B, L + 1, 11 + 2 + 1): states oldest first, the action into each state, the mask."""
    B = len(b)
    states = torch.cat([b.hs_n, b.s_n[:, None]], dim=1)
    prev_a = torch.cat([b.ha_n.new_zeros(B, 1, 2), b.ha_n], dim=1)
    mask = torch.cat([b.mask, b.mask.new_ones(B, 1)], dim=1)[..., None]
    return torch.cat([states, prev_a, mask], dim=2)


class GRUModel(WorldModel):
    arch = "gru"

    def __init__(self, norm, dscaler, hidden: int) -> None:
        super().__init__(norm, dscaler)
        self.gru = nn.GRU(OBS_DIM + 3, hidden, batch_first=True)
        self.head = mlp_stack([hidden + 2] + [hidden] * c.GRU_HEAD_HIDDEN_LAYERS + [OBS_DIM])

    def forward(self, b: Batch) -> torch.Tensor:
        _, h = self.gru(sequence_inputs(b))
        return self.head(torch.cat([h[-1], b.a_n], dim=1))
