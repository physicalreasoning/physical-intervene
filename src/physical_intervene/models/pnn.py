"""pnn (exploration, not in DESIGN.md): mlp's backbone trained on a Gaussian negative log-likelihood.

The probabilistic network of PETS (Chua et al. 2018) without the ensemble: the residual MLP outputs
a mean and a diagonal log-variance for the standardised state change, and is trained on the
Gaussian NLL instead of the shared one-step loss. The log-variance is softly bounded by learned
limits, as in PETS, with a small penalty that keeps the limits tight. The prediction is the mean,
so scoring is deterministic.
"""

from __future__ import annotations

import torch
from torch import nn

from physical_intervene import constants as c
from physical_intervene.models.common import Batch
from physical_intervene.models.mlp import MLP
from physical_intervene.sim import OBS_DIM


class PNN(MLP):
    arch = "pnn"

    def __init__(self, norm, dscaler, hidden: int) -> None:
        super().__init__(norm, dscaler, hidden)
        self.out = nn.Linear(hidden, 2 * OBS_DIM)
        self.max_logvar = nn.Parameter(torch.full((OBS_DIM,), c.PNN_INIT_MAX_LOGVAR))
        self.min_logvar = nn.Parameter(torch.full((OBS_DIM,), c.PNN_INIT_MIN_LOGVAR))

    def mean_logvar(self, b: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        out = super().forward(b)
        mean, raw = out[:, :OBS_DIM], out[:, OBS_DIM:]
        logvar = self.max_logvar - nn.functional.softplus(self.max_logvar - raw)
        logvar = self.min_logvar + nn.functional.softplus(logvar - self.min_logvar)
        return mean, logvar

    def forward(self, b: Batch) -> torch.Tensor:
        return self.mean_logvar(b)[0]

    def nll(self, b: Batch, z: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Gaussian NLL per dimension (constant dropped), weighted by the shared loss's
        per-dimension weights w, plus the PETS penalty on the log-variance limits."""
        mean, logvar = self.mean_logvar(b)
        per_dim = 0.5 * (logvar + (mean - z) ** 2 * torch.exp(-logvar))
        penalty = c.PNN_LOGVAR_LIMIT_PENALTY * (self.max_logvar.sum() - self.min_logvar.sum())
        return (per_dim * w).sum(-1).mean() + penalty
