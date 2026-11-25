"""rssm (main, different objective): a recurrent state-space model trained on the ELBO.

Structure and objective follow PlaNet (Hafner et al. 2019): a deterministic GRU state h_k and a
Gaussian latent z_k, with
- transition h_k = GRU(h_{k-1}, [z_{k-1}, a_{k-1}]);
- prior p(z_k | h_k) and posterior q(z_k | h_k, s_k);
- observation model p(delta_s_{k-1} | h_k, z_k): a Gaussian with fixed standard deviation on the
  standardised state change into s_k. The state change, not the state, is decoded, because it is
  what the model must predict and it is about 30 times smaller than the state's own spread.
- loss per step: reconstruction plus KL weight x max(KL(q || p), free nats).

Filtering starts with h = 0, z = 0 and a zero previous action at the first real state; padded
history steps leave the state unchanged. A one-step prediction filters s_{t-8}, ..., s_t, then
takes one prior step with a_t and decodes; it is the mean over RSSM_PREDICTION_SAMPLES samples,
drawn from a generator seeded per run, so scoring the same checkpoint twice gives the same answer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from physical_intervene import constants as c
from physical_intervene.models.common import Batch, WorldModel, mlp_stack
from physical_intervene.sim import OBS_DIM


@dataclass
class Gaussian:
    mean: torch.Tensor
    std: torch.Tensor

    def sample(self, generator: torch.Generator | None) -> torch.Tensor:
        eps = torch.randn(self.mean.shape, generator=generator, dtype=self.mean.dtype)
        return self.mean + self.std * eps


def kl_divergence(q: Gaussian, p: Gaussian) -> torch.Tensor:
    """KL(q || p) of diagonal Gaussians, summed over the latent dimensions."""
    var_ratio = (q.std / p.std) ** 2
    t = ((q.mean - p.mean) / p.std) ** 2
    return 0.5 * (var_ratio + t - 1.0 - torch.log(var_ratio)).sum(-1)


@dataclass
class ElboTerms:
    recon: torch.Tensor  # mean over scored steps of the reconstruction negative log-likelihood
    kl: torch.Tensor  # mean over scored steps of KL(q || p), unclipped
    kl_clipped: torch.Tensor  # mean over scored steps of max(KL, free nats)
    loss: torch.Tensor  # recon + kl_weight * kl_clipped, the training loss
    recon_sq: torch.Tensor  # mean squared standardised error of the decoded mean (diagnostic)


class RSSM(WorldModel):
    arch = "rssm"
    stochastic = True

    def __init__(self, norm, dscaler, units: int, latent: int, dec_hidden: int) -> None:
        super().__init__(norm, dscaler)
        self.units, self.latent = units, latent
        self.cell = nn.GRUCell(latent + 2, units)
        self.prior_net = nn.Linear(units, 2 * latent)
        self.post_net = nn.Linear(units + OBS_DIM, 2 * latent)
        self.decoder = mlp_stack([units + latent, dec_hidden, OBS_DIM])
        self.free_nats = c.RSSM_FREE_NATS
        self.kl_weight = c.RSSM_KL_WEIGHT
        self.n_samples = c.RSSM_PREDICTION_SAMPLES
        self.sampling_seed = 0  # set per run by train.py; recorded in the checkpoint

    @staticmethod
    def _gaussian(out: torch.Tensor) -> Gaussian:
        mean, raw = out.chunk(2, dim=-1)
        return Gaussian(mean, nn.functional.softplus(raw) + c.RSSM_MIN_STD)

    def prior(self, h: torch.Tensor) -> Gaussian:
        return self._gaussian(self.prior_net(h))

    def posterior(self, h: torch.Tensor, s: torch.Tensor) -> Gaussian:
        return self._gaussian(self.post_net(torch.cat([h, s], dim=-1)))

    def decode(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([h, z], dim=-1))

    def filter(
        self, states: torch.Tensor, prev_a: torch.Tensor, mask: torch.Tensor, generator=None
    ) -> tuple[list, torch.Tensor, torch.Tensor]:
        """Run the posterior over a sequence. states (B, K, 11), prev_a (B, K, 2), mask (B, K).

        Returns per-step (h_k, z_k, prior_k, posterior_k) and the final (h, z)."""
        B, K, _ = states.shape
        h = states.new_zeros(B, self.units)
        z = states.new_zeros(B, self.latent)
        steps = []
        for k in range(K):
            h_new = self.cell(torch.cat([z, prev_a[:, k]], dim=-1), h)
            prior, post = self.prior(h_new), self.posterior(h_new, states[:, k])
            z_new = post.sample(generator)
            steps.append((h_new, z_new, prior, post))
            m = mask[:, k : k + 1]
            h = m * h_new + (1 - m) * h
            z = m * z_new + (1 - m) * z
        return steps, h, z

    def elbo(
        self,
        states: torch.Tensor,
        prev_a: torch.Tensor,
        mask: torch.Tensor,
        target_z: torch.Tensor,
        scored: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> ElboTerms:
        """ELBO terms on a window. target_z[:, k] is the standardised change into state k, scored
        where scored[:, k] is 1 (never at the first step, which has no previous state)."""
        steps, _, _ = self.filter(states, prev_a, mask, generator)
        recon, kl, sq = [], [], []
        for k, (h, z, prior, post) in enumerate(steps):
            err = (self.decode(h, z) - target_z[:, k]) / c.RSSM_OBS_STD
            recon.append(0.5 * (err * err).sum(-1))
            sq.append((err * err).mean(-1))
            kl.append(kl_divergence(post, prior))
        recon_t, kl_t, sq_t = torch.stack(recon, 1), torch.stack(kl, 1), torch.stack(sq, 1)
        w = scored / scored.sum()
        kl_clip = torch.clamp(kl_t, min=self.free_nats)
        r, k_, kc = (recon_t * w).sum(), (kl_t * w).sum(), (kl_clip * w).sum()
        return ElboTerms(
            recon=r, kl=k_, kl_clipped=kc, loss=r + self.kl_weight * kc, recon_sq=(sq_t * w).sum()
        )

    @staticmethod
    def window_inputs(b: Batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The history and s_t of a one-step batch as a filtering window."""
        B = len(b)
        states = torch.cat([b.hs_n, b.s_n[:, None]], dim=1)
        prev_a = torch.cat([b.ha_n.new_zeros(B, 1, 2), b.ha_n], dim=1)
        mask = torch.cat([b.mask, b.mask.new_ones(B, 1)], dim=1)
        return states, prev_a, mask

    def sample_prediction(self, b: Batch, generator: torch.Generator | None) -> torch.Tensor:
        """One sampled one-step prediction per row of b."""
        _, h, z = self.filter(*self.window_inputs(b), generator=generator)
        h1 = self.cell(torch.cat([z, b.a_n], dim=-1), h)
        return self.decode(h1, self.prior(h1).sample(generator))

    def forward(self, b: Batch) -> torch.Tensor:
        return self.predict_z(b)

    def predict_z(self, b: Batch, generator: torch.Generator | None = None) -> torch.Tensor:
        if generator is None:
            generator = torch.Generator().manual_seed(self.sampling_seed)
        n = self.n_samples
        out = self.sample_prediction(b.repeat(n), generator)
        return out.view(len(b), n, -1).mean(1)
