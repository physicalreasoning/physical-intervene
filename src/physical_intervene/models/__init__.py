"""Learned models (DESIGN.md, Models) and their widths.

Each architecture has one free size per width, chosen so its parameter count (for contact_wm, its
correction network's) is the closest to WIDTH_PARAM_COUNTS. rssm and pnn are exploration only (not
in DESIGN.md); rssm at medium width is fixed at a 200-unit GRU with a 30-dimensional latent.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import torch

from physical_intervene import constants as c
from physical_intervene.data import InputNorm, TargetScaler
from physical_intervene.models.common import WIDTH_NAMES, Batch, WorldModel, make_batch
from physical_intervene.models.contact_wm import ContactWM
from physical_intervene.models.gru import GRUModel
from physical_intervene.models.mlp import MLP, Blind, MLPMultiStep, MLPShortcut
from physical_intervene.models.pnn import PNN
from physical_intervene.models.rssm import RSSM

__all__ = ["Batch", "WorldModel", "make_batch", "build", "size_for", "ARCHITECTURES", "WIDTH_NAMES"]

ARCHITECTURES = ("mlp", "gru", "mlp-ms", "contact_wm", "blind", "mlp-shortcut")
EXPLORATION_ARCHITECTURES = ("rssm", "pnn")  # Phase 0 candidates only, not in DESIGN.md
_CLASSES = {
    "mlp": MLP,
    "gru": GRUModel,
    "contact_wm": ContactWM,
    "blind": Blind,
    "mlp-shortcut": MLPShortcut,
    "pnn": PNN,
    "mlp-ms": MLPMultiStep,
}


def _dummy() -> tuple[InputNorm, TargetScaler]:
    z, o = np.zeros(11), np.ones(11)
    return InputNorm(z, o, np.zeros(2), np.ones(2)), TargetScaler(z, o)


def rssm_sizes(units: int) -> dict:
    return {
        "units": units,
        "latent": max(1, round(units * c.RSSM_LATENT_PER_UNIT)),
        "dec_hidden": max(1, round(units * c.RSSM_DECODER_HIDDEN_PER_UNIT)),
    }


def _construct(arch: str, size: int, norm: InputNorm, dscaler: TargetScaler) -> WorldModel:
    if arch == "rssm":
        return RSSM(norm, dscaler, **rssm_sizes(size))
    return _CLASSES[arch](norm, dscaler, hidden=size)


@cache
def count_for(arch: str, size: int) -> int:
    with torch.device("meta"):
        return _construct(arch, size, *_dummy()).n_params()


@cache
def size_for(arch: str, width: str) -> int:
    """The free size (hidden width, or GRU units for rssm) of an architecture at a width."""
    if arch == "rssm" and width == "medium":
        return c.RSSM_MEDIUM_GRU_UNITS
    target = c.WIDTH_PARAM_COUNTS[WIDTH_NAMES.index(width)]
    lo, hi = 1, 4096
    while hi - lo > 1:  # the count grows with the size
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if count_for(arch, mid) <= target else (lo, mid)
    return min((lo, hi), key=lambda s: abs(count_for(arch, s) - target))


def build(arch: str, width: str, norm: InputNorm, dscaler: TargetScaler) -> WorldModel:
    model = _construct(arch, size_for(arch, width), norm, dscaler)
    model.width = width
    return model


def param_table() -> dict:
    out = {}
    for arch in ARCHITECTURES:
        out[arch] = {}
        for w in WIDTH_NAMES:
            s = size_for(arch, w)
            out[arch][w] = {"size": s, "params": count_for(arch, s)}
            if arch == "rssm":
                out[arch][w].update(rssm_sizes(s))
    return out
