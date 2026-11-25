"""Baselines (DESIGN.md, Models): copy, constvel, knn. Each returns the predicted state change in
physical units for a set of examples; the box target follows from it as for every model."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from physical_intervene import constants as c
from physical_intervene.data import Examples


def copy(ex: Examples) -> np.ndarray:
    """The box (and the pusher) do not move."""
    return np.zeros_like(ex.s)


def constvel(ex: Examples) -> np.ndarray:
    """Box and pusher keep their current velocities over the step."""
    s, dt = ex.s, c.CONTROL_STEP_S
    d = np.zeros_like(s)
    th = np.arctan2(s[:, 3], s[:, 2])
    th1 = th + s[:, 6] * dt
    d[:, 0:2] = s[:, 4:6] * dt
    d[:, 2] = np.cos(th1) - s[:, 2]
    d[:, 3] = np.sin(th1) - s[:, 3]
    d[:, 7:9] = s[:, 9:11] * dt
    return d


class KNN:
    """Mean state change of the k nearest training transitions in (s_t, a_t), each dimension
    standardised on the training split."""

    def __init__(self, train: Examples, k: int = c.KNN_NEIGHBOURS) -> None:
        x = np.concatenate([train.s, train.a], axis=1)
        self.mean, self.std = x.mean(0), x.std(0)
        self.tree = cKDTree((x - self.mean) / self.std)
        self.delta, self.k = train.delta, k

    def __call__(self, ex: Examples) -> np.ndarray:
        x = (np.concatenate([ex.s, ex.a], axis=1) - self.mean) / self.std
        _, idx = self.tree.query(x, k=self.k)
        return self.delta[idx].mean(axis=1)
