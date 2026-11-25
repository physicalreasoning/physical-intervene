"""Metrics and matching rules (DESIGN.md, Target and loss; Matching observational loss).

std-MSE is TargetScaler.std_mse (data.py); L_obs is the std-MSE of the box target over the
validation transitions. Matching uses validation loss only.
"""

from __future__ import annotations

import numpy as np

from physical_intervene import constants as c
from physical_intervene.data import theta

_CORNERS = 0.5 * c.BOX_SIDE_M * np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=float)


def corners_after(s_t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """World positions of the box's 4 corners after a step with box target y: (..., 4, 2)."""
    th = theta(s_t) + y[..., 2]
    ct, st = np.cos(th)[..., None], np.sin(th)[..., None]
    centre = s_t[..., 0:2] + y[..., 0:2]
    x = ct * _CORNERS[:, 0] - st * _CORNERS[:, 1]
    yy = st * _CORNERS[:, 0] + ct * _CORNERS[:, 1]
    return centre[..., None, :] + np.stack([x, yy], -1)


def corner_rmse_mm(s_t: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """RMS over transitions of the mean distance between predicted and true corners, in mm."""
    d = np.linalg.norm(corners_after(s_t, y_pred) - corners_after(s_t, y_true), axis=-1).mean(-1)
    return float(1e3 * np.sqrt(np.mean(d * d)))


def best(curve: list[float]) -> tuple[int, float]:
    i = int(np.argmin(curve))
    return i, float(curve[i])


def v_star(best_by_arch: dict[str, list[float]], factor: float = c.MATCH_TARGET_FACTOR) -> float:
    """V* = factor x the largest per-architecture median of best L_obs over seeds."""
    return factor * max(float(np.median(v)) for v in best_by_arch.values())


def matched_checkpoint(losses: list[float], target: float) -> int:
    """Index of the checkpoint whose L_obs is closest to the target (first one on a tie)."""
    return int(np.argmin(np.abs(np.asarray(losses) - target)))


def excluded(losses: list[float], target: float, tol: float = c.MATCHED_LOSS_REL_TOL) -> bool:
    """A run is excluded if its L_obs never comes within tol (relative) of the target."""
    return not bool(np.any(np.abs(np.asarray(losses) - target) <= tol * target))


def passes_match_screen(l_a: float, l_b: float, tol: float = c.MATCHED_LOSS_REL_TOL) -> bool:
    return abs(l_a - l_b) / min(l_a, l_b) <= tol


def plateau(updates: list[int], losses: list[float]) -> dict:
    """The plateau criterion (DESIGN.md, Models, Training) at the last checkpoint and the first
    checkpoint where it holds.

    At checkpoint u (a multiple of CHECKPOINT_EVERY_UPDATES with u - 10 x 1,000 also a checkpoint),
    the criterion holds if the best L_obs so far improved by no more than 1% over the last 10
    checkpoints at 1,000-update spacing: (best(u - 10,000) - best(u)) / best(u - 10,000) <= 0.01."""
    span = c.PLATEAU_CHECKPOINTS * c.CHECKPOINT_EVERY_UPDATES
    best_so_far = np.minimum.accumulate(np.asarray(losses, float))
    at = dict(zip(updates, best_so_far, strict=True))
    rows = []
    for u in updates:
        if u % c.CHECKPOINT_EVERY_UPDATES == 0 and (u - span) in at:
            rel = (at[u - span] - at[u]) / at[u - span]
            rows.append((u, float(rel), bool(rel <= c.PLATEAU_MIN_REL_IMPROVEMENT)))
    first = next((u for u, _, ok in rows if ok), None)
    return {
        "evaluable": bool(rows),
        "met_at_last_checkpoint": rows[-1][2] if rows and rows[-1][0] == updates[-1] else None,
        "last_rel_improvement": rows[-1][1] if rows else None,
        "first_update_met": first,
        "budget_rule_if_pilot": None if first is None else int(np.ceil(first * c.BUDGET_FACTOR_OVER_PLATEAU)),
    }
