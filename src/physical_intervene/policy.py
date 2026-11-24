"""Scripted behaviour policy (DESIGN.md, Data).

The scripted part moves the pusher round the box to the side opposite the goal, then pushes the
box toward the goal at 0.1 m/s. Its action is that mean plus Gaussian noise of 0.02 m/s per axis,
truncated to the action box (drawn by inverse CDF, so the density is the renormalised Gaussian).
With probability epsilon the action is instead uniform on the action box; epsilon is
epsilon_near where the gap is at most 15 mm. The mean is a function of the observation and the
goal only (no internal phase), so the policy's density at a state is well defined.

Each draw consumes exactly one number from the epsilon stream and two each from the noise and
uniform streams, whichever branch is taken, so the streams stay aligned with the steps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr, ndtri

from physical_intervene import constants as c
from physical_intervene.sim import BOX_HALF_M, obs_gap

A_MAX = c.ACTION_MAX_SPEED_M_S
SIGMA = c.POLICY_NOISE_STD_M_S
LOG_UNIFORM_DENSITY = -2.0 * np.log(2.0 * A_MAX)
EPSILON_NEAR = c.POLICY_EPSILON_NEAR
ORBIT_RADIUS_M = BOX_HALF_M * np.sqrt(2.0) + c.PUSHER_RADIUS_M + c.POLICY_ORBIT_CLEARANCE_M


@dataclass(frozen=True)
class Streams:
    """The three random streams one sequence of policy draws uses."""

    eps: np.random.Generator
    noise: np.random.Generator
    uniform: np.random.Generator


def epsilon(obs: np.ndarray, epsilon_near: float = EPSILON_NEAR) -> float:
    return epsilon_near if float(obs_gap(obs)) <= c.POLICY_NEAR_BOX_GAP_M else c.POLICY_EPSILON


def _unit(angle: float) -> np.ndarray:
    return np.array([np.cos(angle), np.sin(angle)])


def face_distance(box_theta: float, direction: np.ndarray) -> float:
    """Distance from the box centre to its boundary along a unit direction."""
    local = np.array(
        [
            np.cos(box_theta) * direction[0] + np.sin(box_theta) * direction[1],
            -np.sin(box_theta) * direction[0] + np.cos(box_theta) * direction[1],
        ]
    )
    return BOX_HALF_M / max(abs(local[0]), abs(local[1]))


def _toward(target: np.ndarray, pusher: np.ndarray) -> np.ndarray:
    v = (target - pusher) * c.POLICY_APPROACH_GAIN / c.CONTROL_STEP_S
    speed = np.hypot(v[0], v[1])
    if speed > c.POLICY_APPROACH_MAX_SPEED_M_S:
        v *= c.POLICY_APPROACH_MAX_SPEED_M_S / speed
    return v


def scripted_mode_and_mean(obs: np.ndarray, goal: np.ndarray) -> tuple[str, np.ndarray]:
    """The scripted controller: ("push" | "approach" | "orbit", mean action), before noise."""
    box, pusher = obs[0:2], obs[7:9]
    theta = float(np.arctan2(obs[3], obs[2]))
    to_goal = np.asarray(goal, float) - box
    dist = np.hypot(to_goal[0], to_goal[1])
    u = to_goal / dist if dist > 1e-9 else np.array([1.0, 0.0])
    u_perp = np.array([-u[1], u[0]])
    contact_dist = face_distance(theta, -u) + c.PUSHER_RADIUS_M  # centre to pusher centre in contact
    rel = pusher - box
    along, lateral = float(rel @ u), float(rel @ u_perp)

    in_corridor = (
        abs(lateral) <= c.POLICY_PUSH_LATERAL_TOL_M
        and -(contact_dist + c.POLICY_APPROACH_CLEARANCE_M + 0.01)
        <= along
        <= -contact_dist + c.POLICY_PUSH_BEHIND_TOL_M
    )
    if in_corridor:
        mean = c.POLICY_PUSH_SPEED_M_S * u - (lateral / c.POLICY_PUSH_LATERAL_TIME_S) * u_perp
        return "push", np.clip(mean, -A_MAX, A_MAX)

    alpha_p = float(np.arctan2(rel[1], rel[0]))
    alpha_t = float(np.arctan2(-u[1], -u[0]))
    d_alpha = float(np.pi - np.mod(np.pi - (alpha_t - alpha_p), 2 * np.pi))
    if abs(d_alpha) <= np.deg2rad(c.POLICY_ORBIT_SWITCH_DEG):
        approach = box - u * (contact_dist + c.POLICY_APPROACH_CLEARANCE_M)
        return "approach", np.clip(_toward(approach, pusher), -A_MAX, A_MAX)
    step = np.sign(d_alpha) * min(abs(d_alpha), np.deg2rad(c.POLICY_ORBIT_STEP_DEG))
    waypoint = box + ORBIT_RADIUS_M * _unit(alpha_p + step)
    return "orbit", np.clip(_toward(waypoint, pusher), -A_MAX, A_MAX)


def scripted_mean(obs: np.ndarray, goal: np.ndarray) -> np.ndarray:
    return scripted_mode_and_mean(obs, goal)[1]


def _trunc_bounds(mean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (-A_MAX - mean) / SIGMA, (A_MAX - mean) / SIGMA


def sample(
    obs: np.ndarray, goal: np.ndarray, streams: Streams, epsilon_near: float = EPSILON_NEAR
) -> tuple[np.ndarray, bool]:
    """One action from the behaviour policy; returns (action, drawn_from_uniform)."""
    u_eps = streams.eps.random()
    u_noise = streams.noise.random(2)
    u_unif = streams.uniform.random(2)
    if u_eps < epsilon(obs, epsilon_near):
        return -A_MAX + 2.0 * A_MAX * u_unif, True
    mean = scripted_mean(obs, goal)
    lo, hi = _trunc_bounds(mean)
    p_lo, p_hi = ndtr(lo), ndtr(hi)
    z = ndtri(p_lo + u_noise * (p_hi - p_lo))
    return np.clip(mean + SIGMA * z, -A_MAX, A_MAX), False


def log_density_scripted(action: np.ndarray, obs: np.ndarray, goal: np.ndarray) -> float:
    """Log-density of the truncated Gaussian part (-inf outside the action box)."""
    action = np.asarray(action, float)
    if np.any(np.abs(action) > A_MAX):
        return -np.inf
    mean = scripted_mean(obs, goal)
    lo, hi = _trunc_bounds(mean)
    z = (action - mean) / SIGMA
    log_pdf = -0.5 * z**2 - 0.5 * np.log(2 * np.pi) - np.log(SIGMA)
    log_mass = np.log(ndtr(hi) - ndtr(lo))
    return float(np.sum(log_pdf - log_mass))


def log_density(
    action: np.ndarray, obs: np.ndarray, goal: np.ndarray, epsilon_near: float = EPSILON_NEAR
) -> float:
    """Log-density of the mixture (1 - eps) * truncated Gaussian + eps * uniform, as `sample` draws."""
    action = np.asarray(action, float)
    if np.any(np.abs(action) > A_MAX):
        return -np.inf
    eps = epsilon(obs, epsilon_near)
    return float(
        np.logaddexp(
            np.log1p(-eps) + log_density_scripted(action, obs, goal), np.log(eps) + LOG_UNIFORM_DENSITY
        )
    )
