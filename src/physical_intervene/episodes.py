"""Behaviour episodes, split seed ranges and random streams.

Every episode has one integer seed. Each random stream an episode uses is
SeedSequence(seed, spawn_key=(k,)) for a fixed k per purpose, so the streams are independent of
each other, and adding or removing one (for example the shadow actions) changes no other draw.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from physical_intervene import constants as c
from physical_intervene import policy
from physical_intervene.sim import Sim, box_target, gap

# One spawn key per purpose. Never renumber: the data depend on these.
STREAM_KEYS = {
    "init": 0,  # box start pose, pusher start position, first goal
    "goal": 1,  # goals drawn after the box reaches one
    "eps": 2,  # behaviour policy: epsilon draws
    "noise": 3,  # behaviour policy: Gaussian noise
    "uniform": 4,  # behaviour policy: uniform actions
    "shadow_eps": 5,  # shadow actions, their own three streams
    "shadow_noise": 6,
    "shadow_uniform": 7,
    "select": 8,  # test-like state selection: the drawn start step
    "interventions": 9,  # the 24 intervention actions
    "feasibility_eps": 10,  # positive-control feasibility: policy draws for m
    "feasibility_noise": 11,
    "feasibility_uniform": 12,
}

SPLIT_RANGES = {
    "train": c.SEED_RANGE_TRAIN,
    "validation": c.SEED_RANGE_VALIDATION,
    "smoke": c.SEED_RANGE_SMOKE,
    "pilot": c.SEED_RANGE_PILOT,
}


def rng(seed: int, purpose: str) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(STREAM_KEYS[purpose],)))


def policy_streams(seed: int, prefix: str = "") -> policy.Streams:
    return policy.Streams(
        eps=rng(seed, prefix + "eps"),
        noise=rng(seed, prefix + "noise"),
        uniform=rng(seed, prefix + "uniform"),
    )


def smoke_seeds(block: str, n: int, attempt: int = c.SMOKE_ATTEMPT) -> np.ndarray:
    """Seeds for a block of the smoke split ("episodes", "pool" or "states")."""
    offset = {
        "episodes": c.SMOKE_EPISODES_OFFSET,
        "pool": c.SMOKE_POOL_OFFSET,
        "states": c.SMOKE_STATES_OFFSET,
    }[block]
    start = c.SEED_RANGE_SMOKE[0] + attempt * c.SMOKE_ATTEMPT_STRIDE + offset
    seeds = np.arange(start, start + n, dtype=np.int64)
    if seeds[-1] >= c.SEED_RANGE_SMOKE[1]:
        raise ValueError("smoke seeds out of range")
    return seeds


def draw_goal(box_xy: np.ndarray, g: np.random.Generator) -> np.ndarray:
    lo, hi = c.GOAL_XY_RANGE_M
    while True:
        goal = g.uniform(lo, hi, size=2)
        if np.hypot(*(goal - box_xy)) >= c.GOAL_MIN_DIST_M:
            return goal


def reset_episode(sim: Sim, seed: int) -> np.ndarray:
    """Draw the start pose and the first goal, reset the simulator; returns the goal."""
    g = rng(seed, "init")
    box_xy = g.uniform(*c.BOX_START_XY_RANGE_M, size=2)
    box_theta = float(np.pi - g.uniform(0.0, 2.0 * np.pi))  # uniform on (-pi, pi]
    while True:
        pusher_xy = g.uniform(*c.PUSHER_START_XY_RANGE_M, size=2)
        if gap(box_xy, box_theta, pusher_xy) >= c.PUSHER_START_MIN_DIST_M:
            break
    sim.reset(box_xy, box_theta, pusher_xy)
    return draw_goal(box_xy, g)


class EpisodeRunner:
    """Steps one behaviour episode; used for full episodes and for test-like state selection."""

    def __init__(self, sim: Sim, seed: int, shadow: bool, epsilon_near: float = policy.EPSILON_NEAR):
        self.sim, self.seed, self.shadow, self.epsilon_near = sim, seed, shadow, epsilon_near
        self.goal = reset_episode(sim, seed)
        self.goal_rng = rng(seed, "goal")
        self.streams = policy_streams(seed)
        self.shadow_streams = policy_streams(seed, "shadow_") if shadow else None
        self.t = 0

    def current_goal(self) -> np.ndarray:
        """The goal at the current step (a new one is drawn once the box reaches the old)."""
        box = self.sim.data.qpos[0:2]
        if np.hypot(*(self.goal - box)) < c.GOAL_REACHED_DIST_M:
            self.goal = draw_goal(box.copy(), self.goal_rng)
        return self.goal

    def policy_step(self) -> dict:
        """Observe, choose the action (and shadow action), but do not step the simulator."""
        obs = self.sim.obs()
        goal = self.current_goal().copy()
        action, unif = policy.sample(obs, goal, self.streams, self.epsilon_near)
        out = {
            "obs": obs,
            "goal": goal,
            "action": action,
            "action_uniform": unif,
            "state": self.sim.get_state(),
            "gap": self.sim.gap(),
        }
        if self.shadow_streams is not None:
            out["shadow"], out["shadow_uniform"] = policy.sample(
                obs, goal, self.shadow_streams, self.epsilon_near
            )
        return out

    def env_step(self, action: np.ndarray) -> tuple[bool, np.ndarray]:
        q0 = self.sim.qpos
        contact = self.sim.step(action)
        self.t += 1
        return contact, box_target(q0, self.sim.qpos)


def run_episode(sim: Sim, seed: int, shadow: bool = True, epsilon_near: float = policy.EPSILON_NEAR) -> dict:
    """A full behaviour episode of 100 control steps, as arrays (obs has the final step too)."""
    runner = EpisodeRunner(sim, seed, shadow, epsilon_near)
    rows = []
    for _ in range(c.EPISODE_CONTROL_STEPS):
        row = runner.policy_step()
        row["contact"], row["target"] = runner.env_step(row["action"])
        rows.append(row)
    out = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    out["obs"] = np.concatenate([out["obs"], sim.obs()[None]], axis=0)
    out["seed"] = np.int64(seed)
    return out


TRANSITION_KEYS = ("obs", "goal", "action", "action_uniform", "state", "gap", "contact", "target")


def generate(
    sim: Sim, seeds: np.ndarray, shadow: bool = True, epsilon_near: float = policy.EPSILON_NEAR
) -> dict:
    """Episodes stacked along a leading axis."""
    eps = [run_episode(sim, int(s), shadow, epsilon_near) for s in seeds]
    return {k: np.stack([e[k] for e in eps]) for k in eps[0]}


def save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)


def load(path: Path) -> dict:
    with np.load(path) as f:
        return {k: f[k] for k in f.files}
