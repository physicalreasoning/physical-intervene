"""Test-like states, the 24 interventions and their labels (DESIGN.md, Interventions).

Labels use only the simulator's own outcome. The diagnostic label (`label_vs_reference`) compares
the box's outcome under the action with its outcome over the same step with contacts disabled,
so a box that is already sliding is not counted as pushed; it is evidence for an open design
question and is not used for anything else.
"""

from __future__ import annotations

import numpy as np

from physical_intervene import constants as c
from physical_intervene import policy
from physical_intervene.episodes import EpisodeRunner, rng
from physical_intervene.sim import Sim, box_target, wrap_angle

LABELS = ("no_contact", "graze", "push")
KIND_LOGGED, KIND_AIMED, KIND_UNIFORM = 0, 1, 2
# Action index 0 is the logged action, 1..12 aimed, 13..24 uniform.
KINDS = np.array(
    [KIND_LOGGED] + [KIND_AIMED] * c.N_AIMED_INTERVENTIONS + [KIND_UNIFORM] * c.N_UNIFORM_INTERVENTIONS
)


def push_thresholds(
    d_push: float = c.PUSH_MIN_DISPLACEMENT_M, theta_push_deg: float = c.PUSH_MIN_ROTATION_DEG
):
    return d_push, np.deg2rad(theta_push_deg)


def label(
    contact: np.ndarray,
    target: np.ndarray,
    d_push: float = c.PUSH_MIN_DISPLACEMENT_M,
    theta_push_deg: float = c.PUSH_MIN_ROTATION_DEG,
) -> np.ndarray:
    """0 no_contact, 1 graze, 2 push, from the contact flag and the box target (vectorised)."""
    d, th = push_thresholds(d_push, theta_push_deg)
    target = np.asarray(target, float)
    moved = (np.hypot(target[..., 0], target[..., 1]) >= d) | (np.abs(target[..., 2]) >= th)
    return np.where(np.asarray(contact, bool), np.where(moved, 2, 1), 0)


def label_vs_reference(contact: np.ndarray, target: np.ndarray, reference: np.ndarray, **kw) -> np.ndarray:
    """Diagnostic label: as `label`, but on the difference from the no-contact reference outcome."""
    diff = np.asarray(target, float) - np.asarray(reference, float)
    diff[..., 2] = wrap_angle(diff[..., 2])
    return label(contact, diff, **kw)


def draw_interventions(seed: int, obs: np.ndarray) -> np.ndarray:
    """12 aimed then 12 uniform actions, from the episode's own intervention stream."""
    g = rng(seed, "interventions")
    a = c.ACTION_MAX_SPEED_M_S
    speed = g.uniform(*c.AIMED_SPEED_RANGE_M_S, size=c.N_AIMED_INTERVENTIONS)
    offset = g.uniform(
        -np.deg2rad(c.AIMED_MAX_ANGLE_DEG), np.deg2rad(c.AIMED_MAX_ANGLE_DEG), size=c.N_AIMED_INTERVENTIONS
    )
    to_box = obs[0:2] - obs[7:9]
    angle = np.arctan2(to_box[1], to_box[0]) + offset
    aimed = speed[:, None] * np.stack([np.cos(angle), np.sin(angle)], axis=1)
    uniform = g.uniform(-a, a, size=(c.N_UNIFORM_INTERVENTIONS, 2))
    return np.concatenate([aimed, uniform], axis=0)


def select_state(sim: Sim, seed: int, epsilon_near: float = policy.EPSILON_NEAR) -> dict | None:
    """The test-like state of a fresh episode, or None if it has none (the episode is skipped).

    The first step at or after a step drawn uniformly from [10, 90] whose gap is at most 10 mm.
    """
    lo, hi = c.TEST_STATE_STEP_RANGE
    start = int(rng(seed, "select").integers(lo, hi + 1))
    runner = EpisodeRunner(sim, seed, shadow=False, epsilon_near=epsilon_near)
    for t in range(c.EPISODE_CONTROL_STEPS):
        row = runner.policy_step()
        if t >= start and row["gap"] <= c.TEST_STATE_MAX_GAP_M:
            return {
                "seed": seed,
                "step": t,
                "start_step": start,
                "obs": row["obs"],
                "goal": row["goal"],
                "state": row["state"],
                "logged_action": row["action"],
                "gap": row["gap"],
            }
        runner.env_step(row["action"])
    return None


def run_actions(sim: Sim, state: np.ndarray, actions: np.ndarray) -> dict:
    """Restore `state` and run one control step under each action; plus the no-contact reference."""
    contact, target, after = [], [], []
    for a in actions:
        sim.set_state(state)
        q0 = sim.qpos
        contact.append(sim.step(a))
        target.append(box_target(q0, sim.qpos))
        after.append(sim.get_state())
    sim.set_state(state)
    q0 = sim.qpos
    sim.step_without_contact(np.zeros(2))
    reference = box_target(q0, sim.qpos)
    return {
        "contact": np.array(contact),
        "target": np.array(target),
        "after": np.array(after),
        "reference": reference,
    }


def make_states(sim: Sim, seeds_from: int, n_states: int, epsilon_near: float = policy.EPSILON_NEAR) -> dict:
    """n test-like states from consecutive seeds starting at `seeds_from`, skipping episodes with none,
    each with its 25 actions (logged first) run and labelled."""
    rows, skipped, seed = [], [], seeds_from
    while len(rows) < n_states:
        s = select_state(sim, seed, epsilon_near)
        if s is None:
            skipped.append(seed)
        else:
            actions = np.concatenate([s["logged_action"][None], draw_interventions(seed, s["obs"])], axis=0)
            s.update(actions=actions, **run_actions(sim, s["state"], actions))
            s["log_density"] = np.array(
                [policy.log_density(a, s["obs"], s["goal"], epsilon_near) for a in actions]
            )
            rows.append(s)
        seed += 1
    out = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    out["skipped_seeds"] = np.array(skipped, dtype=np.int64)
    out["kinds"] = KINDS
    return out
