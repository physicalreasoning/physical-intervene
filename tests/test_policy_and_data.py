"""Behaviour policy, seed streams, shadow actions and push labels."""

from __future__ import annotations

import numpy as np
import pytest

from physical_intervene import constants as c
from physical_intervene import episodes as E
from physical_intervene import policy
from physical_intervene.interventions import label, label_total_motion
from physical_intervene.sim import Sim

A = c.ACTION_MAX_SPEED_M_S
H = c.BOX_SIDE_M / 2
GOAL = np.array([0.15, 0.05])


def _obs(box_xy, theta, pusher_xy) -> np.ndarray:
    return np.array(
        [box_xy[0], box_xy[1], np.cos(theta), np.sin(theta), 0, 0, 0, pusher_xy[0], pusher_xy[1], 0, 0]
    )


# Far from the box (orbit), and behind it in the push corridor (push mode, near-box epsilon).
STATES = [
    _obs([0.0, 0.0], 0.2, [0.12, 0.1]),
    _obs([0.0, 0.0], 0.0, [-(H + c.PUSHER_RADIUS_M + 0.005), -0.002]),
]


@pytest.mark.parametrize("obs", STATES)
def test_log_density_integrates_to_one(obs: np.ndarray) -> None:
    n = 400
    grid = np.linspace(-A, A, n + 1)
    mid = 0.5 * (grid[1:] + grid[:-1])
    cell = (2 * A / n) ** 2
    total = sum(np.exp(policy.log_density(np.array([x, y]), obs, GOAL)) for x in mid for y in mid) * cell
    assert total == pytest.approx(1.0, abs=5e-3)


@pytest.mark.parametrize("obs", STATES)
def test_log_density_matches_sample_frequencies(obs: np.ndarray) -> None:
    n_draws, n_bins = 40_000, 6
    streams = policy.Streams(*(np.random.default_rng(s) for s in (1, 2, 3)))
    draws = np.array([policy.sample(obs, GOAL, streams)[0] for _ in range(n_draws)])
    edges = np.linspace(-A, A, n_bins + 1)
    counts, _, _ = np.histogram2d(draws[:, 0], draws[:, 1], bins=[edges, edges])
    fine = 30  # integrate the density over each coarse cell on a finer grid
    expected = np.zeros((n_bins, n_bins))
    for i in range(n_bins):
        for j in range(n_bins):
            xs = np.linspace(edges[i], edges[i + 1], fine + 1)
            ys = np.linspace(edges[j], edges[j + 1], fine + 1)
            xm, ym = 0.5 * (xs[1:] + xs[:-1]), 0.5 * (ys[1:] + ys[:-1])
            dens = [np.exp(policy.log_density(np.array([x, y]), obs, GOAL)) for x in xm for y in ym]
            expected[i, j] = np.mean(dens) * (edges[1] - edges[0]) ** 2
    assert expected.sum() == pytest.approx(1.0, abs=1e-2)
    freq = counts / n_draws
    # Binomial standard error, plus a little for the fine-grid integration.
    tol = 4 * np.sqrt(expected * (1 - expected) / n_draws) + 2e-3
    assert np.all(np.abs(freq - expected) <= tol)


def test_epsilon_near_box() -> None:
    assert policy.epsilon(STATES[1], epsilon_near=0.05) == 0.05
    assert policy.epsilon(STATES[0], epsilon_near=0.05) == c.POLICY_EPSILON


def test_shadow_stream_does_not_change_transitions() -> None:
    sim = Sim()
    seeds = np.array([c.SEED_RANGE_SMOKE[0] + 999_999 - k for k in range(2)])  # top of the smoke range
    with_shadow = E.generate(sim, seeds, shadow=True)
    without = E.generate(sim, seeds, shadow=False)
    for k in E.TRANSITION_KEYS:
        assert np.array_equal(
            np.ascontiguousarray(with_shadow[k]).view(np.uint8),
            np.ascontiguousarray(without[k]).view(np.uint8),
        ), k
    assert "shadow" in with_shadow and "shadow" not in without
    assert not np.array_equal(with_shadow["shadow"], with_shadow["action"])


def test_seed_ranges_are_disjoint() -> None:
    ranges = sorted(E.SPLIT_RANGES.values())
    for (a0, a1), (b0, b1) in zip(ranges, ranges[1:], strict=False):
        assert a0 < a1 <= b0 < b1
    assert set(E.SPLIT_RANGES) == {"train", "validation", "smoke", "pilot"}


def test_smoke_blocks_stay_in_smoke_range() -> None:
    lo, hi = c.SEED_RANGE_SMOKE
    blocks = {
        "episodes": c.N_SMOKE_EPISODES,
        "pool": c.N_SMOKE_COVERAGE_POOL_EPISODES,
        "states": 10 * c.N_SMOKE_STATES,
    }
    used = []
    for attempt in range(len(c.POLICY_EPSILON_NEAR_LADDER)):
        for block, n in blocks.items():
            s = E.smoke_seeds(block, n, attempt)
            assert lo <= s[0] and s[-1] < hi
            used.append(set(s.tolist()))
    for i, a in enumerate(used):
        for b in used[i + 1 :]:
            assert not a & b


def test_stream_keys_are_unique() -> None:
    assert len(set(E.STREAM_KEYS.values())) == len(E.STREAM_KEYS)


def test_push_label_on_hand_built_push_and_graze() -> None:
    sim = Sim()
    at_rest = np.zeros(3)  # no-contact reference of a box at rest: it does not move
    # Push: pusher 1 mm behind the middle of the -x face, driven into it at 0.1 m/s.
    sim.reset(np.zeros(2), 0.0, np.array([-(H + c.PUSHER_RADIUS_M + 0.001), 0.0]))
    q0 = sim.qpos
    contact = sim.step(np.array([0.10, 0.0]))
    push_target = np.array([*(sim.qpos[:2] - q0[:2]), sim.qpos[2] - q0[2]])
    assert contact and label(contact, push_target, at_rest) == 2
    # Graze: pusher just touching the face, sliding along it.
    sim.reset(np.zeros(2), 0.0, np.array([-(H + c.PUSHER_RADIUS_M) + 0.0002, -0.03]))
    q0 = sim.qpos
    contact = sim.step(np.array([0.0, 0.05]))
    graze_target = np.array([*(sim.qpos[:2] - q0[:2]), sim.qpos[2] - q0[2]])
    assert contact and label(contact, graze_target, at_rest) == 1
    assert label(False, push_target, at_rest) == 0
    # A box already sliding 4 mm in the step, touched lightly: a graze against its reference,
    # though its total motion (the earlier rule) is over d_push.
    sliding = np.array([0.004, 0.0, 0.0])
    touched = sliding + np.array([0.0002, 0.0, 0.0])
    assert label(True, touched, sliding) == 1
    assert label_total_motion(True, touched) == 2
    # The same box pushed hard enough to change its motion by more than d_push: a push.
    assert label(True, sliding + np.array([0.003, 0.0, 0.0]), sliding) == 2
