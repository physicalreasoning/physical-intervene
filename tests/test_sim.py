"""Simulator wrapper, geometry, scene and constants agreement."""

from __future__ import annotations

import re

import numpy as np
import pytest

from physical_intervene import constants as c
from physical_intervene.sim import SCENE_PATH, Sim, box_target, gap, wrap_angle

R = c.PUSHER_RADIUS_M
H = c.BOX_SIDE_M / 2


@pytest.fixture(scope="module")
def sim() -> Sim:
    return Sim()


def test_save_step_restore_step_is_byte_identical(sim: Sim) -> None:
    sim.reset(np.array([0.0, 0.0]), 0.3, np.array([-0.07, 0.01]))
    sim.step(np.array([0.1, 0.0]))  # non-trivial warm start and velocities
    saved = sim.get_state()
    action = np.array([0.12, -0.03])
    sim.step(action)
    first = sim.get_state()
    sim.step(np.array([-0.1, 0.1]))  # move on, so the restore has something to undo
    sim.set_state(saved)
    sim.step(action)
    second = sim.get_state()
    assert np.array_equal(first.view(np.uint8), second.view(np.uint8))


def test_contact_flag(sim: Sim) -> None:
    sim.reset(np.zeros(2), 0.0, np.array([-(H + R + 0.002), 0.0]))
    assert sim.step(np.array([0.1, 0.0]))
    sim.reset(np.zeros(2), 0.0, np.array([-(H + R + 0.002), 0.0]))
    assert not sim.step(np.array([-0.1, 0.0]))


def test_gap_touching_and_one_cm() -> None:
    assert gap(np.zeros(2), 0.0, np.array([H + R, 0.0])) == pytest.approx(0.0, abs=1e-12)
    assert gap(np.zeros(2), 0.0, np.array([0.0, -(H + R + 0.01)])) == pytest.approx(0.01)
    assert gap(np.zeros(2), 0.0, np.array([H + R - 0.002, 0.0])) == pytest.approx(-0.002)


def test_gap_at_a_corner() -> None:
    p = np.array([H + 0.03, H + 0.04])
    assert gap(np.zeros(2), 0.0, p) == pytest.approx(0.05 - R)


def test_gap_rotated_box() -> None:
    # Rotated 45 degrees: a corner points along +x at distance H * sqrt(2).
    p = np.array([H * np.sqrt(2) + 0.02, 0.0])
    assert gap(np.zeros(2), np.pi / 4, p) == pytest.approx(0.02 - R)
    # Box displaced and rotated 90 degrees is the same square.
    b = np.array([0.1, -0.2])
    assert gap(b, np.pi / 2, b + np.array([0.0, H + R + 0.005])) == pytest.approx(0.005)


def test_wrap_angle_across_pi() -> None:
    assert wrap_angle(np.pi) == pytest.approx(np.pi)
    assert wrap_angle(-np.pi) == pytest.approx(np.pi)
    assert wrap_angle(3 * np.pi / 2) == pytest.approx(-np.pi / 2)
    before = np.array([0.0, 0.0, np.pi - 0.01])
    after = np.array([0.001, 0.0, -np.pi + 0.01])
    assert box_target(before, after) == pytest.approx([0.001, 0.0, 0.02])
    assert box_target(after, before)[2] == pytest.approx(-0.02)


def test_scene_agrees_with_constants(sim: Sim) -> None:
    m = sim.model
    assert m.opt.timestep == c.PHYSICS_STEP_S
    assert m.body("box").mass[0] == pytest.approx(c.BOX_MASS_KG)
    assert m.body("pusher").mass[0] == pytest.approx(c.PUSHER_MASS_KG)
    assert m.geom("box").size[:2] == pytest.approx([H, H])
    assert 2 * m.geom("box").size[2] == pytest.approx(c.BOX_HEIGHT_M)
    assert m.geom("pusher").size[0] == pytest.approx(R)
    assert 2 * (m.geom("pusher").size[0] + m.geom("pusher").size[1]) == pytest.approx(c.PUSHER_HEIGHT_M)
    # The capsule's straight section must lie within the box's height, so contact is planar.
    pusher_z, box_z = m.body("pusher").pos[2], m.body("box").pos[2]
    assert abs(pusher_z - box_z) + m.geom("pusher").size[1] <= m.geom("box").size[2]
    assert m.dof_frictionloss[:2] == pytest.approx([c.BOX_SLIDE_FRICTIONLOSS_N] * 2)
    assert m.dof_frictionloss[2] == pytest.approx(c.BOX_HINGE_FRICTIONLOSS_NM)
    assert m.geom("box").friction[0] == pytest.approx(c.PUSHER_BOX_FRICTION)
    assert m.geom("pusher").friction[0] == pytest.approx(c.PUSHER_BOX_FRICTION)
    assert m.actuator_gainprm[:, 0] == pytest.approx([c.ACTUATOR_KV] * 2)
    assert m.actuator_forcerange[:, 1] == pytest.approx([c.ACTUATOR_FORCE_LIMIT_N] * 2)
    assert m.actuator_ctrlrange[:, 1] == pytest.approx([c.ACTION_MAX_SPEED_M_S] * 2)
    assert chr(0x2014) not in SCENE_PATH.read_text()  # no em dashes in the scene file either
    assert not re.search(r"frictionloss=\"0\"", SCENE_PATH.read_text())


def test_analytic_push_force_margin() -> None:
    f_resist = np.sqrt(2) * c.BOX_SLIDE_FRICTIONLOSS_N
    f_slow = min(c.ACTUATOR_KV * c.AIMED_SPEED_RANGE_M_S[0], c.ACTUATOR_FORCE_LIMIT_N)
    assert f_slow >= c.SMOKE_MIN_PUSH_FORCE_MARGIN * f_resist


def test_simulator_contact_matches_gap(sim: Sim) -> None:
    """Every shallow penetration is detected, at the depth the gap function gives (catches a faulty
    collider on the platform the tests run on)."""
    import mujoco

    rng = np.random.default_rng(0)
    n_checked = 0
    for _ in range(2000):
        theta, angle = rng.uniform(-np.pi, np.pi, size=2)
        r = rng.uniform(H, H * np.sqrt(2) + R)
        p = r * np.array([np.cos(angle), np.sin(angle)])
        g = float(gap(np.zeros(2), theta, p))
        if not -0.003 < g < -1e-5:
            continue
        sim.reset(np.zeros(2), theta, p)
        mujoco.mj_forward(sim.model, sim.data)
        assert sim.data.ncon >= 1
        assert min(sim.data.contact[i].dist for i in range(sim.data.ncon)) == pytest.approx(g, abs=1e-9)
        n_checked += 1
    assert n_checked > 100
