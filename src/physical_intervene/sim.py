"""MuJoCo planar-pushing simulator: load, exact save and restore, one control step, geometry.

The full state is saved and restored with mj_getState / mj_setState and mjSTATE_INTEGRATION
(DESIGN.md, Environment): restoring only qpos and qvel is not bit-exact, because the solver's
warm start is part of the state. The simulator has no randomness and runs single-threaded.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import mujoco
import numpy as np

from physical_intervene import constants as c

SCENE_PATH = Path(__file__).resolve().parent / "assets" / "pushing.xml"
STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION
BOX_HALF_M = c.BOX_SIDE_M / 2
OBS_DIM = 11
OBS_NAMES = (
    "box_x",
    "box_y",
    "box_cos",
    "box_sin",
    "box_vx",
    "box_vy",
    "box_w",
    "pusher_x",
    "pusher_y",
    "pusher_vx",
    "pusher_vy",
)
# qpos and qvel layout, checked when the model loads.
JOINTS = ("box_x", "box_y", "box_yaw", "pusher_x", "pusher_y")


def scene_sha256() -> str:
    return hashlib.sha256(SCENE_PATH.read_bytes()).hexdigest()


def wrap_angle(a: np.ndarray | float) -> np.ndarray | float:
    """Wrap to (-pi, pi]."""
    return np.pi - np.mod(np.pi - np.asarray(a, dtype=float), 2 * np.pi)


def gap(box_xy: np.ndarray, box_theta: np.ndarray, pusher_xy: np.ndarray) -> np.ndarray:
    """Shortest distance from the pusher's surface to the box's surface, in m (vectorised).

    Distance from the cylinder's centre to the rotated square, minus the pusher radius; 0 or
    negative when touching (inside the box the signed distance to the nearest face is used).
    """
    box_xy, pusher_xy = np.asarray(box_xy, float), np.asarray(pusher_xy, float)
    rel = pusher_xy - box_xy
    ct, st = np.cos(box_theta), np.sin(box_theta)
    lx = np.abs(ct * rel[..., 0] + st * rel[..., 1]) - BOX_HALF_M
    ly = np.abs(-st * rel[..., 0] + ct * rel[..., 1]) - BOX_HALF_M
    outside = np.hypot(np.maximum(lx, 0.0), np.maximum(ly, 0.0))
    inside = np.minimum(np.maximum(lx, ly), 0.0)
    return outside + inside - c.PUSHER_RADIUS_M


def obs_gap(obs: np.ndarray) -> np.ndarray:
    """Gap from an observation (or a batch of them)."""
    obs = np.asarray(obs, float)
    theta = np.arctan2(obs[..., 3], obs[..., 2])
    return gap(obs[..., 0:2], theta, obs[..., 7:9])


def box_target(qpos_before: np.ndarray, qpos_after: np.ndarray) -> np.ndarray:
    """(dx, dy, dtheta) of the box over one step, dtheta wrapped to (-pi, pi]."""
    d = np.asarray(qpos_after, float)[..., :3] - np.asarray(qpos_before, float)[..., :3]
    return np.concatenate([d[..., :2], np.asarray(wrap_angle(d[..., 2]))[..., None]], axis=-1)


class Sim:
    """One MuJoCo model and data pair. Not thread-safe; one per process."""

    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        self.data = mujoco.MjData(self.model)
        m = self.model
        names = tuple(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt))
        if names != JOINTS or list(m.jnt_qposadr) != list(range(5)) or list(m.jnt_dofadr) != list(range(5)):
            raise ValueError(f"unexpected joint layout {names}")
        collidable = [g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
        self.box_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "box")
        self.pusher_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "pusher")
        if sorted(collidable) != sorted([self.box_geom, self.pusher_geom]):
            raise ValueError("only the box and the pusher may collide")
        if abs(m.opt.timestep - c.PHYSICS_STEP_S) > 1e-15:
            raise ValueError("timestep differs from constants.py")
        self.state_size = mujoco.mj_stateSize(m, STATE_SPEC)

    # State.
    def get_state(self) -> np.ndarray:
        buf = np.empty(self.state_size, dtype=np.float64)
        mujoco.mj_getState(self.model, self.data, buf, STATE_SPEC)
        return buf

    def set_state(self, state: np.ndarray) -> None:
        state = np.ascontiguousarray(state, dtype=np.float64)
        if state.shape != (self.state_size,):
            raise ValueError(f"state has shape {state.shape}, expected ({self.state_size},)")
        mujoco.mj_setState(self.model, self.data, state, STATE_SPEC)

    def reset(self, box_xy: np.ndarray, box_theta: float, pusher_xy: np.ndarray) -> None:
        """Reset to a pose with everything at rest and no warm start."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = [box_xy[0], box_xy[1], box_theta, pusher_xy[0], pusher_xy[1]]

    # Readouts (from qpos and qvel only, so they are valid right after set_state).
    @property
    def qpos(self) -> np.ndarray:
        return self.data.qpos.copy()

    def obs(self) -> np.ndarray:
        q, v = self.data.qpos, self.data.qvel
        return np.array([q[0], q[1], np.cos(q[2]), np.sin(q[2]), v[0], v[1], v[2], q[3], q[4], v[3], v[4]])

    def gap(self) -> float:
        q = self.data.qpos
        return float(gap(q[0:2], q[2], q[3:5]))

    # Dynamics.
    def step(self, action: np.ndarray, track_contact: bool = True) -> bool:
        """One control step under commanded velocity `action`; returns the contact flag.

        Contact is True if any pusher-box contact exists at any of the 50 physics steps. Only the
        box and the pusher can collide (checked at load), so any contact is a pusher-box contact.
        """
        self.data.ctrl[:] = action
        if not track_contact:
            mujoco.mj_step(self.model, self.data, nstep=c.PHYSICS_STEPS_PER_CONTROL)
            return False
        contact = False
        for _ in range(c.PHYSICS_STEPS_PER_CONTROL):
            mujoco.mj_step(self.model, self.data)
            contact = contact or self.data.ncon > 0
        return contact

    def step_without_contact(self, action: np.ndarray) -> None:
        """One control step with contacts disabled: the box moves as if the pusher were absent."""
        flags = self.model.opt.disableflags
        self.model.opt.disableflags = flags | mujoco.mjtDisableBit.mjDSBL_CONTACT
        try:
            self.step(action, track_contact=False)
        finally:
            self.model.opt.disableflags = flags
