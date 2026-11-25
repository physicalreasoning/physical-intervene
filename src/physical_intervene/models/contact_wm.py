"""contact_wm (main, different structure): hand-written pushing physics with learned numbers.

DESIGN.md, *The structured model*. Over CONTACT_WM_SUBSTEPS sub-steps of the control step:
1. Pusher path: the pusher velocity follows the command with a learned first-order lag.
2. Contact: in the box frame, the pusher's signed distance to a square of learned half-width b
   gives the penetration r - d (learned pusher radius r), a smooth contact indicator
   sigmoid(k (r - d)) (learned sharpness k), the contact point and the outward normal.
3. Pushing law: in contact, while the pusher moves into the box, the box moves quasi-statically.
   With an ellipsoidal limit surface (learned constant c), a force f at contact point (cx, cy)
   gives the box twist (fx, fy, (cx fy - cy fx) / c^2). The friction cone (learned pusher-box
   friction mu) maps to a motion cone at the contact; the pusher's velocity is clamped smoothly
   into it (stick inside, slip on its edge), and the box twist is the one whose contact-point
   velocity equals the clamped velocity. Out of contact, the box keeps its velocity and slows
   under a learned table friction (deceleration mu_t g; angular, mu_t g / c).
4. Correction network: a small MLP on geometric features only (contact-weighted means over the
   sub-steps of the contact indicator, contact point, normal and relative velocity, and the box's
   velocity at the start, all in the box frame) adds a residual to the box's motion. It never sees
   the history. Its last layer starts at zero, so training starts from the physics alone.

No MuJoCo value is written in: every number starts from the generic values in constants.py and is
learned. Only the correction network counts towards the width.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from physical_intervene import constants as c
from physical_intervene.models.common import Batch, WorldModel, mlp_stack

N_FEATURES = 10
N_RESIDUAL = 6  # box frame: dx, dy, dtheta, then vx, vy, omega at the end of the step
LENGTH_UNIT = c.CONTACT_WM_INIT_BOX_HALF_WIDTH_M  # feature scaling only
SPEED_UNIT = c.ACTION_MAX_SPEED_M_S


def softplus_b(x: torch.Tensor, beta: float) -> torch.Tensor:
    return nn.functional.softplus(x, beta=beta, threshold=20.0)


def rotate(v: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate 2-vectors (..., 2) by the angle with the given cos and sin."""
    return torch.stack([cos * v[..., 0] - sin * v[..., 1], sin * v[..., 0] + cos * v[..., 1]], -1)


def square_geometry(rel: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Signed distance from a point (box frame) to a square of half-width b, the outward normal at
    the nearest surface point, and that point."""
    q = rel.abs() - b
    outside = torch.clamp(q, min=0.0)
    d_out = torch.sqrt((outside * outside).sum(-1) + 1e-18)
    is_out = (q > 0).any(-1)
    d_in = torch.clamp(q.max(-1).values, max=0.0)
    d = torch.where(is_out, d_out, d_in)
    sign = torch.where(rel >= 0, 1.0, -1.0).to(rel.dtype)
    n_out = sign * outside / d_out[..., None]
    x_face = (q[..., 0] >= q[..., 1]).to(rel.dtype)
    n_in = sign * torch.stack([x_face, 1.0 - x_face], -1)
    n = torch.where(is_out[..., None], n_out, n_in)
    point = rel - d[..., None] * n
    return d, n, point


def twist_from_force(f: torch.Tensor, cp: torch.Tensor, c_ls: torch.Tensor) -> torch.Tensor:
    """Box twist (vx, vy, omega), up to scale, for a force f at contact point cp (box frame)."""
    m = cp[..., 0] * f[..., 1] - cp[..., 1] * f[..., 0]
    return torch.stack([f[..., 0], f[..., 1], m / c_ls**2], -1)


def contact_point_velocity(tw: torch.Tensor, cp: torch.Tensor) -> torch.Tensor:
    return torch.stack([tw[..., 0] - tw[..., 2] * cp[..., 1], tw[..., 1] + tw[..., 2] * cp[..., 0]], -1)


def twist_from_contact_velocity(v: torch.Tensor, cp: torch.Tensor, c_ls: torch.Tensor) -> torch.Tensor:
    """The quasi-static twist whose contact-point velocity is v (sticking contact)."""
    cx, cy, c2 = cp[..., 0], cp[..., 1], c_ls**2
    den = c2 + cx * cx + cy * cy
    vx = ((c2 + cx * cx) * v[..., 0] + cx * cy * v[..., 1]) / den
    vy = (cx * cy * v[..., 0] + (c2 + cy * cy) * v[..., 1]) / den
    return torch.stack([vx, vy, (cx * vy - cy * vx) / c2], -1)


def push_twist(
    v_pusher: torch.Tensor, n: torch.Tensor, cp: torch.Tensor, mu: torch.Tensor, c_ls: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quasi-static box twist under a pusher moving at v_pusher (box frame), and the pusher's speed
    into the box. The pusher's tangential-to-normal ratio is clamped smoothly into the motion cone."""
    t = torch.stack([-n[..., 1], n[..., 0]], -1)
    v_in = -(v_pusher * n).sum(-1)
    v_in_pos = softplus_b(v_in, 1.0 / c.CONTACT_WM_PUSH_SMOOTHING_M_S)
    # The ratio is clamped into the cone below; the floor only keeps it finite when the pusher
    # moves away (its speed into the box then multiplies the result and is about 0).
    gamma = (v_pusher * t).sum(-1) / torch.clamp(v_in_pos, min=1e-6)
    edges = []
    for sgn in (1.0, -1.0):
        f = -n + sgn * mu[..., None] * t
        vc = contact_point_velocity(twist_from_force(f, cp, c_ls), cp)
        edges.append((vc * t).sum(-1) / (-(vc * n).sum(-1)))
    lo, hi = torch.minimum(*edges), torch.maximum(*edges)
    beta = c.CONTACT_WM_SLIP_SMOOTHING
    g_eff = lo + softplus_b(gamma - lo, beta) - softplus_b(gamma - hi, beta)
    v_cp = v_in_pos[..., None] * (-n + g_eff[..., None] * t)
    return twist_from_contact_velocity(v_cp, cp, c_ls), v_in


class ContactWM(WorldModel):
    arch = "contact_wm"

    def __init__(self, norm, dscaler, hidden: int) -> None:
        super().__init__(norm, dscaler)
        b0 = c.CONTACT_WM_INIT_BOX_HALF_WIDTH_M
        init = {
            "log_lag": math.log(c.CONTACT_WM_INIT_LAG_S),
            "log_half_width": math.log(b0),
            "log_pusher_radius": math.log(c.CONTACT_WM_INIT_PUSHER_RADIUS_M),
            "log_sharpness": math.log(c.CONTACT_WM_INIT_SHARPNESS_PER_M),
            "log_mu_pusher": math.log(c.CONTACT_WM_INIT_FRICTION),
            "log_mu_table": math.log(c.CONTACT_WM_INIT_FRICTION),
            "log_limit_surface": math.log(c.UNIFORM_SQUARE_LIMIT_SURFACE_FACTOR * b0),
        }
        self.physics = nn.ParameterDict({k: nn.Parameter(torch.tensor(v)) for k, v in init.items()})
        self.correction = mlp_stack([N_FEATURES] + [hidden] * c.CONTACT_WM_HIDDEN_LAYERS + [N_RESIDUAL])
        last = self.correction[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.correction.parameters())

    def physical_values(self) -> dict[str, float]:
        return {k.removeprefix("log_"): torch.exp(v).item() for k, v in self.physics.items()}

    def residual_scales(self) -> torch.Tensor:
        s = self.d_scale
        pos = 0.5 * (s[0] + s[1])
        vel = 0.5 * (s[4] + s[5])
        return torch.stack([pos, pos, s[6] * c.CONTROL_STEP_S, vel, vel, s[6]])

    def physics_step(self, s: torch.Tensor, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Physics only: the next state (physical units) and the correction features."""
        p = {k: torch.exp(v) for k, v in self.physics.items()}
        b, r, k_sharp = p["log_half_width"], p["log_pusher_radius"], p["log_sharpness"]
        mu, mu_t, c_ls = p["log_mu_pusher"], p["log_mu_table"], p["log_limit_surface"]
        dt = c.CONTROL_STEP_S / c.CONTACT_WM_SUBSTEPS
        alpha = 1.0 - torch.exp(-dt / p["log_lag"])
        box = s[:, 0:2]
        th = torch.atan2(s[:, 3], s[:, 2])
        v_box, w_box = s[:, 4:6], s[:, 6]
        pusher, v_p = s[:, 7:9], s[:, 9:11]
        cos0, sin0 = torch.cos(th), torch.sin(th)
        feat_v0 = torch.cat(
            [rotate(v_box, cos0, -sin0) / SPEED_UNIT, (w_box * LENGTH_UNIT / SPEED_UNIT)[:, None]], 1
        )
        acc = {"w": 0.0, "cp": 0.0, "n": 0.0, "vrel": 0.0}
        decel, decel_w = mu_t * c.GRAVITY_M_S2, mu_t * c.GRAVITY_M_S2 / c_ls
        stop_beta = 1.0 / c.CONTACT_WM_STOP_SMOOTHING_M_S
        for _ in range(c.CONTACT_WM_SUBSTEPS):
            v_p = v_p + alpha * (a - v_p)
            pusher = pusher + v_p * dt
            cos, sin = torch.cos(th), torch.sin(th)
            rel = rotate(pusher - box, cos, -sin)
            d, n, cp = square_geometry(rel, b)
            w_contact = torch.sigmoid(k_sharp * (r - d))
            vp_box = rotate(v_p, cos, -sin)
            tw_push, _ = push_twist(vp_box, n, cp, mu, c_ls)
            v_box_frame = rotate(v_box, cos, -sin)
            v_cp_box = v_box_frame + torch.stack([-w_box * cp[:, 1], w_box * cp[:, 0]], -1)
            v_rel = vp_box - v_cp_box
            approach = -(v_rel * n).sum(-1)
            w_push = w_contact * torch.sigmoid(approach / c.CONTACT_WM_PUSH_SMOOTHING_M_S)
            # Free motion with table friction.
            speed = torch.sqrt((v_box * v_box).sum(-1) + 1e-18)
            v_free = v_box * (softplus_b(speed - decel * dt, stop_beta) / speed)[:, None]
            w_abs = torch.sqrt(w_box * w_box + 1e-18)
            w_free = w_box * softplus_b(w_abs - decel_w * dt, stop_beta) / w_abs
            v_push_world = rotate(tw_push[:, 0:2], cos, sin)
            v_box = w_push[:, None] * v_push_world + (1 - w_push[:, None]) * v_free
            w_box = w_push * tw_push[:, 2] + (1 - w_push) * w_free
            box = box + v_box * dt
            th = th + w_box * dt
            acc["w"] = acc["w"] + w_push
            acc["cp"] = acc["cp"] + w_push[:, None] * cp / LENGTH_UNIT
            acc["n"] = acc["n"] + w_push[:, None] * n
            acc["vrel"] = acc["vrel"] + w_push[:, None] * v_rel / SPEED_UNIT
        m = c.CONTACT_WM_SUBSTEPS
        feats = torch.cat(
            [acc["w"][:, None] / m, acc["cp"] / m, acc["n"] / m, acc["vrel"] / m, feat_v0], dim=1
        )
        state = {"box": box, "th": th, "v_box": v_box, "w_box": w_box, "pusher": pusher, "v_p": v_p}
        return feats, (cos0, sin0), state

    def predict_delta(self, b: Batch) -> torch.Tensor:
        """The predicted state change in physical units."""
        s = b.s
        feats, (cos0, sin0), st = self.physics_step(s, b.a)
        res = self.correction(feats) * self.residual_scales()
        box = st["box"] + rotate(res[:, 0:2], cos0, sin0)
        th = st["th"] + res[:, 2]
        v_box = st["v_box"] + rotate(res[:, 3:5], cos0, sin0)
        w_box = st["w_box"] + res[:, 5]
        th0 = torch.atan2(s[:, 3], s[:, 2])
        return torch.cat(
            [
                box - s[:, 0:2],
                (torch.cos(th) - torch.cos(th0))[:, None],
                (torch.sin(th) - torch.sin(th0))[:, None],
                v_box - s[:, 4:6],
                (w_box - s[:, 6])[:, None],
                st["pusher"] - s[:, 7:9],
                st["v_p"] - s[:, 9:11],
            ],
            dim=1,
        )

    def forward(self, b: Batch) -> torch.Tensor:
        return (self.predict_delta(b) - self.d_mean) / self.d_scale
