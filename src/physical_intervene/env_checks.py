"""Phase 0 environment checks (docs/SMOKE_TEST.md, Environment checks), simulator only.

Each check returns a plain dict with its numbers, its pass condition and `passed` (True, False,
or None for checks that are reported, not gated). No model of any kind is used here.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.spatial import cKDTree

from physical_intervene import constants as c
from physical_intervene import episodes as E
from physical_intervene import policy
from physical_intervene.interventions import (
    KIND_AIMED,
    KIND_LOGGED,
    KIND_UNIFORM,
    LABELS,
    label,
    label_vs_reference,
)
from physical_intervene.sim import Sim, box_target

LABEL_NO_CONTACT, LABEL_GRAZE, LABEL_PUSH = 0, 1, 2


def _q(x: np.ndarray, qs=(0.05, 0.25, 0.5, 0.75, 0.95)) -> dict:
    x = np.asarray(x, float)
    if x.size == 0:
        return {"n": 0}
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        **{f"q{int(q * 100):02d}": float(np.quantile(x, q)) for q in qs},
    }


# Bit-exact replay -----------------------------------------------------------------------------


def replay(states: dict) -> dict:
    """Restore every stored state, rerun every stored action, compare full states byte for byte."""
    sim = Sim()
    n_states, n_actions = states["actions"].shape[:2]
    mismatches = []
    for i in range(n_states):
        for k in range(n_actions):
            sim.set_state(states["state"][i])
            sim.step(states["actions"][i, k])
            got, want = sim.get_state(), states["after"][i, k]
            if not (got.dtype == want.dtype and np.array_equal(got.view(np.uint8), want.view(np.uint8))):
                mismatches.append([i, k])
    return {
        "n_states": n_states,
        "n_actions_per_state": n_actions,
        "n_compared": n_states * n_actions,
        "n_mismatched": len(mismatches),
        "first_mismatches": mismatches[:10],
        "passed": len(mismatches) == 0,
    }


# Push thresholds ------------------------------------------------------------------------------


def push_thresholds(states: dict) -> dict:
    """Displacement and rotation of every contact intervention (logged action excluded)."""
    contact = states["contact"][:, 1:]
    target = states["target"][:, 1:][contact]
    disp_mm = 1e3 * np.hypot(target[:, 0], target[:, 1])
    rot_deg = np.rad2deg(np.abs(target[:, 2]))
    lab = label(np.ones(len(target), bool), target)
    d, th = c.PUSH_MIN_DISPLACEMENT_M * 1e3, c.PUSH_MIN_ROTATION_DEG
    near = (disp_mm >= 0.5 * d) & (disp_mm < 2 * d)
    return {
        "n_contact_interventions": int(contact.sum()),
        "d_push_mm": d,
        "theta_push_deg": th,
        "graze_share_of_contacts": float(np.mean(lab == LABEL_GRAZE)),
        "displacement_mm": _q(disp_mm),
        "rotation_deg": _q(rot_deg),
        "share_displacement_within_factor_2_of_d_push": float(near.mean()),
        "share_below_d_push_but_rotated_at_least_theta_push": float(np.mean((disp_mm < d) & (rot_deg >= th))),
        "pass_condition": "defaults kept unless the plots show they cut through the bulk of pushes",
        "passed": None,  # decided from the plots and written in the record by the script
    }


# Pusher can move the box ----------------------------------------------------------------------


def pusher_can_move_box() -> dict:
    sim = Sim()
    m = sim.model
    kv = float(m.actuator_gainprm[0, 0])
    force_limit = float(m.actuator_forcerange[0, 1])
    fl_slide = float(m.dof_frictionloss[0])
    f_resist = np.sqrt(2.0) * fl_slide

    def force(v: float) -> float:
        return min(kv * v, force_limit)

    analytic = {
        "kv_N_s_per_m": kv,
        "forcerange_N": force_limit,
        "slide_frictionloss_N": fl_slide,
        "hinge_frictionloss_N_m": float(m.dof_frictionloss[2]),
        "F_resist_N": f_resist,
        "F_at_0.05_N": force(0.05),
        "F_at_0.10_N": force(0.10),
        "margin_at_0.05": force(0.05) / f_resist,
        "pass_condition": f"F(0.05 m/s) >= {c.SMOKE_MIN_PUSH_FORCE_MARGIN} * F_resist",
        "passed": force(0.05) >= c.SMOKE_MIN_PUSH_FORCE_MARGIN * f_resist,
    }

    start_gap = 0.001  # pusher placed 1 mm off the face
    half = c.BOX_SIDE_M / 2
    pushes = []
    cases = [
        (v, name, ang)
        for v in (0.05, 0.10, 0.15)
        for name, ang in (("x", 0.0), ("y", np.pi / 2), ("diagonal", np.pi / 4))
    ]
    for v, name, ang in cases:
        # Box turned so that one face is normal to the push; the pusher sits off the middle of that face.
        u = np.array([np.cos(ang), np.sin(ang)])
        sim.reset(np.zeros(2), ang, -u * (half + c.PUSHER_RADIUS_M + start_gap))
        pushes.append(
            _push_case(
                sim,
                v * u,
                f"centred_{name}_{v:.2f}",
                v,
                fl_slide * (np.sqrt(2) if name == "diagonal" else 1.0),
                kv,
            )
        )
    # Off-centre push at 0.10 m/s: 3 cm from the middle of the -x face.
    sim.reset(np.zeros(2), 0.0, np.array([-(half + c.PUSHER_RADIUS_M + start_gap), 0.03]))
    off = _push_case(sim, np.array([0.10, 0.0]), "off_centre_x_0.10", 0.10, fl_slide, kv)
    centred_ok = all(p["displacement_mm"] >= c.PUSH_MIN_DISPLACEMENT_M * 1e3 for p in pushes)
    off_ok = off["rotation_deg"] >= c.PUSH_MIN_ROTATION_DEG
    simulator = {
        "pushes": pushes + [off],
        "pass_condition": "every centred push moves the box >= default d_push; off-centre push turns it >= "
        "default theta_push",
        "passed": bool(centred_ok and off_ok),
    }
    return {
        "analytic": analytic,
        "simulator": simulator,
        "passed": bool(analytic["passed"] and simulator["passed"]),
    }


def _push_case(sim: Sim, action: np.ndarray, name: str, v: float, f_resist_dir: float, kv: float) -> dict:
    q0 = sim.qpos
    contact = sim.step(action)
    t = box_target(q0, sim.qpos)
    box_speed = float(np.hypot(*sim.data.qvel[0:2]))
    return {
        "case": name,
        "contact": bool(contact),
        "displacement_mm": float(1e3 * np.hypot(t[0], t[1])),
        "rotation_deg": float(np.rad2deg(abs(t[2]))),
        "box_speed_end_m_s": box_speed,
        "expected_steady_speed_m_s": v - f_resist_dir / kv,
    }


# Push rate ------------------------------------------------------------------------------------


def push_rate(states: dict) -> dict:
    lab = label(states["contact"], states["target"])[:, 1:]
    n_push = (lab == LABEL_PUSH).sum(axis=1)
    frac_with_push = float(np.mean(n_push >= 1))
    mean_push = float(n_push.mean())
    kinds = states["kinds"][1:]
    by_kind = {}
    for name, k in (("aimed", KIND_AIMED), ("uniform", KIND_UNIFORM)):
        sub = lab[:, kinds == k]
        by_kind[name] = {lab_name: float(np.mean(sub == j)) for j, lab_name in enumerate(LABELS)}
    logged = label(states["contact"][:, 0], states["target"][:, 0])
    return {
        "n_states": int(len(lab)),
        "frac_states_with_push": frac_with_push,
        "mean_push_per_state": mean_push,
        "push_per_state": _q(n_push),
        "rates_all": {lab_name: float(np.mean(lab == j)) for j, lab_name in enumerate(LABELS)},
        "rates_by_kind": by_kind,
        "rates_logged_action": {lab_name: float(np.mean(logged == j)) for j, lab_name in enumerate(LABELS)},
        "pass_condition": f"frac_states_with_push >= {c.SMOKE_MIN_FRAC_STATES_WITH_PUSH} and "
        f"mean_push_per_state >= {c.SMOKE_MIN_MEAN_PUSH_PER_STATE}",
        "passed": bool(
            frac_with_push >= c.SMOKE_MIN_FRAC_STATES_WITH_PUSH
            and mean_push >= c.SMOKE_MIN_MEAN_PUSH_PER_STATE
        ),
    }


def push_label_diagnostic(states: dict, n_examples: int = 8) -> dict:
    """Standard label against the no-contact-reference label (DESIGN.md design-log issue on moving boxes)."""
    std = label(states["contact"], states["target"])[:, 1:]
    ref = label_vs_reference(states["contact"], states["target"], states["reference"][:, None, :])[:, 1:]
    kinds = states["kinds"][1:]
    out = {}
    for name, mask in (
        ("all", np.ones_like(kinds, bool)),
        ("aimed", kinds == KIND_AIMED),
        ("uniform", kinds == KIND_UNIFORM),
    ):
        s, r = std[:, mask], ref[:, mask]
        contact = s > 0
        out[name] = {
            "n_contact": int(contact.sum()),
            "disagree_share_of_contacts": float(np.mean(s[contact] != r[contact])) if contact.any() else None,
            "push_std_graze_ref": int(np.sum((s == LABEL_PUSH) & (r == LABEL_GRAZE))),
            "graze_std_push_ref": int(np.sum((s == LABEL_GRAZE) & (r == LABEL_PUSH))),
            "push_share_std": float(np.mean(s[contact] == LABEL_PUSH)) if contact.any() else None,
            "push_share_ref": float(np.mean(r[contact] == LABEL_PUSH)) if contact.any() else None,
        }
    ref_speed = np.hypot(states["reference"][:, 0], states["reference"][:, 1])
    out["states_box_moving_without_contact_ge_d_push"] = int(np.sum(ref_speed >= c.PUSH_MIN_DISPLACEMENT_M))
    idx = np.argwhere(std != ref)
    examples = []
    for i, j in idx[:n_examples]:
        k = j + 1
        examples.append(
            {
                "state": int(i),
                "action_index": int(k),
                "kind": "aimed" if kinds[j] == KIND_AIMED else "uniform",
                "label_std": LABELS[std[i, j]],
                "label_ref": LABELS[ref[i, j]],
                "target_mm_deg": _mm_deg(states["target"][i, k]),
                "reference_mm_deg": _mm_deg(states["reference"][i]),
            }
        )
    out["examples"] = examples
    out["passed"] = None
    return out


def _mm_deg(t: np.ndarray) -> list[float]:
    return [round(float(t[0] * 1e3), 3), round(float(t[1] * 1e3), 3), round(float(np.rad2deg(t[2])), 3)]


# Contact coverage in training-like data -------------------------------------------------------


def contact_coverage(eps: dict) -> dict:
    contact, unif = eps["contact"], eps["action_uniform"]
    lab = label(contact, eps["target"])
    n = contact.size
    return {
        "n_transitions": int(n),
        "contact_share": float(contact.mean()),
        "push_share": float(np.mean(lab == LABEL_PUSH)),
        "graze_share": float(np.mean(lab == LABEL_GRAZE)),
        "contacts_from_scripted": int(np.sum(contact & ~unif)),
        "contacts_from_uniform": int(np.sum(contact & unif)),
        "pushes_from_scripted": int(np.sum((lab == LABEL_PUSH) & ~unif)),
        "pushes_from_uniform": int(np.sum((lab == LABEL_PUSH) & unif)),
        "uniform_action_share": float(unif.mean()),
        "passed": None,
    }


# Intervention coverage ------------------------------------------------------------------------


def _flat_transitions(eps: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.concatenate([eps["obs"][:, :-1], eps["action"]], axis=-1).reshape(-1, 13)
    lab = label(eps["contact"], eps["target"]).reshape(-1)
    return x, lab, eps["action_uniform"].reshape(-1)


def intervention_coverage(pool: dict, smoke: dict, states: dict) -> dict:
    x_pool, lab_pool, unif_pool = _flat_transitions(pool)
    mu, sd = x_pool.mean(axis=0), x_pool.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    tree = cKDTree((x_pool[lab_pool == LABEL_PUSH] - mu) / sd)
    k = c.SMOKE_COVERAGE_NEIGHBOURS

    def knn_mean(x: np.ndarray) -> np.ndarray:
        d, _ = tree.query((x - mu) / sd, k=k)
        return d.mean(axis=1)

    x_smoke, lab_smoke, _ = _flat_transitions(smoke)
    d_obs = knn_mean(x_smoke[lab_smoke == LABEL_PUSH])
    tau = float(np.quantile(d_obs, c.SMOKE_COVERAGE_TAIL_QUANTILE))

    lab_int = label(states["contact"], states["target"])[:, 1:]
    kinds = np.broadcast_to(states["kinds"][1:], lab_int.shape)
    obs = np.broadcast_to(states["obs"][:, None, :], (*lab_int.shape, 11))
    x_int = np.concatenate([obs, states["actions"][:, 1:]], axis=-1)
    push = lab_int == LABEL_PUSH
    d_int = knn_mean(x_int[push])
    k_int = kinds[push]
    tail = d_int > tau
    t_share = float(tail.mean())

    def action_share(x: np.ndarray) -> np.ndarray:
        """Share of the squared distance to the 5 neighbours that comes from the 2 action dimensions."""
        z = (x - mu) / sd
        _, idx = tree.query(z, k=k)
        diff2 = (tree.data[idx] - z[:, None, :]) ** 2
        return diff2[..., 11:].sum(axis=(1, 2)) / diff2.sum(axis=(1, 2))

    share_int = action_share(x_int[push])
    share_obs = action_share(x_smoke[lab_smoke == LABEL_PUSH])
    return {
        "n_pool_transitions": int(len(x_pool)),
        "n_pool_push": int(np.sum(lab_pool == LABEL_PUSH)),
        "pool_push_share_from_uniform": float(np.mean(unif_pool[lab_pool == LABEL_PUSH])),
        "n_behaviour_push_smoke": int(len(d_obs)),
        "n_behaviour_push_beyond_tau": int(np.sum(d_obs > tau)),
        "n_push_interventions": int(len(d_int)),
        "tau": tau,
        "T": t_share,
        "T_aimed": float(tail[k_int == KIND_AIMED].mean()),
        "T_uniform": float(tail[k_int == KIND_UNIFORM].mean()),
        "R": float(np.median(d_int) / np.median(d_obs)),
        "d_obs": _q(d_obs),
        "d_int": _q(d_int),
        "diagnostic_action_share_of_sq_distance": {
            "interventions": _q(share_int),
            "behaviour": _q(share_obs),
            "action_dims_share_if_equal": 2 / 13,
        },
        "epsilon_near": policy.EPSILON_NEAR,
        "pass_condition": f"T >= {c.SMOKE_MIN_COVERAGE_TAIL_SHARE}",
        "passed": bool(t_share >= c.SMOKE_MIN_COVERAGE_TAIL_SHARE),
    }


# Policy log-density ---------------------------------------------------------------------------


def policy_log_density(states: dict, pool: dict) -> dict:
    lab = label(states["contact"], states["target"])
    ld = states["log_density"]
    kinds = np.broadcast_to(states["kinds"], lab.shape)
    out = {"logged_action_at_states": _q(ld[:, 0])}
    for kname, kind in (("aimed", KIND_AIMED), ("uniform", KIND_UNIFORM), ("interventions", None)):
        sel = (kinds == kind) if kind is not None else (kinds != KIND_LOGGED)
        out[kname] = {"all": _q(ld[sel])}
        for j, lab_name in enumerate(LABELS):
            out[kname][lab_name] = _q(ld[sel & (lab == j)])
    pool_ld = np.array(
        [
            policy.log_density(a, o, g)
            for a, o, g in zip(
                pool["action"].reshape(-1, 2),
                pool["obs"][:, :-1].reshape(-1, 11),
                pool["goal"].reshape(-1, 2),
                strict=True,
            )
        ]
    )
    pool_lab = label(pool["contact"], pool["target"]).reshape(-1)
    out["pool_logged_actions"] = {
        "all": _q(pool_ld),
        **{n: _q(pool_ld[pool_lab == j]) for j, n in enumerate(LABELS)},
    }
    out["log_uniform_density"] = float(policy.LOG_UNIFORM_DENSITY)
    out["passed"] = None
    return out


# Shadow actions -------------------------------------------------------------------------------


def shadow_actions(sim: Sim, seeds: np.ndarray, with_shadow: dict) -> dict:
    without = E.generate(sim, seeds, shadow=False)
    identical = {
        k: bool(
            np.array_equal(
                np.ascontiguousarray(with_shadow[k]).view(np.uint8),
                np.ascontiguousarray(without[k]).view(np.uint8),
            )
        )
        for k in E.TRANSITION_KEYS
    }
    a, s = with_shadow["action"], with_shadow["shadow"]
    da = (a[:, 1:] - a[:, :-1]).reshape(-1, 2)
    ds = (s[:, 1:] - a[:, :-1]).reshape(-1, 2)
    tests = {}
    for name, x, y in (
        ("dx", da[:, 0], ds[:, 0]),
        ("dy", da[:, 1], ds[:, 1]),
        ("norm", np.hypot(*da.T), np.hypot(*ds.T)),
    ):
        ks = stats.ks_2samp(x, y)
        tests[name] = {"ks_statistic": float(ks.statistic), "p_value": float(ks.pvalue)}
    # Shadow and logged action at the same state should agree in their uniform share, and differ.
    return {
        "byte_identical_with_and_without": identical,
        "uniform_share_logged": float(with_shadow["action_uniform"].mean()),
        "uniform_share_shadow": float(with_shadow["shadow_uniform"].mean()),
        "shadow_equals_logged_share": float(np.mean(np.all(a == s, axis=-1))),
        "ks_tests_delta_vs_previous_action": tests,
        "note": "KS tests treat the two samples as independent; they are paired by state, which makes the "
        "test conservative for equal distributions",
        "pass_condition": "byte-identical transitions; shadow and logged proportions and (a - a_prev) "
        "distributions match (KS p >= 0.01 on each)",
        "passed": bool(all(identical.values()) and all(t["p_value"] >= 0.01 for t in tests.values())),
    }


# Positive-control feasibility -----------------------------------------------------------------


def target_std(eps: dict) -> np.ndarray:
    sd = eps["target"].reshape(-1, 3).std(axis=0)
    return np.maximum(sd, np.array(c.TARGET_STD_FLOORS))


def policy_mean_outcome(
    sim: Sim,
    state: np.ndarray,
    obs: np.ndarray,
    goal: np.ndarray,
    streams: policy.Streams,
    n: int = c.FEASIBILITY_POLICY_DRAWS,
) -> np.ndarray:
    """m: the box target averaged over n behaviour-policy actions at this state."""
    out = np.empty((n, 3))
    for i in range(n):
        a, _ = policy.sample(obs, goal, streams)
        sim.set_state(state)
        q0 = sim.qpos
        sim.step(a, track_contact=False)
        out[i] = box_target(q0, sim.qpos)
    return out.mean(axis=0)


def positive_control_feasibility(sim: Sim, smoke: dict, states: dict) -> dict:
    sd = target_std(smoke)
    n_ep, n_t = smoke["action"].shape[:2]
    d_obs_terms = np.empty((n_ep, n_t))
    for i in range(n_ep):
        streams = E.policy_streams(int(smoke["seed"][i]), "feasibility_")
        for t in range(n_t):
            m = policy_mean_outcome(
                sim, smoke["state"][i, t], smoke["obs"][i, t], smoke["goal"][i, t], streams
            )
            d_obs_terms[i, t] = np.mean(((smoke["target"][i, t] - m) / sd) ** 2)
    lab = label(states["contact"], states["target"])
    d_int_terms, d_logged_terms = [], []
    for i in range(len(states["seed"])):
        streams = E.policy_streams(int(states["seed"][i]), "feasibility_")
        m = policy_mean_outcome(sim, states["state"][i], states["obs"][i], states["goal"][i], streams)
        sq = np.mean(((states["target"][i] - m) / sd) ** 2, axis=-1)
        d_int_terms += list(sq[1:][lab[i, 1:] == LABEL_PUSH])
        d_logged_terms.append(sq[0])
    d_obs, d_int = float(d_obs_terms.mean()), float(np.mean(d_int_terms))
    ratio = d_int / d_obs
    return {
        "target_std_used": sd.tolist(),
        "target_std_source": "box target on the 50 smoke episodes (no training split exists yet), with the "
        "design's floors",
        "n_obs_transitions": int(d_obs_terms.size),
        "n_push_interventions": len(d_int_terms),
        "policy_draws_per_state": c.FEASIBILITY_POLICY_DRAWS,
        "D_obs": d_obs,
        "D_obs_on_contact_transitions": float(d_obs_terms[smoke["contact"]].mean()),
        "D_logged_at_test_like_states": float(np.mean(d_logged_terms)),
        "D_int": d_int,
        "D_int_over_D_obs": ratio,
        "reading": "expected relative gap at matched loss is about p^2 * D_int / E_int(mlp) with "
        "p^2 * D_obs <= V* - L_best(mlp); so it is at least 20% when "
        "E_int(mlp) <= 5 * (D_int / D_obs) * (V* - L_best(mlp))",
        "passed": None,
    }
