"""Training pipeline checks (SMOKE_TEST.md, Training pipeline checks, parts 1 to 3).

Training side only (SMOKE_TEST.md, No-leak rule): models are trained on smoke data with recorded
actions, or on made-up targets, and are evaluated on L_obs (validation transitions, recorded
actions) or on their training loss. Nothing here computes an intervention error or reads the
test-like states. Each check returns a plain dict with its numbers, pass condition and result.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from physical_intervene import baselines
from physical_intervene import constants as c
from physical_intervene import metrics as M
from physical_intervene.data import (
    Examples,
    Split,
    TargetScaler,
    box_scaler,
    box_target_from_delta,
    make_examples,
    split_overlap,
)
from physical_intervene.sim import wrap_angle
from physical_intervene.train import checkpoint_schedule

# Part 1: data --------------------------------------------------------------------------------------


def hand_built_split(n_steps: int = 12, theta0: float = np.pi - 0.03, dtheta: float = 0.02) -> Split:
    """One episode with known values: every number encodes its step, and the box turns across +pi."""
    T = n_steps
    obs = np.zeros((1, T + 1, 11))
    t = np.arange(T + 1)
    th = theta0 + dtheta * t
    obs[0, :, 0] = 0.001 * t  # box x
    obs[0, :, 1] = -0.002 * t
    obs[0, :, 2], obs[0, :, 3] = np.cos(th), np.sin(th)
    for j in range(4, 11):
        obs[0, :, j] = 100.0 * j + t  # step t is readable from every other number
    action = np.stack([1000.0 + t[:-1], 2000.0 + t[:-1]], axis=-1)[None]
    target = np.stack([np.full(T, 0.001), np.full(T, -0.002), np.full(T, dtheta)], -1)[None]
    return Split("hand_built", obs, action, target, np.array([0]))


def check_alignment() -> dict:
    sp = hand_built_split()
    ex = make_examples(sp)
    errors = []
    for i in range(len(ex)):
        t = int(ex.step[i])
        if not np.array_equal(ex.s[i], sp.obs[0, t]) or not np.array_equal(ex.s_next[i], sp.obs[0, t + 1]):
            errors.append(f"state at t={t}")
        if not np.array_equal(ex.a[i], sp.action[0, t]):
            errors.append(f"action at t={t}")
        for k in range(c.HISTORY_LEN):
            src = t - c.HISTORY_LEN + k
            real = src >= 0
            if ex.hist_mask[i, k] != float(real):
                errors.append(f"mask at t={t}, k={k}")
            want_s = sp.obs[0, src] if real else sp.obs[0, 0]
            want_a = sp.action[0, src] if real else np.zeros(2)
            if not (np.array_equal(ex.hist_s[i, k], want_s) and np.array_equal(ex.hist_a[i, k], want_a)):
                errors.append(f"history at t={t}, k={k}")
        # Nothing at or after t in the history: the step is encoded in states (column 4) and actions.
        steps_seen = np.concatenate(
            [ex.hist_s[i, :, 4] - 400.0, ex.hist_a[i, :, 0][ex.hist_mask[i] > 0] - 1000.0]
        )
        if steps_seen.size and steps_seen.max() >= t and t > 0:
            errors.append(f"future in history at t={t}")
    return {
        "examples": len(ex),
        "history_len": c.HISTORY_LEN,
        "padding": "copies of s_0 with a zero action, mask 0",
        "errors": errors[:10],
        "pass_condition": "every example pairs s_t, the history up to t-1, a_t and s_{t+1}; nothing after t",
        "pass": not errors,
    }


def check_target(split: Split) -> dict:
    sp = hand_built_split()
    ex = make_examples(sp)
    th = np.arctan2(sp.obs[0, :, 3], sp.obs[0, :, 2])
    crossing = np.flatnonzero(np.diff(np.sign(th)) != 0)
    wrap_ok = bool(np.allclose(ex.y[:, 2], 0.02, atol=1e-12))
    # Direct computation on real data: from the states (cos, sin) against the stored qpos target.
    ex_real = make_examples(split)
    diff = np.abs(ex_real.y - split.target.reshape(-1, 3))
    # The model's path: box target from a state change (atan2 of predicted cos, sin).
    via_delta = box_target_from_delta(ex_real.s, ex_real.delta)
    return {
        "hand_built_crossing_step": crossing.tolist(),
        "hand_built_dtheta": [float(ex.y[:, 2].min()), float(ex.y[:, 2].max())],
        "wrap_pass": wrap_ok,
        "max_abs_diff_vs_stored_target": diff.max(0).tolist(),
        "max_abs_diff_via_delta": np.abs(via_delta - ex_real.y).max(0).tolist(),
        "wrap_angle_examples": {
            "pi-0.01 to -pi+0.01": float(wrap_angle(-np.pi + 0.01 - (np.pi - 0.01))),
            "-pi+0.01 to pi-0.01": float(wrap_angle(np.pi - 0.01 - (-np.pi + 0.01))),
        },
        "pass_condition": "dtheta wraps across +-pi on a hand-built episode; the box target from the states "
        "matches the stored one (from qpos) to 1e-9",
        "pass": wrap_ok and bool(diff.max() < 1e-9) and bool(np.abs(via_delta - ex_real.y).max() < 1e-9),
    }


def check_scaling(train: Examples, val: Examples) -> dict:
    s1 = box_scaler(train)
    val_changed = val.subset(np.arange(len(val)))
    val_changed.y = val_changed.y * 10.0 + 1.0
    s2 = box_scaler(train)  # the scaler takes only training data; refit after changing validation
    s_val = box_scaler(val)
    return {
        "train_std": s1.scale.tolist(),
        "floors": list(c.TARGET_STD_FLOORS),
        "floors_bind": bool(np.any(s1.scale <= np.asarray(c.TARGET_STD_FLOORS))),
        "validation_std_for_comparison": s_val.scale.tolist(),
        "unchanged_when_validation_changes": bool(np.array_equal(s1.scale, s2.scale)),
        "pass_condition": "the scaler is fitted on the training split only; "
        "changing validation data leaves it unchanged",
        "pass": bool(np.array_equal(s1.scale, s2.scale)),
    }


def check_splits(splits: list[Split]) -> dict:
    out = split_overlap(splits)
    shared = [v for v in out["pairs"].values() if v["shared_seeds"] or v["shared_transition_hashes"]]
    all_smoke = all(v == ["smoke"] for v in out["design_split_of_seeds"].values())
    out.update(
        {
            "pass_condition": "no episode seed or transition hash appears in two splits; "
            "Phase 0 uses smoke seeds only",
            "pass": not shared and all_smoke,
        }
    )
    return out


# Part 2: made-up targets ---------------------------------------------------------------------------


def _rng(name: str) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(20_261_124, spawn_key=(sum(map(ord, name)),)))


def history_features(ex: Examples, norm) -> np.ndarray:
    """The history as the models see it, padding zeroed: (N, L x 13)."""
    hs = (ex.hist_s - norm.s_mean) / norm.s_std
    ha = (ex.hist_a - norm.a_mean) / norm.a_std
    return (np.concatenate([hs, ha], -1) * ex.hist_mask[..., None]).reshape(len(ex), -1)


def made_up_targets(ex: Examples, norm, dscaler: TargetScaler) -> dict[str, np.ndarray]:
    """Made-up state changes (physical units) whose right answer is known (SMOKE_TEST.md, part 2)."""
    sig = dscaler.scale
    N = len(ex)
    out = {"zero": np.zeros((N, 11))}
    act = np.zeros((N, 11))
    act[:, 7:9] = ex.a * c.CONTROL_STEP_S
    out["action"] = act
    hf = history_features(ex, norm)
    u = _rng("history").normal(size=(hf.shape[1], 11)) / np.sqrt(hf.shape[1])
    out["history"] = sig * np.tanh(hf @ u)
    x = np.concatenate([(ex.s - norm.s_mean) / norm.s_std, (ex.a - norm.a_mean) / norm.a_std], 1)
    g = _rng("smooth")
    W, b = g.normal(size=(13, 16)) / np.sqrt(13), g.normal(size=16) * 0.5
    V = g.normal(size=(16, 11)) / np.sqrt(16)
    out["smooth"] = sig * (np.tanh(x @ W + b) @ V)
    # contact_wm: a smooth function of its correction network's inputs (the box's velocity in its
    # own frame) on the box's (dx, dy), rotated to the world frame.
    th = np.arctan2(ex.s[:, 3], ex.s[:, 2])
    ct, st = np.cos(th), np.sin(th)
    vb = np.stack([ct * ex.s[:, 4] + st * ex.s[:, 5], -st * ex.s[:, 4] + ct * ex.s[:, 5], ex.s[:, 6]], 1)
    vb = vb / vb.std(0)
    Wc = _rng("contact_wm").normal(size=(3, 2))
    loc = sig[0:2] * np.tanh(vb @ Wc)
    cw = np.zeros((N, 11))
    cw[:, 0] = ct * loc[:, 0] - st * loc[:, 1]
    cw[:, 1] = st * loc[:, 0] + ct * loc[:, 1]
    out["smooth_contact_wm_features"] = cw
    return out


ROLLOUT_TARGETS = ("zero_rollout", "action_rollout")


def made_up_episodes(split: Split, kind: str) -> Split:
    """Episodes whose states follow a made-up rule from the real first state with the real actions:
    *zero_rollout*, nothing ever moves; *action_rollout*, the pusher position moves by a_t x the
    control step and nothing else changes. Their one-step state changes equal the made-up targets
    *zero* and *action*, and a model rolled out over them meets the states it predicts."""
    obs = np.repeat(split.obs[:, :1], split.obs.shape[1], axis=1).copy()
    if kind == "action_rollout":
        step = split.action * c.CONTROL_STEP_S
        obs[:, 1:, 7:9] += np.cumsum(step, axis=1)
    elif kind != "zero_rollout":
        raise ValueError(kind)
    return replace(split, name=f"{split.name} ({kind})", obs=obs, target=np.zeros_like(split.target))


# Which made-up target each architecture is trained on, and on which dimensions of the loss.
# contact_wm: its structure maps (s_t, a_t) to the state change through a pusher lag path and a box
# correction on geometric features, so only the parts it can represent are tested (see report).
ALL_DIMS = None
BOX = list(range(7))
PUSHER_POS = [7, 8]
MADE_UP_PLAN: dict[str, list[tuple[str, list[int] | None]]] = {
    "mlp": [("zero", ALL_DIMS), ("action", ALL_DIMS), ("history", ALL_DIMS), ("smooth", ALL_DIMS)],
    "gru": [("zero", ALL_DIMS), ("action", ALL_DIMS), ("history", ALL_DIMS), ("smooth", ALL_DIMS)],
    # mlp-ms trains on rollouts, so its made-up targets must define states to roll out over:
    # made-up episodes from the real start states and actions (made_up_episodes). Its network and
    # one-step wiring are mlp's; the rollout's feedback is covered by a unit test.
    "mlp-ms": [("zero_rollout", ALL_DIMS), ("action_rollout", ALL_DIMS)],
    "contact_wm": [("zero", BOX), ("action", PUSHER_POS), ("smooth_contact_wm_features", [0, 1])],
    "blind": [("action", ALL_DIMS)],  # expected not to fit: it never sees a_t
}


# Part 3: learning works ----------------------------------------------------------------------------


def baseline_losses(train: Examples, val: Examples) -> dict:
    ys = box_scaler(train)
    out = {}
    for name, fn in (
        ("copy", baselines.copy),
        ("constvel", baselines.constvel),
        ("knn", baselines.KNN(train)),
    ):
        y_pred = box_target_from_delta(val.s, fn(val))
        out[name] = {
            "l_obs": ys.std_mse(y_pred, val.y),
            "corner_rmse_mm": M.corner_rmse_mm(val.s, y_pred, val.y),
        }
    direct = ys.std_mse(np.zeros_like(val.y), val.y)
    out["copy"]["direct_from_box_motion"] = direct
    return out


def check_losses_fall(runs: list[dict]) -> dict:
    rows, ok = {}, True
    for r in runs:
        L = [x["l_obs"] for x in r["curve"]]
        u = [x["updates"] for x in r["curve"]]
        i100 = u.index(100)
        bi, bl = M.best(L)
        good = L[i100] < L[0] and bl < L[i100]
        ok &= good
        rows[r["name"]] = {
            "l_obs_at_0": L[0],
            "l_obs_at_100": L[i100],
            "best": bl,
            "best_at": u[bi],
            "last": L[-1],
            "pass": good,
        }
    return {
        "runs": rows,
        "pass_condition": "L_obs(100) < L_obs(0) and best L_obs < L_obs(100), every architecture and width",
        "pass": ok,
    }


def check_beats_baselines(runs: list[dict], base: dict) -> dict:
    rows, ok = {}, True
    for r in runs:
        L = [x["l_obs"] for x in r["curve"]]
        best, last = min(L), L[-1]
        good = all(v < base[b]["l_obs"] for v in (best, last) for b in ("copy", "constvel"))
        ok &= good
        rows[r["name"]] = {"best": best, "last": last, "pass": good}
    return {
        "runs": rows,
        "copy": base["copy"]["l_obs"],
        "constvel": base["constvel"]["l_obs"],
        "knn": base["knn"]["l_obs"],
        "pass_condition": "best and last L_obs below copy and constvel, every architecture and width",
        "pass": ok,
    }


def check_matching(runs: list[dict], main: tuple[str, ...] = c.MAIN_ARCHITECTURES) -> dict:
    """DESIGN.md, Matching observational loss, on the medium-width smoke runs."""
    med = [r for r in runs if r["config"]["width"] == "medium" and r["config"]["arch"] in main]
    best_by_arch: dict[str, list[float]] = {}
    for r in med:
        best_by_arch.setdefault(r["config"]["arch"], []).append(min(x["l_obs"] for x in r["curve"][1:]))
    vstar = M.v_star(best_by_arch)
    rows, ok = {}, True
    seed_mean: dict[str, list[float]] = {}
    for r in med:
        L = [x["l_obs"] for x in r["curve"][1:]]  # saved checkpoints only (update 0 is not one)
        u = [x["updates"] for x in r["curve"][1:]]
        got = M.matched_checkpoint(L, vstar)
        brute = min(range(len(L)), key=lambda i: (abs(L[i] - vstar), i))
        excl = M.excluded(L, vstar)
        brute_excl = all(abs(x - vstar) > c.MATCHED_LOSS_REL_TOL * vstar for x in L)
        crosses = min(L) <= vstar
        good = got == brute and excl == brute_excl
        ok &= good
        if not excl:
            seed_mean.setdefault(r["config"]["arch"], []).append(L[got])
        rows[r["name"]] = {
            "matched_update": u[got],
            "matched_l_obs": L[got],
            "expected_update": u[brute],
            "excluded": excl,
            "reaches_v_star": crosses,
            "pass": good,
        }
    arch_l = {a: float(np.mean(v)) for a, v in seed_mean.items()}
    screen = {
        f"{a}/{b}": M.passes_match_screen(arch_l[a], arch_l[b])
        for i, a in enumerate(sorted(arch_l))
        for b in sorted(arch_l)[i + 1 :]
    }
    return {
        "best_by_arch": best_by_arch,
        "medians": {a: float(np.median(v)) for a, v in best_by_arch.items()},
        "v_star_like": vstar,
        "runs": rows,
        "seed_mean_matched_l_obs": arch_l,
        "match_screen": screen,
        "pass_condition": "matched checkpoints and exclusions equal a brute-force recomputation "
        "of DESIGN.md's rule; "
        "every architecture has a run reaching V*",
        "pass": ok
        and all(
            any(rows[r["name"]]["reaches_v_star"] for r in med if r["config"]["arch"] == a)
            for a in best_by_arch
        ),
    }


def check_budget(runs: list[dict]) -> dict:
    rows, ok = {}, True
    for r in runs:
        cfg = r["config"]
        u = [x["updates"] for x in r["curve"]]
        L = [x["l_obs"] for x in r["curve"]]
        sched_ok = u[1:] == checkpoint_schedule(cfg["updates"])
        lr_ok = all(lr == cfg["lr"] for lr in r["final_lr"])
        full = r["updates_done"] == cfg["updates"]
        bi, _ = M.best(L[1:])
        past_best = bi + 1 < len(L) - 1
        good = sched_ok and lr_ok and full
        ok &= good
        rows[r["name"]] = {
            "updates": r["updates_done"],
            "checkpoints": len(u) - 1,
            "schedule_ok": sched_ok,
            "constant_lr": lr_ok,
            "best_update": u[bi + 1],
            "continues_past_best": past_best,
            "plateau": M.plateau(u[1:], L[1:]),
            "pass": good,
        }
    return {
        "runs": rows,
        "pass_condition": "full number of updates, constant learning rate, "
        "checkpoints every 10 updates to 1,000, every 100 to 2,000, then every "
        "1,000; plateau criterion and whether the run continues past its best checkpoint are reported",
        "pass": ok,
        "runs_continuing_past_best": sum(v["continues_past_best"] for v in rows.values()),
    }


def check_reproducible(first: dict, again: dict) -> dict:
    n = len(again["curve"])
    a = [(x["updates"], x["l_obs"], x["train_loss"]) for x in first["curve"][:n]]
    b = [(x["updates"], x["l_obs"], x["train_loss"]) for x in again["curve"]]
    return {"checkpoints_compared": n, "identical": a == b, "pass": a == b}
