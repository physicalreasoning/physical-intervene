"""Phase 0, milestone 3: the intervention side with stand-in models (SMOKE_TEST.md, Intervention-side
checks with stand-in models, Repeated pairs, Format check).

    uv run --extra models python scripts/phase0_intervention.py [--quick]

Reads the smoke data of milestone 1 (data/phase0/milestone1/attemptK/: coverage_pool.npz as the
training split for the target scaler, episodes.npz as validation, states.npz with the 24
interventions) and milestone 2's L_obs curves (results/phase0/milestone2/attemptN/training_checks.json,
training side, recorded actions only), which shape the stand-ins' synthetic checkpoint series.
Writes results/phase0/milestone3/attemptM/{intervention_checks.json, intervention_results_standin.json,
plots/}, and data/phase0/milestone3/attemptM/{state_history.npz, untrained/}.

No trained model is loaded or scored: every model scored here is a stand-in or an untrained
network, and the scorer refuses anything else (scoring.assert_scorable).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
import time
from pathlib import Path

import matplotlib
import numpy as np

from physical_intervene import baselines as B
from physical_intervene import constants as c
from physical_intervene import gates as G
from physical_intervene import results_format as RF
from physical_intervene import scoring as S
from physical_intervene import standins as SI
from physical_intervene.data import InputNorm, box_scaler, delta_scaler, load_split, make_examples
from physical_intervene.train import checkpoint_schedule

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SMOKE = ROOT / "data" / "phase0" / "milestone1" / f"attempt{c.SMOKE_ATTEMPT}"
M2 = ROOT / "results" / "phase0" / "milestone2" / f"attempt{c.MILESTONE2_ATTEMPT}"
M1 = ROOT / "results" / "phase0" / "milestone1" / f"attempt{c.SMOKE_ATTEMPT}"
DATA = ROOT / "data" / "phase0" / "milestone3" / f"attempt{c.MILESTONE3_ATTEMPT}"
OUT = ROOT / "results" / "phase0" / "milestone3" / f"attempt{c.MILESTONE3_ATTEMPT}"
FORBIDDEN = ROOT / "data" / "phase0" / "milestone2"  # trained checkpoints: never opened here
PLACEHOLDER_UPDATES = 12_000  # milestone 2's placeholder budget; the curves stop there
MAIN = list(c.MAIN_ARCHITECTURES)

# Stand-in lineup of the full stand-in run: (key, curve architecture, push noise ratio). Keys only
# make the random draws independent.
LINEUP = {
    "mlp": (101, "mlp", 1.0),
    "gru": (102, "gru", 1.0),
    "mlp-ms": (103, "mlp-ms", c.SMOKE_PLANTED_NOISE_RATIO),
    "contact_wm": (104, "contact_wm", 1.0),
    "mlp-twin": (111, "mlp", 1.0),
    "gru-twin": (112, "gru", 1.0),
    "mlp-ms-twin": (113, "mlp-ms", c.SMOKE_PLANTED_NOISE_RATIO),
    "contact_wm-twin": (114, "contact_wm", 1.0),
    "mlp-shortcut": (120, "mlp", c.SMOKE_PLANTED_NOISE_RATIO),
    "blind": (121, "mlp", 3.0),
}
N_SEEDS = len(c.STANDIN_SEEDS)
# Seeds that must fail to reach V* for an architecture to be dropped (fewer than the minimum left).
N_DROP = N_SEEDS - c.G1_MIN_SEEDS_PER_ARCHITECTURE + 1
BLIND_OBS_FACTOR = 2.4  # blind's L_obs over mlp's (milestone 2: 0.138 against 0.057)


def timed(fn, *a, **kw):
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    return out, time.perf_counter() - t0


def rng(*key: int) -> np.random.Generator:
    return SI.gen(9_000, *key)


# Setup -------------------------------------------------------------------------------------------


def setup() -> dict:
    train_split = load_split(SMOKE / "coverage_pool.npz", "train")
    val_split = load_split(SMOKE / "episodes.npz", "validation")
    train = make_examples(train_split)
    val = make_examples(val_split)
    scaler = box_scaler(train)
    hist_path = DATA / "state_history.npz"
    if not hist_path.exists():
        from physical_intervene import episodes as E
        from physical_intervene import interventions as I
        from physical_intervene.sim import Sim

        states = E.load(SMOKE / "states.npz")
        h = I.state_histories(Sim(), states)
        E.save(hist_path, h)
    iset = S.load_interventions(SMOKE / "states.npz", hist_path)
    ctx = SI.Context(scaler, val.y, val.episode, iset)
    curves = SI.milestone2_curves(M2 / "training_checks.json")
    sched = checkpoint_schedule(PLACEHOLDER_UPDATES)
    for arch in curves:
        for w, (upd, _) in curves[arch].items():
            assert upd == sched, (arch, w)
    return {
        "train": train,
        "val": val,
        "scaler": scaler,
        "norm": InputNorm.fit(train),
        "dscaler": delta_scaler(train.delta),
        "iset": iset,
        "ctx": ctx,
        "curves": curves,
        "schedule": sched,
        "files": {
            "train": {
                "file": str((SMOKE / "coverage_pool.npz").relative_to(ROOT)),
                "sha256": train_split.sha256,
            },
            "validation": {
                "file": str((SMOKE / "episodes.npz").relative_to(ROOT)),
                "sha256": val_split.sha256,
            },
            "states": {"file": str((SMOKE / "states.npz").relative_to(ROOT)), "sha256": iset.sha256},
            "state_history": {"file": str(hist_path.relative_to(ROOT)), "sha256": S.file_sha256(hist_path)},
            "milestone2_curves": {
                "file": str((M2 / "training_checks.json").relative_to(ROOT)),
                "sha256": S.file_sha256(M2 / "training_checks.json"),
            },
        },
    }


def spec(
    env: dict,
    name: str,
    key: int,
    curve_arch: str,
    ratio,
    seeds=c.STANDIN_SEEDS,
    width: str = "medium",
    bias_share: float | None = None,
    obs_factor: float = 1.0,
    obs_cv: float = c.STANDIN_OBS_SEED_CV,
) -> SI.StandIn:
    upd, cur = env["curves"][curve_arch][width]
    base = SI.base_curve(cur) * obs_factor
    f = SI.bias_share_for_cv(c.STANDIN_PUSH_SEED_CV) if bias_share is None else bias_share
    return SI.StandIn(
        name, key, tuple(seeds), upd, SI.seed_curves(base, len(seeds), key, obs_cv), f, ratio, width=width
    )


def fam(env: dict, sp: SI.StandIn) -> G.Family:
    f = SI.family(sp, env["ctx"])
    f.meta["predict"] = lambda i, k, sp=sp: sp.predict(env["ctx"], i, k)
    return f


# Checks ------------------------------------------------------------------------------------------


def check_no_leak() -> dict:
    """The scorer refuses trained models in Phase 0. Tested with synthetic metadata only; no trained
    checkpoint is opened."""
    cases = {
        "trained_true": {"trained": True, "phase": "phase0", "architecture": "mlp", "seed": 11},
        "trained_missing": {"phase": "phase0", "architecture": "mlp"},
        "trained_string": {"trained": "false"},
        "untrained": {"trained": False},
        "standin": dict(SI.STANDIN_META),
    }
    out = {}
    for name, meta in cases.items():
        try:
            S.assert_scorable(meta)
            out[name] = "accepted"
        except S.LeakError:
            out[name] = "refused"
    try:
        S.assert_scorable({"trained": True}, phase="scoring")
        out["trained_true_after_phase0"] = "accepted"
    except S.LeakError:
        out["trained_true_after_phase0"] = "refused"
    # A file whose sidecar says trained: the entry point refuses before reading the file, which holds
    # no checkpoint at all (bytes that torch could not load).
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "fake.pt"
        p.write_bytes(b"not a checkpoint")
        p.with_suffix(".json").write_text(json.dumps(cases["trained_true"]))
        try:
            S.load_checkpoint_for_scoring(p)
            out["file_with_trained_sidecar"] = "accepted"
        except S.LeakError:
            out["file_with_trained_sidecar"] = "refused"
        q = Path(d) / "nometa.pt"
        q.write_bytes(b"not a checkpoint")
        try:
            S.load_checkpoint_for_scoring(q)
            out["file_without_metadata"] = "accepted"
        except S.LeakError:
            out["file_without_metadata"] = "refused"
    expected = {
        "trained_true": "refused",
        "trained_missing": "refused",
        "trained_string": "refused",
        "untrained": "accepted",
        "standin": "accepted",
        "trained_true_after_phase0": "accepted",
        "file_with_trained_sidecar": "refused",
        "file_without_metadata": "refused",
    }
    return {
        "phase": "phase0",
        "results": out,
        "expected": expected,
        "pass_condition": "trained or unlabelled metadata refused in phase0; trained: false accepted",
        "pass": out == expected,
    }


def check_oracle(env: dict) -> dict:
    ctx = env["ctx"]
    ls = S.losses(ctx.scaler, SI.oracle_prediction(ctx), ctx.val_y, ctx.iset)
    e = {
        b: float(S.state_sums(ls.interventions, ctx.iset, b).sum() / ctx.iset.counts(b).sum())
        for b in S.BLOCKS
    }
    return {
        "l_obs": ls.l_obs,
        "e_int": e,
        "logged_action": float(ls.interventions[:, 0].mean()),
        "pass_condition": "L_obs = 0 and E_int = 0 in every block, exactly",
        "pass": ls.l_obs == 0.0 and all(v == 0.0 for v in e.values()),
    }


def check_copy(env: dict) -> dict:
    ctx, val, iset = env["ctx"], env["val"], env["iset"]
    pred = S.predict_baseline(B.copy, val, iset)
    ls = S.losses(ctx.scaler, pred, ctx.val_y, iset)
    scorer = {"l_obs": ls.l_obs}
    direct = {"l_obs": S.copy_std_mse_direct(ctx.scaler, ctx.val_y)}
    with np.load(SMOKE / "states.npz") as f:
        stored_target = f["target"]  # read again, straight from the file
    for b in S.BLOCKS:
        m = iset.mask(b)
        scorer[b] = float(ls.interventions[m].mean())
        direct[b] = S.copy_std_mse_direct(ctx.scaler, stored_target[m])
    diff = max(abs(scorer[k] - direct[k]) for k in scorer)
    floors = np.asarray(c.TARGET_STD_FLOORS)
    return {
        "scorer": scorer,
        "direct_from_box_motion": direct,
        "max_abs_diff": diff,
        "scaler_scale": ctx.scaler.scale.tolist(),
        "scaler_floors": floors.tolist(),
        "floors_respected": bool(np.all(ctx.scaler.scale >= floors)),
        "floor_binding": (ctx.scaler.scale == floors).tolist(),
        "scaler_from": "training split only (the coverage pool stands in for it in Phase 0)",
        "pass_condition": "scorer equals the direct computation to 1e-12 in L_obs and every block; "
        "scales >= floors",
        "pass": bool(diff < 1e-12 and np.all(ctx.scaler.scale >= floors)),
    }


def check_untrained(
    env: dict, archs=("mlp", "gru", "mlp-ms", "contact_wm"), seeds=(11, 22, 33)
) -> tuple[dict, dict]:
    """Untrained networks saved with trained: false, loaded through the scoring entry point. Only
    their L_obs is scored (G0.2 can fail). Their predictions at the 25 actions of every smoke state
    are made, to exercise the scorer's model path (histories, actions, box target), and checked for
    shape and finiteness only: no intervention error of any untrained network is computed, since
    `contact_wm`'s hand-written law makes even its untrained push error informative."""
    ctx, val, iset = env["ctx"], env["val"], env["iset"]
    copy_l = S.copy_std_mse_direct(ctx.scaler, ctx.val_y)
    out = {}
    for arch in archs:
        rows = []
        for seed in seeds:
            path = DATA / "untrained" / f"{arch}_medium_{seed}.pt"
            assert FORBIDDEN not in path.parents
            SI.save_untrained(path, arch, "medium", seed, env["norm"], env["dscaler"], ctx.scaler)
            model, meta, dsc = S.load_checkpoint_for_scoring(path)
            pred = S.predict_model(model, dsc, val, iset, meta)
            rows.append(
                {
                    "seed": seed,
                    "l_obs": ctx.scaler.std_mse(pred.val, ctx.val_y),
                    "intervention_predictions_shape_ok": pred.interventions.shape == iset.y_true.shape,
                    "intervention_predictions_finite": bool(np.all(np.isfinite(pred.interventions))),
                    "params": meta["params"],
                }
            )
        mean_l = float(np.mean([r["l_obs"] for r in rows]))
        beats = bool(mean_l < copy_l)  # one checkpoint each: matched and best are the same
        out[arch] = {
            "runs": rows,
            "seed_mean_l_obs": mean_l,
            "beats_copy": {"matched": beats, "best": beats, "copy": copy_l},
        }
    mlp_fails = not out["mlp"]["beats_copy"]["matched"]
    paths_ok = all(
        r["intervention_predictions_shape_ok"] and r["intervention_predictions_finite"]
        for a in out.values()
        for r in a["runs"]
    )
    return (
        {
            "copy_l_obs": copy_l,
            "results": out,
            "g0_2_with_untrained_mlp_as_main_model": "fail" if mlp_fails else "pass",
            "note": "only L_obs is scored; predictions at the smoke interventions are made to exercise the "
            "model path and checked for shape and finiteness, never scored",
            "pass_condition": "untrained mlp does not beat copy on L_obs, so G0.2 can fail; model path runs",
            "pass": mlp_fails and paths_ok,
        },
        {},
    )


def known_gap(sa: SI.StandIn, ma: G.Match, sb: SI.StandIn, mb: G.Match) -> float:
    """Expected E_int(b) - E_int(a) at the matched checkpoints, over draws of bias and noise."""
    ea = np.mean([sa.expected_e_push(i, int(ma.idx[i])) for i in np.flatnonzero(ma.keep)])
    eb = np.mean([sb.expected_e_push(i, int(mb.idx[i])) for i in np.flatnonzero(mb.keep)])
    return float(eb - ea)


def check_noisy_pair(env: dict, name_b: str, ratio_b: float, key: int, expect_pass: bool) -> dict:
    sa = spec(env, "noisy-a", key, "mlp", 1.0)
    sb = spec(env, name_b, key + 1, "mlp", ratio_b)
    fams = {"noisy-a": fam(env, sa), name_b: fam(env, sb)}
    v, setter = G.v_star_of(list(fams.values()))
    g1 = G.run_g1(fams, ["noisy-a", name_b], v, rng(key))
    p = g1["pairs"][0]
    reading, _ = G.g1_reading(g1["pairs"], {}, positive_pass=True)
    kg = known_gap(sa, g1["matches"]["noisy-a"], sb, g1["matches"][name_b])
    g = p["gap"]
    sign = 1.0 if p["challenger"] == name_b else -1.0
    covers = g["ci_low"] <= sign * kg <= g["ci_high"]
    out = {
        "v_star": v,
        "pair": RF.pair_entry(p),
        "meets": p["meets"],
        "reading_with_positive_control_passed": reading,
        "known_gap": kg,
        "known_rel_gap": SI.rel_gap_for_ratio(ratio_b, sa.bias_share),
        "interval_covers_known_gap": bool(covers),
        "interval_covers_zero": bool(g["ci_low"] <= 0 <= g["ci_high"]),
    }
    if expect_pass:
        out["pass_condition"] = (
            "G1 pass; the interval is reported (its coverage is checked on the repeated noisy-b and noisy-c)"
        )
        out["pass"] = bool(reading == "pass")
    else:
        out["pass_condition"] = (
            "G1 fail; the interval is reported (its calibration is checked on the repeated twins)"
        )
        out["pass"] = bool(reading == "fail")
    return out


def exp_curves(updates: list[int], floors: np.ndarray, start: float, tau: float) -> np.ndarray:
    u = np.asarray(updates, float)
    return floors[:, None] + (start - floors[:, None]) * np.exp(-u[None, :] / tau)


def check_matching(env: dict) -> dict:
    """Synthetic checkpoint series with known L_obs curves (no stand-in noise): expected V*, matched
    checkpoints and exclusions, the 6-of-10 rule, and the not testable paths."""
    upd = env["schedule"]
    u = np.asarray(upd, float)
    taus = {"a": 300.0, "b": 600.0, "c": 2000.0, "d": 500.0}
    floors = {"a": 0.020, "b": 0.050, "c": 0.075, "d": 0.060}
    seeds_f = np.linspace(0.97, 1.03, N_SEEDS)
    res: dict = {}

    def build(floor_over: dict | None = None) -> dict[str, G.Family]:
        fams = {}
        for n in taus:
            fl = floors[n] * seeds_f
            if floor_over and n in floor_over:
                fl = floor_over[n]
            fams[n] = SI.synthetic_family(n, upd, exp_curves(upd, fl, 1.0, taus[n]))
        return fams

    # (a) all reach V*: matched checkpoints against the analytic crossing.
    fams = build()
    v, setter = G.v_star_of(list(fams.values()))
    v_expected = c.MATCH_TARGET_FACTOR * float(
        np.median(floors["c"] * seeds_f + (1 - floors["c"] * seeds_f) * np.exp(-u[-1] / taus["c"]))
    )
    wrong = []
    for n, f in fams.items():
        m = G.match(f, v)
        fl = floors[n] * seeds_f
        for i in range(N_SEEDS):
            ustar = -taus[n] * np.log((v - fl[i]) / (1.0 - fl[i]))
            j = np.searchsorted(u, ustar)  # the two schedule checkpoints around the crossing
            cand = [k for k in (j - 1, j) if 0 <= k < len(u)]
            exp_idx = min(cand, key=lambda k: abs(f.l_obs[i, k] - v))
            if m.idx[i] != exp_idx or m.excluded[i]:
                wrong.append([n, i, int(m.idx[i]), int(exp_idx)])
    res["a_all_reach"] = {
        "v_star": v,
        "v_star_expected": v_expected,
        "set_by": setter,
        "matched_updates": {n: [upd[i] for i in G.match(f, v).idx] for n, f in fams.items()},
        "mismatches": wrong,
        "pass": bool(np.isclose(v, v_expected) and setter == "c" and not wrong),
    }
    # (b) one run never comes within 5% of V*: exactly that run is excluded.
    fl_b = floors["b"] * seeds_f
    fl_b = fl_b.copy()
    fams = build()
    v, _ = G.v_star_of(list(fams.values()))
    fl_b[3] = 1.08 * v
    fams = build({"b": fl_b})
    v2, _ = G.v_star_of(list(fams.values()))
    mb = G.match(fams["b"], v2)
    res["b_one_excluded"] = {
        "v_star": v2,
        "excluded_seeds": np.flatnonzero(mb.excluded).tolist(),
        "dropped": mb.dropped,
        "pass": bool(np.flatnonzero(mb.excluded).tolist() == [3] and not mb.dropped),
    }
    # (c) five of ten runs never reach V*: the architecture is dropped from G1, its pairs untested.
    fams = build()
    v, _ = G.v_star_of(list(fams.values()))
    fl_d = floors["d"] * seeds_f
    fl_d[:N_DROP] = 1.08 * v
    fams = build({"d": fl_d})
    v3, _ = G.v_star_of(list(fams.values()))
    g1 = G.run_g1(fams, list(taus), v3, rng(31), reps=200)
    md = g1["matches"]["d"]
    untested = [(p["challenger"], p["reference"]) for p in g1["pairs"] if not p["tested"]]
    res["c_dropped"] = {
        "v_star": v3,
        "d_excluded": int(md.excluded.sum()),
        "d_dropped": md.dropped,
        "untested_pairs": untested,
        "holm_m": g1["holm_m"],
        "pass": bool(md.dropped and len(untested) == 3 and all("d" in p for p in untested)),
    }
    # (d) not testable: three architectures dropped, so no pair is left.
    fams = build()
    v, _ = G.v_star_of(list(fams.values()))
    over = {}
    for n in ("a", "b", "d"):
        fl = floors[n] * seeds_f
        fl[:N_DROP] = 1.08 * v
        over[n] = fl
    fams = build(over)
    v4, _ = G.v_star_of(list(fams.values()))
    g1 = G.run_g1(fams, list(taus), v4, rng(32), reps=200)
    r4, _ = G.g1_reading(g1["pairs"], {}, True)
    res["d_not_testable_dropped"] = {
        "dropped": [n for n, m in g1["matches"].items() if m.dropped],
        "reading": r4,
        "pass": r4 == "not_testable",
    }
    # (e) not testable: two architectures whose curves jump across V*, so their matched L_obs differ by
    # more than 5% (each run within 5% of V*, none excluded).
    v = 0.1
    hi = np.tile(np.where(u < 5_000, 0.3, 1.045 * v), (N_SEEDS, 1))
    hi[:, -1] = 1.045 * v
    lo = np.tile(np.where(u < 5_000, 0.3, 0.955 * v), (N_SEEDS, 1))
    lo[:, :20] = 0.5
    fams = {"hi": SI.synthetic_family("hi", upd, hi), "lo": SI.synthetic_family("lo", upd, lo)}
    g1 = G.run_g1(fams, ["hi", "lo"], v, rng(33), reps=200)
    r5, _ = G.g1_reading(g1["pairs"], {}, True)
    res["e_not_testable_screen"] = {
        "matched_l": {n: m.mean_l for n, m in g1["matches"].items()},
        "excluded": {n: int(m.excluded.sum()) for n, m in g1["matches"].items()},
        "rel_diff": g1["pairs"][0].get("obs_val_rel_diff"),
        "reading": r5,
        "pass": r5 == "not_testable" and not any(m.excluded.any() for m in g1["matches"].values()),
    }
    return {
        "schedule": f"design schedule to {PLACEHOLDER_UPDATES} updates ({len(upd)} checkpoints)",
        "curves": "L(u) = floor + (1 - floor) exp(-u / tau); floors 0.97 to 1.03 over the seeds",
        "scenarios": res,
        "pass": all(r["pass"] for r in res.values()),
    }


EXPECTED_READING = {
    # (main pair separates, twin separates, positive separates) -> reading
    (True, False, True): "pass",
    (True, False, False): "pass",
    (True, True, True): "inconclusive_not_specific",
    (True, True, False): "inconclusive_not_specific",
    (False, False, True): "fail",
    (False, True, True): "fail",
    (False, False, False): "inconclusive_not_sensitive",
    (False, True, False): "inconclusive_not_sensitive",
}


def verdict_table() -> dict:
    """The verdict logic alone, on hand-made pair results: every combination, plus not testable and a
    twin pair that is not run."""
    rows, ok = [], True
    for (main_sep, twin_sep, pos), want in EXPECTED_READING.items():
        pairs = [
            {
                "tested": True,
                "challenger": "gru",
                "reference": "mlp",
                "matched_within_5pct": True,
                "meets": main_sep,
            }
        ]
        twins = {"mlp": "clean", "gru": "separated" if twin_sep else "clean"}
        got, passing = G.g1_reading(pairs, twins, pos)
        rows.append({"main": main_sep, "twin": twin_sep, "positive": pos, "reading": got, "expected": want})
        ok &= got == want
    extra = {
        "no_pair_screened": G.g1_reading(
            [
                {
                    "tested": True,
                    "challenger": "gru",
                    "reference": "mlp",
                    "matched_within_5pct": False,
                    "meets": True,
                }
            ],
            {},
            True,
        )[0],
        "twin_not_run": G.g1_reading(
            [
                {
                    "tested": True,
                    "challenger": "gru",
                    "reference": "mlp",
                    "matched_within_5pct": True,
                    "meets": True,
                }
            ],
            {"gru": "not_run", "mlp": "clean"},
            False,
        )[0],
    }
    ok &= extra == {"no_pair_screened": "not_testable", "twin_not_run": "pass"}
    return {"combinations": rows, "other": extra, "pass": bool(ok)}


def check_control_readings(env: dict, reps: int) -> dict:
    """End to end with stand-ins: main pair mlp/gru, twin pairs, positive control mlp/mlp-shortcut, each
    set to separate or not, through matching, bootstrap and the verdict."""
    r15 = c.SMOKE_PLANTED_NOISE_RATIO
    rows, ok = [], True
    shared = {
        "mlp": fam(env, spec(env, "mlp", 301, "mlp", 1.0)),
        "mlp-twin": fam(env, spec(env, "mlp-twin", 302, "mlp", 1.0, seeds=c.STANDIN_TWIN_SEEDS)),
    }
    cache: dict = {}

    def get(name, key, curve, ratio, seeds=c.STANDIN_SEEDS):
        k = (name, ratio)
        if k not in cache:
            cache[k] = fam(env, spec(env, name, key, curve, ratio, seeds=seeds))
        return cache[k]

    for (main_sep, twin_sep, pos), want in EXPECTED_READING.items():
        gru_r = r15 if main_sep else 1.0
        twin_r = (1.0 if main_sep else r15) if twin_sep else gru_r
        fams = dict(shared)
        fams["gru"] = get("gru", 310 if main_sep else 311, "gru", gru_r)
        fams["gru-twin"] = get("gru-twin", 320 if twin_r != 1.0 else 321, "gru", twin_r, c.STANDIN_TWIN_SEEDS)
        fams["mlp-shortcut"] = get("mlp-shortcut", 330 if pos else 331, "mlp", r15 if pos else 1.0)
        v, _ = G.v_star_of([fams["mlp"], fams["gru"]])
        g = rng(40, len(rows))
        twins = {}
        for a in ("mlp", "gru"):
            _, twins[a] = G.twin_pair(fams, a, f"{a}-twin", v, g, reps)
        pp = G.positive_pair(fams, "mlp", "mlp-shortcut", v, g, reps)
        g1 = G.run_g1(fams, ["mlp", "gru"], v, g, reps)
        got, _ = G.g1_reading(g1["pairs"], twins, pp["separates"])
        rows.append(
            {
                "set_to": {"main": main_sep, "twin": twin_sep, "positive": pos},
                "observed": {
                    "main_meets": g1["pairs"][0]["meets"],
                    "twins": twins,
                    "positive_separates": pp["separates"],
                },
                "reading": got,
                "expected": want,
            }
        )
        ok &= got == want
    # not testable end to end: gru with too few seeds reaching V*.
    sp = spec(env, "gru", 340, "gru", 1.0)
    v_guess = c.MATCH_TARGET_FACTOR * float(np.median(shared["mlp"].best_l()))
    sp.l_target[:N_DROP] = np.maximum(sp.l_target[:N_DROP], 2.0 * v_guess)
    fams = {"mlp": shared["mlp"], "gru": fam(env, sp)}
    v, _ = G.v_star_of(list(fams.values()))
    g1 = G.run_g1(fams, ["mlp", "gru"], v, rng(41), reps)
    nt, _ = G.g1_reading(g1["pairs"], {}, True)
    # twin not run: mlp-twin with too few seeds reaching V*.
    st = spec(env, "mlp-twin", 350, "mlp", 1.0, seeds=c.STANDIN_TWIN_SEEDS)
    st.l_target[:N_DROP] = np.maximum(st.l_target[:N_DROP], 2.0 * v_guess)
    fams = {"mlp": shared["mlp"], "mlp-twin": fam(env, st)}
    _, twin_st = G.twin_pair(fams, "mlp", "mlp-twin", G.v_star_of([shared["mlp"]])[0], rng(42), reps)
    extra = {
        "not_testable": {"gru_dropped": g1["matches"]["gru"].dropped, "reading": nt},
        "twin_not_run": twin_st,
    }
    ok &= nt == "not_testable" and twin_st == "not_run"
    return {
        "setting": "separating member has push noise ratio 1.5; others 1.0; bias share for 10% seed CV",
        "combinations": rows,
        "other": extra,
        "pass": bool(ok),
    }


def ramp(lo_level: float, hi_level: float, r_lo: float, r_hi: float):
    def r(level: float) -> float:
        t = np.clip((level - lo_level) / (hi_level - lo_level), 0.0, 1.0)
        return float(r_lo + t * (r_hi - r_lo))

    return r


GAT_DRAWS = 10  # independent draws of B per Gap along training scenario


def check_gap_along_training(env: dict, reps: int, draws: int = GAT_DRAWS) -> dict:
    """Synthetic checkpoint series giving each Gap along training reading, and matched_only. Each
    scenario is drawn several times (fresh B and bootstrap), since a reading built on intervals comes
    out as designed only most of the time; the check passes if each scenario's designed reading is
    its most frequent one."""
    r15 = c.SMOKE_PLANTED_NOISE_RATIO
    a = fam(env, spec(env, "A", 401, "mlp", 1.0))
    setter = fam(env, spec(env, "C", 403, "mlp-ms", 1.0))
    v, set_by = G.v_star_of([a, setter])
    vp = c.MATCH_TARGET_FACTOR * float(np.median(a.best_l()))  # V*_pair, expected (B shares A's shape)
    lo, hi = 1.10 * vp, 0.93 * v
    scenarios = {
        "overlap": 1.0,
        "persistent": r15,
        "closing": ramp(lo, hi, 1.0, r15),
        "mixed": ramp(lo, hi, 1.0 / r15, r15),
    }
    out, ok = {}, True
    for i, (name, ratio) in enumerate(scenarios.items()):
        readings, matched_only, example = [], [], None
        for d in range(draws):
            b = fam(env, spec(env, "B", 1_000 + 100 * i + d, "mlp", ratio))
            fams = {"A": a, "B": b, "C": setter}
            g = rng(50, i, d)
            g1 = G.run_g1(fams, ["A", "B", "C"], v, g, reps)
            p = next(p for p in g1["pairs"] if {p["challenger"], p["reference"]} == {"A", "B"})
            orient = ("B", "A") if name != "overlap" else (p["challenger"], p["reference"])
            best = G.compare(
                a, G.best_match(a), b, G.best_match(b), g, reps, states_only=False, orient=orient
            )
            gat = G.gap_along_training(fams[orient[0]], fams[orient[1]], v, orient, g, reps)
            readings.append(gat["reading"])
            matched_only.append(
                bool(p["meets"] and p["matched_within_5pct"] and not G.excludes_zero(best["gap"]))
            )
            if example is None:
                example = {
                    "v_star_pair": gat["v_star_pair"],
                    "levels": gat["levels"],
                    "g1_at_v_star_meets": p["meets"],
                    "best_push_gap": RF._gap(best["gap"]),
                }
        rates = {r: readings.count(r) / draws for r in G.CURVE_READINGS}
        modal = max(rates, key=rates.get)
        out[name] = {
            "reading_rates": rates,
            "modal_reading": modal,
            "matched_only_rate": float(np.mean(matched_only)),
            "first_draw": example,
        }
        ok &= modal == name
    ok &= out["closing"]["matched_only_rate"] > 0.5 and out["persistent"]["matched_only_rate"] < 0.5
    return {
        "v_star": v,
        "v_star_set_by": set_by,
        "draws_per_scenario": draws,
        "setting": "A and B share mlp's curve shape, C (mlp-ms's curve) sets V*; B's push noise ratio as a "
        "function of the target L_obs: 1 (overlap), 1.5 (persistent), 1 at 1.1 x V*_pair rising to 1.5 at "
        "0.93 x V* (closing), 1/1.5 rising to 1.5 over the same range (mixed)",
        "scenarios": out,
        "pass_condition": "each scenario's designed reading is its most frequent; "
        "matched_only in most closing "
        "draws and in few persistent draws",
        "pass": bool(ok),
    }


def check_g2_readings(reps: int) -> dict:
    """G2's reading on made-up run values whose rank correlation is known."""
    g = rng(60)
    x = np.sort(g.uniform(0.05, 0.1, 80))
    cases = {
        "refuted": x * 3 + 1e-4 * g.standard_normal(80),
        "decoupled": g.permutation(x),
    }
    out = {}
    for name, y in cases.items():
        s = G.spearman_boot(x, y, rng(61), reps)
        out[name] = {**s, "reading": G.g2_reading(s["rho"])}
    mid = G.g2_reading(0.65)
    return {
        "cases": out,
        "rho_0_65": mid,
        "pass": out["refuted"]["reading"] == "refuted"
        and out["decoupled"]["reading"] == "decoupled"
        and mid == "inconclusive",
    }


# Repeated pairs --------------------------------------------------------------------------------------


def one_level_spec(
    env: dict, name: str, key: int, ratio: float, f: float, seeds=c.STANDIN_SEEDS
) -> SI.StandIn:
    """A stand-in with one checkpoint at the V*-like level of milestone 2 (0.085), for repeated pairs."""
    level = 0.085
    l_target = SI.seed_curves(np.array([level]), len(seeds), key)
    return SI.StandIn(name, key, tuple(seeds), [1], l_target, f, ratio)


def repeated(env: dict, ratio_b: float, f: float, n: int, reps: int, key0: int) -> dict:
    """n fresh draws of noisy-a against a stand-in with push noise ratio ratio_b; the full procedure
    and the states-only interval."""
    sep_full, sep_states, cover, rel, excl_full, excl_states = 0, 0, 0, [], 0, 0
    for d in range(n):
        sa = one_level_spec(env, "a", key0 + 2 * d, 1.0, f)
        sb = one_level_spec(env, "b", key0 + 2 * d + 1, ratio_b, f)
        fa, fb = fam(env, sa), fam(env, sb)
        ma = G.Match("a", 0.085, np.zeros(N_SEEDS, int), fa.l_obs[:, 0], np.zeros(N_SEEDS, bool), 1)
        mb = G.Match("b", 0.085, np.zeros(N_SEEDS, int), fb.l_obs[:, 0], np.zeros(N_SEEDS, bool), 1)
        p = G.compare(fa, ma, fb, mb, rng(70, key0, d), reps)
        sep_full += G.control_separates(p)
        q = dict(p)
        q["gap"] = p["gap_states_only"]
        sep_states += G.control_separates(q)
        excl_full += G.excludes_zero(p["gap"])
        excl_states += G.excludes_zero(p["gap_states_only"])
        kg = known_gap(sa, ma, sb, mb)
        sign = 1.0 if p["challenger"] == "b" else -1.0
        cover += p["gap"]["ci_low"] <= sign * kg <= p["gap"]["ci_high"]
        e_b, e_a = (
            (p["e_challenger"], p["e_reference"]) if sign > 0 else (p["e_reference"], p["e_challenger"])
        )
        rel.append((e_b - e_a) / e_a)
    return {
        "n": n,
        "rate_full": sep_full / n,
        "rate_states_only": sep_states / n,
        "interval_excludes_zero_full": excl_full / n,
        "interval_excludes_zero_states_only": excl_states / n,
        "known_gap_coverage_full": cover / n,
        "rel_gap_true": SI.rel_gap_for_ratio(ratio_b, f),
        "rel_gap_mean_estimated": float(np.mean(rel)),
    }


def check_repeated(env: dict, n: int, reps: int) -> dict:
    f = SI.bias_share_for_cv(c.STANDIN_PUSH_SEED_CV)
    r_c = SI.ratio_for_rel_gap(c.SMOKE_NEAR_THRESHOLD_REL_GAP, f)
    t0 = time.perf_counter()
    twins = repeated(env, 1.0, f, n, reps, 10_000)
    near = repeated(env, r_c, f, n, reps, 20_000)
    big = repeated(env, c.SMOKE_PLANTED_NOISE_RATIO, f, n, reps, 30_000)
    sweep = {}
    for i, cv in enumerate(c.STANDIN_FALSE_ALARM_SEED_CVS):
        fi = SI.bias_share_for_cv(cv)
        sweep[f"{cv:g}"] = {"bias_share": fi, **repeated(env, 1.0, fi, n, reps, 40_000 + 1_000 * i)}
    secs = time.perf_counter() - t0
    return {
        "draws_per_pair": n,
        "bootstrap_replicates": reps,
        "seed_cv": c.STANDIN_PUSH_SEED_CV,
        "bias_share": f,
        "noisy_c_ratio": r_c,
        "twins": {
            **twins,
            "pass_condition": "full procedure: separates in at most 5% of draws, and its interval excludes "
            "zero in at most 10%",
            "pass": twins["rate_full"] <= c.SMOKE_MAX_FALSE_ALARM_RATE
            and twins["interval_excludes_zero_full"] <= c.SMOKE_MAX_TWIN_INTERVAL_EXCLUDES_ZERO,
        },
        "noisy_c": {
            **near,
            "pass_condition": "the interval covers the known gap in at least 90% of draws; the separation "
            "rate is reported, not gated",
            "pass": near["known_gap_coverage_full"] >= c.SMOKE_MIN_KNOWN_GAP_COVERAGE,
        },
        "noisy_b": {
            **big,
            "pass_condition": "separates in at least 95% of draws, and the interval covers the known gap in "
            "at least 90%",
            "pass": big["rate_full"] >= c.SMOKE_MIN_DETECTION_RATE
            and big["known_gap_coverage_full"] >= c.SMOKE_MIN_KNOWN_GAP_COVERAGE,
        },
        "false_alarm_by_seed_cv": sweep,
        "false_alarm_all_cvs_at_most_5pct": all(
            v["rate_full"] <= c.SMOKE_MAX_FALSE_ALARM_RATE for v in sweep.values()
        ),
        "wall_seconds": secs,
        "pass": bool(
            twins["rate_full"] <= c.SMOKE_MAX_FALSE_ALARM_RATE
            and twins["interval_excludes_zero_full"] <= c.SMOKE_MAX_TWIN_INTERVAL_EXCLUDES_ZERO
            and big["rate_full"] >= c.SMOKE_MIN_DETECTION_RATE
            and big["known_gap_coverage_full"] >= c.SMOKE_MIN_KNOWN_GAP_COVERAGE
            and near["known_gap_coverage_full"] >= c.SMOKE_MIN_KNOWN_GAP_COVERAGE
        ),
    }


# The full stand-in run ---------------------------------------------------------------------------


def standin_run(env: dict, reps: int) -> tuple[dict, dict, dict]:
    ctx, iset = env["ctx"], env["iset"]
    fams = {}
    for name, (key, curve, ratio) in LINEUP.items():
        seeds = c.STANDIN_TWIN_SEEDS if name.endswith("-twin") else c.STANDIN_SEEDS
        obs = BLIND_OBS_FACTOR if name == "blind" else 1.0
        fams[name] = fam(env, spec(env, name, key, curve, ratio, seeds=seeds, obs_factor=obs))
    widths: dict = {}
    for wi, w in enumerate(("small", "large")):
        widths[w] = {}
        for ai, a in enumerate(MAIN):
            _, curve, ratio = LINEUP[a]
            widths[w][a] = fam(
                env, spec(env, a, 200 + 10 * wi + ai, curve, ratio, seeds=c.SECONDARY_WIDTH_SEEDS, width=w)
            )
    m1 = json.loads((M1 / "env_checks.json").read_text())["checks"]["bit_exact_replay"]
    amd = json.loads((M1 / "amd64_replay.json").read_text())["replay_check"]
    g0_replay = {
        "passed": bool(m1["passed"] and amd["passed"]),
        "source": "milestone 1 attempt 2: smoke states, all 25 actions, fresh process in the pinned image "
        "(docker) and on real amd64 hardware",
        "n_compared": amd["n_compared"],
        "n_mismatched": amd["n_mismatched"],
    }
    lab = iset.label[:, 1:]
    counts = {
        "n_train_obs": len(env["train"]),
        "n_val_obs": len(env["val"]),
        "n_interventions_total": int(lab.size),
        "n_held_out_checkpoints": iset.n_states,
        "regime_counts": {s: int((lab == i).sum()) for i, s in enumerate(S.STRATA)},
        "n_states_with_push": int((iset.counts("push") > 0).sum()),
    }
    base = {
        "copy": S.predict_baseline(B.copy, env["val"], iset),
        "constvel": S.predict_baseline(B.constvel, env["val"], iset),
        "knn": None,
    }
    t0 = time.perf_counter()
    primary = RF.build_primary(
        fams,
        widths,
        ctx.scaler,
        iset,
        env["val"].s,
        ctx.val_y,
        ctx.val_episode,
        base,
        counts,
        g0_replay,
        rng(80),
        PLACEHOLDER_UPDATES,
        reps,
    )
    result = {
        "format_version": RF.FORMAT_VERSION,
        "benchmark": "physical-intervene/pushing",
        "device": "cpu",
        "seeds": list(c.STANDIN_SEEDS),
        "updates": PLACEHOLDER_UPDATES,
        "wall_seconds": time.perf_counter() - t0,
        "phase": "phase0",
        "stand_in_run": "Phase 0 stand-ins only (SMOKE_TEST.md, No-leak rule); "
        "no trained model is in this file. "
        "Model names are slots filled by stand-ins (see lineup); knn is not scored in Phase 0",
        "lineup": {
            n: {"key": k, "curve_shape_from": cv, "push_noise_ratio": r, "trained": False}
            for n, (k, cv, r) in LINEUP.items()
        },
        "primary": primary,
    }
    return result, fams, widths


def plot_gap_along_training(fams: dict, path: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for a in MAIN:
        f = fams[a]
        e = np.array([f.run_e("push", np.full(f.n_seeds, k)) for k in range(len(f.updates))]).mean(1)
        lo = f.l_obs.mean(0)
        for x in ax:
            x.plot(lo, e, label=a, lw=1)
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    for x in ax:
        x.set_xlabel("seed-mean L_obs (stand-in)")
        x.set_ylabel("seed-mean E_int, push (stand-in)")
        x.legend()
    fig.suptitle("Stand-in run: E_int against L_obs at every checkpoint (stand-ins, not trained models)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _json_default(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, G.Match):
        return {"idx": x.idx.tolist(), "excluded": x.excluded.tolist()}
    if callable(x):
        return "function of target L_obs"
    raise TypeError(type(x))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--quick", action="store_true", help="fewer replicates and draws, for a smoke run of the script"
    )
    args = ap.parse_args()
    reps = 500 if args.quick else c.BOOTSTRAP_REPLICATES
    n_rep = 20 if args.quick else c.SMOKE_REPEATED_PAIRS
    out_dir = OUT / "quick" if args.quick else OUT
    t_all = time.perf_counter()
    env, t_setup = timed(setup)
    print(f"setup {t_setup:.1f} s", flush=True)

    checks, times = {}, {"setup": t_setup}
    for name, fn in (
        ("no_leak", lambda: check_no_leak()),
        ("oracle", lambda: check_oracle(env)),
        ("copy", lambda: check_copy(env)),
        ("noisy_a_vs_a_prime", lambda: check_noisy_pair(env, "noisy-a'", 1.0, 500, expect_pass=False)),
        (
            "noisy_a_vs_b",
            lambda: check_noisy_pair(env, "noisy-b", c.SMOKE_PLANTED_NOISE_RATIO, 510, expect_pass=True),
        ),
        ("matching", lambda: check_matching(env)),
        ("verdict_logic", lambda: verdict_table()),
        ("control_readings", lambda: check_control_readings(env, reps)),
        ("gap_along_training", lambda: check_gap_along_training(env, reps)),
        ("g2_readings", lambda: check_g2_readings(reps)),
    ):
        checks[name], times[name] = timed(fn)
        print(name, checks[name]["pass"], f"{times[name]:.1f} s", flush=True)
    (checks["untrained_network"], _), times["untrained_network"] = timed(check_untrained, env)
    print("untrained_network", checks["untrained_network"]["pass"], flush=True)

    (result, fams, widths), times["standin_run"] = timed(standin_run, env, reps)
    gates = result["primary"]["gates"]
    expected = {
        "gate_g0_valid": True,
        "gate_g1_reading": "pass",
        "gate_positive_control_pass": True,
        "gate_g1_passing_pairs": sorted([["mlp-ms", a] for a in ("mlp", "gru", "contact_wm")]),
    }
    observed = {
        "gate_g0_valid": gates["gate_g0_valid"],
        "gate_g1_reading": gates["gate_g1_reading"],
        "gate_positive_control_pass": gates["gate_positive_control_pass"],
        "gate_g1_passing_pairs": sorted(gates["gate_g1_passing_pairs"]),
    }
    checks["standin_run"] = {
        "expected": expected,
        "observed": observed,
        "negative_controls": gates["gate_negative_controls"],
        "g2_reading": gates["gate_g2_reading"],
        "g2_general_purpose_reading": result["primary"]["rank_correlation_gate_g2_general_purpose"][
            "gate_g2_reading"
        ],
        "without_v_star_setter": {
            k: result["primary"]["pairwise_gate_g1_without_v_star_setter"][k]
            for k in ("left_out", "v_star", "gate_g1_reading", "gate_g1_passing_pairs")
        },
        "v_star": result["primary"]["matching"]["v_star"],
        "v_star_set_by": result["primary"]["matching"]["set_by"],
        "g0": {
            "replay": result["primary"]["g0"]["replay"]["passed"],
            "beats_copy": result["primary"]["g0"]["beats_copy"],
            "blind_gap": result["primary"]["g0"]["blind"]["matched_push_gap"],
            "blind_pass": result["primary"]["g0"]["blind"]["pass"],
        },
        "note": "twin pairs are expected clean with probability about 1 - the false-alarm rate each",
        "pass_condition": "G0 valid, G1 pass with exactly the three mlp-ms pairs passing, "
        "positive control passes",
        "pass": expected == observed,
    }
    print("standin_run", checks["standin_run"]["pass"], flush=True)
    errors = RF.check_conformance(result)
    checks["format"] = {
        "errors": errors,
        "pass_condition": "the stand-in run's result JSON has every key of INTERVENTION_REPORTING.md and "
        "RESULTS_FORMAT.md, allowed reading values, and readings consistent with their keys",
        "pass": not errors,
    }
    print("format", not errors, errors[:5], flush=True)
    plot_gap_along_training(fams, out_dir / "plots" / "standin_e_int_vs_l_obs.png")

    checks["repeated_pairs"], times["repeated_pairs"] = timed(check_repeated, env, n_rep, reps)
    print("repeated_pairs", checks["repeated_pairs"]["pass"], flush=True)

    f = SI.bias_share_for_cv(c.STANDIN_PUSH_SEED_CV)
    record = {
        "milestone": 3,
        "attempt": c.MILESTONE3_ATTEMPT,
        "phase": "phase0",
        "quick": args.quick,
        "models_scored_on_interventions": "stand-ins (oracle, copy, constvel as a fixed rule, "
        "noisy families) and "
        "untrained networks only; no trained checkpoint was opened",
        "inputs": env["files"],
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "standin_scales": {
            "obs_seed_cv": c.STANDIN_OBS_SEED_CV,
            "push_seed_cv": c.STANDIN_PUSH_SEED_CV,
            "push_bias_share": f,
            "push_over_obs": c.STANDIN_PUSH_OVER_OBS,
            "difficulty_share": c.STANDIN_DIFFICULTY_SHARE,
            "noisy_b_ratio": c.SMOKE_PLANTED_NOISE_RATIO,
            "noisy_b_rel_gap_expected": SI.rel_gap_for_ratio(c.SMOKE_PLANTED_NOISE_RATIO, f),
            "noisy_c_ratio": SI.ratio_for_rel_gap(c.SMOKE_NEAR_THRESHOLD_REL_GAP, f),
            "noisy_c_rel_gap_expected": c.SMOKE_NEAR_THRESHOLD_REL_GAP,
            "how_set": "obs_seed_cv from milestone 2's best L_obs over 3 medium seeds "
            "per architecture (0.8% to "
            "5.4%); push_seed_cv a guess at 3 times that, with the false-alarm rate also measured at "
            f"{list(c.STANDIN_FALSE_ALARM_SEED_CVS)}",
            "bootstrap_replicates": reps,
        },
        "checks": checks,
        "timings_seconds": times,
        "wall_seconds_total": time.perf_counter() - t_all,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "intervention_checks.json").write_text(json.dumps(record, indent=1, default=_json_default))
    (out_dir / "intervention_results_standin.json").write_text(
        json.dumps(result, indent=1, default=_json_default)
    )
    summary = {k: v["pass"] for k, v in checks.items()}
    print(json.dumps(summary, indent=1))
    print(f"total {record['wall_seconds_total']:.0f} s")
    return 0 if all(v is not False for v in summary.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
