"""Phase 0 exploration: how G1's interval behaves when seeds differ a lot on pushes, at the smoke
states' size and at about the test set's size, and two ways to widen its seed part. Stand-ins only.

    uv run --extra models python scripts/phase0_seed_interval.py [--workers 8] [--quick]

Writes results/phase0/milestone3/attemptM/seed_interval_checks.json. Not part of the design: the
design's interval is `design` below; the other two are candidates.

Interval methods, each run on the same draws:
- `design`: DESIGN.md's G1 interval, states (paired) and seeds resampled with replacement;
- `inflated`: as `design`, with each family's seed part of every replicate (its E* minus the
  replicate's E* with seed means fixed) multiplied by sqrt(n / (n - 1));
- `t`: the states-only replicate plus T x sqrt(s_a^2 / n_a + s_b^2 / n_b), with s the standard
  deviation (n - 1) of the per-seed E_int and T drawn from a t distribution with Welch-Satterthwaite
  degrees of freedom;
- `welch`: Welch's t-test on the per-seed E_int, test states held fixed (the usual practice when
  comparing trained models over seeds).

With --seeds-comparison: the same at the test set's size only, with 10 and with 20 seeds per
family (written to seed_interval_checks_seeds.json).

The stand-ins are those of the milestone 3 repeated pairs (constants STANDIN_*), drawn directly on
push transitions: per seed, an L_obs level 0.085 x a log-normal factor (CV STANDIN_OBS_SEED_CV), a
3-dimensional bias, and noise per transition, scaled by the same per-transition difficulty. A
family's realised L_obs is its level times a chi-square draw over the 5,000 x 3 validation errors,
which is its distribution under the stand-in model. "About 1,000 states": 1,000 draws with
replacement from the smoke states that have a push, each copy with its own noise (copies of a
state share its true outcome, so this approximates, not reproduces, 1,000 independent states).

Per setting (number of states, seed spread on push):
- null: 4 main families and 4 twins, no true difference. Twin false alarm (each twin pair, the
  control rule: unadjusted p, interval, 20%), the main pairs' false pass (6 pairs, Holm, G1's three
  conditions; then the G1 reading with the twin pairs), and how often an interval covers zero;
- power: as null, with one main family's push noise set for a true relative gap of 25% (noisy-c).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

from physical_intervene import constants as c
from physical_intervene import gates as G
from physical_intervene import standins as SI

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "phase0" / "milestone3" / f"attempt{c.MILESTONE3_ATTEMPT}"
METHODS = ("design", "inflated", "t", "welch")
MAIN = ("a1", "a2", "a3", "a4")
LEVEL = 0.085  # milestone 2's V*-like level
N_VAL_TERMS = 5_000 * 3
_PUSH: dict = {}


def push_setup() -> dict:
    """Push transitions of the smoke states: state index and difficulty (as standins.Context)."""
    if not _PUSH:
        sys.path.insert(0, str(ROOT / "scripts"))
        import phase0_intervention as P

        env = P.setup()
        ctx = env["ctx"]
        st, _ = np.nonzero(ctx.push)
        _PUSH.update(state=st, difficulty=ctx.difficulty[ctx.push], n_states=ctx.iset.n_states)
    return _PUSH


def layout(n_states: int, key: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Transition -> state index and difficulty for a set of n_states states (the smoke states, or
    draws with replacement from those with a push)."""
    p = push_setup()
    if n_states == p["n_states"]:
        return p["state"], p["difficulty"], n_states
    with_push = np.unique(p["state"])
    pick = SI.gen(key, 1).choice(with_push, size=n_states, replace=True)
    st, d = [], []
    for j, s in enumerate(pick):
        m = p["state"] == s
        st.append(np.full(m.sum(), j))
        d.append(p["difficulty"][m])
    return np.concatenate(st), np.concatenate(d), n_states


def family(g: np.random.Generator, state, diff, n_states, f: float, r: float, S: int = 10) -> dict:
    """Per-seed per-state push loss sums and the family's seed-mean L_obs."""
    level = LEVEL * np.exp(c.STANDIN_OBS_SEED_CV * g.standard_normal(S))
    b = g.standard_normal((S, 1, 3))
    eps = g.standard_normal((S, len(state), 3))
    scale = np.sqrt(c.STANDIN_PUSH_OVER_OBS * level[:, None] * diff[None, :])[..., None]
    err = scale * (np.sqrt(f) * b + np.sqrt(1 - f) * r * eps)
    loss = np.mean(err * err, axis=-1)  # (S, m)
    sums = np.zeros((S, n_states))
    for i in range(S):
        sums[i] = np.bincount(state, weights=loss[i], minlength=n_states)
    l_obs = level * g.chisquare(N_VAL_TERMS, S) / N_VAL_TERMS
    return {"sums": sums, "l_obs": float(l_obs.mean())}


def replicates(fams: dict, n: np.ndarray, reps: int, g: np.random.Generator) -> dict:
    """E* per family and method, one shared state draw W."""
    W = g.multinomial(len(n), np.full(len(n), 1.0 / len(n)), size=reps).astype(float)
    tot = W @ n
    out = {}
    for name, fm in fams.items():
        S = fm["sums"].shape[0]
        M = W @ fm["sums"].T  # (R, S)
        C = g.multinomial(S, np.full(S, 1.0 / S), size=reps).astype(float)
        e_states = M.mean(1) / tot
        e_full = (M * C).sum(1) / S / tot
        k = np.sqrt(S / (S - 1))
        per_seed = fm["sums"].sum(1) / n.sum()
        out[name] = {
            "states": e_states,
            "design": e_full,
            "inflated": e_states + k * (e_full - e_states),
            "e": float(per_seed.mean()),
            "var_mean": float(np.var(per_seed, ddof=1) / S),
            "per_seed": per_seed,
            "S": S,
        }
    return out


def gap(ra: dict, rb: dict, method: str, g: np.random.Generator) -> dict:
    """Gap test of a against b: challenger = the higher E. Returns p, interval, relative gap."""
    ch, ref = (ra, rb) if ra["e"] >= rb["e"] else (rb, ra)
    point = ch["e"] - ref["e"]
    if method == "welch":  # per-seed E_int, n against n, test states held fixed
        va, vb = ch["var_mean"], ref["var_mean"]
        df = (va + vb) ** 2 / (va**2 / (ch["S"] - 1) + vb**2 / (ref["S"] - 1))
        half = stats.t.ppf(0.975, df) * np.sqrt(va + vb)
        p = float(stats.ttest_ind(ch["per_seed"], ref["per_seed"], equal_var=False).pvalue)
        lo, hi = point - half, point + half
        return {
            "p": p,
            "excl": bool(lo > 0 or hi < 0),
            "rel": point / ref["e"],
            "lo": lo,
            "hi": hi,
            "point": point,
        }
    if method == "t":
        va, vb = ch["var_mean"], ref["var_mean"]
        df = (va + vb) ** 2 / (va**2 / (ch["S"] - 1) + vb**2 / (ref["S"] - 1))
        d = ch["states"] - ref["states"] + g.standard_t(df, len(ch["states"])) * np.sqrt(va + vb)
    else:
        d = ch[method] - ref[method]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {
        "p": G.p_two_sided(d),
        "excl": bool(lo > 0 or hi < 0),
        "rel": point / ref["e"],
        "lo": lo,
        "hi": hi,
        "point": point,
    }


def setting(job: dict) -> dict:
    n_states, cv, kind, draws, reps = job["n_states"], job["cv"], job["kind"], job["draws"], job["reps"]
    n_seeds = job.get("seeds", 10)
    t0 = time.perf_counter()
    state, diff, ns = layout(n_states, 7)
    n = np.bincount(state, minlength=ns).astype(float)
    f = SI.bias_share_for_cv(cv)
    r_c = SI.ratio_for_rel_gap(c.SMOKE_NEAR_THRESHOLD_REL_GAP, f)
    counts = {
        m: {
            "twin_sep": 0,
            "twin_excl": 0,
            "main_any_meets": 0,
            "reading_pass": 0,
            "pair_excl": 0,
            "planted_meets": 0,
        }
        for m in METHODS
    }
    for d in range(draws):
        g = SI.gen(job["key"], d)
        fams = {}
        for a in MAIN:
            r = r_c if (kind == "power" and a == "a4") else 1.0
            fams[a] = family(g, state, diff, ns, f, r, n_seeds)
            fams[a + "-twin"] = family(g, state, diff, ns, f, r, n_seeds)
        R = replicates(fams, n, reps, g)
        for m in METHODS:
            twin_status = {}
            for a in MAIN:
                t = gap(R[a], R[a + "-twin"], m, g)
                screen = G.passes_match_screen(fams[a]["l_obs"], fams[a + "-twin"]["l_obs"])
                sep = screen and t["p"] < c.G1_ALPHA and t["excl"] and t["rel"] >= c.G1_MIN_REL_GAP
                twin_status[a] = "separated" if sep else "clean"
                counts[m]["twin_sep"] += sep
                counts[m]["twin_excl"] += t["excl"]
            pairs = []
            for a, b in combinations(MAIN, 2):
                t = gap(R[a], R[b], m, g)
                ch, ref = (a, b) if R[a]["e"] >= R[b]["e"] else (b, a)
                pairs.append(
                    {
                        "tested": True,
                        "challenger": ch,
                        "reference": ref,
                        "t": t,
                        "matched_within_5pct": G.passes_match_screen(fams[a]["l_obs"], fams[b]["l_obs"]),
                    }
                )
                counts[m]["pair_excl"] += t["excl"]
            adj = G.holm([p["t"]["p"] for p in pairs])
            for p, h in zip(pairs, adj, strict=True):
                p["meets"] = bool(h < c.G1_ALPHA and p["t"]["excl"] and p["t"]["rel"] >= c.G1_MIN_REL_GAP)
            meeting = [p for p in pairs if p["meets"] and p["matched_within_5pct"]]
            counts[m]["main_any_meets"] += bool(meeting)
            counts[m]["planted_meets"] += any(p["challenger"] == "a4" for p in meeting)
            reading, _ = G.g1_reading(pairs, twin_status, positive_pass=True)
            counts[m]["reading_pass"] += reading == "pass"
    res = {}
    for m, k in counts.items():
        res[m] = {
            "twin_pair_separates": k["twin_sep"] / (4 * draws),
            "twin_pair_interval_excludes_zero": k["twin_excl"] / (4 * draws),
            "main_pair_interval_excludes_zero": k["pair_excl"] / (6 * draws),
            "g1_some_main_pair_meets": k["main_any_meets"] / draws,
            "g1_reading_pass": k["reading_pass"] / draws,
        }
        if kind == "power":
            res[m]["g1_planted_pair_meets"] = k["planted_meets"] / draws
    return {
        **{k: job[k] for k in ("n_states", "cv", "kind", "draws", "reps")},
        "seeds": n_seeds,
        "bias_share": f,
        "n_push_transitions": int(len(state)),
        "results": res,
        "seconds": time.perf_counter() - t0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument(
        "--seeds-comparison",
        action="store_true",
        help="10 against 20 seeds per family, at the test set's size only",
    )
    args = ap.parse_args()
    draws = {"null": 30, "power": 20} if args.quick else {"null": 1000, "power": 500}
    reps = 500 if args.quick else c.BOOTSTRAP_REPLICATES
    jobs = []
    key = 200_000
    if args.seeds_comparison:
        sizes, key = ((c.N_TEST_STATES, 10), (c.N_TEST_STATES, 20)), 300_000
    else:
        sizes = ((c.N_SMOKE_STATES, 10), (c.N_TEST_STATES, 10))
    for n_states, n_seeds in sizes:
        for cv in (0.10, 0.20, 0.40):
            for kind in ("null", "power"):
                key += 1_000
                jobs.append(
                    {
                        "n_states": n_states,
                        "seeds": n_seeds,
                        "cv": cv,
                        "kind": kind,
                        "draws": draws[kind],
                        "reps": reps,
                        "key": key,
                    }
                )
    t0 = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(ex.map(setting, jobs))
    for r in rows:
        print(r["n_states"], r["cv"], r["kind"], json.dumps(r["results"]), flush=True)
    out = {
        "what": __doc__.split("\n\n")[0],
        "methods": {
            "design": "states (paired) and seeds resampled with replacement (DESIGN.md, G1)",
            "inflated": "design, each family's seed part times sqrt(n / (n - 1))",
            "t": "states-only replicate + t(Welch-Satterthwaite df) x sqrt(s_a^2 / n_a + s_b^2 / n_b)",
            "welch": "Welch's t-test on the per-seed E_int (test states held fixed)",
        },
        "quick": args.quick,
        "settings": rows,
        "wall_seconds": time.perf_counter() - t0,
    }
    name = "seed_interval_checks_seeds.json" if args.seeds_comparison else "seed_interval_checks.json"
    if args.quick:
        name = name.replace(".json", "_quick.json")
    (OUT / name).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
