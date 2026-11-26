"""Score every trained run on the test states and write the result (DESIGN.md, Gates, Results, Seal
step 5).

    uv run --extra models python scripts/score.py --replay results/test_replay.json [--workers N]

Refuses to run unless every hash recorded at freeze 1 and 2 matches, checkpoints included. The
replay file is the output of `python scripts/replay_check.py --check data/main/test_states.npz`
run in a fresh process in the pinned image (G0.1); a second machine's, if given with
--replay-second, is reported, not gated.

Every checkpoint of every run is scored: L_obs on the validation transitions and std-MSE on all 25
actions at every test state. The gates, controls, secondary analyses and the result file come from
results_format.build_primary, the code the Phase 0 stand-in run checked. Writes
results/intervention_results.json and checks it against the format.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from physical_intervene import baselines as B
from physical_intervene import constants as c
from physical_intervene import pipeline as P
from physical_intervene import results_format as RF
from physical_intervene import scoring as S
from physical_intervene.data import box_scaler, load_split, make_examples
from physical_intervene.seal import ROOT, check_frozen

SCORE_BLOCKS = (*S.BLOCKS, "logged")
RNG_SEED = 0  # bootstrap draws of the result (G2's interval, the states-only intervals, descriptive blocks)

_CTX: dict = {}


def context() -> dict:
    if not _CTX:
        train = make_examples(load_split(P.data_path("train"), "train"))
        _CTX["val"] = make_examples(load_split(P.data_path("validation"), "validation"))
        _CTX["iset"] = S.load_interventions(P.data_path("test_states"), P.data_path("test_history"))
        _CTX["scaler"] = box_scaler(train)
        _CTX["train"] = train
    return _CTX


def _score(run_dir: str) -> dict:
    ctx = context()
    return P.score_run(run_dir, ctx["val"], ctx["iset"], ctx["scaler"], SCORE_BLOCKS)


def families(fz: dict, workers: int) -> tuple[dict, dict, dict]:
    rd = P.runs_dir("main")
    groups: dict[tuple[str, str], list[int]] = {}
    for j in P.main_jobs():
        groups.setdefault((j["family"], j["width"]), []).append(j["seed"])
    if fz["models"]["p"] is not None:
        groups[("mlp-shortcut", "medium")] = list(c.MAIN_SEEDS)
    order = [(k, s) for k, seeds in groups.items() for s in seeds]
    dirs = [str(rd / P.run_name(f, w, s)) for (f, w), s in order]
    with ProcessPoolExecutor(workers) as ex:
        scored = list(ex.map(_score, dirs))
    by = {}
    for ((f, w), s), r in zip(order, scored, strict=True):
        by.setdefault((f, w), []).append((s, r))
    ctx = context()
    fams, widths = {}, {"small": {}, "large": {}}
    for (f, w), rows in by.items():
        fam = P.family_from_scored(f, w, [s for s, _ in rows], [r for _, r in rows], ctx["iset"], ctx["val"])
        if w == "medium":
            fams[f] = fam
        else:
            widths[w][f] = fam
    updates = {n: int(fz["models"]["settings"][P.settings_source(n)]["updates"]) for n in fams}
    return fams, widths, updates


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", required=True, help="replay_check.py --check output on the test states")
    ap.add_argument("--replay-second", default=None, help="the same check on a second machine (reported)")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    t0 = time.perf_counter()
    fz = check_frozen(2, checkpoints=True)
    if (
        S.load_interventions(P.data_path("test_states")).sha256
        != json.loads((P.data_dir() / "manifest.json").read_text())["test_states"]["sha256"]
    ):
        raise SystemExit("test_states.npz differs from the one make_data.py recorded")
    rep = json.loads(open(a.replay).read())
    g0_replay = {
        "passed": bool(rep["passed"]),
        "source": "replay_check.py --check on the test states, fresh process in the pinned image",
        "n_compared": rep.get("n_compared"),
        "n_mismatched": rep.get("n_mismatched"),
    }
    if a.replay_second:
        r2 = json.loads(open(a.replay_second).read())
        g0_replay["second_machine"] = {
            "passed": bool(r2["passed"]),
            "platform": r2.get("platform"),
            "gated": False,
        }
    fams, widths, updates = families(fz, a.workers)
    ctx = context()
    iset, val = ctx["iset"], ctx["val"]
    base = {
        "copy": S.predict_baseline(B.copy, val, iset),
        "constvel": S.predict_baseline(B.constvel, val, iset),
        "knn": S.predict_baseline(B.KNN(ctx["train"]), val, iset),
    }
    lab = iset.label[:, 1:]
    counts = {
        "n_train_obs": len(ctx["train"]),
        "n_val_obs": len(val),
        "n_interventions_total": int(lab.size),
        "n_held_out_checkpoints": iset.n_states,
        "regime_counts": {s: int((lab == i).sum()) for i, s in enumerate(S.STRATA)},
        "n_states_with_push": int((iset.counts("push") > 0).sum()),
    }
    primary = RF.build_primary(
        fams,
        widths,
        ctx["scaler"],
        iset,
        val.s,
        val.y,
        val.episode,
        base,
        counts,
        g0_replay,
        np.random.default_rng(RNG_SEED),
        updates,
    )
    result = {
        "format_version": RF.FORMAT_VERSION,
        "benchmark": "physical-intervene/pushing",
        "device": "cpu",
        "seeds": list(c.MAIN_SEEDS),
        "updates": updates,
        "wall_seconds": time.perf_counter() - t0,
        "phase": "scoring",
        "freeze": {
            "design_sha256": fz["design"]["design_sha256"],
            "models_sha256": fz["models_sha256"],
            "test_seed_start": fz["test_seed_start"],
            "amendments": fz["amendments"],
        },
        "primary": primary,
    }
    errors = RF.check_conformance(result)
    result["format_errors"] = errors
    out = ROOT / "results" / "intervention_results.json"
    out.write_text(json.dumps(result, indent=1, default=float))
    print(json.dumps(primary["gates"], indent=1, default=float))
    if errors:
        print(f"format errors: {errors}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
