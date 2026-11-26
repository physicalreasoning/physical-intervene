"""Training after the pilot (DESIGN.md, Models, Controls, Seal step 3).

    uv run --extra models python scripts/train_runs.py main [--workers N] [--family F] [--dry-run]
    uv run --extra models python scripts/train_runs.py shortcut [--workers N] [--dry-run]

main: every main run (4 architectures; 20 seeds at medium width, 5 at small and large), `blind` and
the four twins, each with its architecture's pilot settings and budget (results/pilot/settings.json),
on the training split, L_obs on the validation split, every scheduled checkpoint saved. Resumes:
a run with a run.json is skipped.

shortcut: after every medium-width main run has finished, V* from their best L_obs, D_obs from the
validation split and the simulator (make_data.py validation_policy_mean), L_mlp, and p by the
design's rule; writes results/training/p.json, then trains the 20 `mlp-shortcut` runs at that p
(none if no p fits: the positive control is not run).

No intervention error is computed here. --dry-run uses data/dryrun/, the dry-run pilot settings and
a tiny budget, and skips the freeze check.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from physical_intervene import constants as c
from physical_intervene import episodes as E
from physical_intervene import pipeline as P
from physical_intervene.data import box_scaler, file_sha256, load_split, make_examples

DRY_RUN_UPDATES = 300


def settings(dry_run: bool) -> dict:
    return json.loads((P.results_dir("pilot", dry_run) / "settings.json").read_text())["settings"]


def job(fam: str, width: str, seed: int, st: dict, dry_run: bool, **extra) -> dict:
    s = st[P.settings_source(fam)]
    return {
        "family": fam,
        "width": width,
        "seed": seed,
        **{k: s[k] for k in ("lr", "batch_size", "optimiser")},
        "updates": DRY_RUN_UPDATES if dry_run else s["updates"],
        "train_file": str(P.data_path("train", dry_run)),
        "val_file": str(P.data_path("validation", dry_run)),
        "dir": str(P.runs_dir("main", dry_run) / P.run_name(fam, width, seed)),
        **extra,
    }


def main_runs(a) -> dict:
    st = settings(a.dry_run)
    todo = [j for j in P.main_jobs() if a.family is None or j["family"] == a.family]
    if a.dry_run:  # one seed per family and width is enough to check the wiring
        seen, keep = set(), []
        for j in todo:
            if (j["family"], j["width"]) not in seen:
                seen.add((j["family"], j["width"]))
                keep.append(j)
        todo = keep
    recs = P.run_all([job(j["family"], j["width"], j["seed"], st, a.dry_run) for j in todo], a.workers)
    return {"n_runs": len(recs)}


def shortcut(a) -> dict:
    st = settings(a.dry_run)
    rd = P.runs_dir("main", a.dry_run)
    medians, missing = {}, []
    for arch in c.MAIN_ARCHITECTURES:
        best = []
        seeds = c.MAIN_SEEDS[:1] if a.dry_run else c.MAIN_SEEDS
        for s in seeds:
            d = rd / P.run_name(arch, "medium", s)
            if not (d / "run.json").exists():
                missing.append(d.name)
                continue
            best.append(P.load_run(d)["best_l_obs"])
        medians[arch] = float(np.median(best)) if best else None
    if missing:
        raise SystemExit(f"medium-width main runs not finished: {missing[:5]} ({len(missing)} in all)")
    setter = max(medians, key=medians.get)
    v_star = c.MATCH_TARGET_FACTOR * medians[setter]
    scaler = box_scaler(make_examples(load_split(P.data_path("train", a.dry_run), "train")))
    vm_path = P.data_path("validation_policy_mean", a.dry_run)
    vm = E.load(vm_path)
    d_obs = float(P.d_term(vm["target"], vm["mean_outcome"], scaler.scale).mean())
    l_mlp = medians["mlp"]
    p = P.p_rule(l_mlp, d_obs, v_star)
    rec = {
        "rule": "DESIGN.md, Controls: p is the largest grid value with L_mlp + p^2 D_obs <= V*",
        "v_star": v_star,
        "v_star_set_by": setter,
        "medians_best_l_obs": medians,
        "l_mlp": l_mlp,
        "d_obs": d_obs,
        "d_obs_from": {"file": vm_path.name, "sha256": file_sha256(vm_path)},
        "p": p,
        "positive_control": "run" if p is not None else "not run: no p in the grid fits V*",
    }
    res = P.results_dir("training", a.dry_run)
    res.mkdir(parents=True, exist_ok=True)
    (res / "p.json").write_text(json.dumps(rec, indent=1))
    if p is not None:
        seeds = c.MAIN_SEEDS[:1] if a.dry_run else c.MAIN_SEEDS
        P.run_all([job("mlp-shortcut", "medium", s, st, a.dry_run, shortcut_p=p) for s in seeds], a.workers)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["main", "shortcut"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--family", default=None, help="main only: train one family")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not a.dry_run:
        from physical_intervene.seal import check_frozen

        check_frozen(1)
        if not (P.results_dir("pilot") / "confirmation.json").exists():
            raise SystemExit("the pilot is not finished (results/pilot/confirmation.json)")
    out = main_runs(a) if a.step == "main" else shortcut(a)
    print(json.dumps(out, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
