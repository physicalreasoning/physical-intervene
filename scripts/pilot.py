"""The pilot (DESIGN.md, Models, Pilot rule; Controls, Pilot confirmation), after freeze 1.

    uv run --extra models python scripts/pilot.py grid [--workers N] [--dry-run]
    uv run --extra models python scripts/pilot.py confirm [--workers N] [--dry-run]

grid: for each main architecture at medium width, every setting of the grid on the pilot training
seeds, trained on the pilot's 1,000 episodes until the plateau criterion or the cap; L_obs on the
validation split. Applies steps 3 to 5 of the pilot rule, extends the grid once for an
architecture whose lowest-scoring setting has an edge learning rate, and writes
results/pilot/settings.json. Validation loss only.

confirm: the positive control's confirmation. Pilot V*, L_mlp and D_obs from the pilot's own runs
and the simulator give a pilot p by the design's rule; `mlp-shortcut` (at that p) and `mlp-twin`
are trained on pilot data and scored on the pilot states (push), the only intervention error the
pilot computes. Writes results/pilot/confirmation.json. It never changes any rule or setting.

--dry-run uses data/dryrun/ (make_data.py --dry-run) and a tiny budget, and skips the freeze check;
there the scorer refuses the confirmation runs (no freeze yet), so confirm stops before scoring.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from physical_intervene import constants as c
from physical_intervene import episodes as E
from physical_intervene import gates as G
from physical_intervene import pipeline as P
from physical_intervene.data import box_scaler, file_sha256, load_split, make_examples

DRY_RUN_UPDATES = 300


def grid_jobs(arch: str, lrs, dry_run: bool) -> list[dict]:
    jobs = []
    for lr in lrs:
        for bs in c.PILOT_BATCH_GRID:
            for seed in c.PILOT_TRAINING_SEEDS:
                jobs.append(
                    {
                        "family": arch,
                        "width": "medium",
                        "seed": seed,
                        "lr": lr,
                        "batch_size": bs,
                        "optimiser": c.PILOT_OPTIMISER,
                        "updates": DRY_RUN_UPDATES if dry_run else c.PILOT_MAX_UPDATES,
                        "stop_at_plateau": True,
                        "save": False,
                        "train_file": str(P.data_path("pilot", dry_run)),
                        "val_file": str(P.data_path("validation", dry_run)),
                        "dir": str(P.runs_dir("pilot", dry_run) / P.run_name(arch, "medium", seed, lr, bs)),
                    }
                )
    return jobs


def rows(recs: list[dict]) -> list[dict]:
    return [
        {
            "lr": r["job"]["lr"],
            "batch_size": r["job"]["batch_size"],
            "seed": r["job"]["seed"],
            "best_l_obs": r["best_l_obs"],
            "plateau_point": r["plateau_point"],
            "updates_done": r["updates_done"],
        }
        for r in recs
    ]


def grid(a) -> dict:
    out: dict = {"rule": "DESIGN.md, Models, Pilot rule", "architectures": {}}
    for arch in c.MAIN_ARCHITECTURES:
        recs = P.run_all(grid_jobs(arch, c.PILOT_LR_GRID, a.dry_run), a.workers)
        choice = P.choose_setting(rows(recs), c.PILOT_LR_GRID)
        entry = {"first_choice": choice, "extended_with_lr": None}
        ext = P.edge_extension(choice["edge"])
        if ext is not None:
            recs += P.run_all(grid_jobs(arch, (ext,), a.dry_run), a.workers)
            lrs = tuple(sorted((*c.PILOT_LR_GRID, ext)))
            choice = P.choose_setting(rows(recs), lrs)
            entry["extended_with_lr"] = ext
            entry["best_at_edge_of_extended_grid"] = choice["edge"] is not None
        entry["chosen"] = choice
        entry["runs"] = rows(recs)
        out["architectures"][arch] = entry
    out["settings"] = {
        arch: {k: e["chosen"][k] for k in ("optimiser", "lr", "batch_size", "updates")}
        for arch, e in out["architectures"].items()
    }
    out["data"] = {
        k: {
            "file": str(P.data_path(k, a.dry_run).relative_to(P.ROOT)),
            "sha256": file_sha256(P.data_path(k, a.dry_run)),
        }
        for k in ("pilot", "validation")
    }
    out["note"] = "every choice minimises L_obs on validation data; no intervention error was computed"
    return out


def confirm(a) -> dict:
    res = P.results_dir("pilot", a.dry_run)
    g = json.loads((res / "settings.json").read_text())
    medians = {
        arch: float(
            np.median(
                [
                    r["best_l_obs"]
                    for r in e["runs"]
                    if (r["lr"], r["batch_size"]) == (e["chosen"]["lr"], e["chosen"]["batch_size"])
                ]
            )
        )
        for arch, e in g["architectures"].items()
    }
    setter = max(medians, key=medians.get)
    v_star = c.MATCH_TARGET_FACTOR * medians[setter]
    l_mlp = medians["mlp"]
    pilot_ex = make_examples(load_split(P.data_path("pilot", a.dry_run), "train"))
    scaler = box_scaler(pilot_ex)
    vm = E.load(P.data_path("validation_policy_mean", a.dry_run))
    d_obs = float(P.d_term(vm["target"], vm["mean_outcome"], scaler.scale).mean())
    p = P.p_rule(l_mlp, d_obs, v_star)
    out = {
        "what": "pilot confirmation of the positive control (DESIGN.md, Controls); controls only",
        "pilot_v_star": v_star,
        "pilot_v_star_set_by": setter,
        "pilot_medians_best_l_obs": medians,
        "pilot_l_mlp": l_mlp,
        "pilot_d_obs": d_obs,
        "pilot_p": p,
        "note": "this reveals one absolute E_int of an mlp-like model (mlp-twin) and nothing about how mlp, "
        "gru, mlp-ms and contact_wm compare; the final p is computed afresh from the main runs",
    }
    if p is None:
        out["result"] = "not run: no p in the grid fits the pilot's V*"
        return out
    st = g["settings"]["mlp"]
    jobs = []
    for fam, seeds, extra in (
        ("mlp-shortcut", c.PILOT_TRAINING_SEEDS, {"shortcut_p": p}),
        ("mlp-twin", c.TWIN_SEEDS[: len(c.PILOT_TRAINING_SEEDS)], {}),
    ):
        for s in seeds:
            jobs.append(
                {
                    "family": fam,
                    "width": "medium",
                    "seed": s,
                    **{k: st[k] for k in ("lr", "batch_size", "optimiser")},
                    "updates": DRY_RUN_UPDATES if a.dry_run else st["updates"],
                    "pilot_confirmation": True,
                    "train_file": str(P.data_path("pilot", a.dry_run)),
                    "val_file": str(P.data_path("validation", a.dry_run)),
                    "dir": str(P.runs_dir("pilot_confirmation", a.dry_run) / P.run_name(fam, "medium", s)),
                    **extra,
                }
            )
    P.run_all(jobs, a.workers)
    from physical_intervene import scoring as S
    from physical_intervene.seal import current_phase

    if current_phase() == "phase0":  # dry run: show the guard refuses, and stop before scoring
        meta = json.loads(next((P.runs_dir("pilot_confirmation", a.dry_run)).glob("*/u*.json")).read_text())
        try:
            S.assert_scorable(meta)
        except S.LeakError as e:
            out["result"] = f"dry run, stopped before scoring: {e}"
            return out
        raise SystemExit("the scorer accepted a trained run in phase0")

    iset = S.load_interventions(
        P.data_path("pilot_states", a.dry_run), P.data_path("pilot_history", a.dry_run)
    )
    val = make_examples(load_split(P.data_path("validation", a.dry_run), "validation"))
    fams = {}
    for fam in ("mlp-shortcut", "mlp-twin"):
        dirs = sorted(P.runs_dir("pilot_confirmation", a.dry_run).glob(f"{fam}_medium_s*"))
        scored = [P.score_run(d, val, iset, scaler, ("push",)) for d in dirs]
        seeds = [P.load_run(d)["job"]["seed"] for d in dirs]
        fams[fam] = P.family_from_scored(fam, "medium", seeds, scored, iset, val)
    ms = {n: G.match(f, v_star, min_seeds=1) for n, f in fams.items()}
    sc, tw = G.arm(fams["mlp-shortcut"], ms["mlp-shortcut"]), G.arm(fams["mlp-twin"], ms["mlp-twin"])
    sm = E.load(P.data_path("pilot_states_policy_mean", a.dry_run))
    states = E.load(P.data_path("pilot_states", a.dry_run))
    d_int_terms = P.d_term(states["target"], sm["mean_outcome"][:, None, :], scaler.scale)
    d_int = float(d_int_terms[iset.mask("push")].mean())
    out.update(
        {
            "pilot_d_int": d_int,
            "e_int_mlp_twin": tw.e,
            "e_int_mlp_shortcut": sc.e,
            "observed_rel_gap": (sc.e - tw.e) / tw.e,
            "predicted_rel_gap": p * p * d_int / tw.e,
            "welch": G.welch_test(sc, tw),
            "matched_l_obs": {n: m.l.tolist() for n, m in ms.items()},
            "result": "reported; the rule stays as written whatever this shows (DESIGN.md, Controls)",
        }
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["grid", "confirm"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not a.dry_run:
        from physical_intervene.seal import check_frozen

        check_frozen(1)
    res = P.results_dir("pilot", a.dry_run)
    res.mkdir(parents=True, exist_ok=True)
    out = grid(a) if a.step == "grid" else confirm(a)
    name = "settings.json" if a.step == "grid" else "confirmation.json"
    Path(res / name).write_text(json.dumps(out, indent=1, default=float))
    print(
        json.dumps(
            out.get("settings", {k: out[k] for k in out if k.startswith("pilot_")}), indent=1, default=float
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
