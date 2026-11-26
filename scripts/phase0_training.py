"""Phase 0, milestone 2: the training-side checks (SMOKE_TEST.md, Training pipeline checks 1 to 3,
and the training time part of Compute).

    uv run --extra models python scripts/phase0_training.py [--workers 8] [--quick]

Reads data/phase0/milestone1/attemptK/{coverage_pool,episodes}.npz (never states.npz) and writes
results/phase0/milestone2/attemptN/training_checks.json and plots/, with checkpoints under
data/phase0/milestone2/attemptN/checkpoints/ (gitignored), for N = constants.MILESTONE2_ATTEMPT.
Training side only: no trained model is scored on interventions, and no intervention error of any
kind is computed.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
import numpy as np
import torch

from physical_intervene import constants as c
from physical_intervene import training_checks as TC
from physical_intervene.data import load_split, make_examples
from physical_intervene.models import param_table
from physical_intervene.train import (
    TrainConfig,
    TrainData,
    checkpoint_schedule,
    fit_loss,
    loss_weights,
    multi_step_loss,
    train,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "phase0" / "milestone1" / f"attempt{c.SMOKE_ATTEMPT}"
TRAIN_FILE, VAL_FILE = DATA / "coverage_pool.npz", DATA / "episodes.npz"
CKPT = ROOT / "data" / "phase0" / "milestone2" / f"attempt{c.MILESTONE2_ATTEMPT}" / "checkpoints"
OUT = ROOT / "results" / "phase0" / "milestone2" / f"attempt{c.MILESTONE2_ATTEMPT}"

# Phase 0 placeholders for the *set by pilot* items (DESIGN.md, Models). Not chosen by looking at any
# intervention quantity, and not carried into the pilot: the pilot sets these by L_obs.
PLACEHOLDERS = {
    "optimiser": "adam",
    "lr": 1e-3,
    "batch_size": 256,
    "rssm_seq_batch": 16,
    "updates": 12_000,
}
# Check settings (not design values).
CHECK_SETTINGS = {
    "made_up_updates": 5_000,
    "overfit_updates": 2_000,
    "reproducibility_updates": 2_000,
    "made_up_eval_examples": 5_000,
    "shortcut_p_standin": 0.5,
    "smoke_split": "train on the coverage pool (1,000 episodes, 100,000 transitions); "
    "validate on the 50 smoke "
    "episodes (5,000 transitions)",
    "matching_seeds_medium": [11, 22, 33],
    "other_widths_seed": 11,
}

_TD: dict = {}


def train_data(target: str | None = None) -> TrainData:
    """Per-process cache of the training data (real, or one made-up target)."""
    if "real" not in _TD:
        _TD["split_train"] = load_split(TRAIN_FILE, "train")
        _TD["split_val"] = load_split(VAL_FILE, "validation")
        _TD["real"] = TrainData.build(_TD["split_train"], _TD["split_val"])
    if target is None:
        return _TD["real"]
    if target not in _TD:
        real = _TD["real"]
        if target in TC.ROLLOUT_TARGETS:  # mlp-ms: made-up episodes, the real data's scalers
            split = TC.made_up_episodes(_TD["split_train"], target)
            ex = make_examples(split)
            td = TrainData(
                split, ex, None, real.norm, real.dscaler, real.yscaler, ex.delta, dict(real.data_meta)
            )
        else:
            delta = TC.made_up_targets(real.train, real.norm, real.dscaler)[target]
            td = TrainData.build(_TD["split_train"], None, delta=delta, dscaler=real.dscaler)
        td.data_meta["target"] = f"made-up: {target}"
        _TD[target] = td
    return _TD[target]


def config(arch: str, width: str, seed: int, updates: int, **kw) -> TrainConfig:
    p = PLACEHOLDERS
    return TrainConfig(
        arch=arch,
        width=width,
        seed=seed,
        updates=updates,
        lr=p["lr"],
        batch_size=p["batch_size"],
        seq_batch=p["rssm_seq_batch"],
        optimiser=p["optimiser"],
        shortcut_p=CHECK_SETTINGS["shortcut_p_standin"] if arch == "mlp-shortcut" else None,
        **kw,
    )


def run_job(job: dict) -> dict:
    t0 = time.perf_counter()
    kind, arch = job["kind"], job["arch"]
    name = job["name"]
    if kind == "learn":
        td = train_data()
        cfg = config(arch, job["width"], job["seed"], job["updates"], tag=name)
        r = train(cfg, td, ckpt_dir=CKPT / name)
    elif kind == "repro":
        td = train_data()
        cfg = config(arch, job["width"], job["seed"], job["updates"], tag=name)
        r = train(cfg, td)
    elif kind == "madeup":
        td = train_data(job["target"])
        cfg = config(arch, "medium", 11, job["updates"], loss_dims=job["dims"], target_name=job["target"])
        r = train(cfg, td, evaluate_val=False)
        idx = np.random.default_rng(0).choice(
            len(td.train), CHECK_SETTINGS["made_up_eval_examples"], replace=False
        )
        r["fit_loss"] = fit_loss(r["model"], td, idx, job["dims"])
        r["fit_loss_before"] = r["curve"][0]["train_loss"]
        r["first_train_loss"] = next(
            (x["train_loss"] for x in r["curve"] if x["train_loss"] is not None), None
        )
    elif kind == "overfit":
        base = train_data()
        pool = np.arange(len(base.train))
        if arch == "mlp-ms":  # only starts whose 4-step rollout stays inside the episode
            pool = pool[base.train.step <= base.split.steps - c.MULTI_STEP_HORIZON]
        idx = np.random.default_rng(1).choice(pool, c.OVERFIT_EXAMPLES, replace=False)
        td = TrainData(
            base.split,
            base.train,
            None,
            base.norm,
            base.dscaler,
            base.yscaler,
            base.delta,
            base.data_meta,
            subset=idx,
        )
        cfg = config(arch, "medium", 11, job["updates"])
        r = train(cfg, td, evaluate_val=False)
        r["fit_loss"] = fit_loss(r["model"], td, idx)
        if arch == "mlp-ms":  # its own training loss, which is what it minimised
            with torch.no_grad():
                r["fit_loss_one_step"] = r["fit_loss"]
                r["fit_loss"] = float(multi_step_loss(r["model"], td, idx, loss_weights(None)))
    else:
        raise ValueError(kind)
    r.pop("model")
    r.update({"name": name, "kind": kind, "job": job, "wall_seconds": time.perf_counter() - t0})
    return r


def jobs(quick: bool) -> list[dict]:
    out = []
    for arch in c.MAIN_ARCHITECTURES:
        for seed in CHECK_SETTINGS["matching_seeds_medium"]:
            out.append({"kind": "learn", "arch": arch, "width": "medium", "seed": seed})
        for w in ("small", "large"):
            out.append(
                {"kind": "learn", "arch": arch, "width": w, "seed": CHECK_SETTINGS["other_widths_seed"]}
            )
        out.append(
            {"kind": "repro", "arch": arch, "width": "small", "seed": CHECK_SETTINGS["other_widths_seed"]}
        )
        out.append({"kind": "overfit", "arch": arch})
    for arch in ("blind", "mlp-shortcut"):
        out.append({"kind": "learn", "arch": arch, "width": "medium", "seed": 11})
    for arch, plan in TC.MADE_UP_PLAN.items():
        for target, dims in plan:
            out.append({"kind": "madeup", "arch": arch, "target": target, "dims": dims})
    for j in out:
        j["name"] = "_".join(str(j[k]) for k in ("kind", "arch", "width", "seed", "target") if k in j)
    if quick:
        PLACEHOLDERS["updates"] = 2_000
        for k in ("made_up_updates", "overfit_updates", "reproducibility_updates"):
            CHECK_SETTINGS[k] = 300
        out = [
            j for j in out if j.get("width") in (None, "small", "medium") and j.get("seed", 11) in (11, 22)
        ]
    budget = {
        "learn": PLACEHOLDERS["updates"],
        "repro": CHECK_SETTINGS["reproducibility_updates"],
        "madeup": CHECK_SETTINGS["made_up_updates"],
        "overfit": CHECK_SETTINGS["overfit_updates"],
    }
    for j in out:
        j["updates"] = budget[j["kind"]]
    cost = {"gru": 3, "contact_wm": 3, "mlp-ms": 2}
    width_cost = {"large": 4, "medium": 1.5, "small": 1, None: 1}
    return sorted(
        out,
        key=lambda j: (
            -cost.get(j["arch"], 1) * width_cost[j.get("width")] * (3 if j["kind"] == "learn" else 1)
        ),
    )


# Plots ---------------------------------------------------------------------------------------------


def plot_curves(runs: list[dict], base: dict, path: Path) -> None:
    archs = [
        a
        for a in (*c.MAIN_ARCHITECTURES, "blind", "mlp-shortcut")
        if any(r["config"]["arch"] == a for r in runs)
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    colours = {"small": "tab:blue", "medium": "tab:orange", "large": "tab:green"}
    for ax, arch in zip(axes.flat, archs, strict=False):
        for r in runs:
            if r["config"]["arch"] != arch:
                continue
            u = [x["updates"] for x in r["curve"]][1:]
            L = [x["l_obs"] for x in r["curve"]][1:]
            ls = "-" if r["config"]["seed"] == 11 else ":"
            ax.plot(
                u,
                L,
                ls,
                color=colours[r["config"]["width"]],
                label=f"{r['config']['width']} s{r['config']['seed']}",
            )
        for b, st in (("copy", "--"), ("constvel", "-."), ("knn", ":")):
            ax.axhline(base[b]["l_obs"], color="grey", ls=st, lw=1, label=b)
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_title(arch)
        ax.set_xlabel("updates")
        ax.legend(fontsize=7)
    axes[0, 0].set_ylabel("L_obs (smoke validation)")
    axes[1, 0].set_ylabel("L_obs (smoke validation)")
    fig.suptitle(
        "Phase 0 smoke runs: L_obs over checkpoints (placeholder settings; mechanics, not performance)"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_matching(runs: list[dict], matching: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colours = dict(zip(c.MAIN_ARCHITECTURES, ("tab:blue", "tab:orange", "tab:green", "tab:red"), strict=True))
    for r in runs:
        a = r["config"]["arch"]
        if r["config"]["width"] != "medium" or a not in colours:
            continue
        u = [x["updates"] for x in r["curve"]][1:]
        L = [x["l_obs"] for x in r["curve"]][1:]
        ax.plot(u, L, color=colours[a], lw=1, label=a if r["config"]["seed"] == 11 else None)
        m = matching["runs"][r["name"]]
        ax.plot(
            m["matched_update"],
            m["matched_l_obs"],
            "o",
            color=colours[a],
            mfc="none" if m["excluded"] else colours[a],
        )
    v = matching["v_star_like"]
    ax.axhline(v, color="k", lw=1, label=f"V*-like = {v:.3f}")
    ax.axhspan(
        v * (1 - c.MATCHED_LOSS_REL_TOL),
        v * (1 + c.MATCHED_LOSS_REL_TOL),
        color="k",
        alpha=0.08,
        label="within 5%",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("updates")
    ax.set_ylabel("L_obs (smoke validation)")
    ax.set_title("Matching on smoke runs, medium width (dots: matched checkpoints; open: excluded)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def solo_timing(updates: int = 300) -> dict:
    """Train time per update and L_obs time per checkpoint for each architecture and width, one run
    at a time on one thread (no other run in parallel), projected to the placeholder budget."""
    from physical_intervene.train import evaluate

    td = train_data()
    n_ckpt = len(checkpoint_schedule(PLACEHOLDERS["updates"])) + 1
    rows = {}
    for arch in (*c.MAIN_ARCHITECTURES, "blind", "mlp-shortcut"):
        for w in ("small", "medium", "large"):
            r = train(config(arch, w, 11, updates), td, evaluate_val=False)
            t0 = time.perf_counter()
            evaluate(r["model"], td)
            t_eval = time.perf_counter() - t0
            ms = 1e3 * r["train_seconds"] / updates
            rows[f"{arch}/{w}"] = {
                "ms_per_update": ms,
                "eval_seconds_per_checkpoint": t_eval,
                "minutes_at_placeholder_budget": (ms * PLACEHOLDERS["updates"] / 1e3 + n_ckpt * t_eval) / 60,
            }
            print(arch, w, f"{ms:.1f} ms/update, eval {t_eval:.1f} s", flush=True)
    plan = {f"{a}/medium": 10 for a in c.MAIN_ARCHITECTURES}
    plan.update({f"{a}/{w}": 5 for a in c.MAIN_ARCHITECTURES for w in ("small", "large")})
    controls = {"blind/medium": 10, "mlp-shortcut/medium": 10}
    controls.update({f"{a}/medium": 10 for a in c.MAIN_ARCHITECTURES})  # twins
    hours = {
        name: sum(n * rows[k]["minutes_at_placeholder_budget"] for k, n in runs.items()) / 60
        for name, runs in (("main_80_runs", plan), ("control_60_runs", controls))
    }
    hours["total"] = sum(hours.values())
    return {
        "updates_timed": updates,
        "checkpoints_at_placeholder_budget": n_ckpt,
        "per_architecture": rows,
        "projected_cpu_hours_at_placeholder_budget": hours,
        "note": "one thread, one run at a time; includes L_obs at every checkpoint of the schedule",
    }


def _json_default(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x))


# Main ----------------------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--quick", action="store_true", help="short budgets, to check the script runs")
    ap.add_argument(
        "--timing", action="store_true", help="only time each model alone, into the existing record"
    )
    args = ap.parse_args()
    if args.timing:
        path = OUT / "training_checks.json"
        record = json.loads(path.read_text())
        record["checks"]["5_compute"]["solo"] = solo_timing()
        record["environment"]["solo_timing_python"] = sys.version.split()[0]
        path.write_text(json.dumps(record, indent=1, default=_json_default))
        return 0
    out_dir = OUT if not args.quick else OUT / "quick"
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # Part 1: data.
    split_train, split_val = load_split(TRAIN_FILE, "train"), load_split(VAL_FILE, "validation")
    td = TrainData.build(split_train, split_val)
    val_changed = load_split(VAL_FILE, "validation")
    val_changed.obs = val_changed.obs * 3.0 + 0.5
    td_changed = TrainData.build(split_train, val_changed)
    scaling = TC.check_scaling(td.train, td.val)
    same = {
        "box_scaler": np.array_equal(td.yscaler.scale, td_changed.yscaler.scale),
        "delta_scaler": np.array_equal(td.dscaler.scale, td_changed.dscaler.scale)
        and np.array_equal(td.dscaler.mean, td_changed.dscaler.mean),
        "input_norm": all(
            np.array_equal(a, b)
            for a, b in zip(td.norm.__dict__.values(), td_changed.norm.__dict__.values(), strict=True)
        ),
    }
    scaling["unchanged_when_validation_changes_trainingdata"] = same
    scaling["delta_std"] = td.dscaler.scale.tolist()
    scaling["pass"] = scaling["pass"] and all(same.values())
    data_checks = {
        "alignment": TC.check_alignment(),
        "target": TC.check_target(split_val),
        "scaling": scaling,
        "splits": TC.check_splits([split_train, split_val]),
    }
    base = TC.baseline_losses(td.train, make_examples(split_val))
    print("data checks:", {k: v["pass"] for k, v in data_checks.items()}, flush=True)

    # Parts 2 and 3: training runs, in parallel, one thread each.
    todo = jobs(args.quick)
    results: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(run_job, todo):
            results[r["name"]] = r
            print(f"{r['name']}: {r['wall_seconds']:.0f} s", flush=True)

    learn = [r for r in results.values() if r["kind"] == "learn"]
    main_learn = [r for r in learn if r["config"]["arch"] in c.MAIN_ARCHITECTURES]

    made_up = {}
    for r in results.values():
        if r["kind"] != "madeup":
            continue
        arch, target = r["job"]["arch"], r["job"]["target"]
        expect_fit = arch != "blind"
        near0 = r["fit_loss"] < c.NEAR_ZERO_STD_MSE
        made_up.setdefault(arch, {})[target] = {
            "loss_dims": r["job"]["dims"] or "all 11",
            "fit_loss": r["fit_loss"],
            "first_train_loss": r["first_train_loss"],
            "near_zero": near0,
            "expected_near_zero": expect_fit,
            "pass": near0 == expect_fit,
        }
    wiring = {
        "threshold": c.NEAR_ZERO_STD_MSE,
        "unit": "shared training loss on 5,000 training examples "
        "(std-MSE in the real training split's state-change "
        "scaler; 0.5 box + 0.5 pusher), of the one-step prediction; mlp-ms is trained on rollouts over "
        "made-up episodes (zero_rollout, action_rollout)",
        "results": made_up,
        "pass_condition": f"made-up training loss below {c.NEAR_ZERO_STD_MSE}; "
        "blind is expected not to fit the action target",
        "pass": all(v["pass"] for a in made_up.values() for v in a.values()),
    }
    overfit = {
        r["job"]["arch"]: {
            "fit_loss": r["fit_loss"],
            **({"fit_loss_one_step": r["fit_loss_one_step"]} if "fit_loss_one_step" in r else {}),
            "pass": r["fit_loss"] < c.NEAR_ZERO_STD_MSE,
        }
        for r in results.values()
        if r["kind"] == "overfit"
    }
    repro = {}
    for r in results.values():
        if r["kind"] == "repro":
            first = (
                results[f"learn_{r['job']['arch']}_small_11"]
                if f"learn_{r['job']['arch']}_small_11" in results
                else None
            )
            repro[r["job"]["arch"]] = (
                TC.check_reproducible(first, r) if first else {"pass": False, "note": "no reference run"}
            )
    matching = TC.check_matching(main_learn)
    learning = {
        "overfit_64": {
            "results": overfit,
            "pass_condition": f"training loss on the 64 examples below {c.NEAR_ZERO_STD_MSE}",
            "pass": all(v["pass"] for v in overfit.values()),
        },
        "losses_fall": TC.check_losses_fall(main_learn),
        "beats_baselines": TC.check_beats_baselines(main_learn, base),
        "matching": matching,
        "budget": TC.check_budget(learn),
        "reproducibility": {
            "results": repro,
            "pass_condition": "a rerun with the same seed gives the same L_obs and training-loss curve, "
            "bit for bit (CPU)",
            "pass": all(v["pass"] for v in repro.values()),
        },
    }
    shortcut = [r for r in learn if r["config"]["arch"] == "mlp-shortcut"]

    # Compute.
    compute = {}
    for r in learn:
        a, w = r["config"]["arch"], r["config"]["width"]
        key = f"{a}/{w}"
        if key in compute:
            continue
        n = r["updates_done"]
        compute[key] = {
            "updates": n,
            "train_seconds": r["train_seconds"],
            "eval_seconds": r["eval_seconds"],
            "ms_per_update": 1e3 * r["train_seconds"] / n,
            "params": r["params"],
        }
    per_run_min = {k: v["ms_per_update"] * PLACEHOLDERS["updates"] / 6e4 for k, v in compute.items()}
    plan = {f"{a}/medium": 10 for a in c.MAIN_ARCHITECTURES}
    plan.update({f"{a}/small": 5 for a in c.MAIN_ARCHITECTURES})
    plan.update({f"{a}/large": 5 for a in c.MAIN_ARCHITECTURES})
    controls = {"blind/medium": 10, "mlp-shortcut/medium": 10}
    controls.update({f"{a}/medium": 10 for a in c.MAIN_ARCHITECTURES})  # twins
    hours_main = sum(n * per_run_min[k] for k, n in plan.items() if k in per_run_min) / 60
    hours_ctrl = sum(n * per_run_min[k] for k, n in controls.items() if k in per_run_min) / 60
    compute_check = {
        "per_architecture": compute,
        "train_minutes_at_placeholder_budget": per_run_min,
        "projected_cpu_hours_80_main_plus_60_control_runs_at_placeholder_budget": {
            "main": hours_main,
            "controls": hours_ctrl,
            "total": hours_main + hours_ctrl,
        },
        "design_estimate": "minutes per run for mlp and mlp-ms, tens of minutes for gru and contact_wm "
        "on one CPU core; "
        "80 + 60 runs in three to five days of CPU",
        "note": "one thread per run; 8 runs in parallel on the machine below, "
        "so times include some contention. "
        "Train time excludes evaluation at checkpoints (eval_seconds). The real budget is set by the pilot.",
    }

    plot_curves(learn, base, out_dir / "plots" / "loss_curves.png")
    plot_matching(main_learn, matching, out_dir / "plots" / "matching.png")

    record = {
        "milestone": 2,
        "phase": "phase0",
        "smoke_attempt": c.SMOKE_ATTEMPT,
        "quick": args.quick,
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "torch_threads_per_run": 1,
            "parallel_runs": args.workers,
        },
        "data": td.data_meta,
        "placeholders_set_by_pilot": PLACEHOLDERS,
        "check_settings": CHECK_SETTINGS,
        "chosen_values": {
            "history_padding": "copies of s_0 with a zero action, plus a mask input "
            "(1 real, 0 padding); all models",
            "dtheta_from_prediction": "atan2 of the predicted (cos, sin) = current + predicted change, "
            "minus the current "
            "angle, wrapped to (-pi, pi]",
            "delta_state_std_floors": list(c.DELTA_STATE_STD_FLOORS),
            "near_zero_threshold": c.NEAR_ZERO_STD_MSE,
        },
        "params": param_table(),
        "baselines": base,
        "checks": {
            "1_data": data_checks,
            "2_wiring": wiring,
            "3_learning": learning,
            "5_compute": compute_check,
        },
        "controls_trained_not_scored": {
            r["name"]: {
                "best_l_obs": min(x["l_obs"] for x in r["curve"][1:]),
                "swapped_share": r.get("shortcut_swapped_share"),
            }
            for r in learn
            if r["config"]["arch"] in ("blind", "mlp-shortcut")
        },
        "shortcut_swapped_share": {r["name"]: r["shortcut_swapped_share"] for r in shortcut},
        "contact_wm_learned_values": {
            r["name"]: r["contact_wm_physical_values"]
            for r in results.values()
            if "contact_wm_physical_values" in r
        },
        "curves": {r["name"]: r["curve"] for r in learn},
        "checkpoints_dir": str(CKPT.relative_to(ROOT)),
        "wall_seconds_total": time.time() - t_start,
    }
    passes = {
        "1_data": all(v["pass"] for v in data_checks.values()),
        "2_wiring": wiring["pass"],
        **{f"3_{k}": v["pass"] for k, v in learning.items()},
    }
    record["summary"] = passes
    (out_dir / "training_checks.json").write_text(json.dumps(record, indent=1, default=_json_default))
    print(json.dumps(passes, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
