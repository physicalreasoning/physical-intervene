"""Phase 0, milestone 1: generate the smoke data and run the environment checks.

    uv run python scripts/phase0_env.py \
        [--docker-replay results/phase0/milestone1/attemptK/docker_replay.json]

Writes data/phase0/milestone1/attemptK/*.npz (gitignored),
results/phase0/milestone1/attemptK/env_checks.json and results/phase0/milestone1/attemptK/plots/,
for K = constants.SMOKE_ATTEMPT.
Simulator and scripted policy only: no model of any kind is built, trained or run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
import mujoco
import numpy as np

from physical_intervene import constants as c
from physical_intervene import env_checks as checks
from physical_intervene import episodes as E
from physical_intervene import interventions as I
from physical_intervene import policy
from physical_intervene.sim import Sim, scene_sha256

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
# Written after reading the attempt's threshold plot (top row); checked again on every attempt.
THRESHOLD_DECISION = (
    "defaults kept (d_push 2 mm, theta_push 1 deg). Against the no-contact reference, 13% of contacts "
    "change nothing (a spike at zero) and the rest form a continuum peaking at 3 to 5 mm. The only gap, "
    "between zero and about 0.01 mm, separates no-effect touches from all others, not grazes from "
    "pushes, so the rule has no gap to move to. 2 mm lies near the median of contact effects"
)
# Each Phase 0 attempt (constants.SMOKE_ATTEMPT) keeps its own data and record.
DATA = ROOT / "data" / "phase0" / "milestone1" / f"attempt{c.SMOKE_ATTEMPT}"
OUT = ROOT / "results" / "phase0" / "milestone1" / f"attempt{c.SMOKE_ATTEMPT}"


def timed(fn, *args, **kw):
    t0 = time.perf_counter()
    out = fn(*args, **kw)
    return out, time.perf_counter() - t0


def plot_thresholds(states: dict, path: Path) -> None:
    contact = states["contact"][:, 1:]
    kinds = np.broadcast_to(states["kinds"][1:], contact.shape)[contact]
    t = states["target"][:, 1:][contact]
    disp = 1e3 * np.hypot(t[:, 0], t[:, 1])
    rot = np.rad2deg(np.abs(t[:, 2]))
    ref = states["reference"][:, None, :].repeat(contact.shape[1], axis=1)[contact]
    diff = t - ref
    diff[:, 2] = np.pi - np.mod(np.pi - diff[:, 2], 2 * np.pi)
    disp_ref = 1e3 * np.hypot(diff[:, 0], diff[:, 1])
    rot_ref = np.rad2deg(np.abs(diff[:, 2]))
    d, th = c.PUSH_MIN_DISPLACEMENT_M * 1e3, c.PUSH_MIN_ROTATION_DEG
    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    for row, (dd, rr, title) in enumerate(
        (
            (disp_ref, rot_ref, "against no-contact reference (design label)"),
            (disp, rot, "total motion (earlier rule, reported)"),
        )
    ):
        floor = 1e-4
        for kind, colour, name in ((I.KIND_AIMED, "C0", "aimed"), (I.KIND_UNIFORM, "C1", "uniform")):
            m = kinds == kind
            ax[row, 0].scatter(
                np.maximum(dd[m], floor), np.maximum(rr[m], floor), s=4, alpha=0.5, c=colour, label=name
            )
        ax[row, 0].set(
            xscale="log",
            yscale="log",
            xlabel="box displacement (mm)",
            ylabel="|rotation| (deg)",
            title=f"contact interventions: {title}",
        )
        ax[row, 0].axvline(d, c="k", ls="--")
        ax[row, 0].axhline(th, c="k", ls="--")
        ax[row, 0].legend()
        bins = np.logspace(-3, 1.5, 60)
        ax[row, 1].hist(np.maximum(dd, 1e-3), bins=bins)
        ax[row, 1].set(xscale="log", xlabel="box displacement (mm)", ylabel="count", title="displacement")
        ax[row, 1].axvline(d, c="k", ls="--")
        bins = np.logspace(-4, 1.5, 60)
        ax[row, 2].hist(np.maximum(rr, 1e-4), bins=bins)
        ax[row, 2].set(xscale="log", xlabel="|rotation| (deg)", ylabel="count", title="rotation")
        ax[row, 2].axvline(th, c="k", ls="--")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_coverage_and_density(states: dict, pool: dict, path: Path) -> None:
    lab = I.state_label(states)
    ld = states["log_density"]
    kinds = np.broadcast_to(states["kinds"], lab.shape)
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(-6, 8, 57)
    ax.hist(ld[:, 0], bins=bins, histtype="step", density=True, label="logged action at test-like states")
    for kind, name in ((I.KIND_AIMED, "aimed"), (I.KIND_UNIFORM, "uniform")):
        sel = (kinds == kind) & (lab == 2)
        ax.hist(ld[sel], bins=bins, histtype="step", density=True, label=f"{name} interventions, push")
    ax.axvline(policy.LOG_UNIFORM_DENSITY, c="k", ls=":", label="log uniform density")
    ax.set(xlabel="behaviour-policy log-density of the action", ylabel="density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_episodes(smoke: dict, path: Path, n: int = 6) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, i in zip(axes.flat, range(n), strict=False):
        o = smoke["obs"][i]
        ax.plot(o[:, 0], o[:, 1], c="C1", label="box")
        ax.plot(o[:, 7], o[:, 8], c="C0", lw=0.8, label="pusher")
        g = np.unique(smoke["goal"][i], axis=0)
        ax.scatter(g[:, 0], g[:, 1], marker="x", c="k", label="goals")
        ax.set(aspect="equal", title=f"seed {int(smoke['seed'][i])}")
    axes.flat[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docker-replay", type=Path, help="JSON written by the replay runs in the Docker image")
    args = ap.parse_args()
    (OUT / "plots").mkdir(parents=True, exist_ok=True)
    sim = Sim()

    smoke_seeds = E.smoke_seeds("episodes", c.N_SMOKE_EPISODES)
    pool_seeds = E.smoke_seeds("pool", c.N_SMOKE_COVERAGE_POOL_EPISODES)
    state_start = int(E.smoke_seeds("states", 1)[0])
    smoke, t_smoke = timed(E.generate, sim, smoke_seeds, shadow=True)
    pool, t_pool = timed(E.generate, sim, pool_seeds, shadow=True)
    states, t_states = timed(I.make_states, sim, state_start, c.N_SMOKE_STATES)
    for name, d in (("episodes", smoke), ("coverage_pool", pool), ("states", states)):
        E.save(DATA / f"{name}.npz", d)
    print(f"generated: smoke {t_smoke:.1f}s, pool {t_pool:.1f}s, states {t_states:.1f}s", flush=True)

    # Bit-exact replay in a fresh process.
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "replay_check.py"), "--check", str(DATA / "states.npz")],
        capture_output=True,
        text=True,
    )
    replay = json.loads(proc.stdout.strip().splitlines()[-1])
    replay["run"] = "fresh local process"
    replay_entry = {"local": replay, "pass_condition": "every smoke state and all 25 actions byte-identical"}
    if args.docker_replay and args.docker_replay.exists():
        replay_entry["docker"] = json.loads(args.docker_replay.read_text())
    replay_entry["passed"] = bool(replay["passed"] and replay_entry.get("docker", {}).get("passed", True))

    results: dict = {}
    results["bit_exact_replay"] = replay_entry
    results["push_thresholds"] = checks.push_thresholds(states)
    # DESIGN.md rule: move the defaults only to a gap between grazes and pushes that they cut above.
    # Read from plots/push_thresholds.png (top row) before this text was written; see the report.
    results["push_thresholds"]["decision"] = THRESHOLD_DECISION
    results["pusher_can_move_box"] = checks.pusher_can_move_box()
    results["push_rate"] = checks.push_rate(states)
    results["push_label_diagnostic"] = checks.push_label_diagnostic(states)
    results["contact_coverage"] = checks.contact_coverage(smoke)
    results["intervention_coverage"] = checks.intervention_coverage(pool, smoke, states)
    results["policy_log_density"] = checks.policy_log_density(states, pool)
    shadow, t_shadow = timed(checks.shadow_actions, sim, smoke_seeds, smoke)
    results["shadow_actions"] = shadow
    feas, t_feas = timed(checks.positive_control_feasibility, sim, smoke, states)
    results["positive_control_feasibility"] = feas
    print("checks done", flush=True)

    n_ctrl_steps = (smoke["action"].size + pool["action"].size) // 2
    results["compute"] = {
        "wall_s_smoke_episodes": t_smoke,
        "wall_s_coverage_pool": t_pool,
        "wall_s_test_like_states_with_25_actions": t_states,
        "wall_s_shadow_regeneration": t_shadow,
        "wall_s_positive_control_feasibility": t_feas,
        "control_steps_in_episodes": int(n_ctrl_steps),
        "ms_per_episode_control_step": 1e3 * (t_smoke + t_pool) / n_ctrl_steps,
        "projected_s_train_plus_validation": (t_smoke + t_pool)
        / n_ctrl_steps
        * (c.N_TRAIN_EPISODES + c.N_VAL_EPISODES)
        * 100,
        "design_estimate": "data generation is a few CPU-minutes (DESIGN.md, Compute)",
        "training_time": "milestone 2",
        "passed": None,
    }

    plot_thresholds(states, OUT / "plots" / "push_thresholds.png")
    plot_coverage_and_density(states, pool, OUT / "plots" / "log_density.png")
    plot_episodes(smoke, OUT / "plots" / "smoke_episodes.png")

    record = {
        "milestone": "phase0 milestone 1: simulator and environment checks",
        "models_used": "none",
        "seeds": {
            "split_ranges": {k: list(v) for k, v in E.SPLIT_RANGES.items()},
            "smoke_attempt": c.SMOKE_ATTEMPT,
            "smoke_episodes": [int(smoke_seeds[0]), int(smoke_seeds[-1])],
            "coverage_pool": [int(pool_seeds[0]), int(pool_seeds[-1])],
            "test_like_states": [state_start, int(states["seed"][-1])],
            "skipped_state_seeds": states["skipped_seeds"].tolist(),
            "stream_keys": E.STREAM_KEYS,
        },
        "scene_sha256": scene_sha256(),
        "data_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DATA.glob("*.npz"))
        },
        "epsilon_near": policy.EPSILON_NEAR,
        "versions": {
            "mujoco": mujoco.__version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "platform": {"platform": platform.platform(), "machine": platform.machine()},
        "checks": results,
    }
    (OUT / "env_checks.json").write_text(json.dumps(record, indent=2, default=_json_default) + "\n")
    for name, r in results.items():
        print(f"{name}: passed={r.get('passed')}")
    return 0


def _json_default(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x))


if __name__ == "__main__":
    sys.exit(main())
