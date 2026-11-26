"""Generate the data of the run after Phase 0 (DESIGN.md, Data, Interventions, Controls), simulator only.

    python scripts/make_data.py --split train|validation|pilot|pilot_states|validation_policy_mean
                                        |pilot_states_policy_mean|test_states [--workers N] [--dry-run]

All ground truth is generated in the pinned image (DESIGN.md, Environment): the script refuses to run
outside linux/x86_64 unless --dry-run, which writes a few episodes to data/dryrun/ to check the
pipeline. Writes data/main/<file>.npz and records its sha256, seeds and versions in
data/main/manifest.json.

- train, validation, pilot: behaviour episodes with shadow actions, consecutive seeds from the
  start of each split's range.
- pilot_states, test_states: test-like states with their 24 labelled interventions, and the model
  inputs' history at each (`*_history.npz`). The test states need freeze 2: their first seed is
  derived from the sha256 of frozen/models.json, and the script checks every recorded hash first.
- validation_policy_mean, pilot_states_policy_mean: m, the box target averaged over 100
  behaviour-policy actions, at every validation transition and every pilot state, for D_obs and
  D_int (DESIGN.md, Controls). No model is involved.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mujoco
import numpy as np

from physical_intervene import constants as c
from physical_intervene import episodes as E
from physical_intervene import interventions as I
from physical_intervene import pipeline as P
from physical_intervene.data import file_sha256
from physical_intervene.env_checks import policy_mean_outcome
from physical_intervene.sim import Sim

EPS_NEAR = c.POLICY_EPSILON_NEAR
STATE_CHUNK = 64


def _episodes(seeds: list[int]) -> dict:
    return E.generate(Sim(), np.asarray(seeds), shadow=True, epsilon_near=EPS_NEAR)


def episodes_parallel(seeds: np.ndarray, workers: int) -> dict:
    chunks = [list(map(int, ch)) for ch in np.array_split(seeds, max(1, min(len(seeds), workers * 4)))]
    with ProcessPoolExecutor(workers) as ex:
        parts = list(ex.map(_episodes, chunks))
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def _state_rows(seeds: list[int]) -> list[dict | None]:
    sim = Sim()
    return [I.state_row(sim, s, EPS_NEAR) for s in seeds]


def states_parallel(start: int, n: int, workers: int) -> dict:
    """The same states as interventions.make_states (consecutive seeds, skipping episodes with
    none), computed in parallel blocks and kept in seed order."""
    rows, skipped, seed = [], [], start
    with ProcessPoolExecutor(workers) as ex:
        while len(rows) < n:
            block = list(range(seed, seed + STATE_CHUNK * workers))
            chunks = [block[i : i + STATE_CHUNK] for i in range(0, len(block), STATE_CHUNK)]
            for ch, res in zip(chunks, ex.map(_state_rows, chunks), strict=True):
                for s, r in zip(ch, res, strict=True):
                    if len(rows) == n:
                        break
                    if r is None:
                        skipped.append(s)
                    else:
                        rows.append(r)
            seed = block[-1] + 1
    return I.stack_states(rows, skipped)


def _histories(args: tuple[dict, list[int]]) -> dict:
    states, idx = args
    part = {k: states[k][idx] for k in ("seed", "step", "obs", "logged_action")}
    return I.state_histories(Sim(), part, EPS_NEAR)


def histories_parallel(states: dict, workers: int) -> dict:
    idx = [list(ch) for ch in np.array_split(np.arange(len(states["seed"])), workers * 4) if len(ch)]
    with ProcessPoolExecutor(workers) as ex:
        parts = list(ex.map(_histories, [(states, i) for i in idx]))
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def _policy_mean_episode(args: tuple[int, np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    seed, state, obs, goal = args
    sim, streams = Sim(), E.policy_streams(seed, "feasibility_")
    return np.stack([policy_mean_outcome(sim, state[t], obs[t], goal[t], streams) for t in range(len(state))])


def policy_mean_episodes(eps: dict, workers: int) -> dict:
    """m at every transition of every episode, each episode's draws from its own feasibility streams."""
    args = [
        (int(eps["seed"][i]), eps["state"][i], eps["obs"][i, :-1], eps["goal"][i])
        for i in range(len(eps["seed"]))
    ]
    with ProcessPoolExecutor(workers) as ex:
        m = np.stack(list(ex.map(_policy_mean_episode, args)))
    return {"mean_outcome": m, "target": eps["target"], "seed": eps["seed"]}


def _policy_mean_state(args: tuple[int, np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    seed, state, obs, goal = args
    return policy_mean_outcome(Sim(), state, obs, goal, E.policy_streams(seed, "feasibility_"))


def policy_mean_states(states: dict, workers: int) -> dict:
    args = [
        (int(states["seed"][i]), states["state"][i], states["obs"][i], states["goal"][i])
        for i in range(len(states["seed"]))
    ]
    with ProcessPoolExecutor(workers) as ex:
        m = np.stack(list(ex.map(_policy_mean_state, args)))
    return {"mean_outcome": m, "seed": states["seed"]}


def environment() -> dict:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
    }


def record(dry_run: bool, key: str, path: Path, info: dict) -> None:
    man = P.data_dir(dry_run) / "manifest.json"
    d = json.loads(man.read_text()) if man.exists() else {}
    d[key] = {
        "file": path.name,
        "sha256": file_sha256(path),
        "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "environment": environment(),
        **info,
    }
    man.write_text(json.dumps(d, indent=1))


def seeds_info(seeds: np.ndarray) -> dict:
    return {"first_seed": int(seeds[0]), "last_seed": int(seeds[-1]), "n": int(len(seeds))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--split",
        required=True,
        choices=[
            *P.SPLIT_EPISODES,
            "pilot_states",
            "test_states",
            "validation_policy_mean",
            "pilot_states_policy_mean",
        ],
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not a.dry_run and not (sys.platform.startswith("linux") and platform.machine() == "x86_64"):
        raise SystemExit("ground truth is generated in the pinned linux/amd64 image (DESIGN.md, Environment)")
    out_dir = P.data_dir(a.dry_run)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    split = a.split
    if split in P.SPLIT_EPISODES:
        seeds = P.episode_seeds(split, a.dry_run)
        path = P.data_path(split, a.dry_run)
        if path.exists():
            raise SystemExit(f"{path} exists; data are generated once")
        E.save(path, episodes_parallel(seeds, a.workers))
        record(a.dry_run, split, path, {**seeds_info(seeds), "epsilon_near": EPS_NEAR, "shadow": True})
    elif split in ("pilot_states", "test_states"):
        if split == "test_states":
            if a.dry_run:
                raise SystemExit("test states exist only after freeze 2; the dry run stops at pilot states")
            from physical_intervene.seal import check_frozen

            start = check_frozen(2)["test_seed_start"]
            n, key_h = c.N_TEST_STATES, "test_history"
        else:
            start, n, key_h = P.pilot_states_start(), c.N_PILOT_STATES, "pilot_history"
        if a.dry_run:
            n = P.DRY_RUN_STATES
        path = P.data_path(split, a.dry_run)
        if path.exists():
            raise SystemExit(f"{path} exists; data are generated once")
        states = states_parallel(start, n, a.workers)
        E.save(path, states)
        record(
            a.dry_run,
            split,
            path,
            {
                "first_seed": start,
                "n_states": n,
                "skipped_seeds": states["skipped_seeds"].tolist(),
                "epsilon_near": EPS_NEAR,
            },
        )
        hpath = P.data_path(key_h, a.dry_run)
        E.save(hpath, histories_parallel(states, a.workers))
        record(a.dry_run, key_h, hpath, {"states_sha256": file_sha256(path)})
    elif split == "validation_policy_mean":
        src = P.data_path("validation", a.dry_run)
        path = P.data_path(split, a.dry_run)
        E.save(path, policy_mean_episodes(E.load(src), a.workers))
        record(
            a.dry_run,
            split,
            path,
            {"from": src.name, "from_sha256": file_sha256(src), "policy_draws": c.FEASIBILITY_POLICY_DRAWS},
        )
    elif split == "pilot_states_policy_mean":
        src = P.data_path("pilot_states", a.dry_run)
        path = P.data_path(split, a.dry_run)
        E.save(path, policy_mean_states(E.load(src), a.workers))
        record(
            a.dry_run,
            split,
            path,
            {"from": src.name, "from_sha256": file_sha256(src), "policy_draws": c.FEASIBILITY_POLICY_DRAWS},
        )
    print(json.dumps({"split": split, "dir": str(out_dir), "seconds": round(time.perf_counter() - t0, 1)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
