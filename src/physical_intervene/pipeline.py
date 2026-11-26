"""The run after freeze 1 (DESIGN.md, Seal, Models, Controls): file layout, split seeds, the pilot
rule and the rule for p as pure functions, training jobs, and scoring saved runs into gate families.

Scripts: `make_data.py` (simulator only), `pilot.py`, `train_runs.py`, `freeze.py`, `score.py`.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from physical_intervene import constants as c
from physical_intervene import gates as G
from physical_intervene.seal import ROOT

# Layout ------------------------------------------------------------------------------------------

DATA_FILES = {
    "train": "train.npz",
    "validation": "validation.npz",
    "pilot": "pilot.npz",
    "pilot_states": "pilot_states.npz",
    "pilot_history": "pilot_history.npz",
    "validation_policy_mean": "validation_policy_mean.npz",
    "pilot_states_policy_mean": "pilot_states_policy_mean.npz",
    "test_states": "test_states.npz",
    "test_history": "test_history.npz",
}
# Files that exist before freeze 2 and are hashed in it (the test files are made after it).
FREEZE2_DATA = (
    "train",
    "validation",
    "pilot",
    "pilot_states",
    "pilot_history",
    "validation_policy_mean",
    "pilot_states_policy_mean",
)
SPLIT_EPISODES = {
    "train": (c.SEED_RANGE_TRAIN, c.N_TRAIN_EPISODES),
    "validation": (c.SEED_RANGE_VALIDATION, c.N_VAL_EPISODES),
    "pilot": (c.SEED_RANGE_PILOT, c.N_PILOT_EPISODES),
}
DRY_RUN_EPISODES = {"train": 20, "validation": 5, "pilot": 20}
DRY_RUN_STATES = 8


def data_dir(dry_run: bool = False) -> Path:
    return ROOT / "data" / ("dryrun" if dry_run else "main")


def data_path(key: str, dry_run: bool = False) -> Path:
    return data_dir(dry_run) / DATA_FILES[key]


def runs_dir(stage: str, dry_run: bool = False) -> Path:
    """stage: "pilot" (grid), "pilot_confirmation" or "main"."""
    return ROOT / "data" / ("dryrun_runs" if dry_run else "runs") / stage


def results_dir(stage: str, dry_run: bool = False) -> Path:
    return ROOT / "results" / ("dryrun" if dry_run else "") / stage


def episode_seeds(split: str, dry_run: bool = False) -> np.ndarray:
    (start, stop), n = SPLIT_EPISODES[split]
    if dry_run:
        n = DRY_RUN_EPISODES[split]
    seeds = np.arange(start, start + n, dtype=np.int64)
    if seeds[-1] >= stop:
        raise ValueError(f"{split} seeds out of range")
    return seeds


def pilot_states_start() -> int:
    return c.SEED_RANGE_PILOT[0] + c.PILOT_STATES_OFFSET


# Families and settings ---------------------------------------------------------------------------

TWINS = {a: f"{a}-twin" for a in c.MAIN_ARCHITECTURES}


def family_arch(family: str) -> str:
    """The architecture a family trains: a twin trains its architecture."""
    return family[: -len("-twin")] if family.endswith("-twin") else family


def settings_source(family: str) -> str:
    """The architecture whose pilot settings a family uses (DESIGN.md, Pilot rule, step 6)."""
    if family in ("blind", "mlp-shortcut"):
        return "mlp"
    return family_arch(family)


def run_name(family: str, width: str, seed: int, lr: float | None = None, bs: int | None = None) -> str:
    name = f"{family}_{width}_s{seed}"
    if lr is not None:
        name += f"_lr{lr:g}_bs{bs}"
    return name


def large_width_record(dry_run: bool = False) -> Path:
    return results_dir("training", dry_run) / "large_width.json"


def large_width_dropped(dry_run: bool = False) -> bool:
    """The compute fallback (DESIGN.md, Compute): true once train_runs.py main --drop-large recorded it."""
    f = large_width_record(dry_run)
    return f.exists() and bool(json.loads(f.read_text())["dropped"])


def main_jobs(drop_large: bool = False) -> list[dict]:
    """Every run trained before p is known: main runs at three widths (two if the large width is
    dropped), blind and the twins. The `mlp-shortcut` runs come after (train_runs.py shortcut)."""
    jobs = []
    for a in c.MAIN_ARCHITECTURES:
        jobs += [{"family": a, "width": "medium", "seed": s} for s in c.MAIN_SEEDS]
        for w in ("small",) if drop_large else ("small", "large"):
            jobs += [{"family": a, "width": w, "seed": s} for s in c.SECONDARY_WIDTH_SEEDS]
    jobs += [{"family": "blind", "width": "medium", "seed": s} for s in c.MAIN_SEEDS]
    for a in c.MAIN_ARCHITECTURES:
        jobs += [{"family": TWINS[a], "width": "medium", "seed": s} for s in c.TWIN_SEEDS]
    return jobs


# Pilot rule (DESIGN.md, Models, Pilot rule) ------------------------------------------------------


def setting_scores(rows: list[dict]) -> list[dict]:
    """Group grid runs (dicts with lr, batch_size, seed, best_l_obs, plateau_point) by setting."""
    out: dict[tuple, dict] = {}
    for r in rows:
        k = (r["lr"], r["batch_size"])
        out.setdefault(k, {"lr": r["lr"], "batch_size": r["batch_size"], "runs": []})["runs"].append(r)
    for s in out.values():
        s["score"] = float(np.median([r["best_l_obs"] for r in s["runs"]]))
    return sorted(out.values(), key=lambda s: (s["lr"], s["batch_size"]))


def choose_setting(rows: list[dict], lr_grid: tuple[float, ...]) -> dict:
    """Steps 3 to 5 for one architecture. Returns the chosen setting, whether the lowest-scoring
    setting has an edge learning rate of `lr_grid` (step 4), and the budget."""
    settings = setting_scores(rows)
    low = min(s["score"] for s in settings)
    lowest = min(settings, key=lambda s: s["score"])
    tied = [s for s in settings if s["score"] <= low * (1 + c.PILOT_TIE_REL_TOL)]
    chosen = min(tied, key=lambda s: (s["lr"], s["batch_size"]))
    points = [
        r["plateau_point"] if r["plateau_point"] is not None else c.PILOT_MAX_UPDATES for r in chosen["runs"]
    ]
    edge = None
    if lowest["lr"] == min(lr_grid):
        edge = "low"
    elif lowest["lr"] == max(lr_grid):
        edge = "high"
    return {
        "lr": chosen["lr"],
        "batch_size": chosen["batch_size"],
        "optimiser": c.PILOT_OPTIMISER,
        "updates": int(np.ceil(max(points) * c.BUDGET_FACTOR_OVER_PLATEAU)),
        "plateau_points": [r["plateau_point"] for r in chosen["runs"]],
        "seeds_without_plateau": [r["seed"] for r in chosen["runs"] if r["plateau_point"] is None],
        "lowest": {"lr": lowest["lr"], "batch_size": lowest["batch_size"], "score": lowest["score"]},
        "n_tied": len(tied),
        "edge": edge,
        "settings": [{k: s[k] for k in ("lr", "batch_size", "score")} for s in settings],
    }


def edge_extension(edge: str | None) -> float | None:
    if edge is None:
        return None
    return c.PILOT_LR_EDGE_EXTENSION[0] if edge == "low" else c.PILOT_LR_EDGE_EXTENSION[1]


# Positive control's p (DESIGN.md, Controls) ------------------------------------------------------


def p_rule(l_mlp: float, d_obs: float, v_star: float) -> float | None:
    """The largest p in the grid with L_mlp + p^2 D_obs <= V*, or None (control not run)."""
    ok = [p for p in c.SHORTCUT_P_GRID if l_mlp + p * p * d_obs <= v_star]
    return max(ok) if ok else None


def d_term(target: np.ndarray, mean_outcome: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """(f - m)^2 in std-MSE units, per transition (mean over the 3 box dimensions)."""
    z = (np.asarray(target) - np.asarray(mean_outcome)) / np.asarray(scale)
    return np.mean(z * z, axis=-1)


# Training jobs -----------------------------------------------------------------------------------

_TD: dict = {}


def train_data(train_file: Path, val_file: Path):
    from physical_intervene.data import load_split
    from physical_intervene.train import TrainData

    key = (str(train_file), str(val_file))
    if key not in _TD:
        _TD.clear()
        _TD[key] = TrainData.build(load_split(train_file, "train"), load_split(val_file, "validation"))
    return _TD[key]


def run_job(job: dict) -> dict:
    """Train one run and write run.json (and checkpoints unless job["save"] is False). Skips a run
    whose run.json exists, so an interrupted batch resumes."""
    from physical_intervene.train import TrainConfig, train

    out_dir = Path(job["dir"])
    done = out_dir / "run.json"
    if done.exists():
        return json.loads(done.read_text())
    t0 = time.perf_counter()
    cfg = TrainConfig(
        arch=family_arch(job["family"]),
        width=job["width"],
        seed=int(job["seed"]),
        updates=int(job["updates"]),
        lr=float(job["lr"]),
        batch_size=int(job["batch_size"]),
        seq_batch=16,
        optimiser=job.get("optimiser", c.PILOT_OPTIMISER),
        shortcut_p=job.get("shortcut_p"),
        stop_at_plateau=bool(job.get("stop_at_plateau", False)),
        family=job["family"],
        pilot_confirmation=bool(job.get("pilot_confirmation", False)),
        tag=out_dir.name,
    )
    td = train_data(Path(job["train_file"]), Path(job["val_file"]))
    r = train(cfg, td, ckpt_dir=out_dir if job.get("save", True) else None)
    r.pop("model")
    from physical_intervene.metrics import plateau

    ups = [x["updates"] for x in r["curve"]]
    losses = [x["l_obs"] for x in r["curve"]]
    pl = plateau(ups, losses)
    i = int(np.argmin(losses))
    rec = {
        "job": {k: v for k, v in job.items() if k != "dir"},
        "config": asdict(cfg),
        "params": r["params"],
        "updates_done": r["updates_done"],
        "stopped_at_plateau": r["stopped_at_plateau"],
        "plateau": pl,
        "plateau_point": pl["first_update_met"],
        "best_l_obs": float(losses[i]),
        "best_updates": ups[i],
        "curve": [{"updates": u, "l_obs": float(v)} for u, v in zip(ups, losses, strict=True)],
        "train_seconds": r["train_seconds"],
        "wall_seconds": time.perf_counter() - t0,
        "data": td.data_meta,
    }
    if "shortcut_swapped_share" in r:
        rec["shortcut_swapped_share"] = r["shortcut_swapped_share"]
    out_dir.mkdir(parents=True, exist_ok=True)
    done.write_text(json.dumps(rec, indent=1, default=float))
    return rec


def run_all(jobs: list[dict], workers: int, log=print) -> list[dict]:
    """Run jobs in parallel (one thread each), slowest kinds first."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    cost = {"contact_wm": 4, "gru": 3, "mlp-ms": 2}
    wcost = {"large": 4, "medium": 1.5, "small": 1}
    jobs = sorted(
        jobs, key=lambda j: -cost.get(family_arch(j["family"]), 1) * wcost[j["width"]] * j["updates"]
    )
    out = []
    if workers <= 1:
        for j in jobs:
            out.append(run_job(j))
            log(f"done {Path(j['dir']).name}")
        return out
    with ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(run_job, j): j for j in jobs}
        for f in as_completed(futs):
            out.append(f.result())
            log(f"done {Path(futs[f]['dir']).name} ({len(out)}/{len(jobs)})")
    return out


def load_run(d: Path) -> dict:
    return json.loads((Path(d) / "run.json").read_text())


def checkpoint_paths(d: Path) -> list[Path]:
    return sorted(Path(d).glob("u*.pt"))


# Scoring saved runs into gate families -----------------------------------------------------------


def score_run(run_dir: Path, val, iset, scaler, blocks: tuple[str, ...]) -> dict:
    """L_obs and per-state block sums at every checkpoint of one run, from its saved weights. Goes
    through the scorer's entry point, so the no-leak guard and (after freeze 2) the hash check apply."""
    from physical_intervene import scoring as S

    paths = checkpoint_paths(run_dir)
    l_obs, sums, updates = [], {b: [] for b in blocks}, []
    for p in paths:
        model, meta, dsc = S.load_checkpoint_for_scoring(p)
        pred = S.predict_model(model, dsc, val, iset, meta)
        ls = S.losses(scaler, pred, val.y, iset)
        l_obs.append(ls.l_obs)
        for b in blocks:
            sums[b].append(
                S.state_sums(ls.interventions, iset, b) if b != "logged" else ls.interventions[:, 0]
            )
        updates.append(int(meta["updates"]))
    return {
        "updates": updates,
        "l_obs": np.array(l_obs),
        "sums": {b: np.array(v) for b, v in sums.items()},
        "paths": [str(p) for p in paths],
    }


def family_from_scored(name: str, width: str, seeds: list[int], scored: list[dict], iset, val) -> G.Family:
    """A gate family from scored runs (one per seed, same checkpoint schedule). meta["predict"](i, k)
    reloads checkpoint k of seed i for the descriptive blocks."""
    from physical_intervene import scoring as S

    updates = scored[0]["updates"]
    for s in scored:
        if s["updates"] != updates:
            raise ValueError(f"{name} {width}: seeds have different checkpoint schedules")
    blocks = list(scored[0]["sums"])
    sums = {b: np.stack([s["sums"][b] for s in scored]) for b in blocks}
    counts = {b: iset.counts(b) for b in blocks if b != "logged"}
    counts["logged"] = np.ones(iset.n_states, int)

    def predict(i: int, k: int):
        model, meta, dsc = S.load_checkpoint_for_scoring(Path(scored[i]["paths"][k]))
        return S.predict_model(model, dsc, val, iset, meta)

    return G.Family(
        name,
        list(seeds),
        list(updates),
        np.stack([s["l_obs"] for s in scored]),
        sums,
        counts,
        width,
        {"predict": predict},
    )
