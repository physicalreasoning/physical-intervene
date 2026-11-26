"""Rehearse freeze 2 and scoring in a scratch copy of the repo, before any of it happens for real.

    uv run --extra models python scripts/rehearse_scoring.py [--out DIR] [--workers N] [--states N]

The steps after training (freeze.py --stage 2, the test states, score.py) cannot run before freeze 2,
so they were never run end to end. This script runs them, unchanged, in a copy of the repo:

1. Copies the working tree (tracked and new, not ignored files) and data/main/ (training,
   validation and pilot files, manifest) to DIR.
2. Writes an untrained checkpoint schedule (`trained: false`, three checkpoints) for every run that
   freeze 2 expects, in the checkpoint format train.py writes, and a placeholder p.json.
3. Runs `freeze.py --stage 2` in the copy.
4. Makes stand-in test states from Phase 0's smoke states (with their history), records them with
   make_data.py's own `record`, and writes a passing replay file.
5. Runs `score.py` in the copy and reports whether it finishes and the result conforms to the format.
   If scoring fails, the copy is refrozen with the test states in place and scored again, to find
   anything further down the path; the first failure is still reported.

No-leak: every checkpoint is untrained (initial weights), as Phase 0's stand-ins, so no main
architecture's intervention error after training is computed; the stand-in test states are smoke
states, never pilot or test states. The real repo is never written: every step runs with the copy's
src/ first on the path, and the script refuses to start if the copy would resolve to this repo.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SMOKE_STATES = "data/phase0/milestone1/attempt2/states.npz"  # read from the real repo
DATA_COPY = (
    "train.npz",
    "validation.npz",
    "pilot.npz",
    "pilot_states.npz",
    "pilot_history.npz",
    "validation_policy_mean.npz",
    "pilot_states_policy_mean.npz",
    "manifest.json",
)
CKPT_UPDATES = (10, 20, 30)  # the untrained checkpoint schedule of every run


def run(cmd: list[str], root: Path, check: bool = True) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    r = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise SystemExit(f"{' '.join(cmd[1:3])} failed:\n{r.stdout[-2000:]}\n{r.stderr[-4000:]}")
    return r


def copy_repo(out: Path) -> None:
    if out.exists():
        raise SystemExit(f"{out} exists; give a new --out")
    files = subprocess.run(  # tracked files and new ones not ignored: the working tree as it is
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    for rel in files.stdout.split():
        src = REPO / rel
        if src.is_file():
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out / rel)
    (out / "data/main").mkdir(parents=True)
    for name in DATA_COPY:
        shutil.copy2(REPO / "data/main" / name, out / "data/main" / name)
    shutil.copy2(Path(__file__), out / "scripts" / Path(__file__).name)


# Steps that run inside the copy (its src/ first on the path) --------------------------------------


def inside_checkpoints() -> None:
    import torch

    from physical_intervene import constants as c
    from physical_intervene import pipeline as P
    from physical_intervene.models import build
    from physical_intervene.seal import ROOT

    td = P.train_data(P.data_path("train"), P.data_path("validation"))
    jobs = P.main_jobs(P.large_width_dropped())
    jobs += [{"family": "mlp-shortcut", "width": "medium", "seed": s} for s in c.MAIN_SEEDS]
    for j in jobs:
        d = P.runs_dir("main") / P.run_name(j["family"], j["width"], j["seed"])
        arch = P.family_arch(j["family"]) if j["family"] != "mlp-shortcut" else "mlp"
        for k, u in enumerate(CKPT_UPDATES):
            torch.manual_seed(1000 * j["seed"] + k)
            model = build(arch, j["width"], td.norm, td.dscaler)
            meta = {
                "architecture": arch,
                "width": j["width"],
                "seed": j["seed"],
                "updates": u,
                "trained": False,
                "phase": "training",
                "family": j["family"],
                "params": model.n_params(),
                "torch": torch.__version__,  # as train.py stores it (a TorchVersion)
                "note": "scoring rehearsal: initial weights, never trained",
            }
            p = d / f"u{u:06d}.pt"
            p.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "meta": meta,
                    "state_dict": model.state_dict(),
                    "input_norm": td.norm.to_dict(),
                    "delta_scaler": td.dscaler.to_dict(),
                    "box_scaler": td.yscaler.to_dict(),
                },
                p,
            )
            p.with_suffix(".json").write_text(json.dumps(meta, indent=1))
        (d / "run.json").write_text(json.dumps({"note": "scoring rehearsal, untrained", "job": j}))
    conf = json.loads((ROOT / "results/pilot/confirmation.json").read_text())
    training = P.results_dir("training")
    training.mkdir(parents=True, exist_ok=True)
    (training / "p.json").write_text(
        json.dumps(
            {
                "note": "scoring rehearsal placeholder (pilot values), not the rule's output",
                "v_star": conf["pilot_v_star"],
                "v_star_set_by": conf["pilot_v_star_set_by"],
                "p": conf["pilot_p"],
                "d_obs": conf["pilot_d_obs"],
                "l_mlp": conf["pilot_l_mlp"],
            },
            indent=1,
        )
    )
    print(json.dumps({"runs": len(jobs), "checkpoints": len(jobs) * len(CKPT_UPDATES)}))


def inside_states(n_states: int, workers: int) -> None:
    import importlib.util

    import numpy as np

    from physical_intervene import episodes as E
    from physical_intervene import pipeline as P
    from physical_intervene.seal import ROOT

    spec = importlib.util.spec_from_file_location("make_data", ROOT / "scripts/make_data.py")
    md = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(md)
    with np.load(Path(os.environ["REHEARSAL_REAL_REPO"]) / SMOKE_STATES) as f:
        d = {k: f[k] for k in f.files}
    keep = {"skipped_seeds", "kinds"}
    states = {k: (v if k in keep else v[:n_states]) for k, v in d.items()}
    path, hpath = P.data_path("test_states"), P.data_path("test_history")
    E.save(path, states)
    md.record(
        False,
        "test_states",
        path,
        {
            "first_seed": int(states["seed"][0]),
            "n_states": n_states,
            "note": "scoring rehearsal: Phase 0 smoke states",
        },
    )
    E.save(hpath, md._histories((states, np.arange(n_states))))  # serial: few states
    md.record(False, "test_history", hpath, {"states_sha256": md.file_sha256(path)})
    replay = {
        "n_states": n_states,
        "n_compared": n_states * 25,
        "n_mismatched": 0,
        "passed": True,
        "note": "scoring rehearsal placeholder",
    }
    (ROOT / "results/test_replay.json").write_text(json.dumps(replay))


# Driver -------------------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--states", type=int, default=40, help="stand-in test states (of Phase 0's 200)")
    ap.add_argument("--inside", choices=["checkpoints", "states"], default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.inside:
        from physical_intervene.seal import ROOT

        if ROOT.resolve() == Path(os.environ.get("REHEARSAL_REAL_REPO", "")).resolve():
            raise SystemExit("refusing: the copy resolves to the real repo")
        inside_checkpoints() if a.inside == "checkpoints" else inside_states(a.states, a.workers)
        return 0

    out = (a.out or Path(os.environ.get("TMPDIR", "/tmp")) / "pi-scoring-rehearsal").resolve()
    if out == REPO or REPO in out.parents:
        raise SystemExit("--out must be outside the repo")
    os.environ["REHEARSAL_REAL_REPO"] = str(REPO)
    py = sys.executable
    me = ["scripts/rehearse_scoring.py"]
    steps: dict[str, dict] = {}

    copy_repo(out)
    where = run([py, "-c", "from physical_intervene.seal import ROOT; print(ROOT)"], out).stdout.strip()
    if Path(where).resolve() != out:
        raise SystemExit(f"the copy's package resolves to {where}, not {out}; stopping")
    run([py, *me, "--inside", "checkpoints"], out)
    steps["checkpoints"] = {"ok": True}
    r = run([py, "scripts/freeze.py", "--stage", "2"], out, check=False)
    steps["freeze2"] = {"returncode": r.returncode, "out": (r.stdout + r.stderr).strip()[-600:]}
    if r.returncode:
        print(json.dumps(steps, indent=1))
        return 1
    run([py, *me, "--inside", "states", "--states", str(a.states), "--workers", str(a.workers)], out)
    steps["test_states"] = {"ok": True}

    def score(tag: str) -> int:
        r = run(
            [py, "scripts/score.py", "--replay", "results/test_replay.json", "--workers", str(a.workers)],
            out,
            check=False,
        )
        res = out / "results/intervention_results.json"
        steps[tag] = {
            "returncode": r.returncode,
            "tail": (r.stdout + r.stderr).strip()[-1500:],
            "format_errors": json.loads(res.read_text()).get("format_errors") if res.exists() else None,
        }
        return r.returncode

    # As in the real run order: freeze 2, then the test states, then scoring.
    if score("score_in_run_order"):
        # Continue the rehearsal past that failure, in the copy only: refreeze with the test states
        # already in the manifest, to find anything further down the scoring path.
        (out / "frozen/models.json").unlink()
        r = run([py, "scripts/freeze.py", "--stage", "2"], out, check=False)
        steps["freeze2_after_test_states"] = {"returncode": r.returncode}
        score("score_after_refreeze")
    print(json.dumps(steps, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
