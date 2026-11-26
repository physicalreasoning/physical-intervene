"""Write a freeze record (DESIGN.md, Seal), as forward's scripts/freeze.py.

    uv run python scripts/freeze.py --stage 1 --image physical-intervene
    uv run python scripts/freeze.py --stage 2

Stage 1, before any training on the training split: frozen/design.json with DESIGN.md's sha256, the
sha256 of every file that generates data, trains or scores (seal.freeze1_code), the Phase 0 record,
the MuJoCo, Python and torch versions, the image digest, every seed and the values Phase 0 fixed.
Run on a clean checkout of main, after building the image from it.

Stage 2, after training, before any test state exists: frozen/models.json with the pilot's settings
and confirmation, V*, p with D_obs and L_mlp, and the sha256 of the data files, of every run record
and of every checkpoint. The test seed is derived from this file's sha256 (printed).

A record is written once. Then commit only that file, push, and post the commit hash and the
recorded sha256 on physicalreasoning/forward#5. --dry-run writes to data/dryrun/frozen/ and skips
the clean-checkout and image checks.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import subprocess
import sys

from physical_intervene import constants as c
from physical_intervene import episodes as E
from physical_intervene import pipeline as P
from physical_intervene import seal
from physical_intervene.seal import ROOT, sha256_file


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def locked(*names: str) -> dict:
    text = (ROOT / "uv.lock").read_text()
    return {n: seal.locked_versions(n, text) for n in names}


def image_info(image: str) -> dict:
    inspect = json.loads(
        subprocess.run(
            ["docker", "image", "inspect", image], check=True, capture_output=True, text=True
        ).stdout
    )[0]
    probe = (
        "import json, platform, mujoco, numpy; "
        "print(json.dumps({'python': platform.python_version(), 'mujoco': mujoco.__version__, "
        "'numpy': numpy.__version__, 'machine': platform.machine()}))"
    )
    run = subprocess.run(
        ["docker", "run", "--rm", image, "python", "-c", probe], check=True, capture_output=True, text=True
    )
    return {
        "name": image,
        "id": inspect["Id"],
        "repo_digests": inspect.get("RepoDigests", []),
        "inside": json.loads(run.stdout),
    }


def seeds() -> dict:
    span = lambda split: {"first": int(P.episode_seeds(split)[0]), "n": int(len(P.episode_seeds(split)))}  # noqa: E731
    return {
        "train_episodes": span("train"),
        "validation_episodes": span("validation"),
        "pilot_episodes": span("pilot"),
        "pilot_states_first_seed": P.pilot_states_start(),
        "smoke": {
            "attempt": c.SMOKE_ATTEMPT,
            "episodes_first": int(E.smoke_seeds("episodes", 1)[0]),
            "pool_first": int(E.smoke_seeds("pool", 1)[0]),
            "states_first": int(E.smoke_seeds("states", 1)[0]),
        },
        "main_training": list(c.MAIN_SEEDS),
        "secondary_width_training": list(c.SECONDARY_WIDTH_SEEDS),
        "twin_training": list(c.TWIN_SEEDS),
        "pilot_training": list(c.PILOT_TRAINING_SEEDS),
        "shadow_action_streams": "per episode seed: SeedSequence(seed, spawn_key=(k,)) with k = "
        f"{E.STREAM_KEYS['shadow_eps']}, {E.STREAM_KEYS['shadow_noise']}, {E.STREAM_KEYS['shadow_uniform']}",
        "test": f"from {c.TEST_SEED_BASE} + the first 12 hex digits of sha256(frozen/models.json)",
    }


def stage1(a) -> dict:
    if not a.dry_run:
        if git("status", "--porcelain"):
            raise SystemExit("the checkout has uncommitted changes; freeze from a clean main")
        if not a.image:
            raise SystemExit("--image is required: the pinned image built from this checkout")
    return {
        "freeze": 1,
        "frozen_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "HEAD"),
        "design_file": seal.DESIGN_FILE,
        "design_sha256": sha256_file(seal.DESIGN_FILE),
        "code_sha256": {p: sha256_file(p) for p in seal.freeze1_code()},
        "phase0_record_sha256": {p: sha256_file(p) for p in seal.PHASE0_RECORD},
        "versions": {
            "python_host": platform.python_version(),
            "uv_lock": locked("torch", "mujoco", "numpy", "scipy"),
            "note": "the image's own Python, MuJoCo and numpy are under image.inside; torch is not in "
            "the image (data generation and replay do not use it)",
        },
        "image": image_info(a.image) if a.image else None,
        "seeds": seeds(),
        "phase0_values": {
            "d_push_m": c.PUSH_MIN_DISPLACEMENT_M,
            "theta_push_deg": c.PUSH_MIN_ROTATION_DEG,
            "epsilon_near": c.POLICY_EPSILON_NEAR,
        },
    }


def expected_runs(p: float | None) -> list[str]:
    names = [P.run_name(j["family"], j["width"], j["seed"]) for j in P.main_jobs(P.large_width_dropped())]
    if p is not None:
        names += [P.run_name("mlp-shortcut", "medium", s) for s in c.MAIN_SEEDS]
    return names


def stage2(a) -> dict:
    if not a.dry_run:
        seal.check_frozen(1)
    pilot, training = P.results_dir("pilot", a.dry_run), P.results_dir("training", a.dry_run)
    files = {
        "settings": pilot / "settings.json",
        "confirmation": pilot / "confirmation.json",
        "p": training / "p.json",
    }
    if P.large_width_record(a.dry_run).exists():
        files["large_width"] = P.large_width_record(a.dry_run)
    rec = {k: json.loads(f.read_text()) for k, f in files.items()}
    rd = P.runs_dir("main", a.dry_run)
    present = sorted(d.name for d in rd.iterdir() if (d / "run.json").exists())
    if not a.dry_run:
        missing = sorted(set(expected_runs(rec["p"]["p"])) - set(present))
        if missing:
            raise SystemExit(f"runs not finished: {missing[:5]} ({len(missing)} in all)")
    ckpt = {}
    for name in present:
        for f in sorted((rd / name).iterdir()):
            if f.suffix in (".pt", ".json"):
                rel = f.relative_to(ROOT).as_posix()
                ckpt[rel] = sha256_file(rel)
    data = {P.data_path(k, a.dry_run).relative_to(ROOT).as_posix(): None for k in P.FREEZE2_DATA}
    data[(P.data_dir(a.dry_run) / "manifest.json").relative_to(ROOT).as_posix()] = None
    data = {k: sha256_file(k) for k in data}
    data.update({f.relative_to(ROOT).as_posix(): sha256_file(f.relative_to(ROOT)) for f in files.values()})
    design_rec = seal.DESIGN_RECORD if not a.dry_run else "data/dryrun/frozen/design.json"
    return {
        "freeze": 2,
        "frozen_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "design_record_sha256": sha256_file(design_rec),
        "amendments_record_sha256": (
            sha256_file(seal.AMENDMENTS_RECORD) if (ROOT / seal.AMENDMENTS_RECORD).exists() else None
        ),
        "settings": rec["settings"]["settings"],
        "pilot_confirmation": rec["confirmation"],
        "v_star": rec["p"]["v_star"],
        "v_star_set_by": rec["p"]["v_star_set_by"],
        "p": rec["p"]["p"],
        "d_obs": rec["p"]["d_obs"],
        "l_mlp": rec["p"]["l_mlp"],
        "large_width_dropped": P.large_width_dropped(a.dry_run),
        "n_runs": len(present),
        "data_sha256": data,
        "checkpoint_sha256": ckpt,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=[1, 2], required=True)
    ap.add_argument("--image", default=None, help="stage 1: the pinned image's name or tag")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    out_dir = ROOT / ("data/dryrun/frozen" if a.dry_run else seal.FROZEN)
    out = out_dir / ("design.json" if a.stage == 1 else "models.json")
    if out.exists():
        raise SystemExit(f"{out.relative_to(ROOT)} exists: each freeze is written once")
    rec = stage1(a) if a.stage == 1 else stage2(a)
    out_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1, default=float))
    h = sha256_file(out.relative_to(ROOT))
    print(f"{out.relative_to(ROOT)} sha256 {h}")
    if a.stage == 1:
        print(f"DESIGN.md sha256 {rec['design_sha256']}")
    else:
        print(f"test states start at seed {seal.test_seed_start(h)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
