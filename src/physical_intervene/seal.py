"""Sealing (DESIGN.md, Seal): the two freeze records, the phase they imply, and the hash check that
every run after freeze 1 makes first.

- Freeze 1, `frozen/design.json`: DESIGN.md's sha256, the sha256 of every file that generates data,
  trains or scores (FREEZE1_CODE), versions, the image digest, the Phase 0 record, the seeds and the
  values Phase 0 fixed. Written by `scripts/freeze.py --stage 1`.
- Freeze 2, `frozen/models.json`: the pilot's settings, V*, p with D_obs and L_mlp, the sha256 of
  the data files and of every checkpoint. The test seed is derived from this file's sha256.
- Amendments (DESIGN.md, Deviations after freeze 1): `frozen/amendments.json` lists each file in
  `frozen/amendments/` with its sha256 and, if the amendment changes a hashed file, that file's new
  sha256. A hashed file may differ from its freeze 1 hash only if an amendment records its new one.
  Freeze 2 hashes `frozen/amendments.json` itself.

`check_frozen` raises SystemExit naming anything that changed, as forward's `seal.check_frozen`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from physical_intervene import constants as c

ROOT = Path(__file__).resolve().parents[2]
FROZEN = "frozen"
DESIGN_RECORD = f"{FROZEN}/design.json"
MODELS_RECORD = f"{FROZEN}/models.json"
AMENDMENTS_DIR = f"{FROZEN}/amendments"
AMENDMENTS_RECORD = f"{FROZEN}/amendments.json"

DESIGN_FILE = "docs/DESIGN.md"
PHASE0_RECORD = ("results/phase0_smoke.json", "results/phase0/REPORT_phase0.md")
# Scripts that generate data, train or score after freeze 1, and the freeze itself.
FREEZE1_SCRIPTS = (
    "scripts/make_data.py",
    "scripts/pilot.py",
    "scripts/train_runs.py",
    "scripts/score.py",
    "scripts/freeze.py",
    "scripts/replay_check.py",
)
FREEZE1_ENV = ("Dockerfile", "pyproject.toml", "uv.lock")

PHASES = ("phase0", "training", "scoring")


def sha256_file(path: str | Path, root: Path = ROOT) -> str:
    return hashlib.sha256((Path(root) / path).read_bytes()).hexdigest()


def package_files(root: Path = ROOT) -> list[str]:
    """Every file of the package (code, scene file and other assets), sorted; no caches."""
    pkg = Path(root) / "src" / "physical_intervene"
    out = [
        p.relative_to(root).as_posix()
        for p in pkg.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    ]
    return sorted(out)


def freeze1_code(root: Path = ROOT) -> list[str]:
    return [*package_files(root), *FREEZE1_SCRIPTS, *FREEZE1_ENV]


def locked_versions(name: str, lock_text: str) -> list[dict]:
    """Every version of a package pinned in uv.lock, with the Python versions it applies to (uv
    pins one version per Python range when they differ, e.g. numpy for 3.11 and for 3.12+)."""
    import tomllib

    return [
        {"version": p["version"], "python": p.get("resolution-markers", ["any"])}
        for p in tomllib.loads(lock_text).get("package", [])
        if p.get("name") == name
    ]


def current_phase(root: Path = ROOT) -> str:
    """phase0 before freeze 1; training between the freezes; scoring after freeze 2."""
    if (Path(root) / MODELS_RECORD).exists():
        return "scoring"
    if (Path(root) / DESIGN_RECORD).exists():
        return "training"
    return "phase0"


def test_seed_start(models_sha256: str) -> int:
    """The first test-state seed: TEST_SEED_BASE + the first 12 hex digits of freeze 2's sha256."""
    return c.TEST_SEED_BASE + int(models_sha256[:12], 16)


def _load(path: str, root: Path) -> dict:
    p = Path(root) / path
    if not p.exists():
        raise SystemExit(f"{path} does not exist: that freeze has not happened")
    return json.loads(p.read_text())


def _changed(files: dict[str, str], root: Path) -> list[str]:
    out = []
    for p, h in files.items():
        f = Path(root) / p
        if not f.exists():
            out.append(f"{p} (missing)")
        elif sha256_file(p, root) != h:
            out.append(p)
    return out


def check_amendments(root: Path = ROOT) -> list[dict]:
    """The amendments in date order: each file in frozen/amendments/ is listed in amendments.json,
    and none has changed. Each entry: {"file", "sha256", "code_sha256": {path: new sha256}}."""
    d = Path(root) / AMENDMENTS_DIR
    present = sorted(p.relative_to(root).as_posix() for p in d.glob("*")) if d.exists() else []
    rec_path = Path(root) / AMENDMENTS_RECORD
    listed = json.loads(rec_path.read_text())["amendments"] if rec_path.exists() else []
    names = [a["file"] for a in listed]
    problems = [f"{p} (not listed in {AMENDMENTS_RECORD})" for p in present if p not in names]
    problems += _changed({a["file"]: a["sha256"] for a in listed}, root)
    if problems:
        raise SystemExit(f"amendments changed or unlisted: {problems}")
    return listed


def amended_code_sha256(design: dict, amendments: list[dict]) -> dict[str, str]:
    """Freeze 1's code hashes, with each amendment's new hashes applied in order."""
    code = dict(design["code_sha256"])
    for a in amendments:
        code.update(a.get("code_sha256", {}))
    return code


def check_frozen(stage: int, root: Path = ROOT, checkpoints: bool = False) -> dict:
    """Load the freeze records up to `stage` and refuse (SystemExit) if any recorded hash differs.
    Stage 2 also checks the data files; `checkpoints=True` re-hashes every checkpoint as well."""
    design = _load(DESIGN_RECORD, root)
    amendments = check_amendments(root)
    code = amended_code_sha256(design, amendments)
    files = {design["design_file"]: design["design_sha256"], **code}
    files.update(design["phase0_record_sha256"])
    changed = _changed(files, root)
    recorded_pkg = {p for p in code if p.startswith("src/")}
    changed += [f"{p} (added after freeze 1)" for p in package_files(root) if p not in recorded_pkg]
    out = {"design": design, "amendments": amendments}
    if stage >= 2:
        models = _load(MODELS_RECORD, root)
        if models["design_record_sha256"] != sha256_file(DESIGN_RECORD, root):
            changed.append(DESIGN_RECORD)
        rec = Path(root) / AMENDMENTS_RECORD
        now = sha256_file(AMENDMENTS_RECORD, root) if rec.exists() else None
        if models.get("amendments_record_sha256") != now:
            changed.append(f"{AMENDMENTS_RECORD} (differs from freeze 2)")
        changed += _changed(models["data_sha256"], root)
        if checkpoints:
            changed += _changed(models["checkpoint_sha256"], root)
        out["models"] = models
        out["models_sha256"] = sha256_file(MODELS_RECORD, root)
        out["test_seed_start"] = test_seed_start(out["models_sha256"])
    if changed:
        raise SystemExit(f"changed since the freeze: {changed}")
    return out
