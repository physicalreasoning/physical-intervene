"""Bit-exact replay check (docs/SMOKE_TEST.md), meant to run in a fresh process.

    python scripts/replay_check.py --generate data/smoke/states.npz   # make states and truth
    python scripts/replay_check.py --check data/smoke/states.npz      # restore, rerun, compare

--check restores every stored state, reruns its 25 stored actions and compares each resulting
full integration state with the stored one byte for byte. It prints a JSON summary and exits 1 on
any mismatch. --generate makes the smoke test-like states alone (no coverage pool), for running
both steps inside the Docker image.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import mujoco

from physical_intervene import constants as c
from physical_intervene import env_checks, episodes, interventions
from physical_intervene.sim import Sim


def main() -> int:
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--generate", type=Path)
    group.add_argument("--check", type=Path)
    args = ap.parse_args()
    env = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "mujoco": mujoco.__version__,
    }
    if args.generate:
        start = int(episodes.smoke_seeds("states", 1)[0])
        states = interventions.make_states(Sim(), start, c.N_SMOKE_STATES)
        episodes.save(args.generate, states)
        print(json.dumps({"generated": str(args.generate), "n_states": c.N_SMOKE_STATES, **env}))
        return 0
    result = env_checks.replay(episodes.load(args.check))
    print(json.dumps({**result, **env}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
