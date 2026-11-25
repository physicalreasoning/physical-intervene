"""The Phase 0 record (SMOKE_TEST.md, Record): results/phase0_smoke.json.

    uv run python scripts/phase0_record.py

Collects the current attempt of each milestone (environment checks, training-side checks,
intervention-side checks), the values Phase 0 fixes for freeze 1, every rerun and its reason, the
sha256 of every record it summarises, and the two lists of quantities computed (training side,
intervention side). Reads only result files; runs nothing.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from physical_intervene import constants as c

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "phase0"
M1 = RES / "milestone1" / f"attempt{c.SMOKE_ATTEMPT}"
M2 = RES / "milestone2" / f"attempt{c.MILESTONE2_ATTEMPT}"
M3 = RES / "milestone3" / f"attempt{c.MILESTONE3_ATTEMPT}"

RERUNS = [
    {
        "milestone": 1,
        "attempt": 0,
        "folder": "results/phase0/milestone1/attempt0/",
        "reason": "first run, epsilon_near 0.2",
        "outcome": "intervention coverage failed (T 0.066 on the earlier distance, needs 0.25); the other "
        "gated checks passed",
    },
    {
        "milestone": 1,
        "attempt": 1,
        "folder": "results/phase0/milestone1/attempt1/",
        "reason": "rerun on new smoke seeds: coverage measure replaced (action novelty given the state), "
        "epsilon_near lowered to 0.05",
        "outcome": "every check passed",
    },
    {
        "milestone": 1,
        "attempt": 2,
        "folder": "results/phase0/milestone1/attempt2/",
        "reason": "rerun on new smoke seeds: push label measured against the no-contact reference (a box "
        "already sliding is not counted as pushed by a light touch)",
        "outcome": "every check passed; bit-exact replay also on real amd64 hardware",
    },
    {
        "milestone": 2,
        "attempt": 0,
        "folder": "results/phase0/milestone2/attempt0/",
        "reason": "first run, with rssm as the fourth architecture and checkpoints every 100 updates",
        "outcome": "rssm learned slowest, set V* and failed two checks; "
        "contact_wm skipped the band around V* "
        "between two checkpoints",
    },
    {
        "milestone": 2,
        "attempt": 1,
        "folder": "results/phase0/milestone2/attempt1/",
        "reason": "mlp-ms in place of rssm; checkpoints every 10 updates to 1,000, every 100 to 2,000, then "
        "every 1,000",
        "outcome": "every training-side check passed",
    },
    {
        "milestone": 3,
        "attempt": 0,
        "folder": "results/phase0/milestone3/attempt0/",
        "reason": "first run of the intervention side with stand-ins "
        "(10 seeds, bootstrap over states and seeds)",
        "outcome": "every check passed after two SMOKE_TEST.md wordings "
        "(single-draw twin interval; constvel); "
        "G1's interval found too narrow when seeds disagree a lot on pushes (up to 9% false alarms)",
    },
    {
        "milestone": 3,
        "attempt": 1,
        "folder": "results/phase0/milestone3/attempt1/",
        "reason": "design change: 20 seeds per architecture and Welch's t-test over the seeds for G1",
        "outcome": "see intervention_checks",
    },
]

TRAINING_SIDE = [
    "data checks: alignment, target and angle wrap, scaler from the training split only, split overlap",
    "baselines copy, constvel and knn: L_obs on the validation transitions (recorded actions)",
    "training loss and L_obs (with corner RMSE) at every checkpoint of every learning run, "
    "recorded actions only",
    "L_obs on validation transitions split by contact label and by box motion (log entry, ad hoc, not kept)",
    "made-up-target training losses (zero, action, history, smooth function; contact_wm and mlp-ms variants)",
    "overfit loss on 64 examples",
    "contact_wm's learned physical values",
    "mlp-shortcut swapped share (stand-in p 0.5); controls trained, not scored",
    "V*-like level, matched checkpoints, exclusions and match screen on smoke runs",
    "plateau criterion; reproducibility of curves; timings",
    "exploration: pnn and rssm-dv3 L_obs curves and made-up targets",
]
INTERVENTION_SIDE = [
    "simulator only (milestone 1): replay of the 25 actions at each smoke state, push thresholds and labels, "
    "push rate, intervention coverage, policy log-density, shadow actions, D_obs and D_int (positive-control "
    "feasibility)",
    "stand-ins (milestone 3, attempts 0 and 1): oracle, copy and constvel (fixed rules) and noisy stand-in "
    "families scored on "
    "validation transitions and on all 25 actions at every smoke state: L_obs, std-MSE per stratum, corner "
    "and angle errors, logged-action reference",
    "untrained networks (mlp, gru, mlp-ms, contact_wm, 3 seeds each, trained: false): L_obs, through the "
    "scoring entry point; their push std-MSE was computed in earlier runs, never reported, and deleted",
    "with stand-ins only: V*, matching, exclusions, match screen, gaps, intervals (seed-level; states "
    "only), p, Holm, G0.2, G0.3, G1, control pairs and readings, G2, best_push_gap, matched_only, gap along "
    "training, G1 without the V*-setting architecture, G2 over the general-purpose runs, per-seed view",
    "repeated stand-in pairs (twins, noisy-c, noisy-b, false alarm at several bias scales): separation rates",
    "exploration with stand-ins: G1's false-alarm rate and power for several interval methods, at 200 and "
    "1,000 states, with 10 and 20 seeds",
    "the stand-in run's result JSON and its format check",
]
NEVER_COMPUTED = [
    "the intervention error (any stratum) of any trained model, main or control",
    "any gap, G1, G2 or control-pair result between trained models",
    "any measure of a trained model's reliance on the action using real-style data",
    "anything on pilot seeds or test states",
]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    env = json.loads((M1 / "env_checks.json").read_text())
    amd = json.loads((M1 / "amd64_replay.json").read_text())
    tr = json.loads((M2 / "training_checks.json").read_text())
    it = json.loads((M3 / "intervention_checks.json").read_text())
    if it.get("quick"):
        raise SystemExit("milestone 3 record is a --quick run; rerun it in full first")
    env_pass = {k: v.get("passed") for k, v in env["checks"].items() if isinstance(v, dict) and "passed" in v}
    rep = it["checks"]["repeated_pairs"]
    record = {
        "phase": "phase0",
        "status": "complete: every gated Phase 0 check ran; see failed_checks and open issues in the report",
        "freeze1_values": {
            "d_push_m": c.PUSH_MIN_DISPLACEMENT_M,
            "theta_push_deg": c.PUSH_MIN_ROTATION_DEG,
            "epsilon_near": c.POLICY_EPSILON_NEAR,
            "smoke_attempt": c.SMOKE_ATTEMPT,
            "smoke_seeds": env["seeds"],
            "push_threshold_decision": "defaults kept: no gap between grazes and pushes",
            "graze_share_of_contacts": env["checks"]["push_thresholds"]["graze_share_of_contacts"],
        },
        "reruns": RERUNS,
        "environment_checks": {
            "attempt": c.SMOKE_ATTEMPT,
            "gated": env_pass,
            "amd64_replay": amd["replay_check"]["passed"],
            "push_rate": {
                k: env["checks"]["push_rate"][k] for k in ("frac_states_with_push", "mean_push_per_state")
            },
            "intervention_coverage_T": env["checks"]["intervention_coverage"]["T"],
            "positive_control_feasibility": {
                k: env["checks"]["positive_control_feasibility"].get(k)
                for k in ("D_obs", "D_int", "D_int_over_D_obs")
            },
        },
        "training_checks": {"attempt": c.MILESTONE2_ATTEMPT, "summary": tr["summary"]},
        "intervention_checks": {
            "attempt": c.MILESTONE3_ATTEMPT,
            "summary": {k: v["pass"] for k, v in it["checks"].items()},
            "standin_scales": it["standin_scales"],
            "repeated_pairs": {
                "draws": rep["draws_per_pair"],
                "twins_false_alarm_full": rep["twins"]["rate_full"],
                "twins_false_alarm_states_only": rep["twins"]["rate_states_only"],
                "twins_interval_excludes_zero_full": rep["twins"]["interval_excludes_zero_full"],
                "twins_interval_excludes_zero_states_only": rep["twins"][
                    "interval_excludes_zero_states_only"
                ],
                "noisy_c_detection_full": rep["noisy_c"]["rate_full"],
                "noisy_b_detection_full": rep["noisy_b"]["rate_full"],
                "false_alarm_by_seed_cv": {
                    k: {"full": v["rate_full"], "states_only": v["rate_states_only"]}
                    for k, v in rep["false_alarm_by_seed_cv"].items()
                },
            },
        },
        "failed_checks": [f"milestone 3: {k}" for k, v in it["checks"].items() if v["pass"] is False],
        "records_sha256": {
            str(p.relative_to(ROOT)): sha(p)
            for p in (
                M1 / "env_checks.json",
                M1 / "amd64_replay.json",
                M2 / "training_checks.json",
                M3 / "intervention_checks.json",
                M3 / "intervention_results_standin.json",
            )
        },
        "quantities_computed": {"training_side": TRAINING_SIDE, "intervention_side": INTERVENTION_SIDE},
        "never_computed": NEVER_COMPUTED,
        "report": "results/phase0/REPORT_phase0.md",
    }
    out = ROOT / "results" / "phase0_smoke.json"
    out.write_text(json.dumps(record, indent=1))
    print(json.dumps(record["intervention_checks"]["summary"], indent=1))
    print("failed:", record["failed_checks"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
