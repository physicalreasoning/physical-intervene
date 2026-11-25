"""Phase 0 exploration: candidates to replace or fix rssm, on smoke data (training side only).

Trains candidates with milestone 2's placeholder settings and compares them with the first
milestone 2 run, with rssm (results/phase0/milestone2/attempt0/training_checks.json):
- pnn: mlp's backbone trained on a Gaussian NLL (PETS-style), prediction = mean;
- rssm-dv3: the design's rssm with DreamerV3's KL terms (free nats 1, KL balancing), scored with
  the design's rule (16 samples) and deterministically (posterior means, prior mean);
- mlp-ms: mlp trained on a weighted multi-step (open-loop rollout) loss.

Runs: learning runs at medium width (pnn and mlp-ms 3 seeds, rssm-dv3 2 seeds), pnn and mlp-ms
small and large (seed 11), the made-up targets (pnn all four, rssm-dv3 the history target rssm
failed). Then the matching rule on the medium runs with each candidate in place of rssm. No
interventions; never opens states.npz.

    uv run python scripts/phase0_model_exploration.py [--workers N] [--candidates mlp-ms ...]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import phase0_training as P  # noqa: E402

from physical_intervene import constants as c  # noqa: E402
from physical_intervene import train as T  # noqa: E402
from physical_intervene import training_checks as TC  # noqa: E402
from physical_intervene.data import box_target_from_delta  # noqa: E402
from physical_intervene.train import fit_loss, loss_weights, train, weighted_loss  # noqa: E402

OUT = ROOT / "results" / "phase0" / "exploration"
M2 = ROOT / "results" / "phase0" / "milestone2" / "attempt0" / "training_checks.json"  # with rssm
CANDIDATES = {"pnn": ("pnn", "design"), "rssm-dv3": ("rssm", "dv3"), "mlp-ms": ("mlp-ms", "design")}


def _l_obs_mean_rule(model, td) -> float:
    """rssm-dv3 scored with its deterministic prediction."""
    with torch.no_grad():
        z = model.predict_mean(td.val_batch()).double().numpy()
    delta = z * td.dscaler.scale + td.dscaler.mean
    return td.yscaler.std_mse(box_target_from_delta(td.val.s, delta), td.val.y)


def run_job(job: dict) -> dict:
    t0 = time.perf_counter()
    arch, variant = CANDIDATES[job["candidate"]]
    if job["kind"] == "learn":
        td = P.train_data()
        cfg = P.config(arch, job["width"], job["seed"], job["updates"], rssm_variant=variant)
        orig = T.evaluate
        if arch == "rssm":  # also score the deterministic rule at every checkpoint

            def evaluate(model, td_):
                row = orig(model, td_)
                row["l_obs_mean_rule"] = _l_obs_mean_rule(model, td_)
                return row

            T.evaluate = evaluate
        try:
            r = train(cfg, td)
        finally:
            T.evaluate = orig
        out = {"curve": r["curve"], "params": r["params"], "train_seconds": r["train_seconds"]}
    else:  # made-up target, as in milestone 2's check
        td = P.train_data(job["target"])
        cfg = P.config(
            arch,
            "medium",
            11,
            job["updates"],
            loss_dims=job["dims"],
            target_name=job["target"],
            rssm_variant=variant,
        )
        r = train(cfg, td, evaluate_val=False)
        idx = np.random.default_rng(0).choice(
            len(td.train), P.CHECK_SETTINGS["made_up_eval_examples"], replace=False
        )
        out = {"fit_loss": fit_loss(r["model"], td, idx, job["dims"])}
        if arch == "rssm":
            with torch.no_grad():
                zp = r["model"].predict_mean(td.batch(idx))
            out["fit_loss_mean_rule"] = float(weighted_loss(zp, td.z(idx), loss_weights(job["dims"])))
    out.update(job=job, name=job["name"], config=r["config"], wall_seconds=time.perf_counter() - t0)
    return out


def jobs() -> list[dict]:
    out = [{"kind": "learn", "candidate": "rssm-dv3", "width": "medium", "seed": s} for s in (11, 22)]
    out += [{"kind": "learn", "candidate": "pnn", "width": "medium", "seed": s} for s in (11, 22, 33)]
    out += [{"kind": "learn", "candidate": "pnn", "width": w, "seed": 11} for w in ("small", "large")]
    out.append({"kind": "madeup", "candidate": "rssm-dv3", "target": "history", "dims": None})
    for target, dims in TC.MADE_UP_PLAN["mlp"]:
        out.append({"kind": "madeup", "candidate": "pnn", "target": target, "dims": dims})
    # mlp-ms: no made-up targets; a made-up state change defines no states to roll out over, and
    # its network and one-step wiring are mlp's, which pass them.
    out += [{"kind": "learn", "candidate": "mlp-ms", "width": "medium", "seed": s} for s in (11, 22, 33)]
    out += [{"kind": "learn", "candidate": "mlp-ms", "width": w, "seed": 11} for w in ("small", "large")]
    for j in out:
        j["updates"] = (
            P.PLACEHOLDERS["updates"] if j["kind"] == "learn" else P.CHECK_SETTINGS["made_up_updates"]
        )
        j["name"] = "_".join(str(j[k]) for k in ("kind", "candidate", "width", "seed", "target") if k in j)
    return out


def matching(m2: dict, runs: list[dict]) -> dict:
    """The matching rule on medium runs: mlp, gru, contact_wm from milestone 2, plus each
    candidate (and the design's rssm, for reference) in turn."""
    base = []
    for name, curve in m2["curves"].items():
        parts = name.split("_")
        if parts[0] != "learn" or "medium" not in parts:
            continue
        arch = "contact_wm" if name.startswith("learn_contact_wm") else parts[1]
        base.append({"name": name, "config": {"arch": arch, "width": "medium"}, "curve": curve})
    out = {}
    fourth = {"rssm (design)": [r for r in base if r["config"]["arch"] == "rssm"]}
    for cand in CANDIDATES:
        fourth[cand] = [
            {"name": r["name"], "config": {"arch": cand, "width": "medium"}, "curve": r["curve"]}
            for r in runs
            if r["job"]["kind"] == "learn" and r["job"]["candidate"] == cand and r["job"]["width"] == "medium"
        ]
    three = [r for r in base if r["config"]["arch"] in ("mlp", "gru", "contact_wm")]
    for label, rs in fourth.items():
        archs = ("mlp", "gru", "contact_wm", rs[0]["config"]["arch"])
        out[label] = TC.check_matching(three + rs, main=archs)
    out["no fourth model (mlp, gru, contact_wm)"] = TC.check_matching(
        three, main=("mlp", "gru", "contact_wm")
    )
    return out


def plot(m2: dict, runs: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colours = {"mlp": "tab:blue", "gru": "tab:green", "contact_wm": "tab:red", "rssm": "tab:grey"}
    for name, curve in m2["curves"].items():
        if "medium" not in name or not name.startswith("learn_"):
            continue
        arch = "contact_wm" if "contact_wm" in name else name.split("_")[1]
        if arch not in colours:
            continue
        ax.plot(
            [x["updates"] for x in curve[1:]],
            [x["l_obs"] for x in curve[1:]],
            color=colours[arch],
            lw=1,
            alpha=0.6,
            label=arch if name.endswith("_11") else None,
        )
    for r in runs:
        j = r["job"]
        if j["kind"] != "learn" or j["width"] != "medium":
            continue
        u = [x["updates"] for x in r["curve"][1:]]
        lab = j["seed"] == 11
        if j["candidate"] in ("pnn", "mlp-ms"):
            ax.plot(
                u,
                [x["l_obs"] for x in r["curve"][1:]],
                color={"pnn": "tab:purple", "mlp-ms": "tab:orange"}[j["candidate"]],
                lw=2,
                label=j["candidate"] if lab else None,
            )
        else:
            ax.plot(
                u,
                [x["l_obs"] for x in r["curve"][1:]],
                color="black",
                lw=2,
                label="rssm-dv3 (16 samples)" if lab else None,
            )
            ax.plot(
                u,
                [x["l_obs_mean_rule"] for x in r["curve"][1:]],
                "--",
                color="black",
                lw=1.5,
                label="rssm-dv3 (means)" if lab else None,
            )
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("updates")
    ax.set_ylabel("L_obs (smoke validation)")
    ax.set_title("Medium width, placeholder settings: candidates against milestone 2 runs")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--candidates",
        nargs="+",
        choices=list(CANDIDATES),
        default=list(CANDIDATES),
        help="run only these; earlier runs of the others are kept from exploration.json",
    )
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "plots").mkdir(exist_ok=True)
    t0 = time.time()
    js = [j for j in jobs() if j["candidate"] in args.candidates]
    js = sorted(js, key=lambda j: (j["candidate"] != "rssm-dv3", j["kind"] != "learn"))
    runs = []
    path = OUT / "exploration.json"
    if path.exists():  # keep earlier runs of candidates not rerun now
        old = json.loads(path.read_text())["runs"]
        runs = [{"name": n, **r} for n, r in old.items() if r["job"]["candidate"] not in args.candidates]
    with ProcessPoolExecutor(args.workers) as ex:
        for r in ex.map(run_job, js):
            print(r["name"], f"{r['wall_seconds']:.0f}s", flush=True)
            runs.append(r)
    m2 = json.loads(M2.read_text())
    record = {
        "settings": {"placeholders": P.PLACEHOLDERS, "checks": P.CHECK_SETTINGS},
        "constants": {k: getattr(c, k) for k in dir(c) if k.startswith(("PNN_", "RSSM_DV3_", "MULTI_STEP_"))},
        "runs": {r["name"]: {k: v for k, v in r.items() if k != "name"} for r in runs},
        "matching": matching(m2, runs),
        "wall_seconds": time.time() - t0,
    }
    (OUT / "exploration.json").write_text(json.dumps(record, indent=1, default=P._json_default))
    plot(m2, runs, OUT / "plots" / "candidates.png")
    print("done", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
