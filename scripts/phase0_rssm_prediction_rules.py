"""Training-side diagnostic: is rssm's gap prediction noise or slow learning?

Run after scripts/phase0_training.py: uv run python scripts/phase0_rssm_prediction_rules.py

Rescores saved rssm checkpoints on smoke validation L_obs under several prediction rules, and
retrains the history-only made-up target run to score it the same ways. No interventions, no
states.npz.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import phase0_training as P  # noqa: E402

from physical_intervene.data import box_target_from_delta  # noqa: E402
from physical_intervene.models import build  # noqa: E402
from physical_intervene.models.common import Batch  # noqa: E402
from physical_intervene.train import loss_weights, train, weighted_loss  # noqa: E402

torch.set_num_threads(4)
OUT = ROOT / "results" / "phase0" / "milestone2" / "rssm_prediction_rules.json"


def predict(model, b: Batch, mode: str, s_next_n=None, n=16) -> torch.Tensor:
    gen = torch.Generator().manual_seed(model.sampling_seed)
    with torch.no_grad():
        if mode.startswith("samples"):
            return model.predict_z(b, gen) if n == 16 else _samples(model, b, gen, n)
        states, prev_a, mask = model.window_inputs(b)
        if mode == "prior_mean":  # filter as now (sampled posterior), prior mean at the step
            _, h, z = model.filter(states, prev_a, mask, gen)
        elif mode in ("deterministic", "posterior_next"):  # posterior means while filtering
            h, z = _filter_mean(model, states, prev_a, mask)
        h1 = model.cell(torch.cat([z, b.a_n], -1), h)
        if mode == "posterior_next":  # uses s_{t+1}: not a prediction, an upper bound
            return model.decode(h1, model.posterior(h1, s_next_n).mean)
        return model.decode(h1, model.prior(h1).mean)


def _samples(model, b, gen, n):
    outs = []
    for i in range(0, len(b), 500):
        part = Batch(*(x[i : i + 500] for x in b.__dict__.values()))
        o = model.sample_prediction(part.repeat(n), gen)
        outs.append(o.view(len(part), n, -1).mean(1))
    return torch.cat(outs)


def _filter_mean(model, states, prev_a, mask):
    B, K, _ = states.shape
    h = states.new_zeros(B, model.units)
    z = states.new_zeros(B, model.latent)
    for k in range(K):
        h_new = model.cell(torch.cat([z, prev_a[:, k]], -1), h)
        z_new = model.posterior(h_new, states[:, k]).mean
        m = mask[:, k : k + 1]
        h, z = m * h_new + (1 - m) * h, m * z_new + (1 - m) * z
    return h, z


def l_obs(model, td, mode, n=16):
    b = td.val_batch()
    s_next_n = None
    if mode == "posterior_next":
        s_next = td.val.s + td.val.delta
        s_next_n = torch.as_tensor((s_next - td.norm.s_mean) / td.norm.s_std, dtype=torch.float32)
    z = predict(model, b, mode, s_next_n, n).double().numpy()
    delta = z * td.dscaler.scale + td.dscaler.mean
    return td.yscaler.std_mse(box_target_from_delta(td.val.s, delta), td.val.y)


def load(path, td):
    ck = torch.load(path, weights_only=False)
    m = ck["meta"]
    model = build("rssm", m["width"], td.norm, td.dscaler)
    model.load_state_dict(ck["state_dict"])
    model.sampling_seed = m["rssm_sampling_seed"]
    model.eval()
    return model, m


MODES = ["samples16", "prior_mean", "deterministic", "posterior_next"]
res = {"checkpoints": {}, "history_target": {}}
td = P.train_data()
for run in [
    "learn_rssm_medium_11",
    "learn_rssm_medium_22",
    "learn_rssm_medium_33",
    "learn_rssm_small_11",
    "learn_rssm_large_11",
]:
    d = P.CKPT / run
    files = sorted(d.glob("u*.pt"))
    if "medium" not in run:  # small and large: a few checkpoints only
        files = [f for f in files if int(f.stem[1:]) in (1000, 2000, 6000, 12000)]
    rows = []
    for f in files:
        model, m = load(f, td)
        row = {"updates": m["updates"]}
        for mode in MODES:
            row[mode] = l_obs(model, td, mode)
        if m["updates"] == 12000:
            row["samples256"] = l_obs(model, td, "samples", n=256)
        rows.append(row)
        print(run, row, flush=True)
    res["checkpoints"][run] = rows

# History-only made-up target: retrain exactly as the check did, then score with each rule.
t0 = time.time()
tdh = P.train_data("history")
cfg = P.config(
    "rssm", "medium", 11, P.CHECK_SETTINGS["made_up_updates"], loss_dims=None, target_name="history"
)
r = train(cfg, tdh, evaluate_val=False)
model = r["model"].eval()
idx = np.random.default_rng(0).choice(
    len(tdh.train), P.CHECK_SETTINGS["made_up_eval_examples"], replace=False
)
b = tdh.batch(idx)
zt = tdh.z(idx)
for mode in ["samples16", "prior_mean", "deterministic"]:
    zp = predict(model, b, mode)
    res["history_target"][mode] = float(weighted_loss(zp, zt, loss_weights(None)))
print("history", res["history_target"], f"{time.time() - t0:.0f}s", flush=True)
OUT.write_text(json.dumps(res, indent=1))
