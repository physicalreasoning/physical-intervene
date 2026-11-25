"""Stand-in models for the intervention side of Phase 0 (SMOKE_TEST.md, Intervention-side checks).

No stand-in is trained, and every one carries `trained: false`. A noisy stand-in's prediction for
seed i at checkpoint k is the simulator's true box target plus an error, in units of the training
split's target scale:

    error = sqrt(E) x (sqrt(f) x b_i + sqrt(1 - f) x r x eps)

- E is the expected per-transition std-MSE: on validation transitions and non-push interventions
  it is the run's target L_obs at that checkpoint; on push interventions it is STANDIN_PUSH_OVER_OBS
  times that, scaled per transition by a fixed difficulty (the same for every stand-in) whose mean
  over push transitions is 1.
- b_i is a 3-dimensional standard Gaussian bias drawn once per seed (one for push transitions, one
  for the others), shared by all of that seed's transitions in the group and by all its
  checkpoints; eps is fresh standard Gaussian noise per transition and checkpoint.
- f is the bias share; a stand-in's seed-to-seed coefficient of variation of E_int on push is
  f x sqrt(2/3) (bias term only).
- r is the push noise ratio (1 for noisy-a; 1.5 for noisy-b; noisy-c's gives a 25% gap). It applies
  to push transitions only and may depend on the target L_obs (synthetic series along training).

Target L_obs curves follow milestone 2's medium-width L_obs curves of the same architecture
(training side, recorded actions only), on the design's checkpoint schedule, times a per-seed
log-normal factor with coefficient of variation STANDIN_OBS_SEED_CV. The realised L_obs is computed
from the stand-in's predictions, as for any model.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from physical_intervene import constants as c
from physical_intervene.data import TargetScaler
from physical_intervene.gates import Family
from physical_intervene.scoring import BLOCKS, InterventionSet, Prediction, losses, state_sums

STANDIN_META = {"trained": False, "standin": True, "phase": c.PHASE}
KEYS = {"curve": 0, "bias": 1, "noise": 2}


def bias_share_for_cv(cv: float) -> float:
    """Bias share f giving a seed-to-seed coefficient of variation cv of E_int (cv = f sqrt(2/3))."""
    f = cv / np.sqrt(2.0 / 3.0)
    if not 0.0 <= f < 1.0:
        raise ValueError(f"cv {cv} needs a bias share of {f:.3f}, outside [0, 1)")
    return float(f)


def ratio_for_rel_gap(rel_gap: float, f: float) -> float:
    """Push noise ratio r giving an expected relative gap rel_gap over r = 1 at bias share f:
    (f + r^2 (1 - f)) / 1 = 1 + rel_gap."""
    return float(np.sqrt(1.0 + rel_gap / (1.0 - f)))


def rel_gap_for_ratio(r: float, f: float) -> float:
    return float((r * r - 1.0) * (1.0 - f))


def gen(*key: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(k) for k in key]))


# Curves ------------------------------------------------------------------------------------------


def milestone2_curves(path: Path) -> dict:
    """{arch: {width: (updates, (n_seeds, K) L_obs)}} from milestone 2's training_checks.json, on the
    checkpoint schedule (update 0, before training, left out)."""
    d = json.loads(Path(path).read_text())["curves"]
    out: dict = {}
    for name, rows in d.items():
        parts = name.split("_")
        if parts[0] != "learn":
            continue
        arch, width, _seed = "_".join(parts[1:-2]), parts[-2], int(parts[-1])
        rows = [r for r in rows if r["updates"] > 0]
        upd = [r["updates"] for r in rows]
        out.setdefault(arch, {}).setdefault(width, []).append((upd, [r["l_obs"] for r in rows]))
    res: dict = {}
    for arch, widths in out.items():
        for width, runs in widths.items():
            upd = runs[0][0]
            assert all(r[0] == upd for r in runs)
            res.setdefault(arch, {})[width] = (upd, np.array([r[1] for r in runs]))
    return res


def base_curve(curves: np.ndarray) -> np.ndarray:
    """Geometric mean over the available seeds' curves."""
    return np.exp(np.log(curves).mean(0))


def seed_curves(base: np.ndarray, n_seeds: int, key: int, cv: float = c.STANDIN_OBS_SEED_CV) -> np.ndarray:
    z = gen(key, KEYS["curve"]).standard_normal(n_seeds)
    return base[None, :] * np.exp(cv * z)[:, None]


# Context and specs -------------------------------------------------------------------------------


@dataclass
class Context:
    """What every stand-in needs: the scaler, the validation targets and the interventions."""

    scaler: TargetScaler
    val_y: np.ndarray  # (N_val, 3)
    val_episode: np.ndarray  # (N_val,)
    iset: InterventionSet
    difficulty: np.ndarray = field(init=False)  # (n, 25), push transitions only differ from 1
    push: np.ndarray = field(init=False)  # (n, 25) bool, push interventions

    def __post_init__(self) -> None:
        self.push = self.iset.mask("push")
        z = self.iset.y_true / self.scaler.scale
        q = np.mean(z * z, axis=-1)
        w = c.STANDIN_DIFFICULTY_SHARE
        d = np.ones_like(q)
        d[self.push] = (1.0 - w) + w * q[self.push] / q[self.push].mean()
        self.difficulty = d


@dataclass
class StandIn:
    """A noisy stand-in family: L_obs target curves per seed, push settings, and its own random key."""

    name: str
    key: int
    seeds: tuple[int, ...]
    updates: list[int]
    l_target: np.ndarray  # (S, K)
    bias_share: float
    push_ratio: float | Callable[[float], float] = 1.0
    push_over_obs: float = c.STANDIN_PUSH_OVER_OBS
    width: str = "medium"

    def ratio(self, level: float) -> float:
        return float(self.push_ratio(level) if callable(self.push_ratio) else self.push_ratio)

    def meta(self) -> dict:
        return {**STANDIN_META, "family": self.name, "key": self.key, "bias_share": self.bias_share}

    def expected_e_push(self, i: int, k: int) -> float:
        """Expected E on push of seed i at checkpoint k, over draws of bias and noise."""
        f, r = self.bias_share, self.ratio(self.l_target[i, k])
        return float(self.push_over_obs * self.l_target[i, k] * (f + r * r * (1.0 - f)))

    def predict(self, ctx: Context, i: int, k: int) -> Prediction:
        seed = self.seeds[i]
        level = float(self.l_target[i, k])
        f, r = self.bias_share, self.ratio(level)
        b = gen(self.key, KEYS["bias"], seed).standard_normal((2, 3))  # push, other
        g = gen(self.key, KEYS["noise"], seed, k)
        val_err = np.sqrt(level) * g.standard_normal(ctx.val_y.shape)
        eps = g.standard_normal(ctx.iset.y_true.shape)
        sf, sn = np.sqrt(f), np.sqrt(1.0 - f)
        e_push = (np.sqrt(self.push_over_obs * level * ctx.difficulty))[..., None] * (
            sf * b[0] + sn * r * eps
        )
        e_other = np.sqrt(level) * (sf * b[1] + sn * eps)
        err = np.where(ctx.push[..., None], e_push, e_other)
        return Prediction(
            val=ctx.val_y + val_err * ctx.scaler.scale,
            interventions=ctx.iset.y_true + err * ctx.scaler.scale,
        )


def oracle_prediction(ctx: Context) -> Prediction:
    return Prediction(val=ctx.val_y.copy(), interventions=ctx.iset.y_true.copy())


# Families ----------------------------------------------------------------------------------------


def family_from_predictor(
    name: str,
    seeds: tuple[int, ...] | list[int],
    updates: list[int],
    predict: Callable[[int, int], Prediction],
    ctx: Context,
    width: str = "medium",
    meta: dict | None = None,
) -> Family:
    """Score every seed at every checkpoint: L_obs and per-state loss sums for every block."""
    S, K, n = len(seeds), len(updates), ctx.iset.n_states
    l_obs = np.empty((S, K))
    sums = {b: np.empty((S, K, n)) for b in (*BLOCKS, "logged")}
    for i in range(S):
        for k in range(K):
            ls = losses(ctx.scaler, predict(i, k), ctx.val_y, ctx.iset)
            l_obs[i, k] = ls.l_obs
            for b in BLOCKS:
                sums[b][i, k] = state_sums(ls.interventions, ctx.iset, b)
            sums["logged"][i, k] = ls.interventions[:, 0]
    counts = {b: ctx.iset.counts(b) for b in BLOCKS}
    counts["logged"] = np.ones(n, int)
    return Family(name, list(seeds), list(updates), l_obs, sums, counts, width, meta or {})


def family(spec: StandIn, ctx: Context) -> Family:
    fam = family_from_predictor(
        spec.name,
        spec.seeds,
        spec.updates,
        lambda i, k: spec.predict(ctx, i, k),
        ctx,
        spec.width,
        spec.meta(),
    )
    fam.meta["spec"] = spec
    return fam


def synthetic_family(name: str, updates: list[int], l_obs: np.ndarray, n_states: int = 4) -> Family:
    """A family with given L_obs curves and zero intervention losses (for matching checks)."""
    S, K = l_obs.shape
    sums = {b: np.zeros((S, K, n_states)) for b in (*BLOCKS, "logged")}
    counts = {b: np.ones(n_states, int) for b in (*BLOCKS, "logged")}
    return Family(name, list(range(S)), list(updates), np.asarray(l_obs, float), sums, counts)


def untrained_meta(arch: str, width: str, seed: int) -> dict:
    return {
        "architecture": arch,
        "width": width,
        "seed": seed,
        "updates": 0,
        "trained": False,
        "phase": c.PHASE,
        "note": "initial weights, never trained (Phase 0 stand-in)",
    }


def save_untrained(path: Path, arch: str, width: str, seed: int, norm, dscaler, yscaler) -> dict:
    """An untrained network saved in the checkpoint format, with `trained: false`."""
    import torch

    from physical_intervene.models import build

    torch.manual_seed(seed)
    model = build(arch, width, norm, dscaler)
    meta = {**untrained_meta(arch, width, seed), "params": model.n_params(), "torch": str(torch.__version__)}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "meta": meta,
            "state_dict": model.state_dict(),
            "input_norm": norm.to_dict(),
            "delta_scaler": dscaler.to_dict(),
            "box_scaler": yscaler.to_dict(),
        },
        path,
    )
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    return meta
