"""Scoring on interventions (DESIGN.md, Target and loss, Interventions, Gates).

The intervention side of the pipeline: the test-like states and their 24 labelled interventions,
std-MSE per transition on the box target (the training split's scaler with the design's floors),
blocks by stratum (no contact, graze, push, all contact, all), seed averaging, the corner error in
mm, the angle error in degrees, and the logged-action reference (action 0 at each state, which is
never an intervention and never enters E_int).

**No-leak guard** (SMOKE_TEST.md, *No-leak rule*; DESIGN.md, *Seal*). The phase comes from the freeze
records (seal.current_phase). In Phase 0 the scorer refuses any model whose metadata does not say
`trained: false`; metadata that lacks the key is refused too. Between the freezes it also accepts
the pilot's positive-control confirmation runs (`pilot_confirmation: true`) and nothing else
trained. After freeze 2 it accepts trained models, and loading a checkpoint first checks every
recorded hash (seal.check_frozen). The guard reads a checkpoint's metadata before its weights are
loaded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from physical_intervene import constants as c
from physical_intervene import interventions as I
from physical_intervene.data import Examples, TargetScaler, box_target_from_delta, file_sha256
from physical_intervene.metrics import corner_rmse_mm
from physical_intervene.sim import wrap_angle

STRATA = I.LABELS  # no_contact, graze, push
HARD = "push"
BLOCKS = ("all", "no_contact", "graze", "push", "all_contact")
_BLOCK_LABELS = {"all": (0, 1, 2), "no_contact": (0,), "graze": (1,), "push": (2,), "all_contact": (1, 2)}


class LeakError(RuntimeError):
    """Raised when Phase 0 would score a trained model on interventions."""


def _phase(phase: str | None) -> str:
    if phase is None:
        from physical_intervene.seal import current_phase

        return current_phase()
    return phase


def assert_scorable(meta: dict, phase: str | None = None) -> None:
    """Refuse a model on interventions unless the phase allows it (see the module's docstring)."""
    phase = _phase(phase)
    if phase not in ("phase0", "training", "scoring"):
        raise ValueError(f"unknown phase {phase!r}")
    if phase == "scoring" or meta.get("trained") is False:
        return
    if phase == "training" and meta.get("pilot_confirmation") is True:
        return
    raise LeakError(
        f"phase is {phase!r}: the scorer refuses this trained model on interventions "
        f"(metadata trained={meta.get('trained', 'missing')!r}, "
        f"pilot_confirmation={meta.get('pilot_confirmation', 'missing')!r}, "
        f"architecture={meta.get('architecture', '?')}, seed={meta.get('seed', '?')})"
    )


_CHECKED: set = set()


def require_frozen_for_scoring(phase: str) -> None:
    """After freeze 2, every recorded hash must match before any trained model is scored (once per
    process)."""
    if phase == "scoring" and "scoring" not in _CHECKED:
        from physical_intervene.seal import check_frozen

        check_frozen(2)
        _CHECKED.add("scoring")


def load_checkpoint_for_scoring(path: Path, phase: str | None = None):
    """The scoring entry point for saved models: (model, metadata, state-change scaler). The sidecar
    metadata (.json) is checked first, then the metadata stored in the checkpoint itself, before any
    weight is loaded into a model."""
    phase = _phase(phase)
    require_frozen_for_scoring(phase)
    path = Path(path)
    side = path.with_suffix(".json")
    if not side.exists():
        raise LeakError(f"{path.name}: no metadata file, so the scorer cannot tell whether it is trained")
    assert_scorable(json.loads(side.read_text()), phase)
    import torch
    from torch.torch_version import TorchVersion

    from physical_intervene.data import InputNorm
    from physical_intervene.models import build

    # Training stores torch.__version__ (a TorchVersion, a str subclass) in the metadata; allow that
    # one class and nothing else, so the safe loader still refuses any other object.
    with torch.serialization.safe_globals([TorchVersion]):
        ck = torch.load(path, map_location="cpu", weights_only=True)
    assert_scorable(ck["meta"], phase)
    meta = ck["meta"]
    model = build(
        meta["architecture"],
        meta["width"],
        InputNorm.from_dict(ck["input_norm"]),
        TargetScaler.from_dict(ck["delta_scaler"]),
    )
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, meta, TargetScaler.from_dict(ck["delta_scaler"])


# Intervention set --------------------------------------------------------------------------------


@dataclass
class InterventionSet:
    """Test-like states with all 25 actions (index 0 the logged action, then 12 aimed, 12 uniform)."""

    s: np.ndarray  # (n, 11)
    actions: np.ndarray  # (n, 25, 2)
    y_true: np.ndarray  # (n, 25, 3), the simulator's box target
    label: np.ndarray  # (n, 25): 0 no_contact, 1 graze, 2 push
    kinds: np.ndarray  # (25,)
    hist_s: np.ndarray | None = None  # (n, L, 11)
    hist_a: np.ndarray | None = None  # (n, L, 2)
    hist_mask: np.ndarray | None = None  # (n, L)
    file: str = ""
    sha256: str = ""

    @property
    def n_states(self) -> int:
        return len(self.s)

    def mask(self, block: str) -> np.ndarray:
        """(n, 25) mask of the interventions in a block; the logged action (column 0) is never in one."""
        m = np.isin(self.label, _BLOCK_LABELS[block])
        m[:, 0] = False
        return m

    def counts(self, block: str) -> np.ndarray:
        return self.mask(block).sum(1)

    def logged_mask(self) -> np.ndarray:
        m = np.zeros_like(self.label, dtype=bool)
        m[:, 0] = True
        return m

    def examples(self) -> Examples:
        """Model inputs for all 25 actions at every state, flattened state-major (n x 25)."""
        if self.hist_s is None:
            raise ValueError("this intervention set has no histories; learned models need them")
        k = self.actions.shape[1]
        rep = lambda x: np.repeat(x, k, axis=0)  # noqa: E731
        s = rep(self.s)
        a = self.actions.reshape(-1, 2)
        y = self.y_true.reshape(-1, 3)
        return Examples(
            episode=np.repeat(np.arange(self.n_states), k),
            step=np.tile(np.arange(k), self.n_states),
            s=s,
            hist_s=rep(self.hist_s),
            hist_a=rep(self.hist_a),
            hist_mask=rep(self.hist_mask),
            a=a,
            s_next=np.full_like(s, np.nan),  # unknown to the scorer; the box target is y
            delta=np.full_like(s, np.nan),
            y=y,
            shadow=None,
        )


def load_interventions(states_path: Path, history_path: Path | None = None) -> InterventionSet:
    states_path = Path(states_path)
    with np.load(states_path) as f:
        d = {k: f[k] for k in f.files}
    out = InterventionSet(
        s=d["obs"],
        actions=d["actions"],
        y_true=d["target"],
        label=I.state_label(d),
        kinds=d["kinds"],
        file=states_path.name,
        sha256=file_sha256(states_path),
    )
    if history_path is not None:
        with np.load(history_path) as f:
            h = {k: f[k] for k in f.files}
        if not np.array_equal(h["seed"], d["seed"]):
            raise ValueError("history file does not belong to these states")
        out.hist_s, out.hist_a, out.hist_mask = h["hist_s"], h["hist_a"], h["hist_mask"]
    return out


# Predictions and losses --------------------------------------------------------------------------


@dataclass
class Prediction:
    """One run's box-target predictions at one checkpoint."""

    val: np.ndarray  # (N_val, 3), validation transitions with recorded actions
    interventions: np.ndarray  # (n, 25, 3), logged action first


def predict_model(
    model, dscaler: TargetScaler, val: Examples, iset: InterventionSet, meta: dict, phase: str | None = None
) -> Prediction:
    """A learned model's predictions (dscaler: its state-change scaler); refuses a trained model in
    Phase 0 (the no-leak guard)."""
    phase = _phase(phase)
    assert_scorable(meta, phase)
    require_frozen_for_scoring(phase)
    from physical_intervene.models import make_batch
    from physical_intervene.train import predict_delta

    out = []
    for ex in (val, iset.examples()):
        delta = predict_delta(model, make_batch(ex, model.norm), dscaler)
        out.append(box_target_from_delta(ex.s, delta))
    return Prediction(val=out[0], interventions=out[1].reshape(iset.y_true.shape))


def predict_baseline(fn, val: Examples, iset: InterventionSet) -> Prediction:
    """A baseline that maps Examples to a predicted state change (baselines.copy, constvel)."""
    ex_int = Examples(
        episode=np.zeros(iset.n_states * 25, int),
        step=np.zeros(iset.n_states * 25, int),
        s=np.repeat(iset.s, 25, axis=0),
        hist_s=None,
        hist_a=None,
        hist_mask=None,
        a=iset.actions.reshape(-1, 2),
        s_next=None,
        delta=None,
        y=iset.y_true.reshape(-1, 3),
        shadow=None,
    )
    out = [box_target_from_delta(ex.s, fn(ex)) for ex in (val, ex_int)]
    return Prediction(val=out[0], interventions=out[1].reshape(iset.y_true.shape))


@dataclass
class Losses:
    """Per-transition std-MSE of one prediction."""

    val: np.ndarray  # (N_val,)
    interventions: np.ndarray  # (n, 25)

    @property
    def l_obs(self) -> float:
        return float(self.val.mean())


def losses(scaler: TargetScaler, pred: Prediction, val_y: np.ndarray, iset: InterventionSet) -> Losses:
    return Losses(
        val=scaler.per_sample(pred.val, val_y),
        interventions=scaler.per_sample(pred.interventions, iset.y_true),
    )


def state_sums(loss_int: np.ndarray, iset: InterventionSet, block: str) -> np.ndarray:
    """Sum over each state's interventions in the block: (..., n) from (..., n, 25)."""
    return np.where(iset.mask(block), loss_int, 0.0).sum(-1)


# Blocks (descriptive, not gated) -----------------------------------------------------------------


def angle_rmse_deg(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    d = wrap_angle(y_pred[..., 2] - y_true[..., 2])
    return float(np.rad2deg(np.sqrt(np.mean(d * d))))


def block_summary(
    scaler: TargetScaler,
    iset: InterventionSet,
    y_pred: np.ndarray,
    block: str,
    rng: np.random.Generator,
    reps: int = c.DESCRIPTIVE_BOOTSTRAP_REPLICATES,
) -> dict:
    """A stratum block of one prediction (for a model, the seed-averaged prediction): n, std-MSE with
    a 95% interval resampling states, corner RMSE (mm) and angle RMSE (degrees)."""
    m = iset.mask(block) if block != "logged" else iset.logged_mask()
    n = int(m.sum())
    if n == 0:
        return {"n": 0, "std_mse": None, "std_mse_ci95": None, "corner_rmse_mm": None, "angle_rmse_deg": None}
    per = scaler.per_sample(y_pred, iset.y_true)
    sums, cnt = np.where(m, per, 0.0).sum(1), m.sum(1)
    keep = cnt > 0
    sums, cnt = sums[keep], cnt[keep]
    W = rng.multinomial(len(cnt), np.full(len(cnt), 1.0 / len(cnt)), size=reps).astype(float)
    boot = (W @ sums) / (W @ cnt)
    s_rep = np.broadcast_to(iset.s[:, None, :], (*m.shape, iset.s.shape[1]))[m]
    out = {
        "n": n,
        "std_mse": float(per[m].mean()),
        "std_mse_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "corner_rmse_mm": corner_rmse_mm(s_rep, y_pred[m], iset.y_true[m]),
        "angle_rmse_deg": angle_rmse_deg(y_pred[m], iset.y_true[m]),
    }
    return out


def blocks(scaler: TargetScaler, iset: InterventionSet, y_pred: np.ndarray, rng: np.random.Generator) -> dict:
    return {b: block_summary(scaler, iset, y_pred, b, rng) for b in BLOCKS}


def logged_action_block(scaler: TargetScaler, iset: InterventionSet, y_pred: np.ndarray) -> dict:
    """The logged action at each state: the observational reference on the same states."""
    m = iset.logged_mask()
    per = scaler.per_sample(y_pred, iset.y_true)
    return {
        "n": int(m.sum()),
        "std_mse": float(per[m].mean()),
        "corner_rmse_mm": corner_rmse_mm(iset.s, y_pred[:, 0], iset.y_true[:, 0]),
        "angle_rmse_deg": angle_rmse_deg(y_pred[:, 0], iset.y_true[:, 0]),
    }


def obs_val_ci95(
    per_transition: np.ndarray,
    episode: np.ndarray,
    rng: np.random.Generator,
    reps: int = c.DESCRIPTIVE_BOOTSTRAP_REPLICATES,
) -> list[float]:
    """95% interval of L_obs (per-transition loss already averaged over seeds), resampling whole
    validation episodes."""
    ids, inv = np.unique(episode, return_inverse=True)
    sums = np.bincount(inv, weights=per_transition)
    cnt = np.bincount(inv).astype(float)
    W = rng.multinomial(len(ids), np.full(len(ids), 1.0 / len(ids)), size=reps).astype(float)
    boot = (W @ sums) / (W @ cnt)
    return [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


def copy_std_mse_direct(scaler: TargetScaler, y_true: np.ndarray) -> float:
    """`copy`'s std-MSE computed straight from the box motion: the mean over transitions of the mean
    over (dx, dy, dtheta) of (y / scale)^2. A second path, for the stand-in check."""
    z = np.asarray(y_true, float) / scaler.scale
    return float(np.mean(np.sum(z * z, axis=-1) / z.shape[-1]))
