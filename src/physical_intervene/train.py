"""Training loop (DESIGN.md, Models, Training: full budget, no early stopping).

Every run trains its full number of updates at a constant learning rate and is evaluated (L_obs on
the validation transitions) at every checkpoint of the schedule: every 100 updates up to 2,000,
then every 1,000. It runs on CPU, single-threaded, with deterministic algorithms, so one seed gives
one loss curve. All randomness of a run comes from its seed through separate streams:
initialisation, batch order, the mlp-shortcut swap, and rssm's sampling in training and in
prediction. So changing p, for example, does not change the batch order.

Every saved checkpoint records what milestone 3's scorer needs to refuse it in Phase 0: the
architecture, width and seed, the update count, that it is trained, the phase, and the data it was
trained on (file name and sha256).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from physical_intervene import constants as c
from physical_intervene.data import (
    BOX_DIMS,
    PUSHER_DIMS,
    Examples,
    InputNorm,
    Split,
    TargetScaler,
    box_scaler,
    box_target_from_delta,
    delta_scaler,
    make_examples,
    training_loss_weights,
)
from physical_intervene.metrics import corner_rmse_mm, plateau
from physical_intervene.models import Batch, WorldModel, build, make_batch
from physical_intervene.models.common import norm_action
from physical_intervene.seal import current_phase

STREAMS = {"init": 0, "batch": 1, "shortcut": 2, "rssm_train": 3, "rssm_predict": 4}


def stream_seed(seed: int, purpose: str) -> int:
    ss = np.random.SeedSequence(seed, spawn_key=(STREAMS[purpose],))
    return int(ss.generate_state(1, dtype=np.uint64)[0] % (2**63 - 1))


def checkpoint_schedule(updates: int) -> list[int]:
    fine = range(
        c.CHECKPOINT_FINE_EVERY_UPDATES, c.CHECKPOINT_FINE_UNTIL_UPDATES + 1, c.CHECKPOINT_FINE_EVERY_UPDATES
    )
    dense = range(
        c.CHECKPOINT_FINE_UNTIL_UPDATES + c.CHECKPOINT_DENSE_EVERY_UPDATES,
        c.CHECKPOINT_DENSE_UNTIL_UPDATES + 1,
        c.CHECKPOINT_DENSE_EVERY_UPDATES,
    )
    sparse = range(
        c.CHECKPOINT_DENSE_UNTIL_UPDATES + c.CHECKPOINT_EVERY_UPDATES, updates + 1, c.CHECKPOINT_EVERY_UPDATES
    )
    sched = [u for u in (*fine, *dense) if u <= updates] + list(sparse)
    if not sched or sched[-1] != updates:
        sched.append(updates)  # the final update is always a checkpoint
    return sched


@dataclass
class TrainConfig:
    arch: str
    width: str
    seed: int
    updates: int
    lr: float
    batch_size: int  # transitions per update (one-step models)
    seq_batch: int  # sequences per update (rssm)
    seq_len: int = c.RSSM_SEQUENCE_LEN
    shortcut_p: float | None = None
    optimiser: str = "adam"
    checkpoint_every: list[int] | None = None  # default: the design's schedule
    loss_dims: list[int] | None = None  # made-up checks only: restrict the loss to these dims
    target_name: str = "real"
    tag: str = ""
    rssm_variant: str = "design"  # exploration only: "dv3" uses DreamerV3's KL terms
    stop_at_plateau: bool = False  # pilot rule: stop at the first checkpoint meeting the plateau criterion
    family: str = ""  # the design's name when it differs from arch (e.g. "mlp-twin" trains arch "mlp")
    pilot_confirmation: bool = False  # the pilot's positive-control confirmation runs (DESIGN.md, Controls)


@dataclass
class TrainData:
    """Everything a run needs from the data, with scalers fitted on the training split only."""

    split: Split
    train: Examples
    val: Examples | None
    norm: InputNorm
    dscaler: TargetScaler
    yscaler: TargetScaler
    delta: np.ndarray  # (N, 11) training targets in physical units (real or made up)
    data_meta: dict = field(default_factory=dict)
    subset: np.ndarray | None = None  # train only on these example indices (overfit check)
    _cache: dict = field(default_factory=dict, repr=False)

    @classmethod
    def build(
        cls,
        train_split: Split,
        val_split: Split | None,
        delta: np.ndarray | None = None,
        dscaler: TargetScaler | None = None,
    ) -> TrainData:
        """Made-up targets pass the real data's delta scaler, so their losses are in the same unit
        as the real training loss."""
        train = make_examples(train_split)
        val = None if val_split is None else make_examples(val_split)
        delta = train.delta if delta is None else delta
        dscaler = delta_scaler(train.delta) if dscaler is None else dscaler
        meta = {
            "train": {
                "file": train_split.file,
                "sha256": train_split.sha256,
                "seeds": _seed_range(train_split),
            }
        }
        if val_split is not None:
            meta["validation"] = {
                "file": val_split.file,
                "sha256": val_split.sha256,
                "seeds": _seed_range(val_split),
            }
        return cls(train_split, train, val, InputNorm.fit(train), dscaler, box_scaler(train), delta, meta)

    def batch(self, idx: np.ndarray | None = None) -> Batch:
        if "train_batch" not in self._cache:
            self._cache["train_batch"] = make_batch(self.train, self.norm)
            self._cache["train_z"] = torch.as_tensor(
                (self.delta - self.dscaler.mean) / self.dscaler.scale, dtype=torch.float32
            )
        full = self._cache["train_batch"]
        if idx is None:
            return full
        t = torch.as_tensor(idx)
        return Batch(*(x[t] for x in full.__dict__.values()))

    def batch_delta(self, idx: np.ndarray) -> torch.Tensor:
        """True state change of training examples, physical units (float32)."""
        if "train_delta" not in self._cache:
            self._cache["train_delta"] = torch.as_tensor(self.delta, dtype=torch.float32)
        return self._cache["train_delta"][torch.as_tensor(idx)]

    def z(self, idx: np.ndarray) -> torch.Tensor:
        self.batch()
        return self._cache["train_z"][torch.as_tensor(idx)]

    def val_batch(self) -> Batch:
        if "val_batch" not in self._cache:
            self._cache["val_batch"] = make_batch(self.val, self.norm)
        return self._cache["val_batch"]

    def episode_arrays(self) -> dict[str, torch.Tensor]:
        """Standardised episode arrays for rssm windows."""
        if "episodes" not in self._cache:
            sp, n = self.split, self.norm
            T = sp.steps
            z = (self.delta - self.dscaler.mean) / self.dscaler.scale
            self._cache["episodes"] = {
                "s": torch.as_tensor((sp.obs - n.s_mean) / n.s_std, dtype=torch.float32),
                "a": torch.as_tensor(norm_action(sp.action, n), dtype=torch.float32),
                "z": torch.as_tensor(z.reshape(sp.n_episodes, T, -1), dtype=torch.float32),
            }
        return self._cache["episodes"]


def _seed_range(split: Split) -> list[int]:
    return [int(split.seed.min()), int(split.seed.max())]


def loss_weights(loss_dims: list[int] | None) -> torch.Tensor:
    """The shared training loss's per-dimension weights; restricted (and renormalised to the same
    box/pusher split) when a made-up check uses only some dimensions."""
    w = training_loss_weights()
    if loss_dims is not None:
        keep = np.zeros(len(w), bool)
        keep[loss_dims] = True
        w = np.where(keep, 1.0, 0.0)
        for part in (BOX_DIMS, PUSHER_DIMS):
            k = keep[part].sum()
            if k:
                w[part] = np.where(keep[part], 1.0 / k, 0.0)
        w /= w.sum()
    return torch.as_tensor(w, dtype=torch.float32)


def weighted_loss(pred: torch.Tensor, z: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    d = pred - z
    return ((d * d) * w).sum(-1).mean()


# Evaluation ---------------------------------------------------------------------------------------


@torch.no_grad()
def predict_delta(model: WorldModel, b: Batch, dscaler: TargetScaler, chunk: int = 4096) -> np.ndarray:
    """Predicted state change in physical units (float64)."""
    model.eval()
    gen = torch.Generator().manual_seed(getattr(model, "sampling_seed", 0))
    outs = []
    for i in range(0, len(b), chunk):
        part = Batch(*(x[i : i + chunk] for x in b.__dict__.values()))
        outs.append(model.predict_z(part, gen).double().numpy())
    model.train()
    return np.concatenate(outs) * dscaler.scale + dscaler.mean


def evaluate(model: WorldModel, td: TrainData) -> dict:
    """L_obs (std-MSE of the box target on the validation transitions) and corner RMSE."""
    delta = predict_delta(model, td.val_batch(), td.dscaler)
    y_pred = box_target_from_delta(td.val.s, delta)
    return {
        "l_obs": td.yscaler.std_mse(y_pred, td.val.y),
        "corner_rmse_mm": corner_rmse_mm(td.val.s, y_pred, td.val.y),
    }


def fit_loss(model: WorldModel, td: TrainData, idx: np.ndarray, loss_dims: list[int] | None = None) -> float:
    """The shared training loss of the model's prediction on training examples (made-up and overfit
    checks). For rssm this is its one-step prediction, not its ELBO."""
    delta = predict_delta(model, td.batch(idx), td.dscaler)
    z_pred = torch.as_tensor((delta - td.dscaler.mean) / td.dscaler.scale, dtype=torch.float32)
    return float(weighted_loss(z_pred, td.z(idx), loss_weights(loss_dims)))


# Training -----------------------------------------------------------------------------------------


def multi_step_loss(model: WorldModel, td: TrainData, idx: np.ndarray, w: torch.Tensor) -> torch.Tensor:
    """mlp-ms's training loss (DESIGN.md, Target and loss): an open-loop rollout of
    MULTI_STEP_HORIZON steps from the true start idx.

    Step h feeds back the model's own predicted state as s and into the history (with the true
    action of the step before), takes the true action a_{t+h}, and is scored on the state error
    (s_hat_{t+h+1} - s_{t+h+1}) / delta scale with the shared per-dimension weights. At h = 0 this is
    exactly the shared one-step loss. idx must leave MULTI_STEP_HORIZON - 1 steps in the episode."""
    n = td.norm
    s_mean, s_std = (torch.as_tensor(x, dtype=torch.float32) for x in (n.s_mean, n.s_std))
    d_mean, d_scale = (torch.as_tensor(x, dtype=torch.float32) for x in (td.dscaler.mean, td.dscaler.scale))
    b = td.batch(idx)
    s, s_n, hs_n, ha_n, mask = b.s, b.s_n, b.hs_n, b.ha_n, b.mask
    total = torch.zeros(())
    for h, alpha in enumerate(c.MULTI_STEP_WEIGHTS):
        bh = td.batch(idx + h)  # true action and true next state of step t + h
        step = Batch(s_n, hs_n, ha_n, mask, bh.a_n, s, bh.a)
        s_next = s + model(step) * d_scale + d_mean
        err = (s_next - (bh.s + td.batch_delta(idx + h))) / d_scale
        total = total + alpha * ((err * err) * w).sum(-1).mean()
        hs_n = torch.cat([hs_n[:, 1:], s_n[:, None]], dim=1)
        ha_n = torch.cat([ha_n[:, 1:], bh.a_n[:, None]], dim=1)
        mask = torch.cat([mask[:, 1:], mask.new_ones(len(idx), 1)], dim=1)
        s, s_n = s_next, (s_next - s_mean) / s_std
    return total


def rssm_windows(td: TrainData, e: np.ndarray, t0: np.ndarray, length: int) -> tuple[torch.Tensor, ...]:
    """Windows of HISTORY_LEN burn-in states plus `length` scored transitions from step t0.

    State k of a window is step t0 - HISTORY_LEN + k (k = 0 .. HISTORY_LEN + length); steps before
    0 are masked. The first state's previous action is zero, as in one-step prediction."""
    ep = td.episode_arrays()
    L = c.HISTORY_LEN
    K = L + length + 1
    idx = t0[:, None] - L + np.arange(K)[None, :]
    mask = idx >= 0
    ci = np.clip(idx, 0, None)
    et = torch.as_tensor(e)[:, None]
    states = ep["s"][et, torch.as_tensor(ci)]
    a_idx = np.clip(idx - 1, 0, None)
    a_ok = (idx - 1 >= 0) & (np.arange(K)[None, :] >= 1)
    prev_a = ep["a"][et, torch.as_tensor(a_idx)] * torch.as_tensor(a_ok, dtype=torch.float32)[..., None]
    z = ep["z"][et, torch.as_tensor(a_idx)]
    scored = torch.as_tensor(np.arange(K)[None, :] >= L + 1, dtype=torch.float32).expand(len(e), K)
    return states, prev_a, torch.as_tensor(mask, dtype=torch.float32), z, scored


def train(cfg: TrainConfig, td: TrainData, ckpt_dir: Path | None = None, evaluate_val: bool = True) -> dict:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(stream_seed(cfg.seed, "init"))
    model = build(cfg.arch, cfg.width, td.norm, td.dscaler)
    if cfg.arch == "rssm":
        model.sampling_seed = stream_seed(cfg.seed, "rssm_predict")
        if cfg.rssm_variant == "dv3":
            model.free_nats = c.RSSM_DV3_FREE_NATS
            model.kl_balance = (c.RSSM_DV3_KL_DYN, c.RSSM_DV3_KL_REP)
        elif cfg.rssm_variant != "design":
            raise ValueError(f"unknown rssm variant {cfg.rssm_variant}")
    rng_batch = np.random.default_rng(stream_seed(cfg.seed, "batch"))
    rng_swap = np.random.default_rng(stream_seed(cfg.seed, "shortcut"))
    gen_rssm = torch.Generator().manual_seed(stream_seed(cfg.seed, "rssm_train"))
    if cfg.optimiser != "adam":
        raise ValueError("only adam is implemented")
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)  # constant learning rate, no schedule
    w = loss_weights(cfg.loss_dims)
    schedule = cfg.checkpoint_every or checkpoint_schedule(cfg.updates)
    pool = np.arange(len(td.train)) if td.subset is None else td.subset
    if cfg.arch == "mlp-ms":  # starts that leave the rollout inside the episode
        pool = pool[td.train.step[pool] <= td.split.steps - c.MULTI_STEP_HORIZON]
    swapped = drawn = 0
    if cfg.arch == "mlp-shortcut":
        if cfg.shortcut_p is None or td.split.shadow is None:
            raise ValueError("mlp-shortcut needs p and shadow actions")
        shadow_n = torch.as_tensor(norm_action(td.train.shadow, td.norm), dtype=torch.float32)
        shadow_raw = torch.as_tensor(td.train.shadow, dtype=torch.float32)

    curve: list[dict] = []
    t_train = t_eval = 0.0
    running, n_running = 0.0, 0

    stopped_at_plateau = False

    def record(u: int) -> None:
        nonlocal t_eval, stopped_at_plateau
        t0 = time.perf_counter()
        row = {"updates": u, "train_loss": running / n_running if n_running else None}
        if evaluate_val and td.val is not None:
            row.update(evaluate(model, td))
        curve.append(row)
        if ckpt_dir is not None and u > 0:
            save_checkpoint(ckpt_dir / f"u{u:06d}.pt", model, cfg, td, u)
        if cfg.stop_at_plateau and "l_obs" in row:
            ups = [r["updates"] for r in curve]
            stopped_at_plateau = plateau(ups, [r["l_obs"] for r in curve])["first_update_met"] is not None
        t_eval += time.perf_counter() - t0

    record(0)
    for u in range(1, cfg.updates + 1):
        t0 = time.perf_counter()
        if cfg.arch == "rssm":
            if td.subset is None:
                e = rng_batch.integers(0, td.split.n_episodes, cfg.seq_batch)
                t_start = rng_batch.integers(0, td.split.steps - cfg.seq_len + 1, cfg.seq_batch)
                length = cfg.seq_len
            else:
                pick = (
                    pool
                    if len(pool) <= cfg.seq_batch
                    else rng_batch.choice(pool, cfg.seq_batch, replace=False)
                )
                e, t_start, length = td.train.episode[pick], td.train.step[pick], 1
            states, prev_a, mask, z, scored = rssm_windows(td, e, t_start, length)
            loss = model.elbo(states, prev_a, mask, z, scored, gen_rssm).loss
        else:
            idx = (
                pool if len(pool) <= cfg.batch_size else rng_batch.choice(pool, cfg.batch_size, replace=False)
            )
            b = td.batch(idx)
            if cfg.arch == "mlp-shortcut":
                swap = rng_swap.random(len(idx)) < cfg.shortcut_p
                swapped += int(swap.sum())
                drawn += len(idx)
                st = torch.as_tensor(swap)[:, None]
                ti = torch.as_tensor(idx)
                b = b.with_action(torch.where(st, shadow_raw[ti], b.a), torch.where(st, shadow_n[ti], b.a_n))
            if cfg.arch == "pnn":
                loss = model.nll(b, td.z(idx), w)
            elif cfg.arch == "mlp-ms":
                loss = multi_step_loss(model, td, idx, w)
            else:
                loss = weighted_loss(model(b), td.z(idx), w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        running += loss.item()
        n_running += 1
        t_train += time.perf_counter() - t0
        if u in schedule:
            record(u)
            running, n_running = 0.0, 0
            if stopped_at_plateau:
                break

    out = {
        "config": asdict(cfg),
        "params": model.n_params(),
        "curve": curve,
        "final_lr": [g["lr"] for g in opt.param_groups],
        "updates_done": u if cfg.updates else 0,
        "stopped_at_plateau": stopped_at_plateau,
        "train_seconds": t_train,
        "eval_seconds": t_eval,
        "model": model,
    }
    if cfg.arch == "mlp-shortcut":
        out["shortcut_swapped_share"] = swapped / max(drawn, 1)
    if cfg.arch == "contact_wm":
        out["contact_wm_physical_values"] = model.physical_values()
    return out


def save_checkpoint(path: Path, model: WorldModel, cfg: TrainConfig, td: TrainData, updates: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "architecture": cfg.arch,
        "width": cfg.width,
        "seed": cfg.seed,
        "updates": updates,
        "trained": True,
        "phase": current_phase(),
        "family": cfg.family or cfg.arch,
        "pilot_confirmation": cfg.pilot_confirmation,
        "data": td.data_meta,
        "target": cfg.target_name,
        "params": model.n_params(),
        "config": asdict(cfg),
        "torch": torch.__version__,
    }
    if cfg.arch == "rssm":
        meta["rssm_sampling_seed"] = model.sampling_seed
    torch.save(
        {
            "meta": meta,
            "state_dict": model.state_dict(),
            "input_norm": td.norm.to_dict(),
            "delta_scaler": td.dscaler.to_dict(),
            "box_scaler": td.yscaler.to_dict(),
        },
        path,
    )
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1))
