"""Training examples, target scalers and split checks (DESIGN.md, Target and loss, Data, Models).

An example is one transition (episode e, step t):
- s_t, the current state (11 numbers);
- the history: the last HISTORY_LEN (state, action) pairs before t, oldest first. Before step
  HISTORY_LEN the missing pairs are copies of s_0 with a zero action, and a mask marks them (1 real,
  0 padding). The history never contains anything from step t or later;
- a_t, the action;
- the state change s_{t+1} - s_t (11 numbers), the training target;
- the box target y_t = (dx, dy, dtheta), dtheta wrapped to (-pi, pi], the scored target.

Scalers are fitted on the training split only, by construction: they take only training arrays.
This module never opens the test-like states file (states.npz), which belongs to the intervention side.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from physical_intervene import constants as c
from physical_intervene.sim import OBS_DIM, wrap_angle

FORBIDDEN_FILES = ("states.npz",)  # intervention side: never read by the training side
BOX_DIMS = slice(0, 7)
PUSHER_DIMS = slice(7, 11)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass
class Split:
    """Behaviour episodes of one split, as stored by episodes.generate."""

    name: str
    obs: np.ndarray  # (n, T + 1, 11)
    action: np.ndarray  # (n, T, 2)
    target: np.ndarray  # (n, T, 3), the stored box target from qpos
    seed: np.ndarray  # (n,)
    shadow: np.ndarray | None = None  # (n, T, 2), mlp-shortcut only
    file: str = ""
    sha256: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def n_episodes(self) -> int:
        return self.obs.shape[0]

    @property
    def steps(self) -> int:
        return self.action.shape[1]


def load_split(path: Path, name: str) -> Split:
    path = Path(path)
    if path.name in FORBIDDEN_FILES:
        raise ValueError(f"{path.name} is intervention-side data; the training side never reads it")
    with np.load(path) as f:
        d = {k: f[k] for k in f.files}
    return Split(
        name=name,
        obs=d["obs"],
        action=d["action"],
        target=d["target"],
        seed=d["seed"],
        shadow=d.get("shadow"),
        file=path.name,
        sha256=file_sha256(path),
        extra={"contact": d.get("contact")},
    )


def theta(obs: np.ndarray) -> np.ndarray:
    return np.arctan2(obs[..., 3], obs[..., 2])


def box_target_from_states(s_t: np.ndarray, s_next: np.ndarray) -> np.ndarray:
    """(dx, dy, dtheta) of the box between two observations, dtheta wrapped to (-pi, pi]."""
    d_xy = s_next[..., 0:2] - s_t[..., 0:2]
    d_th = wrap_angle(theta(s_next) - theta(s_t))
    return np.concatenate([d_xy, np.asarray(d_th)[..., None]], axis=-1)


def box_target_from_delta(s_t: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """The box target implied by a (predicted) state change: dtheta from atan2 of the predicted cos
    and sin, minus the current angle, wrapped. The predicted (cos, sin) need not have unit norm."""
    return box_target_from_states(s_t, s_t + delta)


@dataclass
class Examples:
    """Every transition of a split, flattened, with its history (float64, physical units)."""

    episode: np.ndarray  # (N,)
    step: np.ndarray  # (N,)
    s: np.ndarray  # (N, 11)
    hist_s: np.ndarray  # (N, L, 11)
    hist_a: np.ndarray  # (N, L, 2)
    hist_mask: np.ndarray  # (N, L)
    a: np.ndarray  # (N, 2)
    s_next: np.ndarray  # (N, 11)
    delta: np.ndarray  # (N, 11)
    y: np.ndarray  # (N, 3), from the states
    shadow: np.ndarray | None  # (N, 2)

    def __len__(self) -> int:
        return len(self.s)

    def subset(self, idx: np.ndarray) -> Examples:
        return Examples(**{k: (None if v is None else v[idx]) for k, v in self.__dict__.items()})


def history_index(steps: np.ndarray, length: int = c.HISTORY_LEN) -> tuple[np.ndarray, np.ndarray]:
    """Step indices of the history of each step (oldest first) and the mask of real entries.

    Padding positions point at step 0 (their state is s_0; their action is replaced by zero)."""
    k = np.arange(-length, 0)
    idx = steps[:, None] + k[None, :]
    mask = idx >= 0
    return np.where(mask, idx, 0), mask


def make_examples(split: Split, episodes: np.ndarray | None = None) -> Examples:
    e_ids = np.arange(split.n_episodes) if episodes is None else np.asarray(episodes)
    T = split.steps
    e = np.repeat(e_ids, T)
    t = np.tile(np.arange(T), len(e_ids))
    h_idx, mask = history_index(t)
    hist_s = split.obs[e[:, None], h_idx]
    hist_a = np.where(mask[..., None], split.action[e[:, None], h_idx], c.HISTORY_PAD_ACTION)
    s, s_next = split.obs[e, t], split.obs[e, t + 1]
    return Examples(
        episode=e,
        step=t,
        s=s,
        hist_s=hist_s,
        hist_a=hist_a,
        hist_mask=mask.astype(np.float64),
        a=split.action[e, t],
        s_next=s_next,
        delta=s_next - s,
        y=box_target_from_states(s, s_next),
        shadow=None if split.shadow is None else split.shadow[e, t],
    )


@dataclass(frozen=True)
class TargetScaler:
    """Per-dimension mean and floored standard deviation, from the training split only."""

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, y_train: np.ndarray, floors: tuple[float, ...] | np.ndarray) -> TargetScaler:
        y = np.asarray(y_train, float).reshape(-1, np.shape(y_train)[-1])
        return cls(mean=y.mean(axis=0), scale=np.maximum(y.std(axis=0), np.asarray(floors, float)))

    def per_sample(self, pred: np.ndarray, true: np.ndarray) -> np.ndarray:
        """Mean over dimensions of the squared error, each divided by its standard deviation."""
        d = (np.asarray(pred) - np.asarray(true)) / self.scale
        return np.mean(d * d, axis=-1)

    def std_mse(self, pred: np.ndarray, true: np.ndarray) -> float:
        return float(np.mean(self.per_sample(pred, true)))

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> TargetScaler:
        return cls(mean=np.asarray(d["mean"], float), scale=np.asarray(d["scale"], float))


def box_scaler(train: Examples) -> TargetScaler:
    """The std-MSE scaler of the scored box target (DESIGN.md, Target and loss)."""
    return TargetScaler.fit(train.y, c.TARGET_STD_FLOORS)


def delta_scaler(train_delta: np.ndarray) -> TargetScaler:
    """The scaler of the full state change, for the training loss."""
    return TargetScaler.fit(train_delta, c.DELTA_STATE_STD_FLOORS)


@dataclass(frozen=True)
class InputNorm:
    """Standardisation of model inputs (states and actions), from the training split only."""

    s_mean: np.ndarray
    s_std: np.ndarray
    a_mean: np.ndarray
    a_std: np.ndarray

    @classmethod
    def fit(cls, train: Examples) -> InputNorm:
        return cls(s_mean=train.s.mean(0), s_std=train.s.std(0), a_mean=train.a.mean(0), a_std=train.a.std(0))

    def to_dict(self) -> dict:
        return {k: v.tolist() for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict) -> InputNorm:
        return cls(**{k: np.asarray(v, float) for k, v in d.items()})


def training_loss_weights() -> np.ndarray:
    """Per-dimension weights of the shared training loss: 0.5 x mean over the box's 7 numbers plus
    0.5 x mean over the pusher's 4 (each in std-MSE units)."""
    w = np.empty(OBS_DIM)
    w[BOX_DIMS] = c.LOSS_WEIGHT_BOX / 7
    w[PUSHER_DIMS] = c.LOSS_WEIGHT_PUSHER / 4
    return w


# Split checks ------------------------------------------------------------------------------------


def transition_hashes(split: Split) -> set[str]:
    """sha256 of each transition's (s_t, a_t, s_{t+1}) bytes."""
    out = set()
    for e in range(split.n_episodes):
        for t in range(split.steps):
            h = hashlib.sha256()
            for arr in (split.obs[e, t], split.action[e, t], split.obs[e, t + 1]):
                h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
            out.add(h.hexdigest())
    return out


def seed_split_name(seed: int) -> str | None:
    from physical_intervene.episodes import SPLIT_RANGES

    for name, (lo, hi) in SPLIT_RANGES.items():
        if lo <= seed < hi:
            return name
    return None


def split_overlap(splits: list[Split]) -> dict:
    """Seeds and transition hashes shared between any two splits (should be none)."""
    seeds = {s.name: set(int(x) for x in s.seed) for s in splits}
    hashes = {s.name: transition_hashes(s) for s in splits}
    pairs = {}
    for i, a in enumerate(splits):
        for b in splits[i + 1 :]:
            pairs[f"{a.name}/{b.name}"] = {
                "shared_seeds": len(seeds[a.name] & seeds[b.name]),
                "shared_transition_hashes": len(hashes[a.name] & hashes[b.name]),
            }
    return {
        "pairs": pairs,
        "seed_ranges": {s.name: [int(s.seed.min()), int(s.seed.max())] for s in splits},
        "design_split_of_seeds": {
            s.name: sorted({str(seed_split_name(int(x))) for x in s.seed}) for s in splits
        },
        "unique_transitions": {s.name: len(hashes[s.name]) for s in splits},
    }
