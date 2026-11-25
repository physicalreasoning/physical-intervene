"""Training-side data checks (SMOKE_TEST.md, Training pipeline checks, part 1)."""

from __future__ import annotations

import numpy as np
import pytest

from physical_intervene import constants as c
from physical_intervene import training_checks as TC
from physical_intervene.data import (
    Split,
    box_scaler,
    box_target_from_delta,
    load_split,
    make_examples,
    split_overlap,
)
from physical_intervene.metrics import corner_rmse_mm, excluded, matched_checkpoint, plateau, v_star


def test_alignment_on_hand_built_episode() -> None:
    r = TC.check_alignment()
    assert r["pass"], r["errors"]


def test_target_wraps_across_pi() -> None:
    sp = TC.hand_built_split()
    ex = make_examples(sp)
    assert np.allclose(ex.y[:, 2], 0.02, atol=1e-12)
    assert np.allclose(ex.y[:, :2], [0.001, -0.002], atol=1e-15)
    # A predicted change whose (cos, sin) crosses pi gives a small dtheta, not about 2 pi.
    assert np.allclose(box_target_from_delta(ex.s, ex.delta)[:, 2], 0.02, atol=1e-12)


def test_scaler_uses_training_split_only() -> None:
    sp = TC.hand_built_split(n_steps=40)
    ex = make_examples(sp)
    train, val = ex.subset(np.arange(20)), ex.subset(np.arange(20, 40))
    s1 = box_scaler(train)
    val.y = val.y * 100.0
    assert np.array_equal(s1.scale, box_scaler(train).scale)
    assert np.all(s1.scale >= np.asarray(c.TARGET_STD_FLOORS))  # floors apply


def test_split_overlap_detects_shared_seed_and_transition() -> None:
    a = TC.hand_built_split()
    b = TC.hand_built_split()
    b.name, b.seed = "other", np.array([1])
    r = split_overlap([a, b])
    assert r["pairs"]["hand_built/other"]["shared_seeds"] == 0
    assert r["pairs"]["hand_built/other"]["shared_transition_hashes"] == a.steps
    b.obs = b.obs + 1.0
    assert split_overlap([a, b])["pairs"]["hand_built/other"]["shared_transition_hashes"] == 0


def test_training_side_refuses_states_file(tmp_path) -> None:
    p = tmp_path / "states.npz"
    np.savez(p, obs=np.zeros(1))
    with pytest.raises(ValueError):
        load_split(p, "x")


def test_corner_rmse_is_zero_for_exact_and_mm_for_shift() -> None:
    s = np.zeros((3, 11))
    s[:, 2] = 1.0
    y = np.array([[0.001, 0.0, 0.0]] * 3)
    assert corner_rmse_mm(s, y, y) == 0.0
    assert corner_rmse_mm(s, y + [0.002, 0.0, 0.0], y) == pytest.approx(2.0)


def test_matching_rules() -> None:
    curves = {"a": [[1.0, 0.5, 0.4]], "b": [[1.0, 0.8, 0.6, 0.68]]}
    vs = v_star({k: [min(c_) for c_ in v] for k, v in curves.items()})
    assert vs == pytest.approx(0.66)
    assert matched_checkpoint(curves["b"][0], vs) == 3  # 0.68 is closest to 0.66, and within 5%
    assert not excluded(curves["b"][0], vs)
    assert excluded([1.0, 0.9], vs)


def test_plateau_criterion() -> None:
    u = list(range(1000, 13_001, 1000))
    L = [1.0 / k for k in range(1, 12)] + [1 / 11, 1 / 11]
    r = plateau(u, L)
    assert r["evaluable"] and r["first_update_met"] is None
    flat = [0.5] * len(u)
    r = plateau(u, flat)
    assert r["first_update_met"] == 11_000 and r["met_at_last_checkpoint"]


def test_smoke_data_checks_if_present() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "data" / "smoke" / f"attempt{c.SMOKE_ATTEMPT}"
    if not (root / "episodes.npz").exists():
        pytest.skip("smoke data not generated (scripts/phase0_env.py)")
    val = load_split(root / "episodes.npz", "validation")
    assert TC.check_target(val)["pass"]
    assert isinstance(val, Split)
