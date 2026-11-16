"""Consistency checks on the pre-registered constants in docs/DESIGN.md."""

from __future__ import annotations

from physical_intervene import constants as c


def test_timing_is_consistent() -> None:
    assert round(c.CONTROL_STEP_S / c.PHYSICS_STEP_S) == c.PHYSICS_STEPS_PER_CONTROL


def test_intervention_count_matches_design() -> None:
    assert c.N_INTERVENTIONS == 24


def test_seeds_match_evm_intervene() -> None:
    assert c.RANDOM_SEEDS == (11, 22, 33, 44, 55)


def test_gate_thresholds_are_ordered() -> None:
    assert 0.0 < c.G2_DECOUPLED_BELOW < c.G2_REFUTED_AT_OR_ABOVE < 1.0
    assert 0.0 < c.MATCHED_LOSS_REL_TOL < c.G1_MIN_REL_GAP
