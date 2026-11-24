"""Consistency checks on the pre-registered constants in docs/DESIGN.md."""

from __future__ import annotations

from physical_intervene import constants as c


def test_timing_is_consistent() -> None:
    assert round(c.CONTROL_STEP_S / c.PHYSICS_STEP_S) == c.PHYSICS_STEPS_PER_CONTROL


def test_intervention_count_matches_design() -> None:
    assert c.N_INTERVENTIONS == 24


def test_seeds_match_evm_intervene() -> None:
    assert c.EVM_SEEDS == (11, 22, 33, 44, 55)
    assert c.MAIN_SEEDS[: len(c.EVM_SEEDS)] == c.EVM_SEEDS
    assert c.SECONDARY_WIDTH_SEEDS == c.EVM_SEEDS


def test_seed_counts_and_twins() -> None:
    assert len(c.MAIN_SEEDS) == 10 and len(set(c.MAIN_SEEDS)) == 10
    assert len(c.TWIN_SEEDS) == len(c.MAIN_SEEDS)
    assert not set(c.TWIN_SEEDS) & set(c.MAIN_SEEDS)
    assert 0 < c.G1_MIN_SEEDS_PER_ARCHITECTURE <= len(c.MAIN_SEEDS)
    assert 0 < c.G1_MIN_SEEDS_SECONDARY_WIDTHS <= len(c.SECONDARY_WIDTH_SEEDS)


def test_gate_thresholds_are_ordered() -> None:
    assert 0.0 < c.G2_DECOUPLED_BELOW < c.G2_REFUTED_AT_OR_ABOVE < 1.0
    assert 0.0 < c.MATCHED_LOSS_REL_TOL < c.G1_MIN_REL_GAP


def test_test_states_can_reach_the_box_in_one_step() -> None:
    max_travel_m = c.ACTION_MAX_SPEED_M_S * c.CONTROL_STEP_S
    assert c.TEST_STATE_MAX_GAP_M < max_travel_m
    assert c.TEST_STATE_MAX_GAP_M <= c.AIMED_SPEED_RANGE_M_S[1] * c.CONTROL_STEP_S


def test_smoke_push_thresholds_are_feasible() -> None:
    assert 0.0 < c.SMOKE_MIN_FRAC_STATES_WITH_PUSH <= 1.0
    assert 0 < c.SMOKE_MIN_MEAN_PUSH_PER_STATE <= c.N_INTERVENTIONS


def test_coverage_pool_matches_training_split() -> None:
    assert c.N_SMOKE_COVERAGE_POOL_EPISODES == c.N_TRAIN_EPISODES
    assert c.SMOKE_COVERAGE_NEIGHBOURS == c.KNN_NEIGHBOURS
    # Interventions no more unusual than ordinary pushes give a tail share of 1 - quantile.
    assert 1.0 - c.SMOKE_COVERAGE_TAIL_QUANTILE < c.SMOKE_MIN_COVERAGE_TAIL_SHARE < 1.0


def test_epsilon_near_ladder() -> None:
    ladder = c.POLICY_EPSILON_NEAR_LADDER
    assert ladder[0] == c.POLICY_EPSILON
    assert list(ladder) == sorted(ladder, reverse=True) and len(set(ladder)) == len(ladder)
    assert all(e > 0.0 for e in ladder)
    assert abs(c.POLICY_NEAR_BOX_GAP_M - c.ACTION_MAX_SPEED_M_S * c.CONTROL_STEP_S) < 1e-12
    assert c.POLICY_NEAR_BOX_GAP_M >= c.TEST_STATE_MAX_GAP_M
    assert c.POLICY_EPSILON_NEAR in ladder
    # Each ladder value is its own Phase 0 attempt, on its own smoke seeds.
    assert c.SMOKE_ATTEMPT >= ladder.index(c.POLICY_EPSILON_NEAR)
    assert c.SMOKE_COVERAGE_STATE_NEIGHBOURS > c.SMOKE_COVERAGE_NEIGHBOURS


def test_push_force_margin_is_above_one() -> None:
    assert c.SMOKE_MIN_PUSH_FORCE_MARGIN > 1.0


def test_push_threshold_is_below_one_step_travel() -> None:
    assert 0.0 < c.PUSH_MIN_DISPLACEMENT_M < c.ACTION_MAX_SPEED_M_S * c.CONTROL_STEP_S
    assert c.PUSH_MIN_ROTATION_DEG > 0.0


def test_shortcut_grid_is_valid() -> None:
    grid = c.SHORTCUT_P_GRID
    assert list(grid) == sorted(grid)
    assert all(0.0 < p < 1.0 for p in grid)


def test_contact_wm_initial_values_are_not_the_simulator_values() -> None:
    assert c.CONTACT_WM_INIT_BOX_HALF_WIDTH_M != c.BOX_SIDE_M / 2
    assert c.CONTACT_WM_INIT_PUSHER_RADIUS_M != c.PUSHER_RADIUS_M


def test_main_architectures() -> None:
    assert len(set(c.MAIN_ARCHITECTURES)) == 4


def test_actuator_responds_well_within_a_control_step() -> None:
    assert abs(c.ACTUATOR_KV - c.PUSHER_MASS_KG / c.ACTUATOR_TIME_CONSTANT_S) < 1e-9
    assert c.ACTUATOR_TIME_CONSTANT_S <= c.CONTROL_STEP_S / 5


def test_loss_weights_sum_to_one() -> None:
    assert abs(c.LOSS_WEIGHT_BOX + c.LOSS_WEIGHT_PUSHER - 1.0) < 1e-12


def test_widths_increase() -> None:
    w = c.WIDTH_PARAM_COUNTS
    assert len(w) == 3 and list(w) == sorted(w)


def test_training_budget_rule_is_valid() -> None:
    assert c.PLATEAU_CHECKPOINTS >= 1
    assert 0.0 < c.PLATEAU_MIN_REL_IMPROVEMENT < 1.0
    assert c.BUDGET_FACTOR_OVER_PLATEAU > 1.0


def test_checkpoint_schedule() -> None:
    assert 0 < c.CHECKPOINT_DENSE_EVERY_UPDATES < c.CHECKPOINT_EVERY_UPDATES
    assert c.CHECKPOINT_DENSE_UNTIL_UPDATES % c.CHECKPOINT_EVERY_UPDATES == 0
    assert c.CHECKPOINT_EVERY_UPDATES % c.CHECKPOINT_DENSE_EVERY_UPDATES == 0


def test_curve_levels_end_at_v_star() -> None:
    f = c.CURVE_LEVEL_FACTORS
    assert list(f) == sorted(f, reverse=True) and f[-1] == 1.0
