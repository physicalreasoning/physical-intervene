"""Pre-registered constants (docs/DESIGN.md, "Where the numbers live").

Values here are fixed at freeze 1. Where a value is also stated in DESIGN.md, the two must agree.
Items the design marks *set by pilot* (optimiser, learning rate, batch size, number of
updates) and p for mlp-shortcut (computed after the main runs) are not here; they are recorded in
frozen/models.json at freeze 2.
Sources for the defaults are in the project notes, not in this file.
"""

from __future__ import annotations

# Environment and timing.
PHYSICS_STEP_S = 0.002
CONTROL_STEP_S = 0.1
PHYSICS_STEPS_PER_CONTROL = 50
EPISODE_CONTROL_STEPS = 100
ACTION_MAX_SPEED_M_S = 0.15  # each of (vx, vy) in [-0.15, 0.15]
BOX_SIDE_M = 0.10
PUSHER_RADIUS_M = 0.01
PUSHER_MASS_KG = 0.1
# Velocity actuator: force = kv * (ctrl - velocity), so the response time is mass / kv.
ACTUATOR_TIME_CONSTANT_S = 0.010
ACTUATOR_KV = PUSHER_MASS_KG / ACTUATOR_TIME_CONSTANT_S  # 10 N s/m

# Behaviour policy.
POLICY_PUSH_SPEED_M_S = 0.10
POLICY_NOISE_STD_M_S = 0.02
POLICY_EPSILON = 0.2
# Within one step's reach of the box (gap at most ACTION_MAX_SPEED_M_S x CONTROL_STEP_S = 15 mm) the
# policy uses EPSILON_NEAR instead. It starts at the first ladder value and is lowered one step each
# time Phase 0's intervention-coverage check fails (docs/SMOKE_TEST.md); the final value is recorded
# in freeze 1. Never 0.
POLICY_NEAR_BOX_GAP_M = 0.015
POLICY_EPSILON_NEAR_LADDER: tuple[float, ...] = (0.2, 0.05, 0.02)
# Current value. Phase 0 attempt 0 used 0.2 and failed intervention coverage (attempt 0 record).
POLICY_EPSILON_NEAR = 0.05
PUSHER_START_MIN_DIST_M = 0.05

# Splits.
N_TRAIN_EPISODES = 1_000
N_VAL_EPISODES = 100
N_SMOKE_EPISODES = 50
N_SMOKE_STATES = 200
N_PILOT_EPISODES = 200
N_PILOT_STATES = 200
N_TEST_STATES = 1_000

# Held-out states and interventions.
TEST_STATE_STEP_RANGE = (10, 90)
# Gap = shortest distance from the pusher's surface to the box's surface. Must stay below the
# farthest the pusher can travel in one control step (0.15 m/s x 0.1 s = 15 mm).
TEST_STATE_MAX_GAP_M = 0.010
N_AIMED_INTERVENTIONS = 12
N_UNIFORM_INTERVENTIONS = 12
N_INTERVENTIONS = N_AIMED_INTERVENTIONS + N_UNIFORM_INTERVENTIONS
AIMED_SPEED_RANGE_M_S = (0.05, 0.15)
AIMED_MAX_ANGLE_DEG = 45.0

# Phase 0 (smoke test) pass conditions.
SMOKE_MIN_FRAC_STATES_WITH_PUSH = 0.90
SMOKE_MIN_MEAN_PUSH_PER_STATE = 6
SMOKE_PLANTED_NOISE_RATIO = 1.5  # noise std on push transitions, noisy-b over noisy-a
# Pusher can move the box (docs/SMOKE_TEST.md): the actuator force available at the slowest aimed
# intervention speed must be at least this multiple of the force needed to start the box sliding.
SMOKE_MIN_PUSH_FORCE_MARGIN = 2.0
# Intervention coverage (docs/SMOKE_TEST.md): a pool of behaviour episodes from smoke seeds, the
# size of the training split. A pair's distance is the mean distance from its action to the
# SMOKE_COVERAGE_NEIGHBOURS nearest actions taken at its SMOKE_COVERAGE_STATE_NEIGHBOURS nearest pool
# states. tau is this quantile of the behaviour pushes' distances; pass if at least
# SMOKE_MIN_COVERAGE_TAIL_SHARE of push interventions lie farther than tau.
N_SMOKE_COVERAGE_POOL_EPISODES = 1_000
SMOKE_COVERAGE_NEIGHBOURS = 5
SMOKE_COVERAGE_STATE_NEIGHBOURS = 100
SMOKE_COVERAGE_TAIL_QUANTILE = 0.95
SMOKE_MIN_COVERAGE_TAIL_SHARE = 0.25

# Push stratum: a contact transition is a push if the box centre moves at least PUSH_MIN_DISPLACEMENT_M
# or turns at least PUSH_MIN_ROTATION_DEG over the step; otherwise a graze. Defaults; fixed in Phase 0
# before freeze 1 (docs/DESIGN.md, Interventions).
PUSH_MIN_DISPLACEMENT_M = 0.002
PUSH_MIN_ROTATION_DEG = 1.0

# Models.
HISTORY_LEN = 8
RSSM_PREDICTION_SAMPLES = 16
# RSSM structure and objective, following PlaNet (Hafner et al. 2019, Appendix A).
RSSM_SEQUENCE_LEN = 50
RSSM_FREE_NATS = 3.0
RSSM_KL_WEIGHT = 1.0
RSSM_MEDIUM_GRU_UNITS = 200
RSSM_MEDIUM_LATENT_DIM = 30

# Training loss: weights on the box's 7 and the pusher's 4 state numbers (std-MSE each).
LOSS_WEIGHT_BOX = 0.5
LOSS_WEIGHT_PUSHER = 0.5

# Widths: approximate parameter counts for small, medium, large, equal across architectures.
WIDTH_PARAM_COUNTS: tuple[int, ...] = (10_000, 100_000, 1_000_000)

# Training budget (DESIGN.md, Models): no early stopping. The pilot sets each architecture's number
# of updates as the point where L_obs has improved by no more than PLATEAU_MIN_REL_IMPROVEMENT over
# the last PLATEAU_CHECKPOINTS checkpoints at CHECKPOINT_EVERY_UPDATES spacing, times
# BUDGET_FACTOR_OVER_PLATEAU. Learning rate is constant.
PLATEAU_CHECKPOINTS = 10
PLATEAU_MIN_REL_IMPROVEMENT = 0.01
BUDGET_FACTOR_OVER_PLATEAU = 1.5
MAIN_ARCHITECTURES: tuple[str, ...] = ("mlp", "gru", "rssm", "contact_wm")
# contact_wm: sub-steps per control step, and generic initial values for its learned parameters
# (never MuJoCo's values; recorded in freeze 1).
CONTACT_WM_SUBSTEPS = 10
CONTACT_WM_INIT_BOX_HALF_WIDTH_M = 0.04
CONTACT_WM_INIT_PUSHER_RADIUS_M = 0.015
CONTACT_WM_INIT_FRICTION = 0.5
KNN_NEIGHBOURS = 5
# mlp-shortcut: during training, replace a_t by the transition's shadow action (a fresh draw from the
# behaviour policy at the same state) with probability p; p is the largest value in this grid with
# L_mlp + p^2 * D_obs <= V*, computed after the main runs (DESIGN.md, Controls).
SHORTCUT_P_GRID: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
CHECKPOINT_EVERY_UPDATES = 1_000
# Denser checkpoints early, where L_obs falls fastest, so matching at V* does not overshoot.
CHECKPOINT_DENSE_EVERY_UPDATES = 100
CHECKPOINT_DENSE_UNTIL_UPDATES = 2_000
# Gap along training (DESIGN.md, Secondary analyses): shared levels as multiples of V*, plus the
# pair's own V*_pair when lower.
CURVE_LEVEL_FACTORS: tuple[float, ...] = (2.0, 1.5, 1.0)
# Training seeds (docs/background/RATIONALE.md, Why 10 seeds). Medium width, which G1 uses, has 10;
# small and large widths have the first 5, which are evm-intervene's seeds.
MAIN_SEEDS: tuple[int, ...] = tuple(range(11, 111, 11))  # 11, 22, ..., 110
EVM_SEEDS: tuple[int, ...] = (11, 22, 33, 44, 55)
SECONDARY_WIDTH_SEEDS: tuple[int, ...] = EVM_SEEDS
TWIN_SEEDS: tuple[int, ...] = tuple(range(121, 221, 11))  # 121, 132, ..., 220

# Target scaling: std floors for (dx m, dy m, dtheta rad).
TARGET_STD_FLOORS = (1e-4, 1e-4, 1e-3)

# Matching and gates.
MATCH_TARGET_FACTOR = 1.1  # V* = 1.1 x largest per-architecture median best L_obs
MATCHED_LOSS_REL_TOL = 0.05
BOOTSTRAP_REPLICATES = 5_000  # gated intervals and p-values (G1, G2)
DESCRIPTIVE_BOOTSTRAP_REPLICATES = 2_000  # std_mse_ci95, matched_obs_val_ci95; as evm-intervene
G1_ALPHA = 0.05
G1_MIN_REL_GAP = 0.20
G1_MIN_SEEDS_PER_ARCHITECTURE = 6  # of 10; fewer matched seeds left drops the architecture from G1
G1_MIN_SEEDS_SECONDARY_WIDTHS = 3  # of 5, at the small and large widths
G2_DECOUPLED_BELOW = 0.50
G2_REFUTED_AT_OR_ABOVE = 0.80

# ---------------------------------------------------------------------------------------------
# Implementation settings chosen in Phase 0 (milestone 1). Physical values of the scene live in
# assets/pushing.xml; the ones code needs are repeated here and tests check the two agree.
# ---------------------------------------------------------------------------------------------

# Scene (assets/pushing.xml).
BOX_HEIGHT_M = 0.05
BOX_MASS_KG = 0.15
PUSHER_HEIGHT_M = 0.04
BOX_SLIDE_FRICTIONLOSS_N = 0.1  # on each of box_x and box_y
BOX_HINGE_FRICTIONLOSS_NM = 0.004  # on box_yaw
PUSHER_BOX_FRICTION = 0.3
ACTUATOR_FORCE_LIMIT_N = 2.0  # velocity actuators' forcerange is [-2, 2] N

# Behaviour policy regions and controller (policy.py). The goal is a position only.
BOX_START_XY_RANGE_M = (-0.10, 0.10)  # box start x and y, each uniform; yaw uniform on (-pi, pi]
PUSHER_START_XY_RANGE_M = (-0.25, 0.25)  # pusher start x and y, redrawn until the gap is >= 5 cm
GOAL_XY_RANGE_M = (-0.20, 0.20)  # goal x and y, each uniform
GOAL_MIN_DIST_M = 0.05  # a goal is redrawn until it is at least this far from the box centre
GOAL_REACHED_DIST_M = 0.02  # box centre this close to the goal: a new goal is drawn
POLICY_APPROACH_CLEARANCE_M = 0.02  # approach point: this far behind the face, surface to surface
POLICY_ORBIT_CLEARANCE_M = 0.02  # orbit radius: box half-diagonal + pusher radius + this
POLICY_ORBIT_STEP_DEG = 45.0  # orbit waypoint: this far round the box per step at most
POLICY_ORBIT_SWITCH_DEG = 30.0  # within this angle of the push side, go straight to the approach point
POLICY_APPROACH_GAIN = 0.7  # fraction of the distance to the waypoint commanded per control step
POLICY_APPROACH_MAX_SPEED_M_S = 0.12
POLICY_PUSH_LATERAL_TOL_M = 0.01  # push mode only while the pusher is this close to the push line
POLICY_PUSH_LATERAL_TIME_S = 0.3  # lateral correction in push mode: offset / this time
POLICY_PUSH_BEHIND_TOL_M = 0.005  # push mode allows the pusher this far past the face (penetration)

# Seeds (episodes.py). Episode seed ranges per split, [start, stop), disjoint. The test seed is
# never derived before freeze 2 (DESIGN.md, Splits), so it has no range here.
SEED_RANGE_TRAIN = (1_000_000, 2_000_000)
SEED_RANGE_VALIDATION = (2_000_000, 3_000_000)
SEED_RANGE_SMOKE = (3_000_000, 4_000_000)
SEED_RANGE_PILOT = (4_000_000, 5_000_000)
# Inside the smoke range: Phase 0 attempt k (one per epsilon_near ladder value) starts at
# SEED_RANGE_SMOKE[0] + k * SMOKE_ATTEMPT_STRIDE, so a rerun uses new smoke seeds. Attempt 0 ran at
# epsilon_near 0.2; attempt 1 at 0.05; attempt 2 at 0.05 with the push label against the no-contact
# reference.
SMOKE_ATTEMPT = 2
SMOKE_ATTEMPT_STRIDE = 300_000
SMOKE_EPISODES_OFFSET = 0  # the 50 smoke episodes
SMOKE_POOL_OFFSET = 100_000  # the 1,000-episode coverage pool
SMOKE_STATES_OFFSET = 200_000  # one fresh episode per test-like state, skipped seeds included
# Positive-control feasibility: policy draws per state for m (DESIGN.md, Controls).
FEASIBILITY_POLICY_DRAWS = 100
