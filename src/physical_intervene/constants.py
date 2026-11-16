"""Pre-registered constants from docs/DESIGN.md.

Values here are fixed at freeze 1 and must match the design text. Items the design marks
*set by pilot* (model widths, optimiser, learning rate, batch size, number of updates) are not
here; they are recorded in frozen/models.json at freeze 2.
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

# Behaviour policy.
POLICY_PUSH_SPEED_M_S = 0.10
POLICY_NOISE_STD_M_S = 0.02
POLICY_EPSILON = 0.2
PUSHER_START_MIN_DIST_M = 0.05

# Splits.
N_TRAIN_EPISODES = 1_000
N_VAL_EPISODES = 100
N_PILOT_EPISODES = 200
N_PILOT_STATES = 200
N_TEST_STATES = 1_000

# Held-out states and interventions.
TEST_STATE_STEP_RANGE = (10, 90)
TEST_STATE_MAX_GAP_M = 0.03
N_AIMED_INTERVENTIONS = 12
N_UNIFORM_INTERVENTIONS = 12
N_INTERVENTIONS = N_AIMED_INTERVENTIONS + N_UNIFORM_INTERVENTIONS
AIMED_SPEED_RANGE_M_S = (0.05, 0.15)
AIMED_MAX_ANGLE_DEG = 45.0

# Models.
HISTORY_LEN = 8
RSSM_PREDICTION_SAMPLES = 16
KNN_NEIGHBOURS = 5
CHECKPOINT_EVERY_UPDATES = 1_000
RANDOM_SEEDS: tuple[int, ...] = (11, 22, 33, 44, 55)  # same as evm-intervene

# Target scaling: std floors for (dx m, dy m, dtheta rad).
TARGET_STD_FLOORS = (1e-4, 1e-4, 1e-3)

# Matching and gates.
MATCH_TARGET_FACTOR = 1.1  # V* = 1.1 x largest per-architecture median best L_obs
MATCHED_LOSS_REL_TOL = 0.05
BOOTSTRAP_REPLICATES = 5_000
G1_ALPHA = 0.05
G1_MIN_REL_GAP = 0.20
G2_DECOUPLED_BELOW = 0.50
G2_REFUTED_AT_OR_ABOVE = 0.80
