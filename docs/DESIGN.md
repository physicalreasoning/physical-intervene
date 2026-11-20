# physical-intervene, task 1: planar pushing

**Status: draft for discussion (physicalreasoning/forward#5).** It is hashed before any model is
trained (freeze 1) and again before any test state exists (freeze 2). After freeze 1 nothing
below changes except the items marked *set by pilot*. After freeze 2 nothing changes at all. Any
deviation is reported beside the original text, never in place of it. Every result is published,
including a failure to separate.

## Question

Do learned world models with matched observational validation loss predict the same outcome when
an action they were not shown makes the pusher touch the box? Or does low observational error hide
large errors under intervention?

This is the physical counterpart of
[evm-intervene](https://github.com/physicalreasoning/evm-intervene), the market version on
Uniswap V3. It uses the same quantities and gates, and the shared result format drafted in
[INTERVENTION_REPORTING.md](INTERVENTION_REPORTING.md), so the two read side by side (see
*Shared terms*). It also builds on forward's Task 3, interventions on the order book.

## Shared terms with evm-intervene

"evm-intervene" below means what its code computes (`scripts/run_experiment.py`,
`src/evm_intervene/metrics.py`), which is where it differs from its `docs/DESIGN.md`.

| Term | evm-intervene | here |
| --- | --- | --- |
| L_obs, observational validation loss | one-step std-MSE on held-out observational swaps | one-step std-MSE on held-out observational transitions |
| E_int, intervention error | std-MSE on the hard regimes (II cliff + III cascade) | std-MSE on the contact stratum |
| easy stratum | regime I, in-tick | no contact |
| physical-unit error | log-price RMSE, bps | box corner RMSE, mm |
| Δ_int, the issue's *consistency gap* | E_int(challenger) − E_int(reference) at matched checkpoints; JSON `matched_cliff_gap.gap` | same quantity; JSON `matched_contact_gap.gap` |
| std-MSE | `TargetScaler`, training-split std, per-dimension floors | same; floors in this task's units |
| match screen | \|L_a − L_b\| / min(L_a, L_b) ≤ 0.05 | same |
| bootstrap, p-value, Holm | 5,000 replicates over held-out states (one per archive checkpoint, 54); p = min(1, 2(min(#≤0, #≥0)+1)/5,001); Holm across all pairs tested, screened or not | same; held-out simulator states (1,000) |
| seeds | 11, 22, 33, 44, 55 | same |
| E_int aggregation | per-transition loss averaged over seeds, then over all hard transitions, pooled across states | same |
| G0 | simulator matches on-chain price; every model beats `copy` on L_obs (best checkpoints) | bit-exact replay; every main model beats `copy` (matched and best checkpoints); controls (see G0) |
| G1 pass rule | screened pair with Holm p < 0.05, CI excludes zero, relative gap ≥ 20% | same; *not testable* if no pair passes the screen |
| G2 reading | Spearman ρ(L_obs, E_int) over best checkpoints; decoupled if point ρ < 0.50 | same; refuted if point ρ ≥ 0.80 |

### Deliberate differences

These are design choices, not oversights. Each is reported so the two G1s can be read correctly
side by side.

| Item | evm-intervene | here | Why |
| --- | --- | --- | --- |
| Pairs in G1 | each of 3 challengers against one anchor, `conservation_wm` | all 3 pairs of main architectures | there is no structured model that plays the anchor's role |
| Relative gap | Δ / E_int(reference), reference = the anchor | Δ / E_int(reference), reference = the pair member with the lower E_int | with no anchor, this fixes the reference so Δ ≥ 0 and the formula is evm's |
| Loss matching | anchor uses its best checkpoint; other models stop at the checkpoint closest to the anchor's 5-seed mean best L_obs | every architecture takes the checkpoint closest to V* = 1.1 × the largest per-architecture median best L_obs | no model is favoured by being the anchor, and V* is reachable by all |
| Bootstrap resamples | clusters only | clusters and, independently per architecture, the 5 seeds (evm's version is reported too) | stops a seed-to-seed difference from passing as an architecture difference |
| G0.1 | simulator vs on-chain price, median error < 0.05 bps | bit-exact replay of the full integration state | the simulator is the ground truth here, so exactness is checkable |

## Why planar pushing

- **Exact, cheap counterfactuals.** MuJoCo can save its full integration state, restore it and
  step again. From any recorded state we can apply the logged action or any other action and get
  the true outcome of each. A test costs microseconds of simulation.
- **A built-in hard case.** The box moves only when the pusher touches it. Without contact,
  copying the box pose is exactly right, as in-tick swaps are nearly linear in evm-intervene.
  Contact is the analogue of crossing a liquidity cliff: rare in routine data, where the action
  matters most.
- **A shortcut by construction.** The scripted policy makes actions predictable from the state
  and recent history. A model can therefore reach a low validation loss while relying on the
  action less than it should. The state is fully observed, so P(s' | s, a) is the same whether a
  comes from the policy or is forced. Any gap measures what each model learned from correlated,
  unevenly covered data, not missing inputs.
- **Small.** Low-dimensional state, no pixels. Everything trains on a CPU or a free Colab or
  Kaggle GPU.

## Environment

| Item | Value |
| --- | --- |
| Simulator | MuJoCo, exact version pinned in `uv.lock` and recorded in the freeze |
| Platform | linux/amd64, one Docker image pinned by digest, single thread; all ground truth is generated in that image |
| Scene | table plane; box 10 x 10 cm (slide x, slide y, hinge z joints; fixed mass and friction); cylindrical pusher, radius 1 cm, on slide x/y joints driven by velocity actuators |
| Timing | physics step 2 ms; control step 0.1 s (50 physics steps); episodes of 100 control steps |
| Action a_t | commanded pusher velocity (vx, vy), each in [-0.15, 0.15] m/s |
| State s_t | box (x, y, cos θ, sin θ, vx, vy, ω) and pusher (x, y, vx, vy): 11 numbers, fully observed |
| Save / restore | `mj_getState` / `mj_setState` with `mjSTATE_INTEGRATION` (includes the solver warm-start; restoring only qpos and qvel is not bit-exact) |
| Contact label | a transition is *contact* if any pusher-box contact exists at any of its 50 physics steps, in the simulator's own outcome |

## Target and loss

Each model predicts the change in the full state. Scoring uses the box target, three numbers:

y_t = (Δx, Δy, Δθ) of the box over one control step, with Δθ wrapped to (−π, π].

**std-MSE**, as in evm-intervene (`TargetScaler`): the mean over the 3 dimensions of the squared
error, each divided by that dimension's standard deviation on the **training split only**. The
floors on the standard deviation are 1e-4 m, 1e-4 m and 1e-3 rad.

**Physical unit**: the RMS, over transitions, of the mean distance between the predicted and
true positions of the box's 4 corners after the step, in mm. It plays the role of bps in
evm-intervene and is reported, not gated.

## Data

**Behaviour policy (observational).** Each episode draws a goal pose for the box. The policy
moves the pusher around the box to the side opposite the goal, then pushes toward the goal at
0.1 m/s, with Gaussian noise of 0.02 m/s per axis. At each control step, with probability
ε = 0.2, it instead takes an action drawn uniformly from the action box. So every single action is
inside the training support, but most of the training contacts come from the scripted push. The
goal is not observed. Box start pose, pusher start position (at least 5 cm from the box) and goal
are drawn uniformly from fixed regions.

**Splits.** Each split has its own seed range and is never reused.

| Split | Content | Seed |
| --- | --- | --- |
| train | 1,000 episodes (100,000 transitions) | fixed, in freeze 1 |
| validation | 100 episodes (10,000 transitions); L_obs | fixed, in freeze 1 |
| pilot | 200 episodes and 200 test-like states | fixed, in freeze 1; pilot outputs never enter a result |
| test | 1,000 held-out states (see *Interventions*) | derived from the sha256 of `frozen/models.json`, so it cannot exist before every model's weights are hashed |

## Interventions

**Held-out states.** Each test state comes from a fresh behaviour-policy episode. It is the first
step at or after a uniformly drawn step in [10, 90] where the pusher is within 3 cm of the box
surface. Near the box, both contact and no-contact actions are possible.

**Interventions.** From each held-out state, the simulator is restored and run for one control
step under each of K = 24 actions:

- 12 aimed: speed U[0.05, 0.15] m/s, direction toward the box centre ± U[0, 45°];
- 12 uniform: uniform on the action box.

Each action is labelled contact or no contact by the simulator, as above. **Contact is the
primary stratum**, the counterpart of evm-intervene's regimes II and III. A held-out state with no
contact action is dropped from the contact stratum for every model. The logged action from the
same state is also scored, as the observational reference on the same states, and reported
under each model's `logged_action` key. **It is not an intervention and never enters E_int**,
in any stratum, gate or secondary analysis.

For each intervention we also record the behaviour policy's log-density of that action at that
state (mixture of the scripted-plus-noise Gaussian and the uniform), for an exploratory plot.

## Models

All learned models see the same inputs: the current state, the last 8 (state, action) pairs, and
the action. A model may ignore the history but none is denied it. This follows evm-intervene's
equal-observability rule, so a failure cannot come from missing inputs.

| Model | What | Role |
| --- | --- | --- |
| `mlp` | residual MLP on (s_t, history, a_t) → Δs | main |
| `gru` | GRU over the history, then a head on (h_t, a_t) → Δs | main |
| `rssm` | small RSSM (deterministic GRU state plus Gaussian latent), ELBO training; prediction is the mean of 16 samples | main |
| `blind` | as `mlp` with a_t removed | negative control |
| `mlp-random` | as `mlp`, trained on the same number of transitions with ε = 1 (all actions uniform) | positive control |
| `copy` | box does not move | baseline (evm's `copy`) |
| `constvel` | box keeps its current velocity | baseline |
| `knn` | mean Δs of the 5 nearest training transitions in standardised (s_t, a_t) | baseline |

Main models: 3 widths (*small*, *medium*, *large*, exact sizes *set by pilot*) × 5 seeds = 45
runs. Each run saves a checkpoint every 1,000 updates. Controls: medium width, 5 seeds. Optimiser,
learning rate, batch size and number of updates are *set by pilot* on pilot and validation data
only. Each architecture gets its own settings, chosen by the same pilot procedure and budget,
and shared across its seeds and widths.

## Matching observational loss

Fixed now, before training. All steps use validation data only. No test state exists yet.

1. For each medium-width main run, take its best L_obs over checkpoints, and the median over the
   5 seeds per architecture. Let V* = 1.1 × the largest of the three medians, a level every
   architecture can reach.
2. Each medium run's **matched checkpoint** is the one whose L_obs is closest to V*.
3. A pair of architectures passes the **match screen** if their seed-mean matched L_obs satisfy
   |L_a − L_b| / min(L_a, L_b) ≤ 0.05.
4. A run whose L_obs never comes within 5% of V* is excluded, and every exclusion is reported.
   An architecture's matched L_obs, E_int and seed resampling then use its remaining seeds. An
   architecture with fewer than 3 of its 5 seeds left is dropped from G1 and reported; if that
   leaves fewer than two architectures, G1 is *not testable*.

## Gates

Gates run in order. If G0 fails, G1 and G2 are not interpreted, and the result is still
published.

**G0. Simulator and pipeline validity.** All must hold:

1. **Bit-exact replay.** For every held-out state, a fresh process in the pinned image restores
   the saved state and applies the logged action and all 24 interventions. Every resulting full
   integration state is byte-identical to the stored truth (`np.array_equal` on the raw bytes):
   1,000 of 1,000 states, all 25 actions each. The same check on a second machine of the same
   platform is reported, not gated.
2. **Every main model beats `copy` on L_obs**, at both its matched and best checkpoints (seed
   mean). This is evm-intervene's G0.2, which checks best checkpoints only. Controls are not
   main models.
3. **Controls.** On the contact stratum, `blind` has a higher E_int than `mlp-random`, with a 95%
   interval excluding zero (the test can see a gap), and `mlp-random` beats `copy` on E_int (a
   small gap is reachable). The two controls differ in both the input (action removed) and the
   training data (behaviour vs uniform actions). This check only shows the test can detect a gap
   when one is built in; it does not say which of the two differences causes it.

**G1. Matched-loss separation (primary).** For each of the 3 pairs of main architectures:

- In each pair, the *reference* is the architecture with the lower point E_int and the
  *challenger* is the other. Δ_int = E_int(challenger) − E_int(reference) on the contact stratum,
  at matched checkpoints. Each contact transition's std-MSE is averaged over the 5 seeds, then
  over all contact transitions, pooled across states, as in evm-intervene. A state with more
  contact actions therefore weighs more. Because the test is two-sided, choosing the reference
  after scoring does not change p.
- 5,000 paired bootstrap replicates, as in evm-intervene. **Unlike evm-intervene, each replicate
  also resamples the 5 seeds of each architecture with replacement, independently.** The
  held-out states are resampled with replacement and shared across models (paired). Without the
  seed resampling, a difference between two training seeds of the same architecture could pass.
- p = min(1, 2 × (min(#Δ* ≤ 0, #Δ* ≥ 0) + 1) / (5,000 + 1)), as in evm-intervene. Holm
  correction across all 3 pairs, whether or not they pass the screen, as in evm-intervene.
- Relative gap = Δ_int / E_int(reference), evm-intervene's formula (`rel_gap`).
- The same test is also run at best checkpoints and reported as `best_contact_gap`, as
  evm-intervene does. It is not gated.

**Pass**: at least one pair that passes the match screen has a Holm-adjusted p < 0.05, a 95%
interval excluding zero and a relative gap ≥ 20%. **Fail**: no screened pair does. That is a negative result about our own
position, published the same way. If no pair passes the screen, G1 is reported as *not testable*
with the observed L_obs differences.

The issue's original interval (states resampled, seed means fixed), which is evm-intervene's
method, is reported for every pair beside ours.

**G2. Rank correlation.** Spearman ρ(L_obs, E_int contact) across the 45 main runs at their
best checkpoints, with a 95% interval from 5,000 bootstrap resamples of runs. It is computed for
both std-MSE and corner RMSE; **std-MSE is gated**. The reading uses the **point estimate**, as in
evm-intervene (`gate_g2_obs_rank_decoupled`):

- point ρ < 0.50: observational loss fails to rank intervention error (decoupled);
- point ρ ≥ 0.80: observational loss ranks it well, which refutes the position for this task, and
  is reported as such;
- in between: inconclusive.

The interval is reported with every reading. A stricter flag, `decoupled_by_interval` (upper end
of the interval < 0.50), is reported beside it and is not gated. With 45 runs the interval is
wide, so a decoupled reading whose interval reaches 0.50 is stated as such in the write-up.

## Secondary analyses (not gated)

- E_int on the no-contact stratum (every model should be near `copy`; a check, not a finding).
- E_int and the corner error against the policy log-density of the action, in deciles.
- **10-step rollouts**, free-running, under constant interventions, as evm-intervene's 12-step
  rollouts. Validation for this analysis uses 10-step rollouts with logged actions, so the two
  are compared in the same mode.
- G1 repeated at the small and large widths, each with its own V* computed by the same rule
  from that width's runs.
- **Closed-loop**: push-to-goal with each model inside a sampling planner (MPC), success rate
  against L_obs and E_int. Does E_int predict closed-loop success better than L_obs?

## Seal (follows forward's `scripts/freeze.py`)

1. **Freeze 1, before any training.** Write `frozen/design.json`: freeze time, this file's sha256,
   the sha256 of every file that generates data, trains or scores, the MuJoCo, Python and torch
   versions, the image digest, and the train, validation and pilot seeds. Commit and push, and
   post the commit hash and this file's sha256 on forward#5. That comment's timestamp is
   independent evidence.
2. **Pilot**, on pilot seeds only. It may fix only the items marked *set by pilot*. Pilot data,
   models and numbers never enter a result.
3. **Freeze 2, after training, before test generation.** Write `frozen/models.json`: the final
   settings of the *set by pilot* items, the sha256 of every checkpoint used, and V*. Commit, push
   and post as above. The test seed is derived from this file's sha256.
4. **Generate the held-out states, run G0 to G2, publish.** The scorer refuses to run if any
   recorded hash differs.

## Results format

One JSON, `results/intervention_results.json`, in the shared format drafted in
[INTERVENTION_REPORTING.md](INTERVENTION_REPORTING.md). That format is evm-intervene's
`results/mainnet_intervention_results.json` written down: top-level run metadata, then one block
per analysis (here only `primary`). Keys use this task's own names where the stratum or unit
differs:

| evm-intervene key | here |
| --- | --- |
| `regime_1_smooth` | `no_contact` |
| `regime_2_cliff`, `regime_3_cascade` and their union block `cliff_and_cascade` (gated) | `contact` (gated) |
| `*_cliff_*` (e.g. `matched_cliff_gap`, `best_cliff_std_mse`) | `*_contact_*` |
| `price_rmse_bps`, `*_price_bps` | `corner_rmse_mm`, `*_corner_mm` |
| `liquidity_rmse` | `angle_rmse_deg` |
| `n_held_out_checkpoints` (bootstrap clusters) | same key; counts held-out simulator states |
| `oot_obs_*` (out-of-time) | not applicable, omitted |
| `passes_5pct_match_with_conservation` | omitted; each pair's `matched_within_5pct` covers it |

E_int for G1 is the seed mean of `seed_checkpoints[*].matched_contact_std_mse`, as in
evm-intervene. The `std_mse` in each model's stratum blocks is the error of the seed-averaged
prediction, reported and not gated.

Descriptive intervals, as in evm-intervene: `std_mse_ci95` resamples held-out states, on the
seed-averaged prediction; `matched_obs_val_ci95` resamples validation episodes, on the
seed-averaged L_obs. Both use 2,000 replicates, as in evm-intervene; neither is gated. Every
gated interval and p-value uses 5,000.

Added keys, absent in evm-intervene's output: the shared format's optional keys
(`format_version`, `strata`, `units`, `resampled`, `gate_g0_valid`, `gate_g1_reading`,
`gate_g2_reading`), and `matched_contact_gap_states_only` (evm's interval, no seed resampling),
`decoupled_by_interval`, `updates` (evm's `epochs`), `width`, `controls` (the two control
models, laid out as `models`) and `logged_action` (per model, the logged action scored on the
test states). Values below are placeholders.

```json
{
  "format_version": "0.1",
  "benchmark": "physical-intervene/pushing",
  "device": "cpu",
  "seeds": [11, 22, 33, 44, 55],
  "updates": 0,
  "wall_seconds": 0.0,
  "primary": {
    "arm_name": "primary",
    "strata": {"hard": ["contact"], "easy": ["no_contact"], "hard_block": "contact"},
    "units": {"physical_unit": "corner_rmse_mm"},
    "counts": {
      "n_train_obs": 100000, "n_val_obs": 10000,
      "n_interventions_total": 24000, "n_held_out_checkpoints": 1000,
      "regime_counts": {"no_contact": 15000, "contact": 9000}
    },
    "target_scaler_std": [0.0, 0.0, 0.0],
    "copy_obs_val_std_mse": 1.0,
    "baselines": {
      "copy": {
        "obs_val_std_mse": 1.0, "obs_val_corner_rmse_mm": 0.0,
        "interventions": {
          "all":        {"n": 24000, "std_mse": 0.0, "std_mse_ci95": [0.0, 0.0], "corner_rmse_mm": 0.0, "angle_rmse_deg": 0.0},
          "no_contact": {"n": 15000, "std_mse": 0.0, "std_mse_ci95": [0.0, 0.0], "corner_rmse_mm": 0.0, "angle_rmse_deg": 0.0},
          "contact":    {"n": 9000,  "std_mse": 0.0, "std_mse_ci95": [0.0, 0.0], "corner_rmse_mm": 0.0, "angle_rmse_deg": 0.0}
        }
      },
      "constvel": {}, "knn": {}
    },
    "controls": {"blind": {}, "mlp-random": {}},
    "models": {
      "gru": {
        "n_seeds": 5,
        "matched_obs_val_std_mse_mean": 0.31, "matched_obs_val_std_mse_sd": 0.01,
        "matched_obs_val_ci95": [0.29, 0.33], "matched_obs_val_corner_rmse_mm": 0.0,
        "best_obs_val_std_mse_mean": 0.28, "best_obs_val_std_mse_sd": 0.01,
        "beats_copy_on_obs": true,
        "logged_action": {"n": 1000, "std_mse": 0.0, "corner_rmse_mm": 0.0},
        "matched_checkpoint_interventions": {"all": {}, "no_contact": {}, "contact": {}},
        "best_checkpoint_interventions": {"all": {}, "no_contact": {}, "contact": {}}
      },
      "mlp": {}, "rssm": {}
    },
    "pairwise_gate_g1": [
      {"challenger": "gru", "reference": "mlp", "matched_within_5pct": true, "obs_val_rel_diff": 0.02,
       "matched_contact_gap": {"gap": 0.9, "ci_low": 0.3, "ci_high": 1.6, "rel_gap": 0.6,
                               "p_two_sided": 0.004, "holm_p": 0.012, "resampled": ["state", "seed"]},
       "matched_contact_gap_states_only": {"gap": 0.9, "ci_low": 0.4, "ci_high": 1.5, "rel_gap": 0.6,
                                           "p_two_sided": 0.002, "resampled": ["state"]},
       "best_contact_gap": {"gap": 0.0, "ci_low": 0.0, "ci_high": 0.0, "rel_gap": 0.0, "p_two_sided": 1.0}}
    ],
    "rank_correlation_gate_g2": {
      "n_checkpoints": 45,
      "spearman_obs_vs_contact_std_mse": {"rho": 0.2, "p_value": 0.2, "ci_low": -0.1, "ci_high": 0.5},
      "spearman_obs_vs_contact_corner_mm": {"rho": 0.2, "p_value": 0.2, "ci_low": -0.1, "ci_high": 0.5},
      "decoupled_by_interval": false
    },
    "gates": {"gate_g0_valid": true, "gate_g1_matched_separation_pass": true, "gate_g2_obs_rank_decoupled": true,
              "gate_g1_reading": "pass", "gate_g2_reading": "decoupled"},
    "seed_checkpoints": [
      {"family": "gru", "width": "medium", "seed": 11,
       "best_obs_val_loss": 0.28, "matched_obs_val_loss": 0.31,
       "best_contact_std_mse": 0.0, "matched_contact_std_mse": 0.0,
       "best_contact_corner_rmse_mm": 0.0, "matched_contact_corner_rmse_mm": 0.0}
    ]
  }
}
```

## Compute

Data generation is a few CPU-minutes. One-step training on 100,000 transitions takes minutes per
run for `mlp` and tens of minutes for `gru` and `rssm` on one CPU core. 45 main runs plus 10
control runs fit in about a day of CPU, or one free Colab or Kaggle GPU session. The pilot
confirms this. If it does not, the large width is dropped, a fallback stated now.

## Limits, stated now

- One simulated task, one box, fixed physics. A positive result shows the effect exists, not that
  it is large in practice.
- Full observability makes `mlp` a correct model class. If the main models do not separate, that
  is the most likely reason. A partially observed variant (no velocities, or friction drawn per
  episode) is the natural follow-up, not part of this test.
- One-step interventions are a single control step. Longer interventions appear only in the
  secondary rollouts.
- MuJoCo's soft-contact model is the ground truth. The test is about agreeing with this
  simulator, not with real boxes.

## Open questions for the maintainers and @Bangar-Lovejesh

1. Should evm-intervene adopt seed resampling in G1 too, so the two G1s mean the same thing?
   Until then, its `matched_cliff_gap` compares with `matched_contact_gap_states_only` here.
2. evm-intervene's `docs/DESIGN.md` and its code differ in two places: G2 is described as
   including intermediate matched-loss checkpoints, but the code uses the 20 best checkpoints;
   and G2's reading is not stated as point estimate or interval, while the code uses the point
   estimate. This design follows the code. Should evm-intervene's DESIGN.md be updated to match?
3. Is the relative-gap convention here (reference = the lower-E_int member of each pair) the
   right generalisation of evm's anchor, or should both repos name a fixed reference?
