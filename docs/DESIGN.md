# physical-intervene, task 1: planar pushing

**Status: draft for the smoke test (Phase 0), not frozen (physicalreasoning/forward#5).** A smoke test (Phase 0) checks
the pipeline first. The design is then hashed before any model is trained on the training split
(freeze 1) and again before any test state exists (freeze 2). After freeze 1 the text below is
not edited: values it computes by fixed rules (the *set by pilot* items, V*, p) are filled in,
and any other change is a dated amendment, reported beside the original text, never in place of
it (see *Deviations after freeze 1*). Every result is published,
including a failure to separate.

## Summary

One page. It states no rule of its own: where it and *Details* differ, *Details* govern.

**Question.** Two world models with the same observational validation loss: do they predict the
same outcome when an action they were not shown makes a pusher push a box? This tests the
hypothesis in *Prediction is not understanding* for a physical task, as the physical counterpart
of evm-intervene.

| | |
| --- | --- |
| **Environment** | MuJoCo planar pushing: a 10 cm box and a cylindrical pusher driven by commanded velocity, 0.1 s control steps. The full simulator state is saved and restored bit-exactly in one pinned Docker image, so the true outcome of any action from any state costs microseconds. |
| **Data** | A scripted policy pushes the box toward a hidden goal, with noise, and takes a uniform random action 20% of the time (less near the box if Phase 0's coverage check requires). 1,000 training and 100 validation episodes; every model trains on the same data. |
| **Interventions** | 1,000 held-out states where the pusher is within 10 mm of the box. From each, 24 actions (12 aimed at the box, 12 uniform), each labelled by the simulator as *no contact*, *graze* or *push*. Only *push* is gated. |
| **Models** | `mlp` and `gru` (architectures), `rssm` (different objective: ELBO), `contact_wm` (different structure: pushing physics written in, numbers learned). Three widths (10k, 100k, 1M parameters); 10 seeds at medium width, which G1 uses. Baselines: `copy`, `constvel`, `knn`. |
| **Controls** | Positive: `mlp` vs `mlp-shortcut`, trained to rely on the action less, so the gap is present by construction; how much less (p) is computed after the main runs from validation loss and the simulator, so it can reach V*. Negative: each architecture vs its twin on new seeds. Sanity: `blind`, which never sees the action. All run through G1 on the test states. |
| **Metrics** | L_obs: one-step std-MSE of the box's (Δx, Δy, Δθ) on validation transitions, the same for every model. E_int: the same error on push interventions. |
| **Matching** | Every run trains its full budget, with no early stopping; G1 uses each run's checkpoint closest to V* = 1.1 × the largest architecture median of best L_obs, and the gap is also reported at other loss levels and at best checkpoints. Two architectures are matched if their L_obs differ by at most 5%. |

**Gates**, fixed before any model is trained:

- **G0, validity.** All 1,000 states replay byte-identically; every main model beats `copy` on
  L_obs; `blind` is worse than `mlp` on push interventions.
- **G1, matched separation (primary).** *Pass* if a matched pair differs in E_int with
  Holm-adjusted p < 0.05, a 95% interval (resampling states and seeds) excluding zero, and a
  relative gap of at least 20%, and neither architecture's twin pair separates. A *fail* counts as
  evidence of no difference only if the positive control separated. Other readings:
  *inconclusive* (not shown specific, or not shown sensitive) and *not testable* (no pair matched).
- **G2, rank correlation.** Spearman ρ between L_obs and E_int over all main runs: below 0.50,
  observational loss does not rank intervention error; 0.80 or above refutes the position for
  this task.

**Order.** Phase 0 checks the pipeline with stand-in models only and never scores a trained
model on interventions. Freeze 1 hashes this file and the code before any model is trained on
the training split. The pilot sets
only the items marked *set by pilot*, using validation loss. Freeze 2 hashes every checkpoint, and
the test states are generated from a seed derived from that hash, so they cannot exist earlier.
Every result is published, including a failure to separate.

**Limits.** One simulated task with full observability; MuJoCo's soft contact is the ground
truth, not real boxes.

# Details

## Question

Do learned world models with matched observational validation loss predict the same outcome when
an action they were not shown makes the pusher touch the box? Or does low observational error hide
large errors under intervention?

This is the physical counterpart of
[evm-intervene](https://github.com/physicalreasoning/evm-intervene), the market version on
Uniswap V3, and uses the same quantities and gates where possible.

## Background (not frozen)

These files explain or compare; they do not define the test and are not hashed in freeze 1 or 2.
Where they and this file differ, this file governs.

- [Why the design is as it is](background/RATIONALE.md): why planar pushing, why grazes are not
  gated, why `contact_wm`, why the positive control's flaw is known and how p is computed, why 10
  seeds, and why G1 uses one matched level with full training.
- [Comparison with evm-intervene](background/EVM_COMPARISON.md): shared terms, deliberate
  differences, result-key mapping.
- [Results file format](background/RESULTS_FORMAT.md): layout of
  `results/intervention_results.json`, with a placeholder example.
- [Shared reporting format, draft](INTERVENTION_REPORTING.md).
- [Phase 0 smoke-test plan](SMOKE_TEST.md); only its record is hashed (see *Seal*).

**Where the numbers live.** This file states every rule and every number that is an analysis
choice. Settings of the simulated world and of each model's implementation (masses, friction,
actuator gain, solver options, layer sizes) are fixed in hashed files instead: the MuJoCo scene
file, `src/physical_intervene/constants.py` and the model configurations. All are hashed in
freeze 1 (see *Seal*). Where a number appears both here and there, the two must agree.

## Environment

| Item | Value |
| --- | --- |
| Simulator | MuJoCo, exact version pinned in `uv.lock` and recorded in the freeze |
| Platform | linux/amd64, one Docker image pinned by digest, single thread; all ground truth is generated in that image |
| Scene | table plane; box 10 x 10 cm (slide x, slide y, hinge z joints; fixed mass). **Table resistance** is `frictionloss` on the box's three joints: the planar joints carry no gravity load, so contact friction with the table does not act. It applies per axis, so a diagonal slide meets √2 times the resistance of an axis-aligned one; this is stated, not corrected; cylindrical pusher, radius 1 cm, mass 0.1 kg, on slide x/y joints driven by velocity actuators with gain kv = 10 N·s/m (a 10 ms response time); all other physical values in the scene file |
| Timing | physics step 2 ms; control step 0.1 s (50 physics steps); episodes of 100 control steps |
| Action a_t | commanded pusher velocity (vx, vy), each in [-0.15, 0.15] m/s |
| State s_t | box (x, y, cos θ, sin θ, vx, vy, ω) and pusher (x, y, vx, vy): 11 numbers, fully observed |
| Save / restore | `mj_getState` / `mj_setState` with `mjSTATE_INTEGRATION` (includes the solver warm-start; restoring only qpos and qvel is not bit-exact) |
| Contact label | a transition is *contact* if any pusher-box contact exists at any of its 50 physics steps, in the simulator's own outcome |
| Push label | a contact transition is a *push* if, in the simulator's own outcome, the box centre moves at least d_push or the box turns at least θ_push over the step; otherwise it is a *graze*. d_push and θ_push are fixed in Phase 0 (see *Interventions*) |

## Target and loss

Each model predicts the change in the full state. **Training loss**, the same for every model
except `rssm`: 0.5 × the std-MSE over the box's 7 state numbers + 0.5 × the std-MSE over the
pusher's 4, each standardised on the training split. `rssm` trains on the ELBO over sequences
instead (see *Models*). It is the model with a different objective, and `contact_wm` the model
with a different structure, the two kinds of difference the issue asks for. Scoring uses only the
box target, three numbers:

y_t = (Δx, Δy, Δθ) of the box over one control step, with Δθ wrapped to (−π, π].

**std-MSE**, as in evm-intervene (`TargetScaler`): the mean over the 3 dimensions of the squared
error, each divided by that dimension's standard deviation on the **training split only**. The
floors on the standard deviation are 1e-4 m, 1e-4 m and 1e-3 rad.

**L_obs**, the observational validation loss: the one-step std-MSE on y_t over the validation
transitions, computed from each model's prediction. It is the same for every model whatever its
training objective, and it is what matching uses (*Matching observational loss*).

**Physical unit**: the RMS, over transitions, of the mean distance between the predicted and
true positions of the box's 4 corners after the step, in mm. It plays the role of bps in
evm-intervene and is reported, not gated.

## Data

**Behaviour policy (observational).** Each episode draws a goal pose for the box. The policy
moves the pusher around the box to the side opposite the goal, then pushes toward the goal at
0.1 m/s, with Gaussian noise of 0.02 m/s per axis. At each control step, with probability
ε, it instead takes an action drawn uniformly from the action box. ε = 0.2, except within one
step's reach of the box (gap at most 15 mm, the farthest the pusher travels in one control step),
where it is ε_near. ε_near is 0.2 unless Phase 0's intervention-coverage check fails, in which
case it is lowered by the fixed ladder in [SMOKE_TEST.md](SMOKE_TEST.md); its final value is
recorded in freeze 1. ε_near is never 0. So every single action is inside the training support,
but most of the training contacts come from the scripted push. The goal is not observed. Box start pose, pusher start position (at least 5 cm between the pusher's surface and the box's surface) and goal
are drawn uniformly from fixed regions.

**Splits.** Each split has its own seed range and is never reused.

| Split | Content | Seed |
| --- | --- | --- |
| train | 1,000 episodes (100,000 transitions) | fixed, in freeze 1 |
| validation | 100 episodes (10,000 transitions); L_obs | fixed, in freeze 1 |
| smoke | 50 episodes, 200 test-like states and a coverage pool of 1,000 episodes, for Phase 0 only | fixed before Phase 0, recorded in freeze 1; smoke outputs never enter a result |
| pilot | 200 episodes and 200 test-like states | fixed, in freeze 1; pilot outputs never enter a result |
| test | 1,000 held-out states (see *Interventions*) | derived from the sha256 of `frozen/models.json`, so it cannot exist before every model's weights are hashed |

## Interventions

**Held-out states.** Each test state comes from a fresh behaviour-policy episode. It is the first
step at or after a uniformly drawn step in [10, 90] where the gap, the shortest distance from the
pusher's surface to the box's surface, is at most 10 mm. In one control step the pusher travels at
most 0.15 m/s × 0.1 s = 15 mm, and aimed interventions travel 5 to 15 mm, so from a gap of at most
10 mm both contact and no-contact actions are possible. A larger cap would add states from which
no action can reach the box. Phase 0 checks the resulting push rate.

**Interventions.** From each held-out state, the simulator is restored and run for one control
step under each of K = 24 actions:

- 12 aimed: speed U[0.05, 0.15] m/s, direction toward the box centre ± U[0, 45°];
- 12 uniform: uniform on the action box.

Each action is labelled by the simulator, as above, into one of three strata:

| Stratum | Meaning | Role |
| --- | --- | --- |
| `no_contact` | the pusher does not touch the box | reported, not gated |
| `graze` | contact, but the box moves less than d_push and turns less than θ_push | reported, not gated |
| `push` | contact that moves the box at least d_push or turns it at least θ_push | **gated** |

**Push is the primary stratum**, the counterpart of evm-intervene's regimes II and III (a trade
that crosses a cliff, not one that only reaches it). Grazes are reported, not gated. A held-out state
with no push action is dropped from the push stratum for every model.

**Push thresholds.** d_push and θ_push are fixed in Phase 0, before freeze 1, from the simulator
alone: for every contact intervention on the smoke states, Phase 0 records the box centre's
displacement and the absolute rotation over the step, and plots both. The defaults are
d_push = 2 mm and θ_push = 1°. They are kept unless the plot shows they cut through the bulk of
real pushes rather than between grazes and pushes, in which case they are moved to that gap. The
final values, the plots and the share of contact interventions labelled graze are recorded in
freeze 1. The thresholds use only the simulator's true outcome, never a model's prediction.

The logged action from the same state is also scored, as the observational reference on the
same states, and reported under each model's `logged_action` key. **It is not an intervention and never enters E_int**,
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
| `rssm` | RSSM (deterministic GRU state plus Gaussian latent). **Different objective:** trained on sequences with the ELBO instead of the shared one-step loss (*Target and loss*); scored on the same L_obs as every model. Structure and objective follow PlaNet (Hafner et al. 2019): sequences of 50, 3 free nats, KL weight 1, and at medium width a 200-unit GRU with a 30-dimensional latent; prediction is the mean of 16 samples, drawn with a sampling seed fixed per run so that scoring the same checkpoint twice gives the same prediction | main, different objective |
| `contact_wm` | structured: hand-written contact geometry and a quasi-static pushing law, with a small trained correction network; every number learned from the training data (see below) | main, different structure |
| `mlp-shortcut` | as `mlp`, but during training each transition's action input is replaced, with probability p, by that transition's *shadow action* (see *Controls*); at test time it gets the real a_t like every model | positive control |
| `mlp-twin`, `gru-twin`, `rssm-twin`, `contact_wm-twin` | the same architecture, settings and width as `mlp`, `gru`, `rssm`, `contact_wm`, trained on 10 seeds of their own, 121 to 220 in steps of 11 | negative controls |
| `blind` | as `mlp` with a_t removed | sanity control |
| `copy` | box does not move | baseline (evm's `copy`) |
| `constvel` | box keeps its current velocity | baseline |
| `knn` | mean Δs of the 5 nearest training transitions in standardised (s_t, a_t) | baseline |

Main models: 4 architectures at 3 widths, 80 runs: 10 seeds at medium width (11 to 110 in steps
of 11), which G1 uses, and 5 seeds (11 to 55) at the small and large widths, which only secondary
analyses and G2 use. The first five seeds are evm-intervene's (why 10: [RATIONALE.md](background/RATIONALE.md),
*Why 10 seeds*). **Widths** are equal parameter counts
across architectures: about 10k, 100k and 1M parameters (small, medium, large); for `contact_wm`
the count is that of its correction network.

**Training: full budget, no early stopping.** Every run trains for its architecture's full number
of updates, at a **constant learning rate** (no decay schedule), so that any saved checkpoint is a
fair "stopped here" model. Checkpoints are saved every 100 updates for the first 2,000 updates,
where L_obs falls fastest, and every 1,000 updates after that, so no run steps past V* by more
than the match screen allows. The number of updates is *set by pilot*: the point where the
pilot's L_obs has improved by no more than 1% over its last 10 checkpoints at 1,000-update
spacing (the *plateau criterion*), plus 50%. Each final run reports whether it met the plateau
criterion by its last checkpoint. Every run therefore reaches its own floor and continues past
its best checkpoint, which matching at V* and the curves in *Secondary analyses* need. Why:
[RATIONALE.md](background/RATIONALE.md), *Why one matched level, with full training*.
Controls: medium width, 10 seeds each (60 runs); `blind` and `mlp-shortcut` use the main seeds, the twins use their own. Optimiser,
learning rate, batch size and the number of updates are *set by pilot* on pilot and
validation data only. Each architecture gets its own settings, chosen by the same pilot procedure and budget,
and shared across its seeds and widths. **Every pilot choice minimises L_obs on validation data.**
Intervention error on the pilot states may be computed to check the pipeline, but it never
informs any choice, and the pilot report says so. The one exception is below, and concerns
controls only.

Every model, main or control, is trained on the same behaviour-policy data (*Data*). Differences
between models therefore come from the models, never from the data. The one addition is the
shadow action used by `mlp-shortcut` (below): an extra column on the same transitions, drawn from
its own random stream, so the training data are byte-identical with or without it.

### The structured model, `contact_wm`

- **Structure**, hand-written and differentiable:
  1. *Pusher path.* The pusher follows the commanded velocity with a learned first-order lag, over
     10 sub-steps of the control step.
  2. *Contact.* From the box pose in s_t and the pusher path, a smooth contact indicator, the
     contact point and the contact normal at each sub-step, using a learned box half-width, a
     learned pusher radius and a learned contact sharpness.
  3. *Pushing law.* In contact, the box's planar velocity follows the quasi-static pusher-slider
     model with an ellipsoidal limit surface (learned limit-surface constant) and a smoothed
     stick or slip choice set by a learned pusher-box friction coefficient. Out of contact, the
     box keeps its current velocity and decelerates under a learned table-friction term.
  4. *Correction network.* A small MLP on geometric features only (contact point, normal,
     relative velocity, box velocity) adds a residual to the box's motion. It does not see the
     history.
- **Inputs.** The same as every model; `contact_wm` uses s_t and a_t and ignores the history by
  design, which the equal-observability rule allows.
- **Nothing copied from the simulator.** No MuJoCo value (box size, pusher radius, friction,
  solver or contact settings) is written in. Every number above is learned, starting from fixed
  generic initial values recorded in freeze 1 (box half-width 4 cm, pusher radius 1.5 cm,
  friction coefficients 0.5, limit-surface constant from a uniform square of the initial size).
  Only the form of the law is built in.
- **Training** is as for every model: gradient descent on std-MSE (box target, plus the pusher
  target from its lag model), the same training budget rule and checkpoint schedule, settings
  *set by pilot* by L_obs.
- **Matching.** Its rules approximate MuJoCo's soft contact, so its best L_obs may be higher than
  the others'. V* is 1.1 × the largest architecture median, so every architecture can reach it,
  but a high `contact_wm` median raises V* and puts the other models at earlier checkpoints. So
  G1 among the three general-purpose models, with V* from their medians only, is also reported
  (*Secondary analyses*).
- **Passing pairs.** The result lists which main pairs pass (`gate_g1_passing_pairs`).

## Controls

A control is a pair of models whose answer is known before testing, run through exactly the G1
procedure on the same test states.

**Positive control, `mlp` vs `mlp-shortcut`.** The gap G1 looks for is a model that predicts
ordinary transitions as well as another but relies on the action less. `mlp-shortcut` has that
flaw built in, at a known size.

- *Shadow action.* At data generation, every training and validation transition gets a shadow
  action a'_t, a fresh draw from the behaviour policy at the same state with the episode's goal
  (uniform with that state's ε, scripted push plus noise otherwise; see *Data*), from a separate
  random stream. During training, with probability p, `mlp-shortcut` is given a'_t instead of a_t
  as its action input; the target is unchanged.
- *What the flaw is.* No input reveals whether the action was replaced, so the best predictor
  the model can learn is (1 − p) · f(s, h, a) + p · m(s, h), where f is the true outcome and m its
  average over the behaviour policy's actions; its extra error is p² · (f − m)² (derivation in
  [RATIONALE.md](background/RATIONALE.md)).
- *Feasibility, from the simulator alone.* Phase 0 (on smoke states) and the pilot (on pilot
  states) estimate D_obs, the mean of (f − m)² over behaviour transitions, and D_int, its mean over
  push interventions, with m from 100 policy draws per state. (f − m)² is measured in std-MSE, the
  unit of L_obs. No model is involved. Matching
  limits the flaw to roughly p² · D_obs ≤ V* − L_best(`mlp`), and the expected relative gap is then
  about p² · D_int / E_int(`mlp`). If D_int / D_obs is too small for a gap of 20% at matched loss,
  the positive control cannot pass whatever G1's sensitivity, and this is found before freeze 1.
- *p is computed after the main runs, before freeze 2*, by a fixed rule, so that `mlp-shortcut`
  can reach the final V*:
  1. Train the medium-width main runs and compute V* (*Matching observational loss*).
  2. Compute D_obs on the validation transitions with the simulator alone, as in *Feasibility*.
  3. Let L_mlp be the median over `mlp`'s seeds of its best L_obs. p is the largest value in
     {0.1, 0.2, ..., 0.9} with L_mlp + p² · D_obs ≤ V*.
  4. Train the 10 `mlp-shortcut` seeds at that p. Each uses its checkpoint closest to V* and goes
     through the match screen like any run; a miss is reported, not corrected.

  The rule's inputs are validation loss and the simulator only; no intervention error and no test
  state exists at this point. If no p in the grid satisfies step 3, the positive control is
  *not run*, and a G1 *fail* reads *inconclusive: test not shown sensitive*. p, D_obs and L_mlp are
  recorded in freeze 2. Why this rule and not a pilot value:
  [RATIONALE.md](background/RATIONALE.md), *Why p is computed after the main runs*.
- *Pilot confirmation.* The pilot computes E_int on pilot states for controls only, comparing
  `mlp-shortcut` with `mlp-twin`, which has `mlp`'s architecture but is not a main model. It
  reports the observed relative gap beside the predicted one. This reveals one absolute E_int of
  an `mlp`-like model, and nothing about how `mlp`, `gru`, `rssm` and `contact_wm` compare, which is what G1
  tests; the pilot report says so. No main model's intervention error is computed in the pilot.
  The pilot applies the rule above to its own runs to get a pilot p for this comparison; the
  final p is computed afresh from the main runs and never uses the pilot's result. If the pilot
  pair does not separate, the rule stays as written, the result is reported, and the formal
  positive control still runs; there is no tuning on intervention error.

**Negative controls, X vs X-twin for X in `mlp`, `gru`, `rssm`, `contact_wm`.** Each twin is its architecture
retrained on 10 new seeds, so the pair has no real difference: any gap is training noise. G1
should not separate them. The main risk G1 guards against is two seeds of one architecture
differing by chance and that passing as an architecture difference; the twins test that
safeguard on trained models, for every architecture, since `gru`, `rssm` and `contact_wm` may vary more
from seed to seed than `mlp`.

- The twins are kept out of everything else: V* is computed from the main runs only, and the
  twins are not in G2, in the main exclusions or in the Holm correction.
- Each twin is held to the same rules as a main architecture: matched checkpoint closest to the
  same V*, its own exclusions and its own 6-of-10-seeds rule. A twin dropped by that rule leaves
  its architecture's negative control *not run*, reported.
- The states-only interval (evm-intervene's method) is also computed on each twin pair and
  reported. If it separates a twin pair that the full procedure does not, that shows directly why
  this design resamples seeds.

**Sanity control, `blind`.** A model that never sees the action is known to do worse under
intervention, so it does contain the gap, but at a much higher L_obs: it cannot pass the match
screen, and the gap is far larger than the 20% G1 must detect. It cannot show what G1 can detect
at matched loss. It shows only that the pipeline notices a model ignoring the action at all
(G0.3); if it fails, something basic (scoring, contact labels, data) is broken.

A single run of each control is one draw. Phase 0 repeats hand-built stand-in pairs many times to
measure G1's false-alarm and detection rates ([SMOKE_TEST.md](SMOKE_TEST.md)); the controls here
show the same on real trained models and the real test states.

## Matching observational loss

Fixed now, before training. All steps use validation data only. No test state exists yet.

1. **Target.** For each medium-width main run, take its best L_obs over checkpoints, and the
   median over the 10 seeds per architecture. V* = 1.1 × the largest of the four medians. Taking
   the largest median, plus a 10% margin, makes V* a level every architecture can reach.
2. **Matched checkpoint.** Each medium run's matched checkpoint is the one whose L_obs is closest
   to V*.
3. **Match screen.** A pair of architectures is matched if their seed-mean matched L_obs differ by
   at most 5%: |L_a − L_b| / min(L_a, L_b) ≤ 0.05. Since every run aims at the same V*, most pairs
   should pass; the screen catches runs that could not get close.
4. **Exclusions.** A run whose L_obs never comes within 5% of V* is excluded, and every exclusion
   is reported. An architecture's matched L_obs, E_int and seed resampling then use its remaining
   seeds. An architecture with fewer than 6 of its 10 seeds left (3 of 5 at the small and large widths)
   is dropped from G1 and reported;
   if that leaves fewer than two architectures, G1 is *not testable*.

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
   mean). Controls are not main models.
3. **Sanity control.** On the push stratum, at best checkpoints, `blind` has a higher E_int
   than `mlp`, with a 95% interval excluding zero (bootstrap over states and seeds, as in G1).
   A model that never sees the action cannot answer "what if"; if the test cannot see that, it
   cannot see anything. This is not a positive control for G1 (see *Controls*).

**Control pairs (read with G1).** Each control pair goes through the full G1 procedure on the
test states: matched checkpoints at the same V*, the match screen, the same bootstrap over states
and seeds, and the same thresholds (p < 0.05, interval excluding zero, relative gap ≥ 20%). Each
p is unadjusted, since each is a separate pre-specified check, and none enters the Holm
correction of the main pairs. A control pair *separates* if it passes the match screen and meets
all three thresholds.

- **Positive control (sensitivity).** It passes if `mlp` vs `mlp-shortcut` separates, with
  `mlp-shortcut` the higher. If it passes, a G1 *fail* on the main models is read as evidence that
  they do not differ. If it does not pass (no match, or no separation), a G1 *fail* is reported as
  *inconclusive: test not shown sensitive*.
- **Negative controls (specificity).** Each of `mlp` vs `mlp-twin`, `gru` vs `gru-twin`,
  `rssm` vs `rssm-twin` and `contact_wm` vs `contact_wm-twin` should not separate. A main pair counts towards a G1 *pass* only if
  neither of its architectures' twin pairs separates; a main pair that meets the pass conditions
  while one of its twin pairs separated is reported as *inconclusive: test not shown specific*.
  A twin pair that is *not run* (too few twin seeds left) leaves its main pairs unaffected, and
  this is stated beside the verdict. Even a correct test separates a twin pair occasionally by
  chance (Phase 0 estimates how often); the cost is accepted, so that a pass is never read when
  the test has shown it finds gaps that are not there.

**G1. Matched-loss separation (primary).**

*Question:* can two architectures with the same observational validation loss differ in how
well they predict the outcome of an intervention that pushes the box?

Steps, for each of the 6 pairs of main architectures (mlp/gru, mlp/rssm, gru/rssm, and each of
the three with contact_wm), at matched
checkpoints (see *Matching observational loss*):

1. **Intervention error.** E_int is the std-MSE on the push stratum. Each push transition's
   std-MSE is averaged over the architecture's seeds, then over all push transitions, pooled
   across states.
2. **Gap.** The pair member with the lower E_int is the *reference*, the other the *challenger*.
   Δ_int = E_int(challenger) − E_int(reference), and the relative gap is Δ_int / E_int(reference).
3. **Interval.** 5,000 bootstrap replicates. Each replicate resamples the held-out states with
   replacement, the same draw for both models (paired), and resamples each architecture's seeds
   with replacement, independently; then recomputes Δ_int. The 95% interval is the 2.5th to
   97.5th percentile of these replicates.
4. **p-value.** p = min(1, 2 × (min(#Δ* ≤ 0, #Δ* ≥ 0) + 1) / (5,000 + 1)), where Δ* are the
   replicates. The 6 p-values are Holm-corrected together, including pairs that fail the match
   screen.

*Verdict:*

- **Pass:** at least one pair that passes the match screen has a Holm-adjusted p < 0.05, a 95%
  interval excluding zero and a relative gap ≥ 20%, and neither of its twin pairs separated (see
  *Control pairs*).
- **Inconclusive, not shown specific:** some pair meets those three conditions, but for every
  such pair at least one of its twin pairs separated.
- **Fail:** at least one pair passes the match screen, but none meets all three conditions. This
  is a negative result about our own position, published the same way, provided the positive
  control passed; otherwise it is *inconclusive: test not shown sensitive*.
- **Not testable:** no pair passes the match screen. Reported with the observed L_obs
  differences.

`gate_g1_reading` takes the values `pass`, `inconclusive_not_specific`, `fail`,
`inconclusive_not_sensitive` and `not_testable`.

*Also reported, not gated:*

- `matched_push_gap_states_only`: the same test resampling states only, with seed means
  fixed. This is evm-intervene's method and the issue's original interval.
- `best_push_gap`: the same test at best checkpoints, as evm-intervene reports.
- `matched_only`: true for a pair whose G1 result passes at V* but whose `best_push_gap`
  interval includes zero. It does not change `gate_g1_reading`: the issue asks whether models with
  matched validation loss differ, and they do. It says the difference is not seen once both are
  fully trained, so it is likely a difference in learning speed, and the write-up says so.

*Note:* a state with more push actions weighs more in E_int. This is how the design reads the
issue's "averaged over held-out states": transitions are pooled across states, as in
evm-intervene and the shared reporting format, and the bootstrap resamples whole states.

**G2. Rank correlation.** Spearman ρ(L_obs, E_int push) across the 80 main runs at their
best checkpoints, with a 95% interval from 5,000 bootstrap resamples of runs. It is computed for
both std-MSE and corner RMSE; **std-MSE is gated**. The reading uses the **point estimate**, as in
evm-intervene (`gate_g2_obs_rank_decoupled`):

- point ρ < 0.50: observational loss fails to rank intervention error (decoupled);
- point ρ ≥ 0.80: observational loss ranks it well, which refutes the position for this task, and
  is reported as such;
- in between: inconclusive.

The interval is reported with every reading. A stricter flag, `decoupled_by_interval` (upper end
of the interval < 0.50), is reported beside it and is not gated. With 80 runs the interval is
wide, so a decoupled reading whose interval reaches 0.50 is stated as such in the write-up.

## Secondary analyses (not gated)

- **Per-seed view:** each architecture's 10 per-seed E_int values on the push stratum, and for each
  main pair an exact two-sided Mann-Whitney test on them (10 vs 10; smallest possible p = 2/184,756).
- E_int on the no-contact stratum (every model should be near `copy`; a check, not a finding).
- E_int on the graze stratum, and on all contact interventions (graze and push together), with
  G1 repeated on the latter for comparison.
- E_int and the corner error against the policy log-density of the action, in deciles.
- **10-step rollouts**, free-running, under constant interventions, as evm-intervene's 12-step
  rollouts. Validation for this analysis uses 10-step rollouts with logged actions, so the two
  are compared in the same mode.
- G1 among the three general-purpose models only (3 pairs, Holm over 3), with V* computed by the
  same rule from their medians, so that `contact_wm` does not set their matching level.
- **Gap along training.** Every saved checkpoint of every medium-width main run is scored on the
  push stratum, and each architecture's seed-mean E_int is plotted against L_obs, on raw and log
  axes. For each main pair, the G1 procedure (matched checkpoints, bootstrap, interval) is repeated
  at the shared levels 2 × V*, 1.5 × V*, V* and, if lower, the pair's own level V*_pair, computed
  by the *Matching* rule from the pair's two medians only. Reading, fixed now:
  *overlap* if the interval includes zero at every level (in this range, L_obs determines E_int);
  *persistent* if the interval excludes zero with the same sign at V* and at the pair's lowest
  level; *closing* if it excludes zero at V* but not at the pair's lowest level (a difference in
  learning speed); otherwise *mixed*. Not gated, and not Holm-corrected.
- G1 repeated at the small and large widths, each with its own V* computed by the same rule
  from that width's runs.
- **Closed-loop**: push-to-goal with each model inside a sampling planner (MPC), success rate
  against L_obs and E_int. Does E_int predict closed-loop success better than L_obs?

## Phase 0: smoke test (before freeze 1)

Before freeze 1, a smoke test runs the whole pipeline on its own smoke seeds, with stand-in
models whose answers are known in advance. It checks the machinery (bit-exact replay, contact
and push rates, the push thresholds, shadow actions, scoring, gates, control readings, result format), measures G1's
false-alarm and detection rates on repeated stand-in pairs, and checks from the simulator alone
that the positive control is feasible and that push interventions lie far enough from the
behaviour data (intervention coverage). Models may be trained on smoke data to check the training
side (data, wiring, learning, matching), but **no trained model is ever scored on interventions**:
the intervention side is run with stand-ins only. So Phase 0 cannot preview the result. Smoke data, stand-ins and numbers never enter
a result. The plan is in [SMOKE_TEST.md](SMOKE_TEST.md); the record
(`results/phase0_smoke.json` and a short report) is hashed in freeze 1.

## Seal (follows forward's `scripts/freeze.py`)

0. **Phase 0**, as above and in [SMOKE_TEST.md](SMOKE_TEST.md), before freeze 1.
1. **Freeze 1, before any training on the training split.** Write `frozen/design.json`: freeze time, this file's sha256,
   the sha256 of every file that generates data, trains or scores (including the MuJoCo scene
   file, `constants.py` and the model configurations), the MuJoCo, Python and torch
   versions, the image digest, the sha256 of the Phase 0 record, the smoke, train, validation
   and pilot seeds, the main training seeds (11 to 110), the twin training seeds (121 to 220), the seed of the shadow-action
   stream, d_push and θ_push, and ε_near. The background files are not part of the freeze. Commit and
   push, and
   post the commit hash and this file's sha256 on forward#5. That comment's timestamp is
   independent evidence.
2. **Pilot**, on pilot seeds only. It may fix only the items marked *set by pilot*, and runs the
   positive-control feasibility and confirmation in *Controls*. Pilot data, models and numbers
   never enter a result.
3. **Training.** The main runs and the other controls first; then V*, D_obs and p by the rule in
   *Controls*; then the `mlp-shortcut` runs at that p.
4. **Freeze 2, after training, before test generation.** Write `frozen/models.json`: the final
   settings of the *set by pilot* items, the sha256 of every checkpoint used, V*, and p with the
   D_obs and L_mlp it was computed from. Commit, push and post as above. The test seed is derived
   from this file's sha256.
5. **Generate the held-out states, run G0 to G2, publish.** The scorer refuses to run if any
   recorded hash differs.

### Deviations after freeze 1

A problem found after freeze 1 is fixed by an amendment, never by editing this file.

- Each amendment is a file in `frozen/amendments/`, named by date. It quotes the original text
  and the new text, states what was found and the evidence, the stage at which it was found,
  and exactly which data, models and results existed at that point.
- It is committed, its sha256 recorded, and posted on forward#5, as for the freezes. The scorer
  checks amendment hashes like the others.
- **Before freeze 2:** the amendment applies; if it changes the training data, affected models
  are retrained.
- **After freeze 2, before scoring:** if test states already exist, a new test set is generated
  from a new seed derived from the amendment's sha256, and the discarded set is reported.
- **After any gate has been computed:** the frozen analysis stays primary and is published as
  frozen. The amended analysis is reported beside it as secondary, labelled as made after seeing
  results.

## Results

One JSON, `results/intervention_results.json`, written by the scorer. It reports every quantity
named above, including the gate readings `gate_g0_valid`, `gate_g1_reading`,
`gate_positive_control_pass`, `gate_negative_controls`, `gate_g1_passing_pairs` and
`gate_g2_reading`. Layout and key names: [RESULTS_FORMAT.md](background/RESULTS_FORMAT.md).

## Compute

Data generation is a few CPU-minutes. One-step training on 100,000 transitions takes minutes per
run for `mlp` and tens of minutes for `gru`, `rssm` and `contact_wm` (whose sub-stepped
geometry is slower per update) on one CPU core. 80 main runs plus 60 control runs (10 `blind`, 10
`mlp-shortcut`, 40 twins) fit in three to five days of CPU, or one or two free
Colab or Kaggle GPU sessions. The `mlp-shortcut` runs start only after the medium-width main runs
have finished, since p needs their V*; they are among the fastest runs. D_obs on the validation
transitions takes 10,000 × 100 one-step simulations, a few CPU-minutes. The pilot confirms this. If it does not, the large width is dropped,
a fallback stated now. The twins are not dropped to save compute.

## Limits, stated now

- One simulated task, one box, fixed physics. A positive result shows the effect exists, not that
  it is large in practice.
- `contact_wm`'s rules are an approximation: quasi-static pushing ignores the box's momentum
  during contact, and MuJoCo's contact is soft. Where they disagree, `contact_wm` is wrong in a
  way no amount of data fixes, beyond what its correction network absorbs.
- Full observability makes `mlp` a correct model class. If the main models do not separate, that
  is the most likely reason. A partially observed variant (no velocities, or friction drawn per
  episode) is the natural follow-up, not part of this test.
- One-step interventions are a single control step. Longer interventions appear only in the
  secondary rollouts.
- Grazes are excluded from the gated stratum by a threshold fixed in Phase 0. A model that is
  wrong only on grazes is not penalised in G1; its graze error is reported.
- MuJoCo's soft-contact model is the ground truth. The test is about agreeing with this
  simulator, not with real boxes.
- G1 compares models at one loss level, V*, set by the worst architecture. For the others this is
  an earlier checkpoint, where rare transitions such as pushes are likely still being learned. A
  gap there answers the question asked (matched loss, different intervention error) but may come
  from learning speed; *Gap along training* and `matched_only` report which.
- The positive control builds in one kind of gap (relying on the action less). Detecting it
  shows G1 can detect a gap of that size at matched loss, not every kind of difference.
- Each control is one draw on the test states. A clean negative control is consistent with a low
  false-alarm rate but does not measure it; Phase 0's repeated stand-in pairs do.
