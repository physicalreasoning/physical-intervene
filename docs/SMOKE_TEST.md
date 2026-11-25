# physical-intervene: Phase 0 smoke test

**Status: plan, not run.** Phase 0 runs before freeze 1 of [DESIGN.md](DESIGN.md). It is not
part of the pre-registered analysis and is not frozen as a plan: its findings may change both
this file and the design until freeze 1. Its record (`results/phase0_smoke.json` and the report)
is hashed in freeze 1.

*Purpose:* check that every part of the pipeline works and that the setup makes sense, before
anything is frozen. Phase 0 tests the machinery, not the hypothesis. Because it cannot preview the
result, its findings may change [DESIGN.md](DESIGN.md) freely.

## No-leak rule

The hypothesis is about one thing: **how trained models do on interventions.** Phase 0 never
computes it. The pipeline is checked in two halves that are never joined:

| Half | What runs | Models allowed |
| --- | --- | --- |
| **Training side** | data generation, training, checkpoints, L_obs on recorded actions, V*, matching, exclusions | real architectures, trained on smoke data or on made-up targets |
| **Intervention side** | interventions, strata, scoring, gap, interval, p, gates, verdicts, result file | **stand-ins only**: `oracle`, `copy`, untrained networks, hand-built noisy pairs |

**Never computed in Phase 0**, not even "just to check it runs" and not even on smoke states:

- the intervention error (any stratum) of any trained model, main or control;
- any gap, G1, G2 or control-pair result between trained models;
- any measure of how much a trained model relies on the action, using real-style data (for
  example shuffling or zeroing actions on smoke data and comparing losses). Such checks are done
  only on made-up targets (*Training pipeline checks*, part 2).

**Never used in Phase 0:** pilot seeds (they belong to the pilot) and test states (they do not
exist before freeze 2).

The scripts must enforce the split: the scoring entry point refuses a trained checkpoint while
the phase flag is `phase0`. The Phase 0 report lists every quantity computed, so a reviewer can see
that nothing above was.

## Environment checks

On the smoke data, no models involved:

| Check | Pass condition |
| --- | --- |
| Bit-exact replay | every smoke state and all 25 actions replay byte-identically in a fresh process in the pinned image |
| Push thresholds | for every contact intervention, the box centre's end-position difference and absolute rotation difference against the no-contact reference (DESIGN.md, *Environment*, *Push label*) are recorded and plotted, with the total motion over the step beside them; d_push and θ_push are fixed by the rule in DESIGN.md (*Interventions*: defaults 2 mm and 1°, moved only to a gap between grazes and pushes, and kept when there is none). The values, plots and the share of contacts labelled graze are reported |
| Pusher can move the box | checks that a push rate failure cannot come from the scene being too weak. *From the scene file:* the velocity actuator's force is kv · (v_cmd − v), capped by its `forcerange`, so against a box at rest the pusher exerts at most F(v_cmd) = min(kv · v_cmd, forcerange). This is computed at 0.05 m/s (the slowest aimed intervention) and 0.10 m/s (the scripted push), against F_resist, the force needed to start the box sliding in the worst direction: √2 times the `frictionloss` on the box's slide joints (a diagonal push, which meets both joints' friction; DESIGN.md, *Environment*). **Pass: F(0.05 m/s) ≥ 2 · F_resist.** *In the simulator:* from rest, the pusher placed just off the middle of one face, centred pushes at 0.05, 0.10 and 0.15 m/s for one control step, along both axes and the diagonal, each move the box at least the default d_push; an off-centre push at 0.10 m/s turns it at least the default θ_push. The box's displacement and rotation, and its steady speed against the expected v_cmd − F_resist / kv, are reported. If either part fails, the scene (box mass, table resistance or actuator) is changed before freeze 1 and the change recorded |
| Push rate | at least 90% of smoke states have one or more push interventions, and the mean is at least 6 of 24 per state; contact and graze rates are reported beside it |
| Contact coverage in training-like data | the fraction of contact transitions in the smoke episodes, and how many come from the scripted push vs the ε-random actions, are reported |
| Intervention coverage | checks that the interventions are unusual enough, compared with the behaviour data, for the test to have something to find. A *coverage pool* of 1,000 behaviour episodes (the size of the training split) is generated from the smoke seed range, and its transitions are labelled with the push label and thresholds fixed above, each against its own no-contact reference. The question is whether the behaviour policy took actions like the intervention's at states like the intervention's, so the distance is to the actions taken at the nearest states; no prediction is made. For a (state, action) pair, its *state neighbours* are the 100 pool transitions (any outcome) nearest in s_t, each state dimension standardised on the pool, and its distance is the mean Euclidean distance, in m/s, from its action to the 5 nearest actions of those neighbours. d_int is this distance for each push intervention on the smoke states; d_obs is the same for each push transition in the 50 smoke episodes (behaviour actions, not in the pool). τ is the 95th percentile of d_obs: an ordinary push is farther than τ from the actions taken at its nearest states only 5% of the time. The *tail share* T is the share of push interventions with d_int > τ. **Pass: T ≥ 0.25**, five times what interventions no more unusual than ordinary pushes would give. Also reported: T for aimed and uniform interventions separately, R = median(d_int) / median(d_obs), the number of behaviour pushes beyond τ, the share of the pool's push transitions that come from ε-random actions, and, not gated, the same tail share with the earlier distance (mean distance to the 5 nearest pool push transitions in (s_t, a_t), each dimension standardised on the pool, as for `knn`). That earlier distance was gated first and replaced in Phase 0: interventions start at behaviour states, so its 11 state dimensions dominated it, and on exploratory smoke seeds its T stayed at 0.14 from ε_near 0.05 to 0.02, so it could not tell whether lowering ε_near made the interventions more unusual. **If T < 0.25:** ε_near (DESIGN.md, *Data*) is lowered to the next value of the ladder 0.2 → 0.05 → 0.02, and Phase 0 is rerun in full on new smoke seeds, since the data change. The first value that passes is kept. If 0.02 still fails, ε_near stays at 0.02, the failed check is reported, and the design proceeds; the limitation in DESIGN.md (*Limits*, full observability) is then the expected reading of a G1 *fail*. ε_near is never set to 0, which would put near-box uniform interventions outside the training support |
| Policy log-density | for each smoke state, the behaviour policy's log-density (DESIGN.md, *Interventions*, computed with the episode's goal) of its logged action and of each of its 24 interventions, reported by stratum and separately for aimed and uniform interventions, beside the log-density of the logged actions in the coverage pool. Reported, not gated; read together with intervention coverage |
| Compute | wall time of data generation and of one training run per architecture on smoke data, against the *Compute* estimate |
| Shadow actions | checks that shadow actions come from their own stream and are indistinguishable from real actions given the history (*Controls* in DESIGN.md). Generating the smoke data with and without shadow actions gives byte-identical transitions; the shadow actions' scripted/uniform proportions and the distribution of (a'_t − a_{t-1}) match those of the logged actions (a_t − a_{t-1}), reported with a two-sample test |
| Positive-control feasibility | checks whether a built-in gap can be large on push interventions while small on behaviour data. D_obs and D_int (DESIGN.md, *Controls*) are computed from the simulator alone on smoke states, with m from 100 policy draws per state, and their ratio is reported. If D_int / D_obs is so small that no p could give a 20% gap at matched loss for any plausible E_int, the positive control is redesigned before freeze 1 |

## Training pipeline checks

Training side only (see *No-leak rule*). Each check has a known right answer that is not the
result.

**1. Data is right.**

| Check | Pass condition |
| --- | --- |
| Alignment | on a hand-built episode with known values, every example pairs s_t, the history up to t, a_t and s_{t+1}; no off-by-one, and the history contains nothing after t |
| Target | Δθ wraps correctly across ±π; the box's 3 scored numbers match a direct computation from the states |
| Scaling | the std-MSE scaler is computed from the training split only; changing validation data does not change it |
| Splits | no episode seed or transition hash appears in two splits |

**2. Each input is wired in: made-up targets.** Each architecture is trained on smoke-data inputs
with a made-up target whose right answer is known. These targets are not the physics, so they say
nothing about the hypothesis.

| Made-up target | Pass condition | Catches |
| --- | --- | --- |
| zero change | training loss near 0 | loss and scaling bugs |
| pusher change = a_t × 0.1 s | training loss near 0 | the action not reaching the model (a model ignoring it cannot fit this) |
| a fixed function of the history only | training loss near 0, except `contact_wm`, which ignores the history by design | the history not reaching the model |
| a fixed smooth function of s_t and a_t | training loss near 0 | capacity and gradient flow |

Two architectures cannot be trained on every target as worded above, and are tested on what
their training can represent:

- `contact_wm`: its pusher follows its lag model and its correction acts on the box only, from
  geometric features. It is trained on zero change for the box's 7 numbers, pusher position change
  = a_t × 0.1 s, and a fixed smooth function of its correction network's inputs on the box's
  (Δx, Δy).
- `mlp-ms`: it trains on rollouts, which need states that follow the target. It is trained on
  made-up episodes built from each training episode's real first state and logged actions, in
  which nothing moves, or only the pusher position moves by a_t × 0.1 s; their one-step changes
  are the first two targets above. Its network and one-step wiring are `mlp`'s, which takes all
  four targets; the rollout's feedback of predicted states is covered by a unit test (part 4).

**3. Learning works**, on the real smoke data, recorded actions only:

| Check | Pass condition |
| --- | --- |
| Overfit a tiny batch (64 examples) | training loss near 0 |
| Losses fall | L_obs decreases over checkpoints for every architecture and width |
| Beats trivial baselines | every architecture beats `copy` and `constvel` on L_obs |
| Matching runs | curves pass a V*-like level computed by DESIGN.md's rule from these runs; matching and exclusions pick the expected checkpoints |
| Training budget | runs train the full number of updates at a constant learning rate, checkpoints follow DESIGN.md's schedule (every 10 updates up to 1,000, every 100 up to 2,000, then every 1,000), the plateau criterion is reported, and runs continue past their best checkpoint |
| Reproducibility | the same seed gives the same loss curve on CPU |

The smoke split is small (50 episodes); these checks test mechanics, not final performance.

**4. Model-specific unit tests** (in `tests/`, no training needed):

| Model | Check |
| --- | --- |
| `contact_wm` | with parameters set by hand in the test only: pusher far away gives no contact; a push through the centre moves the box straight; an off-centre push turns it; a steep push slips. The smooth contact passes a gradient check |
| `mlp-ms` | the first step of its multi-step loss equals the shared loss; at the second step its input state is its own prediction and (s_t, a_t) has joined the history; the made-up episodes' one-step changes equal the made-up targets |
| `mlp-shortcut` | shadow actions come from their own stream; the share of swapped inputs is p |
| `blind` | its input contains no a_t (the history keeps its past actions, as in DESIGN.md); its output does not change with a_t |

## Intervention-side checks, with stand-in models

Intervention side only (see *No-leak rule*). Each stand-in has 10 "seeds", as the main models at medium width, so seed
resampling is exercised at the size G1 uses. Each stand-in seed's prediction is the true target plus two Gaussian terms: a bias
shared by all of that seed's push transitions (drawn once per seed), and independent noise per
transition. The shared bias makes seeds differ systematically, as trained seeds do; per-transition
noise alone averages out over thousands of transitions and would make seed differences too
small to test the seed resampling. The bias and noise scales are set from the spread expected
between trained seeds and are reported.

| Stand-in | Expected result | What it checks |
| --- | --- | --- |
| `oracle` (the simulator's own outcome) | E_int = 0, L_obs = 0 | targets, alignment of predictions and truth, scoring |
| `copy` | equals the value computed directly from the box-motion data | std-MSE, target scaler, floors |
| untrained network | does not beat `copy` on L_obs | that G0.2 can fail |
| `noisy-a` vs `noisy-a'` (same bias and noise scales, independent draws) | G1 *fail*; interval covers zero | no false positive, including from seed differences |
| `noisy-a` vs `noisy-b` (noise on push transitions 1.5 × larger in standard deviation) | G1 *pass*; the estimated gap's interval covers the known gap | that G1 can pass, and the gap, interval and p are computed correctly |
| synthetic checkpoint series with known L_obs curves, one never reaching V* | the expected matched checkpoints, the expected exclusion | matching, exclusions and the *not testable* path |
| control pairs and verdicts: stand-in positive and twin pairs set to separate or not in every combination | each `gate_g1_reading` value, `gate_positive_control_pass` and `gate_negative_controls` come out as DESIGN.md defines | the control readings and the G1 verdict logic |
| synthetic checkpoint series with a gap that is present at V* and absent at best checkpoints, and series giving each *Gap along training* reading | `matched_only` true for the first; *overlap*, *persistent*, *closing* and *mixed* each come out as DESIGN.md defines | the reported matching analyses |

**Repeated pairs: false-alarm and detection rates.** One run of a pair is one draw; a correct
test still separates a no-difference pair now and then. So the following are each repeated 200
times with fresh draws (stand-in predictions only; the smoke states stay fixed), with the full G1
procedure (states and seeds resampled) and, beside it, the states-only interval:

| Pair | Reported | Pass condition |
| --- | --- | --- |
| `noisy-a` vs `noisy-a'` (twins) | fraction of runs that separate | full procedure: at most 5% (10 of 200). The states-only rate is reported, not gated; if it is clearly higher, that is the reason the design resamples seeds |
| `noisy-a` vs `noisy-c` (push noise set so the true relative gap is 25%, just above the 20% threshold) | fraction of runs that separate | reported, not gated: the detection rate near the threshold, i.e. how sensitive G1 can be with this many states and seeds |
| `noisy-a` vs `noisy-b` (push noise 1.5 × larger, a gap of roughly 125%) | fraction of runs that separate | at least 95% |

The twin false-alarm rate is also the expected chance that a real negative control separates
and voids a G1 pass; it is quoted beside the G1 verdict. If a pass condition fails, the design is
changed before freeze 1, as for any other check.

**Format check.** The result JSON from the stand-in run conforms to
[INTERVENTION_REPORTING.md](INTERVENTION_REPORTING.md).

**Record.** `results/phase0_smoke.json` and a short report, both committed and hashed in
freeze 1. The report lists every quantity computed, in two lists (training side, intervention
side), so it is visible that no trained model was scored on interventions. Smoke data, stand-ins and numbers never enter a result. If a check fails, the design is
changed and Phase 0 rerun on new smoke seeds; each rerun and its reason is listed in the report.
