# Phase 0, milestone 2: data pipeline, models and training-side checks

Training side only (SMOKE_TEST.md, *No-leak rule*). Models were trained on smoke data with recorded
actions, or on made-up targets. No trained model was scored on interventions, no intervention
error of any kind was computed, and the test-like states file (`states.npz`) was never opened.
Record: `training_checks.json`; plots: `plots/`. Checkpoints: `data/phase0/milestone2/checkpoints/`
(gitignored, 657 MB).

## Open issues

| # | Issue | Status | What it would affect |
| --- | --- | --- | --- |
| 1 | **rssm at medium width has 177,291 parameters, not about 100k.** DESIGN.md fixes it at a 200-unit GRU with a 30-dimensional latent; that GRU alone has about 140k parameters, so "equal parameter counts across architectures" and "200 units, 30 latent" cannot both hold. Implemented as written (200 / 30, other layers kept small); small and large widths are solved to 10k and 1M | open | the equal-width rule at the width G1 uses; either the rssm sizes or the width rule must change before freeze 1 |
| 2 | **contact_wm cannot fit most made-up targets as written.** SMOKE_TEST.md exempts it only from the history target, but its structure cannot output, for example, zero pusher motion for a moving pusher (the pusher follows its lag model; the correction acts on the box only, on geometric features). Tested instead on what its structure can represent: zero change on the box's 7 numbers, pusher position change = a_t x 0.1 s, and a smooth function of its correction network's inputs on the box's (dx, dy). All three pass | open | the wording of SMOKE_TEST.md part 2 for contact_wm |
| 3 | **What `blind` removes.** DESIGN.md: "as `mlp` with a_t removed"; SMOKE_TEST.md: "its input contains no action". Implemented as DESIGN.md: a_t removed, the history keeps its 8 past actions | open | what G0.3 tests; under intervention a_t is unrelated to the history either way |
| 4 | **rssm fails two checks at the placeholder settings.** (a) history-only made-up target: 0.021 against 0.01; (b) at small width its best L_obs (0.255) is worse than `constvel` (0.224); it is still falling at 12,000 updates. Its medium and large runs pass | open | rssm's settings are *set by pilot*; G0.2 requires every main model to beat `copy` (which it does, 0.94); the pilot budget should show whether small rssm reaches `constvel`. A follow-up rescoring (`rssm_prediction_rules.json`, from `scripts/phase0_rssm_prediction_rules.py`) shows the gap is not prediction noise: decoding at posterior means while filtering and the prior's mean at the predicted step lowers best medium L_obs only about 10% (0.096 to 0.086, seed 11), and the history target still fails (0.020). Decoding with the posterior at s_{t+1} (uses the answer, so an upper bound) gives 0.023: the latent can hold the next step, but the prior does not predict it |
| 5 | **The first dense checkpoint can overshoot V\*.** On the smoke runs, contact_wm passes the V\*-like level between updates 100 and 200; one of its three seeds lands 5.4% from V\* and is excluded, and the match screen fails contact_wm against mlp and gru (5.5% and 5.3% against 5%, seed means). DESIGN.md says the schedule keeps every run within the match screen. Cause: near V\* contact_wm's L_obs falls from about 0.13 to 0.10 between updates 100 and 200, a step of about 27% of V\*, while the band within 5% of V\* is 10% wide, so whether a checkpoint lands in it is chance. Its matched checkpoints also land below V\* (about 0.102) while mlp's and gru's land above it (about 0.108), because they fall only 4 to 7% per 100 updates there; the match screen fails on that offset. A candidate fix is denser early checkpoints (for example every 10 updates for the first few hundred; contact_wm's evaluation is cheap) | evidence only; denser early checkpoints preferred, not yet written into DESIGN.md | matching and exclusions for any architecture much better than the one that sets V\*; depends on the pilot's learning rate and budget |
| 6 | **No run met the plateau criterion in 12,000 updates** (best-so-far improvement over the last 10,000 updates: 21% to 63%) | expected at placeholder budget | the pilot's budget; only the mechanics are checked here |
| 7 | **CI now installs torch.** `tests.yml` syncs with `--extra models`, so the model tests run; on Linux the PyPI torch wheel pulls about 15 CUDA packages (several GB per CI run). A CPU-only index in `pyproject.toml` would avoid it but changes the lock file | open | CI time only |

## What was built

| File | What |
| --- | --- |
| `src/physical_intervene/data.py` | `Split`, `load_split` (refuses `states.npz`), `make_examples` (s_t, 8-pair history with mask, a_t, state change, box target), `TargetScaler`, `InputNorm`, box target from a predicted state change, split overlap |
| `src/physical_intervene/models/` | `mlp.py` (`mlp`, `blind`, `mlp-shortcut`), `gru.py`, `rssm.py`, `contact_wm.py`, `common.py` (shared batch), `__init__.py` (`build`, width sizing, `param_table`) |
| `src/physical_intervene/baselines.py` | `copy`, `constvel`, `KNN` |
| `src/physical_intervene/train.py` | `train` (full budget, constant learning rate, design checkpoint schedule, L_obs at every checkpoint, seeded streams, single thread, deterministic), `save_checkpoint`, `evaluate`, `fit_loss` |
| `src/physical_intervene/metrics.py` | corner RMSE, V\*, matched checkpoint, exclusion, match screen, plateau criterion |
| `src/physical_intervene/training_checks.py` | the checks of parts 1 to 3, each returning a dict |
| `scripts/phase0_training.py` | runs everything (`--workers N`, `--quick`), and `--timing` for single-run compute timings |
| `tests/test_models.py`, `tests/test_data_pipeline.py` | part 4 unit tests and the data checks |
| `constants.py` | new section: `PHASE = "phase0"`, history padding, state-change floors, layer layouts, contact_wm generic initial values and smoothing, near-zero threshold |

Run: `uv sync --extra models`, then `uv run python scripts/phase0_training.py` (about 40 CPU-minutes
for the longest run; 8 in parallel), then `uv run python scripts/phase0_training.py --timing`.

**Checkpoint metadata** (for milestone 3's scorer): every `.pt` holds `meta` with architecture,
width, seed, updates, `trained: true`, `phase: "phase0"`, the training and validation files with
sha256 and seed ranges, the target, parameter count, config and torch version (also written beside
it as `.json`), plus the input normaliser and both scalers.

## Results

| Check | Result |
| --- | --- |
| 1. Alignment | **pass**: 12-step hand-built episode, every number encodes its step; no off-by-one, nothing at or after t in the history, padding as chosen |
| 1. Target | **pass**: dtheta = 0.02 rad on every step of a hand-built turn across +pi; box target from states vs stored (qpos) on smoke data: max difference 0, 0, 8.9e-16 |
| 1. Scaling | **pass**: box, state-change and input scalers identical when the validation data are changed; train std 3.25 mm, 3.19 mm, 0.0282 rad (floors do not bind) |
| 1. Splits | **pass**: train seeds 3,700,000 to 3,700,999, validation 3,600,000 to 3,600,049, all smoke; 0 shared seeds, 0 shared transition hashes |
| 2. Wiring (made-up targets, threshold 0.01) | **fail (1 of 16)**: mlp, gru all four pass (largest 0.00065, gru history); rssm zero, action, smooth pass (largest 0.0066), **history 0.021**; contact_wm three structural variants pass (largest 0.00013); blind on the action target 0.124, not near 0, as expected |
| 3. Overfit 64 examples | **pass**: mlp 4.2e-6, gru 1.2e-4, rssm 0.0018, contact_wm 0.0043 |
| 3. Losses fall | **pass**, every architecture and width |
| 3. Beats baselines | **fail (1 of 20 runs)**: copy 0.940, constvel 0.224 (knn 0.137). Best L_obs, medium seed 11: mlp 0.057, gru 0.060, rssm 0.097, contact_wm 0.017. **rssm small 0.255** fails against constvel |
| 3. Matching | **pass** (rule reproduced): medians mlp 0.056, gru 0.059, rssm 0.097, contact_wm 0.017; V\*-like 0.107; matched updates mlp 900 to 1,000, gru 1,700 to 1,900, rssm 7,000 to 9,000, contact_wm 200; one exclusion (contact_wm seed 33, issue 5) |
| 3. Budget | **pass**: 12,000 updates, constant learning rate, 29 checkpoints on the design schedule; plateau not met (issue 6); 8 of 22 runs continue past their best checkpoint |
| 3. Reproducibility | **pass**: for each architecture a separate rerun (seed 11, small) gives the same L_obs and training loss at all 21 checkpoints to 2,000 updates, bit for bit |
| 4. Unit tests | **pass**: contact_wm (hand-set: no contact far away, straight centre push, off-centre push turns clockwise, steep push slips onto the cone edge, gradient check on inputs and learned numbers); rssm (KL >= 0, free nats remove the KL gradient, ELBO terms add up and match a hand computation, fixed sampling seed reproduces); mlp-shortcut (with p = 0 identical to mlp, so the swap has its own stream; swapped share within 4 sigma of p); blind (no a_t input, output unchanged by a_t); widths within 3% of target; checkpoint metadata |
| 5. Compute | 9.5 CPU-hours projected for all 140 runs at the placeholder budget (see *Compute*) |

Controls trained, not scored: `mlp-shortcut` (stand-in p = 0.5, swapped share 0.4998) best L_obs
0.080; `blind` 0.138.

contact_wm's learned numbers (medium, seed 11), from generic starts of 4 cm, 1.5 cm, 0.5, 0.5:
half-width 5.27 cm, pusher radius 0.61 cm, pusher-box friction 0.41, table friction 0.069, lag
14.5 ms. Reported for interest; nothing is set from them.

### Compute

Each model alone, one thread, 300 updates timed, projected to the placeholder budget (12,000
updates and L_obs at its 29 checkpoints). Apple arm64, Python 3.14.7, torch 2.15.0.

| Model | ms per update (small / medium / large) | minutes per run (small / medium / large) |
| --- | --- | --- |
| mlp | 1.1 / 1.5 / 5.6 | 0.2 / 0.3 / 1.1 |
| gru | 3.4 / 8.1 / 31.6 | 0.7 / 1.7 / 6.5 |
| rssm | 31.5 / 40.5 / 103.1 | 6.7 / 9.7 / 24.6 |
| contact_wm | 19.6 / 20.5 / 28.5 | 3.9 / 4.1 / 5.7 |
| blind, mlp-shortcut (medium) | 2.0 | 0.4 |

Projected for the 80 main and 60 control runs at this budget: 6.8 + 2.8 = **9.5 CPU-hours**,
against DESIGN.md's "three to five days of CPU". Every run is within the design's per-run
estimate (minutes for mlp, tens of minutes at most for the others). The real budget is set by the
pilot; at 10 times this budget the total is still within the estimate. With 8 runs in parallel
the same runs were 1.3 to 2 times slower per update (memory contention). The wall time of the
whole script (4.6 h) is not a compute figure: the machine appears to have slept during it.

## Values chosen (not fixed by the docs)

| Item | Value | Why |
| --- | --- | --- |
| Smoke split | train on the coverage pool (1,000 episodes, 100,000 transitions); validate on the 50 smoke episodes (5,000) | training-split size, so the timing is realistic |
| History before step 8 | copies of s_0 with a zero action, plus a mask input; rssm leaves its state unchanged on masked steps | keeps every transition, the same for every model |
| dtheta from a prediction | atan2 of (current + predicted) cos and sin, minus the current angle, wrapped | the predicted pair need not have unit norm |
| Placeholders (*set by pilot*) | Adam, learning rate 1e-3, batch 256 transitions (rssm: 16 sequences of 50), 12,000 updates | plain defaults; 12,000 makes the plateau criterion evaluable. Nothing was chosen by looking at any intervention quantity |
| Check budgets | made-up 5,000 updates, overfit 2,000, rerun 2,000 | check settings only |
| "Near 0" | shared training loss below 0.01, in the real training split's state-change scaler (predicting the mean gives about 1) | fixed before the runs |
| Made-up targets | zero; pusher position change = a_t x 0.1 s, others 0; tanh of a fixed random projection of the (padding-zeroed) history; a fixed random tanh network of (s_t, a_t). Scaled by the real state-change std | |
| mlp-shortcut p | stand-in 0.5 (0.3 in the unit tests) | p is computed after the main runs |
| Widths | free size solved to the closest count: mlp 35 / 141 / 483 (9,846 / 99,416 / 1,001,270); gru 39 / 135 / 441 (10,073 / 99,641 / 1,000,199); rssm units 46 / 200 / 478 (10,127 / 177,291 / 1,001,423); contact_wm correction 91 / 307 / 991 (9,925 / 99,781 / 999,925); blind 36 / 142 / 483 | issue 1 for rssm |
| State-change floors | 1e-4 for positions and velocities, 1e-3 for cos, sin and omega | the design gives floors for the box target only |
| gru inputs | GRU over s_{t-8} .. s_t, each with the action into it and the mask; head on (h_t, a_t) | h_t has seen s_t |
| rssm | decodes the standardised state change (not the state), Gaussian with std 1; posterior on (h, s); latent std floor 0.1; training windows of 8 burn-in states plus 50 scored transitions; latent and decoder sizes scale with the GRU units (0.15, 0.25) | the change is about 30 times smaller than the state's spread |
| contact_wm | initial lag 20 ms, contact sharpness 1,000 /m, limit-surface constant 0.7652 x half-width (uniform square), table deceleration mu g; correction features: contact-weighted contact indicator, point, normal and relative velocity, and the start velocity, box frame; residual on box pose and velocity, last layer at zero | generic values, no MuJoCo number |
| Plateau criterion | on the best L_obs so far, compared 10,000 updates apart | the design's wording leaves the comparison open |

## Ambiguities in the docs

- Issues 1 to 3 above.
- DESIGN.md: "Checkpoints are saved every 100 updates ... so no run steps past V\* by more than the
  match screen allows." Not true on these smoke runs for contact_wm (issue 5).
- DESIGN.md, plateau criterion: "improved by no more than 1% over its last 10 checkpoints at
  1,000-update spacing" does not say whether it compares raw or best-so-far L_obs; best-so-far is
  used.

## Every quantity computed

Training side only.
- Data: alignment, target and scaling checks; split seed and transition-hash overlap; scaler and
  normaliser statistics of the training split.
- Baselines on validation transitions: L_obs and corner RMSE of copy, constvel, knn.
- Training runs: training loss and L_obs (with corner RMSE) at every checkpoint, on recorded
  actions only; made-up-target and overfit training losses; contact_wm's learned numbers; the
  mlp-shortcut swapped share; V\*-like level, matched checkpoints, exclusions, match screen; plateau
  criterion; timings.

Not computed: any intervention error, any stratum, any gap, interval or gate between trained models,
and any measure of a trained model's reliance on the action using real-style data. `states.npz`
was not read (the loader refuses it). No pilot, train, validation or test data were generated.

## Commands

| Command | Result |
| --- | --- |
| `uv run ruff check .` | All checks passed |
| `uv run ruff format --check .` | 40 files already formatted |
| `uv run pytest -q` (Python 3.14.7, with the `models` extra) | 77 passed |
| model and data tests on Python 3.11.15 and 3.12.13 (`uv run --isolated --extra models`) | 38 passed each |
| `python scripts/check_docs.py` | no errors |
