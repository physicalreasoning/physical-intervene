# Phase 0, milestone 2, attempt 1: training-side checks with `mlp-ms` and the finer schedule

Training side only (SMOKE_TEST.md, *No-leak rule*). Models were trained on smoke data with recorded
actions, or on made-up targets. No trained model was scored on interventions, no intervention
error of any kind was computed, and the test-like states file (`states.npz`) was never opened.
Record: `training_checks.json`; plots: `plots/`. Checkpoints:
`data/phase0/milestone2/attempt1/checkpoints/` (gitignored, 2.7 GB). Smoke data: milestone 1
attempt 2 (`data/phase0/milestone1/attempt2/`).

**What changed since attempt 0** ([attempt 0's report](../attempt0/REPORT_milestone2.md)):

- `mlp-ms` replaces `rssm` as the model with a different objective (DESIGN.md, *Models* and
  *Target and loss*). In attempt 0, `rssm` learned slowest, set V\* for every pair (0.107) and
  failed two checks. The evidence and the candidates tried are in `../../exploration/` and
  RATIONALE.md, *Why `mlp-ms`*.
- Checkpoints every 10 updates up to 1,000, every 100 up to 2,000, then every 1,000. In attempt 0,
  with every 100 updates, `contact_wm` passed the band around V\* between two checkpoints.
- SMOKE_TEST.md part 2 now words the made-up targets for `contact_wm` and `mlp-ms`.
- The overfit check draws `mlp-ms`'s 64 examples from starts whose 4-step rollout stays in the
  episode, and scores its own training loss. A first rerun drew one example at step 97, which a
  4-step rollout cannot start from, so `mlp-ms` never trained on it and failed (0.029).
- Folders now follow milestone and attempt (`results/phase0/README.md`). Checkpoint metadata
  records the training data path as `data/smoke/attempt2/...`, the folder's name when the runs were
  made; the recorded sha256 identifies the files at their new place.

## Open issues

| # | Issue | Status | What it would affect |
| --- | --- | --- | --- |
| 1 | **What `blind` removes.** DESIGN.md: "as `mlp` with a_t removed"; SMOKE_TEST.md: "its input contains no action". Implemented as DESIGN.md: a_t removed, the history keeps its 8 past actions | **resolved**: DESIGN.md and SMOKE_TEST.md now both say a_t is removed and the history keeps its past actions | what G0.3 tests; under intervention a_t is unrelated to the history either way |
| 2 | **No run met the plateau criterion in 12,000 updates** (best-so-far improvement over the last 10,000 updates: 7% to 55%) | expected at placeholder budget | the pilot's budget; only the mechanics are checked here |
| 3 | **CI installs torch with CUDA packages.** `tests.yml` syncs with `--extra models`; on Linux the PyPI torch wheel pulls about 15 CUDA packages. A CPU-only index in `pyproject.toml` would avoid it but changes the lock file | open (carried from attempt 0) | CI time only |
| 4 | **Checkpoints take 2.7 GB** for these 22 learning runs (121 checkpoints each; the large width dominates). The full experiment has 140 runs, most at medium width | noted | disk for the main runs; checkpoints are needed for *Gap along training* |

Resolved from attempt 0: rssm's parameter count and failed checks (rssm removed); the
`contact_wm` made-up wording (SMOKE_TEST.md part 2); the checkpoint skip (finer schedule, no
exclusion below).

## Results

| Check | Result |
| --- | --- |
| 1. Data (alignment, target, scaling, splits) | **pass**, as in attempt 0 (same data and code) |
| 2. Wiring (made-up targets, threshold 0.01) | **pass**: mlp and gru all four (largest 0.00065, gru history); `mlp-ms` zero_rollout 3.7e-6, action_rollout 9.8e-6; contact_wm three structural variants (largest 0.00013); blind on the action target 0.124, not near 0, as expected |
| 3. Overfit 64 examples | **pass**: mlp 4.2e-6, gru 1.2e-4, `mlp-ms` 2.1e-4 (multi-step; one-step 2.5e-4), contact_wm 0.0043 |
| 3. Losses fall | **pass**, every architecture and width |
| 3. Beats baselines | **pass**: copy 0.940, constvel 0.224 (knn 0.137); every run below both |
| 3. Matching | **pass** (rule reproduced); see below |
| 3. Budget | **pass**: 12,000 updates, constant learning rate, 120 checkpoints on the new schedule; plateau not met (issue 2); 9 of 22 runs continue past their best checkpoint |
| 3. Reproducibility | **pass**: for each architecture a separate rerun (seed 11, small) gives the same L_obs and training loss at all 111 checkpoints to 2,000 updates, bit for bit |
| 4. Unit tests | **pass**: as attempt 0 for contact_wm, mlp-shortcut, blind, widths and metadata; `mlp-ms`: first step of its loss equals the shared loss, second step uses its own prediction with (s_t, a_t) added to the history, made-up episodes reproduce the made-up targets; schedule steps |
| 5. Compute | 7.0 CPU-hours projected for all 140 runs at the placeholder budget (see *Compute*) |

Best L_obs (seed 11; medium: seeds 11, 22, 33):

| Model | small | medium | large |
| --- | --- | --- | --- |
| mlp | 0.078 | 0.057, 0.056, 0.054 | 0.056 |
| gru | 0.120 | 0.060, 0.059, 0.059 | 0.057 |
| `mlp-ms` | 0.100 | 0.077, 0.074, 0.078 | 0.068 |
| contact_wm | 0.021 | 0.017, 0.018, 0.017 | 0.012 |

**Matching** (medium, 3 seeds): medians mlp 0.056, gru 0.059, `mlp-ms` 0.077, contact_wm 0.017;
V\*-like level 0.085, set by `mlp-ms` (attempt 0: 0.107, set by `rssm`).

| Model | matched updates | matched L_obs | excluded |
| --- | --- | --- | --- |
| contact_wm | 480, 420, 420 | 0.0854, 0.0846, 0.0850 | none |
| mlp | 1,900, 1,600, 1,800 | 0.0873, 0.0855, 0.0846 | none |
| gru | 4,000, 4,000, 4,000 | 0.0860, 0.0809, 0.0814 | none |
| `mlp-ms` | 5,000, 4,000, 5,000 | 0.0848, 0.0856, 0.0836 | none |

Every pair passes the match screen (seed means 0.0828 to 0.0858). In attempt 0 one contact_wm seed
was excluded and contact_wm failed the screen against mlp and gru. contact_wm is still matched
early (about 450 of 12,000 updates), as DESIGN.md now states and reads asymmetrically.

Controls trained, not scored: `mlp-shortcut` (stand-in p = 0.5, swapped share 0.4998) best L_obs
0.080; `blind` 0.138.

### Compute

Each model alone, one thread, 300 updates timed, projected to the placeholder budget (12,000
updates and L_obs at its 121 checkpoints). Apple arm64, Python 3.14.7, torch 2.15.0.

| Model | ms per update (small / medium / large) | minutes per run (small / medium / large) |
| --- | --- | --- |
| mlp | 1.1 / 1.5 / 5.8 | 0.2 / 0.3 / 1.2 |
| gru | 3.4 / 8.3 / 30.5 | 0.7 / 1.8 / 6.7 |
| `mlp-ms` | 3.1 / 5.8 / 20.7 | 0.6 / 1.2 / 4.2 |
| contact_wm | 19.4 / 20.2 / 27.6 | 3.9 / 4.1 / 5.6 |
| blind, mlp-shortcut (medium) | 1.5 to 1.6 | 0.3 |

Projected for the 80 main and 60 control runs at this budget: 5.0 + 2.1 = **7.0 CPU-hours**
(attempt 0: 9.5, mostly `rssm`), against DESIGN.md's "three to five days of CPU". The real budget
is set by the pilot. The whole script took 13 minutes on 8 workers.

## Values chosen (not fixed by the docs)

As in attempt 0, except:

| Item | Value | Why |
| --- | --- | --- |
| `mlp-ms` widths | `mlp`'s: 35 / 141 / 483 (9,846 / 99,416 / 1,001,270) | the same network |
| `mlp-ms` rollout starts | steps 0 to 96 of each 100-step episode | the 4-step rollout must stay in the episode |
| `mlp-ms` made-up episodes | from each training episode's real first state and logged actions: nothing moves (*zero_rollout*), or only the pusher position moves by a_t × 0.1 s (*action_rollout*); the real data's scalers | their one-step changes are the *zero* and *action* targets |
| `mlp-ms` overfit examples | 64 drawn from valid rollout starts; pass on its own (multi-step) training loss, one-step loss reported | it minimises the multi-step loss |

## Every quantity computed

Training side only: data checks; baselines on validation transitions; training loss and L_obs
(with corner RMSE) at every checkpoint, on recorded actions only; made-up-target and overfit
training losses; contact_wm's learned numbers; the mlp-shortcut swapped share; V\*-like level,
matched checkpoints, exclusions, match screen; plateau criterion; timings.

Not computed: any intervention error, any stratum, any gap, interval or gate between trained models,
and any measure of a trained model's reliance on the action using real-style data. `states.npz`
was not read.

## Commands

| Command | Result |
| --- | --- |
| `uv run --extra models python scripts/phase0_training.py --workers 8` | all checks pass |
| `uv run --extra models python scripts/phase0_training.py --timing` | solo timings above |
| `uv run ruff check .`, `uv run ruff format --check .` | clean |
| `uv run --extra models pytest -q` | 82 passed |
| `python scripts/check_docs.py` | no errors |
