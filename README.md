# physical-intervene

An intervention test for learned world models, the physical counterpart of
[evm-intervene](https://github.com/physicalreasoning/evm-intervene). It answers
[physicalreasoning/forward#5](https://github.com/physicalreasoning/forward/issues/5): do world
models with matched validation loss predict the same outcome under an action they were not shown?

**Status: design is a draft for the smoke test, not frozen.** No data has been generated and no model trained.

## Task 1: planar pushing

A round pusher pushes a box on a table in MuJoCo. The simulator state can be saved,
restored and re-run under any action, so every intervention has an exact, cheap ground truth.
Contact between pusher and box is the hard case: rare in routine data, and where the action
matters most.

- **Models:** an MLP, a GRU, an MLP trained on a multi-step loss and a structured model with pushing rules built in
  (`contact_wm`, the counterpart of evm-intervene's `conservation_wm`), trained on the same
  observational data, plus a
  positive control with a known gap built in at matched loss (`mlp-shortcut`), negative controls
  (each architecture retrained on new seeds), an action-blind sanity control, and `copy`,
  constant-velocity and nearest-neighbour baselines. Every model
  trains on the same behaviour-policy data.
- **Metric:** box-pose error under 24 held-out interventions from each of 1,000 held-out states,
  on the push stratum (contact that moves the box; grazes reported separately), alongside
  observational validation loss.
- **Gates:** G0 bit-exact replay and controls; G1 matched-loss pairs must differ in intervention
  error (Welch's t-test over 20 seeds per architecture, 95% interval excluding zero, relative
  gap ≥ 20%);
  G2 rank correlation between validation loss and intervention error.

Full design: [docs/DESIGN.md](docs/DESIGN.md). Results use the reporting format shared with
evm-intervene, drafted in
[docs/INTERVENTION_REPORTING.md](docs/INTERVENTION_REPORTING.md).

## Pre-registration

A smoke test (Phase 0) first runs the whole pipeline with stand-in models whose answers are
known, to check that every part works; real architectures may be trained on smoke data to check
the training side, but no trained model is ever scored on interventions. Then
`docs/DESIGN.md` is hashed and posted on forward#5 before any model is trained on the training
split (freeze 1), and
the trained checkpoints are hashed before any test state exists (freeze 2). Later changes are
dated amendments, never edits. See [docs/SMOKE_TEST.md](docs/SMOKE_TEST.md) and *Seal* in the
design.

## Layout

| Path | What |
| --- | --- |
| `docs/DESIGN.md` | pre-registered design |
| `docs/SMOKE_TEST.md` | Phase 0 smoke-test plan, run before freeze 1, not part of the frozen analysis |
| `docs/background/` | background, not frozen: RATIONALE.md, EVM_COMPARISON.md, RESULTS_FORMAT.md |
| `src/physical_intervene/constants.py` | every fixed value from the design, in one place |
| `tests/` | unit tests |
| `scripts/check_docs.py` | docs checks run in CI: relative links resolve, no em dashes |
| `results/phase0/` | Phase 0 records, by milestone and attempt (see its README) |
| `scripts/phase0_env.py` | Phase 0: smoke data and the environment checks (simulator only) |
| `scripts/phase0_training.py` | Phase 0: the training-side checks (data, wiring, learning, compute); needs the `models` extra |
| `scripts/phase0_model_exploration.py` | Phase 0 exploration: candidates for the model with a different objective (rssm with DreamerV3's KL terms, a Gaussian NLL network, `mlp-ms`), compared on smoke data |
| `scripts/phase0_rssm_prediction_rules.py` | Phase 0 exploration: rescores the first draft's rssm checkpoints under other prediction rules |
| `scripts/phase0_intervention.py` | Phase 0: the intervention-side checks with stand-ins only (scorer, gates, control readings, repeated pairs, format check); needs the `models` extra |
| `scripts/phase0_seed_interval.py` | Phase 0 exploration: G1's false-alarm rate and power at large seed spreads and at the test set's size, for the design's interval and two wider ones (stand-ins only) |
| `scripts/phase0_record.py` | Phase 0: writes the record, `results/phase0_smoke.json`, from the three milestones' records |

Planned, following evm-intervene: `src/physical_intervene/` simulator, models, baselines and
metrics; `scripts/` for data generation, training, scoring and freezing; `frozen/` for the
freeze records; `results/` for the result JSON; `docs/REPRODUCE.md`.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked --group dev --extra models
uv run ruff check src tests
uv run pytest -q
```

The `models` extra installs torch, needed by the learned models and their tests.

## Licence

Apache 2.0, as evm-intervene.
