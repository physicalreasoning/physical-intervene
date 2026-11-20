# physical-intervene

An intervention test for learned world models, the physical counterpart of
[evm-intervene](https://github.com/physicalreasoning/evm-intervene). It answers
[physicalreasoning/forward#5](https://github.com/physicalreasoning/forward/issues/5): do world
models with matched validation loss predict the same outcome under an action they were not shown?

**Status: design for review, not frozen.** No data has been generated and no model trained. This
repo lives under a personal account until there is one under the `physicalreasoning` org.

## Task 1: planar pushing

A cylindrical pusher pushes a box on a table in MuJoCo. The simulator state can be saved,
restored and re-run under any action, so every intervention has an exact, cheap ground truth.
Contact between pusher and box is the hard case: rare in routine data, and where the action
matters most.

- **Models:** an MLP, a GRU and a small RSSM, trained on the same observational data, plus an
  action-blind negative control, a positive control trained on random actions, and `copy`,
  constant-velocity and nearest-neighbour baselines.
- **Metric:** box-pose error under 24 held-out interventions from each of 1,000 held-out states,
  on the contact stratum, alongside observational validation loss.
- **Gates:** G0 bit-exact replay and controls; G1 matched-loss pairs must differ in intervention
  error (95% interval over held-out states and seeds excluding zero, relative gap ≥ 20%);
  G2 rank correlation between validation loss and intervention error.

Full design: [docs/DESIGN.md](docs/DESIGN.md). Results use the reporting format shared with
evm-intervene, drafted in
[docs/INTERVENTION_REPORTING.md](docs/INTERVENTION_REPORTING.md).

## Pre-registration

`docs/DESIGN.md` is hashed and posted on forward#5 before any model is trained (freeze 1), and
the trained checkpoints are hashed before any test state exists (freeze 2). See *Seal* in the
design.

## Layout

| Path | What |
| --- | --- |
| `docs/DESIGN.md` | pre-registered design |
| `src/physical_intervene/constants.py` | every fixed value from the design, in one place |
| `tests/` | unit tests |

Planned, following evm-intervene: `src/physical_intervene/` simulator, models, baselines and
metrics; `scripts/` for data generation, training, scoring and freezing; `frozen/` for the
freeze records; `results/` for the result JSON; `docs/REPRODUCE.md`.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked --group dev
uv run ruff check src tests
uv run pytest -q
```

## Licence

Apache 2.0, as evm-intervene.
