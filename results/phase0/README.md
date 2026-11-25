# Phase 0 results

Results are filed by milestone, then by attempt. An attempt is a rerun after a failed check or a
design change; each keeps its own report, so earlier attempts stay readable. The matching data
(gitignored) are under `data/phase0/` with the same layout.

| Folder | What | Report | Data |
| --- | --- | --- | --- |
| `milestone1/attempt0/` | simulator, behaviour policy, interventions, environment checks; first run (ε_near = 0.2) | `REPORT_milestone1.md` | `data/phase0/milestone1/attempt0/` |
| `milestone1/attempt1/` | rerun on new smoke seeds: coverage measure replaced, ε_near = 0.05 | `REPORT_attempt1.md` | `data/phase0/milestone1/attempt1/` |
| `milestone1/attempt2/` | rerun on new smoke seeds: push label against the no-contact reference; amd64 replay check | `REPORT_attempt2.md`, `REPORT_amd64.md` | `data/phase0/milestone1/attempt2/` (the smoke data every later milestone uses) |
| `milestone2/attempt0/` | data pipeline, models, training-side checks; first lineup, with `rssm` | `REPORT_milestone2.md` | checkpoints of the `rssm` runs in `data/phase0/milestone2/attempt0/checkpoints/` (the others were identical to attempt 1's and were overwritten by it) |
| `milestone2/attempt1/` | rerun with `mlp-ms` in place of `rssm` and the finer checkpoint schedule | `REPORT_milestone2.md` | `data/phase0/milestone2/attempt1/checkpoints/` |
| `exploration/` | candidates for the model with a different objective (rssm with DreamerV3's KL terms, a Gaussian NLL network, `mlp-ms`), on smoke data | none; see milestone 2 attempt 1's report | none kept |

The current attempt of each milestone is set in `src/physical_intervene/constants.py`
(`SMOKE_ATTEMPT`, `MILESTONE2_ATTEMPT`).
