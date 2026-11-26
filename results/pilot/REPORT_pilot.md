# Pilot report: data, pilot grid, pilot confirmation, compute projection

Run after freeze 1 on branch `pilot`, 2026-11-26. Freeze 1: commit 117ae28, `frozen/design.json`
sha256 `f55415351f971d7726a81708ef2b3bd0508e6f12b30c7916dad564224999d140`. Records:
`settings.json`, `confirmation.json`, `pilot_states_replay.json` in this folder; data in
`data/main/` (gitignored, described by its `manifest.json`).

## Open issues

| Issue | Status | What it would affect |
| --- | --- | --- |
| The positive control may not separate in the main run: at the pilot's p = 0.4 the predicted relative gap is 18.5%, below G1's 20%, and `mlp-shortcut`'s loss rose about 40% more than p² · D_obs, so 2 of its 3 seeds missed the match screen | accepted as written; the rule is unchanged (DESIGN.md, *Controls*: the rule stays whatever the pilot shows) | if the formal positive control does not pass, a G1 *fail* reads *inconclusive: test not shown sensitive*; a G1 *pass* is unaffected |
| Three amendments after freeze 1, each a fix to code no Phase 0 check reached: the format check without `mlp-shortcut` (`2026-11-26.md`), the scorer loading a trained checkpoint (`2026-11-26-2.md`), the test states changing a file freeze 2 hashes (`2026-11-26-3.md`) | done; committed, listed in `frozen/amendments.json`, posted on forward#5 | no rule or reading; without the second and third, scoring after freeze 2 would have refused to run |
| Main training may run on a second machine (16-core x86_64, NixOS, Python 3.15.2 through a `uv` wrapper; this pilot ran on Apple M1 Pro, Python 3.14.7) | open: machine not yet chosen | the run log must record which machine, Python and torch trained which runs; package versions come from `uv.lock` either way; checkpoints are hashed at freeze 2 |
| `contact_wm`'s lowest-scoring setting is at the edge of its extended grid (learning rate 1e-2); the tie rule chose 3e-3 | reported, as the *Pilot rule* says | none: the chosen setting is inside the grid |

## Data (step 1)

Generated in the pinned image (`sha256:547fa5d0…b233b1`, equal to `image.id` in freeze 1) with
`make_data.py --workers 8`; every manifest entry has machine `x86_64`, Python 3.12.16, MuJoCo
4.0.1, numpy 2.5.5, and `epsilon_near` 0.05 (freeze 1's value). Every file's sha256 matches the
manifest.

| File | sha256 | Seeds | Counts |
| --- | --- | --- | --- |
| `train.npz` | `cd3562fd9ed1e557a5dd4c0d336c016f2d17606b5ad12da77e7c353db8dfcc97` | 1000000 to 1000999 | 1,000 episodes, 100,000 transitions |
| `validation.npz` | `c5de9783f63dd451300cdbc2acc2c7408a2ebc599b1e03256736c7f251e5ffcb` | 2000000 to 2000099 | 100 episodes, 10,000 transitions |
| `pilot.npz` | `6cd0d07084aa86936006fa9536fe4c3f094240465de7d39c1d43118d7a236054` | 4000000 to 4000999 | 1,000 episodes |
| `pilot_states.npz` | `bb6b19b71508e4f25472d901c8713150a003ae9e23cd487a51cd54b5b42ac5da` | 4100000 to 4100199 | 200 states × 25 actions (logged, 12 aimed, 12 uniform); no skipped seeds |
| `pilot_history.npz` | `7cf146675f0c822131f0f9dc58e94e2a3c6e7b7db6a4194d7ea547a6ceaab5ed` | the 200 pilot states | model input histories |
| `validation_policy_mean.npz` | `e962b66b6049235801394ed3b1769c96194133fbc9f26ea96c131fdc14027f39` | validation | 10,000 transitions × 100 policy draws |
| `pilot_states_policy_mean.npz` | `d2a991d667518b649e4b5659d21dd3cba5baff155411a8ff080c7c822b7b0f6e` | pilot states | 200 states × 100 policy draws |

Pilot states: every gap at most 10 mm (largest 9.97 mm), start steps 10 to 90, each state at or
after its start step. Contact rate by kind: aimed 71% to 80%, uniform 43% to 51%. Phase 0's smoke
states had no skipped seeds either.

**Replay (step 2, reported).** `replay_check.py --check data/main/pilot_states.npz` in the image:
5,000 of 5,000 one-step results bit-exact (`pilot_states_replay.json`, sha256
`3e590ee1e1feaa32c1d3c91250705118f6971769e91daf1a6414a2d9460dcb1c`).

## Pilot grid (step 3)

`pilot.py grid --workers 8` on the host. `settings.json` sha256
`7eccc67dee8e0925502e74a9b127136cc1607dfd51fe563aed6af6d17bb3a5f6`. Choices by validation loss
only; each checked by hand against the *Pilot rule*.

| Architecture | Chosen (Adam) | Budget | Plateau points of the chosen seeds | Lowest-scoring setting (median best L_obs) | Settings tied (within 5%) | Grid extended | Chosen seeds without plateau |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `mlp` | lr 3e-4, batch 256 | 73,500 | 44,000, 49,000, 41,000 | 3e-4 / 1,024 (0.0507) | 3 | no | none |
| `gru` | lr 3e-4, batch 256 | 105,000 | 56,000, 70,000, 60,000 | 3e-3 / 1,024 (0.0531) | 5 | to 1e-2; lowest unchanged | none |
| `mlp-ms` | lr 3e-4, batch 1,024 | 100,500 | 64,000, 60,000, 67,000 | 3e-4 / 1,024 (0.0585) | 2 | no | none |
| `contact_wm` | lr 3e-3, batch 1,024 | 136,500 | 57,000, 60,000, 91,000 | 1e-2 / 1,024 (0.00857), at the edge of the extended grid | 2 | to 1e-2 | none |

- `gru`'s grid was extended because its lowest-scoring setting had the largest learning rate,
  although the tie rule then chose 3e-4: step 4 of the rule applies to the lowest-scoring setting,
  not the chosen one.
- Seeds that did not plateau by 100,000 updates are all in settings that were not chosen: the
  1e-4 rows of `gru`, `mlp-ms` and `contact_wm`, and some `contact_wm` rows at 3e-4. No budget is
  set by the cap.
- `contact_wm`'s median best L_obs is about 6 times lower than the other architectures' (0.0087
  against 0.051 to 0.059), as Phase 0 found and DESIGN.md expects.

## Pilot confirmation (step 4)

`pilot.py confirm --workers 8`. `confirmation.json` sha256
`4c7512db7ceeae56a0415d51370ce9645837b62acc9522e49aa78a4a3aa81fc0`. The first attempt trained the
six control runs and failed when the scorer loaded the first checkpoint; nothing was scored. The
fix is amendment `2026-11-26-2.md`; the rerun skipped the trained runs and scored them.

| Quantity | Value |
| --- | --- |
| Pilot V\* (1.1 × largest median best L_obs) | 0.0644, set by `mlp-ms` (median 0.0585) |
| L_mlp | 0.0529 |
| D_obs (validation, 100 policy draws) | 0.0652 |
| D_int (push interventions on the pilot states) | 0.533 |
| Pilot p | 0.4 (0.0529 + 0.16 × 0.0652 = 0.0633 ≤ 0.0644) |
| Matched L_obs relative to V\*, `mlp-twin` | +0.3%, +0.5%, −0.8% (3 seeds kept) |
| Matched L_obs relative to V\*, `mlp-shortcut` | +5.0%, +4.7%, +5.4% (1 seed kept; the screen allows 5%) |
| E_int (push), `mlp-twin` / `mlp-shortcut` | 0.461 / 0.506 |
| Relative gap, predicted / observed | 18.5% / 9.8% |
| Welch's interval | not defined: one `mlp-shortcut` seed passed the match screen, so the per-seed variance is undefined (`ci` and `df` are NaN, p = 1.0) |

`mlp-shortcut`'s best L_obs (about 0.0676) is above what the rule's arithmetic expects
(0.0633): its extra loss over `mlp` was 0.0147 against p² · D_obs = 0.0104. The derivation assumes
the best possible predictor; a trained network falls short of it. As DESIGN.md says: this reveals
one absolute E_int of an `mlp`-like model, and nothing about how `mlp`, `gru`, `mlp-ms` and
`contact_wm` compare. The final p is computed afresh from the main runs, and the rule stays as
written.

**No main model's intervention error was computed** in this milestone, by any script. The only
intervention error computed is the confirmation's, of `mlp-shortcut` and `mlp-twin` on the pilot
states.

## Scoring rehearsal

`scripts/rehearse_scoring.py` (new, not hashed) runs `freeze.py --stage 2`, the recording of test
states and `score.py`, unchanged, in a scratch copy of the repo, with an untrained checkpoint
schedule for all 240 runs and 40 of Phase 0's smoke states as stand-in test states. It found that
recording the test states changed the manifest freeze 2 hashes, so `score.py` refused to run
(fixed by amendment `2026-11-26-3.md`). After the fix, in the run order, `score.py` exits 0 and the
result has no format error (about 75 seconds). Untrained models only, so no model's intervention
error after training was computed; the gate readings in the copy carry no information.

## Compute

| Step | Wall time | CPU time |
| --- | --- | --- |
| Data (6 splits, image, 8 workers) | about 2 minutes | script times 13.2, 1.3, 11.7, 4.4, 54.2, 1.3 s |
| Replay of the pilot states | 7 s | |
| Grid (108 runs) | 7 h 1 min | 43.3 h of training (`train_seconds`); 42.7 h user and system |
| Confirmation | 3 min (training) + 41 s (scoring) | |

Grid runs per architecture: `mlp` 24 runs, 1.45 CPU-h; `gru` 30, 15.94; `mlp-ms` 24, 7.30;
`contact_wm` 30, 18.57. The grid's runs used the 8 workers about 6.2 times in parallel.

**Projection for main training (step 5).** Time per update from the three grid runs at each
architecture's chosen setting (wall time per update, evaluation included, under 8 parallel
workers), times the chosen budget and the number of runs; small and large widths scaled from
medium with Phase 0's per-width timings (`results/phase0/milestone2/attempt1/REPORT_milestone2.md`,
*Compute*); `blind` and `mlp-shortcut` at `mlp`'s setting.

| Architecture | ms per update | Minutes per medium run | Medium, 40 runs (CPU-h) | Small, 5 (CPU-h) | Large, 5 (CPU-h) |
| --- | --- | --- | --- | --- | --- |
| `mlp` | 3.1 | 3.8 | 2.6 | 0.2 | 1.2 |
| `gru` | 18.8 | 32.8 | 21.9 | 1.1 | 10.1 |
| `mlp-ms` | 32.8 | 55.0 | 36.6 | 2.4 | 16.3 |
| `contact_wm` | 31.9 | 72.6 | 48.4 | 5.8 | 8.3 |
| `blind`, `mlp-shortcut` (40 runs) | | | 2.6 | | |

Total: **157.5 CPU-hours (6.6 CPU-days) with the large width, 121.6 (5.1) without**, against
DESIGN.md's "roughly five to eight days of CPU". The pilot confirms the estimate, so the large
width is kept. At the grid's parallel use on this machine that is about 25 hours of wall time.
The per-width scaling uses Phase 0's single-run timings at its own batch size and is the least
certain part.

## Commands

```bash
uv sync --locked --extra models
docker image inspect physical-intervene --format '{{.Id}}'
img python scripts/make_data.py --split S --workers 8   # S: train, validation, pilot, pilot_states,
                                                         # validation_policy_mean, pilot_states_policy_mean
img python scripts/replay_check.py --check data/main/pilot_states.npz > results/pilot/pilot_states_replay.json
uv run --extra models python scripts/pilot.py grid --workers 8
uv run --extra models python scripts/pilot.py confirm --workers 8      # failed at scoring; amendment 2
uv run --extra models python scripts/pilot.py confirm --workers 8      # scored
uv run --extra models python scripts/rehearse_scoring.py --out DIR     # before and after amendment 3
```

Checks at commit 8c4bad2:

```
$ uv run ruff check .
All checks passed!
$ uv run ruff format --check .
68 files already formatted
$ uv run --extra models pytest -q
139 passed in 19.19s
$ python scripts/check_docs.py
(exit 0)
```
