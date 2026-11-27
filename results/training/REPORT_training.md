# Training report: main runs, controls, p and `mlp-shortcut`

Run after the pilot, 2026-11-26 to 2026-11-27, on `main` at 820cd27 (freeze 1, commit 117ae28,
`frozen/design.json` sha256 `f55415351f971d7726a81708ef2b3bd0508e6f12b30c7916dad564224999d140`,
with amendments `2026-11-26.md`, `-2.md` and `-3.md`). Records: `p.json` in this folder. The runs
and every scheduled checkpoint are in `data/runs/main/` on the training machine (gitignored, 33 GB;
freeze 2 hashes them there). Every number below comes from validation loss and the simulator only:
**no intervention error of any model was computed** in this milestone, by any script.

## Open issues

| Issue | Status | What it would affect |
| --- | --- | --- |
| The scorer reports `met_plateau_criterion` as `null` for every run whose budget is not a multiple of 1,000 updates (`mlp`, `mlp-ms`, `contact_wm`, their twins, `blind`, `mlp-shortcut`): it tests the criterion only at the last checkpoint, which is not a tested point for those budgets | amendment `2026-11-27.md` written and committed on its own branch; not yet merged or posted on forward#5. The training machine's checkout must include it before freeze 2 | that one field of the result; no gate, reading, matching step or exclusion reads it, and no run is retrained |
| 2 of 20 `mlp-shortcut` seeds (55, 143) never come within 5% of V\* (best L_obs +5.4%) | reported, as DESIGN.md (*Controls*) says: a miss is reported, not corrected | they are excluded at scoring; 18 seeds remain, above the 12 needed |
| The positive control may not separate: at p = 0.4 the pilot's predicted relative gap was 18.5%, below G1's 20%, and `mlp-shortcut`'s extra L_obs is again 1.40 times p² · D_obs | accepted as written (DESIGN.md, *Controls*) | if it does not pass, a G1 *fail* reads *inconclusive: test not shown sensitive*; a G1 *pass* is unaffected |
| 8 small-width runs never met the plateau criterion (`mlp` small and `gru` small, 4 of 5 each); the budget is set at medium width and shared across widths (*Pilot rule*, step 6) | reported | small runs enter only G2 and the secondary analyses, as points that may sit above their own floor |
| Training used 295 CPU-hours against the expected 204; the large width cost 1.8 to 2.7 times its projection | reported; wall time 21.4 h, inside the expected 19 to 22 h | none on the design (the compute fallback was decided at the pilot) |
| The checkpoints exist only on the training machine | open: where freeze 2 and scoring run | freeze 2 hashes the checkpoints where they are; moving them means checking every hash after the copy |
| `ruff` cannot run on the training machine (the locked ruff 0.17.5 is a generic Linux binary, which NixOS does not start) | ruff was run on the Mac on the same commit | none: the training machine's checkout has no local change |

## Environment

| | |
| --- | --- |
| Machine | shared 16-core AMD Ryzen 9 5950X (32 threads, 62 GB memory), NixOS 26.11 (Zokor), Linux 6.18.62 |
| Python, packages | Python 3.15.2 (Nix, through a `uv` wrapper); torch 2.15.0+cu132 on CPU, numpy 2.5.5, MuJoCo 4.0.1; `uv sync --locked --extra models`, `uv.lock` unchanged |
| Workers | 14 (another user's jobs used about 0.1 cores throughout) |
| Checks before training | checkout `main` at 820cd27, clean; `seal.check_frozen(1)` passes with 3 amendments, phase `training`; all 7 files in `data/main/` match `manifest.json` |

Every main run, control and twin trained on this machine, so no architecture and its twin differ in
machine. The pilot ran on an Apple M1 Pro with Python 3.14.7; package versions are the same.

## Main runs and controls (`train_runs.py main`)

220 runs, all finished: every run did its full budget, its last checkpoint is at the budget, and no
validation loss is NaN or infinite. Settings and budgets from `results/pilot/settings.json`
(`mlp`, `blind` and `mlp-twin`: Adam 3e-4, batch 256, 73,500 updates; `gru`: 3e-4, 256, 105,000;
`mlp-ms`: 3e-4, 1,024, 100,500; `contact_wm`: 3e-3, 1,024, 136,500).

"Plateau point" counts runs that met the plateau criterion at some checkpoint (DESIGN.md, *Models*,
*Training*). "Holds at end" counts runs where it also holds at the last checkpoint on a multiple of
1,000 updates. Wall time is from the first run's start to the last run's end in each row.

| Family, width | Runs | Wall h | CPU-h | Median best L_obs [min, max] | Plateau point | Holds at end | Median plateau point [max] | Disk GB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `mlp` medium | 20 | 0.5 | 1.8 | 0.0525 [0.0514, 0.0541] | 20/20 | 17/20 | 48,500 [57,000] | 1.5 |
| `mlp` small | 5 | 0.0 | 0.1 | 0.0676 [0.0663, 0.0691] | 1/5 | 1/5 | 72,000 | 0.0 |
| `mlp` large | 5 | 0.9 | 2.7 | 0.0511 [0.0508, 0.0523] | 5/5 | 4/5 | 29,000 [38,000] | 3.7 |
| `gru` medium | 20 | 2.9 | 14.1 | 0.0558 [0.0524, 0.0582] | 20/20 | 20/20 | 61,000 [75,000] | 1.7 |
| `gru` small | 5 | 0.4 | 1.1 | 0.0959 [0.0947, 0.0987] | 1/5 | 0/5 | 91,000 | 0.1 |
| `gru` large | 5 | 5.2 | 24.6 | 0.0517 [0.0490, 0.0520] | 5/5 | 5/5 | 35,000 [50,000] | 4.3 |
| `mlp-ms` medium | 20 | 4.4 | 29.3 | 0.0590 [0.0568, 0.0624] | 20/20 | 20/20 | 59,000 [79,000] | 1.7 |
| `mlp-ms` small | 5 | 0.7 | 2.5 | 0.0787 [0.0761, 0.0812] | 5/5 | 2/5 | 85,000 [96,000] | 0.1 |
| `mlp-ms` large | 5 | 12.4 | 59.9 | 0.0586 [0.0566, 0.0597] | 5/5 | 5/5 | 26,000 [27,000] | 4.2 |
| `contact_wm` medium | 20 | 7.7 | 37.4 | 0.0077 [0.0076, 0.0080] | 20/20 | 16/20 | 66,500 [85,000] | 2.0 |
| `contact_wm` small | 5 | 2.3 | 7.8 | 0.0091 [0.0091, 0.0092] | 5/5 | 2/5 | 81,000 [93,000] | 0.1 |
| `contact_wm` large | 5 | 6.4 | 31.9 | 0.0074 [0.0073, 0.0076] | 5/5 | 5/5 | 60,000 [68,000] | 4.9 |
| `blind` | 20 | 0.2 | 1.6 | 0.1324 [0.1312, 0.1335] | 20/20 | 20/20 | 25,500 [28,000] | 1.5 |
| `mlp-twin` | 20 | 0.2 | 1.5 | 0.0525 [0.0509, 0.0543] | 20/20 | 19/20 | 47,500 [55,000] | 1.5 |
| `gru-twin` | 20 | 2.5 | 15.4 | 0.0545 [0.0519, 0.0564] | 20/20 | 20/20 | 61,000 [84,000] | 1.7 |
| `mlp-ms-twin` | 20 | 4.4 | 29.2 | 0.0589 [0.0554, 0.0602] | 20/20 | 20/20 | 62,000 [74,000] | 1.7 |
| `contact_wm-twin` | 20 | 3.8 | 34.3 | 0.0078 [0.0075, 0.0080] | 20/20 | 13/20 | 63,000 [89,000] | 2.0 |

- Every medium run (main and twin) has a plateau point within its budget. 22 medium main runs and
  12 twin runs have their plateau point above budget / 1.5, that is, later than the pilot's slowest
  chosen seed.
- Runs that never met the criterion: `mlp` small seeds 11, 22, 44, 55 and `gru` small seeds 11,
  22, 44, 55. The best checkpoint is the last one in `gru` small seed 33, `mlp-ms` small seed 55
  and `mlp` small seeds 33 and 55: those runs were still improving at the end of their budget.
- Where the criterion was met but does not hold at the end (for example 4 `contact_wm` medium and
  7 `contact_wm-twin` runs), L_obs improved by more than 1% again over the last 10,000 updates.
- `mlp-ms` large reaches its best L_obs early (median at update 17,000), and its L_obs then rises.
- Each twin's median best L_obs is within 2.5% of its architecture's. `contact_wm` is about 7
  times lower than the other three, as DESIGN.md expects; `blind` is about 2.5 times `mlp`.

## p (`train_runs.py shortcut`)

`p.json` (sha256 `aba98b792b22a0776aed5eec2bf6a05c3babca590659dd45f70f9d98be2b6eae`), by the rule in
DESIGN.md, *Controls*:

| Quantity | Main runs | Pilot |
| --- | --- | --- |
| Medians of best L_obs, medium: `mlp`, `gru`, `mlp-ms`, `contact_wm` | 0.05247, 0.05576, 0.05901, 0.00771 | 0.0529 (`mlp`), 0.0585 (`mlp-ms`) |
| V\* (1.1 × largest median) | 0.06491, set by `mlp-ms` | 0.0644, set by `mlp-ms` |
| L_mlp | 0.05247 | 0.0529 |
| D_obs (validation, `validation_policy_mean.npz`, sha256 `e962b667…4027f39`) | 0.06465 | 0.0652 |
| p | **0.4** (0.05247 + 0.16 × 0.06465 = 0.06282 ≤ 0.06491; at 0.5: 0.06863 > V\*) | 0.4 |
| Positive control | run | |

D_obs is in std-MSE, whose scale comes from the training split; the pilot's came from its own
training data, which may explain the small difference (not checked).

## `mlp-shortcut` against V\* (validation loss only)

20 runs at p = 0.4, 73,500 updates each, `mlp`'s settings; the share of actions replaced in
training is 0.3998 to 0.4003. Every run has a plateau point (34,000 to 45,000 updates).

| | Value |
| --- | --- |
| Median best L_obs | 0.06696 (+3.2% above V\*; range +0.1% to +5.4%) |
| Checkpoint closest to V\* | the best checkpoint in every run (no run gets below V\*); seed mean 0.06698, +3.2% |
| Within 5% of V\* at some checkpoint (the exclusion rule) | **18 of 20**; seeds 55 and 143 miss (+5.36%) |
| Extra L_obs over L_mlp | 0.01449 against p² · D_obs = 0.01034, a ratio of 1.40 (pilot: about 1.4) |

So the `mlp` against `mlp-shortcut` pair is expected to pass the 5% match screen by validation loss
(`mlp` near V\*, `mlp-shortcut` about 3.2% above), with 18 seeds. The scorer decides this after
freeze 2 from the matched checkpoints.

## Compute

| Step | Wall time | Training CPU-h |
| --- | --- | --- |
| Main runs and controls, 220 runs | 21.4 h (2026-11-26 21:37 to 2026-11-27 18:59) | 294.9 (12.3 CPU-days of this machine's cores) |
| p and 20 `mlp-shortcut` runs | 9 min (19:49 to 19:58) | 1.4 |

Expected cost (bix ms per update at the chosen settings, times budget and runs; small and large
scaled as in the pilot report) against actual CPU-hours:

| Architecture (with its controls) | Medium | Small | Large |
| --- | --- | --- | --- |
| `mlp` (with `blind`, `mlp-twin`) | 4.8 / 4.8 | 0.2 / 0.1 | 1.5 / 2.6 |
| `gru` (with twin) | 21.7 / 29.5 | 1.1 / 1.1 | 10.0 / 24.6 |
| `mlp-ms` (with twin) | 49.6 / 58.5 | 3.2 / 2.5 | 22.3 / 59.9 |
| `contact_wm` (with twin) | 69.6 / 71.7 | 8.4 / 7.8 | 11.9 / 31.9 |

Total 204 expected, 295 actual (1.44 times). Medium runs cost 1.0 to 1.4 times their estimate; the
large width 1.8 to 2.7 times. The pilot report had marked the per-width scaling, taken from Phase
0's single-run timings, as the least certain part. The wall time still fell inside the expected
19 to 22 hours: the run order puts the costliest runs first, and the large runs ran alongside
cheaper ones. These are this
machine's core-hours (slower cores than the Mac's for these models); the pilot's 6.6 CPU-days was
measured on the Mac.

Disk: `data/runs/main/` is 33 GB with `mlp-shortcut` (32 GB before), inside the expected 30 to 40 GB.

## Commands

On the training machine, from `~/physical-intervene`:

```bash
git pull                                                  # main, 820cd27, already up to date
uv run python -c "from physical_intervene import seal; print(len(seal.check_frozen(1)['amendments']))"   # 3
sha256sum data/main/*.npz                                 # against data/main/manifest.json
# tmux session pi-main
uv run --extra models python scripts/train_runs.py main --workers 14 2>&1 | tee ~/pi-main.log
# tmux session pi-shortcut
PYTHONUNBUFFERED=1 uv run --extra models python scripts/train_runs.py shortcut --workers 14 2>&1 | tee ~/pi-shortcut.log
```

Both exited 0. `PYTHONUNBUFFERED=1` only makes the log show each run as it finishes (the main
run's log was filled only at the end).

Then `results/training/` was copied to the Mac with `rsync`, and the sha256 of `p.json` checked on
both sides. The progress and summary numbers above were read from each run's `run.json` (file
names, `curve`, `best_l_obs`, `plateau`, `train_seconds`); no checkpoint was loaded.

## Checks

On the training machine, at 820cd27 with no local change:

```
$ uv run --extra models pytest -q
138 passed, 1 skipped in 23.73s
$ uv run python scripts/check_docs.py
(exit 0)
$ uv run ruff check .
Could not start dynamically linked executable: ruff
```

The skipped test needs Phase 0 smoke data, which is only on the Mac. `ruff` does not start on NixOS
(see *Open issues*), so it ran on the Mac on branch `training` (820cd27 plus this folder):

```
$ uv run ruff check .
All checks passed!
$ uv run ruff format --check .
70 files already formatted
$ uv run --extra models pytest -q
139 passed in 19.03s
$ uv run python scripts/check_docs.py
(exit 0)
```
