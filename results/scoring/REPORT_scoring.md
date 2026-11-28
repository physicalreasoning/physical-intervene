# Scoring report: freeze 2, test states, scoring

Run after training, 2026-11-27 to 2026-11-28, on `main` at d44d025 (freeze 1: commit 117ae28,
`frozen/design.json` sha256 `f55415351f971d7726a81708ef2b3bd0508e6f12b30c7916dad564224999d140`,
with four amendments; freeze 2: commit 12f4560). Records: `results/intervention_results.json`
(sha256 `07747f85d02660031821b847c3c4d4541d9577f27526dace57814b50987721b9`) and
`results/test_replay.json` (sha256 `0a93501912af46044d3cbb2388a3be5c81a033adcc6f00ab8a8635452b3b4f83`).
Every reading below is quoted from the result as `score.py` wrote it; nothing was recomputed,
re-thresholded or dropped.

## Open issues

| Issue | Status | What it would affect |
| --- | --- | --- |
| `contact_wm`'s matched checkpoint is at 210 to 230 updates of its 136,500 (every seed, and its twin's): its validation loss falls through V\* almost at once, and its best L_obs (0.0077) is about 8 times below V\* (0.0649). The three pairs that pass G1 all compare this early `contact_wm` with general-purpose models trained for 10,000 to 68,000 updates | reported as the result has it; the matching rule is the frozen one (DESIGN.md, *Matching observational loss*) | how the G1 *pass* is read: the gap is not a learning-speed effect (`matched_only` false; the gap grows at best checkpoints and along training), but the matched `contact_wm` is a barely trained model |
| The positive control did not separate: `mlp-shortcut` is 4.7% worse than `mlp` on push interventions, with an interval excluding zero, below G1's 20% (pilot: 9.8% observed, 18.5% predicted) | reported, as DESIGN.md (*Controls*) says | a G1 *fail* would have read *inconclusive: test not shown sensitive*; the reading is *pass*, so it is unaffected. G1's ability to find a 20% gap between trained general-purpose models is not shown |
| No pair of general-purpose models (`mlp`, `gru`, `mlp-ms`) meets G1's 20% threshold: their gaps are 1.5% to 5.5%, each with an interval excluding zero | reported | G1 passes on the `contact_wm` pairs only |
| The first scoring attempt was stopped after 78 minutes and restarted with `OMP_NUM_THREADS=1`: `score.py` leaves torch at its default thread count (16 per worker on this machine), so 14 workers ran 224 threads on 32 and the load reached 195. It wrote nothing; the restart used the same command, code and inputs | done; every number here is from the second attempt | none on the readings. Floating-point sums may differ in the last digits from a run with 16 threads per worker. Any later scoring run on this machine should set `OMP_NUM_THREADS=1` |
| 3 of 1,003 test seeds yielded no test-like state and were skipped (147117219545773, …6081, …6253); the pilot and smoke states had none | reported, as DESIGN.md (*Data*) says | none: 1,000 states as designed |
| G2 reads *refuted* (ρ 0.88), and over the general-purpose models only *inconclusive* (ρ 0.74): `contact_wm` is low on both axes | reported; the gated reading is over all 120 runs (DESIGN.md, *G2*) | how G2 is written up |
| `ruff` cannot run on the scoring machine (the locked ruff is a generic Linux binary, which NixOS does not start) | ruff was run on the Mac on the same commit | none |

## Environment

| Step | Machine | Python, packages | Image |
| --- | --- | --- | --- |
| Rehearsal | Apple M1 Pro, macOS 26.6.2 | Python 3.14.7, torch 2.15.0, numpy 2.5.5, MuJoCo 4.0.1 | n/a |
| Freeze 2, scoring | shared 16-core AMD Ryzen 9 5950X (32 threads, 62 GB), NixOS, Linux 6.18.62 | Python 3.15.2 (Nix, through a `uv` wrapper), torch 2.15.0+cu132 on CPU, numpy 2.5.5, MuJoCo 4.0.1; `uv.lock` unchanged | n/a |
| Test states, replay | Apple M1 Pro, Docker `linux/amd64` | Python 3.12.16, MuJoCo 4.0.1, numpy 2.5.5 (in the image) | `sha256:547fa5d041926f20b907bd990e748e0076543d2cee32598224d2db43b71233b1`, equal to freeze 1's |

Before freeze 2 both checkouts were at `main` d77bbd2, clean, `check_frozen(1)` with 4 amendments,
phase `training`; all 7 data files matched `data/main/manifest.json`; 240 run folders, each with a
`run.json` (220 main and controls, 20 `mlp-shortcut`), 49,730 checkpoints, 33 GB. Every file
moved between the machines with `rsync`, its sha256 checked on both sides.

## Rehearsal

`rehearse_scoring.py` (untrained models, Phase 0 smoke states) in a scratch copy at d77bbd2: exit
0 in 78 s; `score.py` exited 0 and the result conformed to the format. The copy was deleted.

## Freeze 2

`freeze.py --stage 2` on the scoring machine, exit 0 in 57 s (2026-11-27 21:02 UTC).

| | |
| --- | --- |
| `frozen/models.json` sha256 | `85cd6503eb46223cef4a2a61aaba0e1695734ac5a116e94642a5d087db4a5091` |
| Commit | 12f4560 on `main` (merge d44d025), posted on forward#5 |
| First test seed | 147117219545542 |
| `n_runs`, `p`, `large_width_dropped` | 240, 0.4, false |
| V\*, D_obs, L_mlp | 0.06491, 0.06465, 0.05247, equal to `results/training/p.json` |
| Hashed | 11 data and record files; every file of the 240 runs (49,730 checkpoints, 49,730 checkpoint metadata files, 240 run records) |

After the merge, `check_frozen(2, checkpoints=True)` re-hashed every run file in 49 s on the
scoring machine and passed; both checkouts report the same `models_sha256` and test seed.

## Test states and replay

`make_data.py --split test_states --workers 8` in the image (9 s), then `replay_check.py --check`
in a fresh container (12 s).

| | |
| --- | --- |
| `test_states.npz` | `9375be831b9c29c3b3f2d529cdc45252ddcbc24ae10164f2079b1ece9be89f86` |
| `test_history.npz` | `1e054a40092131c0f5850657252c16ec5579a60048350ab7207d806c265865a0` |
| `manifest_test.json` | `60bbd6b70b7841d70abfbbb2093ee5d8eb7d52e072739066f6eaa3ec4af18fc1`; machine `x86_64`, `epsilon_near` 0.05 |
| States | 1,000, seeds 147117219545542 to 147117219546544, 3 skipped |
| Gaps, start steps | every gap at most 10 mm (largest 9.96, median 2.18); start steps 10 to 90 |
| Replay | 25,000 of 25,000 one-step results bit-exact |
| `check_frozen(2)` afterwards | passes (`manifest.json` unchanged) |

Labels by kind (simulator only), with the pilot states beside them:

| Kind | Contact rate | Push / graze / no contact | Pilot contact | Pilot push / graze / no contact |
| --- | --- | --- | --- | --- |
| Aimed (12 per state) | 79.7% | 51.9 / 27.8 / 20.3% | 77.2% | 49.8 / 27.4 / 22.8% |
| Uniform (12 per state) | 50.7% | 21.6 / 29.1 / 49.3% | 46.2% | 21.1 / 25.1 / 53.8% |
| Logged | 73.1% | 50.7 / 22.4 / 26.9% | 74.0% | 52.5 / 21.5 / 26.0% |

Of the 24,000 interventions, 8,817 are pushes (the gated stratum), 6,828 grazes and 8,355 without
contact; 957 of the 1,000 states have at least one push.

## Readings

As `score.py` wrote them (`primary.gates`):

| Key | Value |
| --- | --- |
| `gate_g0_valid` | `true` |
| `gate_g1_reading` | `pass` |
| `gate_positive_control_pass` | `false` |
| `gate_negative_controls` | `mlp` clean, `gru` clean, `mlp-ms` clean, `contact_wm` clean |
| `gate_g1_passing_pairs` | (`mlp`, `contact_wm`), (`gru`, `contact_wm`), (`mlp-ms`, `contact_wm`) |
| `gate_g2_reading` | `refuted` |

- **G0, valid.** The simulator replays every test action exactly, every main model predicts
  ordinary transitions better than copying the current state, and a model that never sees the
  action is much worse under intervention than `mlp`. In numbers: replay 25,000 of 25,000; every
  main model beats `copy` (L_obs 0.965) at matched and best checkpoints; `blind` against `mlp` on
  push at best checkpoints: gap 0.299 std-MSE, 95% interval [0.295, 0.303], relative gap 88%.
- **G1, pass.** At the same validation loss, the structured model `contact_wm` predicts what
  happens when the pusher is moved somewhere new much better than each general-purpose model: their
  push error is 41% to 48% higher than its. Among the general-purpose models the differences are
  real but small (1.5% to 5.5%), below the 20% the design requires. In numbers: see the table below;
  all six pairs pass the match screen; Holm over 6.
- **Positive control, not passed.** `mlp-shortcut`, built to ignore the action 40% of the time, is
  worse than `mlp` under intervention, but only by 4.7%, not the 20% G1 requires, so this control
  does not show G1 can detect a gap of that size. It would matter for a G1 *fail*; G1 passed.
- **Negative controls, clean.** Retraining each architecture on new seeds never produces a pair
  G1 separates: the largest twin gap is 1.5% (`gru`), with an interval excluding zero but far
  below 20%.
- **G2, refuted.** Across the 120 main runs at their best checkpoints, a run with lower validation
  loss nearly always has lower push error: Spearman ρ 0.883, 95% interval [0.810, 0.929]
  (corner RMSE: 0.854 [0.769, 0.911]); `decoupled_by_interval` false. By the design's rule (ρ ≥
  0.80), observational loss ranks intervention error well here, which refutes the position for this
  task. Over the 90 general-purpose runs only (not gated): ρ 0.740 [0.594, 0.843], reading
  *inconclusive*.

### G1 pairs (matched checkpoints, push stratum)

Gap is challenger minus reference E_int in std-MSE; the reference has the lower E_int.

| Challenger, reference | Match (L_obs diff) | Gap [95% interval] | Relative gap | Holm p | States-only interval | Best-checkpoint gap | Along training |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `mlp`, `gru` | yes (0.02%) | 0.0058 [0.0001, 0.0114] | 1.5% | 0.046 | [−0.0033, 0.0153] | 0.0087 [0.0030, 0.0143] | closing |
| `mlp`, `mlp-ms` | yes (0.15%) | 0.0200 [0.0148, 0.0252] | 5.5% | < 1e-5 | [0.0113, 0.0288] | −0.0240 [−0.0283, −0.0197] | persistent |
| `gru`, `mlp-ms` | yes (0.13%) | 0.0142 [0.0097, 0.0187] | 3.9% | < 1e-5 | [0.0038, 0.0254] | −0.0326 [−0.0383, −0.0270] | persistent |
| `mlp`, `contact_wm` | yes (0.04%) | 0.1255 [0.1205, 0.1304] | 48.5% | < 1e-5 | [0.0846, 0.1649] | 0.2983 [0.2952, 0.3014] | persistent |
| `gru`, `contact_wm` | yes (0.02%) | 0.1197 [0.1155, 0.1239] | 46.3% | < 1e-5 | [0.0807, 0.1597] | 0.2896 [0.2847, 0.2946] | persistent |
| `mlp-ms`, `contact_wm` | yes (0.11%) | 0.1055 [0.1019, 0.1091] | 40.8% | < 1e-5 | [0.0667, 0.1450] | 0.3223 [0.3192, 0.3254] | persistent |

- `matched_only` is false for every pair. Each `contact_wm` pair's `asymmetric_reading` is
  `evidence_for_structure`. On all contact interventions (graze and push) the `contact_wm` gaps
  are 20% to 24%.
- At best checkpoints `mlp-ms` has the higher push error against `mlp` and `gru` (negative gaps):
  the order of those two pairs reverses between V\* and full training.
- Without the architecture that sets V\* (`mlp-ms` left out, V\* 0.0613, Holm over 3), the reading
  is also *pass*, on (`mlp`, `contact_wm`) and (`gru`, `contact_wm`).

Seed means of E_int at matched checkpoints (from `seed_checkpoints`): `mlp` 0.384, `gru` 0.379,
`mlp-ms` 0.364, `contact_wm` 0.259; at best checkpoints 0.339, 0.330, 0.363, 0.040. Baselines on
push (std-MSE): `copy` 2.316, constant velocity 0.787, nearest neighbour 0.642.

## Matching

V\* 0.06491, set by `mlp-ms`. Mean matched L_obs and the updates of the matched checkpoints:

| Family | Excluded seeds | Seeds left | Mean matched L_obs | Matched at updates |
| --- | --- | --- | --- | --- |
| `mlp` | none | 20 | 0.06495 | 10,000 to 13,000 |
| `gru` | none | 20 | 0.06494 | 27,000 to 38,000 |
| `mlp-ms` | none | 20 | 0.06485 | 23,000 to 68,000 |
| `contact_wm` | none | 20 | 0.06492 | 210 to 230 |
| `mlp-twin` | none | 20 | 0.06463 | 10,000 to 16,000 |
| `gru-twin` | none | 20 | 0.06490 | 24,000 to 34,000 |
| `mlp-ms-twin` | none | 20 | 0.06493 | 23,000 to 99,000 |
| `contact_wm-twin` | none | 20 | 0.06459 | 210 to 230 |
| `mlp-shortcut` | 55, 143 | 18 | 0.06682 | 29,000 to 42,000 |
| `blind` | all 20 (never within 5% of V\*) | 0, dropped | n/a | 16,000 to 22,000 |

Every main pair and every control pair passed the match screen (largest difference 2.9%,
`mlp-shortcut` against `mlp`). `blind` takes part only in G0.3, at best checkpoints. Every medium
run has a plateau point (`met_plateau_criterion` true); 8 small runs do not, as the training
report said.

## Controls

| Pair | Role | Match (L_obs diff) | Gap [95% interval] | Relative gap | p | States-only interval | Separates |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `mlp-shortcut`, `mlp` | positive | yes (2.9%) | 0.0181 [0.0127, 0.0234] | 4.7% | 7.6e-8 | n/a | no |
| `mlp`, `mlp-twin` | negative | yes (0.49%) | 0.0008 [−0.0057, 0.0072] | 0.2% | 0.81 | [−0.0018, 0.0034] | no |
| `gru-twin`, `gru` | negative | yes (0.06%) | 0.0056 [0.0006, 0.0106] | 1.5% | 0.030 | [0.0025, 0.0087] | no |
| `mlp-ms-twin`, `mlp-ms` | negative | yes (0.12%) | 0.0020 [−0.0038, 0.0077] | 0.5% | 0.50 | [−0.0014, 0.0054] | no |
| `contact_wm-twin`, `contact_wm` | negative | yes (0.52%) | 0.0001 [−0.0035, 0.0037] | 0.03% | 0.96 | [−0.0005, 0.0007] | no |

The positive control's relative gap (4.7%) is below the pilot's (9.8% observed, 18.5% predicted).
The `gru` twin pair has an interval excluding zero and p 0.030, and the states-only interval also
excludes zero, but its 1.5% is far from 20%, so it does not separate.

## Compute

| Step | Machine | Wall time |
| --- | --- | --- |
| Rehearsal | Mac | 78 s |
| Freeze 2 | scoring machine | 57 s |
| `check_frozen(2, checkpoints=True)` | scoring machine | 49 s |
| Test states, replay | Mac, image | 16 s, 12 s |
| Scoring, first attempt (stopped, nothing written) | scoring machine | 78 min (23:38 to 00:56) |
| Scoring, second attempt | scoring machine | 86 min (00:56:23 to 02:22:28, `wall_seconds` 5,164) |

Estimate: 1 to 3 hours. The second attempt took 86 minutes with 14 workers at one thread each.
The first attempt ran with torch's default threads (16 per worker): load near 195 on 32 threads,
and its progress was far slower. A timing test with untrained checkpoints and smoke states on the
scoring machine, under that load, showed one thread per worker 1.5 to 6.3 times faster per
checkpoint than 16 (`mlp` 0.055 against 0.23 s, `contact_wm` 0.12 against 0.77 s, `gru` 0.58
against 1.25 s, `mlp` large 0.33 against 0.52 s, at 200 states). On the Mac the same test differed
little (1.2 to 1.5 times).

## Commands

On the Mac, from the checkout:

```bash
uv run --extra models python scripts/rehearse_scoring.py --out <scratch folder outside the repo>
img python scripts/make_data.py --split test_states --workers 8
img python scripts/replay_check.py --check data/main/test_states.npz > results/test_replay.json
```

On the scoring machine, from `~/physical-intervene`:

```bash
uv run --extra models python scripts/freeze.py --stage 2 2>&1 | tee ~/pi-freeze2.log
uv run python -c "from physical_intervene import seal; seal.check_frozen(2, checkpoints=True)"
# tmux session pi-score; first attempt without OMP_NUM_THREADS, stopped
OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 uv run --extra models python scripts/score.py \
    --replay results/test_replay.json --workers 14 2>&1 | tee ~/pi-score.log
```

`score.py` exited 0 and printed the readings above; `results_format.check_conformance` on the
result returns no error. `frozen/models.json` moved from the scoring machine to the Mac, and the
test files and replay the other way, with `rsync`; the result came back the same way. Every sha256
matched on both sides.

## Checks

On the scoring machine, at d44d025 with only the two result files untracked:

```
$ uv run --extra models pytest -q
139 passed, 1 skipped in 24.99s
$ uv run python scripts/check_docs.py
(exit 0)
```

The skipped test needs Phase 0 smoke data, which is only on the Mac. On the Mac, same commit:

```
$ uv run ruff check .
All checks passed!
$ uv run ruff format --check .
71 files already formatted
$ uv run --extra models pytest -q
140 passed in 19.28s
$ uv run python scripts/check_docs.py
(exit 0)
```
