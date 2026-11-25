# Phase 0 report (smoke test)

The record of Phase 0 (SMOKE_TEST.md, *Record*), with `results/phase0_smoke.json`. Phase 0 checks
the pipeline in two halves that were never joined: the training side with real architectures
trained on smoke data, and the intervention side with stand-ins only. No trained model was scored
on interventions. Each milestone's report gives the details.

| Milestone | Current record | Report |
| --- | --- | --- |
| 1: simulator, behaviour policy, data, interventions, environment checks | `milestone1/attempt2/` | `REPORT_attempt2.md`, `REPORT_amd64.md` |
| 2: data pipeline, models, training-side checks | `milestone2/attempt1/` | `REPORT_milestone2.md` |
| 3: scorer, gates, control readings, repeated pairs, format, with stand-ins | `milestone3/attempt1/` (attempt 0: 10 seeds, and the explorations of G1's interval) | `REPORT_milestone3.md` |

## Open issues

| # | Issue | Status | What it would affect |
| --- | --- | --- | --- |
| 1 | Single-draw `noisy-a` vs `noisy-a'`: G1 *fail* as expected, but the interval excludes zero (p = 0.013), failing the check's old wording | resolved: SMOKE_TEST.md now checks the interval's calibration on the repeated twins (excludes zero in at most 10% of 200 draws; observed 7%) | none on the design (milestone 3, issue 1) |
| 2 | G1's interval was too narrow when seeds differ a lot on pushes (10 seeds, bootstrap): with no true difference, G1 read *pass* in 4% to 7% of runs at a 20% to 40% seed spread | resolved: DESIGN.md now uses 20 seeds and Welch's t-test over the seeds; attempt 1: twins separate in at most 5.0% up to a 40% spread, interval covers the known gap in 95% to 100% | none |
| 3 | No run met the plateau criterion at the placeholder budget | expected | the pilot's budget (milestone 2, issue 2) |
| 4 | `constvel` scored on smoke interventions as a fixed rule; it was not named in SMOKE_TEST.md's stand-in list | resolved: the list now names it (and says `knn` is not scored) | none |
| 5 | New result fields for the two secondary analyses added in milestone 2 and for `contact_wm`'s asymmetric reading | accepted (in RESULTS_FORMAT.md) | none |
| 6 | Single-draw `noisy-a` vs `noisy-b` (attempt 1): G1 *pass* as expected, but its interval [0.2034, 0.2329] misses the known gap 0.2020; over 200 repeated draws coverage is 97% | resolved: reworded as issue 1; coverage now checked on the repeated `noisy-b` and `noisy-c` draws (at least 90% of 200; observed 97% and 97.5%) | none |
| 7 | Untrained networks' push error | resolved: the push std-MSE of untrained networks was computed in attempt 0 and in the first runs of attempt 1, to exercise the model path; it was never reported, and was deleted from attempt 0's record and dropped from the code (attempt 1 was rerun without it). While deleting it, a file search printed attempt 1's stored values to the screen once | none |

## Values Phase 0 fixes for freeze 1

d_push = 2 mm and θ_push = 1° (defaults kept: no gap between grazes and pushes; 39.5% of contact
interventions are grazes); ε_near = 0.05; smoke seeds of attempt 2 (episodes 3,600,000 to
3,600,049, coverage pool 3,700,000 to 3,700,999, test-like states 3,800,000 to 3,800,199).

## Checks

| Check | Result |
| --- | --- |
| Environment (milestone 1, attempt 2) | every gated check passes: bit-exact replay (local, pinned image, real amd64), pusher can move the box, push rate (92.5% of states, 9.3 per state), intervention coverage (T = 0.35), shadow actions |
| Training side (milestone 2, attempt 1) | every check passes: data, wiring, overfit, losses fall, beats baselines, matching, budget, reproducibility |
| Intervention side (milestone 3, attempt 1: 20 seeds, Welch's t-test) | every check passes (after the rewording in open issue 6). Repeated pairs: twins separate in 0% of 200 draws (≤ 5%) and their interval excludes zero in 6.5% (≤ 10%); `noisy-c` 92%, `noisy-b` 100% (≥ 95%); known gap covered in 97% and 97.5% (≥ 90%) |

The twin false-alarm rate to quote beside a G1 verdict, from stand-ins (attempt 1): 0% at a 10% seed
spread on pushes, 0.5% at 20%, 5.0% at 40%.

## Reruns

| Milestone, attempt | Reason | Outcome |
| --- | --- | --- |
| 1, 0 | first run, ε_near 0.2 | intervention coverage failed (T = 0.066, needs 0.25) |
| 1, 1 | new smoke seeds; coverage measured as action novelty given the state; ε_near 0.05 | every check passed |
| 1, 2 | new smoke seeds; push label against the no-contact reference | every check passed; replay also on real amd64 |
| 2, 0 | first run, with `rssm`, checkpoints every 100 updates | `rssm` set V\* and failed two checks; `contact_wm` skipped the band around V\* |
| 2, 1 | `mlp-ms` in place of `rssm`; finer checkpoint schedule | every check passed |
| 3, 0 | first run of the intervention side (10 seeds, bootstrap over states and seeds) | every check passed after two SMOKE_TEST.md rewordings; G1's interval too narrow at large seed spreads |
| 3, 1 | design change: 20 seeds, Welch's t-test over the seeds | as above |

## Every quantity computed

**Training side** (real architectures, smoke data, recorded actions only):

- data checks: alignment, target and angle wrap, scaler from the training split only, split overlap;
- baselines `copy`, `constvel`, `knn`: L_obs on validation transitions;
- training loss and L_obs (with corner RMSE) at every checkpoint of every learning run;
- L_obs on validation transitions split by contact label and box motion (one ad hoc check);
- made-up-target and overfit training losses; `contact_wm`'s learned physical values;
- `mlp-shortcut`'s swapped share (controls trained, not scored);
- V\*-like level, matched checkpoints, exclusions, match screen; plateau criterion;
  reproducibility; timings;
- exploration: `pnn` and `rssm` variants, L_obs and made-up targets.

**Intervention side** (simulator, stand-ins and untrained networks only):

- simulator only: replay of the 25 actions at each smoke state, push thresholds and labels, push
  rate, intervention coverage, policy log-density, shadow actions, D_obs and D_int;
- stand-ins (`oracle`, `copy`, `constvel`, noisy families): L_obs, std-MSE per stratum, corner and
  angle errors, logged-action reference;
- untrained networks (`trained: false`): L_obs; their predictions at the smoke interventions
  checked for shape only (push std-MSE computed earlier and deleted, open issue 7);
- with stand-ins only: V\*, matching, gaps, intervals, p, Holm, G0.2, G0.3, G1, control readings,
  G2, `best_push_gap`, `matched_only`, gap along training, both new secondary analyses, per-seed
  view, repeated-pair rates; the stand-in result file and its format check.

**Never computed:** the intervention error of any trained model; any gap, G1, G2 or control-pair
result between trained models; any measure of a trained model's reliance on the action with
real-style data; anything on pilot seeds or test states.
