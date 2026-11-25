# Phase 0, milestone 1, attempt 2: push label against the no-contact reference

Rerun of the environment checks on new smoke seeds after a design change to the push label.
Simulator and scripted policy only: no model of any kind was built, trained or run. Everything
not listed below is as in attempt 1 ([REPORT_attempt1.md](../attempt1/REPORT_attempt1.md)) and
attempt 0 ([REPORT_milestone1.md](../attempt0/REPORT_milestone1.md)).

## Open issues

None blocks milestone 2. Each must be settled, or explicitly accepted, before freeze 1.

| # | Issue | Status | What it would affect |
| --- | --- | --- | --- |
| 1 | **Pusher collision shape.** The pusher collides as a vertical capsule (radius 1 cm), because MuJoCo's box-cylinder collider missed contacts in the linux/amd64 image. | **resolved**: capsule accepted; DESIGN.md and README now say "round pusher" and describe the capsule | none |
| 2 | **Bit-exact replay on real amd64 hardware.** The image ran under Rosetta on Apple Silicon; replay and the collider fix are confirmed only there. | **deferred**: run on an Ubuntu amd64 machine at the end of Phase 0 | ground truth for the test is generated in the image, so this must hold before it is generated |
| 3 | **Where d_push sits under the new label.** Against the reference, 2 mm lies near the median of contact effects, closer to the bulk of pushes than under total motion. The rule keeps it because the only gap in the plot separates no-effect touches from all others, not grazes from pushes. | **resolved**: defaults accepted by the user (d_push 2 mm, θ_push 1°) | none |

**Margins to note (not issues).** None recurs after freeze 1.
- Push rate passes with 92.5% of states having a push, against 90% (mean 9.3, against 6). The
  same label on attempt 1's seeds gives 97.0%.
- Intervention coverage: T = 0.346 against 0.25 (0.346 on attempt 1's seeds with the same label).
- The slowest diagonal push in *Pusher can move the box* moves the box 2.13 mm against 2 mm (scene
  unchanged).

## What changed and why

| | Attempt 1 | Attempt 2 |
| --- | --- | --- |
| Push label | contact, and the box moves ≥ d_push or turns ≥ θ_push in total | contact, and the box's outcome differs from the no-contact reference by ≥ d_push or ≥ θ_push |
| Threshold rule wording | "moved to that gap" | moved only to a gap between grazes and pushes that the defaults cut above; kept when there is no gap |
| ε_near | 0.05 | 0.05 |
| Smoke seeds: episodes, pool, states | 3,300,000+, 3,400,000+, 3,500,000+ | 3,600,000+, 3,700,000+, 3,800,000+ |

1. **Push label.** The *no-contact reference* is the same step, from the same saved state, with
   contacts disabled, so the box moves only under its own momentum and table resistance.
   - Under the earlier rule, a light touch on a box that was already sliding counted as a push.
     On attempt 1, 25.6% of contact interventions were labelled push by the earlier rule and graze
     against the reference, never the other way round.
   - Changed in DESIGN.md (*Environment*, *Push label*; *Interventions*, strata and *Push
     thresholds*) and SMOKE_TEST.md (*Push thresholds*, *Intervention coverage*).
   - Behaviour episodes now store each step's reference, computed after the episode from the saved
     states, so the transitions are unchanged (the shadow-action byte check covers this).
2. **Threshold rule wording.** The earlier text assumed a gap between grazes and pushes that the
   data do not show. The rule now states what happens without one: the defaults are kept.

**Same seeds, new label.** Attempt 1's stored data, rescored with the new label
(`../attempt1/rescored_push_label.json`):
- push rate 97.0% of states, mean 9.0;
- coverage T rises from 0.278 to 0.346;
- graze share of contacts rises from 18% to 44%.

## Results

| Check | Result |
| --- | --- |
| Bit-exact replay | **pass**: 5,000 of 5,000 locally (fresh process) and in the image (`sha256:6b7201f3...5dd810`, generate and check in separate containers, Rosetta) |
| Push thresholds | defaults accepted (see *Open issues*, row 3): 13.2% of contacts change nothing; the effect's quantiles (5/25/50/75/95%) are 0, 0.60, 2.26, 4.30, 7.66 mm; graze share of contacts 0.395 |
| Pusher can move the box | **pass** (scene unchanged; from rest the reference is zero, so the new label changes nothing here) |
| Push rate | **pass**: 92.5% of states have a push, mean 9.29 of 24. Shares of all interventions: no contact 0.360, graze 0.253, push 0.387. Aimed: 0.205, 0.247, 0.547. Uniform: 0.515, 0.258, 0.226 |
| Contact coverage | contact share 0.412, push share 0.300, graze share 0.112; contacts 1,981 from scripted actions, 80 from uniform ones; pushes 1,472 and 28 |
| Intervention coverage | **pass**: T = 0.346 (aimed 0.248, uniform 0.584), R = 1.82, τ = 0.024 m/s. Earlier measure, not gated: T = 0.109 |
| Policy log-density | medians: logged 5.26, pool pushes 5.28, aimed pushes 2.22, uniform pushes −0.54 |
| Shadow actions | **pass**: byte-identical (including the new reference column); uniform share 0.101 logged, 0.110 shadow; KS p = 0.44, 0.69, 0.49 |
| Positive-control feasibility | D_obs = 0.066, D_int = 0.542, D_int / D_obs = 8.3 (1,857 push interventions) |
| Compute | smoke 1.8 s, pool 34.9 s (now including the references), states 3.2 s; train plus validation projected at about 38 s |

Skipped episodes: 0.

**Earlier label, for comparison.** The total-motion rule would label 21.6% of contacts differently
(aimed 13.2%, uniform 35.3%), all push → graze under the new label. In 58 of 200 states the box
slides at least 2 mm in the step with no contact.

**Plot.** `plots/push_thresholds.png`: the top row is the design's label (against the reference);
the bottom row is total motion, the earlier rule.

## Every quantity computed

The same list as attempt 0, plus the no-contact reference outcome of every behaviour step. No
model was built, trained, loaded or scored, and no intervention error was computed.

## Commands and output

```
$ uv run ruff check .
All checks passed!
$ uv run ruff format --check .
23 files already formatted
$ uv run pytest -q
39 passed
$ python scripts/check_docs.py
(no output, exit 0)
```

Run: `uv run python scripts/phase0_env.py --docker-replay results/phase0/milestone1/attempt2/docker_replay.json`
with `SMOKE_ATTEMPT = 2` and `POLICY_EPSILON_NEAR = 0.05` in `constants.py`.
