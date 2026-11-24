# Phase 0, milestone 1: simulator and environment checks

Branch `phase0-simulator` (from `smoke-test`), not committed. Simulator and scripted policy only:
no model of any kind was built, trained or run. Smoke seeds only; no pilot, train, validation or
test data was generated, and no test seed was derived.

## Summary

| Check | Result |
| --- | --- |
| Bit-exact replay | **pass**: 5,000 of 5,000 byte-identical locally (fresh process) and in the Docker image (generated and checked in two separate containers) |
| Push thresholds | defaults kept (2 mm, 1°); graze share of contacts 0.172. The design's rule assumes a gap between grazes and pushes, and there is none |
| Pusher can move the box | **pass** after a scene change (box 0.2 → 0.15 kg); analytic margin 3.54, slowest centred push 2.13 mm |
| Push rate | **pass**: 94.5% of states have a push, mean 12.9 of 24 |
| Contact coverage | reported: 38.8% of smoke transitions are contacts; 1,628 from scripted actions, 313 from uniform ones |
| Intervention coverage | **fail**: T = 0.066 (needs ≥ 0.25), R = 0.98. ε_near not changed: the coordinator decides whether to rerun |
| Policy log-density | reported |
| Shadow actions | **pass**: byte-identical transitions; KS p = 0.49, 0.62, 0.37 |
| Positive-control feasibility | reported: D_obs = 0.104, D_int = 0.518, ratio 5.0 |
| Compute | about 27 s to generate all smoke data; 1,100 train plus validation episodes projected at about 25 s |

There are two scene changes, both described under *Values chosen*. One of them, replacing the
cylinder collider with a capsule, needs the coordinator's sign-off.

## 1. What was built

| File | What |
| --- | --- |
| `src/physical_intervene/assets/pushing.xml` | the scene |
| `src/physical_intervene/sim.py` | `Sim`: `get_state`/`set_state` (mjSTATE_INTEGRATION), `reset`, `step` (50 physics steps, returns the contact flag), `step_without_contact` (reference), `obs`; `gap`, `box_target`, `wrap_angle`, `scene_sha256` |
| `src/physical_intervene/policy.py` | `scripted_mode_and_mean`, `sample`, `log_density`, `epsilon`, `Streams` |
| `src/physical_intervene/episodes.py` | stream keys, split seed ranges, `smoke_seeds`, `run_episode`, `generate`, `save`/`load` |
| `src/physical_intervene/interventions.py` | `select_state`, `draw_interventions`, `run_actions`, `make_states`, `label`, `label_vs_reference` |
| `src/physical_intervene/env_checks.py` | one function per check, each returning a dict |
| `scripts/phase0_env.py` | generates the smoke data in `data/smoke/` (gitignored), runs every check, writes `results/phase0/env_checks.json` and the plots |
| `scripts/replay_check.py` | `--check FILE` replays stored states in a fresh process; `--generate FILE` makes the 200 smoke states alone (for use in Docker) |
| `Dockerfile`, `.dockerignore` | linux/amd64, `python:3.12-slim-bookworm` pinned by digest, uv 0.11.7, `uv sync --locked --no-dev`, BLAS/OpenMP limited to one thread |
| `tests/test_sim.py`, `tests/test_policy_and_data.py` | 21 new tests (39 in all) |

Commands to run:

```bash
uv run python scripts/phase0_env.py --docker-replay results/phase0/docker_replay.json
docker build --platform linux/amd64 -t physical-intervene:phase0 .
docker run --rm --platform linux/amd64 -v "$PWD/data/docker:/out" physical-intervene:phase0 \
  python scripts/replay_check.py --generate /out/states.npz
docker run --rm --platform linux/amd64 -v "$PWD/data/docker:/out" physical-intervene:phase0 \
  python scripts/replay_check.py --check /out/states.npz
```

`results/phase0/docker_replay.json` holds the output of the second Docker command, with a note on
how it was run. `results/phase0/cross_platform_replay.json` holds the image's results replayed on
the Mac (reported, not gated). Matplotlib was added to the dev group.

## 2. Check results

Every number is in `env_checks.json`. Scene sha256: `749b329c...25d9e` (full value in the JSON).
Versions: MuJoCo 4.0.1 on both platforms. Locally: Python 3.14.7, macOS arm64. In the image:
Python 3.12.16, x86_64.

**Bit-exact replay: pass.** All 200 states × 25 actions are byte-identical (`np.array_equal` on
raw bytes) in a fresh local process. In the image, the truth was generated in one container and
checked in another: 5,000 of 5,000 match. Image id: `sha256:70832086...cefa3b8`. The image ran
under Rosetta on Apple Silicon (`/proc/cpuinfo` reports "VirtualApple"), so a real amd64 machine
has not been tested. Cross-platform (not gated): replaying the image's truth on the Mac gives 532
of 5,000 byte-identical steps, a largest qpos difference of 1.3e-15 and identical labels.

**Push thresholds: defaults kept.** 3,112 contact interventions. Displacement quantiles
(5/25/50/75/95%): 0.24, 1.90, 4.17, 6.97, 10.6 mm. Rotation quantiles: 0.002, 0.03, 0.34, 1.64,
4.9°. 17.2% of contacts are grazes, and 9.1% are pushes on rotation alone. The plot
(`plots/push_thresholds.png`) shows one displacement mode at 5 to 8 mm with a long left tail and
no gap. 2 mm sits on the lower shoulder, not through the mode, so by DESIGN.md's rule the
defaults stay. θ_push = 1° sits near the rotation mode, but it acts only through the "or", on
the 9.1%.

**Pusher can move the box: pass.** Analytic part: F(0.05 m/s) = 0.5 N, F_resist = √2 × 0.1 =
0.141 N, margin 3.54 (needs 2). Simulator part, with the pusher starting 1 mm off the middle of a
face and the box turned so the push is normal to that face:

| Push | 0.05 m/s | 0.10 m/s | 0.15 m/s |
| --- | --- | --- | --- |
| x or y | 2.32 mm | 5.99 mm | 9.71 mm |
| diagonal | 2.13 mm | 5.76 mm | 9.45 mm |

Off-centre push at 0.10 m/s: 1.27°. End speeds are 0.035 to 0.137 m/s, within 0.006 m/s of the
expected v_cmd − F_resist/kv (diagonal pushes use √2 × the friction). The margin at 0.05 m/s is
thin (2.13 mm against 2 mm) and depends on the 1 mm start gap, which the docs do not fix.

**Push rate: pass.** 189 of 200 states (94.5%) have a push; mean 12.89 of 24 per state, median
12. Shares of all 4,800 interventions: no contact 0.352, graze 0.111, push 0.537. Aimed: 0.211,
0.155, 0.635. Uniform: 0.493, 0.068, 0.440. Logged action: 0.28, 0.16, 0.56.

**Contact coverage (reported).** 5,000 smoke transitions: contact 0.388, push 0.351, graze
0.037. Contacts: 1,628 from scripted actions and 313 from uniform ones. Pushes: 1,480 scripted
and 276 uniform. Uniform-action share: 0.197.

**Intervention coverage: fail.** Pool: 100,000 transitions, 33,387 of them pushes, 15.8% of
those from uniform actions. 1,756 behaviour pushes in the smoke episodes; τ = 2.08, with 88
pushes beyond it. 2,578 push interventions: **T = 0.066** (aimed 0.047, uniform 0.094),
**R = 0.98**. ε_near is 0.2, the first ladder value, and was left unchanged as the brief asks.

A diagnostic I added (not in the plan) explains part of the result. It measures the share of
the squared kNN distance that comes from the two action dimensions: median 0.25 for
interventions against 0.16 for behaviour pushes (2/13 = 0.15 if every dimension counted the
same). Interventions do differ in their actions. But they start from behaviour states, and the
11 state dimensions dominate both distances, so d_int and d_obs come out nearly the same. I
expect a lower ε_near to raise T only a little, because it mainly changes the pool's actions.
The ladder rerun is the coordinator's call (design-log issue 2).

**Policy log-density (reported).** Medians: logged actions at the smoke states 4.88; pool logged
actions 4.87 (pool pushes 4.92); aimed interventions 1.59 (push 1.80); uniform interventions 0.80
(push 0.80). The floor, log(ε × 1/0.09) = 0.80, is reached by 50% of uniform and over 25% of
aimed interventions. Plot: `plots/log_density.png`.

**Shadow actions: pass.** With and without shadow actions, obs, goal, action, uniform flag,
state, gap, contact and target are byte-identical. Uniform share: logged 0.197, shadow 0.203. KS
tests of (a'_t − a_{t−1}) against (a_t − a_{t−1}): dx p = 0.49, dy p = 0.62, norm p = 0.37. The
KS test treats the samples as independent, but they are paired by state. My pass bar was
p ≥ 0.01; the docs give none.

**Positive-control feasibility (reported).** Target standard deviations come from the 50 smoke
episodes, since there is no training split yet: 2.90 mm, 2.97 mm and 0.0295 rad (the floors do
not bind). D_obs = 0.104 over 5,000 behaviour transitions (0.230 on contact transitions); D_int
= 0.518 over 2,578 push interventions; **D_int / D_obs = 5.0**. The gap at matched loss is
≥ 20% when E_int(mlp) ≤ 5 × 5.0 × (V* − L_best(mlp)) = 25 × the matching slack. That seems
plausible, so no redesign is indicated, but V* and E_int do not exist yet. 100 policy draws per
state, from the feasibility streams.

**Compute (data generation only).** Smoke episodes 1.2 s, coverage pool 22.8 s, 200 states with
25 actions 3.2 s, feasibility 86 s; 0.23 ms per episode control step. Training time belongs to
milestone 2.

## 3. Diagnostic push label (design-log issue 5)

The reference outcome is the same step from the same state with contacts disabled
(`mjDSBL_CONTACT`), so the pusher cannot touch the box. The diagnostic label applies the same
thresholds to (outcome − reference).

| | contacts | disagree | push → graze | graze → push | push share (design / reference) |
| --- | --- | --- | --- | --- | --- |
| all | 3,112 | 25.8% | 802 | 0 | 0.83 / 0.57 |
| aimed | 1,894 | 14.0% | 265 | 0 | 0.80 / 0.66 |
| uniform | 1,218 | 44.1% | 537 | 0 | 0.87 / 0.43 |

- In 58 of 200 states the box slides at least 2 mm in the step with no contact at all.
- In 56 states, all 24 actions are *push* under the design's label and none under the reference
  label.
- 516 contacts produce exactly the reference outcome: the pusher touches the box without
  changing its motion (for example, a pusher that starts touching and moves away).
- Example (state 0, uniform action 13): the box moves (−0.18, 3.99) mm and −0.003° under the
  action, and exactly the same with contacts disabled. The design's label calls it a push.
- Under the reference label the push rate still passes: 94.5% of states, mean 8.88 per state.

This supports issue 5: the design's label calls a sliding box pushed. No rule was changed.
Examples are in `env_checks.json` under `push_label_diagnostic`.

## 4. Values chosen (the docs do not fix them)

**Scene** (`pushing.xml`; `constants.py` repeats them and a test checks the two agree):

- Box: 10 × 10 × 5 cm, **0.15 kg**. Changed from 0.2 kg, at which the diagonal 0.05 m/s push
  moved the box only 1.98 mm (fail). 0.15 kg passes both parts. 0.1 kg was rejected because its
  off-centre push turned the box only 0.86°.
- Box frictionloss: 0.1 N on each slide joint and 0.004 N·m on the hinge (about the torque of
  0.1 N of uniform friction under the square). A box released at 0.09 m/s stops within about
  6 mm.
- **Pusher collision geom: a vertical capsule, not MuJoCo's cylinder.** It has radius 1 cm, a
  2 cm straight section and a 4 cm total height, 0.1 kg. With the cylinder, the image's native-CCD
  box-cylinder collider missed contacts: 124 of 1,093 random penetrating configurations, and 35
  of 158 shallow ones (under 3 mm). The Mac missed none. In the image's first replay run, 1,647
  of 5,000 steps differed from the Mac's by more than 1e-6, up to 16 cm or 9°, with the pusher
  passing 6 mm into the box. The capsule uses MuJoCo's analytic box-capsule collider: exact to
  1e-17 on both platforms, and the cross-platform difference fell to 1e-15. The straight section
  lies within the box's height and the round caps touch nothing (a test checks this), so in the
  plane it is the same 1 cm disc. **DESIGN.md says "cylindrical pusher"; please confirm the
  capsule is acceptable or reword the design.** A new test checks MuJoCo's contact depth against
  the analytic gap. It passes locally and in the image, and fails on a collider that misses.
- Pusher-box friction 0.3, condim 3, contact solref 0.01 s (stiffer than the 0.02 s default).
- Actuator forcerange ±2 N: above kv × 0.15 = 1.5 N, so it never caps the force against a box at
  rest.
- Integrator implicitfast, Newton solver, pyramidal cone, 100 iterations, tolerance 1e-8.
- Only the box and pusher can collide (table contype and conaffinity are 0; checked at load).

**Policy** (`constants.py`):

- Box start x, y uniform on [−0.10, 0.10] m; yaw uniform on (−π, π].
- Pusher start x, y uniform on [−0.25, 0.25] m, redrawn until the gap is ≥ 5 cm.
- The goal is a position only, x, y uniform on [−0.20, 0.20] m, at least 5 cm from the box. When
  the box comes within 2 cm of it, a new goal is drawn from the episode's goal stream, and goals
  are stored per step. The shadow actions and the log-density use the current goal.
- The controller has no memory; its mode depends only on the state:
  - *push* when the pusher is within 1 cm of the line through the box centre toward the goal,
    behind the face: 0.1 m/s toward the goal plus a lateral correction (offset / 0.3 s);
  - otherwise *approach*: go to a point 2 cm behind the face when within 30° of the push side;
  - otherwise *orbit*: go round the box at the half-diagonal + 3 cm, in steps of at most 45°.
  - Approach and orbit command 0.7 × the distance per step, capped at 0.12 m/s.
- The Gaussian part is truncated to the action box (inverse-CDF sampling), so `log_density` is
  the exact density of `sample`. Its integral over the action box is 1 to within 5e-3, and it
  matches sampled frequencies (tests).
- Every draw uses one ε number and two each from the noise and uniform streams, whatever the
  branch.

**Seeds** (`constants.py`):

- Episode seed ranges: train [1,000,000, 2,000,000), validation [2,000,000, 3,000,000), smoke
  [3,000,000, 4,000,000), pilot [4,000,000, 5,000,000). A test checks they are disjoint.
- Smoke attempt k starts at 3,000,000 + 300,000 k, with the episodes at offset 0, the pool at
  +100,000 and the states at +200,000. Attempt 0 used 3,000,000 to 3,000,049, 3,100,000 to
  3,100,999 and 3,200,000 to 3,200,199.
- **Skipped episodes: 0** (every one of the 200 had a test-like state).
- Streams: `SeedSequence(seed, spawn_key=(k,))`, with fixed k for init, goal, ε, noise, uniform,
  three shadow streams, selection, interventions and three feasibility streams.

**Other:**

- Aimed interventions use an angle offset uniform on [−45°, 45°].
- The contact flag is `ncon > 0` after any of the 50 physics steps.
- D_obs uses the 50 smoke episodes' transitions.
- The diagnostic reference uses contacts disabled, not a "move away" command, which could still
  touch a box that is moving toward the pusher.

## 5. Ambiguous or questionable points in the docs

1. *Push thresholds* (DESIGN.md): "moved to that gap" assumes a gap between grazes and pushes.
   On these data there is none, so the rule cannot move the thresholds; it keeps them by default.
2. *Pusher can move the box* (SMOKE_TEST.md): "the pusher placed just off the middle of one
   face" does not give the start gap. With 1 mm, the slowest diagonal push passes by 0.13 mm.
3. *Environment* (DESIGN.md) says "cylindrical pusher"; see the capsule change above.
4. *Platform*: the Docker check ran under Rosetta, so "the same check on a second machine of the
   same platform" still needs a real amd64 machine. Truth generated on the Mac is not bit-exact in
   the image (532 of 5,000), so all ground truth must come from the image, as DESIGN.md says. The
   local venv is Python 3.14 and the image is 3.12; the lock covers both.
5. *Intervention coverage*: with distances over 13 standardised dimensions, state dimensions
   dominate. Interventions start from behaviour states, so T stays near 1 − 0.95 + a little
   whatever the actions are. The ε_near ladder may not be able to reach T ≥ 0.25.
6. *Push label* (issue 5): see section 3. 26% of contact labels change with a no-contact
   reference, and 56 states are all-push only because the box was sliding.
7. *Shadow actions*: the docs give no significance level for the two-sample test; I used
   p ≥ 0.01, per axis and on the norm.

## 6. Every quantity computed (no model involved)

Simulator outcomes (states, contact flags, box targets, full integration states) of behaviour
episodes and interventions; gaps; no-contact reference outcomes; labels (design and diagnostic);
the push-threshold displacement and rotation distributions; the pusher-check forces,
displacements, rotations and speeds; push, contact and graze rates; contact counts by action
source; kNN distances in standardised (s, a) between pool, smoke and intervention transitions,
τ, T, R and the action share of distance; behaviour-policy log-densities; shadow-action shares
and KS tests; the policy-mean outcome m from 100 policy draws, D_obs and D_int; box-target
standard deviations on smoke episodes; wall times. No model was built, trained, loaded or
scored, and no intervention error was computed.

## 7. Commands and output

```
$ uv run ruff check .
All checks passed!
$ uv run ruff format --check .
21 files already formatted
$ uv run pytest -q
39 passed in 14.55s
$ python scripts/check_docs.py
(no output, exit 0)
```

The tests also pass inside the image (39 passed, using `uv sync --locked --group dev` in a
container).
