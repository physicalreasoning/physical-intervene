# Phase 0, attempt 2: bit-exact replay on native amd64 hardware

Measurement only: no code, scene file, Dockerfile or docs were changed. Raw values are in
[amd64_replay.json](amd64_replay.json).

## 1. Verdict

| Check | Result |
| --- | --- |
| Bit-exact replay (gated) | **pass**: 5000 compared, 0 mismatched |
| Collider test (`test_simulator_contact_matches_gap`) | **pass** (39 of 39 tests passed) |
| Comparison with the Rosetta run (reported) | **match**: all three digests are identical |

Real amd64 hardware and the Rosetta run produce the same bits for the attempt 2 smoke seeds.

## 2. Machine and versions

| | |
| --- | --- |
| `uname -m` | x86_64 |
| CPU | Intel(R) Core(TM) i7-8565U CPU @ 1.80GHz |
| AVX2 | present |
| Cores (`nproc`) | 8 |
| OS | Ubuntu 24.04.5 LTS (kernel 7.0.0-38-generic) |
| Docker | 29.1.3, build 29.1.3-0ubuntu3~24.04.2 |
| Commit | `30f3720` (`30f37201f344b6831ad09facfc08e500a01275e7`), branch `phase0-simulator` |
| Image id | `sha256:65803726872236395b483d4817163c5e87604763924209ef2f3c9bbcd2d1a221` |
| Inside the image | Python 3.12.16, MuJoCo 4.0.1 |

## 3. Commands run and their output

### Unit tests (step 4)

```bash
docker run --rm --platform linux/amd64 -v "$PWD:/repo:ro" -e UV_PROJECT_ENVIRONMENT=/tmp/venv -w /tmp \
  physical-intervene:phase0 sh -c "cp -r /repo/src /repo/tests /repo/pyproject.toml /repo/uv.lock /repo/README.md /tmp/ \
  && uv sync --locked --group dev -q && /tmp/venv/bin/pytest -q -p no:cacheprovider tests"
```

```
.......................................                                  [100%]
39 passed in 95.58s (0:01:35)
```

### Replay check (step 5)

Generate and check ran in two separate `docker run` calls (fresh processes).

```bash
docker run --rm --platform linux/amd64 -v "$PWD/data/amd64:/out" physical-intervene:phase0 \
  python scripts/replay_check.py --generate /out/states.npz
docker run --rm --platform linux/amd64 -v "$PWD/data/amd64:/out" physical-intervene:phase0 \
  python scripts/replay_check.py --check /out/states.npz
```

```
{"generated": "/out/states.npz", "n_states": 200, "platform": "Linux-7.0.0-38-generic-x86_64-with-glibc2.36", "machine": "x86_64", "python": "3.12.16", "mujoco": "4.0.1"}
{"n_states": 200, "n_actions_per_state": 25, "n_compared": 5000, "n_mismatched": 0, "first_mismatches": [], "passed": true, "platform": "Linux-7.0.0-38-generic-x86_64-with-glibc2.36", "machine": "x86_64", "python": "3.12.16", "mujoco": "4.0.1"}
```

### Digest against the Rosetta run (step 6)

```bash
docker run --rm --platform linux/amd64 -v "$PWD/data/amd64:/out" physical-intervene:phase0 \
  python /out/digest.py /out/states.npz
```

```
state+actions+after+contact+target+reference 485aae8c4c37efa7de19dcb981cd802e7ce85480d4dfa8a2b714169d6ef7ac64
state ce1cb71eac4104a472ecc8007e2c2791b88726c16e124900698782457d6295c5
after 301ccb80830d79b3271902d09ffc91c2edf2cde09678c2a31d28f593c32be9a4
seeds 3800000 3800199 skipped 0
```

| Line | Matches Rosetta |
| --- | --- |
| state+actions+after+contact+target+reference | yes |
| state | yes |
| after | yes |
| seeds 3800000 to 3800199, skipped 0 | yes |

## 4. Anything unexpected

Nothing failed. One difference to note: the image id here
(`sha256:658037268722...`) is not the one recorded for the Rosetta run
(`sha256:6b7201f32230...`, in [docker_replay.json](docker_replay.json)). The Dockerfile and lock
file are the same; the id also depends on the build itself (for example layer timestamps), so a
fresh build is expected to differ. The versions inside the image and all digests match.
