"""The run after freeze 1 (DESIGN.md, Pilot rule, Controls, Seal): the pilot rule, the rule for p, the
job list, the freeze records' hash check and the test seed. No model is trained here except the
stop-at-plateau check, and nothing is scored on interventions."""

from __future__ import annotations

import json

import numpy as np
import pytest

from physical_intervene import constants as c
from physical_intervene import pipeline as P
from physical_intervene import seal

# Pilot rule --------------------------------------------------------------------------------------


def grid_rows(scores: dict[tuple[float, int], list[float]], plateau: int | None = 20_000) -> list[dict]:
    rows = []
    for (lr, bs), vals in scores.items():
        for seed, v in zip(c.PILOT_TRAINING_SEEDS, vals, strict=True):
            rows.append({"lr": lr, "batch_size": bs, "seed": seed, "best_l_obs": v, "plateau_point": plateau})
    return rows


def full_grid(best: tuple[float, int], low: float = 1.0, other: float = 2.0) -> dict:
    return {
        (lr, bs): [low] * 3 if (lr, bs) == best else [other] * 3
        for lr in c.PILOT_LR_GRID
        for bs in c.PILOT_BATCH_GRID
    }


def test_choice_is_lowest_median_and_budget_from_largest_plateau() -> None:
    scores = full_grid((1e-3, 256))
    scores[(1e-3, 256)] = [1.0, 0.9, 5.0]  # median 1.0, not the mean
    rows = grid_rows(scores)
    rows[0]["plateau_point"] = 30_000  # one seed of the first setting; not the chosen one
    ch = P.choose_setting(rows, c.PILOT_LR_GRID)
    assert (ch["lr"], ch["batch_size"]) == (1e-3, 256)
    assert ch["updates"] == int(20_000 * c.BUDGET_FACTOR_OVER_PLATEAU)
    assert ch["edge"] is None


def test_ties_within_band_go_to_smaller_lr_then_smaller_batch() -> None:
    scores = full_grid((1e-3, 1_024))
    scores[(3e-4, 1_024)] = [1.0 * (1 + c.PILOT_TIE_REL_TOL) * 0.999] * 3  # inside the band
    scores[(1e-3, 256)] = [1.0 * (1 + c.PILOT_TIE_REL_TOL) * 1.01] * 3  # outside
    ch = P.choose_setting(grid_rows(scores), c.PILOT_LR_GRID)
    assert (ch["lr"], ch["batch_size"]) == (3e-4, 1_024)
    assert ch["n_tied"] == 2


def test_edge_is_read_from_the_lowest_setting() -> None:
    for lr, edge in ((min(c.PILOT_LR_GRID), "low"), (max(c.PILOT_LR_GRID), "high")):
        ch = P.choose_setting(grid_rows(full_grid((lr, 256))), c.PILOT_LR_GRID)
        assert ch["edge"] == edge
        assert P.edge_extension(edge) == c.PILOT_LR_EDGE_EXTENSION[0 if edge == "low" else 1]
    assert P.edge_extension(None) is None


def test_seed_without_plateau_counts_as_the_cap() -> None:
    rows = grid_rows(full_grid((1e-3, 256)))
    for r in rows:
        if (r["lr"], r["batch_size"]) == (1e-3, 256) and r["seed"] == c.PILOT_TRAINING_SEEDS[1]:
            r["plateau_point"] = None
    ch = P.choose_setting(rows, c.PILOT_LR_GRID)
    assert ch["updates"] == int(c.PILOT_MAX_UPDATES * c.BUDGET_FACTOR_OVER_PLATEAU)
    assert ch["seeds_without_plateau"] == [c.PILOT_TRAINING_SEEDS[1]]


# p, D_obs and the jobs ---------------------------------------------------------------------------


def test_p_rule() -> None:
    assert P.p_rule(1.0, 1.0, 1.0 + 0.25) == 0.5
    assert P.p_rule(1.0, 1.0, 1.0 + 0.0099) is None  # 0.1^2 = 0.01 does not fit
    assert P.p_rule(1.0, 0.0, 1.0) == max(c.SHORTCUT_P_GRID)


def test_d_term_is_std_mse() -> None:
    y, m, s = np.array([[1.0, 2.0, 3.0]]), np.zeros((1, 3)), np.array([1.0, 2.0, 3.0])
    assert P.d_term(y, m, s)[0] == pytest.approx(1.0)


def test_job_list_and_settings_sources() -> None:
    jobs = P.main_jobs()
    n_main = len(c.MAIN_ARCHITECTURES) * (len(c.MAIN_SEEDS) + 2 * len(c.SECONDARY_WIDTH_SEEDS))
    assert len(jobs) == n_main + len(c.MAIN_SEEDS) + len(c.MAIN_ARCHITECTURES) * len(c.TWIN_SEEDS)
    assert len({P.run_name(j["family"], j["width"], j["seed"]) for j in jobs}) == len(jobs)
    assert P.settings_source("gru-twin") == "gru" and P.family_arch("gru-twin") == "gru"
    assert P.settings_source("blind") == "mlp" and P.settings_source("mlp-shortcut") == "mlp"
    assert P.family_arch("mlp-shortcut") == "mlp-shortcut"
    dropped = P.main_jobs(drop_large=True)
    assert len(dropped) == len(jobs) - len(c.MAIN_ARCHITECTURES) * len(c.SECONDARY_WIDTH_SEEDS)
    assert not any(j["width"] == "large" for j in dropped)


def test_split_seeds_are_disjoint_and_in_range() -> None:
    seen = set()
    for split, ((lo, hi), n) in P.SPLIT_EPISODES.items():
        s = P.episode_seeds(split)
        assert len(s) == n and s[0] == lo and s[-1] < hi
        assert not seen & set(s.tolist())
        seen |= set(s.tolist())
    assert P.pilot_states_start() > P.episode_seeds("pilot")[-1]
    assert P.pilot_states_start() + 10 * c.N_PILOT_STATES < c.SEED_RANGE_PILOT[1]


# Seal --------------------------------------------------------------------------------------------


def fake_repo(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "DESIGN.md").write_text("design")
    pkg = tmp_path / "src" / "physical_intervene"
    pkg.mkdir(parents=True)
    (pkg / "a.py").write_text("x = 1")
    (tmp_path / "frozen").mkdir()
    rec = {
        "design_file": "docs/DESIGN.md",
        "design_sha256": seal.sha256_file("docs/DESIGN.md", tmp_path),
        "code_sha256": {
            "src/physical_intervene/a.py": seal.sha256_file("src/physical_intervene/a.py", tmp_path)
        },
        "phase0_record_sha256": {},
    }
    (tmp_path / seal.DESIGN_RECORD).write_text(json.dumps(rec))
    return tmp_path


def test_check_frozen_passes_then_refuses_any_change(tmp_path) -> None:
    root = fake_repo(tmp_path)
    seal.check_frozen(1, root)
    (root / "src" / "physical_intervene" / "a.py").write_text("x = 2")
    with pytest.raises(SystemExit, match="a.py"):
        seal.check_frozen(1, root)


def test_check_frozen_refuses_added_package_file(tmp_path) -> None:
    root = fake_repo(tmp_path)
    (root / "src" / "physical_intervene" / "b.py").write_text("y = 1")
    with pytest.raises(SystemExit, match="added after freeze 1"):
        seal.check_frozen(1, root)


def test_amendments_must_be_listed(tmp_path) -> None:
    root = fake_repo(tmp_path)
    (root / seal.AMENDMENTS_DIR).mkdir()
    f = root / seal.AMENDMENTS_DIR / "2027-01-01.md"
    f.write_text("amendment")
    with pytest.raises(SystemExit, match="not listed"):
        seal.check_frozen(1, root)
    rel = f.relative_to(root).as_posix()
    entry = {"file": rel, "sha256": seal.sha256_file(rel, root)}
    (root / seal.AMENDMENTS_RECORD).write_text(json.dumps({"amendments": [entry]}))
    assert seal.check_frozen(1, root)["amendments"] == [entry]
    f.write_text("amendment, edited")
    with pytest.raises(SystemExit, match="2027-01-01.md"):
        seal.check_frozen(1, root)


def test_a_hashed_file_may_change_only_with_an_amendment_recording_its_new_hash(tmp_path) -> None:
    root = fake_repo(tmp_path)
    code = "src/physical_intervene/a.py"
    (root / code).write_text("x = 2")  # the fix
    with pytest.raises(SystemExit, match="a.py"):
        seal.check_frozen(1, root)
    (root / seal.AMENDMENTS_DIR).mkdir()
    f = root / seal.AMENDMENTS_DIR / "2027-01-01.md"
    f.write_text("quotes x = 1 and x = 2, and why")
    rel = f.relative_to(root).as_posix()
    entry = {
        "file": rel,
        "sha256": seal.sha256_file(rel, root),
        "code_sha256": {code: seal.sha256_file(code, root)},
    }
    (root / seal.AMENDMENTS_RECORD).write_text(json.dumps({"amendments": [entry]}))
    seal.check_frozen(1, root)
    (root / code).write_text("x = 3")  # a change the amendment does not record
    with pytest.raises(SystemExit, match="a.py"):
        seal.check_frozen(1, root)


def test_stage2_checks_its_own_files_and_derives_the_test_seed(tmp_path) -> None:
    root = fake_repo(tmp_path)
    (root / "d.npz").write_bytes(b"data")
    models = {
        "amendments_record_sha256": None,
        "design_record_sha256": seal.sha256_file(seal.DESIGN_RECORD, root),
        "data_sha256": {"d.npz": seal.sha256_file("d.npz", root)},
        "checkpoint_sha256": {},
    }
    (root / seal.MODELS_RECORD).write_text(json.dumps(models))
    out = seal.check_frozen(2, root)
    assert out["test_seed_start"] == c.TEST_SEED_BASE + int(out["models_sha256"][:12], 16)
    assert out["test_seed_start"] > max(c.SEED_RANGE_PILOT)
    (root / seal.AMENDMENTS_RECORD).write_text(json.dumps({"amendments": []}))  # after freeze 2
    with pytest.raises(SystemExit, match="differs from freeze 2"):
        seal.check_frozen(2, root)
    (root / seal.AMENDMENTS_RECORD).unlink()
    (root / "d.npz").write_bytes(b"changed")
    with pytest.raises(SystemExit, match="d.npz"):
        seal.check_frozen(2, root)


def test_freeze1_code_covers_the_package_and_run_scripts() -> None:
    files = seal.freeze1_code()
    assert "src/physical_intervene/constants.py" in files
    assert any(f.endswith(".xml") for f in files)  # the scene file
    for s in seal.FREEZE1_SCRIPTS:
        assert s in files and (seal.ROOT / s).exists()
    assert not any("__pycache__" in f for f in files)


# Scoring saved runs (untrained networks only, so the no-leak guard allows them) --------------------


def test_score_run_and_family_from_untrained_checkpoints(tmp_path, monkeypatch) -> None:
    pytest.importorskip("torch")
    # The phase comes from the repo's freeze records; fix it, so that after freeze 2 the guard does
    # not re-hash data files a test machine does not have.
    monkeypatch.setattr(seal, "current_phase", lambda root=seal.ROOT: "phase0")
    from physical_intervene import episodes as E
    from physical_intervene import scoring as S
    from physical_intervene import standins as SI
    from physical_intervene.data import InputNorm, Split, box_scaler, delta_scaler, make_examples
    from physical_intervene.sim import Sim

    eps = E.generate(Sim(), np.array([c.SEED_RANGE_SMOKE[1] - 2, c.SEED_RANGE_SMOKE[1] - 1]))
    split = Split("validation", eps["obs"], eps["action"], eps["target"], eps["seed"])
    val = make_examples(split)
    norm, dsc, ysc = InputNorm.fit(val), delta_scaler(val.delta), box_scaler(val)
    g = np.random.default_rng(0)
    n = 3
    iset = S.InterventionSet(
        s=val.s[:n],
        actions=g.uniform(-0.15, 0.15, size=(n, 25, 2)),
        y_true=g.normal(size=(n, 25, 3)) * 0.003,
        label=g.integers(0, 3, size=(n, 25)),
        kinds=np.array([0] + [1] * 12 + [2] * 12),
        hist_s=val.hist_s[:n],
        hist_a=val.hist_a[:n],
        hist_mask=val.hist_mask[:n],
    )
    dirs = []
    for seed in (1, 2):
        d = tmp_path / f"mlp_small_s{seed}"
        for u in (10, 20):
            SI.save_untrained(d / f"u{u:06d}.pt", "mlp", "small", seed * 100 + u, norm, dsc, ysc)
        dirs.append(d)
    scored = [P.score_run(d, val, iset, ysc, ("push", "logged")) for d in dirs]
    assert scored[0]["l_obs"].shape == (2,) and scored[0]["sums"]["push"].shape == (2, n)
    fam = P.family_from_scored("mlp", "small", [1, 2], scored, iset, val)
    assert fam.l_obs.shape == (2, 2) and fam.sums["push"].shape == (2, 2, n)
    pred = fam.meta["predict"](1, 0)
    ls = S.losses(ysc, pred, val.y, iset)
    assert ls.l_obs == pytest.approx(fam.l_obs[1, 0])
    assert np.allclose(S.state_sums(ls.interventions, iset, "push"), fam.sums["push"][1, 0])


def test_stop_at_plateau_stops_at_the_first_checkpoint_meeting_it(monkeypatch) -> None:
    pytest.importorskip("torch")
    from physical_intervene import episodes as E
    from physical_intervene.data import Split
    from physical_intervene.sim import Sim
    from physical_intervene.train import TrainConfig, TrainData, train

    for k, v in {
        "CHECKPOINT_FINE_EVERY_UPDATES": 10,
        "CHECKPOINT_FINE_UNTIL_UPDATES": 10,
        "CHECKPOINT_DENSE_EVERY_UPDATES": 10,
        "CHECKPOINT_DENSE_UNTIL_UPDATES": 10,
        "CHECKPOINT_EVERY_UPDATES": 10,
        "PLATEAU_CHECKPOINTS": 2,
        "PLATEAU_MIN_REL_IMPROVEMENT": 10.0,  # always met once evaluable
    }.items():
        monkeypatch.setattr(c, k, v)
    eps = E.generate(Sim(), np.array([c.SEED_RANGE_SMOKE[1] - 2, c.SEED_RANGE_SMOKE[1] - 1]))
    split = Split("train", eps["obs"], eps["action"], eps["target"], eps["seed"], shadow=eps["shadow"])
    td = TrainData.build(split, split)
    cfg = dict(arch="mlp", width="small", seed=1, updates=200, lr=1e-3, batch_size=32, seq_batch=4)
    r = train(TrainConfig(**cfg, stop_at_plateau=True), td)
    assert r["stopped_at_plateau"] and r["updates_done"] == 20
    r = train(TrainConfig(**cfg), td)
    assert not r["stopped_at_plateau"] and r["updates_done"] == 200


def test_locked_versions_lists_every_python_range() -> None:
    lock = (seal.ROOT / "uv.lock").read_text()
    numpy = seal.locked_versions("numpy", lock)
    assert len(numpy) >= 1 and all(v["version"] and v["python"] for v in numpy)
    pkg = '[[package]]\nname = "{}"\nversion = "{}"\n'
    marker = "resolution-markers = [\"python_full_version {} '3.12'\"]\n"
    text = (
        pkg.format("numpy", "2.4.6")
        + marker.format("<")
        + pkg.format("numpy", "2.5.5")
        + marker.format(">=")
        + pkg.format("torch", "2.15.0")
    )
    assert [v["version"] for v in seal.locked_versions("numpy", text)] == ["2.4.6", "2.5.5"]
    assert seal.locked_versions("torch", text) == [{"version": "2.15.0", "python": ["any"]}]


def test_result_without_positive_control_and_with_per_family_budgets() -> None:
    """build_primary when no p fits (mlp-shortcut not run) and with each family's own budget: the
    positive control is reported as not tested and does not pass, and the result conforms."""
    from physical_intervene import gates as G
    from physical_intervene import results_format as RF
    from physical_intervene import scoring as S
    from physical_intervene import standins as SI
    from physical_intervene.data import TargetScaler

    g = np.random.default_rng(0)
    n, n_val = 6, 40
    th = g.uniform(-np.pi, np.pi, n)
    s = g.normal(size=(n, 11)) * 0.05
    s[:, 2], s[:, 3] = np.cos(th), np.sin(th)
    label = g.integers(0, 3, size=(n, 25))
    label[:, 1] = 2  # every state has a push
    iset = S.InterventionSet(
        s=s,
        actions=g.uniform(-0.15, 0.15, size=(n, 25, 2)),
        y_true=g.normal(size=(n, 25, 3)) * 0.003,
        label=label,
        kinds=np.array([0] + [1] * 12 + [2] * 12),
    )
    val_s = np.repeat(s, n_val // n + 1, axis=0)[:n_val]
    val_y = g.normal(size=(n_val, 3)) * 0.003
    val_episode = np.arange(n_val) // 4
    scaler = TargetScaler(np.zeros(3), np.array([0.003, 0.003, 0.003]))
    ctx = SI.Context(scaler, val_y, val_episode, iset)
    updates = [10, 20, 30]

    def fam(name: str, n_seeds: int, key: int, width: str = "medium") -> G.Family:
        def predict(i: int, k: int) -> S.Prediction:
            r = np.random.default_rng([key, i, k])
            scale = 0.003 * (1.0 / (k + 1)) * (1.5 if name == "blind" else 1.0)
            return S.Prediction(
                val=val_y + r.normal(size=val_y.shape) * scale,
                interventions=iset.y_true + r.normal(size=iset.y_true.shape) * scale,
            )

        f = SI.family_from_predictor(name, list(range(n_seeds)), updates, predict, ctx, width)
        f.meta["predict"] = predict
        return f

    names = [*c.MAIN_ARCHITECTURES, "blind", *(f"{a}-twin" for a in c.MAIN_ARCHITECTURES)]
    fams = {nm: fam(nm, 12, i) for i, nm in enumerate(names)}
    widths = {
        w: {a: fam(a, 5, 100 + 10 * j + i, w) for i, a in enumerate(c.MAIN_ARCHITECTURES)}
        for j, w in enumerate(("small", "large"))
    }
    copy = S.Prediction(val=np.zeros_like(val_y), interventions=np.zeros_like(iset.y_true))
    lab = iset.label[:, 1:]
    counts = {
        "n_train_obs": 1,
        "n_val_obs": n_val,
        "n_interventions_total": int(lab.size),
        "n_held_out_checkpoints": n,
        "regime_counts": {st: int((lab == i).sum()) for i, st in enumerate(S.STRATA)},
    }
    budgets = {nm: 30 + i for i, nm in enumerate(names)}
    primary = RF.build_primary(
        fams,
        widths,
        scaler,
        iset,
        val_s,
        val_y,
        val_episode,
        {"copy": copy, "constvel": None, "knn": None},
        counts,
        {"passed": True},
        np.random.default_rng(1),
        budgets,
        reps=50,
    )
    pos = primary["control_pairs"][0]
    assert pos["challenger"] == "mlp-shortcut" and pos["tested"] is False and pos["separates"] is False
    assert primary["gates"]["gate_positive_control_pass"] is False
    assert primary["gates"]["gate_g1_reading"] != "fail"  # a fail without the positive control is not read
    assert "mlp-shortcut" not in primary["controls"]
    assert primary["models"]["gru"]["updates"] == budgets["gru"]
    assert primary["controls"]["blind"]["updates"] == budgets["blind"]
    result = {
        "format_version": RF.FORMAT_VERSION,
        "benchmark": "test",
        "device": "cpu",
        "seeds": list(range(12)),
        "updates": budgets,
        "wall_seconds": 0.0,
        "primary": primary,
    }
    assert RF.check_conformance(result) == []
