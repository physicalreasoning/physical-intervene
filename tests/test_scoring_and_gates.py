"""Intervention side (SMOKE_TEST.md, No-leak rule and Intervention-side checks): the scorer's guard,
std-MSE blocks, the bootstrap, p-values, Holm, matching, verdicts, readings and the result format.
No trained model is involved."""

from __future__ import annotations

import json

import numpy as np
import pytest

from physical_intervene import constants as c
from physical_intervene import gates as G
from physical_intervene import results_format as RF
from physical_intervene import scoring as S
from physical_intervene import standins as SI
from physical_intervene.data import TargetScaler

# No-leak guard -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "meta",
    [
        {"trained": True, "phase": "phase0"},
        {"phase": "phase0"},  # missing key: refused
        {"trained": "false"},  # not the boolean false: refused
        {"trained": 0},
    ],
)
def test_scorer_refuses_trained_or_unlabelled_in_phase0(meta: dict) -> None:
    with pytest.raises(S.LeakError):
        S.assert_scorable(meta, phase="phase0")


def test_scorer_accepts_untrained_and_standins() -> None:
    S.assert_scorable({"trained": False}, phase="phase0")
    S.assert_scorable(SI.STANDIN_META, phase="phase0")
    S.assert_scorable({"trained": True}, phase="main")  # after Phase 0 the guard steps aside


def test_phase_flag_is_phase0() -> None:
    assert c.PHASE == "phase0"


def test_entry_point_refuses_before_reading_weights(tmp_path) -> None:
    p = tmp_path / "x.pt"
    p.write_bytes(b"not a checkpoint")  # would fail to load if it were read
    p.with_suffix(".json").write_text(json.dumps({"trained": True, "phase": "phase0"}))
    with pytest.raises(S.LeakError):
        S.load_checkpoint_for_scoring(p, phase="phase0")
    q = tmp_path / "y.pt"
    q.write_bytes(b"not a checkpoint")
    with pytest.raises(S.LeakError):
        S.load_checkpoint_for_scoring(q, phase="phase0")


def test_predict_model_refuses_trained_metadata() -> None:
    with pytest.raises(S.LeakError):
        S.predict_model(None, None, None, None, {"trained": True}, phase="phase0")


# Scoring -----------------------------------------------------------------------------------------


def toy_set(n: int = 6, seed: int = 0) -> S.InterventionSet:
    g = np.random.default_rng(seed)
    s = g.normal(size=(n, 11)) * 0.05
    th = g.uniform(-np.pi, np.pi, n)
    s[:, 2], s[:, 3] = np.cos(th), np.sin(th)
    label = g.integers(0, 3, size=(n, 25))
    return S.InterventionSet(
        s=s,
        actions=g.uniform(-0.15, 0.15, size=(n, 25, 2)),
        y_true=g.normal(size=(n, 25, 3)) * np.array([0.003, 0.003, 0.02]),
        label=label,
        kinds=np.array([0] + [1] * 12 + [2] * 12),
    )


def test_blocks_never_include_the_logged_action() -> None:
    iset = toy_set()
    for b in S.BLOCKS:
        assert not iset.mask(b)[:, 0].any()
    assert (iset.mask("all") == (iset.mask("no_contact") | iset.mask("graze") | iset.mask("push"))).all()
    assert (iset.mask("all_contact") == (iset.mask("graze") | iset.mask("push"))).all()
    assert iset.logged_mask()[:, 0].all() and iset.logged_mask().sum() == iset.n_states


def test_copy_direct_equals_scaler_and_floors_apply() -> None:
    iset = toy_set()
    scaler = TargetScaler.fit(iset.y_true.reshape(-1, 3) * 1e-3, c.TARGET_STD_FLOORS)
    assert np.all(scaler.scale >= np.asarray(c.TARGET_STD_FLOORS))
    y = iset.y_true.reshape(-1, 3)
    assert S.copy_std_mse_direct(scaler, y) == pytest.approx(scaler.std_mse(np.zeros_like(y), y), rel=1e-12)


def test_block_summary_of_truth_is_zero() -> None:
    iset = toy_set()
    scaler = TargetScaler(np.zeros(3), np.array([0.003, 0.003, 0.02]))
    out = S.block_summary(scaler, iset, iset.y_true, "push", np.random.default_rng(0), reps=50)
    assert out["std_mse"] == 0.0 and out["corner_rmse_mm"] == 0.0 and out["angle_rmse_deg"] == 0.0
    assert out["n"] == int(iset.mask("push").sum())


# Bootstrap, p, Holm ------------------------------------------------------------------------------


def test_p_value_formula() -> None:
    reps = np.array([1.0] * 90 + [-1.0] * 10)
    assert G.p_two_sided(reps) == pytest.approx(2 * 11 / 101)
    assert G.p_two_sided(np.ones(5000)) == pytest.approx(2 / 5001)
    assert G.p_two_sided(np.zeros(10)) == 1.0


def test_holm_by_hand() -> None:
    p = [0.01, 0.04, 0.03, 0.005, None, 0.5]
    # m = 6, sorted: 0.005, 0.01, 0.03, 0.04, 0.5, 1 -> 0.03, 0.05, 0.12, 0.12 (monotone), 1, 1
    assert G.holm(p) == pytest.approx([0.05, 0.12, 0.12, 0.03, 1.0, 1.0])


def arm(name: str, per_seed_state: np.ndarray, n: np.ndarray) -> G.Arm:
    return G.Arm(name, per_seed_state * n[None, :], n)


def test_boot_e_with_unit_weights_is_e_int() -> None:
    g = np.random.default_rng(1)
    n = g.integers(1, 6, 30)
    a = arm("a", g.uniform(0.5, 1.5, (10, 30)), n)
    W = np.ones((1, 30))
    states = np.arange(30)
    assert G.boot_e(a, W, states, None)[0] == pytest.approx(a.e)
    # E_int: each transition's loss averaged over seeds, then over transitions pooled across states.
    per_transition = np.repeat(a.sums / n[None, :], n, axis=1)
    assert a.e == pytest.approx(per_transition.mean(0).mean())


def test_gap_of_identical_arms_is_zero_and_known_shift_is_recovered() -> None:
    g = np.random.default_rng(2)
    n = g.integers(1, 6, 40)
    base = g.uniform(0.5, 1.5, (10, 40))
    a = arm("a", base, n)
    same = G.gap_test(a, arm("a2", base.copy(), n), np.random.default_rng(3), reps=500, seeds=False)
    assert (
        same["gap"] == 0.0 and same["ci_low"] == 0.0 and same["ci_high"] == 0.0 and same["p_two_sided"] == 1.0
    )
    shifted = arm("b", base + 0.3, n)  # every transition 0.3 worse, every seed
    r = G.gap_test(shifted, a, np.random.default_rng(4), reps=500, seeds=False)
    assert r["gap"] == pytest.approx(0.3)
    assert r["ci_low"] == pytest.approx(0.3) and r["ci_high"] == pytest.approx(0.3)
    assert r["p_two_sided"] == pytest.approx(2 / 501)


def test_states_with_no_transition_in_the_block_are_dropped() -> None:
    n = np.array([0, 2, 3, 0, 1])
    a = arm("a", np.ones((3, 5)), n)
    b = arm("b", 2 * np.ones((3, 5)), n)
    r = G.gap_test(b, a, np.random.default_rng(5), reps=200)
    assert r["gap"] == pytest.approx(1.0) and r["ci_low"] == pytest.approx(1.0)


# Matching and verdicts ---------------------------------------------------------------------------


def test_v_star_matching_and_exclusion() -> None:
    upd = list(range(10, 210, 10))
    u = np.asarray(upd, float)
    good = 0.05 + 0.95 * np.exp(-u[None, :] / 40.0) * np.ones((20, 1))
    slow = 0.08 + 0.92 * np.exp(-u[None, :] / 40.0) * np.ones((20, 1))
    slow[2] += 0.2  # never comes near V*
    fa, fb = SI.synthetic_family("a", upd, good), SI.synthetic_family("b", upd, slow)
    v, setter = G.v_star_of([fa, fb])
    assert setter == "b" and v == pytest.approx(1.1 * np.median(slow.min(1)))
    mb = G.match(fb, v)
    assert mb.excluded.tolist() == [i == 2 for i in range(20)] and not mb.dropped
    ma = G.match(fa, v)
    assert np.all(ma.idx == np.argmin(np.abs(good - v), axis=1))


@pytest.mark.parametrize(
    ("meets", "twin", "positive", "want"),
    [
        (True, "clean", False, "pass"),
        (True, "not_run", False, "pass"),
        (True, "separated", True, "inconclusive_not_specific"),
        (False, "clean", True, "fail"),
        (False, "clean", False, "inconclusive_not_sensitive"),
    ],
)
def test_g1_reading(meets: bool, twin: str, positive: bool, want: str) -> None:
    pairs = [
        {"tested": True, "challenger": "gru", "reference": "mlp", "matched_within_5pct": True, "meets": meets}
    ]
    assert G.g1_reading(pairs, {"gru": twin, "mlp": "clean"}, positive)[0] == want


def test_g1_not_testable_when_no_pair_is_screened() -> None:
    pairs = [
        {"tested": True, "challenger": "gru", "reference": "mlp", "matched_within_5pct": False, "meets": True}
    ]
    assert G.g1_reading(pairs, {}, True)[0] == "not_testable"
    assert G.g1_reading([{"tested": False}], {}, True)[0] == "not_testable"


def level(name: str, lo: float | None, hi: float | None) -> dict:
    return {"level": name, "ci_low": lo, "ci_high": hi, "tested": lo is not None}


@pytest.mark.parametrize(
    ("signs", "want"),
    [
        ((0, 0, 0, 0), "overlap"),
        ((1, 1, 1, 1), "persistent"),
        ((1, 1, 1, 0), "closing"),
        ((1, 1, 1, -1), "mixed"),
        ((0, 0, 0, 1), "mixed"),
        ((-1, -1, -1), "persistent"),  # no V*_pair below V*: the lowest level is V* itself
    ],
)
def test_curve_reading(signs: tuple, want: str) -> None:
    names = ["2.0xV*", "1.5xV*", "V*", "V*_pair"][: len(signs)]
    ci = {1: (0.1, 0.2), -1: (-0.2, -0.1), 0: (-0.1, 0.1)}
    assert G.curve_reading([level(n, *ci[s]) for n, s in zip(names, signs, strict=True)]) == want


def test_g2_reading_thresholds_and_spearman() -> None:
    assert G.g2_reading(0.49) == "decoupled"
    assert G.g2_reading(0.5) == "inconclusive"
    assert G.g2_reading(0.8) == "refuted"
    x = np.linspace(0, 1, 30)
    r = G.spearman_boot(x, x**2, np.random.default_rng(0), reps=200)
    assert r["rho"] == pytest.approx(1.0) and r["ci_low"] == pytest.approx(1.0)


def test_asymmetric_reading() -> None:
    base = {"tested": True, "matched_within_5pct": True, "meets": True}
    assert G.asymmetric_reading({**base, "challenger": "mlp", "reference": "contact_wm"})["reading"] == (
        "evidence_for_structure"
    )
    assert G.asymmetric_reading({**base, "challenger": "contact_wm", "reference": "mlp"})["reading"] == (
        "not_evidence_against"
    )
    assert G.asymmetric_reading({**base, "challenger": "gru", "reference": "mlp"}) is None


# Stand-ins ---------------------------------------------------------------------------------------


def test_bias_share_and_ratio_round_trip() -> None:
    f = SI.bias_share_for_cv(0.10)
    assert f * np.sqrt(2 / 3) == pytest.approx(0.10)
    r = SI.ratio_for_rel_gap(0.25, f)
    assert SI.rel_gap_for_ratio(r, f) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        SI.bias_share_for_cv(0.9)


def test_standin_error_scale_matches_its_expectation() -> None:
    iset = toy_set(n=200, seed=3)
    scaler = TargetScaler(np.zeros(3), np.array([0.003, 0.003, 0.02]))
    val_y = np.random.default_rng(4).normal(size=(5000, 3)) * scaler.scale
    ctx = SI.Context(scaler, val_y, np.repeat(np.arange(50), 100), iset)
    seeds = tuple(range(400))
    sp = SI.StandIn("x", 7, seeds, [1], np.full((len(seeds), 1), 0.1), bias_share=0.3, push_ratio=1.5)
    fam = SI.family(sp, ctx)
    e = fam.run_e("push", np.zeros(len(seeds), int))
    assert e.mean() == pytest.approx(sp.expected_e_push(0, 0), rel=0.05)
    assert fam.l_obs.mean() == pytest.approx(0.1, rel=0.02)
    # The bias is shared by a seed's push transitions: the seed-to-seed spread matches f sqrt(2/3).
    assert np.std(e) / np.mean(e) == pytest.approx(0.3 * np.sqrt(2 / 3) / (0.3 + 2.25 * 0.7), rel=0.25)


def test_conformance_reports_missing_keys() -> None:
    errors = RF.check_conformance({"format_version": "0.1", "primary": {}})
    assert any("missing benchmark" in e for e in errors)
    assert any("missing pairwise_gate_g1" in e for e in errors)


# Histories of test-like states -------------------------------------------------------------------


def test_state_histories_replay_the_stored_step() -> None:
    from physical_intervene import interventions as I
    from physical_intervene.sim import Sim

    sim = Sim()
    seed = c.SEED_RANGE_SMOKE[0] + 999_000  # inside the smoke range, outside every attempt's blocks
    states = I.make_states(sim, seed, 2)
    h = I.state_histories(sim, states)
    assert h["hist_s"].shape == (2, c.HISTORY_LEN, 11) and h["hist_mask"].shape == (2, c.HISTORY_LEN)
    assert np.all(h["hist_mask"][states["step"] >= c.HISTORY_LEN] == 1.0)
    states["obs"][0, 0] += 1e-12  # a stored state the replay does not reproduce
    with pytest.raises(RuntimeError):
        I.state_histories(sim, states)


def test_g1_gap_is_welchs_t_test_over_seeds() -> None:
    from scipy import stats

    g = np.random.default_rng(7)
    n = g.integers(1, 6, 50)
    a = arm("a", g.uniform(1.0, 2.0, (20, 50)), n)
    b = arm("b", g.uniform(0.5, 1.5, (20, 50)), n)
    r = G.welch_test(a, b)
    ref = stats.ttest_ind(a.per_seed, b.per_seed, equal_var=False)
    assert r["p_two_sided"] == pytest.approx(ref.pvalue)
    assert r["gap"] == pytest.approx(a.e - b.e) and r["method"] == c.G1_TEST
    assert np.mean(a.per_seed) == pytest.approx(a.e)  # E_int is the mean of the per-seed values
    # The interval is gap +/- t(0.975, df) x se, and excludes zero exactly when p < 0.05.
    se = np.sqrt(np.var(a.per_seed, ddof=1) / 20 + np.var(b.per_seed, ddof=1) / 20)
    assert r["ci_high"] - r["gap"] == pytest.approx(stats.t.ppf(0.975, r["df"]) * se)
    assert G.excludes_zero(r) == (r["p_two_sided"] < 0.05)
