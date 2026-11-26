"""Model unit tests (SMOKE_TEST.md, Training pipeline checks, part 4). No training on real data."""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from physical_intervene import constants as c  # noqa: E402
from physical_intervene.data import InputNorm, Split, TargetScaler  # noqa: E402
from physical_intervene.models import build, make_batch, size_for  # noqa: E402
from physical_intervene.models.common import Batch  # noqa: E402
from physical_intervene.models.contact_wm import ContactWM, push_twist  # noqa: E402
from physical_intervene.models.mlp import mlp_input_dim  # noqa: E402
from physical_intervene.models.rssm import RSSM, Gaussian, kl_divergence  # noqa: E402
from physical_intervene.seal import PHASES, current_phase  # noqa: E402
from physical_intervene.train import TrainConfig, TrainData, stream_seed, train  # noqa: E402

STANDIN_P = 0.3  # stand-in for mlp-shortcut's p, which is computed only after the main runs


def unit_norm() -> tuple[InputNorm, TargetScaler]:
    z, o = np.zeros(11), np.ones(11)
    return InputNorm(z, o, np.zeros(2), np.ones(2)), TargetScaler(z, o)


def random_split(n_ep: int = 6, T: int = 100, seed: int = 0) -> Split:
    g = np.random.default_rng(seed)
    obs = g.normal(size=(n_ep, T + 1, 11)) * 0.05
    th = g.uniform(-np.pi, np.pi, size=(n_ep, T + 1))
    obs[..., 2], obs[..., 3] = np.cos(th), np.sin(th)
    action = g.uniform(-0.15, 0.15, size=(n_ep, T, 2))
    shadow = g.uniform(-0.15, 0.15, size=(n_ep, T, 2))
    return Split("train", obs, action, np.zeros((n_ep, T, 3)), np.arange(n_ep), shadow=shadow, file="random")


# contact_wm --------------------------------------------------------------------------------------


def hand_set_contact_wm(dtype=torch.float64) -> ContactWM:
    """contact_wm with its numbers set by hand in this test (not MuJoCo's, and not trained)."""
    m = ContactWM(*unit_norm(), hidden=8).to(dtype)
    vals = {
        "log_lag": 0.002,
        "log_half_width": 0.05,
        "log_pusher_radius": 0.01,
        "log_sharpness": 5_000.0,
        "log_mu_pusher": 0.3,
        "log_mu_table": 0.2,
        "log_limit_surface": c.UNIFORM_SQUARE_LIMIT_SURFACE_FACTOR * 0.05,
    }
    with torch.no_grad():
        for k, v in vals.items():
            m.physics[k].copy_(torch.tensor(math.log(v), dtype=dtype))
    return m


def contact_batch(pusher_xy, a, box_theta: float = 0.0, dtype=torch.float64) -> Batch:
    s = torch.zeros(1, 11, dtype=dtype)
    s[0, 2], s[0, 3] = math.cos(box_theta), math.sin(box_theta)
    s[0, 7:9] = torch.tensor(pusher_xy, dtype=dtype)
    s[0, 9:11] = torch.tensor(a, dtype=dtype)  # already moving at the command
    a_t = torch.tensor([a], dtype=dtype)
    z = torch.zeros
    return Batch(z(1, 11), z(1, 8, 11), z(1, 8, 2), z(1, 8), z(1, 2), s, a_t)


def box_motion(m: ContactWM, pusher_xy, a, box_theta: float = 0.0) -> np.ndarray:
    d = m.predict_delta(contact_batch(pusher_xy, a, box_theta)).detach().numpy()[0]
    th0 = box_theta
    dth = math.atan2(math.sin(th0) + d[3], math.cos(th0) + d[2]) - th0
    return np.array([d[0], d[1], dth])


def test_contact_wm_far_pusher_gives_no_contact() -> None:
    m = hand_set_contact_wm()
    dx, dy, dth = box_motion(m, (0.30, 0.0), (0.1, 0.0))
    assert abs(dx) < 1e-9 and abs(dy) < 1e-9 and abs(dth) < 1e-9
    feats, _, _ = m.physics_step(
        contact_batch((0.30, 0.0), (0.1, 0.0)).s, torch.tensor([[0.1, 0.0]], dtype=torch.float64)
    )
    assert feats[0, 0].item() < 1e-9  # contact indicator


def test_contact_wm_centre_push_moves_box_straight() -> None:
    m = hand_set_contact_wm()
    dx, dy, dth = box_motion(m, (-0.0605, 0.0), (0.1, 0.0))
    assert dx > 0.003  # pushed most of the 10 mm the pusher travels
    assert abs(dy) < 1e-9 and abs(dth) < 1e-9


def test_contact_wm_off_centre_push_turns_box() -> None:
    m = hand_set_contact_wm()
    dx, dy, dth = box_motion(m, (-0.0605, 0.03), (0.1, 0.0))
    assert dx > 0.001
    assert dth < -0.01  # a push in +x above the centre turns the box clockwise


def test_contact_wm_steep_push_slips() -> None:
    """Inside the motion cone the contact point follows the pusher (stick); outside, it does not
    (slip) and its tangential-to-normal ratio sits on the cone's edge."""
    dt = torch.float64
    n = torch.tensor([[-1.0, 0.0]], dtype=dt)  # outward normal of the face x = -b
    cp = torch.tensor([[-0.05, 0.0]], dtype=dt)
    mu, c_ls = torch.tensor([0.3], dtype=dt), torch.tensor([0.7652 * 0.05], dtype=dt)
    from physical_intervene.models.contact_wm import contact_point_velocity

    def ratio(angle_deg: float) -> float:
        v = 0.1 * torch.tensor(
            [[math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))]], dtype=dt
        )
        tw, _ = push_twist(v, n, cp, mu, c_ls)
        vc = contact_point_velocity(tw, cp)[0]
        return float(vc[1] / vc[0])

    assert ratio(3.0) == pytest.approx(math.tan(math.radians(3.0)), rel=0.02)  # stick
    steep = ratio(70.0)
    assert steep < 0.5 * math.tan(math.radians(70.0))  # slip: the box does not follow
    assert steep == pytest.approx(ratio(80.0), rel=0.02)  # on the cone edge whatever the angle


def test_contact_wm_smooth_contact_passes_gradient_check() -> None:
    m = hand_set_contact_wm()
    with torch.no_grad():
        m.physics["log_sharpness"].fill_(math.log(500.0))  # soft enough for finite differences
        for p in m.correction.parameters():
            p.normal_(0.0, 0.1)
    base = contact_batch((-0.061, 0.012), (0.08, 0.02), box_theta=0.1)
    base.s[0, 4:7] = torch.tensor([0.01, -0.004, 0.05], dtype=torch.float64)  # |v| has a kink at rest
    s = base.s.clone().requires_grad_(True)
    a = base.a.clone().requires_grad_(True)

    def f(s_, a_):
        return m.predict_delta(Batch(base.s_n, base.hs_n, base.ha_n, base.mask, base.a_n, s_, a_))

    assert torch.autograd.gradcheck(f, (s, a), eps=1e-7, atol=1e-5, rtol=1e-3)
    # And with respect to the learned numbers.
    names = [f"physics.{k}" for k in m.physics]
    phys = tuple(v.detach().clone().requires_grad_(True) for v in m.physics.values())

    def g(*ph):
        return torch.func.functional_call(m, dict(zip(names, ph, strict=True)), (base,), strict=False)

    m.forward = m.predict_delta  # type: ignore[method-assign]
    assert torch.autograd.gradcheck(g, phys, eps=1e-7, atol=1e-5, rtol=1e-3)


def test_contact_wm_width_counts_only_the_correction() -> None:
    m = ContactWM(*unit_norm(), hidden=size_for("contact_wm", "small"))
    assert m.n_params() == sum(p.numel() for p in m.correction.parameters())
    assert sum(p.numel() for p in m.parameters()) == m.n_params() + len(m.physics)


# rssm --------------------------------------------------------------------------------------------


def small_rssm() -> RSSM:
    torch.manual_seed(0)
    return RSSM(*unit_norm(), units=16, latent=4, dec_hidden=8)


def test_rssm_kl_never_negative() -> None:
    g = torch.Generator().manual_seed(1)
    for _ in range(50):
        q = Gaussian(torch.randn(32, 4, generator=g), torch.rand(32, 4, generator=g) + 0.05)
        p = Gaussian(torch.randn(32, 4, generator=g), torch.rand(32, 4, generator=g) + 0.05)
        assert float(kl_divergence(q, p).min()) >= -1e-6
    q = Gaussian(torch.zeros(3, 4), torch.ones(3, 4))
    assert torch.allclose(kl_divergence(q, q), torch.zeros(3), atol=1e-7)


def rssm_window(B: int = 4, K: int = 12):
    g = torch.Generator().manual_seed(2)
    states = torch.randn(B, K, 11, generator=g)
    prev_a = torch.randn(B, K, 2, generator=g)
    prev_a[:, 0] = 0
    mask = torch.ones(B, K)
    z = torch.randn(B, K, 11, generator=g)
    scored = torch.zeros(B, K)
    scored[:, 1:] = 1
    return states, prev_a, mask, z, scored


def test_rssm_free_nats_clip_kl() -> None:
    m = small_rssm()
    m.free_nats = 1e6  # every step below the free nats: the KL term gives no gradient
    terms = m.elbo(*rssm_window(), generator=torch.Generator().manual_seed(3))
    assert terms.kl_clipped.item() == pytest.approx(1e6)
    m.zero_grad()
    (terms.loss - terms.recon).backward()
    assert all(p.grad is None or float(p.grad.abs().max()) == 0.0 for p in m.parameters())
    m.free_nats = 0.0
    terms = m.elbo(*rssm_window(), generator=torch.Generator().manual_seed(3))
    assert terms.kl_clipped.item() == pytest.approx(terms.kl.item(), rel=1e-6)


def test_rssm_elbo_terms_add_up() -> None:
    m = small_rssm()
    states, prev_a, mask, z, scored = rssm_window()
    terms = m.elbo(states, prev_a, mask, z, scored, generator=torch.Generator().manual_seed(4))
    assert terms.loss.item() == pytest.approx((terms.recon + m.kl_weight * terms.kl_clipped).item(), rel=1e-6)
    # Recompute by hand with the same samples.
    steps, _, _ = m.filter(states, prev_a, mask, torch.Generator().manual_seed(4))
    rec = kl = 0.0
    for k in range(1, states.shape[1]):
        h, zz, prior, post = (x.detach() if torch.is_tensor(x) else x for x in steps[k])
        err = (m.decode(h, zz) - z[:, k]) / c.RSSM_OBS_STD
        rec += (0.5 * (err * err).sum(-1)).mean().item()
        kl += kl_divergence(post, prior).mean().item()
    n = states.shape[1] - 1
    assert terms.recon.item() == pytest.approx(rec / n, rel=1e-5)
    assert terms.kl.item() == pytest.approx(kl / n, rel=1e-5)


def one_step_batch(n: int = 5, seed: int = 0) -> Batch:
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    return Batch(r(n, 11), r(n, 8, 11), r(n, 8, 2), torch.ones(n, 8), r(n, 2), r(n, 11), r(n, 2))


def test_rssm_fixed_sampling_seed_gives_identical_predictions() -> None:
    m = small_rssm()
    b = one_step_batch()
    m.sampling_seed = 123
    p1, p2 = m.predict_z(b), m.predict_z(b)
    assert torch.equal(p1, p2)
    m.sampling_seed = 124
    assert not torch.equal(p1, m.predict_z(b))


def test_rssm_medium_width_is_planet_size() -> None:
    assert size_for("rssm", "medium") == c.RSSM_MEDIUM_GRU_UNITS
    m = build("rssm", "medium", *unit_norm())
    assert m.units == 200 and m.latent == 30


# mlp-shortcut and blind --------------------------------------------------------------------------


def tiny_train_data() -> TrainData:
    sp = random_split()
    return TrainData.build(sp, None)


def short_cfg(arch: str, p: float | None = None, updates: int = 30) -> TrainConfig:
    return TrainConfig(
        arch=arch,
        width="small",
        seed=11,
        updates=updates,
        lr=1e-3,
        batch_size=64,
        seq_batch=4,
        shortcut_p=p,
        checkpoint_every=[10, 20, 30][: max(1, updates // 10)],
    )


def test_shortcut_swap_has_its_own_stream() -> None:
    """With p = 0, mlp-shortcut trains exactly as mlp (same init, same batches): the swap draws come
    from their own stream, so they change nothing else."""
    td = tiny_train_data()
    a = train(short_cfg("mlp"), td, evaluate_val=False)
    b = train(short_cfg("mlp-shortcut", 0.0), td, evaluate_val=False)
    assert [x["train_loss"] for x in a["curve"]] == [x["train_loss"] for x in b["curve"]]
    assert b["shortcut_swapped_share"] == 0.0
    assert len({stream_seed(11, s) for s in ("init", "batch", "shortcut", "rssm_train", "rssm_predict")}) == 5


def test_shortcut_swapped_share_is_p() -> None:
    td = tiny_train_data()
    r = train(short_cfg("mlp-shortcut", STANDIN_P, updates=200), td, evaluate_val=False)
    n = 200 * 64
    assert abs(r["shortcut_swapped_share"] - STANDIN_P) < 4 * math.sqrt(STANDIN_P * (1 - STANDIN_P) / n)


def test_shadow_actions_differ_from_actions_and_reach_training() -> None:
    td = tiny_train_data()
    assert td.train.shadow is not None
    assert not np.allclose(td.train.shadow, td.train.a)
    with pytest.raises(ValueError):
        train(short_cfg("mlp-shortcut", None), td, evaluate_val=False)


def test_blind_input_has_no_action() -> None:
    norm, ds = unit_norm()
    blind, mlp = build("blind", "small", norm, ds), build("mlp", "small", norm, ds)
    assert mlp_input_dim(False) == mlp_input_dim(True) - 2
    assert blind.inp.in_features == mlp_input_dim(False)
    b = one_step_batch(seed=5)
    other = b.with_action(b.a * -3.0 + 1.0, b.a_n * 7.0 - 2.0)
    assert blind.inputs(b).shape[1] == mlp.inputs(b).shape[1] - 2
    assert torch.equal(blind(b), blind(other))
    assert not torch.equal(mlp(b), mlp(other))


# Widths and checkpoints --------------------------------------------------------------------------


@pytest.mark.parametrize("arch", ["mlp", "gru", "contact_wm", "blind"])
@pytest.mark.parametrize(
    "width,target", list(zip(("small", "medium", "large"), c.WIDTH_PARAM_COUNTS, strict=True))
)
def test_widths_are_close_to_target(arch: str, width: str, target: int) -> None:
    m = build(arch, width, *unit_norm())
    assert abs(m.n_params() - target) / target < 0.03


def test_checkpoint_records_what_the_scorer_needs(tmp_path) -> None:
    td = tiny_train_data()
    cfg = short_cfg("mlp", updates=10)
    cfg.checkpoint_every = [10]
    train(cfg, td, ckpt_dir=tmp_path, evaluate_val=False)
    ck = torch.load(tmp_path / "u000010.pt", weights_only=False)
    meta = ck["meta"]
    assert meta["architecture"] == "mlp" and meta["width"] == "small" and meta["seed"] == 11
    assert meta["updates"] == 10 and meta["trained"] is True
    assert meta["phase"] == current_phase() and meta["phase"] in PHASES  # read from the freeze records
    assert meta["data"]["train"]["file"] == "random" and "sha256" in meta["data"]["train"]


def test_make_batch_zeroes_padded_actions() -> None:
    from physical_intervene.data import make_examples

    sp = random_split(n_ep=1)
    ex = make_examples(sp)
    norm = InputNorm(np.zeros(11), np.ones(11), np.full(2, 0.5), np.ones(2))
    b = make_batch(ex, norm)
    assert torch.all(b.ha_n[0] == 0)  # t = 0: the whole history is padding
    assert torch.all(b.ha_n[8] != 0)


# Exploration candidates (not in DESIGN.md) ---------------------------------------------------------


def test_pnn_predicts_its_mean_and_nll_has_gradients() -> None:
    norm, ds = unit_norm()
    m = build("pnn", "small", norm, ds)
    b = one_step_batch()
    mean, logvar = m.mean_logvar(b)
    assert torch.equal(m(b), mean) and torch.equal(m.predict_z(b), mean)
    assert torch.all(logvar <= m.max_logvar) and torch.all(logvar >= m.min_logvar)
    loss = m.nll(b, torch.zeros_like(mean), torch.full((11,), 1 / 11))
    loss.backward()
    assert all(p.grad is not None for p in m.parameters())
    assert abs(m.n_params() - c.WIDTH_PARAM_COUNTS[0]) / c.WIDTH_PARAM_COUNTS[0] < 0.03


def test_rssm_kl_balance_value_and_gradients() -> None:
    """DreamerV3's two KL terms have the same value as the plain KL; with no dynamics weight the KL
    sends no gradient to the prior network."""
    m = small_rssm()
    states, prev_a, mask, z, scored = rssm_window()
    m.free_nats = 0.0
    plain = m.elbo(states, prev_a, mask, z, scored, torch.Generator().manual_seed(0))
    m.kl_balance = (0.5, 0.1)
    bal = m.elbo(states, prev_a, mask, z, scored, torch.Generator().manual_seed(0))
    assert torch.allclose(bal.loss, plain.recon + 0.6 * plain.kl)
    m.kl_balance = (0.0, 1.0)  # representation term only: the prior is a fixed target
    m.zero_grad()
    kl_only = m.elbo(states, prev_a, mask, z, scored, torch.Generator().manual_seed(0))
    (kl_only.loss - kl_only.recon).backward()
    assert m.prior_net.weight.grad is None or torch.all(m.prior_net.weight.grad == 0)
    assert m.post_net.weight.grad is not None and torch.any(m.post_net.weight.grad != 0)


def test_multi_step_loss_first_step_is_the_shared_loss_and_feeds_back_predictions(monkeypatch) -> None:
    from physical_intervene import train as T

    sp = random_split(n_ep=3, T=30)
    td = TrainData.build(sp, None)
    torch.manual_seed(0)
    m = build("mlp-ms", "small", td.norm, td.dscaler)
    idx = np.array([0, 5, 31, 70])  # steps 0, 5, 1, 10
    w = T.loss_weights(None)
    monkeypatch.setattr(c, "MULTI_STEP_WEIGHTS", (1.0,))
    one = T.multi_step_loss(m, td, idx, w)
    assert torch.allclose(one, T.weighted_loss(m(td.batch(idx)), td.z(idx), w), rtol=1e-5)
    # two steps: the second step's input state is the first step's prediction, not the true state
    monkeypatch.setattr(c, "MULTI_STEP_WEIGHTS", (0.0, 1.0))
    two = T.multi_step_loss(m, td, idx, w)
    b0, b1 = td.batch(idx), td.batch(idx + 1)
    s1 = b0.s + m(b0) * m.d_scale + m.d_mean
    s1_n = (s1 - torch.as_tensor(td.norm.s_mean, dtype=torch.float32)) / torch.as_tensor(
        td.norm.s_std, dtype=torch.float32
    )
    hs_n = torch.cat([b0.hs_n[:, 1:], b0.s_n[:, None]], 1)
    ha_n = torch.cat([b0.ha_n[:, 1:], b0.a_n[:, None]], 1)  # the pair (s_t, a_t) joins the history
    mask = torch.cat([b0.mask[:, 1:], torch.ones(4, 1)], 1)
    s2 = s1 + m(Batch(s1_n, hs_n, ha_n, mask, b1.a_n, s1, b1.a)) * m.d_scale + m.d_mean
    err = (s2 - torch.as_tensor(sp.obs[[0, 0, 1, 2], [2, 7, 3, 12]], dtype=torch.float32)) / m.d_scale
    assert torch.allclose(two, ((err * err) * w).sum(-1).mean(), rtol=1e-4)


def test_made_up_rollout_episodes_match_the_one_step_targets() -> None:
    from physical_intervene import training_checks as TC
    from physical_intervene.data import make_examples

    sp = random_split(n_ep=3, T=20)
    for kind, one_step in (("zero_rollout", "zero"), ("action_rollout", "action")):
        ex = make_examples(TC.made_up_episodes(sp, kind))
        want = TC.made_up_targets(ex, *unit_norm())[one_step]
        assert np.allclose(ex.delta, want, atol=1e-12)
        assert np.allclose(ex.s[ex.step == 0], make_examples(sp).s[ex.step == 0])
