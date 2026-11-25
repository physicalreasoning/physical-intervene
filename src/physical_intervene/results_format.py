"""The result JSON (docs/background/RESULTS_FORMAT.md, docs/INTERVENTION_REPORTING.md) and a check
that a result conforms to it.

`build_primary` assembles one arm from scored families, whatever produced their predictions
(stand-ins in Phase 0; trained models after freeze 2, never before). Each family's `meta["predict"]`
returns its prediction for (seed index, checkpoint index), used for the descriptive blocks, the
corner errors and the logged-action reference.

Keys added in Phase 0 milestone 3, beyond RESULTS_FORMAT.md as written before it (flagged in the
milestone report and now listed in RESULTS_FORMAT.md):
- `pairwise_gate_g1_without_v_star_setter`: G1 without the architecture whose median sets V*
  (3 pairs, Holm over 3), with its own `v_star`, `left_out`, `pairs` and `gate_g1_reading`;
- `rank_correlation_gate_g2_general_purpose`: G2 over `mlp`, `gru` and `mlp-ms` only, laid out as
  `rank_correlation_gate_g2`, with its own `gate_g2_reading`;
- `asymmetric_reading` on each main pair with `contact_wm` (DESIGN.md, The structured model);
- `matched_all_contact_gap` on each main pair (G1 repeated on all contact interventions,
  DESIGN.md, Secondary analyses); `all_contact` in every stratum block;
- `per_seed_view` on each main pair (per-seed E_int and the exact Mann-Whitney p);
- `matching` (V*, the family that sets it, each family's matched checkpoints and exclusions) and
  `g0` (each G0 check's numbers).
"""

from __future__ import annotations

import numpy as np

from physical_intervene import constants as c
from physical_intervene import gates as G
from physical_intervene.metrics import corner_rmse_mm, plateau
from physical_intervene.scoring import (
    HARD,
    InterventionSet,
    Prediction,
    blocks,
    logged_action_block,
    obs_val_ci95,
)

FORMAT_VERSION = "0.1"
MAIN = list(c.MAIN_ARCHITECTURES)
GENERAL_PURPOSE = ["mlp", "gru", "mlp-ms"]
TWINS = {a: f"{a}-twin" for a in MAIN}
GAP_KEYS = ("gap", "ci_low", "ci_high", "rel_gap", "p_two_sided")


def _gap(g: dict, holm: bool = False) -> dict:
    out = {k: g[k] for k in GAP_KEYS}
    if holm:
        out["holm_p"] = g.get("holm_p")
    for k in ("df", "method", "resampled"):  # G1's test: df, method; the states-only bootstrap: resampled
        if k in g:
            out[k] = g[k]
    return out


def _seed_avg_prediction(fam: G.Family, m: G.Match) -> Prediction:
    preds = [fam.meta["predict"](i, int(m.idx[i])) for i in np.flatnonzero(m.keep)]
    return Prediction(
        val=np.mean([p.val for p in preds], axis=0),
        interventions=np.mean([p.interventions for p in preds], axis=0),
    )


def model_block(
    fam: G.Family,
    m: G.Match,
    scaler,
    iset: InterventionSet,
    val_s: np.ndarray,
    val_y: np.ndarray,
    val_episode: np.ndarray,
    copy_l_obs: float,
    rng: np.random.Generator,
    updates: int,
) -> dict:
    keep = m.keep
    best = G.best_match(fam)
    pb = _seed_avg_prediction(fam, best)
    bc = G.beats_copy(fam, m, copy_l_obs)
    out = {
        "n_seeds": int(keep.sum()),
        "updates": updates,
        "matched_obs_val_std_mse_mean": m.mean_l if keep.any() else None,
        "matched_obs_val_std_mse_sd": float(np.std(m.l[keep], ddof=1)) if keep.sum() > 1 else None,
        "matched_obs_val_ci95": None,
        "matched_obs_val_corner_rmse_mm": None,
        "best_obs_val_std_mse_mean": float(fam.best_l().mean()),
        "best_obs_val_std_mse_sd": float(np.std(fam.best_l(), ddof=1)),
        "beats_copy_on_obs": bool(bc["matched"] and bc["best"]),
        "beats_copy_on_obs_at": {"matched": bc["matched"], "best": bc["best"]},
        "excluded_seeds": [int(s) for s, e in zip(fam.seeds, m.excluded, strict=True) if e],
        "dropped_from_g1": m.dropped,
        "logged_action": logged_action_block(scaler, iset, pb.interventions),
        "logged_action_at": "matched",
        "matched_checkpoint_interventions": None,
        "best_checkpoint_interventions": blocks(scaler, iset, pb.interventions, rng),
    }
    if keep.any():  # a model with every seed excluded (blind, by construction) has no matched blocks
        pm = _seed_avg_prediction(fam, m)
        per_val = np.mean(
            [
                scaler.per_sample(fam.meta["predict"](i, int(m.idx[i])).val, val_y)
                for i in np.flatnonzero(keep)
            ],
            axis=0,
        )
        out["matched_obs_val_ci95"] = obs_val_ci95(per_val, val_episode, rng)
        out["matched_obs_val_corner_rmse_mm"] = corner_rmse_mm(val_s, pm.val, val_y)
        out["logged_action"] = logged_action_block(scaler, iset, pm.interventions)
        out["matched_checkpoint_interventions"] = blocks(scaler, iset, pm.interventions, rng)
    else:
        out["logged_action_at"] = "best (no seed reached V*)"
    return out


def baseline_block(
    pred: Prediction, scaler, iset: InterventionSet, val_s: np.ndarray, val_y: np.ndarray, rng
) -> dict:
    return {
        "obs_val_std_mse": scaler.std_mse(pred.val, val_y),
        "obs_val_corner_rmse_mm": corner_rmse_mm(val_s, pred.val, val_y),
        "interventions": blocks(scaler, iset, pred.interventions, rng),
    }


def seed_rows(fam: G.Family, m: G.Match, ctx_scaler, iset: InterventionSet) -> list[dict]:
    best = G.best_match(fam)
    push = iset.mask(HARD)
    s_rep = np.broadcast_to(iset.s[:, None, :], (*push.shape, iset.s.shape[1]))[push]
    rows = []
    for i, seed in enumerate(fam.seeds):
        row = {
            "family": fam.name,
            "width": fam.width,
            "seed": int(seed),
            "best_obs_val_loss": float(fam.best_l()[i]),
            "matched_obs_val_loss": float(m.l[i]),
            "matched_excluded": bool(m.excluded[i]),
            "met_plateau_criterion": plateau(fam.updates, list(fam.l_obs[i]))["met_at_last_checkpoint"],
            "best_push_std_mse": float(fam.run_e(HARD, best.idx)[i]),
            "matched_push_std_mse": float(fam.run_e(HARD, m.idx)[i]),
        }
        for tag, idx in (("best", best.idx), ("matched", m.idx)):
            y = fam.meta["predict"](i, int(idx[i])).interventions
            row[f"{tag}_push_corner_rmse_mm"] = corner_rmse_mm(s_rep, y[push], iset.y_true[push])
        rows.append(row)
    return rows


def g2_block(fams: list[G.Family], iset: InterventionSet, rng: np.random.Generator) -> dict:
    """Spearman over runs at best checkpoints (std-MSE gated; corner error reported)."""
    x, y, ymm = [], [], []
    push = iset.mask(HARD)
    s_rep = np.broadcast_to(iset.s[:, None, :], (*push.shape, iset.s.shape[1]))[push]
    for f in fams:
        b = f.best_idx()
        x += list(f.best_l())
        y += list(f.run_e(HARD, b))
        for i in range(f.n_seeds):
            yp = f.meta["predict"](i, int(b[i])).interventions
            ymm.append(corner_rmse_mm(s_rep, yp[push], iset.y_true[push]))
    std = G.spearman_boot(np.array(x), np.array(y), rng)
    mm = G.spearman_boot(np.array(x), np.array(ymm), rng)
    reading = G.g2_reading(std["rho"])
    return {
        "n_checkpoints": len(x),
        "spearman_obs_vs_push_std_mse": std,
        "spearman_obs_vs_push_corner_mm": mm,
        "decoupled_by_interval": bool(std["ci_high"] < c.G2_DECOUPLED_BELOW),
        "gate_g2_reading": reading,
    }


def pair_entry(p: dict) -> dict:
    out = {
        "challenger": p["challenger"],
        "reference": p["reference"],
        "tested": p["tested"],
    }
    if not p["tested"]:
        out.update(
            {
                "matched_within_5pct": False,
                "obs_val_rel_diff": None,
                "matched_push_gap": None,
                "not_tested_because": p["not_tested_because"],
            }
        )
        return out
    out.update(
        {
            "matched_within_5pct": p["matched_within_5pct"],
            "obs_val_rel_diff": p["obs_val_rel_diff"],
            "matched_push_gap": _gap(p["gap"], holm="holm_p" in p["gap"]),
        }
    )
    if "gap_states_only" in p:
        out["matched_push_gap_states_only"] = _gap(p["gap_states_only"])
    return out


def build_primary(
    fams: dict[str, G.Family],
    widths: dict[str, dict[str, G.Family]],
    scaler,
    iset: InterventionSet,
    val_s: np.ndarray,
    val_y: np.ndarray,
    val_episode: np.ndarray,
    baselines: dict[str, Prediction | None],
    counts: dict,
    g0_replay: dict,
    rng: np.random.Generator,
    updates: int,
    reps: int = c.BOOTSTRAP_REPLICATES,
) -> dict:
    """The primary arm. `fams` holds the medium-width main families and the controls by name
    (`blind`, `mlp-shortcut`, `<arch>-twin`); `widths` the small and large main families, for G2."""
    v_star, setter = G.v_star_of([fams[a] for a in MAIN])
    copy_l = scaler.std_mse(baselines["copy"].val, val_y)

    # Controls first: their readings enter G1's verdict.
    twin_pairs, twin_status = {}, {}
    for a in MAIN:
        p, st = G.twin_pair(fams, a, TWINS[a], v_star, rng, reps)
        twin_pairs[a], twin_status[a] = p, st
    pos = G.positive_pair(fams, "mlp", "mlp-shortcut", v_star, rng, reps)
    positive_pass = bool(pos["separates"])

    g1 = G.run_g1(fams, MAIN, v_star, rng, reps)
    g1_all_contact = G.run_g1(fams, MAIN, v_star, rng, reps, block="all_contact")
    reading, passing = G.g1_reading(g1["pairs"], twin_status, positive_pass)
    pairs = []
    for p, pc in zip(g1["pairs"], g1_all_contact["pairs"], strict=True):
        e = pair_entry(p)
        if p["tested"]:
            orient = (p["challenger"], p["reference"])
            fa, fb = fams[p["challenger"]], fams[p["reference"]]
            ma, mb = g1["matches"][fa.name], g1["matches"][fb.name]
            best = G.compare(
                fa, G.best_match(fa), fb, G.best_match(fb), rng, reps, states_only=False, orient=orient
            )
            e["best_push_gap"] = _gap(best["gap"])
            e["meets_g1_conditions"] = p["meets"]
            e["matched_only"] = bool(
                p["meets"] and p["matched_within_5pct"] and not G.excludes_zero(best["gap"])
            )
            e["gap_along_training"] = G.gap_along_training(fa, fb, v_star, orient, rng, reps)
            e["matched_all_contact_gap"] = _gap(pc["gap"], holm=True) if pc["tested"] else None
            e["per_seed_view"] = G.mann_whitney(fa, ma, fb, mb)
            ar = G.asymmetric_reading(p)
            if ar is not None:
                e["asymmetric_reading"] = ar
        pairs.append(e)

    # G1 without the architecture that sets V*.
    rest = [a for a in MAIN if a != setter]
    v2, _ = G.v_star_of([fams[a] for a in rest])
    g1b = G.run_g1(fams, rest, v2, rng, reps)
    twin_b = {a: G.twin_pair(fams, a, TWINS[a], v2, rng, reps)[1] for a in rest}
    reading_b, passing_b = G.g1_reading(g1b["pairs"], twin_b, positive_pass)
    without = {
        "left_out": setter,
        "v_star": v2,
        "holm_m": g1b["holm_m"],
        "pairs": [pair_entry(p) | {"meets_g1_conditions": p["meets"]} for p in g1b["pairs"]],
        "negative_controls_at_this_v_star": twin_b,
        "positive_control_from": "primary V*",
        "gate_g1_reading": reading_b,
        "gate_g1_passing_pairs": passing_b,
    }

    # G0.
    g0_copy = {a: G.beats_copy(fams[a], g1["matches"][a], copy_l) for a in MAIN}
    g0_blind = G.g0_blind(fams, "blind", "mlp", rng, reps)
    g0_valid = bool(
        g0_replay["passed"] and all(v["matched"] and v["best"] for v in g0_copy.values()) and g0_blind["pass"]
    )

    # G2: all main runs at all widths; and the general-purpose runs only.
    all_runs = [fams[a] for a in MAIN] + [widths[w][a] for w in widths for a in MAIN]
    gp_runs = [f for f in all_runs if f.name in GENERAL_PURPOSE]
    g2 = g2_block(all_runs, iset, rng)
    g2_gp = g2_block(gp_runs, iset, rng)

    models = {
        a: model_block(
            fams[a], g1["matches"][a], scaler, iset, val_s, val_y, val_episode, copy_l, rng, updates
        )
        for a in MAIN
    }
    controls = {}
    for name in ["blind", "mlp-shortcut", *TWINS.values()]:
        controls[name] = model_block(
            fams[name],
            G.match(fams[name], v_star),
            scaler,
            iset,
            val_s,
            val_y,
            val_episode,
            copy_l,
            rng,
            updates,
        )
    control_pairs = [pair_entry(pos) | {"role": "positive", "separates": positive_pass}]
    for a in MAIN:
        p = twin_pairs[a]
        control_pairs.append(
            pair_entry(p)
            | {"role": "negative", "separates": p.get("separates", False), "status": twin_status[a]}
        )

    seeds = []
    for a in MAIN:
        seeds += seed_rows(fams[a], g1["matches"][a], scaler, iset)
    for fw in widths.values():
        vw, _ = G.v_star_of([fw[a] for a in MAIN])
        for a in MAIN:
            seeds += seed_rows(fw[a], G.match(fw[a], vw, c.G1_MIN_SEEDS_SECONDARY_WIDTHS), scaler, iset)

    base = {}
    for name, pred in baselines.items():
        base[name] = None if pred is None else baseline_block(pred, scaler, iset, val_s, val_y, rng)

    return {
        "arm_name": "primary",
        "strata": {"hard": ["push"], "easy": ["no_contact", "graze"], "hard_block": "push"},
        "units": {"physical_unit": "corner_rmse_mm"},
        "counts": counts,
        "target_scaler_std": scaler.scale.tolist(),
        "copy_obs_val_std_mse": copy_l,
        "baselines": base,
        "models": models,
        "controls": controls,
        "control_pairs": control_pairs,
        "pairwise_gate_g1": pairs,
        "pairwise_gate_g1_without_v_star_setter": without,
        "rank_correlation_gate_g2": {k: v for k, v in g2.items() if k != "gate_g2_reading"},
        "rank_correlation_gate_g2_general_purpose": g2_gp,
        "gates": {
            "gate_g0_valid": g0_valid,
            "gate_g1_matched_separation_pass": reading == "pass",
            "gate_g2_obs_rank_decoupled": g2["gate_g2_reading"] == "decoupled",
            "gate_positive_control_pass": positive_pass,
            "gate_negative_controls": {a: twin_status[a] for a in MAIN},
            "gate_g1_passing_pairs": passing,
            "gate_g1_reading": reading,
            "gate_g2_reading": g2["gate_g2_reading"],
        },
        "matching": {
            "v_star": v_star,
            "set_by": setter,
            "families": {n: G.match(fams[n], v_star).to_dict(fams[n]) for n in fams},
        },
        "g0": {
            "replay": g0_replay,
            "beats_copy": g0_copy,
            "blind": pair_entry(g0_blind) | {"pass": g0_blind["pass"]},
        },
        "seed_checkpoints": seeds,
    }


# Conformance -------------------------------------------------------------------------------------

STRATUM_KEYS = ("n", "std_mse", "std_mse_ci95", "corner_rmse_mm", "angle_rmse_deg")
MODEL_KEYS = (
    "n_seeds",
    "updates",
    "matched_obs_val_std_mse_mean",
    "matched_obs_val_std_mse_sd",
    "matched_obs_val_ci95",
    "best_obs_val_std_mse_mean",
    "best_obs_val_std_mse_sd",
    "beats_copy_on_obs",
    "logged_action",
    "matched_checkpoint_interventions",
    "best_checkpoint_interventions",
)
GAP_REQUIRED = ("gap", "ci_low", "ci_high", "rel_gap", "p_two_sided")
SEED_KEYS = (
    "family",
    "width",
    "seed",
    "best_obs_val_loss",
    "matched_obs_val_loss",
    "met_plateau_criterion",
    "best_push_std_mse",
    "matched_push_std_mse",
    "best_push_corner_rmse_mm",
    "matched_push_corner_rmse_mm",
)
SPEARMAN_KEYS = ("rho", "p_value", "ci_low", "ci_high")


def _need(d: dict, keys, where: str, errors: list[str]) -> None:
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            errors.append(f"{where}: missing {k}")


def _stratum_blocks(d: dict, where: str, errors: list[str]) -> None:
    _need(d, ("all", "no_contact", "graze", "push"), where, errors)
    for b in ("all", "no_contact", "graze", "push"):
        if isinstance(d, dict) and isinstance(d.get(b), dict):
            _need(d[b], STRATUM_KEYS, f"{where}.{b}", errors)


def check_conformance(result: dict) -> list[str]:
    """Errors against INTERVENTION_REPORTING.md and RESULTS_FORMAT.md (empty list: conforms). Checks
    every key either document lists, the allowed reading values, and that the readings agree with
    the keys they are read from."""
    e: list[str] = []
    _need(
        result,
        ("format_version", "benchmark", "device", "seeds", "updates", "wall_seconds", "primary"),
        "top",
        e,
    )
    arm = result.get("primary", {})
    _need(
        arm,
        (
            "arm_name",
            "strata",
            "units",
            "counts",
            "target_scaler_std",
            "copy_obs_val_std_mse",
            "baselines",
            "models",
            "controls",
            "control_pairs",
            "pairwise_gate_g1",
            "rank_correlation_gate_g2",
            "gates",
            "seed_checkpoints",
        ),
        "primary",
        e,
    )
    _need(
        arm.get("counts", {}),
        ("n_train_obs", "n_val_obs", "n_interventions_total", "n_held_out_checkpoints", "regime_counts"),
        "counts",
        e,
    )
    if set(arm.get("counts", {}).get("regime_counts", {})) != {"no_contact", "graze", "push"}:
        e.append("counts.regime_counts: keys must be no_contact, graze, push")
    if len(arm.get("target_scaler_std", [])) != 3:
        e.append("target_scaler_std: needs 3 values")
    for name, b in arm.get("baselines", {}).items():
        if b is None:
            continue
        _need(b, ("obs_val_std_mse", "interventions"), f"baselines.{name}", e)
        _stratum_blocks(b.get("interventions", {}), f"baselines.{name}.interventions", e)
    for group in ("models", "controls"):
        for name, m in arm.get(group, {}).items():
            _need(m, MODEL_KEYS, f"{group}.{name}", e)
            for k in ("matched_checkpoint_interventions", "best_checkpoint_interventions"):
                if m.get(k) is None and k.startswith("matched") and m.get("n_seeds") == 0:
                    continue  # null where it does not apply: no seed reached V*
                _stratum_blocks(m.get(k, {}), f"{group}.{name}.{k}", e)
            _need(
                m.get("logged_action", {}),
                ("n", "std_mse", "corner_rmse_mm"),
                f"{group}.{name}.logged_action",
                e,
            )
    if set(arm.get("models", {})) != set(MAIN):
        e.append(f"models: expected {MAIN}")
    if set(arm.get("controls", {})) != {"blind", "mlp-shortcut", *TWINS.values()}:
        e.append("controls: expected blind, mlp-shortcut and the four twins")
    for i, p in enumerate(arm.get("pairwise_gate_g1", [])):
        w = f"pairwise_gate_g1[{i}]"
        _need(
            p,
            ("challenger", "reference", "matched_within_5pct", "obs_val_rel_diff", "matched_push_gap"),
            w,
            e,
        )
        if p.get("tested", True):
            _need(
                p,
                ("best_push_gap", "matched_push_gap_states_only", "matched_only", "gap_along_training"),
                w,
                e,
            )
            _need(
                p.get("matched_push_gap") or {},
                (*GAP_REQUIRED, "holm_p", "method", "df"),
                w + ".matched_push_gap",
                e,
            )
            _need(
                p.get("matched_push_gap_states_only") or {},
                ("resampled",),
                w + ".matched_push_gap_states_only",
                e,
            )
            if (p.get("matched_push_gap") or {}).get("method") != c.G1_TEST:
                e.append(f"{w}.matched_push_gap.method: expected {c.G1_TEST}")
            _need(p.get("best_push_gap") or {}, GAP_REQUIRED, w + ".best_push_gap", e)
            gat = p.get("gap_along_training", {})
            _need(gat, ("levels", "reading"), w + ".gap_along_training", e)
            if gat.get("reading") not in G.CURVE_READINGS:
                e.append(f"{w}.gap_along_training.reading: {gat.get('reading')!r}")
    if len(arm.get("pairwise_gate_g1", [])) != 6:
        e.append("pairwise_gate_g1: expected the 6 main pairs")
    for i, p in enumerate(arm.get("control_pairs", [])):
        w = f"control_pairs[{i}]"
        _need(
            p,
            ("role", "challenger", "reference", "matched_within_5pct", "separates", "matched_push_gap"),
            w,
            e,
        )
        if p.get("role") == "negative" and p.get("tested", True):
            _need(p, ("matched_push_gap_states_only",), w, e)
    g2 = arm.get("rank_correlation_gate_g2", {})
    _need(
        g2,
        (
            "n_checkpoints",
            "spearman_obs_vs_push_std_mse",
            "spearman_obs_vs_push_corner_mm",
            "decoupled_by_interval",
        ),
        "g2",
        e,
    )
    for k in ("spearman_obs_vs_push_std_mse", "spearman_obs_vs_push_corner_mm"):
        _need(g2.get(k, {}), SPEARMAN_KEYS, f"g2.{k}", e)
    gates = arm.get("gates", {})
    _need(
        gates,
        (
            "gate_g0_valid",
            "gate_g1_matched_separation_pass",
            "gate_g2_obs_rank_decoupled",
            "gate_positive_control_pass",
            "gate_negative_controls",
            "gate_g1_passing_pairs",
            "gate_g1_reading",
            "gate_g2_reading",
        ),
        "gates",
        e,
    )
    for i, r in enumerate(arm.get("seed_checkpoints", [])):
        _need(r, SEED_KEYS, f"seed_checkpoints[{i}]", e)
    if e:
        return e

    # Values and consistency.
    if gates["gate_g1_reading"] not in G.READINGS:
        e.append(f"gate_g1_reading: {gates['gate_g1_reading']!r}")
    if gates["gate_g2_reading"] not in ("decoupled", "inconclusive", "refuted"):
        e.append(f"gate_g2_reading: {gates['gate_g2_reading']!r}")
    if set(gates["gate_negative_controls"].values()) - set(G.TWIN_STATUS):
        e.append("gate_negative_controls: values must be clean, separated or not_run")
    if gates["gate_g1_matched_separation_pass"] != (gates["gate_g1_reading"] == "pass"):
        e.append("gate_g1_matched_separation_pass disagrees with gate_g1_reading")
    if gates["gate_g2_obs_rank_decoupled"] != (gates["gate_g2_reading"] == "decoupled"):
        e.append("gate_g2_obs_rank_decoupled disagrees with gate_g2_reading")
    if G.g2_reading(g2["spearman_obs_vs_push_std_mse"]["rho"]) != gates["gate_g2_reading"]:
        e.append("gate_g2_reading disagrees with the rho it is read from")
    if all(not p["matched_within_5pct"] for p in arm["pairwise_gate_g1"]) != (
        gates["gate_g1_reading"] == "not_testable"
    ):
        e.append("gate_g1_reading not_testable must hold exactly when no pair passes the match screen")
    # E_int of a reference equals gap / rel_gap, and equals the seed mean of matched_push_std_mse.
    e_seed = {}
    for r in arm["seed_checkpoints"]:
        if r["width"] == "medium" and not r.get("matched_excluded", False):
            e_seed.setdefault(r["family"], []).append(r["matched_push_std_mse"])
    for p in arm["pairwise_gate_g1"]:
        g = p.get("matched_push_gap")
        if g and g["rel_gap"] not in (0, None) and np.isfinite(g["rel_gap"]):
            via_gap = g["gap"] / g["rel_gap"]
            via_seeds = float(np.mean(e_seed[p["reference"]]))
            if not np.isclose(via_gap, via_seeds, rtol=1e-9):
                e.append(f"E_int of {p['reference']}: gap/rel_gap {via_gap} vs seed mean {via_seeds}")
    for p in arm["pairwise_gate_g1"]:
        for k in ("matched_push_gap", "matched_push_gap_states_only"):
            g = p.get(k)
            if g and not (g["ci_low"] <= g["ci_high"] and 0 < g["p_two_sided"] <= 1):
                e.append(f"{p['challenger']}/{p['reference']}.{k}: interval or p out of range")
    return e
