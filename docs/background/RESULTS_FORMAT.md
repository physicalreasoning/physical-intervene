# Results file format

**Background, not frozen.** This file explains or compares; it does not define the test and is
not part of freeze 1 or 2. Where it and [DESIGN.md](../DESIGN.md) differ, DESIGN.md governs.

The scorer writes one JSON, `results/intervention_results.json`, in the shared format drafted in
[INTERVENTION_REPORTING.md](../INTERVENTION_REPORTING.md). The mapping from evm-intervene's key
names is in [EVM_COMPARISON.md](EVM_COMPARISON.md).

E_int for G1 is the seed mean of `seed_checkpoints[*].matched_push_std_mse`, as in
evm-intervene. G1's gap (`matched_push_gap`, and every gap of the same test: `best_push_gap`,
the control pairs, G0.3, `gap_along_training`) is Welch's t-test on those per-seed values:
`ci_low`, `ci_high` and `p_two_sided` come from it, `df` is its Welch-Satterthwaite degrees of
freedom and `method` is `welch_t_over_seeds`. `matched_push_gap_states_only` is the bootstrap over
states with seed means fixed (`resampled: ["state"]`), evm-intervene's interval.

The `std_mse` in each model's stratum blocks is the error of the seed-averaged
prediction, reported and not gated.

Descriptive intervals, as in evm-intervene: `std_mse_ci95` resamples held-out states, on the
seed-averaged prediction; `matched_obs_val_ci95` resamples validation episodes, on the
seed-averaged L_obs. Both use 2,000 replicates, as in evm-intervene; neither is gated. G2's
interval and the states-only interval use 5,000.

Added keys, absent in evm-intervene's output: the shared format's optional keys
(`format_version`, `strata`, `units`, `resampled`, `method`, `gate_g0_valid`, `gate_g1_reading`,
`gate_g2_reading`), and `matched_push_gap_states_only` (evm's interval, no seed resampling),
`decoupled_by_interval`, `updates` (evm's `epochs`), `width`, `controls` (the six control
models, laid out as `models`), `control_pairs` (the five control pairs, laid out as
`pairwise_gate_g1` entries, each with a `role` of `positive` or `negative` and a `separates`
flag; twin pairs also carry `matched_push_gap_states_only`), `gates.gate_positive_control_pass`,
`gates.gate_negative_controls` (per architecture: `clean`, `separated` or `not_run`),
`gates.gate_g1_passing_pairs` (the main pairs that meet the pass conditions),
`logged_action` (per model, the logged action scored on the test states), and, from
DESIGN.md's matching rules, `matched_only` and `gap_along_training` per pair (levels and
reading) and `met_plateau_criterion` per run. `updates` at the top level is the largest
architecture's budget; each model block gives its own `updates`. Values below are
placeholders.

Added in Phase 0, next to the secondary analyses above (DESIGN.md, *Secondary analyses* and
*The structured model*); they do not change any gate:

- `pairwise_gate_g1_without_v_star_setter`: G1 without the architecture whose median sets V\*:
  `left_out`, its own `v_star` (the same rule on the other three medians), `holm_m` (3), `pairs`
  laid out as `pairwise_gate_g1` entries, `negative_controls_at_this_v_star`, `gate_g1_reading`
  and `gate_g1_passing_pairs`. The positive control is read at the primary V\*.
- `rank_correlation_gate_g2_general_purpose`: G2 over `mlp`, `gru` and `mlp-ms` only (90 runs),
  laid out as `rank_correlation_gate_g2`, with its own `gate_g2_reading`.
- `asymmetric_reading`, on each main pair with `contact_wm`: `structured_model`, `lower_at_v_star`
  (`structured` or `other`) and `reading`: `evidence_for_structure` (it has the lower E_int and the
  pair meets G1's conditions), `not_evidence_against` (it has the higher E_int and the pair meets
  them) or `no_separation`.
- `matched_all_contact_gap`, on each main pair: G1 repeated on all contact interventions (graze
  and push), Holm over the 6 pairs; and an `all_contact` block beside each stratum block.
- `per_seed_view`, on each main pair: each seed's E_int and the exact Mann-Whitney p.
- `matching` (V\*, the architecture that sets it, every family's matched checkpoints and
  exclusions) and `g0` (the numbers behind each G0 check).

```json
{
  "format_version": "0.1",
  "benchmark": "physical-intervene/pushing",
  "device": "cpu",
  "seeds": [11, 22, 33, 44, 55, 66, 77, 88, 99, 110, 121, 132, 143, 154, 165, 176, 187, 198, 209, 220],
  "updates": 0,
  "wall_seconds": 0.0,
  "primary": {
    "arm_name": "primary",
    "strata": {"hard": ["push"], "easy": ["no_contact", "graze"], "hard_block": "push"},
    "units": {"physical_unit": "corner_rmse_mm"},
    "counts": {
      "n_train_obs": 100000, "n_val_obs": 10000,
      "n_interventions_total": 24000, "n_held_out_checkpoints": 1000,
      "regime_counts": {"no_contact": 15000, "graze": 2000, "push": 7000}
    },
    "target_scaler_std": [0.0, 0.0, 0.0],
    "copy_obs_val_std_mse": 1.0,
    "baselines": {
      "copy": {
        "obs_val_std_mse": 1.0, "obs_val_corner_rmse_mm": 0.0,
        "interventions": {
          "all":        {"n": 24000, "std_mse": 0.0, "std_mse_ci95": [0.0, 0.0], "corner_rmse_mm": 0.0, "angle_rmse_deg": 0.0},
          "no_contact": {"n": 15000, "std_mse": 0.0, "std_mse_ci95": [0.0, 0.0], "corner_rmse_mm": 0.0, "angle_rmse_deg": 0.0},
          "graze":      {"n": 2000,  "std_mse": 0.0, "std_mse_ci95": [0.0, 0.0], "corner_rmse_mm": 0.0, "angle_rmse_deg": 0.0},
          "push":       {"n": 7000,  "std_mse": 0.0, "std_mse_ci95": [0.0, 0.0], "corner_rmse_mm": 0.0, "angle_rmse_deg": 0.0}
        }
      },
      "constvel": {}, "knn": {}
    },
    "controls": {"blind": {}, "mlp-shortcut": {}, "mlp-twin": {}, "gru-twin": {}, "mlp-ms-twin": {}, "contact_wm-twin": {}},
    "control_pairs": [
      {"role": "positive", "challenger": "mlp-shortcut", "reference": "mlp", "matched_within_5pct": true,
       "obs_val_rel_diff": 0.01, "separates": true,
       "matched_push_gap": {"gap": 0.0, "ci_low": 0.0, "ci_high": 0.0, "rel_gap": 0.0,
                               "p_two_sided": 1.0, "df": 0.0, "method": "welch_t_over_seeds"}},
      {"role": "negative", "challenger": "mlp-twin", "reference": "mlp", "matched_within_5pct": true,
       "obs_val_rel_diff": 0.01, "separates": false,
       "matched_push_gap": {"gap": 0.0, "ci_low": 0.0, "ci_high": 0.0, "rel_gap": 0.0,
                               "p_two_sided": 1.0, "df": 0.0, "method": "welch_t_over_seeds"},
       "matched_push_gap_states_only": {"gap": 0.0, "ci_low": 0.0, "ci_high": 0.0, "rel_gap": 0.0,
                                           "p_two_sided": 1.0, "resampled": ["state"]}}
    ],
    "models": {
      "gru": {
        "n_seeds": 20, "updates": 0,
        "matched_obs_val_std_mse_mean": 0.31, "matched_obs_val_std_mse_sd": 0.01,
        "matched_obs_val_ci95": [0.29, 0.33], "matched_obs_val_corner_rmse_mm": 0.0,
        "best_obs_val_std_mse_mean": 0.28, "best_obs_val_std_mse_sd": 0.01,
        "beats_copy_on_obs": true,
        "logged_action": {"n": 1000, "std_mse": 0.0, "corner_rmse_mm": 0.0},
        "matched_checkpoint_interventions": {"all": {}, "no_contact": {}, "graze": {}, "push": {}},
        "best_checkpoint_interventions": {"all": {}, "no_contact": {}, "graze": {}, "push": {}}
      },
      "mlp": {}, "mlp-ms": {}, "contact_wm": {}
    },
    "pairwise_gate_g1": [
      {"challenger": "gru", "reference": "mlp", "matched_within_5pct": true, "obs_val_rel_diff": 0.02,
       "matched_push_gap": {"gap": 0.9, "ci_low": 0.3, "ci_high": 1.6, "rel_gap": 0.6,
                               "p_two_sided": 0.004, "holm_p": 0.012, "df": 0.0, "method": "welch_t_over_seeds"},
       "matched_push_gap_states_only": {"gap": 0.9, "ci_low": 0.4, "ci_high": 1.5, "rel_gap": 0.6,
                                           "p_two_sided": 0.002, "resampled": ["state"]},
       "best_push_gap": {"gap": 0.0, "ci_low": 0.0, "ci_high": 0.0, "rel_gap": 0.0, "p_two_sided": 1.0},
       "matched_only": true,
       "gap_along_training": {"levels": [
          {"level": "2.0xV*", "obs_val_loss": 0.62, "gap": 0.0, "ci_low": 0.0, "ci_high": 0.0},
          {"level": "1.5xV*", "obs_val_loss": 0.47, "gap": 0.0, "ci_low": 0.0, "ci_high": 0.0},
          {"level": "V*", "obs_val_loss": 0.31, "gap": 0.0, "ci_low": 0.0, "ci_high": 0.0},
          {"level": "V*_pair", "obs_val_loss": 0.30, "gap": 0.0, "ci_low": 0.0, "ci_high": 0.0}],
         "reading": "closing"},
       "matched_all_contact_gap": {"gap": 0.0, "ci_low": 0.0, "ci_high": 0.0, "rel_gap": 0.0,
                                   "p_two_sided": 1.0, "holm_p": 1.0, "df": 0.0, "method": "welch_t_over_seeds"},
       "per_seed_view": {"gru": [0.0], "mlp": [0.0], "p_two_sided": 1.0}},
      {"challenger": "contact_wm", "reference": "mlp", "matched_within_5pct": true,
       "asymmetric_reading": {"structured_model": "contact_wm", "lower_at_v_star": "other",
                              "reading": "not_evidence_against"}}
    ],
    "pairwise_gate_g1_without_v_star_setter": {
      "left_out": "mlp-ms", "v_star": 0.0, "holm_m": 3, "pairs": [],
      "gate_g1_reading": "fail", "gate_g1_passing_pairs": []},
    "rank_correlation_gate_g2": {
      "n_checkpoints": 120,
      "spearman_obs_vs_push_std_mse": {"rho": 0.2, "p_value": 0.2, "ci_low": -0.1, "ci_high": 0.5},
      "spearman_obs_vs_push_corner_mm": {"rho": 0.2, "p_value": 0.2, "ci_low": -0.1, "ci_high": 0.5},
      "decoupled_by_interval": false
    },
    "rank_correlation_gate_g2_general_purpose": {
      "n_checkpoints": 90,
      "spearman_obs_vs_push_std_mse": {"rho": 0.2, "p_value": 0.2, "ci_low": -0.1, "ci_high": 0.5},
      "spearman_obs_vs_push_corner_mm": {"rho": 0.2, "p_value": 0.2, "ci_low": -0.1, "ci_high": 0.5},
      "decoupled_by_interval": false, "gate_g2_reading": "decoupled"},
    "gates": {"gate_g0_valid": true, "gate_g1_matched_separation_pass": true, "gate_g2_obs_rank_decoupled": true,
              "gate_positive_control_pass": true,
              "gate_negative_controls": {"mlp": "clean", "gru": "clean", "mlp-ms": "clean", "contact_wm": "clean"},
              "gate_g1_passing_pairs": [["gru", "mlp"]],
              "gate_g1_reading": "pass", "gate_g2_reading": "decoupled"},
    "seed_checkpoints": [
      {"family": "gru", "width": "medium", "seed": 11,
       "best_obs_val_loss": 0.28, "matched_obs_val_loss": 0.31, "met_plateau_criterion": true,
       "best_push_std_mse": 0.0, "matched_push_std_mse": 0.0,
       "best_push_corner_rmse_mm": 0.0, "matched_push_corner_rmse_mm": 0.0}
    ]
  }
}
```
