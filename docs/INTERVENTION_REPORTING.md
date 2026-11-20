# Intervention tests: shared reporting format

**Status (2026-11-16): a draft for discussion
([physicalreasoning/forward#5](https://github.com/physicalreasoning/forward/issues/5)), not
frozen.** It proposes one reporting format for every intervention test that answers forward#5,
so that their results read side by side. Two repos would use it:

- [evm-intervene](https://github.com/physicalreasoning/evm-intervene): markets, Uniswap V3 with
  Aave liquidations. It has results, in `results/mainnet_intervention_results.json`.
- physical-intervene (this repo): planar pushing in MuJoCo. It has a draft design and a
  scaffold, no data or models yet.

Today the only format is what evm-intervene's `scripts/run_experiment.py` writes. Nothing
describes it, so any commit could change it unnoticed. This draft writes that output down as
the format. **evm-intervene's current output already conforms; it needs no change.** Where the
format needs something evm-intervene does not write as a key, this draft says how to read it
from the keys evm-intervene already has. Suggested changes to evm-intervene are listed at the
end and are optional. Each repo then points here, not at the other's output.

## What every test reports

Each test fixes these before training, in its own design file.

| Term | Definition |
| --- | --- |
| state | a held-out initial state from which the simulator is reset and run under each intervention (evm: one held-out archive checkpoint; physical: one saved simulator state) |
| hard stratum | the interventions the test is about (evm: tick-cliff and cascade; physical: contact). Gated. |
| easy stratum | the remaining interventions (evm: in-tick; physical: no contact). Reported, not gated. |
| std-MSE | per-transition mean over target dimensions of the squared error, each divided by that dimension's standard deviation on the **training split only**, with stated floors |
| physical-unit error | an RMSE in the test's own unit (evm: log-price, bps; physical: box corners, mm). Reported, not gated. |
| L_obs, validation loss | std-MSE on held-out observational transitions |
| E_int, intervention error | std-MSE on the hard stratum: each transition's loss averaged over seeds, then the mean over all hard-stratum transitions, pooled across states |
| best checkpoint | per run, the checkpoint with the lowest L_obs |
| matched checkpoint | per run, the checkpoint whose L_obs is closest to a matching target fixed before any test data exists. Each test states its target rule. |
| match screen | two models pass if \|L_a − L_b\| / min(L_a, L_b) ≤ 0.05, using seed-mean matched L_obs |
| consistency gap, Δ_int | E_int(challenger) − E_int(reference) at matched checkpoints, on the hard stratum |
| relative gap | Δ_int / E_int(reference) |
| interval | 95% percentile interval from 5,000 paired bootstrap replicates, resampling held-out states with replacement, shared across the two models. A test may also resample other units (e.g. seeds); the `resampled` key, or this document for evm-intervene, says which. |
| p | min(1, 2 × (min(#Δ\* ≤ 0, #Δ\* ≥ 0) + 1) / 5,001), Holm-adjusted across **every pair tested**, whether or not it passes the match screen |
| rank correlation | Spearman ρ(L_obs, E_int) across all main runs at best checkpoints, with a 95% interval from 5,000 bootstrap resamples of runs |

**Gates.** The rules below are common to every test. Each test fixes in its design what
it checks for G0, its matching target, and which units its gated interval resamples.

- **G0, validity.** The simulator reproduces its ground truth, and every main learned model
  beats `copy` on L_obs. Each test states its own G0 checks and which checkpoints they use.
- **G1, matched separation.** Reading:
  - *pass*: at least one pair passes the match screen and has Holm p < 0.05, an interval
    excluding zero and a relative gap ≥ 0.20 (evm's code checks `ci_low > 0`; the same thing
    here, since a negative gap cannot reach a relative gap of 0.20);
  - *not testable*: no pair passes the match screen;
  - *fail*: otherwise.
- **G2, rank correlation.** Reading, on the point estimate of ρ for std-MSE: *decoupled* if
  ρ < 0.50, *refuted* if ρ ≥ 0.80, *inconclusive* in between. The interval is always reported.

A failure at any gate is published in the same format.

## File

One JSON per test, in `results/`. Top-level metadata, then one block per analysis arm. Keys
come from evm-intervene's current output unless marked **optional**. The example uses
evm-intervene's names.

```json
{
  "benchmark": "evm-intervene",
  "device": "cuda",
  "seeds": [11, 22, 33, 44, 55],
  "wall_seconds": 0.0,
  "<arm>": {
    "arm_name": "primary",
    "counts": {
      "n_train_obs": 0, "n_val_obs": 0,
      "n_interventions_total": 0, "n_held_out_checkpoints": 0,
      "regime_counts": {"<stratum>": 0}
    },
    "target_scaler_std": [0.0, 0.0, 0.0],
    "copy_obs_val_std_mse": 0.0,
    "baselines": {
      "<name>": {
        "obs_val_std_mse": 0.0,
        "interventions": {
          "all":          {"n": 0, "std_mse": 0.0, "std_mse_ci95": [0.0, 0.0], "<unit>": 0.0},
          "<stratum>":    {"n": 0, "std_mse": 0.0, "std_mse_ci95": [0.0, 0.0], "<unit>": 0.0},
          "<hard block>": {"n": 0, "std_mse": 0.0, "std_mse_ci95": [0.0, 0.0], "<unit>": 0.0}
        }
      }
    },
    "models": {
      "<family>": {
        "n_seeds": 5,
        "matched_obs_val_std_mse_mean": 0.0, "matched_obs_val_std_mse_sd": 0.0,
        "matched_obs_val_ci95": [0.0, 0.0],
        "best_obs_val_std_mse_mean": 0.0, "best_obs_val_std_mse_sd": 0.0,
        "beats_copy_on_obs": true,
        "matched_checkpoint_interventions": {"all": {}, "<stratum>": {}, "<hard block>": {}},
        "best_checkpoint_interventions": {"all": {}, "<stratum>": {}, "<hard block>": {}}
      }
    },
    "pairwise_gate_g1": [
      {"challenger": "<family>", "reference": "<family>",
       "matched_within_5pct": true, "obs_val_rel_diff": 0.0,
       "matched_<hard>_gap": {"gap": 0.0, "ci_low": 0.0, "ci_high": 0.0, "rel_gap": 0.0,
                              "p_two_sided": 1.0, "holm_p": 1.0},
       "best_<hard>_gap": {"gap": 0.0, "ci_low": 0.0, "ci_high": 0.0, "rel_gap": 0.0, "p_two_sided": 1.0}}
    ],
    "rank_correlation_gate_g2": {
      "n_checkpoints": 0,
      "spearman_obs_vs_<hard>_std_mse": {"rho": 0.0, "p_value": 1.0, "ci_low": 0.0, "ci_high": 0.0},
      "spearman_obs_vs_<hard>_<unit>": {"rho": 0.0, "p_value": 1.0, "ci_low": 0.0, "ci_high": 0.0}
    },
    "gates": {
      "gate_g1_matched_separation_pass": false,
      "gate_g2_obs_rank_decoupled": true
    },
    "seed_checkpoints": [
      {"family": "<family>", "seed": 11,
       "best_obs_val_loss": 0.0, "matched_obs_val_loss": 0.0,
       "best_<hard>_std_mse": 0.0, "matched_<hard>_std_mse": 0.0}
    ]
  }
}
```

### Names per test

Keys use each test's own names. This table maps them, so a reader can line the two files up.

| Placeholder | evm-intervene | physical-intervene |
| --- | --- | --- |
| `<stratum>` (stratum blocks, `regime_counts`) | `regime_1_smooth`, `regime_2_cliff`, `regime_3_cascade` | `no_contact`, `contact` |
| `<hard block>`, the gated stratum block | `cliff_and_cascade` | `contact` |
| `<hard>`, in gap, Spearman and seed keys | `cliff` | `contact` |
| `<unit>`, physical-unit error | `price_rmse_bps` (`price_bps` in Spearman keys only) | `corner_rmse_mm` (`corner_mm` in Spearman keys only) |
| training length, top level | `epochs` | `updates` |
| other per-stratum errors | `liquidity_rmse` | `angle_rmse_deg` |
| G1 reference | the anchor, `conservation_wm` | in each pair, the member with the lower E_int |
| interval resamples | held-out states (`n_held_out_checkpoints` of them); seeds fixed | held-out states and seeds; the states-only interval is reported beside it |

### Where each quantity is read

| Quantity | Key |
| --- | --- |
| L_obs, matched | `models.<family>.matched_obs_val_std_mse_mean` |
| E_int, matched (the quantity G1 tests) | mean over seeds of `seed_checkpoints[family].matched_<hard>_std_mse`; equals `gap / rel_gap` of any pair with that family as reference |
| Δ_int, interval, p | `pairwise_gate_g1[*].matched_<hard>_gap` |
| ρ for G2 | `rank_correlation_gate_g2.spearman_obs_vs_<hard>_std_mse.rho` |

### Descriptive intervals

Two intervals in the file are descriptive, not part of any gate, and are computed differently
from the gated `interval`. Both use 2,000 bootstrap replicates, not 5,000:

- `std_mse_ci95`, in each stratum block: resamples held-out states, on the std-MSE of the
  seed-averaged prediction (for baselines, the single prediction);
- `matched_obs_val_ci95`: resamples blocks of validation transitions, on the per-transition
  L_obs averaged over seeds.

Each test states how it computes them.

### Not E_int

`models.<family>.*_checkpoint_interventions.<block>.std_mse` is **not** E_int: it is the
std-MSE of the seed-averaged *prediction*, which is lower than the seed-averaged loss. It is
reported, not gated. In evm-intervene's primary arm the two are 22,563 and 22,601 for
`conservation_wm`.

### Reading the gates from the keys

Every test can be read from the keys above; a test may also write the reading directly.

| Gate | Read from |
| --- | --- |
| G0 | the test's own G0 record (evm: `gate_g0_median_lt_0_05_bps_pass` in `results/simulator_verification.json`) and every main model's `beats_copy_on_obs` |
| G1 | *pass* if `gate_g1_matched_separation_pass` is true; else *not testable* if every pair's `matched_within_5pct` is false; else *fail* |
| G2 | from `spearman_obs_vs_<hard>_std_mse.rho` and the thresholds above. `gate_g2_obs_rank_decoupled` is true only for *decoupled*; false means *inconclusive* or *refuted* |

Read this way, evm-intervene's primary arm is G1 *not testable* (no pair passes the screen)
and G2 *decoupled* (ρ = 0.39). Its crash-exposed sensitivity arm is G1 *pass* and G2
*decoupled*.

### Rules

- A key's value may be `null` where it does not apply (evm: `obs_val_std_mse` of
  `single_tick_xyk` and `amm_only_exact`, which have no observational prediction).
- A test may add keys. It may not drop or redefine a key listed here. Extra analyses (rollouts,
  scaling, planners) go in their own files, not in this one.
- **Optional** keys that a test may write, with these meanings:
  - `format_version`: version of this document the file follows;
  - `strata`: `{"hard": [...], "easy": [...], "hard_block": "..."}`, the stratum names;
  - `units`: `{"physical_unit": "..."}`;
  - `resampled`, on each gap: the units the interval resamples, e.g. `["state", "seed"]`;
  - `gates.gate_g0_valid`, `gates.gate_g1_reading` (`"pass"`, `"fail"`, `"not_testable"`),
    `gates.gate_g2_reading` (`"decoupled"`, `"inconclusive"`, `"refuted"`).
- A model that is a control, not a main model, goes under `controls`, with the same layout as
  `models`, so that G0's "every main model" and G2's runs are unambiguous.

## Per repo

**evm-intervene.** Conforms as it is. Suggestions, all optional and up to its maintainer:

- write the optional keys above, so the file states its own strata, units and gate readings;
- drop `passes_5pct_match_with_conservation`, which each pair's `matched_within_5pct` covers.

Its other keys not shown above are evm-specific extras: `counts.n_oot_obs`, the `oot_obs_*`
keys, `*_price_rmse_bps` beside L_obs, and the per-seed `*_cliff_price_rmse_bps`.

**physical-intervene.** Written to this format from the start, with its own names (table
above) and all optional keys. It also adds `matched_contact_gap_states_only` (the interval
resampling states only, as evm-intervene's), `decoupled_by_interval` in G2, `width` in `seed_checkpoints`, `controls` for its two control
models, and `logged_action` per model (the logged action scored on the test states, an
observational reference that is not an intervention).

## Open questions

1. **Seed resampling in G1.** physical-intervene resamples seeds as well as states, so that a
   seed-to-seed difference cannot pass as an architecture difference. evm-intervene resamples
   states only. Should evm-intervene also resample seeds, so that both G1s mean the same thing?
   Until then, compare evm-intervene's `matched_cliff_gap` with physical-intervene's
   `matched_contact_gap_states_only`.
2. **Reference model in G1.** evm-intervene tests each model against one anchor
   (`conservation_wm`). physical-intervene has no natural anchor and tests all pairs, taking the
   lower-E_int member as the reference. Should the format allow both, as now, or fix one rule?
3. **Matching target.** evm-intervene matches to the anchor's 5-seed mean best L_obs.
   physical-intervene plans V\* = 1.1 × the largest per-architecture median best L_obs. Is the
   target rule per test, as this draft says, or shared?
4. **evm-intervene's design text.** Its `docs/DESIGN.md` says G2 also uses intermediate
   matched-loss checkpoints, but the code uses the 20 best checkpoints. It also does not say
   whether G2 reads the point estimate or the interval; the code uses the point estimate, and
   it does not record a G1 *not testable* outcome. This draft follows the code. Would
   evm-intervene's maintainer want its DESIGN.md updated to match?
5. **Where the format lives.** Here in physical-intervene for now, while it is a draft. Once
   agreed, should it move to forward (where #5 started) or to a separate small repo that both
   depend on?
