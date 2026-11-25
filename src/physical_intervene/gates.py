"""Matching, intervals, p-values, the gates and their readings (DESIGN.md, Matching observational loss,
Controls, Gates, Secondary analyses).

The inputs are per-checkpoint losses only, so this module works the same for stand-ins and, after
freeze 2, for trained models. A *family* is one architecture (or control) at one width: its seeds,
L_obs at every checkpoint, and, for each block of interventions, the per-state sum of per-transition
std-MSE at every checkpoint.

G1's gap (and that of the control pairs, G0.3, best checkpoints and gap along training) is Welch's
t-test on the per-seed E_int (DESIGN.md, G1 steps 3 and 4): the test states are held fixed.

The states-only interval (evm-intervene's) is a vectorised bootstrap: a replicate is a vector of
state counts W (multinomial, shared by both members of a pair). With S the per-seed, per-state loss
sums and n the per-state transition counts, E* = mean over seeds of (S W) / (n . W). `boot_e` can
also resample seeds (seed counts C per family), which an earlier draft's G1 did; kept for the
Phase 0 explorations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from scipy import stats

from physical_intervene import constants as c
from physical_intervene.metrics import excluded, matched_checkpoint, passes_match_screen

READINGS = ("pass", "inconclusive_not_specific", "fail", "inconclusive_not_sensitive", "not_testable")
TWIN_STATUS = ("clean", "separated", "not_run")
CURVE_READINGS = ("overlap", "persistent", "closing", "mixed")


@dataclass
class Family:
    name: str
    seeds: list[int]
    updates: list[int]
    l_obs: np.ndarray  # (S, K)
    sums: dict[str, np.ndarray]  # block -> (S, K, n_states)
    counts: dict[str, np.ndarray]  # block -> (n_states,)
    width: str = "medium"
    meta: dict = field(default_factory=dict)

    @property
    def n_seeds(self) -> int:
        return len(self.seeds)

    def best_idx(self) -> np.ndarray:
        return np.argmin(self.l_obs, axis=1)

    def best_l(self) -> np.ndarray:
        return self.l_obs.min(axis=1)

    def run_e(self, block: str, idx: np.ndarray) -> np.ndarray:
        """Each seed's own E on a block at the chosen checkpoint (mean over the block's transitions)."""
        s = self.sums[block][np.arange(self.n_seeds), idx]
        return s.sum(-1) / self.counts[block].sum()


# Matching ----------------------------------------------------------------------------------------


def v_star_of(fams: list[Family], factor: float = c.MATCH_TARGET_FACTOR) -> tuple[float, str]:
    """V* = factor x the largest per-architecture median of best L_obs, and the family that sets it."""
    med = {f.name: float(np.median(f.best_l())) for f in fams}
    name = max(med, key=med.get)
    return factor * med[name], name


@dataclass
class Match:
    family: str
    target: float
    idx: np.ndarray  # (S,) matched checkpoint of every seed
    l: np.ndarray  # (S,) its L_obs
    excluded: np.ndarray  # (S,) bool
    min_seeds: int

    @property
    def keep(self) -> np.ndarray:
        return ~self.excluded

    @property
    def dropped(self) -> bool:
        return int(self.keep.sum()) < self.min_seeds

    @property
    def mean_l(self) -> float:
        return float(self.l[self.keep].mean()) if self.keep.any() else float("nan")

    def to_dict(self, fam: Family) -> dict:
        return {
            "target": self.target,
            "matched_updates": [int(fam.updates[i]) for i in self.idx],
            "matched_l_obs": self.l.tolist(),
            "excluded_seeds": [int(s) for s, e in zip(fam.seeds, self.excluded, strict=True) if e],
            "seeds_left": int(self.keep.sum()),
            "dropped": self.dropped,
            "mean_matched_l_obs": self.mean_l,
        }


def match(fam: Family, target: float, min_seeds: int = c.G1_MIN_SEEDS_PER_ARCHITECTURE) -> Match:
    idx = np.array([matched_checkpoint(list(row), target) for row in fam.l_obs])
    exc = np.array([excluded(list(row), target) for row in fam.l_obs])
    return Match(fam.name, target, idx, fam.l_obs[np.arange(fam.n_seeds), idx], exc, min_seeds)


def best_match(fam: Family, min_seeds: int = c.G1_MIN_SEEDS_PER_ARCHITECTURE) -> Match:
    """Best checkpoints, as a Match with no exclusions."""
    idx = fam.best_idx()
    return Match(fam.name, float("nan"), idx, fam.best_l(), np.zeros(fam.n_seeds, bool), min_seeds)


# Arms and the bootstrap --------------------------------------------------------------------------


@dataclass
class Arm:
    """One family at chosen checkpoints, restricted to its kept seeds, on one block."""

    name: str
    sums: np.ndarray  # (S_kept, n_states)
    n: np.ndarray  # (n_states,)

    @property
    def e(self) -> float:
        return float(self.sums.sum(1).mean() / self.n.sum())

    @property
    def per_seed(self) -> np.ndarray:
        """Each seed's E: its mean std-MSE over the block's transitions, pooled across states."""
        return self.sums.sum(1) / self.n.sum()


def arm(fam: Family, m: Match, block: str = "push") -> Arm:
    keep = np.flatnonzero(m.keep)
    return Arm(fam.name, fam.sums[block][keep, m.idx[keep]], fam.counts[block])


def _state_draws(n_states: int, reps: int, rng: np.random.Generator) -> np.ndarray:
    return rng.multinomial(n_states, np.full(n_states, 1.0 / n_states), size=reps).astype(float)


def boot_e(a: Arm, W: np.ndarray, states: np.ndarray, rng: np.random.Generator | None) -> np.ndarray:
    """E* for each replicate. W: (R, n_used) state counts over the used states; rng None keeps the seed
    means fixed (the states-only interval)."""
    S = a.sums[:, states]  # (seeds, n_used)
    M = W @ S.T  # (R, seeds)
    tot = W @ a.n[states]
    if rng is None:
        num = M.mean(1)
    else:
        k = S.shape[0]
        C = rng.multinomial(k, np.full(k, 1.0 / k), size=W.shape[0]).astype(float)
        num = (M * C).sum(1) / k
    return num / tot


def p_two_sided(reps: np.ndarray) -> float:
    """p = min(1, 2 x (min(#D* <= 0, #D* >= 0) + 1) / (R + 1)) (DESIGN.md, G1 step 4)."""
    r = len(reps)
    return float(min(1.0, 2.0 * (min(np.sum(reps <= 0), np.sum(reps >= 0)) + 1) / (r + 1)))


def gap_test(
    challenger: Arm,
    reference: Arm,
    rng: np.random.Generator,
    reps: int = c.BOOTSTRAP_REPLICATES,
    seeds: bool = True,
) -> dict:
    """Gap E(challenger) - E(reference), its 95% interval and p. States are resampled with one draw
    for both (paired); with seeds=True each family's seeds are resampled independently."""
    if not np.array_equal(challenger.n, reference.n):
        raise ValueError("the two arms are not on the same interventions")
    states = np.flatnonzero(reference.n > 0)  # a state with no transition in the block is dropped
    W = _state_draws(len(states), reps, rng)
    d = boot_e(challenger, W, states, rng if seeds else None) - boot_e(
        reference, W, states, rng if seeds else None
    )
    gap = challenger.e - reference.e
    return {
        "gap": gap,
        "ci_low": float(np.percentile(d, 2.5)),
        "ci_high": float(np.percentile(d, 97.5)),
        "rel_gap": gap / reference.e if reference.e > 0 else float("inf"),
        "p_two_sided": p_two_sided(d),
        "resampled": ["state", "seed"] if seeds else ["state"],
    }


def welch_test(challenger: Arm, reference: Arm) -> dict:
    """G1's gap: Welch's t-test on the two arms' per-seed E, with its 95% interval
    gap +/- t(0.975, df) x sqrt(s_a^2 / n_a + s_b^2 / n_b), df by Welch-Satterthwaite."""
    if not np.array_equal(challenger.n, reference.n):
        raise ValueError("the two arms are not on the same interventions")
    a, b = challenger.per_seed, reference.per_seed
    va, vb = np.var(a, ddof=1) / len(a), np.var(b, ddof=1) / len(b)
    gap = challenger.e - reference.e
    se = float(np.sqrt(va + vb))
    if se == 0.0:  # no spread at all (made-up inputs only)
        df, half, p = float("inf"), 0.0, (1.0 if gap == 0 else 0.0)
    else:
        df = float((va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1)))
        half = float(stats.t.ppf(0.975, df) * se)
        p = float(2 * stats.t.sf(abs(gap) / se, df))
    return {
        "gap": gap,
        "ci_low": gap - half,
        "ci_high": gap + half,
        "rel_gap": gap / reference.e if reference.e > 0 else float("inf"),
        "p_two_sided": min(1.0, p),
        "df": df,
        "method": c.G1_TEST,
    }


def excludes_zero(g: dict) -> bool:
    return bool(g["ci_low"] > 0 or g["ci_high"] < 0)


def meets(g: dict, p_key: str = "p_two_sided") -> bool:
    """The three G1 thresholds: p < 0.05, interval excluding zero, relative gap >= 20%."""
    return bool(g[p_key] < c.G1_ALPHA and excludes_zero(g) and g["rel_gap"] >= c.G1_MIN_REL_GAP)


def holm(p: list[float | None]) -> list[float]:
    """Holm-adjusted p-values. A pair that could not be tested (an architecture dropped) counts as
    p = 1, so the correction stays over every pair the design names."""
    q = np.array([1.0 if x is None else float(x) for x in p])
    m = len(q)
    order = np.argsort(q, kind="stable")
    adj = np.maximum.accumulate((m - np.arange(m)) * q[order])
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    return out.tolist()


# G1 and the control pairs ------------------------------------------------------------------------


def compare(
    fa: Family,
    ma: Match,
    fb: Family,
    mb: Match,
    rng: np.random.Generator,
    reps: int = c.BOOTSTRAP_REPLICATES,
    block: str = "push",
    states_only: bool = True,
    orient: tuple[str, str] | None = None,
) -> dict:
    """One pair at chosen checkpoints: reference = lower E (unless `orient` fixes (challenger,
    reference)), match screen, G1's gap (Welch's t-test over the seeds) and the states-only
    bootstrap interval."""
    if ma.dropped or mb.dropped:
        return {
            "tested": False,
            "challenger": fa.name,
            "reference": fb.name,
            "not_tested_because": [m.family for m in (ma, mb) if m.dropped],
        }
    aa, ab = arm(fa, ma, block), arm(fb, mb, block)
    if orient is not None:
        ch, ref = (aa, ab) if orient[0] == fa.name else (ab, aa)
    else:
        ch, ref = (aa, ab) if aa.e > ab.e else (ab, aa)
    l_ch, l_ref = (ma.mean_l, mb.mean_l) if ch is aa else (mb.mean_l, ma.mean_l)
    out = {
        "tested": True,
        "challenger": ch.name,
        "reference": ref.name,
        "e_challenger": ch.e,
        "e_reference": ref.e,
        "matched_within_5pct": bool(passes_match_screen(l_ch, l_ref)),
        "obs_val_rel_diff": abs(l_ch - l_ref) / min(l_ch, l_ref),
        "gap": welch_test(ch, ref),
    }
    if states_only:
        out["gap_states_only"] = gap_test(ch, ref, rng, reps, seeds=False)
    return out


def control_separates(pair: dict, challenger: str | None = None) -> bool:
    """A control pair separates if it passes the match screen and meets the three thresholds with its
    unadjusted p (and, for the positive control, with the named challenger the higher)."""
    if not pair.get("tested"):
        return False
    ok = pair["matched_within_5pct"] and meets(pair["gap"])
    return bool(ok and (challenger is None or pair["challenger"] == challenger))


def g1_reading(pairs: list[dict], twin_status: dict[str, str], positive_pass: bool) -> tuple[str, list]:
    """DESIGN.md, G1 verdict. `pairs`: main pairs with `matched_within_5pct`, `meets` (Holm p, interval,
    relative gap) and the two architectures. Returns the reading and the passing pairs."""
    screened = [p for p in pairs if p.get("tested") and p["matched_within_5pct"]]
    if not screened:
        return "not_testable", []
    meeting = [p for p in screened if p["meets"]]
    passing = [
        p
        for p in meeting
        if twin_status.get(p["challenger"]) != "separated" and twin_status.get(p["reference"]) != "separated"
    ]
    if passing:
        return "pass", [[p["challenger"], p["reference"]] for p in passing]
    if meeting:
        return "inconclusive_not_specific", []
    return ("fail" if positive_pass else "inconclusive_not_sensitive"), []


def main_pairs(names: list[str]) -> list[tuple[str, str]]:
    return list(combinations(names, 2))


def run_g1(
    fams: dict[str, Family],
    names: list[str],
    target: float,
    rng: np.random.Generator,
    reps: int = c.BOOTSTRAP_REPLICATES,
    min_seeds: int = c.G1_MIN_SEEDS_PER_ARCHITECTURE,
    block: str = "push",
) -> dict:
    """Matching at `target` and every pair of `names`, Holm over all of them."""
    matches = {n: match(fams[n], target, min_seeds) for n in names}
    pairs = [
        compare(fams[a], matches[a], fams[b], matches[b], rng, reps, block) for a, b in main_pairs(names)
    ]
    adj = holm([p["gap"]["p_two_sided"] if p["tested"] else None for p in pairs])
    for p, h in zip(pairs, adj, strict=True):
        if p["tested"]:
            p["gap"]["holm_p"] = h
            p["meets"] = meets(p["gap"], "holm_p")
        else:
            p["meets"] = False
    return {"target": target, "matches": matches, "pairs": pairs, "holm_m": len(pairs)}


def twin_pair(
    fams: dict[str, Family],
    arch: str,
    twin: str,
    target: float,
    rng: np.random.Generator,
    reps: int = c.BOOTSTRAP_REPLICATES,
    min_seeds: int = c.G1_MIN_SEEDS_PER_ARCHITECTURE,
) -> tuple[dict, str]:
    ma, mt = match(fams[arch], target, min_seeds), match(fams[twin], target, min_seeds)
    p = compare(fams[arch], ma, fams[twin], mt, rng, reps)
    p["role"] = "negative"
    if not p["tested"]:
        return p, "not_run"
    p["separates"] = control_separates(p)
    return p, ("separated" if p["separates"] else "clean")


def positive_pair(
    fams: dict[str, Family],
    arch: str,
    flawed: str,
    target: float,
    rng: np.random.Generator,
    reps: int = c.BOOTSTRAP_REPLICATES,
    min_seeds: int = c.G1_MIN_SEEDS_PER_ARCHITECTURE,
) -> dict:
    ma, mf = match(fams[arch], target, min_seeds), match(fams[flawed], target, min_seeds)
    p = compare(fams[arch], ma, fams[flawed], mf, rng, reps, states_only=False)
    p["role"] = "positive"
    p["separates"] = control_separates(p, challenger=flawed)
    return p


# G0 ------------------------------------------------------------------------------------------------


def beats_copy(fam: Family, m: Match, copy_l_obs: float) -> dict:
    """G0.2 for one main model: seed-mean L_obs below copy's at matched and at best checkpoints."""
    matched = m.mean_l
    best = float(fam.best_l().mean())
    return {"matched": bool(matched < copy_l_obs), "best": bool(best < copy_l_obs), "copy": copy_l_obs}


def g0_blind(
    fams: dict[str, Family],
    blind: str,
    ref: str,
    rng: np.random.Generator,
    reps: int = c.BOOTSTRAP_REPLICATES,
) -> dict:
    """G0.3: at best checkpoints, on push, blind has the higher E_int with a 95% interval (Welch's
    t-test over the seeds, as in G1) excluding zero."""
    p = compare(
        fams[blind], best_match(fams[blind]), fams[ref], best_match(fams[ref]), rng, reps, orient=(blind, ref)
    )
    p["pass"] = bool(p["gap"]["gap"] > 0 and p["gap"]["ci_low"] > 0)
    return p


# G2 ------------------------------------------------------------------------------------------------


def spearman_boot(
    x: np.ndarray, y: np.ndarray, rng: np.random.Generator, reps: int = c.BOOTSTRAP_REPLICATES
) -> dict:
    """Spearman rho with a 95% interval from resampling runs (ties from repeated runs get average
    ranks)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    r = stats.spearmanr(x, y)
    idx = rng.integers(0, len(x), size=(reps, len(x)))
    rx, ry = stats.rankdata(x[idx], axis=1), stats.rankdata(y[idx], axis=1)
    rx -= rx.mean(1, keepdims=True)
    ry -= ry.mean(1, keepdims=True)
    den = np.sqrt((rx * rx).sum(1) * (ry * ry).sum(1))
    rho = np.where(den > 0, (rx * ry).sum(1) / np.where(den > 0, den, 1), np.nan)
    rho = rho[np.isfinite(rho)]
    return {
        "rho": float(r.statistic),
        "p_value": float(r.pvalue),
        "ci_low": float(np.percentile(rho, 2.5)),
        "ci_high": float(np.percentile(rho, 97.5)),
    }


def g2_reading(rho: float) -> str:
    if rho < c.G2_DECOUPLED_BELOW:
        return "decoupled"
    if rho >= c.G2_REFUTED_AT_OR_ABOVE:
        return "refuted"
    return "inconclusive"


# Gap along training --------------------------------------------------------------------------------


def curve_reading(levels: list[dict]) -> str:
    """DESIGN.md, Gap along training: overlap if the interval includes zero at every level; persistent
    if it excludes zero with the same sign at V* and at the pair's lowest level; closing if it
    excludes zero at V* but not at the lowest level; otherwise mixed. A level that could not be
    tested counts as including zero."""

    def sign(lv: dict) -> int:
        if not lv.get("tested", True) or lv.get("ci_low") is None:
            return 0
        return 1 if lv["ci_low"] > 0 else (-1 if lv["ci_high"] < 0 else 0)

    signs = [sign(lv) for lv in levels]
    at_v = next(sign(lv) for lv in levels if lv["level"] == "V*")
    lowest = signs[-1]
    if all(s == 0 for s in signs):
        return "overlap"
    if at_v != 0 and lowest == at_v:
        return "persistent"
    if at_v != 0 and lowest == 0:
        return "closing"
    return "mixed"


def gap_along_training(
    fa: Family,
    fb: Family,
    v_star: float,
    orient: tuple[str, str],
    rng: np.random.Generator,
    reps: int = c.BOOTSTRAP_REPLICATES,
    min_seeds: int = c.G1_MIN_SEEDS_PER_ARCHITECTURE,
) -> dict:
    """The pair at 2 x V*, 1.5 x V*, V* and, if lower, V*_pair (the matching rule on the pair's two
    medians only). Each level uses the matched checkpoints, exclusions and seed rule of *Matching*,
    and the gap keeps the orientation of the pair at V*."""
    v_pair, _ = v_star_of([fa, fb])
    lv = [(f"{k:.1f}xV*" if k != 1.0 else "V*", k * v_star) for k in c.CURVE_LEVEL_FACTORS]
    if v_pair < v_star:
        lv.append(("V*_pair", v_pair))
    out = []
    for name, level in lv:
        ma, mb = match(fa, level, min_seeds), match(fb, level, min_seeds)
        p = compare(fa, ma, fb, mb, rng, reps, states_only=False, orient=orient)
        row = {"level": name, "obs_val_loss": level, "tested": p["tested"]}
        if p["tested"]:
            row.update({k: p["gap"][k] for k in ("gap", "ci_low", "ci_high", "rel_gap")})
            row["matched_within_5pct"] = p["matched_within_5pct"]
        else:
            row.update(
                {"gap": None, "ci_low": None, "ci_high": None, "not_tested_because": p["not_tested_because"]}
            )
        out.append(row)
    return {"levels": out, "v_star_pair": v_pair, "reading": curve_reading(out)}


def asymmetric_reading(pair: dict, structured: str = "contact_wm") -> dict | None:
    """DESIGN.md, The structured model: how a pair with contact_wm is read. Lower E_int at V* and the
    pair meets G1's conditions: conservative evidence for its structure. Higher: not read as
    evidence against it (read through gap_along_training and matched_only)."""
    if not pair.get("tested") or structured not in (pair["challenger"], pair["reference"]):
        return None
    lower = "structured" if pair["reference"] == structured else "other"
    if not pair["meets"] or not pair["matched_within_5pct"]:
        reading = "no_separation"
    elif lower == "structured":
        reading = "evidence_for_structure"
    else:
        reading = "not_evidence_against"
    return {"structured_model": structured, "lower_at_v_star": lower, "reading": reading}


def mann_whitney(fa: Family, ma: Match, fb: Family, mb: Match, block: str = "push") -> dict:
    """Per-seed view: each seed's E_int, and an exact two-sided Mann-Whitney test between the two."""
    ea = fa.run_e(block, ma.idx)[ma.keep]
    eb = fb.run_e(block, mb.idx)[mb.keep]
    r = stats.mannwhitneyu(ea, eb, alternative="two-sided", method="exact")
    return {fa.name: ea.tolist(), fb.name: eb.tolist(), "p_two_sided": float(r.pvalue)}
