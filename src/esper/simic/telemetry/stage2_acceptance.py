"""Stage-2 HRA MAJOR-1 acceptance scorer (pure predicates).

Implements the frozen gate in
docs/analysis/2026-07-06-stage2-hra-major1-acceptance-gate.md. Pure functions over
per-seed paired metric series; no I/O, no torch, no scipy. The packet CLI (reads Karn
``ppo_updates`` telemetry and emits the verdict markdown) is a thin wrapper over these.

Statistics are dependency-free: the one-sided Wilcoxon signed-rank p-value is computed
EXACTLY by enumerating sign-flips of the rank statistic (valid/cheap at the n<=~20 this
gate uses). The n=5 tier is a sign-consistency screen, matching the repo's PDR-0019/0026
sign-test lineage; n=10 is the powered Wilcoxon claim.
"""

from __future__ import annotations

import enum
import itertools
import math

import numpy as np

# Valid seed-count tiers (§10): n=5 direction screen, n=10 powered claim.
_VALID_TIERS = (5, 10)
# Exact Wilcoxon enumerates 2**m sign patterns; cap m so a mis-sized input cannot
# silently hang the process (measured: m=24 ~36s, m=30 ~40min). n<=10 keeps m<=10.
_MAX_WILCOXON_M = 20


def _require_finite(name: str, values: list[float]) -> np.ndarray:
    """Fail loud on empty or non-finite input (spec §1; repo forbids silent masking)."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values (NaN/Inf)")
    return arr


def _validate_tier(deltas: list[float], n: int) -> None:
    """A seed count that isn't a pre-registered tier, or that disagrees with the data
    length, changes which test runs — refuse it rather than score a silent wrong verdict."""
    if n not in _VALID_TIERS:
        raise ValueError(f"n must be one of {_VALID_TIERS} (pre-registered tiers), got {n}")
    if len(deltas) != n:
        raise ValueError(f"len(deltas)={len(deltas)} does not match tier n={n}")


class GateResult(enum.Enum):
    """Per-leg outcome."""

    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"


class Verdict(enum.Enum):
    """Composite Stage-2 acceptance outcome (§7)."""

    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    INCONCLUSIVE = "INCONCLUSIVE"
    SCREEN_PASS = "SCREEN_PASS"  # n=5 screen cleared every gate → run n=10; NEVER banked as ACCEPT
    INVALID = "INVALID"


def level(series: list[float]) -> float:
    """Per-run level = median over the (post-burn-in) update series (§2)."""
    return float(np.median(_require_finite("series", series)))


def vol(series: list[float]) -> float:
    """Per-run volatility = IQR (P75 - P25) over the update series (§2)."""
    arr = _require_finite("series", series)
    return float(np.percentile(arr, 75) - np.percentile(arr, 25))


def _paired_bootstrap_se(
    values: list[float], n_boot: int = 2000, rng: np.random.Generator | None = None
) -> float:
    """Bootstrap SE of the per-seed median (the noise of a paired level comparison).

    The default fixed seed is deliberate: δ is a frozen, pre-registered threshold, so its
    calibration must be reproducible. Do not remove the seed without a new PDR.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    vals = np.asarray(values, dtype=float)
    resamples = rng.choice(vals, size=(n_boot, len(vals)), replace=True)
    medians = np.median(resamples, axis=1)
    return float(np.std(medians, ddof=1))


def calibrate_delta(
    off_ev_levels: list[float],
    off_ev_iqrs: list[float],
    floor: float = 0.05,
    rng: np.random.Generator | None = None,
) -> float:
    """Freeze the ev_sum non-inferiority margin δ from the OFF arms only (§3, §11).

    Primary: max(floor, bootstrap SE of the OFF per-seed EV-level median).
    Fallback (<5 OFF runs): max(floor, 0.5 * median of the OFF within-run IQRs).
    An empty OFF corpus is a data failure, not a "quiet corpus" — fail loud.
    """
    levels = _require_finite("off_ev_levels", off_ev_levels)
    if len(levels) >= 5:
        return max(floor, _paired_bootstrap_se(list(levels), rng=rng))
    iqrs = _require_finite("off_ev_iqrs", off_ev_iqrs)
    return max(floor, 0.5 * float(np.median(iqrs)))


def _wilcoxon_signed_rank_p_greater(deltas: list[float]) -> float:
    """Exact one-sided p-value for H1: median(deltas) > 0.

    Standard Wilcoxon signed-rank: drop zeros, rank |delta| with midranks for ties,
    W+ = sum of ranks of the positive differences. The exact null enumerates all 2^m
    equally-likely sign assignments; p = P(W+ >= observed).
    """
    vals = [float(d) for d in deltas if d != 0.0 and math.isfinite(d)]
    if any(not math.isfinite(d) for d in deltas):
        raise ValueError("wilcoxon input contains non-finite values (NaN/Inf)")
    m = len(vals)
    if m == 0:
        return 1.0
    if m > _MAX_WILCOXON_M:
        raise ValueError(
            f"exact sign-flip enumeration infeasible for m={m} nonzero deltas "
            f"(2**{m} patterns); this gate is pre-registered for n<=10"
        )
    absvals = np.abs(vals)
    order = np.argsort(absvals, kind="mergesort")
    sorted_abs = absvals[order]
    ranks = np.empty(m)
    i = 0
    while i < m:  # average (mid) ranks for ties in |delta|
        j = i
        while j + 1 < m and sorted_abs[j + 1] == sorted_abs[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # ranks are 1-based
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    positive = np.array([1.0 if v > 0 else 0.0 for v in vals])
    w_obs = float(np.sum(ranks * positive))
    count_ge = 0
    total = 0
    for bits in itertools.product((0.0, 1.0), repeat=m):
        if float(np.sum(ranks * np.asarray(bits))) >= w_obs:
            count_ge += 1
        total += 1
    return count_ge / total


def leg_a(delta_a: list[float], delta: float, n: int) -> GateResult:
    """LEG A — ev_sum non-inferiority floor (§3). Hard MAJOR-1 gate.

    Δ_A(s) = level(ev_sum_ON) - level(explained_variance_OFF), per seed.
    n=5: sign-consistency screen, >=4/5 seeds non-inferior (Δ_A >= -δ).
    n=10: one-sided Wilcoxon on {Δ_A + δ} rejects "ON worse than OFF-δ" AND median >= -δ.
    """
    _validate_tier(delta_a, n)
    _require_finite("delta_a", delta_a)
    non_inferior = [d >= -delta for d in delta_a]
    if n < 10:
        return GateResult.PASS if sum(non_inferior) >= 4 else GateResult.FAIL
    p = _wilcoxon_signed_rank_p_greater([d + delta for d in delta_a])
    if p < 0.05 and float(np.median(delta_a)) >= -delta:
        return GateResult.PASS
    return GateResult.FAIL


def leg_b(delta_b: list[float], eps: float, n: int) -> GateResult:
    """LEG B — advantage-path VOLATILITY reduction (§4). Gated downstream signal.

    Δ_B(s) = per-seed relative change (ON vs OFF) in IQR-over-updates of the
    return-scale-invariant residual sqrt(1 − ev), ev = ev_sum (ON) /
    explained_variance (OFF) over non-floored updates (§8B). The raw
    IQR(pre_norm_advantage_std) is a descriptive covariate ONLY, not the gated
    statistic — confounded by Var(returns_total) via pre_norm_advantage_std² =
    (1 − ev_sum)·Var(returns_total) (§4; level variant rejected §12.1).
    n=5: >=4/5 seeds reduce (Δ_B<0) → PASS; >=3/5 → INCONCLUSIVE; else FAIL.
    n=10: median >= 0 → FAIL; else one-sided Wilcoxon (ON<OFF, p<0.05) AND
          median <= -ε → PASS; otherwise INCONCLUSIVE (null band → owner adjudication).
    """
    _validate_tier(delta_b, n)
    _require_finite("delta_b", delta_b)
    if n < 10:
        reduced = sum(d < 0 for d in delta_b)
        if reduced >= 4:
            return GateResult.PASS
        if reduced >= 3:
            return GateResult.INCONCLUSIVE
        return GateResult.FAIL
    med = float(np.median(delta_b))
    if med >= 0:
        return GateResult.FAIL
    p_less = _wilcoxon_signed_rank_p_greater([-d for d in delta_b])
    if p_less < 0.05 and med <= -eps:
        return GateResult.PASS
    return GateResult.INCONCLUSIVE


def mech_guard(ev_main_vol_on: list[float], expl_vol_off: list[float]) -> bool:
    """MECH — EV_main mechanism guard (§5). Necessary, not a discriminator.

    ON per-seed ev_main volatility must fall below the OFF single-head explained_variance
    volatility (the only honest baseline) on a >=80% majority of seeds (>=4/5 at the screen
    tier; >=⌈0.8n⌉ generalises it to n=10 — pending drl-expert confirmation). A failure
    signals the decomposition did not do its own bookkeeping (structural bug) → REJECT.
    """
    _require_finite("ev_main_vol_on", ev_main_vol_on)
    _require_finite("expl_vol_off", expl_vol_off)
    lower = sum(on < off for on, off in zip(ev_main_vol_on, expl_vol_off, strict=True))
    return lower >= math.ceil(0.8 * len(ev_main_vol_on))


def safety_g1(d_val_acc: list[float], tau_acc: float) -> GateResult:
    """G1 — val-accuracy non-regression (§6). Paired per-seed Δ (percentage points)."""
    arr = _require_finite("d_val_acc", d_val_acc)
    if float(np.median(arr)) >= -tau_acc:
        return GateResult.PASS
    return GateResult.FAIL


def safety_g2(d_added_params: list[float], delta_param_max: float) -> bool:
    """G2 — added-params non-inflation (§6). Split from G1 so an accuracy regression
    cannot hide behind a param reduction."""
    arr = _require_finite("d_added_params", d_added_params)
    return float(np.median(arr)) <= delta_param_max


def composite_verdict(
    *,
    valid: bool,
    n: int,
    leg_a_result: GateResult,
    leg_b_result: GateResult,
    mech_hold: bool,
    g1_result: GateResult,
    g2_hold: bool,
    g3_hold: bool,
    g4_hold: bool,
) -> Verdict:
    """Combine the legs and guards into the Stage-2 verdict (§7).

    INVALID short-circuits (§1: no interpretation on a broken run). Hard REJECT on the
    MAJOR-1 floor (LEG A), the MECH structural guard, an informative G1 regression, or a
    materially-worsened downstream (LEG B FAIL). A clean run banks ACCEPT ONLY at the n=10
    claim tier — a clean n=5 screen is SCREEN_PASS (proceed to n=10), never ACCEPT.
    Everything else — a downstream null band, or a safety soft-fail despite a LEG-B pass —
    is INCONCLUSIVE and routes to owner adjudication. G3/G4 are required, not defaulted, so
    a run cannot be scored without explicitly evaluating churn/guard-channel safety.
    """
    if n not in _VALID_TIERS:
        raise ValueError(f"n must be one of {_VALID_TIERS} (pre-registered tiers), got {n}")
    if not valid:
        return Verdict.INVALID
    if leg_a_result is GateResult.FAIL or not mech_hold or g1_result is GateResult.FAIL:
        return Verdict.REJECT
    if leg_b_result is GateResult.FAIL:
        return Verdict.REJECT
    # Clean run requires the spec's literal conjunction LEG_A==PASS ∧ G1==PASS ∧ LEG_B==PASS
    # ∧ safety — "not FAIL" is not "PASS" for a 3-valued enum, so an INCONCLUSIVE leg_a/g1
    # must not clear it.
    clean = (
        leg_a_result is GateResult.PASS
        and g1_result is GateResult.PASS
        and leg_b_result is GateResult.PASS
        and g2_hold
        and g3_hold
        and g4_hold
    )
    if not clean:
        return Verdict.INCONCLUSIVE
    return Verdict.ACCEPT if n >= 10 else Verdict.SCREEN_PASS
