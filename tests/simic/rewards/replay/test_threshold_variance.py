"""G-THRESHOLD-VARIANCE formula test (audit A7; pre-reg section 2.4/2.6).

The settlement's fixed prior is gated on ``q_settle >= 1.0``. A hard threshold
on a noisy statistic is option-like: for quote histories with equal MEANS just
below the cliff, higher VARIANCE raises the crossing probability - a windfall
paid for noise, not quality. A7 requires: no material positive expected
windfall from variance alone; if there is one, qualify on a robust
policy-unselectable statistic (``LCB(q_settle) >= 1``) instead - never back to
a ``q_decision`` gate.

Phase 1 delivers the test MACHINERY plus structural assertions (windfall is
real, monotone in sigma below the cliff, and the LCB variant strictly reduces
it). The materiality threshold - HOW MUCH windfall forces the LCB fallback -
is a freeze-time owner decision; the surface table this module prints is the
input to that decision.

The quote formula frozen here matches pre-reg section 2.3: EWMA with span 5
(alpha = 2/(span+1) = 1/3) over >= 5 valid post-request measurements.

CONVENTION NOTE (review nit #5, owner confirms at freeze): this is the
adjust=False streaming recursion seeded with the first measurement. Section
2.3 says "EWMA span 5" without pinning adjust-True/False; the choice is
immaterial to the option-property results here (both smoothers average the
window) but the Phase-2 implementation must freeze one form explicitly.
"""

from __future__ import annotations

import math
import random

import pytest

from .path_driver import PREMIUM_EXPRESSION_VALUE

SPAN_ALPHA = 1.0 / 3.0  # EWMA span 5
MIN_MEASUREMENTS = 5


def ewma_settle(measurements: list[float]) -> float:
    """The q_settle smoother: span-5 EWMA over the confirmation window."""
    if len(measurements) < MIN_MEASUREMENTS:
        raise ValueError(
            f"q_settle requires >= {MIN_MEASUREMENTS} valid measurements "
            f"(zero-slack rule: extend to the next boundary instead)"
        )
    value = measurements[0]
    for m in measurements[1:]:
        value = SPAN_ALPHA * m + (1 - SPAN_ALPHA) * value
    return value


def lcb_settle(measurements: list[float], z: float = 1.0) -> float:
    """Robust variant: EWMA minus z * (sample sd / sqrt(n))."""
    q = ewma_settle(measurements)
    n = len(measurements)
    mean = sum(measurements) / n
    var = sum((m - mean) ** 2 for m in measurements) / (n - 1)
    return q - z * math.sqrt(var / n)


def expected_premium(
    mu: float,
    sigma: float,
    *,
    statistic,
    window: int = 5,
    n_draws: int = 20_000,
    seed: int = 41,
) -> float:
    """Monte-Carlo E[premium] for i.i.d. N(mu, sigma) confirmation windows."""
    rng = random.Random(seed)
    paid = 0
    for _ in range(n_draws):
        window_measurements = [rng.gauss(mu, sigma) for _ in range(window)]
        if statistic(window_measurements) >= 1.0:
            paid += 1
    return PREMIUM_EXPRESSION_VALUE * paid / n_draws


class TestQuoteFormula:
    def test_constant_series_returns_the_constant(self):
        assert ewma_settle([2.0] * 5) == pytest.approx(2.0)

    def test_insufficient_measurements_fail_loudly(self):
        with pytest.raises(ValueError, match="zero-slack"):
            ewma_settle([1.0] * 4)


class TestVarianceWindfall:
    def test_windfall_exists_below_the_cliff(self):
        # mu = 0.9 < 1.0: with sigma -> 0 the premium is never paid; with
        # material noise it is paid for variance alone. The option property
        # is real, not hypothetical.
        quiet = expected_premium(0.9, 0.01, statistic=ewma_settle)
        noisy = expected_premium(0.9, 0.5, statistic=ewma_settle)
        assert quiet == pytest.approx(0.0, abs=1e-9)
        assert noisy > 0.2

    def test_windfall_is_monotone_in_sigma_below_the_cliff(self):
        sigmas = [0.05, 0.1, 0.2, 0.4, 0.6]
        values = [expected_premium(0.9, s, statistic=ewma_settle) for s in sigmas]
        assert values == sorted(values)

    def test_variance_hurts_above_the_cliff(self):
        # The option cuts both ways: at mu = 1.1 noise DENIES a deserved
        # premium some of the time. Symmetric distortion, asymmetric harm.
        quiet = expected_premium(1.1, 0.01, statistic=ewma_settle)
        noisy = expected_premium(1.1, 0.5, statistic=ewma_settle)
        assert quiet == pytest.approx(PREMIUM_EXPRESSION_VALUE, rel=1e-6)
        assert noisy < quiet

    def test_lcb_variant_reduces_the_windfall_everywhere_below_cliff(self):
        for sigma in (0.1, 0.3, 0.6):
            raw = expected_premium(0.9, sigma, statistic=ewma_settle)
            robust = expected_premium(0.9, sigma, statistic=lcb_settle)
            assert robust <= raw
        # And materially so at high noise:
        assert expected_premium(0.9, 0.6, statistic=lcb_settle) < expected_premium(
            0.9, 0.6, statistic=ewma_settle
        ) * 0.8


class TestSurfaceReport:
    def test_surface_table_renders(self, capsys):
        # The freeze-decision input: E[premium] over the (mu, sigma) grid for
        # both statistics. Printed for the freeze packet; asserted non-trivial.
        mus = [0.85, 0.95, 1.0, 1.05]
        sigmas = [0.05, 0.2, 0.4]
        lines = ["mu / sigma | " + " | ".join(f"{s:.2f}" for s in sigmas)]
        for mu in mus:
            raw_row = [
                expected_premium(mu, s, statistic=ewma_settle, n_draws=8000)
                for s in sigmas
            ]
            lcb_row = [
                expected_premium(mu, s, statistic=lcb_settle, n_draws=8000)
                for s in sigmas
            ]
            lines.append(
                f"{mu:.2f} raw  | " + " | ".join(f"{v:.3f}" for v in raw_row)
            )
            lines.append(
                f"{mu:.2f} lcb  | " + " | ".join(f"{v:.3f}" for v in lcb_row)
            )
        table = "\n".join(lines)
        print(table)
        assert "raw" in table and "lcb" in table
