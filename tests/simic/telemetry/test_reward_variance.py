"""Phase 0 (Stage-0): batch per-term reward variance-share decomposition.

``compute_variance_shares`` consumes a batch of finalized SHAPED/ESCROW steps and
returns, per additend, ``Cov(R_i, R)/Var(R)`` (population convention, matching the
existing ``cov_rcf_return_share`` in ``ppo_agent``) plus the coefficient of variation,
the top-line ``share_attribution`` (the ``bounded_attribution``/``R_cf`` share GATE 1
reads), and the reconciliation diagnostic ``residual_share`` (must be ~0).

GATE-0 reconciliation (reviewer-corrected): the real test is ``residual_share ≈ 0``,
NOT the tautological "shares sum to 1" (which holds by covariance linearity for any
classification once ``residual`` closes the sum).
"""

from __future__ import annotations

import pytest

from esper.leyline.telemetry_contracts import RewardComponentsTelemetry
from esper.simic.telemetry.reward_variance import compute_variance_shares


def test_shares_recover_known_decomposition() -> None:
    """R = bounded_attribution + (constant pbrs_bonus). The varying attribution term
    carries ALL the return variance (share ~1); the constant term and residual carry
    none (share ~0); shares sum to 1."""
    steps = [
        (3.0, RewardComponentsTelemetry(bounded_attribution=1.0, pbrs_bonus=2.0)),
        (5.0, RewardComponentsTelemetry(bounded_attribution=3.0, pbrs_bonus=2.0)),
        (7.0, RewardComponentsTelemetry(bounded_attribution=5.0, pbrs_bonus=2.0)),
    ]
    report = compute_variance_shares(steps)

    assert report["share_attribution"] == pytest.approx(1.0, abs=1e-9)
    assert report["shares"]["bounded_attribution"] == pytest.approx(1.0, abs=1e-9)
    assert report["shares"]["pbrs_bonus"] == pytest.approx(0.0, abs=1e-9)
    assert report["residual_share"] == pytest.approx(0.0, abs=1e-9)
    assert report["shares_sum"] == pytest.approx(1.0, abs=1e-9)


def test_residual_share_flags_a_misclassified_term() -> None:
    """If a live term is NOT captured by a named additend it leaks into residual; the
    residual variance-share becomes non-negligible — the real misclassification signal
    (NOT shares≠1, which stays 1 by construction)."""
    # reward_raw carries a varying hidden term (e.g. pending_auto_prune_penalty) that no
    # component field tracks: R = bounded_attribution + hidden, hidden varies.
    steps = [
        (1.0 + 0.0, RewardComponentsTelemetry(bounded_attribution=1.0)),
        (3.0 + 2.0, RewardComponentsTelemetry(bounded_attribution=3.0)),
        (5.0 + 4.0, RewardComponentsTelemetry(bounded_attribution=5.0)),
    ]
    report = compute_variance_shares(steps)
    # Shares STILL sum to 1 (tautology) ...
    assert report["shares_sum"] == pytest.approx(1.0, abs=1e-9)
    # ... but the hidden variance shows up in residual_share, the real diagnostic.
    assert report["residual_share"] > 0.1


def test_signed_rents_do_not_break_reconciliation() -> None:
    """occupancy_rent / fossilized_rent (positive magnitudes the reward subtracts) enter
    negated; reconciliation holds and residual is ~0."""
    steps = [
        (0.8, RewardComponentsTelemetry(bounded_attribution=1.0, occupancy_rent=0.2)),
        (2.7, RewardComponentsTelemetry(bounded_attribution=3.0, occupancy_rent=0.3)),
        (4.5, RewardComponentsTelemetry(bounded_attribution=5.0, occupancy_rent=0.5)),
    ]
    report = compute_variance_shares(steps)
    assert report["residual_share"] == pytest.approx(0.0, abs=1e-9)
    assert report["shares_sum"] == pytest.approx(1.0, abs=1e-9)


def test_coefficient_of_variation_reported() -> None:
    """Per-term CoV = std/(|mean|+eps) is reported for each additend."""
    steps = [
        (3.0, RewardComponentsTelemetry(bounded_attribution=1.0, pbrs_bonus=2.0)),
        (5.0, RewardComponentsTelemetry(bounded_attribution=3.0, pbrs_bonus=2.0)),
    ]
    report = compute_variance_shares(steps)
    assert "bounded_attribution" in report["cov"]
    # constant term has zero std -> CoV ~ 0.
    assert report["cov"]["pbrs_bonus"] == pytest.approx(0.0, abs=1e-9)
    assert report["cov"]["bounded_attribution"] > 0.0


def test_degenerate_zero_variance_batch_is_safe() -> None:
    """A constant-reward batch (Var(R)=0) must not divide-by-zero; shares default to 0."""
    steps = [
        (2.0, RewardComponentsTelemetry(bounded_attribution=2.0)),
        (2.0, RewardComponentsTelemetry(bounded_attribution=2.0)),
    ]
    report = compute_variance_shares(steps)
    assert report["share_attribution"] == pytest.approx(0.0)
    assert report["residual_share"] == pytest.approx(0.0)
