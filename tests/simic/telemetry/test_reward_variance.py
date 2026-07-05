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
from esper.simic.telemetry.reward_variance import compute_return_variance_shares


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


def test_return_variance_shares_match_hand_computed_covariance() -> None:
    # Single 3-step segment, gamma=0.5, two components summing to reward=[1,0,1].
    # Value-free discounted return-to-go:
    #   R_total = [1.25, 0.5, 1.0], R_a = [1.0, 0.0, 0.0], R_b = [0.25, 0.5, 1.0]
    #   Var(R_total) = 7/24 ; Cov(R_a, R_total) = 8/24 -> share_a = 8/7
    #   share_b = 1 - 8/7 = -1/7  (correlated components -> shares may leave [0, 1])
    report = compute_return_variance_shares(
        component_rewards={"a": [1.0, 0.0, 0.0], "b": [0.0, 0.0, 1.0]},
        dones=[False, False, True],
        gamma=0.5,
    )

    assert report["shares"]["a"] == pytest.approx(8.0 / 7.0, rel=1e-9)
    assert report["shares"]["b"] == pytest.approx(-1.0 / 7.0, rel=1e-9)
    assert report["shares_sum"] == pytest.approx(1.0, abs=1e-9)


def test_gate_outputs_attribution_and_main_variance_share() -> None:
    # Same arithmetic, relabelled so the cf stream is `bounded_attribution`:
    #   R_cf = [0.25, 0.5, 1.0], R_main = R_total - R_cf = [1.0, 0.0, 0.0]
    #   Var(R_total) = 7/72 ; Var(R_main) = 2/9 = 16/72
    #   share_attribution = Cov(R_cf, R)/Var(R) = -1/7  (the >0.40 gate read)
    #   residual_share    = Cov(R_residual, R)/Var(R) = 8/7  (the ~0 reconciliation term)
    #   r_main_var_share  = Var(R_main)/Var(R) = (16/72)/(7/72) = 16/7  (the smoothness leg)
    # r_main_var_share is a DISTINCT quantity from any covariance share (it is NOT bounded
    # by [0, 1] — here R_main is anti-correlated with R_cf so it carries MORE return
    # variance than the total): the gate's "is R_main the smoother stream" leg reads this
    # variance-share, not the fragile CoV.
    report = compute_return_variance_shares(
        component_rewards={"bounded_attribution": [0.0, 0.0, 1.0], "residual": [1.0, 0.0, 0.0]},
        dones=[False, False, True],
        gamma=0.5,
    )

    assert report["share_attribution"] == pytest.approx(-1.0 / 7.0, rel=1e-9)
    assert report["residual_share"] == pytest.approx(8.0 / 7.0, rel=1e-9)
    assert report["r_main_var_share"] == pytest.approx(16.0 / 7.0, rel=1e-9)


def test_degenerate_zero_variance_batch_returns_finite_zero_shares() -> None:
    # Two single-step episodes with identical returns -> Var(R_total) = 0. A real PPO
    # update can hit this; the decomposition must return finite 0.0 shares, never nan.
    import math

    report = compute_return_variance_shares(
        component_rewards={"bounded_attribution": [1.0, 1.0], "residual": [0.0, 0.0]},
        dones=[True, True],
        gamma=0.5,
    )

    assert report["shares"]["bounded_attribution"] == 0.0
    assert report["share_attribution"] == 0.0
    assert report["r_main_var_share"] == 0.0
    assert all(math.isfinite(v) for v in report["shares"].values())


def test_return_to_go_resets_at_episode_boundaries() -> None:
    # Two 2-step episodes (done at steps 1 and 3). If the reset were ignored (one
    # 4-step segment), the returns and therefore share_a would differ. With the reset:
    #   R_a = [1, 0, 1, 0], R_b = [0.5, 1, 0.5, 1], R_total = [1.5, 1, 1.5, 1]
    #   share_a = Cov(R_a, R)/Var(R) = 0.125 / 0.0625 = 2.0
    report = compute_return_variance_shares(
        component_rewards={"a": [1.0, 0.0, 1.0, 0.0], "b": [0.0, 1.0, 0.0, 1.0]},
        dones=[False, True, False, True],
        gamma=0.5,
    )

    assert report["shares"]["a"] == pytest.approx(2.0, rel=1e-9)
    assert report["shares_sum"] == pytest.approx(1.0, abs=1e-9)
