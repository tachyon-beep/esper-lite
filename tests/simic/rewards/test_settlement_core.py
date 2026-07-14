"""Settlement pure core (Phase 2, PDR-0090/0097): boundary math, quotes, gates, packages.

Spec: docs/plans/ready/2026-07-14-phase2-settlement-implementation.md §0/§3 +
pre-reg §2.2–§2.4. Everything here is the flag-off pure layer — no trainer
integration, no reward-path change beyond the behavior-preserving ratio_penalty
extraction (asserted identical below).
"""

from __future__ import annotations

import math

import pytest

from esper.leyline import DEFAULT_GAMMA
from esper.leyline.fossil_settlement import FossilSettlementConfig
from esper.simic.rewards import ContributionRewardConfig, RewardMode
from esper.simic.rewards.contribution import _compute_ratio_penalty
from esper.simic.rewards.settlement import (
    BoundaryPackage,
    annuity_per_epoch,
    boundary_with_extension,
    next_boundary,
    q_settle,
    settlement_gate,
    settlement_premium,
)

CFG = FossilSettlementConfig(abort_payment_mode="settle_partial")
RCFG = ContributionRewardConfig(reward_mode=RewardMode.SHAPED)


class TestBoundarySchedule:
    # §0: boundaries at multiples of w_settle; settle at the first boundary
    # >= R + min_window (the lock-burden overlay convention, cross-checked).
    @pytest.mark.parametrize("request_epoch,expected", [(3, 10), (5, 10), (7, 20), (10, 20), (94, 100), (96, 110)])
    def test_next_boundary(self, request_epoch, expected):
        assert next_boundary(request_epoch, CFG) == expected

    def test_full_cadence_needs_no_extension(self):
        # Measurements every epoch in [R+1, B]: window [4..10] has 7 >= 5.
        measured = frozenset(range(4, 11))
        assert boundary_with_extension(3, measured, CFG, max_epochs=150) == 10

    def test_sparse_measurements_extend_to_next_boundary(self):
        # R=3 -> B0=10, but only 3 valid measurements in [4,10] -> extend to 20.
        measured = frozenset({4, 6, 9, 12, 15, 18, 19})
        assert boundary_with_extension(3, measured, CFG, max_epochs=150) == 20

    def test_horizon_overflow_returns_none(self):
        # No qualifying boundary <= max_epochs: the request should have been
        # late-masked; the pure core reports None and the caller fails loud.
        assert boundary_with_extension(96, frozenset(), CFG, max_epochs=100) is None


class TestQuoteAndGate:
    def test_q_settle_constant_series(self):
        assert q_settle([2.0] * 5, CFG) == pytest.approx(2.0)

    def test_q_settle_zero_slack_fails_loud(self):
        with pytest.raises(ValueError, match="min_measurements"):
            q_settle([1.0] * 4, CFG)

    def test_raw_gate_pays_at_threshold(self):
        assert settlement_gate([1.0] * 5, CFG) is True
        assert settlement_gate([0.99] * 5, CFG) is False

    def test_lcb_gate_subtracts_dispersion(self):
        lcb_cfg = FossilSettlementConfig(gate_form="lcb", abort_payment_mode="settle_partial")
        # Constant series at threshold: sd=0 -> passes under lcb too.
        assert settlement_gate([1.0] * 5, lcb_cfg) is True
        # Noisy series with EWMA barely above 1.0: lcb denies what raw pays.
        # (ewma = 1.179; sd = 0.548 -> lcb = 1.179 - 0.548/sqrt(5) = 0.934)
        noisy = [1.5, 0.5, 1.5, 0.5, 1.5]
        assert settlement_gate(noisy, CFG) is True
        assert settlement_gate(noisy, lcb_cfg) is False


class TestPremiumAndPackage:
    def test_premium_is_the_frozen_expression_from_shipping_constants(self):
        # Matched-control consolidation (PDR-0083/0084): base + terminal_scale*tanh(1/ceiling),
        # evaluated in code from the reward config's actual values.
        expected = 0.5 + 3.0 * math.tanh(1.0 / 3.0)
        assert settlement_premium(RCFG) == pytest.approx(expected, abs=1e-15)

    def test_eligible_package_is_discount_neutral(self):
        # PV at request equals the shipping prior: prior_paid * gamma^d == premium * legitimacy.
        from esper.simic.rewards.settlement import boundary_package

        pkg = boundary_package(
            request_epoch=3,
            boundary_epoch=10,
            legitimacy_request=1.0,
            window_measurements=[3.0] * 7,
            window_cf_measurements=[3.0] * 7,
            reward_config=RCFG,
            settlement_config=CFG,
        )
        assert isinstance(pkg, BoundaryPackage)
        assert pkg.branch == "prior"
        d = 7
        assert pkg.d == d
        pv_at_request = pkg.prior_paid * DEFAULT_GAMMA**d
        assert pv_at_request == pytest.approx(settlement_premium(RCFG) * 1.0, abs=1e-12)

    def test_noncontributing_branch_also_discount_neutral(self):
        from esper.simic.rewards.settlement import boundary_package

        pkg = boundary_package(
            request_epoch=3,
            boundary_epoch=10,
            legitimacy_request=0.8,
            window_measurements=[0.5] * 7,  # q_settle < 1.0
            window_cf_measurements=[0.5] * 7,
            reward_config=RCFG,
            settlement_config=CFG,
        )
        assert pkg.branch == "noncontributing"
        assert pkg.prior_paid * DEFAULT_GAMMA**pkg.d == pytest.approx(
            RCFG.fossilize_noncontributing_penalty, abs=1e-12
        )


class TestAnnuityLaw:
    # T_annuity == T_provisional at frozen args, ALL branches (PDR-0096/0097).
    def test_clean_case_equals_weighted_quote(self):
        # progress > q, cf healthy, ratio benign, post-warmup: annuity == 1.0 * q.
        value = annuity_per_epoch(
            q_settle_value=3.0,
            cf_windowed=3.0,
            progress_at_boundary=20.0,
            germination_epoch=12,
            reward_config=RCFG,
        )
        assert value == pytest.approx(3.0, abs=1e-12)

    def test_negative_quote_pays_negative(self):
        # Signed continuity (§2.5.2): negative q_settle -> negative annuity.
        value = annuity_per_epoch(
            q_settle_value=-0.5,
            cf_windowed=2.0,
            progress_at_boundary=20.0,
            germination_epoch=12,
            reward_config=RCFG,
        )
        assert value == pytest.approx(-0.5, abs=1e-12)

    def test_ransomware_signature_is_penalized(self):
        # High LOO quote, tiny windowed cf: the restored cross-check fires
        # (the exploit PDR-0096 closed). Penalty relative to the clean case.
        gamed = annuity_per_epoch(
            q_settle_value=8.0,
            cf_windowed=0.5,  # ratio 16 >> hacking_ratio_threshold 5
            progress_at_boundary=20.0,
            germination_epoch=12,
            reward_config=RCFG,
        )
        clean = annuity_per_epoch(
            q_settle_value=8.0,
            cf_windowed=8.0,
            progress_at_boundary=20.0,
            germination_epoch=12,
            reward_config=RCFG,
        )
        assert gamed < clean

    def test_zero_total_ransomware_branch_fires(self):
        # cf <= safe_threshold with a high quote: the second branch (:508-509).
        value = annuity_per_epoch(
            q_settle_value=8.0,
            cf_windowed=0.0,
            progress_at_boundary=20.0,
            germination_epoch=12,
            reward_config=RCFG,
        )
        clean = annuity_per_epoch(
            q_settle_value=8.0,
            cf_windowed=8.0,
            progress_at_boundary=20.0,
            germination_epoch=12,
            reward_config=RCFG,
        )
        assert value < clean

    def test_legit_low_cf_seed_not_ransomware_penalized_below_quote_gate(self):
        # PDR-0096 reversal-trigger case (false-positive direction): quote at
        # or below the anti-gaming activation gate (q <= 1.0) never triggers
        # the ratio branch regardless of cf (production law: c > 1.0 required).
        value = annuity_per_epoch(
            q_settle_value=1.0,
            cf_windowed=0.05,
            progress_at_boundary=20.0,
            germination_epoch=12,
            reward_config=RCFG,
        )
        assert value == pytest.approx(1.0, abs=1e-12)


class TestRatioPenaltyExtraction:
    # The contribution.py inline block is now _compute_ratio_penalty; identical
    # behavior is guarded here directly (the call site is guarded by the full
    # rewards suite staying green).
    def test_benign_ratio_is_zero(self):
        assert _compute_ratio_penalty(3.0, 3.0, 1.0, RCFG) == 0.0

    def test_hostage_ratio_fires(self):
        assert _compute_ratio_penalty(8.0, 0.5, 1.0, RCFG) < 0.0

    def test_below_activation_gate_is_zero(self):
        assert _compute_ratio_penalty(0.9, 0.01, 1.0, RCFG) == 0.0

    def test_discounted_attribution_disables_check(self):
        # attribution_discount < 0.5 (cf badly negative) skips the ratio check
        # (the penalty branch handles that case instead) - production law.
        assert _compute_ratio_penalty(8.0, -2.0, 0.1, RCFG) == 0.0
