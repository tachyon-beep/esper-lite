"""Integration tests for D3 anti-timing-gaming reward fixes.

These tests verify the complete behavior of timing discount + harmonic
attribution working together to discourage early germination gaming.

The anti-gaming formula (harmonic/timing discount) applies when:
- seed_contribution >= progress (seed claims credit beyond host improvement)

When contribution < progress, attribution is simply capped at contribution.
"""

import math

import pytest

from esper.leyline import LifecycleOp, SeedStage
from esper.simic.rewards import (
    compute_contribution_reward,
    ContributionRewardConfig,
    SeedInfo,
)


class TestAntiTimingGamingIntegration:
    """End-to-end tests for D3 anti-timing-gaming fixes."""

    def test_early_germination_gaming_scenario(self) -> None:
        """Simulate the exact scenario that caused the anti-pattern.

        Gaming pattern: Seed germinated early, has high counterfactual contribution
        (perhaps through dependency creation) but host was improving anyway.

        The contribution exceeds progress, triggering the attribution formula.
        With early germination, timing discount also applies.
        """
        # Configuration with all D3 fixes enabled
        config_d3 = ContributionRewardConfig(
            contribution_weight=1.0,
            germination_warmup_epochs=10,
            germination_discount_floor=0.4,
            attribution_formula="harmonic",
            disable_timing_discount=False,
        )

        # Baseline configuration (pre-D3 behavior)
        config_baseline = ContributionRewardConfig(
            contribution_weight=1.0,
            attribution_formula="geometric",
            disable_timing_discount=True,
        )

        # Simulate seed germinated at epoch 2, now at epoch 100
        seed = SeedInfo(
            stage=SeedStage.TRAINING.value,
            total_improvement=5.0,  # Host improved 5% since germination
            improvement_since_stage_start=0.5,
            epochs_in_stage=1,
            seed_params=1000,
            previous_stage=SeedStage.GERMINATED.value,
            previous_epochs_in_stage=0,
            seed_age_epochs=98,  # epoch 100 - 98 = germinated at epoch 2
            interaction_sum=0.0,
            boost_received=0.0,
        )

        # Gaming scenario: seed claims 30% contribution but host only improved 5%
        # contribution (30) >= progress (5), so formula applies
        _, comp_d3 = compute_contribution_reward(
            action=LifecycleOp.WAIT,
            seed_contribution=30.0,  # Large claimed contribution
            val_acc=51.0,
            seed_info=seed,
            epoch=100,
            max_epochs=150,
            acc_at_germination=46.0,  # progress = 5.0
            config=config_d3,
            return_components=True,
        )

        _, comp_baseline = compute_contribution_reward(
            action=LifecycleOp.WAIT,
            seed_contribution=30.0,
            val_acc=51.0,
            seed_info=seed,
            epoch=100,
            max_epochs=150,
            acc_at_germination=46.0,
            config=config_baseline,
            return_components=True,
        )

        # Expected calculations:
        # Baseline (geometric): sqrt(5 * 30) = sqrt(150) ~ 12.25
        # D3 Harmonic: 2 * 5 * 30 / (5 + 30) = 300/35 ~ 8.57
        # D3 Timing: epoch 2, warmup 10, floor 0.4
        #   discount = 0.4 + (1.0 - 0.4) * (2/10) = 0.4 + 0.12 = 0.52
        # D3 Combined: 8.57 * 0.52 ~ 4.46

        expected_baseline = math.sqrt(5 * 30)  # ~12.25
        expected_harmonic = 2 * 5 * 30 / (5 + 30)  # ~8.57
        expected_discount = 0.4 + (1.0 - 0.4) * (2 / 10)  # 0.52
        expected_d3 = expected_harmonic * expected_discount  # ~4.46

        assert comp_baseline.bounded_attribution == pytest.approx(expected_baseline, rel=0.01)
        assert comp_d3.bounded_attribution == pytest.approx(expected_d3, rel=0.01)
        assert comp_d3.timing_discount == pytest.approx(expected_discount, rel=0.01)

        # D3 reduces attribution significantly for this gaming scenario
        reduction_factor = comp_baseline.bounded_attribution / comp_d3.bounded_attribution
        assert reduction_factor > 2.5, f"Expected >2.5x reduction, got {reduction_factor:.2f}x"

    def test_legitimate_late_germination_not_penalized(self) -> None:
        """Seeds germinated after warmup with contribution >= progress get full timing credit."""
        config_d3 = ContributionRewardConfig(
            contribution_weight=1.0,
            germination_warmup_epochs=10,
            germination_discount_floor=0.4,
            attribution_formula="harmonic",
            disable_timing_discount=False,
        )

        # Seed germinated at epoch 50, now at epoch 100
        seed = SeedInfo(
            stage=SeedStage.TRAINING.value,
            total_improvement=8.0,
            improvement_since_stage_start=0.5,
            epochs_in_stage=1,
            seed_params=1000,
            previous_stage=SeedStage.GERMINATED.value,
            previous_epochs_in_stage=0,
            seed_age_epochs=50,  # germinated at epoch 50 (after warmup)
            interaction_sum=0.0,
            boost_received=0.0,
        )

        # Legitimate scenario: contribution >= progress, both values close
        # contribution (10) >= progress (8), so harmonic formula applies
        _, components = compute_contribution_reward(
            action=LifecycleOp.WAIT,
            seed_contribution=10.0,  # Slightly more than progress
            val_acc=51.0,
            seed_info=seed,
            epoch=100,
            max_epochs=150,
            acc_at_germination=43.0,  # progress = 8.0
            config=config_d3,
            return_components=True,
        )

        # When contribution and progress are close, harmonic ~ geometric
        # Harmonic: 2 * 8 * 10 / (8 + 10) = 160 / 18 ~ 8.89
        # Timing discount: epoch 50 >= warmup 10 -> discount = 1.0
        # Total: 8.89 * 1.0 = 8.89

        expected_harmonic = 2 * 8 * 10 / (8 + 10)  # ~8.89

        assert components.bounded_attribution == pytest.approx(expected_harmonic, rel=0.01)
        assert components.timing_discount == pytest.approx(1.0)

    def test_contribution_less_than_progress_capped_at_contribution(self) -> None:
        """When contribution < progress, attribution is capped at contribution value.

        This is the normal case - seed contributed less than host improved.
        No anti-gaming formula applies; attribution = contribution.
        """
        config = ContributionRewardConfig(
            contribution_weight=1.0,
            attribution_formula="harmonic",  # Doesn't matter when contrib < progress
            disable_timing_discount=False,
        )

        seed = SeedInfo(
            stage=SeedStage.TRAINING.value,
            total_improvement=10.0,
            improvement_since_stage_start=0.5,
            epochs_in_stage=1,
            seed_params=1000,
            previous_stage=SeedStage.GERMINATED.value,
            previous_epochs_in_stage=0,
            seed_age_epochs=50,  # After warmup, so timing_discount = 1.0
            interaction_sum=0.0,
            boost_received=0.0,
        )

        # contribution (3) < progress (10), so capped at contribution
        _, components = compute_contribution_reward(
            action=LifecycleOp.WAIT,
            seed_contribution=3.0,
            val_acc=55.0,
            seed_info=seed,
            epoch=100,
            max_epochs=150,
            acc_at_germination=45.0,  # progress = 10.0
            config=config,
            return_components=True,
        )

        # When contribution < progress: attributed = contribution
        # timing_discount still applies but is 1.0 (after warmup)
        assert components.bounded_attribution == pytest.approx(3.0)
        assert components.timing_discount == pytest.approx(1.0)

    def test_timing_discount_linear_interpolation(self) -> None:
        """Verify timing discount interpolates linearly from floor to 1.0."""
        config = ContributionRewardConfig(
            contribution_weight=1.0,
            germination_warmup_epochs=10,
            germination_discount_floor=0.4,
            attribution_formula="geometric",
            disable_timing_discount=False,
        )

        # Test at various germination epochs
        test_cases = [
            # (germination_epoch, expected_discount)
            (0, 0.4),  # Floor at epoch 0
            (5, 0.7),  # Midpoint: 0.4 + 0.6 * 0.5 = 0.7
            (10, 1.0),  # Full credit at warmup
            (20, 1.0),  # Full credit after warmup
        ]

        for germ_epoch, expected_discount in test_cases:
            seed = SeedInfo(
                stage=SeedStage.TRAINING.value,
                total_improvement=5.0,
                improvement_since_stage_start=0.5,
                epochs_in_stage=1,
                seed_params=1000,
                previous_stage=SeedStage.GERMINATED.value,
                previous_epochs_in_stage=0,
                seed_age_epochs=100 - germ_epoch,  # Compute age from current epoch
                interaction_sum=0.0,
                boost_received=0.0,
            )

            # Use contribution > progress to ensure formula applies
            _, components = compute_contribution_reward(
                action=LifecycleOp.WAIT,
                seed_contribution=10.0,
                val_acc=50.0,
                seed_info=seed,
                epoch=100,
                max_epochs=150,
                acc_at_germination=45.0,  # progress = 5.0
                config=config,
                return_components=True,
            )

            assert components.timing_discount == pytest.approx(expected_discount, rel=0.01), (
                f"At germination epoch {germ_epoch}: "
                f"expected discount {expected_discount}, got {components.timing_discount}"
            )

    def test_pbrs_germination_bonus_discounted_early(self) -> None:
        """D3: PBRS germination bonus is discounted for early germination.

        This is the key fix: the agent was germinating early because
        PBRS gives ~0.25 immediate reward for transitioning to GERMINATED stage.
        With timing discount, early germination gets reduced PBRS bonus.
        """
        # Configuration with D3 timing discount enabled
        config_d3 = ContributionRewardConfig(
            germination_warmup_epochs=10,
            germination_discount_floor=0.4,
            disable_timing_discount=False,
            disable_pbrs=False,  # PBRS must be enabled
        )

        # Baseline without timing discount
        config_baseline = ContributionRewardConfig(
            disable_timing_discount=True,
            disable_pbrs=False,
        )

        # Germinate at epoch 0 (earliest possible, maximum penalty)
        reward_d3_early, comp_d3_early = compute_contribution_reward(
            action=LifecycleOp.GERMINATE,
            seed_contribution=None,  # No seed yet
            val_acc=45.0,
            seed_info=None,  # Empty slot
            epoch=0,
            max_epochs=150,
            config=config_d3,
            return_components=True,
        )

        reward_baseline_early, comp_baseline_early = compute_contribution_reward(
            action=LifecycleOp.GERMINATE,
            seed_contribution=None,
            val_acc=45.0,
            seed_info=None,
            epoch=0,
            max_epochs=150,
            config=config_baseline,
            return_components=True,
        )

        # Germinate at epoch 15 (after warmup, full bonus)
        reward_d3_late, comp_d3_late = compute_contribution_reward(
            action=LifecycleOp.GERMINATE,
            seed_contribution=None,
            val_acc=45.0,
            seed_info=None,
            epoch=15,
            max_epochs=150,
            config=config_d3,
            return_components=True,
        )

        # PBRS bonus baseline: gamma * phi_germinated * pbrs_weight
        # ≈ 0.99 * 1.0 * 0.3 ≈ 0.297, minus germinate_cost (-0.15) ≈ 0.147
        # At epoch 0: discount = 0.4, so PBRS portion is 0.4 * 0.297 ≈ 0.119
        # Total action_shaping ≈ 0.119 - 0.15 ≈ -0.03 (now NET NEGATIVE!)

        # D3 early germination should get significantly less action_shaping
        assert comp_d3_early.action_shaping < comp_baseline_early.action_shaping, (
            f"D3 early ({comp_d3_early.action_shaping}) should be less than "
            f"baseline ({comp_baseline_early.action_shaping})"
        )

        # D3 late germination (after warmup) should get full bonus
        assert comp_d3_late.action_shaping == pytest.approx(
            comp_baseline_early.action_shaping, rel=0.01
        ), "Late germination should get full PBRS bonus"

        # Verify the discount is significant (at least 40% reduction at epoch 0)
        reduction = 1 - (comp_d3_early.action_shaping / comp_baseline_early.action_shaping)
        assert reduction >= 0.35, f"Expected >= 35% reduction, got {reduction:.1%}"
