"""Baseline regression tests for SHAPED reward mode.

These tests capture the current behavior of compute_contribution_reward
BEFORE the sparse reward experiment changes. They serve as a safety net
to ensure the existing SHAPED mode behavior is preserved.

Created as risk reduction before sparse-reward-experiment implementation.
"""

import pytest
from esper.leyline import SeedStage
from esper.leyline.factored_actions import LifecycleOp
from esper.simic.rewards import (
    compute_contribution_reward,
    ContributionRewardConfig,
    SeedInfo,
)


class TestShapedModeBaseline:
    """Baseline tests for SHAPED mode regression detection."""

    def test_no_seed_dormant_slot_reward(self):
        """Reward for WAIT action with no active seed (DORMANT slot).

        This captures the baseline compute_rent behavior.
        """
        config = ContributionRewardConfig()

        reward = compute_contribution_reward(
            action=LifecycleOp.WAIT,
            seed_contribution=None,
            val_acc=70.0,
            seed_info=None,
            epoch=10,
            max_epochs=25,
            total_params=100_000,
            host_params=100_000,
            acc_at_germination=None,
            acc_delta=0.5,
        )

        # Should be negative (compute rent for idle slot)
        # Exact value may vary, but should be small negative
        assert isinstance(reward, float)
        assert -1.0 < reward < 0.5  # Reasonable bounds for idle slot

    def test_blending_seed_with_positive_contribution(self):
        """Reward for BLENDING stage seed with positive counterfactual contribution."""
        config = ContributionRewardConfig()

        seed_info = SeedInfo(
            stage=SeedStage.BLENDING.value,
            improvement_since_stage_start=1.5,
            total_improvement=2.0,
            epochs_in_stage=3,
            seed_params=50_000,
            previous_stage=SeedStage.TRAINING.value,
            previous_epochs_in_stage=2,
            seed_age_epochs=5,
        )

        reward = compute_contribution_reward(
            action=LifecycleOp.WAIT,
            seed_contribution=2.5,  # Positive contribution
            val_acc=75.0,
            seed_info=seed_info,
            epoch=15,
            max_epochs=25,
            total_params=150_000,
            host_params=100_000,
            acc_at_germination=70.0,
            acc_delta=0.3,
        )

        # Positive contribution should yield positive reward
        assert reward > 0, f"Positive contribution should yield positive reward, got {reward}"

    def test_terminal_epoch_includes_bonus(self):
        """Terminal epoch should include accuracy bonus."""
        config = ContributionRewardConfig()

        seed_info = SeedInfo(
            stage=SeedStage.FOSSILIZED.value,
            improvement_since_stage_start=0.0,
            total_improvement=3.0,
            epochs_in_stage=1,
            seed_params=50_000,
            previous_stage=SeedStage.PROBATIONARY.value,
            previous_epochs_in_stage=3,
            seed_age_epochs=10,
        )

        # Non-terminal epoch
        reward_mid = compute_contribution_reward(
            action=LifecycleOp.WAIT,
            seed_contribution=None,  # Fossilized, no contribution
            val_acc=80.0,
            seed_info=seed_info,
            epoch=15,
            max_epochs=25,
            total_params=150_000,
            host_params=100_000,
            acc_at_germination=70.0,
            acc_delta=0.0,
            num_fossilized_seeds=1,
            num_contributing_fossilized=1,
        )

        # Terminal epoch
        reward_terminal = compute_contribution_reward(
            action=LifecycleOp.WAIT,
            seed_contribution=None,
            val_acc=80.0,
            seed_info=seed_info,
            epoch=25,  # Terminal
            max_epochs=25,
            total_params=150_000,
            host_params=100_000,
            acc_at_germination=70.0,
            acc_delta=0.0,
            num_fossilized_seeds=1,
            num_contributing_fossilized=1,
        )

        # Terminal should have higher reward (bonus)
        assert reward_terminal > reward_mid, (
            f"Terminal reward ({reward_terminal}) should exceed mid-episode ({reward_mid})"
        )

    def test_fossilize_action_bonus(self):
        """FOSSILIZE action in PROBATIONARY stage should get bonus."""
        seed_info = SeedInfo(
            stage=SeedStage.PROBATIONARY.value,
            improvement_since_stage_start=0.5,
            total_improvement=2.0,
            epochs_in_stage=3,
            seed_params=50_000,
            previous_stage=SeedStage.BLENDING.value,
            previous_epochs_in_stage=5,
            seed_age_epochs=10,
        )

        reward = compute_contribution_reward(
            action=LifecycleOp.FOSSILIZE,
            seed_contribution=1.5,
            val_acc=75.0,
            seed_info=seed_info,
            epoch=15,
            max_epochs=25,
            total_params=150_000,
            host_params=100_000,
            acc_at_germination=70.0,
            acc_delta=0.2,
        )

        # Valid fossilize should get a positive reward
        assert reward > 0, f"Valid FOSSILIZE should be rewarded, got {reward}"

    def test_invalid_fossilize_penalty(self):
        """FOSSILIZE in non-PROBATIONARY stage should get penalty."""
        seed_info = SeedInfo(
            stage=SeedStage.TRAINING.value,  # Can't fossilize from TRAINING
            improvement_since_stage_start=0.5,
            total_improvement=1.0,
            epochs_in_stage=2,
            seed_params=50_000,
            previous_stage=SeedStage.GERMINATED.value,
            previous_epochs_in_stage=1,
            seed_age_epochs=3,
        )

        reward = compute_contribution_reward(
            action=LifecycleOp.FOSSILIZE,
            seed_contribution=None,  # No contribution in TRAINING
            val_acc=72.0,
            seed_info=seed_info,
            epoch=10,
            max_epochs=25,
            total_params=150_000,
            host_params=100_000,
            acc_at_germination=70.0,
            acc_delta=0.3,
        )

        # Invalid fossilize should get penalty
        assert reward < 0, f"Invalid FOSSILIZE should be penalized, got {reward}"

    def test_reward_components_sum(self):
        """Reward components should sum to total reward."""
        seed_info = SeedInfo(
            stage=SeedStage.BLENDING.value,
            improvement_since_stage_start=1.0,
            total_improvement=1.5,
            epochs_in_stage=2,
            seed_params=40_000,
            previous_stage=SeedStage.TRAINING.value,
            previous_epochs_in_stage=3,
            seed_age_epochs=5,
        )

        reward, components = compute_contribution_reward(
            action=LifecycleOp.WAIT,
            seed_contribution=1.2,
            val_acc=73.0,
            seed_info=seed_info,
            epoch=12,
            max_epochs=25,
            total_params=140_000,
            host_params=100_000,
            acc_at_germination=70.0,
            acc_delta=0.2,
            return_components=True,
        )

        # Components should exist
        assert components is not None
        assert hasattr(components, 'total_reward')

        # Total should match
        assert abs(components.total_reward - reward) < 0.001, (
            f"Component total ({components.total_reward}) != returned reward ({reward})"
        )


class TestShapedModeEdgeCases:
    """Edge case tests for SHAPED mode."""

    def test_zero_params_no_crash(self):
        """Zero params should not cause division errors."""
        reward = compute_contribution_reward(
            action=LifecycleOp.WAIT,
            seed_contribution=None,
            val_acc=70.0,
            seed_info=None,
            epoch=10,
            max_epochs=25,
            total_params=0,
            host_params=0,
            acc_at_germination=None,
            acc_delta=0.0,
        )

        import math
        assert math.isfinite(reward), f"Reward should be finite, got {reward}"

    def test_extreme_accuracy_bounded(self):
        """Extreme accuracy values should produce bounded rewards."""
        import math

        for acc in [0.0, 50.0, 100.0]:
            reward = compute_contribution_reward(
                action=LifecycleOp.WAIT,
                seed_contribution=None,
                val_acc=acc,
                seed_info=None,
                epoch=10,
                max_epochs=25,
                total_params=100_000,
                host_params=100_000,
                acc_at_germination=None,
                acc_delta=0.0,
            )

            assert math.isfinite(reward), f"acc={acc} produced non-finite reward: {reward}"
            assert -10.0 < reward < 10.0, f"acc={acc} produced unbounded reward: {reward}"
