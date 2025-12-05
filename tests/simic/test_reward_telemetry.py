"""Tests for reward component telemetry."""

import pytest

from esper.simic.reward_telemetry import RewardComponentsTelemetry
from esper.simic.rewards import (
    compute_shaped_reward,
    compute_contribution_reward,
    SeedInfo,
    RewardConfig,
    ContributionRewardConfig,
)


class TestRewardComponentsTelemetry:
    """Tests for RewardComponentsTelemetry."""

    def test_from_shaped_reward(self):
        """Can capture components from compute_shaped_reward."""
        # Create a mock action enum
        from enum import IntEnum

        class MockAction(IntEnum):
            WAIT = 0
            GERMINATE = 1
            FOSSILIZE = 2
            CULL = 3

        seed_info = SeedInfo(
            stage=3,  # TRAINING
            improvement_since_stage_start=1.5,
            total_improvement=2.0,
            epochs_in_stage=5,
            seed_params=10000,
            previous_stage=2,
            seed_age_epochs=8,
        )

        # Use the extended version that returns components
        reward, components = compute_shaped_reward(
            action=MockAction.WAIT,
            acc_delta=0.5,
            val_acc=65.0,
            seed_info=seed_info,
            epoch=10,
            max_epochs=25,
            total_params=10000,
            host_params=100000,
            return_components=True,
        )

        assert isinstance(components, RewardComponentsTelemetry)
        assert components.base_acc_delta != 0.0
        assert components.total_reward == reward

    def test_components_sum_to_total(self):
        """Component rewards sum to total reward."""
        from enum import IntEnum

        class MockAction(IntEnum):
            WAIT = 0

        seed_info = SeedInfo(
            stage=3,
            improvement_since_stage_start=1.0,
            total_improvement=1.5,
            epochs_in_stage=3,
            seed_params=5000,
            previous_stage=2,
            seed_age_epochs=5,
        )

        reward, components = compute_shaped_reward(
            action=MockAction.WAIT,
            acc_delta=0.3,
            val_acc=60.0,
            seed_info=seed_info,
            epoch=5,
            max_epochs=25,
            total_params=5000,
            host_params=100000,
            return_components=True,
        )

        computed_sum = (
            components.base_acc_delta
            + components.compute_rent
            + components.stage_bonus
            + components.pbrs_bonus
            + components.action_shaping
            + components.terminal_bonus
        )
        assert abs(computed_sum - components.total_reward) < 1e-6
