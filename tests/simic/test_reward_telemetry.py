"""Tests for reward component telemetry."""

import pytest


from esper.leyline import LifecycleOp, SeedStage
from esper.leyline.telemetry_contracts import RewardComponentsTelemetry
from esper.simic.rewards import (
    compute_contribution_reward,
    SeedInfo,
)


class TestRewardComponentsTelemetry:
    """Tests for RewardComponentsTelemetry."""

    def test_from_contribution_reward(self):
        """Can capture components from compute_contribution_reward."""
        seed_info = SeedInfo(
            stage=SeedStage.TRAINING.value,
            improvement_since_stage_start=1.5,
            total_improvement=2.0,
            epochs_in_stage=5,
            seed_params=10000,
            previous_stage=SeedStage.GERMINATED.value,
            seed_age_epochs=8,
        )

        # Use the extended version that returns components
        # seed_contribution=None uses proxy signal path
        reward, components = compute_contribution_reward(
            action=LifecycleOp.WAIT,
            seed_contribution=None,
            val_acc=65.0,
            seed_info=seed_info,
            epoch=10,
            max_epochs=25,
            total_params=10000,
            host_params=100000,
            acc_delta=0.5,  # Proxy signal
            return_components=True,
        )

        assert isinstance(components, RewardComponentsTelemetry)
        assert components.total_reward == reward

    def test_components_sum_to_total(self):
        """Component rewards sum to total reward."""
        seed_info = SeedInfo(
            stage=SeedStage.TRAINING.value,
            improvement_since_stage_start=1.0,
            total_improvement=1.5,
            epochs_in_stage=3,
            seed_params=5000,
            previous_stage=SeedStage.GERMINATED.value,
            seed_age_epochs=5,
        )

        reward, components = compute_contribution_reward(
            action=LifecycleOp.WAIT,
            seed_contribution=None,
            val_acc=60.0,
            seed_info=seed_info,
            epoch=5,
            max_epochs=25,
            total_params=5000,
            host_params=100000,
            acc_delta=0.3,  # Proxy signal
            return_components=True,
        )

        # ContributionReward uses bounded_attribution instead of base_acc_delta
        # B6-CR-01: Include ALL components for accurate accounting
        computed_sum = (
            components.bounded_attribution
            + components.blending_warning
            + components.holding_warning
            + components.compute_rent
            + components.alpha_shock
            + components.pbrs_bonus
            + components.action_shaping
            + components.terminal_bonus
            + components.synergy_bonus
        )
        assert abs(computed_sum - components.total_reward) < 1e-6

    def test_timing_discount_round_trip(self):
        """D3: timing_discount survives serialization round-trip."""
        # Create telemetry with non-default timing_discount
        original = RewardComponentsTelemetry(
            total_reward=1.5,
            bounded_attribution=1.0,
            timing_discount=0.65,  # Non-default value to catch serialization bugs
        )

        # Serialize and deserialize
        data = original.to_dict()
        restored = RewardComponentsTelemetry.from_dict(data)

        # Verify round-trip preserves timing_discount
        assert restored.timing_discount == original.timing_discount
        assert restored.timing_discount == 0.65

    def test_from_dict_requires_d2_d3_and_drip_fields(self):
        """Missing current reward component fields should fail loudly."""
        data = RewardComponentsTelemetry(
            total_reward=1.0,
            bounded_attribution=0.5,
        ).to_dict()
        del data["timing_discount"]

        with pytest.raises(KeyError, match="timing_discount"):
            RewardComponentsTelemetry.from_dict(data)

    def test_action_success_defaults_to_unknown_until_execution(self):
        """Reward components must not claim success before action execution."""
        original = RewardComponentsTelemetry()

        data = original.to_dict()
        restored = RewardComponentsTelemetry.from_dict(data)

        assert original.action_success is None
        assert data["action_success"] is None
        assert restored.action_success is None

    def test_from_dict_parses_false_action_success_string(self):
        """String boolean telemetry from intermediate exports preserves False."""
        data = RewardComponentsTelemetry(action_success=True).to_dict()
        data["action_success"] = "False"

        restored = RewardComponentsTelemetry.from_dict(data)

        assert restored.action_success is False

    def test_from_dict_parses_true_action_success_string(self):
        """String boolean telemetry from intermediate exports preserves True."""
        data = RewardComponentsTelemetry(action_success=False).to_dict()
        data["action_success"] = "True"

        restored = RewardComponentsTelemetry.from_dict(data)

        assert restored.action_success is True

    def test_from_dict_rejects_invalid_action_success_string(self):
        """Malformed action_success strings should fail instead of guessing."""
        data = RewardComponentsTelemetry(action_success=True).to_dict()
        data["action_success"] = "truthy"

        with pytest.raises(ValueError, match="action_success"):
            RewardComponentsTelemetry.from_dict(data)
