"""Tests for SIMPLIFIED reward mode."""
import pytest
from esper.simic.rewards import RewardMode


def test_simplified_mode_exists():
    """RewardMode.SIMPLIFIED should be a valid enum member."""
    assert hasattr(RewardMode, "SIMPLIFIED")
    assert RewardMode.SIMPLIFIED.value == "simplified"


def test_simplified_mode_string_conversion():
    """SIMPLIFIED mode should round-trip through string."""
    mode = RewardMode("simplified")
    assert mode == RewardMode.SIMPLIFIED


from esper.simic.rewards import (
    compute_simplified_reward,
    ContributionRewardConfig,
    SeedInfo,
    STAGE_TRAINING,
    STAGE_PROBATIONARY,
)
from esper.leyline.factored_actions import LifecycleOp


class TestComputeSimplifiedReward:
    """Tests for the simplified 3-component reward."""

    def test_non_terminal_returns_pbrs_plus_cost(self):
        """Non-terminal steps: PBRS + intervention cost only."""
        config = ContributionRewardConfig()
        seed_info = SeedInfo(
            stage=STAGE_TRAINING,
            improvement_since_stage_start=0.5,
            total_improvement=1.0,
            epochs_in_stage=3,
            seed_params=1000,
            previous_stage=STAGE_TRAINING,
            previous_epochs_in_stage=2,
            seed_age_epochs=5,
        )

        # WAIT action: only PBRS, no intervention cost
        reward_wait = compute_simplified_reward(
            action=LifecycleOp.WAIT,
            seed_info=seed_info,
            epoch=10,
            max_epochs=25,
            val_acc=65.0,
            num_contributing_fossilized=0,
            config=config,
        )

        # GERMINATE action: PBRS + intervention cost
        reward_germinate = compute_simplified_reward(
            action=LifecycleOp.GERMINATE,
            seed_info=None,  # No seed when germinating
            epoch=10,
            max_epochs=25,
            val_acc=65.0,
            num_contributing_fossilized=0,
            config=config,
        )

        # Germinate should have small negative cost
        assert reward_germinate < 0.0  # Intervention cost

    def test_terminal_includes_accuracy_and_fossilize_bonus(self):
        """Terminal step: PBRS + cost + accuracy + fossilize bonus."""
        config = ContributionRewardConfig()

        # Terminal with 2 contributing fossilized seeds
        reward = compute_simplified_reward(
            action=LifecycleOp.WAIT,
            seed_info=None,
            epoch=25,
            max_epochs=25,
            val_acc=75.0,
            num_contributing_fossilized=2,
            config=config,
        )

        # Should have: accuracy bonus (75/100 * 3 = 2.25) + fossilize bonus (2 * 2 = 4)
        # Total terminal ~ 6.25
        assert reward > 5.0
        assert reward < 8.0

    def test_no_attribution_component(self):
        """SIMPLIFIED should NOT include bounded_attribution."""
        config = ContributionRewardConfig()
        seed_info = SeedInfo(
            stage=STAGE_PROBATIONARY,
            improvement_since_stage_start=5.0,  # High improvement
            total_improvement=10.0,
            epochs_in_stage=3,
            seed_params=1000,
            previous_stage=STAGE_PROBATIONARY,
            previous_epochs_in_stage=2,
            seed_age_epochs=15,
        )

        # With SHAPED, high improvement would give large attribution reward
        # With SIMPLIFIED, there should be NO attribution component
        reward = compute_simplified_reward(
            action=LifecycleOp.WAIT,
            seed_info=seed_info,
            epoch=15,
            max_epochs=25,
            val_acc=70.0,
            num_contributing_fossilized=0,
            config=config,
        )

        # Reward should be small (just PBRS epoch progress)
        # Not inflated by attribution
        assert abs(reward) < 1.0

    def test_no_warning_components(self):
        """SIMPLIFIED should NOT include blending_warning or probation_warning."""
        config = ContributionRewardConfig()
        seed_info = SeedInfo(
            stage=STAGE_PROBATIONARY,
            improvement_since_stage_start=-5.0,  # Negative (would trigger warnings)
            total_improvement=-3.0,
            epochs_in_stage=5,  # Long time (would trigger probation_warning)
            seed_params=1000,
            previous_stage=STAGE_PROBATIONARY,
            previous_epochs_in_stage=4,
            seed_age_epochs=20,
        )

        # With SHAPED, this would trigger severe probation_warning (-9.0 or worse)
        # With SIMPLIFIED, no warning penalties
        reward = compute_simplified_reward(
            action=LifecycleOp.WAIT,
            seed_info=seed_info,
            epoch=20,
            max_epochs=25,
            val_acc=60.0,
            num_contributing_fossilized=0,
            config=config,
        )

        # Should NOT have the -9.0 probation_warning
        assert reward > -2.0


from esper.simic.rewards import compute_reward


class TestComputeRewardDispatcher:
    """Test that compute_reward dispatches to SIMPLIFIED correctly."""

    def test_simplified_mode_dispatches(self):
        """compute_reward with SIMPLIFIED mode should use simplified logic."""
        config = ContributionRewardConfig(reward_mode=RewardMode.SIMPLIFIED)
        seed_info = SeedInfo(
            stage=STAGE_PROBATIONARY,
            improvement_since_stage_start=-5.0,  # Would trigger warnings in SHAPED
            total_improvement=-3.0,
            epochs_in_stage=5,
            seed_params=1000,
            previous_stage=STAGE_PROBATIONARY,
            previous_epochs_in_stage=4,
            seed_age_epochs=20,
        )

        # Call through dispatcher
        reward = compute_reward(
            action=LifecycleOp.WAIT,
            seed_contribution=5.0,  # Would give attribution in SHAPED
            val_acc=60.0,
            host_max_acc=65.0,
            seed_info=seed_info,
            epoch=20,
            max_epochs=25,
            total_params=100000,
            host_params=90000,
            acc_at_germination=55.0,
            acc_delta=0.5,
            num_fossilized_seeds=1,
            num_contributing_fossilized=1,
            config=config,
        )

        # Should NOT have probation_warning (-9.0) or attribution (+5.0)
        # Should be small (just PBRS)
        assert -2.0 < reward < 2.0


from esper.simic.training.config import TrainingConfig


class TestABTestingConfig:
    """Test A/B testing configuration."""

    def test_ab_reward_modes_field_exists(self):
        """TrainingConfig should have ab_reward_modes field."""
        config = TrainingConfig()
        # Default should be None (all envs use reward_mode)
        assert config.ab_reward_modes is None

    def test_ab_reward_modes_splits_envs(self):
        """ab_reward_modes should specify per-env reward modes."""
        config = TrainingConfig(
            n_envs=8,
            ab_reward_modes=["shaped", "shaped", "shaped", "shaped",
                            "simplified", "simplified", "simplified", "simplified"],
        )
        assert len(config.ab_reward_modes) == 8
        assert config.ab_reward_modes[0] == "shaped"
        assert config.ab_reward_modes[4] == "simplified"

    def test_ab_reward_modes_validation(self):
        """ab_reward_modes length must match n_envs."""
        with pytest.raises(ValueError, match="ab_reward_modes.*must match.*n_envs"):
            TrainingConfig(
                n_envs=8,
                ab_reward_modes=["shaped", "simplified"],  # Wrong length
            )
