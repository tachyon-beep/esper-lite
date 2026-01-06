"""Tests for SIMPLIFIED reward mode."""


from esper.leyline import LifecycleOp
from esper.simic.rewards import (
    STAGE_HOLDING,
    STAGE_TRAINING,
    ContributionRewardConfig,
    ContributionRewardInputs,
    RewardMode,
    SeedInfo,
    compute_reward,
    compute_simplified_reward,
)
def test_simplified_mode_exists():
    """RewardMode.SIMPLIFIED should be a valid enum member."""
    assert hasattr(RewardMode, "SIMPLIFIED")
    assert RewardMode.SIMPLIFIED.value == "simplified"


def test_simplified_mode_string_conversion():
    """SIMPLIFIED mode should round-trip through string."""
    mode = RewardMode("simplified")
    assert mode == RewardMode.SIMPLIFIED


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
        compute_simplified_reward(
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
            stage=STAGE_HOLDING,
            improvement_since_stage_start=5.0,  # High improvement
            total_improvement=10.0,
            epochs_in_stage=3,
            seed_params=1000,
            previous_stage=STAGE_HOLDING,
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
        """SIMPLIFIED should NOT include blending_warning or holding_warning."""
        config = ContributionRewardConfig()
        seed_info = SeedInfo(
            stage=STAGE_HOLDING,
            improvement_since_stage_start=-5.0,  # Negative (would trigger warnings)
            total_improvement=-3.0,
            epochs_in_stage=5,  # Long time (would trigger holding_warning)
            seed_params=1000,
            previous_stage=STAGE_HOLDING,
            previous_epochs_in_stage=4,
            seed_age_epochs=20,
        )

        # With SHAPED, this would trigger holding_warning (-0.25 at epoch 5)
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

        # Should NOT have the -0.25 holding_warning (SIMPLIFIED skips warnings)
        assert reward > -2.0


class TestComputeRewardDispatcher:
    """Test that compute_reward dispatches to SIMPLIFIED correctly."""

    def test_simplified_mode_dispatches(self):
        """compute_reward with SIMPLIFIED mode should use simplified logic."""
        config = ContributionRewardConfig(reward_mode=RewardMode.SIMPLIFIED)
        seed_info = SeedInfo(
            stage=STAGE_HOLDING,
            improvement_since_stage_start=-5.0,  # Would trigger warnings in SHAPED
            total_improvement=-3.0,
            epochs_in_stage=5,
            seed_params=1000,
            previous_stage=STAGE_HOLDING,
            previous_epochs_in_stage=4,
            seed_age_epochs=20,
        )

        # Call through dispatcher
        reward = compute_reward(
            ContributionRewardInputs(
                action=LifecycleOp.WAIT,
                seed_contribution=5.0,  # Would give attribution in SHAPED
                val_acc=60.0,
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
        )

        # Should NOT have holding_warning (-9.0) or attribution (+5.0)
        # Should be small (just PBRS)
        assert -2.0 < reward < 2.0
