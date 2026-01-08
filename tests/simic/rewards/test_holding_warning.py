"""Tests for holding_warning penalty in HOLDING stage.

Bug fixed (2026-01-08): SET_ALPHA_TARGET was exempt from holding_warning,
creating an exploit where Tamiyo could turntable alpha settings to avoid
the indecision penalty while collecting dense positive rewards.
"""

import pytest

from esper.leyline import LifecycleOp, MIN_PRUNE_AGE, SeedStage
from esper.simic.rewards import compute_contribution_reward, ContributionRewardConfig
from esper.simic.rewards.types import SeedInfo


def holding_config() -> ContributionRewardConfig:
    """Config that isolates holding_warning from other reward components."""
    return ContributionRewardConfig(
        # Zero out other shaping to isolate holding_warning
        contribution_weight=1.0,
        disable_pbrs=True,
        disable_terminal_reward=True,
        disable_anti_gaming=True,
        rent_weight=0.0,
        alpha_shock_coef=0.0,
        germinate_cost=0.0,
        fossilize_cost=0.0,
        prune_cost=0.0,
        set_alpha_target_cost=0.0,
        germinate_with_seed_penalty=0.0,
    )


def holding_seed_info(epochs_in_stage: int = 3) -> SeedInfo:
    """Create a SeedInfo for a seed in HOLDING stage."""
    return SeedInfo(
        stage=SeedStage.HOLDING.value,
        improvement_since_stage_start=0.0,
        total_improvement=5.0,
        epochs_in_stage=epochs_in_stage,
        seed_params=1000,
        previous_stage=SeedStage.BLENDING.value,
        previous_epochs_in_stage=5,
        seed_age_epochs=MIN_PRUNE_AGE + 5,
        interaction_sum=0.0,
        boost_received=0.0,
    )


class TestHoldingWarningPenalty:
    """Tests for holding_warning penalty in HOLDING stage."""

    def test_wait_in_holding_triggers_warning(self) -> None:
        """WAIT in HOLDING with positive attribution triggers holding_warning."""
        config = holding_config()
        seed_info = holding_seed_info(epochs_in_stage=3)

        reward, components = compute_contribution_reward(
            action=LifecycleOp.WAIT,
            seed_contribution=5.0,  # Positive contribution
            val_acc=50.0,
            seed_info=seed_info,
            epoch=10,
            max_epochs=100,
            config=config,
            return_components=True,
            acc_at_germination=45.0,  # Positive progress
        )

        # WAIT should trigger holding_warning
        assert components.holding_warning < 0, "WAIT in HOLDING should incur penalty"

    def test_set_alpha_target_in_holding_triggers_warning(self) -> None:
        """SET_ALPHA_TARGET in HOLDING with positive attribution triggers holding_warning.

        Bug fix: Previously SET_ALPHA_TARGET was exempt, creating turntabling exploit.
        """
        config = holding_config()
        seed_info = holding_seed_info(epochs_in_stage=3)

        reward, components = compute_contribution_reward(
            action=LifecycleOp.SET_ALPHA_TARGET,
            seed_contribution=5.0,  # Positive contribution
            val_acc=50.0,
            seed_info=seed_info,
            epoch=10,
            max_epochs=100,
            config=config,
            return_components=True,
            acc_at_germination=45.0,  # Positive progress
        )

        # SET_ALPHA_TARGET should ALSO trigger holding_warning (the fix)
        assert components.holding_warning < 0, (
            "SET_ALPHA_TARGET in HOLDING should incur penalty (turntabling fix)"
        )

    def test_fossilize_in_holding_exempt_from_warning(self) -> None:
        """FOSSILIZE in HOLDING is exempt - it's a terminal decision."""
        config = holding_config()
        seed_info = holding_seed_info(epochs_in_stage=3)

        reward, components = compute_contribution_reward(
            action=LifecycleOp.FOSSILIZE,
            seed_contribution=5.0,
            val_acc=50.0,
            seed_info=seed_info,
            epoch=10,
            max_epochs=100,
            config=config,
            return_components=True,
            acc_at_germination=45.0,
        )

        # FOSSILIZE is exempt - it resolves the holding state
        assert components.holding_warning == 0.0, "FOSSILIZE should be exempt from penalty"

    def test_prune_in_holding_exempt_from_warning(self) -> None:
        """PRUNE in HOLDING is exempt - it's a terminal decision."""
        config = holding_config()
        seed_info = holding_seed_info(epochs_in_stage=3)

        reward, components = compute_contribution_reward(
            action=LifecycleOp.PRUNE,
            seed_contribution=5.0,
            val_acc=50.0,
            seed_info=seed_info,
            epoch=10,
            max_epochs=100,
            config=config,
            return_components=True,
            acc_at_germination=45.0,
        )

        # PRUNE is exempt - it resolves the holding state
        assert components.holding_warning == 0.0, "PRUNE should be exempt from penalty"

    def test_holding_warning_requires_positive_attribution(self) -> None:
        """Holding warning only fires when bounded_attribution > 0."""
        config = holding_config()
        seed_info = holding_seed_info(epochs_in_stage=3)

        # Negative contribution = no penalty (seed is hurting, agent should prune)
        reward, components = compute_contribution_reward(
            action=LifecycleOp.SET_ALPHA_TARGET,
            seed_contribution=-2.0,  # Negative contribution
            val_acc=50.0,
            seed_info=seed_info,
            epoch=10,
            max_epochs=100,
            config=config,
            return_components=True,
            acc_at_germination=45.0,
        )

        # No penalty when attribution is negative
        assert components.holding_warning == 0.0, (
            "No penalty when seed contribution is negative"
        )

    def test_holding_warning_requires_epochs_in_stage_ge_2(self) -> None:
        """Holding warning only fires after epochs_in_stage >= 2."""
        config = holding_config()
        seed_info = holding_seed_info(epochs_in_stage=1)  # Just entered HOLDING

        reward, components = compute_contribution_reward(
            action=LifecycleOp.SET_ALPHA_TARGET,
            seed_contribution=5.0,
            val_acc=50.0,
            seed_info=seed_info,
            epoch=10,
            max_epochs=100,
            config=config,
            return_components=True,
            acc_at_germination=45.0,
        )

        # No penalty in first epoch of HOLDING
        assert components.holding_warning == 0.0, (
            "No penalty in first epoch of HOLDING stage"
        )


class TestHoldingWarningParity:
    """Integration tests ensuring WAIT and SET_ALPHA_TARGET have equal penalties."""

    def test_wait_and_set_alpha_target_same_penalty_in_holding(self) -> None:
        """WAIT and SET_ALPHA_TARGET should incur the same holding_warning.

        This prevents any action from being a 'free WAIT' loophole.
        """
        config = holding_config()
        seed_info = holding_seed_info(epochs_in_stage=5)

        # WAIT
        _, wait_components = compute_contribution_reward(
            action=LifecycleOp.WAIT,
            seed_contribution=5.0,
            val_acc=50.0,
            seed_info=seed_info,
            epoch=10,
            max_epochs=100,
            config=config,
            return_components=True,
            acc_at_germination=45.0,
        )

        # SET_ALPHA_TARGET
        _, set_alpha_components = compute_contribution_reward(
            action=LifecycleOp.SET_ALPHA_TARGET,
            seed_contribution=5.0,
            val_acc=50.0,
            seed_info=seed_info,
            epoch=10,
            max_epochs=100,
            config=config,
            return_components=True,
            acc_at_germination=45.0,
        )

        # Both should have the SAME holding_warning
        assert wait_components.holding_warning == set_alpha_components.holding_warning, (
            f"WAIT ({wait_components.holding_warning}) and SET_ALPHA_TARGET "
            f"({set_alpha_components.holding_warning}) should have same penalty"
        )
        assert wait_components.holding_warning < 0, "Both should be penalized"
