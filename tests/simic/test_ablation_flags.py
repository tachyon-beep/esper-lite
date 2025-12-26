"""Tests for ablation configuration flags."""

import pytest
from esper.leyline import LifecycleOp, SeedStage
from esper.simic.training.config import TrainingConfig
from esper.simic.rewards import (
    compute_contribution_reward,
    ContributionRewardConfig,
    SeedInfo,
)


def test_ablation_flags_exist():
    """TrainingConfig has ablation flags with correct defaults."""
    config = TrainingConfig()
    assert config.disable_pbrs is False
    assert config.disable_terminal_reward is False
    assert config.disable_anti_gaming is False


def test_ablation_flags_settable():
    """Ablation flags can be set via constructor."""
    config = TrainingConfig(
        disable_pbrs=True,
        disable_terminal_reward=True,
        disable_anti_gaming=True,
    )
    assert config.disable_pbrs is True
    assert config.disable_terminal_reward is True
    assert config.disable_anti_gaming is True


def test_ablation_flags_in_to_train_kwargs():
    """Ablation flags are passed through to_train_kwargs()."""
    config = TrainingConfig(disable_pbrs=True)
    kwargs = config.to_train_kwargs()
    assert kwargs.get("disable_pbrs") is True
    assert kwargs.get("disable_terminal_reward") is False
    assert kwargs.get("disable_anti_gaming") is False


def test_ablation_flags_all_in_to_train_kwargs():
    """All ablation flags are passed through to_train_kwargs()."""
    config = TrainingConfig(
        disable_pbrs=True,
        disable_terminal_reward=True,
        disable_anti_gaming=True,
    )
    kwargs = config.to_train_kwargs()
    assert kwargs["disable_pbrs"] is True
    assert kwargs["disable_terminal_reward"] is True
    assert kwargs["disable_anti_gaming"] is True


def test_ablation_flags_from_dict():
    """Ablation flags can be set via from_dict()."""
    data = {
        "disable_pbrs": True,
        "disable_terminal_reward": True,
        "disable_anti_gaming": True,
    }
    config = TrainingConfig.from_dict(data)
    assert config.disable_pbrs is True
    assert config.disable_terminal_reward is True
    assert config.disable_anti_gaming is True


def test_ablation_flags_in_to_dict():
    """Ablation flags are included in to_dict()."""
    config = TrainingConfig(
        disable_pbrs=True,
        disable_terminal_reward=True,
        disable_anti_gaming=True,
    )
    data = config.to_dict()
    assert data["disable_pbrs"] is True
    assert data["disable_terminal_reward"] is True
    assert data["disable_anti_gaming"] is True


# =============================================================================
# Reward Computation Tests - Verify flags affect behavior
# =============================================================================


def _make_seed_info(stage: int = SeedStage.BLENDING.value) -> SeedInfo:
    """Create a seed info for testing."""
    return SeedInfo(
        stage=stage,
        improvement_since_stage_start=0.5,
        total_improvement=1.0,
        epochs_in_stage=3,
        seed_params=1000,
        previous_stage=SeedStage.TRAINING.value,
        previous_epochs_in_stage=2,
        seed_age_epochs=5,
    )


def test_disable_pbrs_zeroes_pbrs_bonus():
    """disable_pbrs=True should zero out PBRS stage progression bonus."""
    seed_info = _make_seed_info()

    # With PBRS enabled (default)
    config_enabled = ContributionRewardConfig(disable_pbrs=False)
    reward_enabled, components_enabled = compute_contribution_reward(
        action=LifecycleOp.WAIT,
        seed_contribution=0.5,
        val_acc=60.0,
        seed_info=seed_info,
        epoch=10,
        max_epochs=25,
        config=config_enabled,
        return_components=True,
    )

    # With PBRS disabled
    config_disabled = ContributionRewardConfig(disable_pbrs=True)
    reward_disabled, components_disabled = compute_contribution_reward(
        action=LifecycleOp.WAIT,
        seed_contribution=0.5,
        val_acc=60.0,
        seed_info=seed_info,
        epoch=10,
        max_epochs=25,
        config=config_disabled,
        return_components=True,
    )

    # PBRS bonus should be zero when disabled
    assert components_disabled.pbrs_bonus == 0.0
    # PBRS bonus should be non-zero when enabled (for progressing seed)
    assert components_enabled.pbrs_bonus != 0.0
    # Total reward should differ
    assert reward_enabled != reward_disabled


def test_disable_terminal_reward_zeroes_terminal_bonus():
    """disable_terminal_reward=True should zero out terminal accuracy bonus."""
    seed_info = _make_seed_info(stage=SeedStage.HOLDING.value)

    # At terminal epoch with terminal reward enabled
    config_enabled = ContributionRewardConfig(disable_terminal_reward=False)
    reward_enabled, components_enabled = compute_contribution_reward(
        action=LifecycleOp.WAIT,
        seed_contribution=0.5,
        val_acc=60.0,
        seed_info=seed_info,
        epoch=25,  # terminal
        max_epochs=25,
        config=config_enabled,
        return_components=True,
        num_contributing_fossilized=1,
    )

    # With terminal reward disabled
    config_disabled = ContributionRewardConfig(disable_terminal_reward=True)
    reward_disabled, components_disabled = compute_contribution_reward(
        action=LifecycleOp.WAIT,
        seed_contribution=0.5,
        val_acc=60.0,
        seed_info=seed_info,
        epoch=25,  # terminal
        max_epochs=25,
        config=config_disabled,
        return_components=True,
        num_contributing_fossilized=1,
    )

    # Terminal bonus should be zero when disabled
    assert components_disabled.terminal_bonus == 0.0
    # Terminal bonus should be non-zero when enabled
    assert components_enabled.terminal_bonus != 0.0
    # Total reward should differ
    assert reward_enabled != reward_disabled


def test_disable_anti_gaming_zeroes_ratio_penalty():
    """disable_anti_gaming=True should zero out ratio_penalty."""
    # Create seed with high contribution but low improvement (triggers ratio penalty)
    seed_info = SeedInfo(
        stage=SeedStage.BLENDING.value,
        improvement_since_stage_start=0.01,
        total_improvement=0.05,  # Low improvement
        epochs_in_stage=3,
        seed_params=1000,
        previous_stage=SeedStage.TRAINING.value,
        previous_epochs_in_stage=2,
        seed_age_epochs=5,
    )

    # With anti-gaming enabled (high contribution, low improvement -> ratio penalty)
    config_enabled = ContributionRewardConfig(disable_anti_gaming=False)
    reward_enabled, components_enabled = compute_contribution_reward(
        action=LifecycleOp.WAIT,
        seed_contribution=2.0,  # High contribution >> improvement
        val_acc=60.0,
        seed_info=seed_info,
        epoch=10,
        max_epochs=25,
        config=config_enabled,
        return_components=True,
        acc_at_germination=59.0,
    )

    # With anti-gaming disabled
    config_disabled = ContributionRewardConfig(disable_anti_gaming=True)
    reward_disabled, components_disabled = compute_contribution_reward(
        action=LifecycleOp.WAIT,
        seed_contribution=2.0,  # Same high contribution
        val_acc=60.0,
        seed_info=seed_info,
        epoch=10,
        max_epochs=25,
        config=config_disabled,
        return_components=True,
        acc_at_germination=59.0,
    )

    # Ratio penalty should be zero when anti-gaming is disabled
    assert components_disabled.ratio_penalty == 0.0
    # Ratio penalty should be non-zero when enabled (contribution >> improvement)
    assert components_enabled.ratio_penalty != 0.0


def test_disable_anti_gaming_zeroes_alpha_shock():
    """disable_anti_gaming=True should zero out alpha_shock penalty."""
    seed_info = _make_seed_info()

    # With anti-gaming enabled and alpha changes
    config_enabled = ContributionRewardConfig(disable_anti_gaming=False)
    reward_enabled, components_enabled = compute_contribution_reward(
        action=LifecycleOp.WAIT,
        seed_contribution=0.5,
        val_acc=60.0,
        seed_info=seed_info,
        epoch=10,
        max_epochs=25,
        config=config_enabled,
        return_components=True,
        alpha_delta_sq_sum=0.5,  # Non-zero alpha changes
    )

    # With anti-gaming disabled
    config_disabled = ContributionRewardConfig(disable_anti_gaming=True)
    reward_disabled, components_disabled = compute_contribution_reward(
        action=LifecycleOp.WAIT,
        seed_contribution=0.5,
        val_acc=60.0,
        seed_info=seed_info,
        epoch=10,
        max_epochs=25,
        config=config_disabled,
        return_components=True,
        alpha_delta_sq_sum=0.5,  # Same alpha changes
    )

    # Alpha shock should be zero when anti-gaming is disabled
    assert components_disabled.alpha_shock == 0.0
    # Alpha shock should be non-zero when enabled with alpha changes
    assert components_enabled.alpha_shock != 0.0


def test_all_ablation_flags_combined():
    """All ablation flags can be used together."""
    seed_info = SeedInfo(
        stage=SeedStage.BLENDING.value,
        improvement_since_stage_start=0.01,
        total_improvement=0.05,
        epochs_in_stage=3,
        seed_params=1000,
        previous_stage=SeedStage.TRAINING.value,
        previous_epochs_in_stage=2,
        seed_age_epochs=5,
    )

    config = ContributionRewardConfig(
        disable_pbrs=True,
        disable_terminal_reward=True,
        disable_anti_gaming=True,
    )

    reward, components = compute_contribution_reward(
        action=LifecycleOp.WAIT,
        seed_contribution=2.0,
        val_acc=60.0,
        seed_info=seed_info,
        epoch=25,  # terminal
        max_epochs=25,
        config=config,
        return_components=True,
        acc_at_germination=59.0,
        alpha_delta_sq_sum=0.5,
        num_contributing_fossilized=1,
    )

    # All ablated components should be zero
    assert components.pbrs_bonus == 0.0
    assert components.terminal_bonus == 0.0
    assert components.ratio_penalty == 0.0
    assert components.alpha_shock == 0.0
