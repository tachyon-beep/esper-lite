"""Golden-value tests for reward outputs."""

from __future__ import annotations

import pytest

from esper.leyline import LifecycleOp, LossRewardConfig, SeedStage
from esper.simic.rewards import (
    ContributionRewardConfig,
    ContributionRewardInputs,
    LossRewardInputs,
    RewardMode,
    SeedInfo,
    compute_loss_reward,
    compute_reward,
)


def test_contribution_reward_golden_simplified_pbrs() -> None:
    """Simplified reward: PBRS + action cost stays stable."""
    config = ContributionRewardConfig(reward_mode=RewardMode.SIMPLIFIED)
    seed_info = SeedInfo(
        stage=SeedStage.TRAINING.value,
        improvement_since_stage_start=0.0,
        total_improvement=0.0,
        epochs_in_stage=2,
        seed_params=10_000,
        previous_stage=SeedStage.GERMINATED.value,
        previous_epochs_in_stage=1,
        seed_age_epochs=5,
    )

    reward = compute_reward(
        ContributionRewardInputs(
            action=LifecycleOp.PRUNE,
            seed_contribution=None,
            val_acc=70.0,
            seed_info=seed_info,
            epoch=5,
            max_epochs=10,
            total_params=110_000,
            host_params=100_000,
            acc_at_germination=65.0,
            acc_delta=0.1,
            committed_val_acc=70.0,
            fossilized_seed_params=0,
            num_contributing_fossilized=0,
            config=config,
        )
    )

    assert reward == pytest.approx(0.07610000000000011, abs=1e-6)


def test_contribution_reward_golden_basic_mode() -> None:
    """Basic reward: accuracy improvement minus rent stays stable."""
    config = ContributionRewardConfig(
        reward_mode=RewardMode.BASIC,
        basic_acc_delta_weight=5.0,
        param_budget=500_000,
        param_penalty_weight=0.1,
    )

    reward = compute_reward(
        ContributionRewardInputs(
            action=LifecycleOp.WAIT,
            seed_contribution=None,
            val_acc=70.0,
            seed_info=None,
            epoch=1,
            max_epochs=10,
            total_params=125_000,
            host_params=100_000,
            acc_at_germination=None,
            acc_delta=2.0,
            committed_val_acc=70.0,
            fossilized_seed_params=0,
            effective_seed_params=25_000,
            config=config,
        )
    )

    assert reward == pytest.approx(0.095, abs=1e-6)


def test_loss_reward_golden_terminal() -> None:
    """Loss reward: normalized loss delta + terminal bonus stays stable."""
    config = LossRewardConfig.default()
    reward = compute_loss_reward(
        LossRewardInputs(
            action=LifecycleOp.WAIT,
            loss_delta=-0.05,
            val_loss=1.5,
            seed_info=None,
            epoch=10,
            max_epochs=10,
            total_params=120,
            host_params=100,
            config=config,
        )
    )

    assert reward == pytest.approx(2.890883922160302, abs=1e-6)
