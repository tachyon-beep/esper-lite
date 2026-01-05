"""Utilities for SIMPLIFIED reward tests.

These helpers keep the test cases extremely explicit: each test focuses on a
single simplified component (PBRS, uniform intervention cost, terminal bonus)
without dragging in SHAPED/ESCROW-only machinery.
"""

from __future__ import annotations

from esper.leyline import MIN_PRUNE_AGE, LifecycleOp, SeedStage
from esper.simic.rewards import ContributionRewardConfig, RewardMode, SeedInfo


def simplified_config(
    *,
    pbrs_weight: float = 1.0,
    gamma: float = 0.9,
    epoch_progress_bonus: float = 0.2,
    max_progress_bonus: float = 2.0,
    disable_pbrs: bool = False,
    disable_terminal_reward: bool = True,
) -> ContributionRewardConfig:
    """Return a config with predictable PBRS and terminal toggles."""
    return ContributionRewardConfig(
        reward_mode=RewardMode.SIMPLIFIED,
        pbrs_weight=pbrs_weight,
        gamma=gamma,
        epoch_progress_bonus=epoch_progress_bonus,
        max_progress_bonus=max_progress_bonus,
        disable_pbrs=disable_pbrs,
        disable_terminal_reward=disable_terminal_reward,
    )


def seed_info(
    *,
    stage: SeedStage = SeedStage.TRAINING,
    epochs_in_stage: int = 1,
    previous_stage: SeedStage = SeedStage.GERMINATED,
    previous_epochs_in_stage: int = 0,
    improvement_since_stage_start: float = 0.0,
    total_improvement: float = 0.0,
    seed_params: int = 0,
    seed_age_epochs: int = MIN_PRUNE_AGE,
    interaction_sum: float = 0.0,
    boost_received: float = 0.0,
) -> SeedInfo:
    """Convenience constructor with safe defaults for simplified tests."""
    return SeedInfo(
        stage=stage.value,
        improvement_since_stage_start=improvement_since_stage_start,
        total_improvement=total_improvement,
        epochs_in_stage=epochs_in_stage,
        seed_params=seed_params,
        previous_stage=previous_stage.value,
        previous_epochs_in_stage=previous_epochs_in_stage,
        seed_age_epochs=seed_age_epochs,
        interaction_sum=interaction_sum,
        boost_received=boost_received,
    )


__all__ = [
    "LifecycleOp",
    "SeedStage",
    "RewardMode",
    "ContributionRewardConfig",
    "SeedInfo",
    "seed_info",
    "simplified_config",
]

