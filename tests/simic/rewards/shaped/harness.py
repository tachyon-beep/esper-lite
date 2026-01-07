"""Utilities for SHAPED reward tests.

These helpers keep the test cases extremely explicit: each test focuses on a
single shaped component (attribution, ratio penalty, holding rent, etc.) without
unrelated shaping terms.
"""

from __future__ import annotations

from dataclasses import replace

from esper.leyline import MIN_PRUNE_AGE, LifecycleOp, SeedStage
from esper.simic.rewards import ContributionRewardConfig, RewardMode, SeedInfo


def shaped_config(
    *,
    contribution_weight: float = 1.0,
    proxy_confidence_factor: float = 0.3,
    disable_pbrs: bool = True,
    disable_terminal_reward: bool = True,
    disable_anti_gaming: bool = True,
    rent_weight: float = 0.0,
    rent_host_params_floor: int = 1,
    alpha_shock_coef: float = 0.0,
    pbrs_weight: float = 0.0,
    terminal_acc_weight: float = 0.0,
    fossilize_terminal_scale: float = 0.0,
    prune_good_seed_penalty: float = 0.0,
    prune_hurting_bonus: float = 0.0,
    prune_acceptable_bonus: float = 0.0,
    prune_cost: float = 0.0,
    fossilize_cost: float = 0.0,
    germinate_cost: float = 0.0,
    set_alpha_target_cost: float = 0.0,
) -> ContributionRewardConfig:
    """Return a config that isolates the SHAPED path.

    By default we:
    - disable PBRS and terminal bonus
    - disable anti-gaming (ratio penalty + alpha shock)
    - zero out intervention costs and prune shaping
    """
    return ContributionRewardConfig(
        reward_mode=RewardMode.SHAPED,
        contribution_weight=contribution_weight,
        proxy_confidence_factor=proxy_confidence_factor,
        disable_pbrs=disable_pbrs,
        disable_terminal_reward=disable_terminal_reward,
        disable_anti_gaming=disable_anti_gaming,
        rent_weight=rent_weight,
        rent_host_params_floor=rent_host_params_floor,
        alpha_shock_coef=alpha_shock_coef,
        pbrs_weight=pbrs_weight,
        terminal_acc_weight=terminal_acc_weight,
        fossilize_terminal_scale=fossilize_terminal_scale,
        prune_good_seed_penalty=prune_good_seed_penalty,
        prune_hurting_bonus=prune_hurting_bonus,
        prune_acceptable_bonus=prune_acceptable_bonus,
        prune_cost=prune_cost,
        fossilize_cost=fossilize_cost,
        germinate_cost=germinate_cost,
        set_alpha_target_cost=set_alpha_target_cost,
        # Remove action shaping we are not testing in SHAPED specs.
        germinate_with_seed_penalty=0.0,
        invalid_fossilize_penalty=0.0,
        prune_fossilized_penalty=0.0,
        fossilize_base_bonus=0.0,
        fossilize_contribution_scale=0.0,
        fossilize_noncontributing_penalty=0.0,
        early_prune_threshold=0,
        early_prune_penalty=0.0,
        # D2 capacity economics - disable for isolated SHAPED tests
        first_germinate_bonus=0.0,
    )


def seed_info(
    *,
    stage: SeedStage = SeedStage.TRAINING,
    total_improvement: float = 0.0,
    improvement_since_stage_start: float = 0.0,
    epochs_in_stage: int = 1,
    seed_age_epochs: int = MIN_PRUNE_AGE,
    previous_stage: SeedStage = SeedStage.GERMINATED,
    previous_epochs_in_stage: int = 0,
    seed_params: int = 0,
    interaction_sum: float = 0.0,
    boost_received: float = 0.0,
) -> SeedInfo:
    """Convenience constructor with safe defaults for shaped tests."""
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


def with_prune_good_seed_penalty(
    config: ContributionRewardConfig, *, prune_good_seed_penalty: float
) -> ContributionRewardConfig:
    """Return a copy of config with a different prune_good_seed_penalty."""
    return replace(config, prune_good_seed_penalty=prune_good_seed_penalty)


__all__ = [
    "LifecycleOp",
    "SeedStage",
    "RewardMode",
    "ContributionRewardConfig",
    "SeedInfo",
    "seed_info",
    "shaped_config",
    "with_prune_good_seed_penalty",
]

