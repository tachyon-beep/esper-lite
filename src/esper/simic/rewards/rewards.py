"""Reward Computation for Tamiyo Seed Lifecycle Controller.

This module provides the public reward dispatch surface while keeping
implementation details in dedicated modules.
"""

from __future__ import annotations

from enum import Enum
from typing import cast

from esper.leyline import DEFAULT_GAMMA
from .contribution import (
    ContributionRewardConfig,
    RewardMode,
    compute_basic_reward,
    compute_contribution_reward,
    compute_minimal_reward,
    compute_scaffold_hindsight_credit,
    compute_simplified_reward,
    compute_sparse_reward,
    get_intervention_cost,
    INTERVENTION_COSTS,
    _check_reward_hacking,
    _check_ransomware_signature,
    _contribution_fossilize_shaping,
    _contribution_pbrs_bonus,
    _contribution_prune_shaping,
)
from .loss_primary import compute_loss_reward, compute_pbrs_stage_bonus
from .reward_telemetry import RewardComponentsTelemetry
from .shaping import (
    STAGE_POTENTIALS,
    compute_pbrs_bonus,
    compute_potential,
    compute_seed_potential,
)
from .types import (
    ContributionRewardInputs,
    LossRewardInputs,
    SeedInfo,
    STAGE_BLENDING,
    STAGE_FOSSILIZED,
    STAGE_GERMINATED,
    STAGE_HOLDING,
    STAGE_TRAINING,
)


class RewardFamily(Enum):
    """Top-level reward family selection."""

    CONTRIBUTION = "contribution"
    LOSS = "loss"


def compute_reward(
    inputs: ContributionRewardInputs,
) -> float | tuple[float, RewardComponentsTelemetry]:
    """Unified reward computation dispatcher."""
    config = inputs.config
    if config is None:
        config = ContributionRewardConfig()

    if config.reward_mode in (RewardMode.SHAPED, RewardMode.ESCROW):
        return compute_contribution_reward(
            action=inputs.action,
            seed_contribution=inputs.seed_contribution,
            val_acc=inputs.val_acc,
            seed_info=inputs.seed_info,
            epoch=inputs.epoch,
            max_epochs=inputs.max_epochs,
            total_params=inputs.total_params,
            host_params=inputs.host_params,
            config=config,
            acc_at_germination=inputs.acc_at_germination,
            acc_delta=inputs.acc_delta,
            return_components=inputs.return_components,
            num_fossilized_seeds=inputs.num_fossilized_seeds,
            num_contributing_fossilized=inputs.num_contributing_fossilized,
            slot_id=inputs.slot_id,
            seed_id=inputs.seed_id,
            effective_seed_params=inputs.effective_seed_params,
            alpha_delta_sq_sum=inputs.alpha_delta_sq_sum,
            stable_val_acc=inputs.stable_val_acc,
            escrow_credit_prev=inputs.escrow_credit_prev,
            # D2: Capacity Economics (slot saturation prevention)
            n_active_seeds=inputs.n_active_seeds,
            seeds_germinated_this_episode=inputs.seeds_germinated_this_episode,
        )

    if config.reward_mode == RewardMode.SPARSE:
        if inputs.committed_val_acc is None:
            raise ValueError("committed_val_acc is required for RewardMode.SPARSE")
        reward = compute_sparse_reward(
            committed_val_acc=inputs.committed_val_acc,
            fossilized_seed_params=inputs.fossilized_seed_params,
            epoch=inputs.epoch,
            max_epochs=inputs.max_epochs,
            config=config,
        )

    elif config.reward_mode == RewardMode.MINIMAL:
        seed_age = inputs.seed_info.seed_age_epochs if inputs.seed_info else None
        if inputs.committed_val_acc is None:
            raise ValueError("committed_val_acc is required for RewardMode.MINIMAL")
        reward = compute_minimal_reward(
            committed_val_acc=inputs.committed_val_acc,
            fossilized_seed_params=inputs.fossilized_seed_params,
            epoch=inputs.epoch,
            max_epochs=inputs.max_epochs,
            action=inputs.action,
            seed_age=seed_age,
            config=config,
        )

    elif config.reward_mode == RewardMode.BASIC:
        reward, rent_penalty, growth_ratio, pbrs_bonus, fossilize_bonus = compute_basic_reward(
            acc_delta=inputs.acc_delta,
            effective_seed_params=inputs.effective_seed_params,
            total_params=inputs.total_params,
            host_params=inputs.host_params,
            config=config,
            epoch=inputs.epoch,
            max_epochs=inputs.max_epochs,
            seed_info=inputs.seed_info,
            action=inputs.action,
            seed_contribution=inputs.seed_contribution,
        )
        if inputs.return_components:
            components = RewardComponentsTelemetry()
            components.total_reward = reward
            components.action_name = inputs.action.name
            components.epoch = inputs.epoch
            components.seed_stage = inputs.seed_info.stage if inputs.seed_info else None
            components.val_acc = inputs.val_acc
            components.base_acc_delta = inputs.acc_delta
            components.compute_rent = -rent_penalty
            components.growth_ratio = growth_ratio
            components.pbrs_bonus = pbrs_bonus
            components.fossilize_terminal_bonus = fossilize_bonus
            return reward, components

    elif config.reward_mode == RewardMode.SIMPLIFIED:
        reward = compute_simplified_reward(
            action=inputs.action,
            seed_info=inputs.seed_info,
            epoch=inputs.epoch,
            max_epochs=inputs.max_epochs,
            val_acc=inputs.val_acc,
            num_contributing_fossilized=inputs.num_contributing_fossilized,
            config=config,
        )

    else:
        raise ValueError(f"Unknown reward mode: {config.reward_mode}")

    if inputs.return_components:
        components = RewardComponentsTelemetry()
        components.total_reward = reward
        components.action_name = inputs.action.name
        components.epoch = inputs.epoch
        components.seed_stage = inputs.seed_info.stage if inputs.seed_info else None
        components.val_acc = inputs.val_acc
        components.base_acc_delta = inputs.acc_delta
        return reward, components

    return reward


def compute_reward_for_family(
    reward_family: RewardFamily,
    inputs: ContributionRewardInputs | LossRewardInputs,
) -> float:
    """Dispatch reward based on family (contribution vs loss-primary)."""
    if reward_family == RewardFamily.CONTRIBUTION:
        contribution_inputs = cast(ContributionRewardInputs, inputs)
        if contribution_inputs.return_components:
            raise ValueError(
                "compute_reward_for_family expects return_components=False"
            )
        return cast(float, compute_reward(contribution_inputs))
    if reward_family == RewardFamily.LOSS:
        loss_inputs = cast(LossRewardInputs, inputs)
        return compute_loss_reward(loss_inputs)
    raise ValueError(f"Unknown reward family: {reward_family}")


__all__ = [
    "ContributionRewardConfig",
    "RewardMode",
    "RewardFamily",
    "SeedInfo",
    "compute_reward",
    "compute_reward_for_family",
    "compute_contribution_reward",
    "compute_sparse_reward",
    "compute_minimal_reward",
    "compute_basic_reward",
    "compute_simplified_reward",
    "compute_loss_reward",
    "compute_scaffold_hindsight_credit",
    "compute_potential",
    "compute_pbrs_bonus",
    "compute_pbrs_stage_bonus",
    "compute_seed_potential",
    "get_intervention_cost",
    "INTERVENTION_COSTS",
    "DEFAULT_GAMMA",
    "STAGE_POTENTIALS",
    "STAGE_TRAINING",
    "STAGE_BLENDING",
    "STAGE_FOSSILIZED",
    "STAGE_HOLDING",
    "STAGE_GERMINATED",
    "_contribution_pbrs_bonus",
    "_contribution_prune_shaping",
    "_contribution_fossilize_shaping",
    "_check_reward_hacking",
    "_check_ransomware_signature",
]
