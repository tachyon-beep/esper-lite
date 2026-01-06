"""Loss-primary reward computation."""

from __future__ import annotations

import math

from esper.leyline import DEFAULT_GAMMA, LossRewardConfig, SeedStage
from .shaping import STAGE_POTENTIALS
from .types import LossRewardInputs, SeedInfo


def compute_pbrs_stage_bonus(
    seed_info: SeedInfo,
    config: LossRewardConfig,
    gamma: float = DEFAULT_GAMMA,
) -> float:
    """PBRS-compatible stage bonus using potential function."""
    previous_stage = seed_info.previous_stage

    current_potential = STAGE_POTENTIALS[SeedStage(seed_info.stage)]
    previous_potential = STAGE_POTENTIALS[SeedStage(previous_stage)]

    return config.stage_potential_weight * (
        gamma * current_potential - previous_potential
    )


def compute_loss_reward(
    inputs: LossRewardInputs,
) -> float:
    """Compute loss-primary reward for seed lifecycle control."""
    config = inputs.config
    if config is None:
        config = LossRewardConfig.default()

    reward = 0.0

    normalized_delta = inputs.loss_delta / config.typical_loss_delta_std
    clipped = max(-config.max_loss_delta, min(normalized_delta, config.max_loss_delta))
    if clipped > 0:
        clipped *= config.regression_penalty_scale
    reward += (-clipped) * config.loss_delta_weight

    if inputs.host_params > 0 and inputs.total_params > inputs.host_params:
        in_grace = False
        if inputs.seed_info is not None:
            in_grace = inputs.seed_info.seed_age_epochs < config.grace_epochs
        if not in_grace:
            growth_ratio = (
                inputs.total_params - inputs.host_params
            ) / inputs.host_params
            scaled_cost = math.log(1.0 + growth_ratio)
            rent_penalty = config.compute_rent_weight * scaled_cost
            rent_penalty = min(rent_penalty, config.max_rent_penalty)
            reward -= rent_penalty

    if inputs.seed_info is not None:
        reward += compute_pbrs_stage_bonus(inputs.seed_info, config)

    if inputs.epoch == inputs.max_epochs:
        improvement = config.baseline_loss - inputs.val_loss
        achievable_range = config.achievable_range or 1.0
        normalized = max(0.0, min(improvement / achievable_range, 1.0))
        reward += normalized * config.terminal_loss_weight

    return reward


__all__ = [
    "compute_pbrs_stage_bonus",
    "compute_loss_reward",
]
