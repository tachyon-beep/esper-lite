"""Reward computation and telemetry for Tamiyo seed lifecycle training.

This subpackage contains:
- rewards.py: Dense reward computation with PBRS, contribution signals, penalties
- reward_telemetry.py: Per-component reward breakdown for debugging
"""

from .reward_telemetry import RewardComponentsTelemetry

from .rewards import (
    # Config classes
    LossRewardConfig,
    ContributionRewardConfig,
    RewardMode,
    RewardFamily,
    # Seed info
    SeedInfo,
    # Reward functions
    compute_reward,
    compute_reward_for_family,
    compute_contribution_reward,
    compute_sparse_reward,
    compute_minimal_reward,
    compute_simplified_reward,
    compute_loss_reward,
    compute_scaffold_hindsight_credit,
    # PBRS utilities
    compute_potential,
    compute_pbrs_bonus,
    compute_pbrs_stage_bonus,
    compute_seed_potential,
    # Intervention costs
    get_intervention_cost,
    INTERVENTION_COSTS,
    # Stage constants
    DEFAULT_GAMMA,
    STAGE_POTENTIALS,
    STAGE_TRAINING,
    STAGE_BLENDING,
    STAGE_FOSSILIZED,
    STAGE_HOLDING,
    STAGE_GERMINATED,
    # Internal helpers (exported for testing)
    _contribution_pbrs_bonus,
    _contribution_prune_shaping,
    _contribution_fossilize_shaping,
    _check_reward_hacking,
    _check_ransomware_signature,
)

__all__ = [
    # Telemetry
    "RewardComponentsTelemetry",
    # Config classes
    "LossRewardConfig",
    "ContributionRewardConfig",
    "RewardMode",
    "RewardFamily",
    # Seed info
    "SeedInfo",
    # Reward functions
    "compute_reward",
    "compute_reward_for_family",
    "compute_contribution_reward",
    "compute_sparse_reward",
    "compute_minimal_reward",
    "compute_simplified_reward",
    "compute_loss_reward",
    "compute_scaffold_hindsight_credit",
    # PBRS utilities
    "compute_potential",
    "compute_pbrs_bonus",
    "compute_pbrs_stage_bonus",
    "compute_seed_potential",
    # Intervention costs
    "get_intervention_cost",
    "INTERVENTION_COSTS",
    # Stage constants
    "DEFAULT_GAMMA",
    "STAGE_POTENTIALS",
    "STAGE_TRAINING",
    "STAGE_BLENDING",
    "STAGE_FOSSILIZED",
    "STAGE_HOLDING",
    "STAGE_GERMINATED",
    # Internal helpers (exported for testing)
    "_contribution_pbrs_bonus",
    "_contribution_prune_shaping",
    "_contribution_fossilize_shaping",
    "_check_reward_hacking",
    "_check_ransomware_signature",
]
