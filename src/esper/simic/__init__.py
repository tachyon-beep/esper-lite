"""Simic - RL Training Infrastructure for Tamiyo

This package contains the reinforcement learning infrastructure for training
the Tamiyo seed lifecycle controller:

- rewards: Reward computation for seed lifecycle control
- episodes: Episode data structures and collection
- features: Feature extraction (hot path)
- networks: Policy network architectures
- ppo: Online PPO training
- iql: Offline IQL/CQL training

Public API:
    from esper.simic.rewards import compute_shaped_reward, SeedInfo
    from esper.simic.episodes import Episode, TrainingSnapshot, EpisodeCollector
    from esper.simic.features import obs_to_base_features
    from esper.simic.networks import PolicyNetwork

Note: PPO and IQL are heavy modules - import them directly when needed:
    from esper.simic.ppo import PPOAgent
    from esper.simic.iql import IQL
"""

# Core data structures
from esper.simic.episodes import (
    TrainingSnapshot,
    ActionTaken,
    StepOutcome,
    DecisionPoint,
    Episode,
    EpisodeCollector,
    DatasetManager,
    snapshot_from_signals,
    action_from_decision,
)

# Rewards
from esper.simic.rewards import (
    RewardConfig,
    SeedInfo,
    compute_shaped_reward,
    compute_potential,
    compute_pbrs_bonus,
    get_intervention_cost,
    INTERVENTION_COSTS,
    STAGE_TRAINING,
    STAGE_BLENDING,
    STAGE_FOSSILIZED,
)

# Features (hot path)
from esper.simic.features import (
    safe,
    obs_to_base_features,
    telemetry_to_features,
)

# Networks
from esper.simic.networks import (
    PolicyNetwork,
    print_confusion_matrix,
)

# NOTE: We don't import ppo or iql here because they're heavy.
# Import them directly when needed:
#   from esper.simic.ppo import PPOAgent
#   from esper.simic.iql import IQL

__all__ = [
    # Episodes
    "TrainingSnapshot",
    "ActionTaken",
    "StepOutcome",
    "DecisionPoint",
    "Episode",
    "EpisodeCollector",
    "DatasetManager",
    "snapshot_from_signals",
    "action_from_decision",

    # Rewards
    "RewardConfig",
    "SeedInfo",
    "compute_shaped_reward",
    "compute_potential",
    "compute_pbrs_bonus",
    "get_intervention_cost",
    "INTERVENTION_COSTS",
    "STAGE_TRAINING",
    "STAGE_BLENDING",
    "STAGE_FOSSILIZED",

    # Features
    "safe",
    "obs_to_base_features",
    "telemetry_to_features",

    # Networks
    "PolicyNetwork",
    "print_confusion_matrix",
]
