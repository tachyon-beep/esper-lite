"""Simic - RL Training Infrastructure for Tamiyo

This package contains the reinforcement learning infrastructure for training
the Tamiyo seed lifecycle controller:

- buffers: Trajectory buffers
- normalization: Observation normalization
- networks: Policy network architectures
- rewards: Reward computation
- features: Feature extraction (hot path)
- episodes: Episode data structures
- ppo: PPO agent
- training: Training loops
- vectorized: Multi-GPU training
"""

# Actions
from esper.leyline import Action, SimicAction  # SimicAction is deprecated alias

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

# Buffers
from esper.simic.buffers import (
    RolloutStep,
    RolloutBuffer,
)

# Normalization
from esper.simic.normalization import RunningMeanStd

# Rewards
from esper.simic.rewards import (
    RewardConfig,
    LossRewardConfig,
    SeedInfo,
    compute_shaped_reward,
    compute_potential,
    compute_pbrs_bonus,
    compute_pbrs_stage_bonus,
    compute_loss_reward,
    compute_seed_potential,
    get_intervention_cost,
    STAGE_TRAINING,
    STAGE_BLENDING,
    STAGE_FOSSILIZED,
)

# Features (hot path)
from esper.simic.features import (
    safe,
    obs_to_base_features,
    TaskConfig,
    normalize_observation,
)

# Networks
from esper.simic.networks import (
    PolicyNetwork,
    print_confusion_matrix,
    ActorCritic,
    QNetwork,
    VNetwork,
)

# NOTE: Heavy modules imported on demand:
#   from esper.simic.ppo import PPOAgent
#   from esper.simic.training import train_ppo
#   from esper.simic.vectorized import train_ppo_vectorized

__all__ = [
    # Actions
    "Action",
    "SimicAction",  # deprecated alias

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

    # Buffers
    "RolloutStep",
    "RolloutBuffer",

    # Normalization
    "RunningMeanStd",

    # Rewards
    "RewardConfig",
    "LossRewardConfig",
    "SeedInfo",
    "compute_shaped_reward",
    "compute_potential",
    "compute_pbrs_bonus",
    "compute_pbrs_stage_bonus",
    "compute_loss_reward",
    "compute_seed_potential",
    "get_intervention_cost",
    "STAGE_TRAINING",
    "STAGE_BLENDING",
    "STAGE_FOSSILIZED",

    # Features
    "safe",
    "obs_to_base_features",
    "TaskConfig",
    "normalize_observation",

    # Networks
    "PolicyNetwork",
    "print_confusion_matrix",
    "ActorCritic",
    "QNetwork",
    "VNetwork",
]
