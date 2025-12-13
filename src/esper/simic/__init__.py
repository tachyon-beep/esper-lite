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

# Core data structures
from esper.simic.episodes import (
    TrainingSnapshot,
    ActionTaken,
    StepOutcome,
    DecisionPoint,
    Episode,
    DatasetManager,
)

from esper.simic.prioritized_buffer import (
    SumTree,
    PrioritizedReplayBuffer,
)

# Normalization
from esper.simic.normalization import RunningMeanStd

# Curriculum
from esper.simic.curriculum import BlueprintCurriculum, CurriculumStats

# Rewards
from esper.simic.rewards import (
    LossRewardConfig,
    ContributionRewardConfig,
    SeedInfo,
    compute_contribution_reward,
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

# Telemetry
from esper.simic.telemetry_config import (
    TelemetryLevel,
    TelemetryConfig,
)
from esper.simic.ppo_telemetry import (
    PPOHealthTelemetry,
    ValueFunctionTelemetry,
)
from esper.simic.reward_telemetry import (
    RewardComponentsTelemetry,
)
from esper.simic.memory_telemetry import (
    MemoryMetrics,
    collect_memory_metrics,
)
from esper.simic.gradient_collector import (
    GradientHealthMetrics,
)
from esper.simic.debug_telemetry import (
    LayerGradientStats,
    collect_per_layer_gradients,
    NumericalStabilityReport,
    check_numerical_stability,
    RatioExplosionDiagnostic,
)
from esper.simic.anomaly_detector import (
    AnomalyDetector,
    AnomalyReport,
)

# NOTE: Heavy modules imported on demand:
#   from esper.simic.ppo import PPOAgent
#   from esper.simic.training import train_ppo
#   from esper.simic.vectorized import train_ppo_vectorized

__all__ = [
    # Episodes
    "TrainingSnapshot",
    "ActionTaken",
    "StepOutcome",
    "DecisionPoint",
    "Episode",
    "DatasetManager",

    # Buffers
    "SumTree",
    "PrioritizedReplayBuffer",

    # Normalization
    "RunningMeanStd",

    # Curriculum
    "BlueprintCurriculum",
    "CurriculumStats",

    # Rewards
    "LossRewardConfig",
    "ContributionRewardConfig",
    "SeedInfo",
    "compute_contribution_reward",
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

    # Telemetry
    "TelemetryLevel",
    "TelemetryConfig",
    "PPOHealthTelemetry",
    "ValueFunctionTelemetry",
    "RewardComponentsTelemetry",
    "MemoryMetrics",
    "collect_memory_metrics",
    "GradientHealthMetrics",
    "LayerGradientStats",
    "collect_per_layer_gradients",
    "NumericalStabilityReport",
    "check_numerical_stability",
    "RatioExplosionDiagnostic",
    "AnomalyDetector",
    "AnomalyReport",
]
