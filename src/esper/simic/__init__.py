"""Simic - RL Training Infrastructure for Tamiyo

This package contains the reinforcement learning infrastructure for training
the Tamiyo seed lifecycle controller:

- buffers: Trajectory buffers
- normalization: Observation normalization
- action_masks: Masked action distributions
- rewards: Reward computation
- features: Feature extraction (hot path)
- ppo: PPO agent
- training: Training loops
- vectorized: Multi-GPU training
- debug_telemetry: Per-layer gradient debugging
- anomaly_detector: Phase-dependent anomaly detection
"""

# Normalization
from esper.simic.normalization import RunningMeanStd

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
    TaskConfig,
    normalize_observation,
)

# Action Masks
from esper.simic.action_masks import (
    MaskedCategorical,
    InvalidStateMachineError,
)

# Telemetry
from esper.simic.telemetry_config import (
    TelemetryLevel,
    TelemetryConfig,
)
from esper.simic.reward_telemetry import (
    RewardComponentsTelemetry,
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
#   from esper.simic.vectorized import train_ppo_vectorized
#   from esper.simic.training import train_heuristic

__all__ = [
    # Normalization
    "RunningMeanStd",

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
    "TaskConfig",
    "normalize_observation",

    # Action Masks
    "MaskedCategorical",
    "InvalidStateMachineError",

    # Telemetry
    "TelemetryLevel",
    "TelemetryConfig",
    "RewardComponentsTelemetry",
    "GradientHealthMetrics",
    "LayerGradientStats",
    "collect_per_layer_gradients",
    "NumericalStabilityReport",
    "check_numerical_stability",
    "RatioExplosionDiagnostic",
    "AnomalyDetector",
    "AnomalyReport",
]
