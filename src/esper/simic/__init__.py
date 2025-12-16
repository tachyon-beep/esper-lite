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
- parallel_env_state: Per-environment state container
- telemetry/: Telemetry emission helpers
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
)

# Action Masks
from esper.simic.action_masks import (
    MaskedCategorical,
    InvalidStateMachineError,
    build_slot_states,
    compute_action_masks,
    compute_batch_masks,
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
    DualGradientStats,
    SeedGradientCollector,
    collect_dual_gradients_async,
    materialize_dual_grad_stats,
    materialize_grad_stats,
    collect_seed_gradients,
    collect_seed_gradients_async,
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

# Parallel environment state
from esper.simic.parallel_env_state import ParallelEnvState

# Telemetry emitters (pure functions extracted from vectorized.py)
from esper.simic.telemetry import (
    emit_with_env_context,
    emit_batch_completed,
    emit_ppo_update_event,
    check_performance_degradation,
    aggregate_layer_gradient_health,
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

    # Action Masks
    "MaskedCategorical",
    "InvalidStateMachineError",
    "build_slot_states",
    "compute_action_masks",
    "compute_batch_masks",

    # Telemetry
    "TelemetryLevel",
    "TelemetryConfig",
    "RewardComponentsTelemetry",
    "GradientHealthMetrics",
    "DualGradientStats",
    "SeedGradientCollector",
    "collect_dual_gradients_async",
    "materialize_dual_grad_stats",
    "materialize_grad_stats",
    "collect_seed_gradients",
    "collect_seed_gradients_async",
    "LayerGradientStats",
    "collect_per_layer_gradients",
    "NumericalStabilityReport",
    "check_numerical_stability",
    "RatioExplosionDiagnostic",
    "AnomalyDetector",
    "AnomalyReport",

    # Parallel environment state
    "ParallelEnvState",

    # Telemetry emitters
    "emit_with_env_context",
    "emit_batch_completed",
    "emit_ppo_update_event",
    "check_performance_degradation",
    "aggregate_layer_gradient_health",
]
