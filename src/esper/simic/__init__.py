"""Simic - RL Training Infrastructure for Tamiyo

This package contains the reinforcement learning infrastructure for training
the Tamiyo seed lifecycle controller:

Core Modules:
- ppo: PPO agent and policy learning
- tamiyo_network: Factored recurrent actor-critic network
- tamiyo_buffer: Trajectory buffer management
- advantages: Per-head advantage computation

Control:
- action_masks: Masked action distributions
- features: Feature extraction (hot path)
- normalization: Observation/reward preprocessing

Subpackages:
- rewards/: Reward computation (PBRS, contribution signals, penalties)
- telemetry/: Diagnostics (config, gradients, anomaly detection, emitters)

Training:
- training: Training loops
- vectorized: Multi-GPU PPO training
- parallel_env_state: Per-environment state container
- config: Hyperparameter configuration
"""

# Control (observation/action preprocessing)
from esper.simic.control import (
    RunningMeanStd,
    safe,
    TaskConfig,
    MaskedCategorical,
    InvalidStateMachineError,
    build_slot_states,
    compute_action_masks,
    compute_batch_masks,
)

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

# Reward telemetry (from rewards subpackage)
from esper.simic.rewards import (
    RewardComponentsTelemetry,
)

# Telemetry (from telemetry subpackage)
from esper.simic.telemetry import (
    # Config
    TelemetryLevel,
    TelemetryConfig,
    # Gradient collection
    GradientHealthMetrics,
    DualGradientStats,
    SeedGradientCollector,
    collect_dual_gradients_async,
    materialize_dual_grad_stats,
    materialize_grad_stats,
    collect_seed_gradients,
    collect_seed_gradients_async,
    # Debug telemetry
    LayerGradientStats,
    collect_per_layer_gradients,
    NumericalStabilityReport,
    check_numerical_stability,
    RatioExplosionDiagnostic,
    # Anomaly detection
    AnomalyDetector,
    AnomalyReport,
    # Emitters
    emit_with_env_context,
    emit_batch_completed,
    emit_ppo_update_event,
    check_performance_degradation,
    aggregate_layer_gradient_health,
)

# Parallel environment state
from esper.simic.parallel_env_state import ParallelEnvState

# NOTE: Heavy modules imported on demand:
#   from esper.simic.agent import PPOAgent
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
