"""Simic - RL Training Infrastructure for Tamiyo

This package contains the reinforcement learning infrastructure for training
the Tamiyo seed lifecycle controller:

Core Modules:
- ppo: PPO agent and policy learning
- network: Factored recurrent actor-critic network
- rollout_buffer: Trajectory buffer management
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

NOTE: This module uses PEP 562 lazy imports. Heavy modules (tamiyo policy features,
telemetry with torch, training loops) are only loaded when accessed.
"""

__all__ = [
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
    "RewardComponentsTelemetry",
    # Features (from tamiyo.policy - HEAVY)
    "safe",
    "TaskConfig",
    # Action Masks (from tamiyo.policy - HEAVY)
    "MaskedCategorical",
    "InvalidStateMachineError",
    "build_slot_states",
    "compute_action_masks",
    "compute_batch_masks",
    # Telemetry
    "TelemetryLevel",
    "TelemetryConfig",
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
    "emit_with_env_context",
    "emit_batch_completed",
    "emit_ppo_update_event",
    "check_performance_degradation",
    "aggregate_layer_gradient_health",
    # Parallel environment state
    "ParallelEnvState",
]


from typing import Any


def __getattr__(name: str) -> Any:
    """Lazy import using PEP 562.

    Heavy modules (tamiyo policy features/masks with torch, telemetry with torch,
    training loops) are only loaded when accessed, not at package import time.
    """
    # Features (HEAVY - from tamiyo.policy which loads torch)
    if name in ("safe", "TaskConfig"):
        from esper.tamiyo.policy.features import safe, TaskConfig
        mapping: dict[str, Any] = {
            "safe": safe,
            "TaskConfig": TaskConfig,
        }
        return mapping[name]

    # Action Masks (HEAVY - from tamiyo.policy which loads torch)
    if name in ("MaskedCategorical", "InvalidStateMachineError", "build_slot_states",
                "compute_action_masks", "compute_batch_masks"):
        from esper.tamiyo.policy.action_masks import (
            MaskedCategorical,
            InvalidStateMachineError,
            build_slot_states,
            compute_action_masks,
            compute_batch_masks,
        )
        return {"MaskedCategorical": MaskedCategorical, "InvalidStateMachineError": InvalidStateMachineError,
                "build_slot_states": build_slot_states, "compute_action_masks": compute_action_masks,
                "compute_batch_masks": compute_batch_masks}[name]

    # Rewards (lightweight)
    if name in ("LossRewardConfig", "ContributionRewardConfig", "SeedInfo",
                "compute_contribution_reward", "compute_potential", "compute_pbrs_bonus",
                "compute_pbrs_stage_bonus", "compute_loss_reward", "compute_seed_potential",
                "get_intervention_cost", "STAGE_TRAINING", "STAGE_BLENDING",
                "STAGE_FOSSILIZED", "RewardComponentsTelemetry"):
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
            RewardComponentsTelemetry,
        )
        return {"LossRewardConfig": LossRewardConfig, "ContributionRewardConfig": ContributionRewardConfig,
                "SeedInfo": SeedInfo, "compute_contribution_reward": compute_contribution_reward,
                "compute_potential": compute_potential, "compute_pbrs_bonus": compute_pbrs_bonus,
                "compute_pbrs_stage_bonus": compute_pbrs_stage_bonus, "compute_loss_reward": compute_loss_reward,
                "compute_seed_potential": compute_seed_potential, "get_intervention_cost": get_intervention_cost,
                "STAGE_TRAINING": STAGE_TRAINING, "STAGE_BLENDING": STAGE_BLENDING,
                "STAGE_FOSSILIZED": STAGE_FOSSILIZED, "RewardComponentsTelemetry": RewardComponentsTelemetry}[name]

    # Telemetry (HEAVY - uses torch for gradient collection)
    if name in ("TelemetryLevel", "TelemetryConfig", "GradientHealthMetrics",
                "DualGradientStats", "SeedGradientCollector", "collect_dual_gradients_async",
                "materialize_dual_grad_stats", "materialize_grad_stats",
                "collect_seed_gradients", "collect_seed_gradients_async",
                "LayerGradientStats", "collect_per_layer_gradients",
                "NumericalStabilityReport", "check_numerical_stability",
                "RatioExplosionDiagnostic", "AnomalyDetector", "AnomalyReport",
                "emit_with_env_context", "emit_batch_completed", "emit_ppo_update_event",
                "check_performance_degradation", "aggregate_layer_gradient_health"):
        from esper.simic.telemetry import (
            TelemetryLevel,
            TelemetryConfig,
            GradientHealthMetrics,
            DualGradientStats,
            SeedGradientCollector,
            collect_dual_gradients_async,
            materialize_dual_grad_stats,
            materialize_grad_stats,
            collect_seed_gradients,
            collect_seed_gradients_async,
            LayerGradientStats,
            collect_per_layer_gradients,
            NumericalStabilityReport,
            check_numerical_stability,
            RatioExplosionDiagnostic,
            AnomalyDetector,
            AnomalyReport,
            emit_with_env_context,
            emit_batch_completed,
            emit_ppo_update_event,
            check_performance_degradation,
            aggregate_layer_gradient_health,
        )
        return {"TelemetryLevel": TelemetryLevel, "TelemetryConfig": TelemetryConfig,
                "GradientHealthMetrics": GradientHealthMetrics, "DualGradientStats": DualGradientStats,
                "SeedGradientCollector": SeedGradientCollector, "collect_dual_gradients_async": collect_dual_gradients_async,
                "materialize_dual_grad_stats": materialize_dual_grad_stats, "materialize_grad_stats": materialize_grad_stats,
                "collect_seed_gradients": collect_seed_gradients, "collect_seed_gradients_async": collect_seed_gradients_async,
                "LayerGradientStats": LayerGradientStats, "collect_per_layer_gradients": collect_per_layer_gradients,
                "NumericalStabilityReport": NumericalStabilityReport, "check_numerical_stability": check_numerical_stability,
                "RatioExplosionDiagnostic": RatioExplosionDiagnostic, "AnomalyDetector": AnomalyDetector,
                "AnomalyReport": AnomalyReport, "emit_with_env_context": emit_with_env_context,
                "emit_batch_completed": emit_batch_completed, "emit_ppo_update_event": emit_ppo_update_event,
                "check_performance_degradation": check_performance_degradation,
                "aggregate_layer_gradient_health": aggregate_layer_gradient_health}[name]

    # Training (lightweight - ParallelEnvState is just a dataclass container)
    if name == "ParallelEnvState":
        from esper.simic.training import ParallelEnvState
        return ParallelEnvState

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
