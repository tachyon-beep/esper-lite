"""Telemetry subsystem for simic training.

This package contains telemetry emission helpers and configuration.
"""

from esper.simic.telemetry.emitters import (
    emit_with_env_context,
    emit_batch_completed,
    emit_last_action,
    compute_grad_norm_surrogate,
    aggregate_layer_gradient_health,
    emit_ppo_update_event,
    emit_action_distribution,
    emit_cf_unavailable,
    emit_throughput,
    emit_reward_summary,
    emit_mask_hit_rates,
    check_performance_degradation,
    apply_slot_telemetry,
)

__all__ = [
    "emit_with_env_context",
    "emit_batch_completed",
    "emit_last_action",
    "compute_grad_norm_surrogate",
    "aggregate_layer_gradient_health",
    "emit_ppo_update_event",
    "emit_action_distribution",
    "emit_cf_unavailable",
    "emit_throughput",
    "emit_reward_summary",
    "emit_mask_hit_rates",
    "check_performance_degradation",
    "apply_slot_telemetry",
]
