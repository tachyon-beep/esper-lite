"""Telemetry subsystem for simic training.

This package contains:
- emitters.py: Pure telemetry emission functions
- telemetry_config.py: Configuration for telemetry levels
- debug_telemetry.py: Per-layer gradient statistics (expensive, debug-only)
- gradient_collector.py: Async gradient collection for seed telemetry
- anomaly_detector.py: Phase-dependent training anomaly detection
"""

# Telemetry configuration
from .telemetry_config import (
    TelemetryLevel,
    TelemetryConfig,
)

# Debug telemetry (per-layer gradients)
from .debug_telemetry import (
    LayerGradientStats,
    collect_per_layer_gradients,
    NumericalStabilityReport,
    check_numerical_stability,
    RatioExplosionDiagnostic,
)

# Gradient collection
from .gradient_collector import (
    GradientHealthMetrics,
    DualGradientStats,
    SeedGradientCollector,
    collect_dual_gradients_async,
    collect_host_gradients_async,
    collect_seed_gradients_only_async,
    materialize_dual_grad_stats,
    materialize_grad_stats,
    collect_seed_gradients,
    collect_seed_gradients_async,
)

# Anomaly detection
from .anomaly_detector import (
    AnomalyDetector,
    AnomalyReport,
)

# LSTM health monitoring (P4-8)
from .lstm_health import (
    LSTMHealthMetrics,
    compute_lstm_health,
)

# Gradient EMA drift detection (P4-9)
from .gradient_ema import GradientEMATracker

# torch.profiler integration (P4-5)
from .profiler import training_profiler

# Telemetry emitters (pure functions)
from .emitters import (
    emit_with_env_context,
    emit_batch_completed,
    emit_last_action,
    compute_grad_norm_surrogate,
    aggregate_layer_gradient_health,
    emit_ppo_update_event,
    emit_action_distribution,
    emit_throughput,
    emit_reward_summary,
    emit_mask_hit_rates,
    check_performance_degradation,
    apply_slot_telemetry,
)

__all__ = [
    # Config
    "TelemetryLevel",
    "TelemetryConfig",
    # Debug telemetry
    "LayerGradientStats",
    "collect_per_layer_gradients",
    "NumericalStabilityReport",
    "check_numerical_stability",
    "RatioExplosionDiagnostic",
    # Gradient collection
    "GradientHealthMetrics",
    "DualGradientStats",
    "SeedGradientCollector",
    "collect_dual_gradients_async",
    "collect_host_gradients_async",
    "collect_seed_gradients_only_async",
    "materialize_dual_grad_stats",
    "materialize_grad_stats",
    "collect_seed_gradients",
    "collect_seed_gradients_async",
    # Anomaly detection
    "AnomalyDetector",
    "AnomalyReport",
    # LSTM health monitoring (P4-8)
    "LSTMHealthMetrics",
    "compute_lstm_health",
    # Gradient EMA drift detection (P4-9)
    "GradientEMATracker",
    # torch.profiler integration (P4-5)
    "training_profiler",
    # Emitters
    "emit_with_env_context",
    "emit_batch_completed",
    "emit_last_action",
    "compute_grad_norm_surrogate",
    "aggregate_layer_gradient_health",
    "emit_ppo_update_event",
    "emit_action_distribution",
    "emit_throughput",
    "emit_reward_summary",
    "emit_mask_hit_rates",
    "check_performance_degradation",
    "apply_slot_telemetry",
]
