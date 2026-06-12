"""Telemetry subsystem for simic training.

This package contains:
- emitters.py: Pure telemetry emission functions
- telemetry_config.py: Configuration for telemetry levels
- debug_telemetry.py: Per-layer gradient statistics (expensive, debug-only)
- gradient_collector.py: Async gradient collection for seed telemetry
- anomaly_detector.py: Phase-dependent training anomaly detection
"""

from typing import TYPE_CHECKING, Any
import importlib

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # Config
    "TelemetryLevel": ("esper.simic.telemetry.telemetry_config", "TelemetryLevel"),
    "TelemetryConfig": ("esper.simic.telemetry.telemetry_config", "TelemetryConfig"),
    # Debug telemetry
    "LayerGradientStats": ("esper.simic.telemetry.debug_telemetry", "LayerGradientStats"),
    "collect_per_layer_gradients": ("esper.simic.telemetry.debug_telemetry", "collect_per_layer_gradients"),
    "NumericalStabilityReport": ("esper.simic.telemetry.debug_telemetry", "NumericalStabilityReport"),
    "check_numerical_stability": ("esper.simic.telemetry.debug_telemetry", "check_numerical_stability"),
    "RatioExplosionDiagnostic": ("esper.simic.telemetry.debug_telemetry", "RatioExplosionDiagnostic"),
    # Gradient collection
    "GradientHealthMetrics": ("esper.simic.telemetry.gradient_collector", "GradientHealthMetrics"),
    "DualGradientStats": ("esper.simic.telemetry.gradient_collector", "DualGradientStats"),
    "SeedGradientCollector": ("esper.simic.telemetry.gradient_collector", "SeedGradientCollector"),
    "collect_dual_gradients_async": ("esper.simic.telemetry.gradient_collector", "collect_dual_gradients_async"),
    "collect_host_gradients_async": ("esper.simic.telemetry.gradient_collector", "collect_host_gradients_async"),
    "collect_seed_gradients_only_async": ("esper.simic.telemetry.gradient_collector", "collect_seed_gradients_only_async"),
    "materialize_dual_grad_stats": ("esper.simic.telemetry.gradient_collector", "materialize_dual_grad_stats"),
    "materialize_grad_stats": ("esper.simic.telemetry.gradient_collector", "materialize_grad_stats"),
    "collect_seed_gradients": ("esper.simic.telemetry.gradient_collector", "collect_seed_gradients"),
    "collect_seed_gradients_async": ("esper.simic.telemetry.gradient_collector", "collect_seed_gradients_async"),
    # Anomaly detection
    "AnomalyDetector": ("esper.simic.telemetry.anomaly_detector", "AnomalyDetector"),
    "AnomalyReport": ("esper.simic.telemetry.anomaly_detector", "AnomalyReport"),
    # LSTM health monitoring (P4-8)
    "LSTMHealthMetrics": ("esper.simic.telemetry.lstm_health", "LSTMHealthMetrics"),
    "compute_lstm_health": ("esper.simic.telemetry.lstm_health", "compute_lstm_health"),
    # Gradient EMA drift detection (P4-9)
    "GradientEMATracker": ("esper.simic.telemetry.gradient_ema", "GradientEMATracker"),
    # torch.profiler integration (P4-5)
    "training_profiler": ("esper.simic.telemetry.profiler", "training_profiler"),
    # Telemetry emitters
    "emit_with_env_context": ("esper.simic.telemetry.emitters", "emit_with_env_context"),
    "emit_batch_completed": ("esper.simic.telemetry.emitters", "emit_batch_completed"),
    "emit_last_action": ("esper.simic.telemetry.emitters", "emit_last_action"),
    "compute_grad_norm_surrogate": ("esper.simic.telemetry.emitters", "compute_grad_norm_surrogate"),
    "aggregate_layer_gradient_health": ("esper.simic.telemetry.emitters", "aggregate_layer_gradient_health"),
    "emit_ppo_update_event": ("esper.simic.telemetry.emitters", "emit_ppo_update_event"),
    "emit_action_distribution": ("esper.simic.telemetry.emitters", "emit_action_distribution"),
    "emit_throughput": ("esper.simic.telemetry.emitters", "emit_throughput"),
    "emit_reward_summary": ("esper.simic.telemetry.emitters", "emit_reward_summary"),
    "emit_mask_hit_rates": ("esper.simic.telemetry.emitters", "emit_mask_hit_rates"),
    "check_performance_degradation": ("esper.simic.telemetry.emitters", "check_performance_degradation"),
    "apply_slot_telemetry": ("esper.simic.telemetry.emitters", "apply_slot_telemetry"),
    # Value function metrics (TELE-220 to TELE-228)
    "compute_value_function_metrics": ("esper.leyline.value_metrics", "compute_value_function_metrics"),
    "ValueFunctionMetricsDict": ("esper.leyline.value_metrics", "ValueFunctionMetricsDict"),
    # Observation statistics (TELE-OBS)
    "ObservationStatsTelemetry": ("esper.simic.telemetry.observation_stats", "ObservationStatsTelemetry"),
    "compute_observation_stats": ("esper.simic.telemetry.observation_stats", "compute_observation_stats"),
}

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
    # Value function metrics (TELE-220 to TELE-228)
    "compute_value_function_metrics",
    "ValueFunctionMetricsDict",
    # Observation statistics (TELE-OBS)
    "ObservationStatsTelemetry",
    "compute_observation_stats",
]

if TYPE_CHECKING:
    from esper.leyline.value_metrics import (
        ValueFunctionMetricsDict as ValueFunctionMetricsDict,
        compute_value_function_metrics as compute_value_function_metrics,
    )
    from esper.simic.telemetry.anomaly_detector import (
        AnomalyDetector as AnomalyDetector,
        AnomalyReport as AnomalyReport,
    )
    from esper.simic.telemetry.debug_telemetry import (
        LayerGradientStats as LayerGradientStats,
        NumericalStabilityReport as NumericalStabilityReport,
        RatioExplosionDiagnostic as RatioExplosionDiagnostic,
        check_numerical_stability as check_numerical_stability,
        collect_per_layer_gradients as collect_per_layer_gradients,
    )
    from esper.simic.telemetry.emitters import (
        aggregate_layer_gradient_health as aggregate_layer_gradient_health,
        apply_slot_telemetry as apply_slot_telemetry,
        check_performance_degradation as check_performance_degradation,
        compute_grad_norm_surrogate as compute_grad_norm_surrogate,
        emit_action_distribution as emit_action_distribution,
        emit_batch_completed as emit_batch_completed,
        emit_last_action as emit_last_action,
        emit_mask_hit_rates as emit_mask_hit_rates,
        emit_ppo_update_event as emit_ppo_update_event,
        emit_reward_summary as emit_reward_summary,
        emit_throughput as emit_throughput,
        emit_with_env_context as emit_with_env_context,
    )
    from esper.simic.telemetry.gradient_collector import (
        DualGradientStats as DualGradientStats,
        GradientHealthMetrics as GradientHealthMetrics,
        SeedGradientCollector as SeedGradientCollector,
        collect_dual_gradients_async as collect_dual_gradients_async,
        collect_host_gradients_async as collect_host_gradients_async,
        collect_seed_gradients as collect_seed_gradients,
        collect_seed_gradients_async as collect_seed_gradients_async,
        collect_seed_gradients_only_async as collect_seed_gradients_only_async,
        materialize_dual_grad_stats as materialize_dual_grad_stats,
        materialize_grad_stats as materialize_grad_stats,
    )
    from esper.simic.telemetry.gradient_ema import GradientEMATracker as GradientEMATracker
    from esper.simic.telemetry.lstm_health import (
        LSTMHealthMetrics as LSTMHealthMetrics,
        compute_lstm_health as compute_lstm_health,
    )
    from esper.simic.telemetry.observation_stats import (
        ObservationStatsTelemetry as ObservationStatsTelemetry,
        compute_observation_stats as compute_observation_stats,
    )
    from esper.simic.telemetry.profiler import training_profiler as training_profiler
    from esper.simic.telemetry.telemetry_config import (
        TelemetryConfig as TelemetryConfig,
        TelemetryLevel as TelemetryLevel,
    )


def __getattr__(name: str) -> Any:
    """Lazy import telemetry exports and cache them on first access."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path)
        value = module.__dict__[attr_name]
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Expose the telemetry public API without importing heavy modules."""
    return sorted(set(globals().keys()) | set(__all__))
