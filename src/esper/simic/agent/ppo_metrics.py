"""PPO metrics aggregation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from esper.simic.telemetry.value_metrics import ValueFunctionMetricsDict

from .types import FinitenessGateFailure, PPOUpdateMetrics


@dataclass(slots=True)
class PPOUpdateMetricsResult:
    """Aggregated PPO update metrics plus success flag."""

    metrics: PPOUpdateMetrics
    update_performed: bool


@dataclass(slots=True)
class PPOUpdateMetricsBuilder:
    """Aggregate per-epoch PPO metrics into a single update payload."""

    metrics: dict[str, list[Any]]
    finiteness_failures: list[FinitenessGateFailure]
    epochs_completed: int
    head_entropies: dict[str, list[float]]
    head_grad_norms: dict[str, list[float]]
    head_nan_detected: dict[str, bool]
    head_inf_detected: dict[str, bool]
    lstm_health_history: dict[str, list[float | bool]]
    log_prob_min_across_epochs: float
    log_prob_max_across_epochs: float
    head_ratio_max_across_epochs: dict[str, float]
    joint_ratio_max_across_epochs: float
    value_func_metrics: ValueFunctionMetricsDict
    cuda_memory_metrics: dict[str, float]
    head_names: tuple[str, ...]

    def finalize(self) -> PPOUpdateMetricsResult:
        aggregated_result: PPOUpdateMetrics = {}

        if self.epochs_completed == 0:
            aggregated_result["ppo_update_performed"] = False
            aggregated_result["finiteness_gate_skip_count"] = len(self.finiteness_failures)
            aggregated_result["ratio_max"] = float("nan")
            aggregated_result["ratio_min"] = float("nan")
            aggregated_result["policy_loss"] = float("nan")
            aggregated_result["value_loss"] = float("nan")
            aggregated_result["entropy"] = float("nan")
            aggregated_result["approx_kl"] = float("nan")
            aggregated_result["clip_fraction"] = float("nan")
            aggregated_result["explained_variance"] = float("nan")
            aggregated_result["pre_clip_grad_norm"] = float("nan")
            if self.finiteness_failures:
                aggregated_result["finiteness_gate_failures"] = self.finiteness_failures
            return PPOUpdateMetricsResult(metrics=aggregated_result, update_performed=False)

        aggregated_result["ppo_update_performed"] = True
        aggregated_result["finiteness_gate_skip_count"] = len(self.finiteness_failures)

        for k, v in self.metrics.items():
            if not v:
                aggregated_result[k] = 0.0  # type: ignore[literal-required]
                continue

            first = v[0]
            if k == "finiteness_gate_failures":
                aggregated_result[k] = v  # type: ignore[literal-required]
            elif k == "early_stop_epoch":
                aggregated_result[k] = first  # type: ignore[literal-required]
            elif k == "ratio_diagnostic":
                aggregated_result[k] = first  # type: ignore[literal-required]
            elif k in ("ratio_max", "value_max", "pre_clip_grad_norm"):
                aggregated_result[k] = max(v)  # type: ignore[literal-required]
            elif k in ("ratio_min", "value_min"):
                aggregated_result[k] = min(v)  # type: ignore[literal-required]
            else:
                aggregated_result[k] = sum(v) / len(v)  # type: ignore[literal-required]

        aggregated_result["head_entropies"] = self.head_entropies
        aggregated_result["head_grad_norms"] = self.head_grad_norms

        log_prob_min = self.log_prob_min_across_epochs
        log_prob_max = self.log_prob_max_across_epochs
        if log_prob_min == float("inf"):
            log_prob_min = float("nan")
        if log_prob_max == float("-inf"):
            log_prob_max = float("nan")
        aggregated_result["log_prob_min"] = log_prob_min
        aggregated_result["log_prob_max"] = log_prob_max

        for key in self.head_names:
            ratio_key = f"head_{key}_ratio_max"
            max_val = self.head_ratio_max_across_epochs[key]
            aggregated_result[ratio_key] = max_val if max_val != float("-inf") else 1.0  # type: ignore[literal-required]
        aggregated_result["joint_ratio_max"] = (
            self.joint_ratio_max_across_epochs if self.joint_ratio_max_across_epochs != float("-inf") else 1.0
        )

        aggregated_result["head_nan_detected"] = self.head_nan_detected
        aggregated_result["head_inf_detected"] = self.head_inf_detected

        if self.lstm_health_history["lstm_h_rms"]:
            aggregated_result["lstm_h_rms"] = (
                sum(self.lstm_health_history["lstm_h_rms"]) / len(self.lstm_health_history["lstm_h_rms"])
            )
            aggregated_result["lstm_c_rms"] = (
                sum(self.lstm_health_history["lstm_c_rms"]) / len(self.lstm_health_history["lstm_c_rms"])
            )
            aggregated_result["lstm_h_env_rms_mean"] = (
                sum(self.lstm_health_history["lstm_h_env_rms_mean"])
                / len(self.lstm_health_history["lstm_h_env_rms_mean"])
            )
            aggregated_result["lstm_c_env_rms_mean"] = (
                sum(self.lstm_health_history["lstm_c_env_rms_mean"])
                / len(self.lstm_health_history["lstm_c_env_rms_mean"])
            )
            aggregated_result["lstm_h_env_rms_max"] = max(self.lstm_health_history["lstm_h_env_rms_max"])
            aggregated_result["lstm_c_env_rms_max"] = max(self.lstm_health_history["lstm_c_env_rms_max"])
            aggregated_result["lstm_h_max"] = max(self.lstm_health_history["lstm_h_max"])
            aggregated_result["lstm_c_max"] = max(self.lstm_health_history["lstm_c_max"])
            aggregated_result["lstm_has_nan"] = any(self.lstm_health_history["lstm_has_nan"])
            aggregated_result["lstm_has_inf"] = any(self.lstm_health_history["lstm_has_inf"])
        else:
            aggregated_result["lstm_h_rms"] = None
            aggregated_result["lstm_c_rms"] = None
            aggregated_result["lstm_h_env_rms_mean"] = None
            aggregated_result["lstm_h_env_rms_max"] = None
            aggregated_result["lstm_c_env_rms_mean"] = None
            aggregated_result["lstm_c_env_rms_max"] = None
            aggregated_result["lstm_h_max"] = None
            aggregated_result["lstm_c_max"] = None
            aggregated_result["lstm_has_nan"] = None
            aggregated_result["lstm_has_inf"] = None

        aggregated_result["v_return_correlation"] = self.value_func_metrics["v_return_correlation"]
        aggregated_result["td_error_mean"] = self.value_func_metrics["td_error_mean"]
        aggregated_result["td_error_std"] = self.value_func_metrics["td_error_std"]
        aggregated_result["bellman_error"] = self.value_func_metrics["bellman_error"]
        aggregated_result["return_p10"] = self.value_func_metrics["return_p10"]
        aggregated_result["return_p50"] = self.value_func_metrics["return_p50"]
        aggregated_result["return_p90"] = self.value_func_metrics["return_p90"]
        aggregated_result["return_variance"] = self.value_func_metrics["return_variance"]
        aggregated_result["return_skewness"] = self.value_func_metrics["return_skewness"]

        if self.cuda_memory_metrics:
            for k, v in self.cuda_memory_metrics.items():
                aggregated_result[k] = v  # type: ignore[literal-required]

        return PPOUpdateMetricsResult(metrics=aggregated_result, update_performed=True)


__all__ = [
    "PPOUpdateMetricsResult",
    "PPOUpdateMetricsBuilder",
]
