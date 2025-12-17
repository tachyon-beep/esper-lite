"""Telemetry emission helpers for vectorized PPO training.

These are pure functions that format and emit telemetry events.
They do not modify training state.

Extracted from vectorized.py to reduce file size and improve clarity.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import torch
from torch import nn

from esper.leyline import TelemetryEvent, TelemetryEventType
from .debug_telemetry import LayerGradientStats
from esper.nissa import get_hub

if TYPE_CHECKING:
    from .telemetry_config import TelemetryConfig


def emit_with_env_context(hub, env_idx: int, device: str, event: TelemetryEvent) -> None:
    """Safely emit telemetry with env_id/device injected and no shared mutation.

    Creates a new event with the additional context rather than mutating the input.
    """
    data = dict(event.data) if event.data else {}
    data["env_id"] = env_idx
    data["device"] = device
    new_event = dataclasses.replace(event, data=data)
    hub.emit(new_event)


def emit_batch_completed(
    hub,
    *,
    batch_idx: int,
    episodes_completed: int,
    total_episodes: int,
    env_final_accs: list[float],
    avg_acc: float,
    rolling_avg_acc: float,
    avg_reward: float,
    start_episode: int,
    requested_episodes: int,
) -> None:
    """Emit batch completion telemetry with resume-aware totals."""
    clamped_completed = min(episodes_completed, total_episodes)
    hub.emit(
        TelemetryEvent(
            event_type=TelemetryEventType.BATCH_COMPLETED,
            data={
                "batch_idx": batch_idx,
                "episodes_completed": clamped_completed,
                "total_episodes": total_episodes,
                "start_episode": start_episode,
                "requested_episodes": requested_episodes,
                "env_accuracies": env_final_accs,
                "avg_accuracy": avg_acc,
                "rolling_accuracy": rolling_avg_acc,
                "avg_reward": avg_reward,
            },
        )
    )


def emit_last_action(
    *,
    env_id: int,
    epoch: int,
    factored_action,
    slot_id: str,
    masked: dict[str, bool],
    success: bool,
) -> dict:
    """Emit per-step last-action detail for debugging and UIs."""
    hub = get_hub()
    data = {
        "kind": "last_action",
        "env_id": env_id,
        "inner_epoch": epoch,
        "op": factored_action.op.name,
        "slot_id": slot_id,
        "blueprint_id": factored_action.blueprint_id,
        "blend_id": factored_action.blend_algorithm_id,
        "op_masked": bool(masked.get("op", False)),
        "slot_masked": bool(masked.get("slot", False)),
        "blueprint_masked": bool(masked.get("blueprint", False)),
        "blend_masked": bool(masked.get("blend", False)),
        "action_success": success,
    }
    hub.emit(TelemetryEvent(
        event_type=TelemetryEventType.ANALYTICS_SNAPSHOT,
        epoch=epoch,
        severity="debug",
        message="Last action",
        data=data,
    ))
    return data


def compute_grad_norm_surrogate(module: nn.Module) -> float | None:
    """Compute a cheap grad-norm surrogate (single device sync)."""
    grad_norm_sq = None
    for param in module.parameters():
        if param.grad is None:
            continue
        g = param.grad.detach()
        g_sq = (g.float() * g.float()).sum()
        grad_norm_sq = g_sq if grad_norm_sq is None else grad_norm_sq + g_sq
    if grad_norm_sq is None:
        return None
    return float(torch.sqrt(grad_norm_sq).item())


def aggregate_layer_gradient_health(
    layer_stats: list[LayerGradientStats],
) -> dict[str, int | float]:
    """Aggregate per-layer gradient stats into summary metrics.

    Args:
        layer_stats: List from collect_per_layer_gradients()

    Returns:
        Dict with dead_layers, exploding_layers, nan_grad_count, layer_gradient_health
    """
    if not layer_stats:
        return {
            "dead_layers": 0,
            "exploding_layers": 0,
            "nan_grad_count": 0,
            "layer_gradient_health": 1.0,
        }

    n_layers = len(layer_stats)
    dead = sum(1 for s in layer_stats if s.zero_fraction > 0.9)
    exploding = sum(1 for s in layer_stats if s.large_fraction > 0.1)
    nan_count = sum(s.nan_count for s in layer_stats)

    # Compute health score: 1.0 = perfect, penalize dead/exploding layers
    health = 1.0
    health -= (dead / n_layers) * 0.5  # Dead layers reduce health
    health -= (exploding / n_layers) * 0.8  # Exploding is worse
    health -= min(nan_count / 100, 0.5)  # NaNs are bad
    health = max(0.0, min(1.0, health))

    return {
        "dead_layers": dead,
        "exploding_layers": exploding,
        "nan_grad_count": nan_count,
        "layer_gradient_health": health,
    }


def emit_ppo_update_event(
    *,
    hub,
    metrics: dict,
    episodes_completed: int,
    batch_idx: int,
    epoch: int,
    optimizer,
    grad_norm: float | None,
    update_time_ms: float | None,
) -> None:
    """Emit PPO update completion telemetry with optional vitals."""
    lr = None
    if optimizer is not None:
        try:
            lr = optimizer.param_groups[0].get("lr")
        except (AttributeError, IndexError, KeyError, TypeError):
            lr = None

    # Compute per-head entropy averages for logging (P3-1)
    head_entropies_avg = {}
    if "head_entropies" in metrics:
        for head, values in metrics["head_entropies"].items():
            avg_entropy = sum(values) / len(values) if values else 0.0
            head_entropies_avg[f"{head}_entropy"] = avg_entropy

    data = {
        "inner_epoch": epoch,  # Final inner epoch (typically max_epochs)
        "batch": batch_idx + 1,
        "episodes_completed": episodes_completed,
        "train_steps": metrics.get("train_steps", 0),
        # Core losses
        "policy_loss": metrics.get("policy_loss", 0.0),
        "value_loss": metrics.get("value_loss", 0.0),
        "entropy": metrics.get("entropy", 0.0),
        "entropy_coef": metrics.get("entropy_coef", 0.0),
        # PPO health (KL, clipping) - normalized to kl_divergence for Karn
        "kl_divergence": metrics.get("approx_kl", 0.0),
        "clip_fraction": metrics.get("clip_fraction", 0.0),
        # Ratio statistics (early warning for policy collapse)
        "ratio_max": metrics.get("ratio_max", 1.0),
        "ratio_min": metrics.get("ratio_min", 1.0),
        "ratio_std": metrics.get("ratio_std", 0.0),
        # Value function health (negative = critic broken)
        "explained_variance": metrics.get("explained_variance", 0.0),
        # Early stopping info
        "early_stop_epoch": metrics.get("early_stop_epoch"),
        # Episode-level metrics
        "avg_accuracy": metrics.get("avg_accuracy", 0.0),
        "avg_reward": metrics.get("avg_reward", 0.0),
        "rolling_avg_accuracy": metrics.get("rolling_avg_accuracy", 0.0),
        # Vitals
        "lr": lr,
        "grad_norm": grad_norm,
        "update_time_ms": update_time_ms,
        # Gradient layer health (Task 1)
        "dead_layers": metrics.get("dead_layers", 0),
        "exploding_layers": metrics.get("exploding_layers", 0),
        "nan_grad_count": metrics.get("nan_grad_count", 0),
        # Gradient health score (Task 2)
        "layer_gradient_health": metrics.get("layer_gradient_health"),
        # Entropy collapse flag (Task 3)
        "entropy_collapsed": metrics.get("entropy", 1.0) < 0.1,
    }
    # Add per-head entropy (P3-1)
    data.update(head_entropies_avg)

    hub.emit(TelemetryEvent(
        event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
        epoch=episodes_completed,  # Monotonic per-batch epoch id (NOT inner epoch!)
        data=data,
    ))


def emit_action_distribution(
    *,
    hub,
    batch_idx: int,
    episodes_completed: int,
    action_counts: dict[str, int],
    success_counts: dict[str, int],
) -> None:
    """Emit per-batch action distribution summary."""
    hub.emit(TelemetryEvent(
        event_type=TelemetryEventType.ANALYTICS_SNAPSHOT,
        epoch=episodes_completed,
        data={
            "kind": "action_distribution",
            "batch": batch_idx,
            "episodes_completed": episodes_completed,
            "action_counts": dict(action_counts),
            "success_counts": dict(success_counts),
        },
    ))


def emit_cf_unavailable(
    hub,
    *,
    env_id: int,
    slot_id: str,
    reason: str,
) -> None:
    """Emit marker event when counterfactual baseline is unavailable."""
    hub.emit(TelemetryEvent(
        event_type=TelemetryEventType.COUNTERFACTUAL_COMPUTED,
        slot_id=slot_id,
        severity="warning",
        data={
            "env_id": env_id,
            "slot_id": slot_id,
            "available": False,
            "reason": reason,
        },
    ))


def emit_throughput(
    *,
    hub,
    env_id: int,
    batch_idx: int,
    episodes_completed: int,
    step_time_ms: float,
    dataloader_wait_ms: float,
) -> None:
    """Emit per-env throughput metrics for this batch."""
    fps = 1000.0 / step_time_ms if step_time_ms > 0 else None
    hub.emit(TelemetryEvent(
        event_type=TelemetryEventType.ANALYTICS_SNAPSHOT,
        epoch=episodes_completed,
        data={
            "kind": "throughput",
            "env_id": env_id,
            "batch": batch_idx,
            "episodes_completed": episodes_completed,
            "fps": fps,
            "step_time_ms": step_time_ms,
            "dataloader_wait_ms": dataloader_wait_ms,
        },
    ))


def emit_reward_summary(
    *,
    hub,
    env_id: int,
    batch_idx: int,
    summary: dict[str, float],
    episodes_completed: int = 0,
) -> None:
    """Emit compact reward summary for this batch."""
    hub.emit(TelemetryEvent(
        event_type=TelemetryEventType.ANALYTICS_SNAPSHOT,
        epoch=episodes_completed,
        data={
            "kind": "reward_summary",
            "env_id": env_id,
            "batch": batch_idx,
            "episodes_completed": episodes_completed,
            "summary": dict(summary),
        },
    ))


def emit_mask_hit_rates(
    *,
    hub,
    batch_idx: int,
    episodes_completed: int,
    mask_hits: dict[str, int],
    mask_total: dict[str, int],
) -> None:
    """Emit per-head mask hit rates for this batch."""
    hub.emit(TelemetryEvent(
        event_type=TelemetryEventType.ANALYTICS_SNAPSHOT,
        epoch=episodes_completed,
        data={
            "kind": "mask_hit_rates",
            "batch": batch_idx,
            "episodes_completed": episodes_completed,
            "mask_hits": dict(mask_hits),
            "mask_total": dict(mask_total),
        },
    ))


# TODO: [UNWIRED TELEMETRY] - Call check_performance_degradation() at end of each epoch
# with current accuracy vs rolling average. See telemetry-phase3.md Task 5 for integration notes.
def check_performance_degradation(
    hub,
    *,
    current_acc: float,
    rolling_avg_acc: float,
    degradation_threshold: float = 0.1,
    env_id: int = 0,
    training_progress: float = 1.0,
    warmup_threshold: float = 0.1,
) -> bool:
    """Emit PERFORMANCE_DEGRADATION if accuracy dropped significantly.

    PPO has natural 15-20% accuracy variance during early training, so we
    skip emissions during warmup (first 10% of training by default) to
    avoid false positives from normal policy exploration.

    Args:
        hub: Telemetry hub for event emission
        current_acc: Current accuracy value
        rolling_avg_acc: Rolling average accuracy for comparison
        degradation_threshold: Minimum relative drop to trigger (default 10%)
        env_id: Environment ID for attribution
        training_progress: Progress through training (0.0 to 1.0)
        warmup_threshold: Skip emissions below this progress (default 0.1 = 10%)

    Returns True if event was emitted.
    """
    # Skip during warmup - PPO has high variance early in training
    if training_progress < warmup_threshold:
        return False

    if rolling_avg_acc <= 0:
        return False

    drop = (rolling_avg_acc - current_acc) / rolling_avg_acc

    if drop < degradation_threshold:
        return False

    hub.emit(TelemetryEvent(
        event_type=TelemetryEventType.PERFORMANCE_DEGRADATION,
        severity="warning",
        data={
            "current_acc": current_acc,
            "rolling_avg_acc": rolling_avg_acc,
            "drop_percent": drop * 100,
            "threshold_percent": degradation_threshold * 100,
            "env_id": env_id,
            "training_progress": training_progress,
        },
    ))
    return True


def apply_slot_telemetry(
    env_state,
    *,
    ops_telemetry_enabled: bool,
    lifecycle_only: bool,
    inner_epoch: int | None = None,
    global_epoch: int | None = None,
) -> None:
    """Configure slot telemetry and fast_mode based on current telemetry settings.

    Lifecycle-only mode keeps lightweight lifecycle events enabled even when
    ops telemetry is disabled and slots are running in fast_mode.
    """
    for slot in env_state.model.seed_slots.values():
        slot.telemetry_inner_epoch = inner_epoch
        slot.telemetry_global_epoch = global_epoch
        slot.fast_mode = not ops_telemetry_enabled
        if ops_telemetry_enabled:
            slot.on_telemetry = env_state.telemetry_cb
            slot.telemetry_lifecycle_only = False
            continue

        slot.telemetry_lifecycle_only = lifecycle_only
        slot.on_telemetry = env_state.telemetry_cb if lifecycle_only else None


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
