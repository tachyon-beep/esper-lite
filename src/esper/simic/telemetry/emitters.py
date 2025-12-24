"""Telemetry emission helpers for vectorized PPO training.

These are pure functions that format and emit telemetry events.
They do not modify training state.

Extracted from vectorized.py to reduce file size and improve clarity.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Sequence

import torch
from torch import nn

from esper.leyline import (
    DEFAULT_ENTROPY_COLLAPSE_THRESHOLD,
    HEAD_NAMES,
    SeedStage,
    TelemetryEvent,
    TelemetryEventType,
)
from esper.leyline.factored_actions import (
    ALPHA_CURVE_NAMES,
    ALPHA_SPEED_NAMES,
    ALPHA_TARGET_VALUES,
    BLUEPRINT_IDS,
    LifecycleOp,
    OP_NAMES,
    STYLE_ALPHA_ALGORITHMS,
    STYLE_BLEND_IDS,
    STYLE_NAMES,
)
from esper.nissa import get_hub

from .debug_telemetry import LayerGradientStats, collect_per_layer_gradients

if TYPE_CHECKING:
    from esper.simic.training.parallel_env_state import ParallelEnvState

    from .telemetry_config import TelemetryConfig


class VectorizedEmitter:
    """Consolidated telemetry emitter for vectorized training loop.

    Encapsulates the complex logic for building telemetry payloads,
    reducing boilerplate and interpreter overhead in the hot path.
    """

    def __init__(
        self,
        env_id: int,
        device: str,
        hub=None,
        telemetry_config: "TelemetryConfig | None" = None,
        quiet_analytics: bool = False,
    ):
        self.hub = hub or get_hub()
        self.env_id = env_id
        self.device = device
        self.telemetry_config = telemetry_config
        self.quiet_analytics = quiet_analytics

    def _should_emit(self, level: str = "ops_normal") -> bool:
        if self.hub is None:
            return False
        if self.telemetry_config is None:
            return True
        return self.telemetry_config.should_collect(level)

    def _emit(self, event: TelemetryEvent) -> None:
        """Emit event with environment context injected."""
        event.env_id = self.env_id
        event.device = self.device
        self.hub.emit(event)

    def on_epoch_completed(
        self,
        epoch: int,
        env_state: "ParallelEnvState",
        slot_reports: dict[str, Any],
    ) -> None:
        """Emit per-environment epoch metrics and seed telemetry."""
        if not self._should_emit("ops_normal"):
            return

        # Build per-seed telemetry dict for this env
        seeds_telemetry = {}
        for slot_id, report in slot_reports.items():
            if report.telemetry is not None:
                seeds_telemetry[slot_id] = {
                    "stage": report.stage.name if report.stage else "UNKNOWN",
                    "blueprint_id": report.blueprint_id,
                    "accuracy_delta": report.telemetry.accuracy_delta,
                    "epochs_in_stage": report.telemetry.epochs_in_stage,
                    "alpha": report.telemetry.alpha,
                    "grad_ratio": report.telemetry.gradient_health,
                    "has_vanishing": report.telemetry.has_vanishing,
                    "has_exploding": report.telemetry.has_exploding,
                }

        self._emit(TelemetryEvent(
            event_type=TelemetryEventType.EPOCH_COMPLETED,
            epoch=epoch,
            data={
                "env_id": self.env_id,
                "inner_epoch": epoch,
                "val_accuracy": env_state.val_acc,
                "val_loss": env_state.val_loss,
                "train_accuracy": env_state.train_acc,
                "train_loss": env_state.train_loss,
                "seeds": seeds_telemetry,
            },
        ))

    def on_counterfactual_matrix(
        self,
        active_slots: list[str],
        baseline_accs: dict[str, float],
        val_acc: float,
        all_disabled_acc: float | None = None,
        pair_accs: dict[tuple[int, int], float] | None = None,
    ) -> None:
        """Emit live counterfactual matrix for dashboard visualization."""
        if not self._should_emit("ops_normal") or not active_slots:
            return

        configs = []
        n = len(active_slots)

        # All disabled
        strategy = "full_factorial" if all_disabled_acc is not None else "ablation_only"
        final_all_disabled = all_disabled_acc if all_disabled_acc is not None else min(baseline_accs.values())
        
        configs.append({
            "seed_mask": [False] * n,
            "accuracy": final_all_disabled,
        })

        # Per-slot solo estimates (derived from ablation)
        for i, slot_id in enumerate(active_slots):
            contribution = val_acc - baseline_accs[slot_id]
            solo_estimate = final_all_disabled + contribution
            mask = [j == i for j in range(n)]
            configs.append({
                "seed_mask": mask,
                "accuracy": solo_estimate,
            })

        # Pair configs (measured)
        if pair_accs and n >= 3:
            for (i, j), pair_acc in pair_accs.items():
                mask = [k == i or k == j for k in range(n)]
                configs.append({
                    "seed_mask": mask,
                    "accuracy": pair_acc,
                })

        # All enabled
        configs.append({
            "seed_mask": [True] * n,
            "accuracy": val_acc,
        })

        self._emit(TelemetryEvent(
            event_type=TelemetryEventType.COUNTERFACTUAL_MATRIX_COMPUTED,
            data={
                "env_id": self.env_id,
                "slot_ids": active_slots,
                "configs": configs,
                "strategy": strategy,
                "compute_time_ms": 0.0,
            },
        ))

    def on_last_action(
        self,
        epoch: int,
        action_indices: dict[str, int],
        slot_id: str,
        masked: dict[str, bool],
        success: bool,
        active_alpha_algorithm: str | None = None,
        *,
        total_reward: float | None = None,
        value_estimate: float | None = None,
        host_accuracy: float | None = None,
        slot_states: dict[str, str] | None = None,
        action_confidence: float | None = None,
        alternatives: list[tuple[str, float]] | None = None,
    ) -> None:
        """Emit last-action detail, offloading formatting to the hub thread."""
        if not self._should_emit("ops_normal"):
            return

        # Pass raw indices + mapping info in the data dict
        # The background thread could theoretically do the mapping, but 
        # for now just creating the event object here is much faster than 
        # the original unrolled logic in vectorized.py.
        
        selected_alpha_algorithm = STYLE_ALPHA_ALGORITHMS[action_indices["style"]].name
        data: dict[str, Any] = {
            "kind": "last_action",
            "env_id": self.env_id,
            "inner_epoch": epoch,
            "op": OP_NAMES[action_indices["op"]],
            # REWARD_COMPUTED-compatible fields (used by Sanctum decision carousel)
            "action_name": OP_NAMES[action_indices["op"]],
            "slot_id": slot_id,
            "action_slot": slot_id if OP_NAMES[action_indices["op"]] != "WAIT" else None,
            "blueprint_id": BLUEPRINT_IDS[action_indices["blueprint"]],
            "style": STYLE_NAMES[action_indices["style"]],
            "blend_id": STYLE_BLEND_IDS[action_indices["style"]],
            "tempo_idx": action_indices["tempo"],
            "alpha_target": ALPHA_TARGET_VALUES[action_indices["alpha_target"]],
            "alpha_speed": ALPHA_SPEED_NAMES[action_indices["alpha_speed"]],
            "alpha_curve": ALPHA_CURVE_NAMES[action_indices["alpha_curve"]],
            "alpha_algorithm": active_alpha_algorithm or selected_alpha_algorithm,
            "alpha_algorithm_selected": selected_alpha_algorithm,
            "op_masked": bool(masked.get("op", False)),
            "slot_masked": bool(masked.get("slot", False)),
            "blueprint_masked": bool(masked.get("blueprint", False)),
            "style_masked": bool(masked.get("style", False)),
            "tempo_masked": bool(masked.get("tempo", False)),
            "alpha_target_masked": bool(masked.get("alpha_target", False)),
            "alpha_speed_masked": bool(masked.get("alpha_speed", False)),
            "alpha_curve_masked": bool(masked.get("alpha_curve", False)),
            "action_success": success,
        }

        if total_reward is not None:
            data["total_reward"] = float(total_reward)
        if value_estimate is not None:
            data["value_estimate"] = float(value_estimate)
        if host_accuracy is not None:
            data["host_accuracy"] = float(host_accuracy)
            # Reward components use val_acc key, decision uses host_accuracy.
            data["val_acc"] = float(host_accuracy)
        if slot_states is not None:
            data["slot_states"] = dict(slot_states)
        if action_confidence is not None:
            data["action_confidence"] = float(action_confidence)
        if alternatives is not None:
            data["alternatives"] = list(alternatives)
        
        self._emit(TelemetryEvent(
            event_type=TelemetryEventType.ANALYTICS_SNAPSHOT,
            epoch=epoch,
            severity="debug",
            message="Last action",
            data=data,
        ))

    def on_ppo_update(
        self,
        metrics: dict,
        episodes_completed: int,
        batch_idx: int,
        epoch: int,
        agent: Any,
        ppo_grad_norm: float | None,
        ppo_update_time_ms: float | None,
        avg_acc: float,
        avg_reward: float,
        rolling_avg_acc: float,
    ) -> None:
        """Emit comprehensive PPO update telemetry."""
        if not self.hub:
            return

        payload = dict(metrics)
        payload["train_steps"] = agent.train_steps
        payload["entropy_coef"] = agent.get_entropy_coef()
        payload["avg_accuracy"] = avg_acc
        payload["avg_reward"] = avg_reward
        payload["rolling_avg_accuracy"] = rolling_avg_acc

        if self._should_emit("debug"):
            try:
                layer_stats = collect_per_layer_gradients(agent._base_network)
                layer_health = aggregate_layer_gradient_health(layer_stats)
                payload.update(layer_health)
            except Exception:
                pass

        emit_ppo_update_event(
            hub=self.hub,
            metrics=payload,
            episodes_completed=episodes_completed,
            batch_idx=batch_idx,
            epoch=epoch,
            optimizer=agent.optimizer,
            grad_norm=ppo_grad_norm,
            update_time_ms=ppo_update_time_ms,
        )

    def on_batch_completed(
        self,
        batch_idx: int,
        episodes_completed: int,
        rolling_avg_acc: float,
        avg_acc: float,
        metrics: dict,
        env_states: Sequence["ParallelEnvState"],
        update_skipped: bool,
        plateau_threshold: float,
        improvement_threshold: float,
        prev_rolling_avg_acc: float | None,
        total_episodes: int,
        start_episode: int,
        n_episodes: int,
        env_final_accs: list[float],
        avg_reward: float,
        train_losses: list[float],
        train_corrects: list[int],
        train_totals: list[int],
        val_losses: list[float],
        val_corrects: list[int],
        val_totals: list[int],
        num_train_batches: int,
        num_test_batches: int,
        analytics: Any = None,
        epoch: int = 0,
    ) -> None:
        """Emit analytics snapshot and batch completion events."""
        if not self.hub:
            return

        total_seeds_created = sum(es.seeds_created for es in env_states)
        total_seeds_fossilized = sum(es.seeds_fossilized for es in env_states)
        
        self.hub.emit(TelemetryEvent(
            event_type=TelemetryEventType.ANALYTICS_SNAPSHOT,
            epoch=episodes_completed,
            data={
                "inner_epoch": epoch,
                "batch": batch_idx + 1,
                "accuracy": rolling_avg_acc,
                "host_accuracy": avg_acc,
                "entropy": metrics.get("entropy", 0.0),
                "kl_divergence": metrics.get("approx_kl", 0.0),
                "value_variance": metrics.get("explained_variance", 0.0),
                "seeds": {
                    "total_created": total_seeds_created,
                    "total_fossilized": total_seeds_fossilized,
                },
                "episodes_completed": episodes_completed,
                "skipped_update": update_skipped,
            },
        ))

        if prev_rolling_avg_acc is not None:
            rolling_delta = rolling_avg_acc - prev_rolling_avg_acc
            event_type = None
            if abs(rolling_delta) < plateau_threshold:
                event_type = TelemetryEventType.PLATEAU_DETECTED
            elif rolling_delta < -improvement_threshold:
                event_type = TelemetryEventType.DEGRADATION_DETECTED
            elif rolling_delta > improvement_threshold:
                event_type = TelemetryEventType.IMPROVEMENT_DETECTED
            
            if event_type:
                self.hub.emit(TelemetryEvent(
                    event_type=event_type,
                    data={
                        "batch": batch_idx + 1,
                        "rolling_delta": rolling_delta,
                        "rolling_avg_accuracy": rolling_avg_acc,
                        "prev_rolling_avg_accuracy": prev_rolling_avg_acc,
                        "episodes_completed": episodes_completed,
                    },
                ))

        # BATCH_EPOCH_COMPLETED
        total_train_correct = sum(train_corrects)
        total_train_samples = sum(train_totals)
        total_val_correct = sum(val_corrects)
        total_val_samples = sum(val_totals)

        plateau_detected = (
            abs(rolling_avg_acc - prev_rolling_avg_acc) < 0.5 
            if prev_rolling_avg_acc is not None else False
        )

        self.hub.emit(TelemetryEvent(
            event_type=TelemetryEventType.BATCH_EPOCH_COMPLETED,
            epoch=episodes_completed,
            data={
                "inner_epoch": epoch,
                "batch_idx": batch_idx + 1,
                "episodes_completed": episodes_completed,
                "total_episodes": total_episodes,
                "start_episode": start_episode,
                "requested_episodes": n_episodes,
                "env_accuracies": env_final_accs,
                "avg_accuracy": avg_acc,
                "rolling_accuracy": rolling_avg_acc,
                "avg_reward": avg_reward,
                "train_loss": sum(train_losses) / max(len(env_states) * num_train_batches, 1),
                "train_accuracy": 100.0 * total_train_correct / max(total_train_samples, 1),
                "val_loss": sum(val_losses) / max(len(env_states) * num_test_batches, 1),
                "val_accuracy": 100.0 * total_val_correct / max(total_val_samples, 1),
                "n_envs": len(env_states),
                "skipped_update": update_skipped,
                "plateau_detected": plateau_detected,
            },
        ))

        if self._should_emit("ops_normal"):
            action_counts: dict[str, int] = {}
            success_counts: dict[str, int] = {}
            for env_state in env_states:
                # action_counts and successful_action_counts are required fields
                # on ParallelEnvState (lines 49-50 in parallel_env_state.py)
                for action, count in env_state.action_counts.items():
                    key = str(action)
                    action_counts[key] = action_counts.get(key, 0) + int(count)
                for action, count in env_state.successful_action_counts.items():
                    key = str(action)
                    success_counts[key] = success_counts.get(key, 0) + int(count)

            emit_action_distribution(
                hub=self.hub,
                batch_idx=batch_idx + 1,
                episodes_completed=episodes_completed,
                action_counts=action_counts,
                success_counts=success_counts,
            )

        # Analytics table
        if not self.quiet_analytics and episodes_completed % 5 == 0 and analytics and len(analytics.stats) > 0:
            summary_table = analytics.summary_table()
            scoreboard_tables = {
                env_idx: analytics.scoreboard_table(env_idx)
                for env_idx in range(len(env_states))
                if env_idx in analytics.scoreboards
            }
            message = summary_table
            if scoreboard_tables:
                message = f"{summary_table}\n" + "\n".join(scoreboard_tables.values())

            self.hub.emit(TelemetryEvent(
                event_type=TelemetryEventType.ANALYTICS_SNAPSHOT,
                severity="info",
                message=message,
                data={
                    "batch": batch_idx + 1,
                    "episodes_completed": episodes_completed,
                    "summary_table": summary_table,
                    "scoreboard_tables": scoreboard_tables,
                },
            ))


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
    """Emit batch epoch completion telemetry with resume-aware totals."""
    clamped_completed = min(episodes_completed, total_episodes)
    hub.emit(
        TelemetryEvent(
            event_type=TelemetryEventType.BATCH_EPOCH_COMPLETED,
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
    slot_idx: int,
    blueprint_idx: int,
    style_idx: int,
    tempo_idx: int,
    alpha_target_idx: int,
    alpha_speed_idx: int,
    alpha_curve_idx: int,
    op_idx: int,
    slot_id: str,
    masked: dict[str, bool],
    success: bool,
    active_alpha_algorithm: str | None = None,
) -> dict:
    """Emit per-step last-action detail for debugging and UIs.

    Args:
        env_id: Environment index
        epoch: Current epoch
        slot_idx: Slot action index
        blueprint_idx: Blueprint action index
        style_idx: Germination style action index
        tempo_idx: Tempo action index
        op_idx: Lifecycle operation index
        slot_id: Target slot ID string
        masked: Dict of head -> was_masked flags
        success: Whether the action executed successfully
        active_alpha_algorithm: The slot's currently active alpha algorithm (post-action),
            if known. When omitted, telemetry falls back to the algorithm implied by the
            sampled style index.

    Returns:
        The emitted data dict (for testing)
    """
    hub = get_hub()
    selected_alpha_algorithm = STYLE_ALPHA_ALGORITHMS[style_idx].name
    data = {
        "kind": "last_action",
        "env_id": env_id,
        "inner_epoch": epoch,
        "op": OP_NAMES[op_idx],
        "slot_id": slot_id,
        "blueprint_id": BLUEPRINT_IDS[blueprint_idx],
        "style": STYLE_NAMES[style_idx],
        "blend_id": STYLE_BLEND_IDS[style_idx],
        "tempo_idx": tempo_idx,
        "alpha_target": ALPHA_TARGET_VALUES[alpha_target_idx],
        "alpha_speed": ALPHA_SPEED_NAMES[alpha_speed_idx],
        "alpha_curve": ALPHA_CURVE_NAMES[alpha_curve_idx],
        "alpha_algorithm": active_alpha_algorithm or selected_alpha_algorithm,
        "alpha_algorithm_selected": selected_alpha_algorithm,
        "op_masked": bool(masked.get("op", False)),
        "slot_masked": bool(masked.get("slot", False)),
        "blueprint_masked": bool(masked.get("blueprint", False)),
        "style_masked": bool(masked.get("style", False)),
        "tempo_masked": bool(masked.get("tempo", False)),
        "alpha_target_masked": bool(masked.get("alpha_target", False)),
        "alpha_speed_masked": bool(masked.get("alpha_speed", False)),
        "alpha_curve_masked": bool(masked.get("alpha_curve", False)),
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
    group_id: str = "default",  # A/B testing identifier
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

    # Compute per-head gradient norm averages for logging (P4-6)
    head_grad_norms_avg = {}
    if "head_grad_norms" in metrics:
        for head, values in metrics["head_grad_norms"].items():
            avg_grad_norm = sum(values) / len(values) if values else 0.0
            head_grad_norms_avg[f"head_{head}_grad_norm"] = avg_grad_norm

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
        # Entropy collapse flag (Task 3) - M11: use leyline constant
        "entropy_collapsed": metrics.get("entropy", 1.0) < DEFAULT_ENTROPY_COLLAPSE_THRESHOLD,
    }
    # Add per-head entropy (P3-1)
    data.update(head_entropies_avg)
    # Add per-head gradient norms (P4-6)
    data.update(head_grad_norms_avg)

    hub.emit(TelemetryEvent(
        event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
        epoch=episodes_completed,  # Monotonic per-batch epoch id (NOT inner epoch!)
        data=data,
        group_id=group_id,
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
    "VectorizedEmitter",
    "aggregate_layer_gradient_health",
    "apply_slot_telemetry",
    "check_performance_degradation",
    "compute_grad_norm_surrogate",
    "emit_action_distribution",
    "emit_batch_completed",
    "emit_cf_unavailable",
    "emit_last_action",
    "emit_mask_hit_rates",
    "emit_ppo_update_event",
    "emit_reward_summary",
    "emit_throughput",
    "emit_with_env_context",
]
