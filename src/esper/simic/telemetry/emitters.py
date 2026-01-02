"""Telemetry emission helpers for vectorized PPO training.

These are pure functions that format and emit telemetry events.
They do not modify training state.

Extracted from vectorized.py to reduce file size and improve clarity.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Any, Sequence

import torch
from torch import nn

from esper.leyline import (
    ALPHA_CURVE_NAMES,
    ALPHA_SPEED_NAMES,
    ALPHA_TARGET_VALUES,
    AnalyticsSnapshotPayload,
    BatchEpochCompletedPayload,
    BLUEPRINT_IDS,
    CounterfactualMatrixPayload,
    DEFAULT_ENTROPY_COLLAPSE_THRESHOLD,
    EpochCompletedPayload,
    OP_NAMES,
    PPOUpdatePayload,
    STYLE_ALPHA_ALGORITHMS,
    STYLE_BLEND_IDS,
    STYLE_NAMES,
    TelemetryEvent,
    TelemetryEventType,
    TrendDetectedPayload,
)
from esper.leyline.telemetry import HeadTelemetry
from esper.nissa import get_hub

from .debug_telemetry import LayerGradientStats, collect_per_layer_gradients

if TYPE_CHECKING:
    from esper.simic.rewards.reward_telemetry import RewardComponentsTelemetry
    from esper.simic.training.parallel_env_state import ParallelEnvState

    from .telemetry_config import TelemetryConfig

_logger = logging.getLogger(__name__)


class VectorizedEmitter:
    """Consolidated telemetry emitter for vectorized training loop.

    Encapsulates the complex logic for building telemetry payloads,
    reducing boilerplate and interpreter overhead in the hot path.
    """

    def __init__(
        self,
        env_id: int,
        device: str,
        hub: Any = None,
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
        event.env_id = self.env_id  # type: ignore[attr-defined]
        event.device = self.device  # type: ignore[attr-defined]
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
                    # Inter-slot interaction metrics (from SeedMetrics - always present via default_factory)
                    # Note: These are zero for n>3 seeds due to factorial complexity limits
                    "contribution_velocity": report.metrics.contribution_velocity,
                    "interaction_sum": report.metrics.interaction_sum,
                    "boost_received": report.metrics.boost_received,
                    "upstream_alpha_sum": report.metrics.upstream_alpha_sum,
                    "downstream_alpha_sum": report.metrics.downstream_alpha_sum,
                }

        self._emit(TelemetryEvent(
            event_type=TelemetryEventType.EPOCH_COMPLETED,
            epoch=epoch,
            data=EpochCompletedPayload(
                env_id=self.env_id,
                val_accuracy=env_state.val_acc,
                val_loss=env_state.val_loss,
                inner_epoch=epoch,
                seeds=seeds_telemetry,
            ),
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
            data=CounterfactualMatrixPayload(
                env_id=self.env_id,
                slot_ids=tuple(active_slots),
                configs=tuple(configs),
                strategy=strategy,
                compute_time_ms=0.0,
            ),
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
        decision_entropy: float | None = None,
        reward_components: "RewardComponentsTelemetry | None" = None,
        head_telemetry: HeadTelemetry | None = None,
    ) -> None:
        """Emit last-action telemetry from the training loop.

        This is the PRODUCTION path called by vectorized.py during training.
        It accepts action indices as a dict and includes rich context (rewards,
        confidence, alternatives) for the Sanctum TUI decision cards.

        Note: There is also a standalone `emit_last_action()` function at module
        level used for UNIT TESTING. That function has explicit *_idx parameters
        and returns a dict for test assertions. Both must emit the same fields
        to AnalyticsSnapshotPayload - if you add a field here, add it there too.

        Args:
            epoch: Current training epoch
            action_indices: Dict mapping head names to action indices
            slot_id: Target slot for the action
            masked: Dict mapping head names to mask flags
            success: Whether the action executed successfully
            active_alpha_algorithm: Current alpha algorithm for the slot
            total_reward: Total reward for this step
            value_estimate: Value function estimate
            host_accuracy: Current host model validation accuracy
            slot_states: Dict mapping slot IDs to state descriptions
            action_confidence: Confidence (probability) of the chosen action
            alternatives: Top-2 alternative actions with probabilities
            decision_entropy: Entropy of the action distribution
            reward_components: Typed dataclass with full reward breakdown (may be None for LOSS family)
            head_telemetry: Typed dataclass with per-head confidence and entropy values
        """
        if not self._should_emit("ops_normal"):
            return

        # Pass raw indices + mapping info in the data dict
        # The background thread could theoretically do the mapping, but
        # for now just creating the event object here is much faster than
        # the original unrolled logic in vectorized.py.

        action_name = OP_NAMES[action_indices["op"]]
        blueprint_id = BLUEPRINT_IDS[action_indices["blueprint"]]
        style_idx = action_indices["style"]
        style = STYLE_NAMES[style_idx]
        blend_id = STYLE_BLEND_IDS[style_idx]
        selected_alpha_algorithm = STYLE_ALPHA_ALGORITHMS[style_idx].name
        alpha_algorithm = active_alpha_algorithm or selected_alpha_algorithm
        tempo_idx = action_indices["tempo"]
        alpha_target = ALPHA_TARGET_VALUES[action_indices["alpha_target"]]
        alpha_speed = ALPHA_SPEED_NAMES[action_indices["alpha_speed"]]
        alpha_curve = ALPHA_CURVE_NAMES[action_indices["alpha_curve"]]
        alpha_target_masked = bool(masked.get("alpha_target", False))
        alpha_speed_masked = bool(masked.get("alpha_speed", False))
        alpha_curve_masked = bool(masked.get("alpha_curve", False))
        op_masked = bool(masked.get("op", False))
        slot_masked = bool(masked.get("slot", False))
        blueprint_masked = bool(masked.get("blueprint", False))
        style_masked = bool(masked.get("style", False))
        tempo_masked = bool(masked.get("tempo", False))

        self._emit(TelemetryEvent(
            event_type=TelemetryEventType.ANALYTICS_SNAPSHOT,
            epoch=epoch,
            severity="debug",
            message="Last action",
            data=AnalyticsSnapshotPayload(
                kind="last_action",
                env_id=self.env_id,
                inner_epoch=epoch,
                total_reward=total_reward,
                action_name=action_name,
                action_confidence=action_confidence,
                value_estimate=value_estimate,
                # Pass typed dataclasses directly - replaces individual component fields
                reward_components=reward_components,
                head_telemetry=head_telemetry,
                # Decision context for TamiyoBrain Decision Cards
                slot_states=slot_states,
                alternatives=alternatives,
                decision_entropy=decision_entropy,
                # Head choice fields (for decision card sub-decision display)
                slot_id=slot_id,
                blueprint_id=blueprint_id,
                tempo_idx=tempo_idx,
                style=style,
                blend_id=blend_id,
                alpha_target=alpha_target,
                alpha_speed=alpha_speed,
                alpha_curve=alpha_curve,
                alpha_algorithm=alpha_algorithm,
                alpha_algorithm_selected=selected_alpha_algorithm,
                action_success=success,
                op_masked=op_masked,
                slot_masked=slot_masked,
                blueprint_masked=blueprint_masked,
                style_masked=style_masked,
                tempo_masked=tempo_masked,
                alpha_target_masked=alpha_target_masked,
                alpha_speed_masked=alpha_speed_masked,
                alpha_curve_masked=alpha_curve_masked,
            ),
        ))

    def on_ppo_update(
        self,
        metrics: dict[str, Any],
        episodes_completed: int,
        batch_idx: int,
        epoch: int,
        agent: Any,
        ppo_grad_norm: float,  # MANDATORY: computed during backward pass
        ppo_update_time_ms: float,  # MANDATORY: timing always captured
        avg_acc: float,
        avg_reward: float,
        rolling_avg_acc: float,
    ) -> None:
        """Emit comprehensive PPO update telemetry."""
        if not self.hub:
            return

        payload = dict(metrics)
        payload["entropy_coef"] = agent.get_entropy_coef()

        if self._should_emit("debug"):
            try:
                # Access network via policy bundle (Tamiyo migration renamed _base_network)
                layer_stats = collect_per_layer_gradients(agent.policy.network)
                layer_health = aggregate_layer_gradient_health(layer_stats)
                payload.update(layer_health)
            except (RuntimeError, AttributeError) as e:
                # HIGH-02 fix: Narrow to expected gradient collection failures
                _logger.warning("Failed to collect per-layer gradient stats: %s", e)

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
        metrics: dict[str, Any],
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
            data=AnalyticsSnapshotPayload(
                kind="batch_stats",
                inner_epoch=epoch,
                batch=batch_idx + 1,
                accuracy=rolling_avg_acc,
                host_accuracy=avg_acc,
                entropy=metrics.get("entropy", 0.0),
                kl_divergence=metrics.get("approx_kl", 0.0),
                value_variance=metrics.get("explained_variance", 0.0),
                seeds_created=total_seeds_created,
                seeds_fossilized=total_seeds_fossilized,
                episodes_completed=episodes_completed,
                skipped_update=update_skipped,
            ),
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
                    data=TrendDetectedPayload(
                        batch_idx=batch_idx + 1,
                        episodes_completed=episodes_completed,
                        rolling_delta=rolling_delta,
                        rolling_avg_accuracy=rolling_avg_acc,
                        prev_rolling_avg_accuracy=prev_rolling_avg_acc,
                    ),
                ))

        # BATCH_EPOCH_COMPLETED
        self.hub.emit(TelemetryEvent(
            event_type=TelemetryEventType.BATCH_EPOCH_COMPLETED,
            epoch=episodes_completed,
            data=BatchEpochCompletedPayload(
                episodes_completed=episodes_completed,
                batch_idx=batch_idx + 1,
                avg_accuracy=avg_acc,
                avg_reward=avg_reward,
                total_episodes=total_episodes,
                n_envs=len(env_states),
                rolling_accuracy=rolling_avg_acc,
                env_accuracies=tuple(env_final_accs),
            ),
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
                data=AnalyticsSnapshotPayload(
                    kind="summary_table",
                    batch=batch_idx + 1,
                    episodes_completed=episodes_completed,
                    summary_table=summary_table,
                    scoreboard_tables=scoreboard_tables,
                ),
            ))


def emit_with_env_context(hub: Any, env_idx: int, device: str, event: TelemetryEvent) -> None:
    """Emit telemetry with env_id injected for per-environment events.

    Creates a new event with the additional context rather than mutating the input.

    For typed payloads from slots (which don't know their env_id), replaces the
    sentinel env_id=-1 with the actual env_id.

    NOTE: Only use for per-env events (seed lifecycle, epoch completed, etc.).
    Batch-level events (PPO updates, batch completed) should use hub.emit() directly.

    Raises:
        TypeError: If event.data is None, a dict, or missing env_id attribute.
            This indicates a caller is emitting untyped payloads (fix the emitter).
    """
    if event.data is None:
        raise TypeError(
            f"emit_with_env_context requires typed payload, got None for {event.event_type}"
        )
    if isinstance(event.data, dict):
        raise TypeError(
            f"emit_with_env_context requires typed payload, got dict for {event.event_type}. "
            "Migrate emitter to use typed dataclass payload."
        )

    # Typed payload - replace sentinel env_id with actual env_id
    # Type ignore: union type from TelemetryPayload doesn't expose env_id,
    # but all seed-lifecycle payloads have it (checked at runtime above)
    payload = dataclasses.replace(event.data, env_id=env_idx)  # type: ignore[arg-type]
    new_event = dataclasses.replace(event, data=payload)
    hub.emit(new_event)


def emit_batch_completed(
    hub: Any,
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
            data=BatchEpochCompletedPayload(
                episodes_completed=clamped_completed,
                batch_idx=batch_idx,
                avg_accuracy=avg_acc,
                avg_reward=avg_reward,
                total_episodes=total_episodes,
                n_envs=len(env_final_accs),
                start_episode=start_episode,
                requested_episodes=requested_episodes,
                rolling_accuracy=rolling_avg_acc,
                env_accuracies=tuple(env_final_accs),
            ),
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
) -> dict[str, Any]:
    """Standalone last-action emitter for UNIT TESTING.

    This function exists separately from VectorizedEmitter.on_last_action() to
    allow testing telemetry emission without instantiating the full emitter class.
    It has explicit *_idx parameters (vs a dict) and returns the payload for
    test assertions.

    IMPORTANT: Both this function and on_last_action() must emit the same fields
    to AnalyticsSnapshotPayload. If you add a field to one, add it to the other.
    The on_last_action() method also includes additional context (rewards,
    confidence, head_telemetry) that this test helper omits.

    Args:
        env_id: Environment index
        epoch: Current epoch
        slot_idx: Slot action index
        blueprint_idx: Blueprint action index
        style_idx: Germination style action index
        tempo_idx: Tempo action index
        alpha_target_idx: Alpha target action index
        alpha_speed_idx: Alpha speed action index
        alpha_curve_idx: Alpha curve action index
        op_idx: Lifecycle operation index
        slot_id: Target slot ID string
        masked: Dict of head -> was_masked flags
        success: Whether the action executed successfully
        active_alpha_algorithm: The slot's currently active alpha algorithm (post-action),
            if known. When omitted, telemetry falls back to the algorithm implied by the
            sampled style index.

    Returns:
        The emitted data dict (for test assertions)
    """
    hub = get_hub()
    selected_alpha_algorithm = STYLE_ALPHA_ALGORITHMS[style_idx].name
    payload = AnalyticsSnapshotPayload(
        kind="last_action",
        env_id=env_id,
        inner_epoch=epoch,
        action_name=OP_NAMES[op_idx],
        slot_id=slot_id,
        blueprint_id=BLUEPRINT_IDS[blueprint_idx],
        style=STYLE_NAMES[style_idx],
        blend_id=STYLE_BLEND_IDS[style_idx],
        tempo_idx=tempo_idx,
        alpha_target=ALPHA_TARGET_VALUES[alpha_target_idx],
        alpha_speed=ALPHA_SPEED_NAMES[alpha_speed_idx],
        alpha_curve=ALPHA_CURVE_NAMES[alpha_curve_idx],
        alpha_algorithm=active_alpha_algorithm or selected_alpha_algorithm,
        alpha_algorithm_selected=selected_alpha_algorithm,
        op_masked=bool(masked.get("op", False)),
        slot_masked=bool(masked.get("slot", False)),
        blueprint_masked=bool(masked.get("blueprint", False)),
        style_masked=bool(masked.get("style", False)),
        tempo_masked=bool(masked.get("tempo", False)),
        alpha_target_masked=bool(masked.get("alpha_target", False)),
        alpha_speed_masked=bool(masked.get("alpha_speed", False)),
        alpha_curve_masked=bool(masked.get("alpha_curve", False)),
        action_success=success,
    )
    hub.emit(TelemetryEvent(
        event_type=TelemetryEventType.ANALYTICS_SNAPSHOT,
        epoch=epoch,
        severity="debug",
        message="Last action",
        data=payload,
    ))
    # Return dict for backwards compatibility with tests
    return {
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


def compute_grad_norm_surrogate(module: nn.Module) -> float:
    """Compute gradient L2 norm with float64 overflow protection.

    Uses fused _foreach_norm for minimal GPU-CPU sync overhead (single sync).
    Float64 prevents overflow: float32 overflows at ~1e19 due to squaring,
    float64 handles up to ~1e154.

    MUST be called after loss.backward() - having no gradients is a bug.
    """
    grads = [p.grad for p in module.parameters() if p.grad is not None]
    # No gradients after backward() indicates a bug (e.g., torch.compile wrapper issue)
    assert grads, "No gradients found - did loss.backward() run? Check torch.compile wrapper."
    # Upcast to float64 for overflow protection on large norms
    grads_double = [g.double() for g in grads]
    # Fused kernel: computes all per-tensor norms in one launch
    per_param_norms = torch._foreach_norm(grads_double, ord=2)
    # Single reduction via vector_norm (fused) and sync point
    total_norm = torch.linalg.vector_norm(torch.stack(per_param_norms))
    return float(total_norm.item())


def aggregate_layer_gradient_health(
    layer_stats: list[LayerGradientStats],
) -> dict[str, int | float | dict[str, float]]:
    """Aggregate per-layer gradient stats into summary metrics.

    Args:
        layer_stats: List from collect_per_layer_gradients()

    Returns:
        Dict with dead_layers, exploding_layers, nan_grad_count, layer_gradient_health
        layer_gradient_health is a dict mapping layer_name -> health_score (0-1)
    """
    if not layer_stats:
        return {
            "dead_layers": 0,
            "exploding_layers": 0,
            "nan_grad_count": 0,
            "layer_gradient_health": {},
        }

    dead = sum(1 for s in layer_stats if s.zero_fraction > 0.9)
    exploding = sum(1 for s in layer_stats if s.large_fraction > 0.1)
    nan_count = sum(s.nan_count for s in layer_stats)

    # Per-layer health scores: 1.0 = perfect, penalize based on stats
    # Score indicates: 1.0=healthy, 0.5=warning, 0.0=dead/exploding
    per_layer_health: dict[str, float] = {}
    for s in layer_stats:
        health = 1.0
        # Dead layer: >90% zeros
        if s.zero_fraction > 0.9:
            health = 0.0
        # Exploding layer: >10% large gradients
        elif s.large_fraction > 0.1:
            health = 0.1
        # Warning: >50% zeros or >5% large
        elif s.zero_fraction > 0.5 or s.large_fraction > 0.05:
            health = 0.5
        # Slight concern: >30% zeros
        elif s.zero_fraction > 0.3:
            health = 0.7
        # NaN/Inf always critical
        if s.nan_count > 0 or s.inf_count > 0:
            health = 0.0
        per_layer_health[s.layer_name] = health

    return {
        "dead_layers": dead,
        "exploding_layers": exploding,
        "nan_grad_count": nan_count,
        "layer_gradient_health": per_layer_health,
    }


def emit_ppo_update_event(
    *,
    hub: Any,
    metrics: dict[str, Any],
    episodes_completed: int,
    batch_idx: int,
    epoch: int,
    optimizer: Any,
    grad_norm: float,  # MANDATORY: must be provided after backward pass
    update_time_ms: float,  # MANDATORY: timing must always be captured
    group_id: str = "default",  # A/B testing identifier
) -> None:
    """Emit PPO update completion telemetry.

    All core metrics are MANDATORY - this function will fail loudly if
    the metrics dict is missing required keys. This prevents bug-hiding
    patterns where defaults silently mask missing data.
    """
    # PT-07 fix: Direct access - all torch.optim.Optimizer subclasses have param_groups[0]["lr"]
    # If this fails, the optimizer is fundamentally broken and we should fail loudly
    lr = optimizer.param_groups[0]["lr"] if optimizer is not None else None

    # Compute per-head entropy averages for logging (P3-1)
    # Key format: head_{name}_entropy to match aggregator field names
    head_entropies_avg = {}
    if "head_entropies" in metrics:
        for head, values in metrics["head_entropies"].items():
            avg_entropy = sum(values) / len(values)  # Fail on empty list
            head_entropies_avg[f"head_{head}_entropy"] = avg_entropy

    # Compute per-head gradient norm averages for logging (P4-6)
    # No defensive pattern - empty lists should fail loudly (indicates PPO bug)
    head_grad_norms_avg = {}
    if "head_grad_norms" in metrics:
        for head, values in metrics["head_grad_norms"].items():
            avg_grad_norm = sum(values) / len(values)  # Fail on empty list
            head_grad_norms_avg[f"head_{head}_grad_norm"] = avg_grad_norm

    hub.emit(TelemetryEvent(
        event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
        epoch=episodes_completed,  # Monotonic per-batch epoch id (NOT inner epoch!)
        data=PPOUpdatePayload(
            # MANDATORY metrics - fail loudly if missing (no bug-hiding defaults)
            policy_loss=metrics["policy_loss"],
            value_loss=metrics["value_loss"],
            entropy=metrics["entropy"],
            grad_norm=grad_norm,
            kl_divergence=metrics["approx_kl"],
            clip_fraction=metrics["clip_fraction"],
            nan_grad_count=metrics.get("nan_grad_count", 0),
            # BUG FIX: Pre-clip gradient norm for detecting explosion vs healthy gradients
            pre_clip_grad_norm=metrics["pre_clip_grad_norm"],
            # Optional fields (computed once per update, not per-epoch)
            explained_variance=metrics.get("explained_variance"),
            entropy_loss=0.0,
            # MANDATORY advantage statistics - computed in PPO update
            advantage_mean=metrics["advantage_mean"],
            advantage_std=metrics["advantage_std"],
            advantage_skewness=metrics["advantage_skewness"],
            advantage_kurtosis=metrics["advantage_kurtosis"],
            advantage_positive_ratio=metrics["advantage_positive_ratio"],
            # MANDATORY ratio statistics - computed in PPO update
            ratio_mean=metrics["ratio_mean"],
            ratio_min=metrics["ratio_min"],
            ratio_max=metrics["ratio_max"],
            ratio_std=metrics["ratio_std"],
            # MANDATORY log prob extremes (NaN predictor)
            log_prob_min=metrics["log_prob_min"],
            log_prob_max=metrics["log_prob_max"],
            # MANDATORY value function statistics for drift monitoring
            value_mean=metrics["value_mean"],
            value_std=metrics["value_std"],
            value_min=metrics["value_min"],
            value_max=metrics["value_max"],
            # Q-values (Policy V2 op-conditioned critic)
            # Use NaN default to distinguish "missing" from "zero" (CRIT-01 fix)
            q_germinate=metrics.get("q_germinate", float("nan")),
            q_advance=metrics.get("q_advance", float("nan")),
            q_fossilize=metrics.get("q_fossilize", float("nan")),
            q_prune=metrics.get("q_prune", float("nan")),
            q_wait=metrics.get("q_wait", float("nan")),
            q_set_alpha=metrics.get("q_set_alpha", float("nan")),
            q_variance=metrics.get("q_variance", float("nan")),
            q_spread=metrics.get("q_spread", float("nan")),
            lr=lr,
            entropy_coef=metrics.get("entropy_coef"),
            inf_grad_count=0,
            dead_layers=metrics.get("dead_layers", 0),
            exploding_layers=metrics.get("exploding_layers", 0),
            layer_gradient_health=metrics.get("layer_gradient_health"),
            entropy_collapsed=metrics["entropy"] < DEFAULT_ENTROPY_COLLAPSE_THRESHOLD,
            update_time_ms=update_time_ms,  # Already validated by assert above
            early_stop_epoch=metrics.get("early_stop_epoch"),
            head_slot_entropy=head_entropies_avg.get("head_slot_entropy"),
            head_blueprint_entropy=head_entropies_avg.get("head_blueprint_entropy"),
            head_slot_grad_norm=head_grad_norms_avg.get("head_slot_grad_norm"),
            head_blueprint_grad_norm=head_grad_norms_avg.get("head_blueprint_grad_norm"),
            head_style_grad_norm=head_grad_norms_avg.get("head_style_grad_norm"),
            head_tempo_grad_norm=head_grad_norms_avg.get("head_tempo_grad_norm"),
            head_alpha_target_grad_norm=head_grad_norms_avg.get("head_alpha_target_grad_norm"),
            head_alpha_speed_grad_norm=head_grad_norms_avg.get("head_alpha_speed_grad_norm"),
            head_alpha_curve_grad_norm=head_grad_norms_avg.get("head_alpha_curve_grad_norm"),
            head_op_grad_norm=head_grad_norms_avg.get("head_op_grad_norm"),
            head_style_entropy=head_entropies_avg.get("head_style_entropy"),
            head_tempo_entropy=head_entropies_avg.get("head_tempo_entropy"),
            head_alpha_target_entropy=head_entropies_avg.get("head_alpha_target_entropy"),
            head_alpha_speed_entropy=head_entropies_avg.get("head_alpha_speed_entropy"),
            head_alpha_curve_entropy=head_entropies_avg.get("head_alpha_curve_entropy"),
            head_op_entropy=head_entropies_avg.get("head_op_entropy"),
            # Per-head ratio max (Policy V2 - multi-head ratio explosion detection)
            head_slot_ratio_max=metrics.get("head_slot_ratio_max", 1.0),
            head_blueprint_ratio_max=metrics.get("head_blueprint_ratio_max", 1.0),
            head_style_ratio_max=metrics.get("head_style_ratio_max", 1.0),
            head_tempo_ratio_max=metrics.get("head_tempo_ratio_max", 1.0),
            head_alpha_target_ratio_max=metrics.get("head_alpha_target_ratio_max", 1.0),
            head_alpha_speed_ratio_max=metrics.get("head_alpha_speed_ratio_max", 1.0),
            head_alpha_curve_ratio_max=metrics.get("head_alpha_curve_ratio_max", 1.0),
            head_op_ratio_max=metrics.get("head_op_ratio_max", 1.0),
            joint_ratio_max=metrics.get("joint_ratio_max", 1.0),
            # Per-head NaN/Inf flags (for indicator lights)
            head_nan_detected=metrics.get("head_nan_detected"),
            head_inf_detected=metrics.get("head_inf_detected"),
            # Gradient quality metrics (per DRL expert)
            clip_fraction_positive=metrics.get("clip_fraction_positive", 0.0),
            clip_fraction_negative=metrics.get("clip_fraction_negative", 0.0),
            gradient_cv=metrics.get("gradient_cv", 0.0),
            # Infrastructure metrics (per PyTorch expert)
            cuda_memory_allocated_gb=metrics.get("cuda_memory_allocated_gb", 0.0),
            cuda_memory_reserved_gb=metrics.get("cuda_memory_reserved_gb", 0.0),
            cuda_memory_peak_gb=metrics.get("cuda_memory_peak_gb", 0.0),
            cuda_memory_fragmentation=metrics.get("cuda_memory_fragmentation", 0.0),
            inner_epoch=epoch,
            batch=batch_idx + 1,
            # BUG FIX: Track actual PPO update count (inner_epoch was misleading)
            ppo_updates_count=metrics["ppo_updates_count"],
        ),
        group_id=group_id,
    ))


def emit_action_distribution(
    *,
    hub: Any,
    batch_idx: int,
    episodes_completed: int,
    action_counts: dict[str, int],
    success_counts: dict[str, int],
) -> None:
    """Emit per-batch action distribution summary."""
    hub.emit(TelemetryEvent(
        event_type=TelemetryEventType.ANALYTICS_SNAPSHOT,
        epoch=episodes_completed,
        data=AnalyticsSnapshotPayload(
            kind="action_distribution",
            action_counts=dict(action_counts),
        ),
    ))


def emit_throughput(
    *,
    hub: Any,
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
        data=AnalyticsSnapshotPayload(
            kind="throughput",
            env_id=env_id,
            batch=batch_idx,
            episodes_completed=episodes_completed,
            fps=fps,
            step_time_ms=step_time_ms,
            dataloader_wait_ms=dataloader_wait_ms,
        ),
    ))


def emit_reward_summary(
    *,
    hub: Any,
    env_id: int,
    batch_idx: int,
    summary: dict[str, float],
    episodes_completed: int = 0,
) -> None:
    """Emit compact reward summary for this batch.

    Includes scaffold hindsight credit debugging fields (Phase 3.2):
    - hindsight_credit: Total credit applied (post-cap)
    - scaffold_count: Number of scaffolds that contributed
    - avg_scaffold_delay: Average epochs since scaffolding interactions
    """
    # Extract scaffold metrics for telemetry (Phase 3.2)
    scaffold_count = int(summary.get("scaffold_count", 0))
    scaffold_delay_total = summary.get("scaffold_delay_total", 0.0)
    avg_scaffold_delay = (
        scaffold_delay_total / scaffold_count if scaffold_count > 0 else None
    )
    hindsight_credit = summary.get("hindsight_credit", 0.0)

    hub.emit(TelemetryEvent(
        event_type=TelemetryEventType.ANALYTICS_SNAPSHOT,
        epoch=episodes_completed,
        data=AnalyticsSnapshotPayload(
            kind="reward_summary",
            env_id=env_id,
            batch=batch_idx,
            episodes_completed=episodes_completed,
            summary=dict(summary),
            # Scaffold hindsight credit debugging (Phase 3.2)
            hindsight_credit=hindsight_credit if hindsight_credit > 0 else None,
            scaffold_count=scaffold_count if scaffold_count > 0 else None,
            avg_scaffold_delay=avg_scaffold_delay,
        ),
    ))


def emit_mask_hit_rates(
    *,
    hub: Any,
    batch_idx: int,
    episodes_completed: int,
    mask_hits: dict[str, int],
    mask_total: dict[str, int],
) -> None:
    """Emit per-head mask hit rates for this batch."""
    hub.emit(TelemetryEvent(
        event_type=TelemetryEventType.ANALYTICS_SNAPSHOT,
        epoch=episodes_completed,
        data=AnalyticsSnapshotPayload(
            kind="mask_hit_rates",
            batch=batch_idx,
            episodes_completed=episodes_completed,
            mask_hits=dict(mask_hits),
            mask_total=dict(mask_total),
        ),
    ))


def check_performance_degradation(
    hub: Any,
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

    from esper.leyline import PerformanceDegradationPayload

    hub.emit(TelemetryEvent(
        event_type=TelemetryEventType.PERFORMANCE_DEGRADATION,
        severity="warning",
        data=PerformanceDegradationPayload(
            env_id=env_id,
            current_acc=current_acc,
            rolling_avg_acc=rolling_avg_acc,
            drop_percent=drop * 100,
            threshold_percent=degradation_threshold * 100,
            training_progress=training_progress,
        ),
    ))
    return True


def apply_slot_telemetry(
    env_state: Any,
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
    "emit_last_action",
    "emit_mask_hit_rates",
    "emit_ppo_update_event",
    "emit_reward_summary",
    "emit_throughput",
    "emit_with_env_context",
]
