from __future__ import annotations

import math
from typing import Any

import torch
import torch.amp as torch_amp

from esper.leyline import (
    AnomalyDetectedPayload,
    SeedStage,
    SlotConfig,
    TelemetryEvent,
    TelemetryEventType,
)
from esper.simic.agent import PPOAgent
from esper.simic.agent.types import PPOUpdateMetrics
from esper.simic.control import RunningMeanStd
from esper.simic.telemetry import (
    AnomalyReport,
    TelemetryConfig,
    check_numerical_stability,
    collect_per_layer_gradients,
)


def _advance_active_seed(model: Any, slot_id: str) -> bool:
    """Advance lifecycle for the active seed in the specified slot.

    Args:
        model: MorphogeneticModel instance
        slot_id: Target slot ID (e.g., "r0c0", "r0c1", "r0c2")

    Returns:
        True if the seed successfully fossilized, False otherwise.
    """
    if not model.has_active_seed_in_slot(slot_id):
        return False

    slot = model.seed_slots[slot_id]
    seed_state = slot.state
    if seed_state is None:
        return False
    current_stage = seed_state.stage

    # Tamiyo only finalizes; mechanical blending/advancement handled by Kasmina.
    # NOTE: Leyline VALID_TRANSITIONS allow HOLDING → FOSSILIZED (finalize).
    if current_stage == SeedStage.HOLDING:
        gate_result = slot.advance_stage(SeedStage.FOSSILIZED)
        if gate_result.passed:
            slot.set_alpha(1.0)
            return True
        # Gate check failure is normal; reward shaping will penalize
        return False
    return False


def _resolve_target_slot(
    slot_idx: int,
    *,
    enabled_slots: list[str],
    slot_config: SlotConfig,
) -> tuple[str, bool]:
    """Map a slot head action index to a canonical slot ID.

    slot_idx is defined in SlotConfig.slot_ids order (canonical), independent of
    the caller-provided enabled slot list order.
    """
    try:
        slot_id = slot_config.slot_id_for_index(slot_idx)
    except IndexError:
        # Out-of-range should be impossible (network head size == slot_config.num_slots),
        # but return a deterministic slot for logging and mark it invalid.
        return enabled_slots[0], False
    return slot_id, slot_id in enabled_slots


def _calculate_entropy_anneal_steps(
    entropy_anneal_episodes: int,
    n_envs: int,
    ppo_updates_per_batch: int,
) -> int:
    """Convert episode-based entropy annealing to step-based with multi-update batches."""
    if entropy_anneal_episodes <= 0:
        return 0
    if n_envs <= 0:
        raise ValueError("n_envs must be positive when computing entropy anneal steps")

    updates_per_batch = max(1, ppo_updates_per_batch)
    batches_for_anneal = math.ceil(entropy_anneal_episodes / n_envs)
    return batches_for_anneal * updates_per_batch


def _aggregate_ppo_metrics(update_metrics: list[PPOUpdateMetrics]) -> dict[str, Any]:
    """Aggregate metrics across multiple PPO updates for a single batch."""
    if not update_metrics:
        return {}

    aggregated: dict[str, Any] = {}
    keys = {k for metrics in update_metrics for k in metrics.keys()}
    for key in keys:
        values: list[Any] = [
            metrics.get(key)
            for metrics in update_metrics
            if key in metrics and metrics.get(key) is not None
        ]
        if not values:
            continue
        if key == "ratio_max" or key.endswith("_ratio_max"):
            # All ratio max fields (ratio_max, head_*_ratio_max, joint_ratio_max)
            # should take the maximum across PPO updates, not average
            aggregated[key] = max(values)
        elif key == "ratio_min":
            aggregated[key] = min(values)
        elif key == "value_min":
            aggregated[key] = min(values)
        elif key == "value_max":
            aggregated[key] = max(values)
        elif key == "value_mean":
            # Average of means across environments
            aggregated[key] = sum(values) / len(values)
        elif key == "value_std":
            # Cannot simply average std - take max as conservative approximation
            # (proper solution requires pooled variance with means, but max ensures
            # we don't underestimate variance which could mask instability)
            aggregated[key] = max(values)
        elif key == "early_stop_epoch":
            aggregated[key] = min(values)
        elif key in ("head_entropies", "head_grad_norms"):
            # Dict[head_name, List[float]] - merge lists from multiple PPO updates
            # Take max per-head value across all updates (conservative for monitoring)
            merged: dict[str, float] = {}
            for update_dict in values:
                if isinstance(update_dict, dict):
                    for head, head_values in update_dict.items():
                        if isinstance(head_values, list) and head_values:
                            max_val = max(head_values)
                            if head not in merged or max_val > merged[head]:
                                merged[head] = max_val
            # Return in format emitter expects: dict[head, list[float]]
            aggregated[key] = {h: [v] for h, v in merged.items()}
        elif isinstance(values[0], dict):
            aggregated[key] = values[0]
        else:
            aggregated[key] = sum(values) / len(values)
    return aggregated


def _run_ppo_updates(
    agent: PPOAgent,
    ppo_updates_per_batch: int,
    raw_states_for_normalizer_update: list[torch.Tensor],
    obs_normalizer: RunningMeanStd,
    use_amp: bool,
    amp_dtype: torch.dtype | None,  # Required: explicit dtype or None for no AMP
) -> dict[str, Any]:
    """Run one or more PPO updates on the current buffer and aggregate metrics."""
    # P1 FIX: RECURRENT POLICY STALENESS GUARD
    # Multiple external PPO updates with LSTM policies cause hidden state staleness:
    # Update 1 changes policy weights, but Update 2+ uses the SAME hidden states
    # from rollout collection (computed with old weights). This creates the exact
    # mismatch that recurrent_n_epochs=1 is designed to prevent.
    #
    # The agent's internal recurrent_n_epochs=1 safety is bypassed by this external loop.
    # Standard R2D2/Recurrent PPO would require burn-in (recomputing hidden states
    # for each update), which is not implemented here.
    if ppo_updates_per_batch > 1 and agent.lstm_hidden_dim > 0:
        raise ValueError(
            f"ppo_updates_per_batch={ppo_updates_per_batch} is incompatible with recurrent (LSTM) "
            f"policies. After the first update, policy weights change but hidden states remain "
            f"from the original rollout, causing gradient corruption. Use ppo_updates_per_batch=1 "
            f"for LSTM policies, or implement hidden state burn-in. See PPOAgent C4 comment."
        )

    # C5 FIX: Update observation normalizer BEFORE PPO update.
    # This ensures batch N's observations are normalized with stats that include batch N,
    # preventing the one-batch lag that compounds distribution shift over training.
    #
    # DESIGN NOTE: Normalizer updates even if PPO update subsequently fails. This is
    # intentional - the collected observations are valid environment data regardless
    # of training outcome. Reverting stats on failure would reintroduce one-batch lag
    # and the observations were already used during rollout collection anyway.
    if raw_states_for_normalizer_update:
        all_raw_states = torch.cat(raw_states_for_normalizer_update, dim=0)
        obs_normalizer.update(all_raw_states)

    update_metrics: list[PPOUpdateMetrics] = []
    buffer_cleared = False
    updates_to_run = max(1, ppo_updates_per_batch)

    for update_idx in range(updates_to_run):
        clear_buffer = update_idx == updates_to_run - 1
        if use_amp and torch.cuda.is_available() and amp_dtype is not None:
            with torch_amp.autocast(device_type="cuda", dtype=amp_dtype):  # type: ignore[attr-defined]
                metrics = agent.update(clear_buffer=clear_buffer)
        else:
            metrics = agent.update(clear_buffer=clear_buffer)
        if metrics:
            update_metrics.append(metrics)
        buffer_cleared = buffer_cleared or clear_buffer

        approx_kl = metrics.get("approx_kl") if metrics else None
        if agent.target_kl is not None and approx_kl is not None:
            # approx_kl is already aggregated to float by PPOAgent.update()
            if approx_kl > 1.5 * agent.target_kl:
                break

    if not buffer_cleared:
        agent.buffer.reset()

    aggregated_metrics = _aggregate_ppo_metrics(update_metrics)

    # BUG FIX: Track actual PPO update count (was reporting inner_epoch=150 always)
    # This tells consumers how many PPO gradient updates occurred in this batch.
    # Early stopping on KL divergence may reduce this below ppo_updates_per_batch.
    aggregated_metrics["ppo_updates_count"] = len(update_metrics)

    return aggregated_metrics


def _handle_telemetry_escalation(
    anomaly_report: AnomalyReport | None,
    telemetry_config: TelemetryConfig | None,
) -> None:
    """Escalate telemetry on anomaly."""
    if telemetry_config is None:
        return

    if (
        telemetry_config.auto_escalate_on_anomaly
        and anomaly_report is not None
        and anomaly_report.has_anomaly
    ):
        telemetry_config.escalate_temporarily()


def _emit_anomaly_diagnostics(
    hub: Any,
    anomaly_report: AnomalyReport | None,
    agent: PPOAgent,
    batch_epoch_id: int,
    batch_idx: int,
    max_epochs: int,
    total_episodes: int,
    collect_debug: bool,
    ratio_diagnostic: dict[str, Any] | None = None,
    group_id: str | None = None,
    collect_per_layer_gradients_fn=collect_per_layer_gradients,
    check_numerical_stability_fn=check_numerical_stability,
) -> None:
    """Emit anomaly telemetry, optionally with expensive diagnostics when debug is enabled."""
    if hub is None or anomaly_report is None or not anomaly_report.has_anomaly:
        return

    event_type_map = {
        "ratio_explosion": TelemetryEventType.RATIO_EXPLOSION_DETECTED,
        "ratio_collapse": TelemetryEventType.RATIO_COLLAPSE_DETECTED,
        "value_collapse": TelemetryEventType.VALUE_COLLAPSE_DETECTED,
        "numerical_instability": TelemetryEventType.NUMERICAL_INSTABILITY_DETECTED,
    }

    gradient_stats = None
    stability_report = None
    if collect_debug:
        gradient_stats = collect_per_layer_gradients_fn(agent.policy.network)
        stability_report = check_numerical_stability_fn(agent.policy.network)

    for anomaly_type in anomaly_report.anomaly_types:
        event_type = event_type_map.get(
            anomaly_type,
            TelemetryEventType.GRADIENT_ANOMALY,  # fallback
        )

        # Prepare gradient stats as tuple of dicts
        gradient_stats_tuple = None
        if collect_debug and gradient_stats is not None:
            gradient_stats_tuple = tuple(gs.to_dict() for gs in gradient_stats[:5])

        # Prepare stability report dict
        stability_dict = None
        if collect_debug and stability_report is not None:
            stability_dict = stability_report.to_dict()

        # Create typed payload
        payload = AnomalyDetectedPayload(
            anomaly_type=anomaly_type,
            episode=batch_epoch_id,
            batch=batch_idx + 1,
            inner_epoch=max_epochs,
            total_episodes=total_episodes,
            detail=anomaly_report.details.get(anomaly_type, ""),
            gradient_stats=gradient_stats_tuple,
            stability=stability_dict,
            ratio_diagnostic=ratio_diagnostic,
        )

        hub.emit(
            TelemetryEvent(
                event_type=event_type,
                epoch=batch_epoch_id,  # Anomalies detected at batch boundary
                group_id=group_id if group_id is not None else "",
                data=payload,
                severity="debug" if collect_debug else "warning",
            )
        )
