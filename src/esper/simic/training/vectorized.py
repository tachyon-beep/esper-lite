"""Vectorized multi-GPU PPO training for Tamiyo.

This module implements high-performance vectorized PPO training using:
- Multiple parallel environments
- CUDA streams for async GPU execution
- Inverted control flow (batch-first iteration)
- SharedBatchIterator: Single DataLoader serving all environments

Key Architecture:
Instead of N independent DataLoaders (N × workers = massive IPC overhead),
we use ONE SharedBatchIterator with combined batch size, then split batches
across environments. This reduces worker processes from N×M to just M.

Performance comparison (4 envs, 4 workers each):
- Old (independent): 16 worker processes, 16× IPC overhead
- New (shared): 4 worker processes, 1× IPC overhead

Usage:
    from esper.simic.training import train_ppo_vectorized

    agent, history = train_ppo_vectorized(
        n_episodes=100,
        n_envs=4,
        devices=["cuda:0", "cuda:1"],
    )
"""

from __future__ import annotations

import logging
import math
import random
import threading
import time
import warnings
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable

_logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as torch_amp

# NOTE: get_task_spec imported lazily inside train_ppo_vectorized to avoid circular import:
#   runtime -> simic.rewards -> simic -> simic.training -> vectorized -> runtime
from esper.utils.data import SharedBatchIterator
from esper.leyline import (
    SeedStage,
    SeedTelemetry,
    TelemetryEvent,
    TelemetryEventType,
    SlotConfig,
    DEFAULT_GAMMA,
    DEFAULT_EPISODE_LENGTH,
    DEFAULT_LSTM_HIDDEN_DIM,
    DEFAULT_N_ENVS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_CLIP_RATIO,
    DEFAULT_ENTROPY_COEF,
    DEFAULT_ENTROPY_COEF_MIN,
    DEFAULT_GOVERNOR_SENSITIVITY,
    DEFAULT_GOVERNOR_ABSOLUTE_THRESHOLD,
    DEFAULT_GOVERNOR_DEATH_PENALTY,
    DEFAULT_GOVERNOR_HISTORY_WINDOW,
    DEFAULT_MIN_PANICS_BEFORE_ROLLBACK,
    HEAD_NAMES,
)
from esper.leyline.factored_actions import (
    FactoredAction,
    LifecycleOp,
    OP_NAMES,
    BLUEPRINT_IDS,
    BLEND_IDS,
    TempoAction,
    TEMPO_TO_EPOCHS,
    OP_WAIT,
    OP_GERMINATE,
    OP_CULL,
    OP_FOSSILIZE,
)
from esper.tamiyo.policy.action_masks import build_slot_states, compute_action_masks
from esper.leyline.slot_id import validate_slot_ids
from esper.simic.telemetry import (
    AnomalyDetector,
    AnomalyReport,
    collect_per_layer_gradients,
    check_numerical_stability,
    LayerGradientStats,
    collect_host_gradients_async,
    collect_seed_gradients_only_async,
    materialize_dual_grad_stats,
    TelemetryConfig,
    compute_lstm_health,  # P4-8
    GradientEMATracker,  # P4-9
)
from esper.simic.control import RunningMeanStd, RewardNormalizer
from esper.tamiyo.policy.features import MULTISLOT_FEATURE_SIZE, get_feature_size
from esper.simic.agent import PPOAgent, signals_to_features
from esper.simic.rewards import (
    compute_reward,
    compute_loss_reward,
    RewardMode,
    RewardFamily,
    ContributionRewardConfig,
    SeedInfo,
)
from esper.leyline import DEFAULT_MIN_FOSSILIZE_CONTRIBUTION
from esper.nissa import get_hub, BlueprintAnalytics, DirectoryOutput
from esper.tolaria import TolariaGovernor
from esper.karn.health import HealthMonitor
from esper.simic.attribution import CounterfactualHelper
from esper.simic.telemetry.emitters import (
    emit_with_env_context,
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
from .parallel_env_state import ParallelEnvState


# =============================================================================
# GPU-Accurate Timing (P4-1)
# =============================================================================


class CUDATimer:
    """GPU-accurate timing using CUDA events.

    CUDA events measure actual GPU kernel execution time, not CPU-side overhead.
    This is critical for async CUDA operations where time.perf_counter() would
    only measure the time to enqueue operations, not their actual execution.

    Falls back to CPU timing when CUDA unavailable.
    """

    def __init__(self, device: str = "cuda"):
        self.use_cuda = device.startswith("cuda") and torch.cuda.is_available()
        if self.use_cuda:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
        else:
            self.start_time = 0.0

    def start(self) -> None:
        """Record start time."""
        if self.use_cuda:
            self.start_event.record()
        else:
            self.start_time = time.perf_counter()

    def stop(self) -> float:
        """Record end time and return elapsed milliseconds.

        For CUDA: synchronizes to ensure GPU work is complete before measuring.
        """
        if self.use_cuda:
            self.end_event.record()
            self.end_event.synchronize()
            return self.start_event.elapsed_time(self.end_event)
        else:
            return (time.perf_counter() - self.start_time) * 1000.0


# =============================================================================
# Seed Management Helpers
# =============================================================================


def _advance_active_seed(model, slot_id: str) -> bool:
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
    current_stage = seed_state.stage

    # Tamiyo only finalizes; mechanical blending/advancement handled by Kasmina.
    # NOTE: Leyline VALID_TRANSITIONS only allow PROBATIONARY → FOSSILIZED.
    if current_stage == SeedStage.PROBATIONARY:
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


def _aggregate_ppo_metrics(update_metrics: list[dict]) -> dict:
    """Aggregate metrics across multiple PPO updates for a single batch."""
    if not update_metrics:
        return {}

    aggregated: dict = {}
    keys = {k for metrics in update_metrics for k in metrics.keys()}
    for key in keys:
        values = [metrics[key] for metrics in update_metrics if key in metrics and metrics[key] is not None]
        if not values:
            continue
        if key == "ratio_max":
            aggregated[key] = max(values)
        elif key == "ratio_min":
            aggregated[key] = min(values)
        elif key == "early_stop_epoch":
            aggregated[key] = min(values)
        elif key == "head_entropies":
            # Aggregate per-head entropy: concatenate lists from multiple updates
            aggregated[key] = values[0]  # Just use first update's head_entropies
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
) -> dict:
    """Run one or more PPO updates on the current buffer and aggregate metrics."""
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

    update_metrics: list[dict] = []
    buffer_cleared = False
    updates_to_run = max(1, ppo_updates_per_batch)

    for update_idx in range(updates_to_run):
        clear_buffer = update_idx == updates_to_run - 1
        if use_amp and torch.cuda.is_available():
            with torch_amp.autocast(device_type="cuda", dtype=torch.float16):  # type: ignore[call-arg]
                metrics = agent.update(clear_buffer=clear_buffer)
        else:
            metrics = agent.update(clear_buffer=clear_buffer)
        if metrics:
            update_metrics.append(metrics)
        buffer_cleared = buffer_cleared or clear_buffer

        approx_kl = metrics.get("approx_kl") if metrics else None
        if agent.target_kl is not None and approx_kl is not None:
            if approx_kl > 1.5 * agent.target_kl:
                break

    if not buffer_cleared:
        agent.buffer.reset()

    aggregated_metrics = _aggregate_ppo_metrics(update_metrics)

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
    ratio_diagnostic: dict | None = None,
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
        gradient_stats = collect_per_layer_gradients(agent.network)
        stability_report = check_numerical_stability(agent.network)

    for anomaly_type in anomaly_report.anomaly_types:
        event_type = event_type_map.get(
            anomaly_type,
            TelemetryEventType.GRADIENT_ANOMALY,  # fallback
        )

        data = {
            "episode": batch_epoch_id,
            "batch": batch_idx + 1,
            "episodes_completed": batch_epoch_id,
            "inner_epoch": max_epochs,
            "detail": anomaly_report.details.get(anomaly_type, ""),
            "total_episodes": total_episodes,
        }

        if collect_debug:
            # gradient_stats/stability_report set when collect_debug=True above
            data["gradient_stats"] = [gs.to_dict() for gs in gradient_stats[:5]]  # type: ignore[index,union-attr]
            data["stability"] = stability_report.to_dict()  # type: ignore[union-attr]
        if ratio_diagnostic is not None:
            data["ratio_diagnostic"] = ratio_diagnostic

        hub.emit(TelemetryEvent(
            event_type=event_type,
            epoch=batch_epoch_id,  # Anomalies detected at batch boundary
            data=data,
            severity="debug" if collect_debug else "warning",
        ))


# =============================================================================
# Vectorized PPO Training
# =============================================================================

def _compute_batched_bootstrap_values(
    agent: PPOAgent,
    post_action_data: list[dict[str, Any]],
    obs_normalizer: RunningMeanStd,
    device: str,
) -> list[float]:
    """Compute bootstrap values for all truncated envs in single forward pass.

    Args:
        agent: PPO agent with network
        post_action_data: List of dicts with keys:
            - features: list[float] - post-action observation features
            - hidden: tuple[Tensor, Tensor] - LSTM hidden state
            - masks: dict[str, Tensor] - action masks for each head
        obs_normalizer: Observation normalizer
        device: Device string

    Returns:
        List of bootstrap values (one per entry in post_action_data)
    """
    if not post_action_data:
        return []

    # Stack all features
    features_batch = torch.tensor(
        [d["features"] for d in post_action_data],
        dtype=torch.float32,
        device=device,
    )
    features_normalized = obs_normalizer.normalize(features_batch)

    # Stack hidden states: each is [layers, 1, hidden_dim], need [layers, batch, hidden_dim]
    try:
        hidden_h = torch.cat([d["hidden"][0] for d in post_action_data], dim=1)
        hidden_c = torch.cat([d["hidden"][1] for d in post_action_data], dim=1)
    except RuntimeError as e:
        shapes_h = [d["hidden"][0].shape for d in post_action_data]
        shapes_c = [d["hidden"][1].shape for d in post_action_data]
        raise RuntimeError(
            f"LSTM hidden state shape mismatch during bootstrap batching. "
            f"Expected all states to be [layers, 1, hidden_dim]. "
            f"Got h shapes: {shapes_h}, c shapes: {shapes_c}. "
            f"Original error: {e}"
        ) from e

    # Stack masks
    masks_batch = {
        key: torch.stack([d["masks"][key] for d in post_action_data])
        for key in HEAD_NAMES
    }

    # Single forward pass
    with torch.inference_mode():
        result = agent.network.get_action(
            features_normalized,
            hidden=(hidden_h, hidden_c),
            slot_mask=masks_batch["slot"],
            blueprint_mask=masks_batch["blueprint"],
            blend_mask=masks_batch["blend"],
            tempo_mask=masks_batch["tempo"],
            op_mask=masks_batch["op"],
            deterministic=True,
        )

    return result.values.tolist()


def train_ppo_vectorized(
    n_episodes: int = 100,
    n_envs: int = DEFAULT_N_ENVS,
    max_epochs: int = DEFAULT_EPISODE_LENGTH,
    device: str = "cuda:0",
    devices: list[str] | None = None,
    task: str = "cifar10",
    use_telemetry: bool = True,
    lr: float = DEFAULT_LEARNING_RATE,
    clip_ratio: float = DEFAULT_CLIP_RATIO,
    entropy_coef: float = DEFAULT_ENTROPY_COEF,  # From leyline
    entropy_coef_start: float | None = None,
    entropy_coef_end: float | None = None,
    entropy_coef_min: float = DEFAULT_ENTROPY_COEF_MIN,  # From leyline
    adaptive_entropy_floor: bool = False,
    entropy_anneal_episodes: int = 0,
    gamma: float = DEFAULT_GAMMA,
    ppo_updates_per_batch: int = 1,
    save_path: str | None = None,
    resume_path: str | None = None,
    seed: int = 42,
    num_workers: int | None = None,
    gpu_preload: bool = False,
    amp: bool = False,
    lstm_hidden_dim: int = DEFAULT_LSTM_HIDDEN_DIM,
    chunk_length: int = DEFAULT_EPISODE_LENGTH,  # Must match max_epochs (from leyline)
    telemetry_config: "TelemetryConfig | None" = None,
    telemetry_lifecycle_only: bool = False,
    plateau_threshold: float = 0.5,
    improvement_threshold: float = 2.0,
    slots: list[str] | None = None,
    max_seeds: int | None = None,
    reward_mode: str = "shaped",
    param_budget: int = 500_000,
    param_penalty_weight: float = 0.1,
    sparse_reward_scale: float = 1.0,
    reward_family: str = "contribution",
    ab_reward_modes: list[str] | None = None,
    quiet_analytics: bool = False,
    telemetry_dir: str | None = None,
    ready_event: "threading.Event | None" = None,
) -> tuple[PPOAgent, list[dict]]:
    """Train PPO with vectorized environments using INVERTED CONTROL FLOW.

    Key architecture: Instead of iterating environments then dataloaders,
    we iterate dataloader batches FIRST, then run all environments in parallel
    using CUDA streams. This ensures both GPUs are working simultaneously.

    Args:
        n_episodes: Total episodes to train
        n_envs: Number of parallel environments
        max_epochs: Max epochs per episode (RL timesteps per episode)
        device: Device for policy network
        devices: List of devices for environments (e.g., ["cuda:0", "cuda:1"])
        use_telemetry: Whether to use telemetry features
        lr: Learning rate
        clip_ratio: PPO clip ratio
        entropy_coef: Entropy coefficient
        gamma: Discount factor
        ppo_updates_per_batch: Number of PPO updates per batch of episodes.
            Higher values improve sample efficiency but risk policy divergence.
            With KL early stopping enabled, values of 2-4 are often safe.
            Default: 1 (standard PPO behavior)
        save_path: Optional path to save model
        resume_path: Optional path to resume from checkpoint
        seed: Random seed for reproducibility
        plateau_threshold: Rolling average delta threshold below which training is considered
            plateaued (emits PLATEAU_DETECTED event). Compares current vs previous batch's
            rolling average. Scale-dependent: adjust for accuracy scales (e.g., 0-1 vs 0-100).
        improvement_threshold: Rolling average delta threshold above which training shows
            significant improvement/degradation (emits IMPROVEMENT_DETECTED/DEGRADATION_DETECTED).
            Events align with displayed rolling_avg_accuracy trend.

    Returns:
        Tuple of (trained_agent, training_history)
    """
    from esper.tolaria import create_model
    from esper.tamiyo import SignalTracker

    def _parse_device(device_str: str) -> torch.device:
        """Parse a device string with an actionable error on failure."""
        try:
            return torch.device(device_str)
        except Exception as exc:  # pragma: no cover - torch raises varied exceptions
            raise ValueError(f"Invalid device '{device_str}': {exc}") from exc

    def _validate_cuda_device(
        device_str: str, *, require_explicit_index: bool
    ) -> None:
        """Fail fast on invalid CUDA device requests."""
        dev = _parse_device(device_str)
        if dev.type != "cuda":
            return

        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device '{device_str}' requested but CUDA is not available. "
                "Use CPU devices or install CUDA drivers."
            )

        if require_explicit_index and dev.index is None:
            raise ValueError(
                f"CUDA device '{device_str}' must include an explicit index like 'cuda:0'."
            )

        if dev.index is None:
            return

        available = torch.cuda.device_count()
        if dev.index >= available:
            raise RuntimeError(
                f"CUDA device '{device_str}' requested but only {available} device(s) are available."
            )

    if not slots:
        raise ValueError("slots parameter is required and cannot be empty")

    # Get task spec early (needed for model creation to derive slot_config)
    # Lazy import to avoid circular dependency
    from esper.runtime import get_task_spec
    task_spec = get_task_spec(task)
    ActionEnum = task_spec.action_enum

    # Derive slot_config from host's injection specs
    # Create a temporary model to query the host's injection topology
    temp_device = "cpu"  # Use CPU for temp model to avoid GPU allocation
    temp_model = create_model(task=task_spec, device=temp_device, slots=slots)
    slot_config = SlotConfig.from_specs(temp_model.host.injection_specs())
    # Calculate host_params while we have the model (constant across all envs)
    host_params_baseline = sum(p.numel() for p in temp_model.host.parameters() if p.requires_grad)
    del temp_model  # Free memory immediately

    # Compute effective seed limit
    # max_seeds=None means unlimited (use 0 to indicate no limit)
    effective_max_seeds = max_seeds if max_seeds is not None else 0

    if devices is None:
        devices = [device]

    if not devices:
        raise ValueError("devices must be a non-empty list")

    # Policy device may be specified as "cuda" without an index, but env devices must be explicit.
    _validate_cuda_device(device, require_explicit_index=False)
    for env_device in devices:
        _validate_cuda_device(env_device, require_explicit_index=True)

    if len(devices) > n_envs:
        raise ValueError(
            f"n_envs={n_envs} must be >= len(devices)={len(devices)} so every requested device "
            "runs at least one environment."
        )

    # Create reward config based on mode
    reward_family_enum = RewardFamily(reward_family)
    reward_mode_enum = RewardMode(reward_mode)
    if reward_family_enum == RewardFamily.LOSS and reward_mode_enum != RewardMode.SHAPED:
        raise ValueError("reward_mode applies only to contribution rewards. Use default when reward_family=loss.")

    reward_config = ContributionRewardConfig(
        reward_mode=reward_mode_enum,
        param_budget=param_budget,
        param_penalty_weight=param_penalty_weight,
        sparse_reward_scale=sparse_reward_scale,
    )
    loss_reward_config = task_spec.loss_reward_config

    # Per-environment reward configs for A/B testing
    if ab_reward_modes is not None:
        if len(ab_reward_modes) != n_envs:
            raise ValueError(
                f"ab_reward_modes length ({len(ab_reward_modes)}) must match n_envs ({n_envs})"
            )
        env_reward_configs = []
        for env_idx, mode_str in enumerate(ab_reward_modes):
            env_mode = RewardMode(mode_str)
            env_config = ContributionRewardConfig(
                reward_mode=env_mode,
                param_budget=param_budget,
                param_penalty_weight=param_penalty_weight,
                sparse_reward_scale=sparse_reward_scale,
            )
            env_reward_configs.append(env_config)
        _logger.info(
            "A/B testing enabled: %s",
            {mode: ab_reward_modes.count(mode) for mode in set(ab_reward_modes)}
        )
    else:
        env_reward_configs = [reward_config] * n_envs

    # Map environments to devices in round-robin (needed for SharedBatchIterator)
    env_device_map = [devices[i % len(devices)] for i in range(n_envs)]

    # DataLoader settings (used for SharedBatchIterator + diagnostics).
    batch_size_per_env = task_spec.dataloader_defaults.get("batch_size", 128)
    if task_spec.name == "cifar10":
        batch_size_per_env = 512  # High-throughput setting for CIFAR
    effective_workers = num_workers if num_workers is not None else 4

    # State dimension: base features (dynamic based on slot count) + telemetry features when enabled
    # For 3 slots: 23 base + 3*9 slot features = 50, plus 3*10 telemetry if enabled
    base_feature_size = get_feature_size(slot_config)
    telemetry_size = slot_config.num_slots * SeedTelemetry.feature_dim() if use_telemetry else 0
    state_dim = base_feature_size + telemetry_size

    # Use EMA momentum for stable normalization during long training runs
    # (prevents distribution shift that can break PPO ratio calculations)
    obs_normalizer = RunningMeanStd((state_dim,), device=device, momentum=0.99)

    # Reward normalizer for critic stability (prevents value loss explosion)
    # Essential after ransomware fix where reward magnitudes changed significantly
    reward_normalizer = RewardNormalizer(clip=10.0)

    # Convert episode-based annealing to step-based (respecting multi-update batches)
    entropy_anneal_steps = _calculate_entropy_anneal_steps(
        entropy_anneal_episodes=entropy_anneal_episodes,
        n_envs=n_envs,
        ppo_updates_per_batch=ppo_updates_per_batch,
    )

    # ==========================================================================
    # Blueprint Analytics + Nissa Hub Wiring
    # ==========================================================================
    hub = get_hub()
    analytics = BlueprintAnalytics(quiet=quiet_analytics)
    hub.add_backend(analytics)

    # Optional file-based telemetry logging
    dir_output = None
    if telemetry_dir and use_telemetry:
        dir_output = DirectoryOutput(telemetry_dir)
        hub.add_backend(dir_output)
        _logger.info(f"Telemetry logging to: {telemetry_dir}")

    # Mapping diagnostics: required for multi-GPU sign-off.
    unique_env_devices = list(dict.fromkeys(devices))
    env_device_counts = {dev: 0 for dev in unique_env_devices}
    for mapped_device in env_device_map:
        env_device_counts[mapped_device] += 1

    entropy_anneal_summary = None
    if entropy_anneal_episodes > 0:
        entropy_anneal_summary = {
            "start": entropy_coef_start or entropy_coef,
            "end": entropy_coef_end or entropy_coef,
            "episodes": entropy_anneal_episodes,
            "steps": entropy_anneal_steps,
        }

    dataloader_summary = {
        "mode": "gpu_preload" if gpu_preload else "shared_batch_iterator",
        "batch_size_per_env": batch_size_per_env,
        "num_workers": None if gpu_preload else effective_workers,
        "pin_memory": not gpu_preload,
    }

    hub.emit(TelemetryEvent(
        event_type=TelemetryEventType.TRAINING_STARTED,
        message=(
            f"PPO vectorized training initialized: policy_device={device}, "
            f"env_device_counts={env_device_counts}"
        ),
        data={
            "episode_id": f"ppo_{seed}_{n_episodes}ep",
            "seed": seed,
            "task": task,
            "topology": task_spec.topology,
            "task_type": task_spec.task_type,
            "reward_mode": reward_mode,
            "max_epochs": max_epochs,
            "n_envs": n_envs,
            "n_episodes": n_episodes,
            "use_telemetry": use_telemetry,
            "state_dim": state_dim,
            "policy_device": device,
            "env_devices": list(devices),
            "env_device_counts": env_device_counts,
            "env_device_map_strategy": "round_robin",
            "resume_path": str(resume_path) if resume_path else None,
            "lr": lr,
            "clip_ratio": clip_ratio,
            "entropy_coef": entropy_coef,
            "entropy_anneal": entropy_anneal_summary,
            "gpu_preload": gpu_preload,
            "dataloader": dataloader_summary,
            "param_budget": param_budget,
            "param_penalty_weight": param_penalty_weight,
            "sparse_reward_scale": sparse_reward_scale,
            "host_params": host_params_baseline,
            "slot_ids": list(slot_config.slot_ids),
        },
    ))

    ops_telemetry_enabled = use_telemetry and (
        telemetry_config is None or telemetry_config.should_collect("ops_normal")
    )
    if telemetry_lifecycle_only and not ops_telemetry_enabled:
        hub.emit(TelemetryEvent(
            event_type=TelemetryEventType.ANALYTICS_SNAPSHOT,
            severity="warning",
            message="Ops telemetry disabled; emitting lifecycle-only seed telemetry",
            data={
                "telemetry_lifecycle_only": True,
                "use_telemetry": use_telemetry,
                "telemetry_level": telemetry_config.level.name if telemetry_config is not None else None,
            },
        ))

    # Create SharedBatchIterator - single DataLoader serving all environments
    # This eliminates the N×M worker overhead from N independent DataLoaders
    if gpu_preload:
        # GPU-resident data loading: 8x faster than CPU DataLoader workers
        # SharedGPUBatchIterator: ONE DataLoader per device, splits batches across envs
        # CRITICAL: Multiple DataLoaders sharing cached GPU tensors causes data corruption
        # (race condition when concurrent iterators access same tensor storage)
        from esper.utils.data import SharedGPUBatchIterator

        # Create generator for reproducible shuffling
        gen = torch.Generator()
        gen.manual_seed(seed)

        # Create shared GPU iterators for train and test
        shared_train_iter = SharedGPUBatchIterator(
            batch_size_per_env=512,
            n_envs=n_envs,
            env_devices=env_device_map,
            shuffle=True,
            generator=gen,
            is_train=True,
        )

        shared_test_iter = SharedGPUBatchIterator(
            batch_size_per_env=512,
            n_envs=n_envs,
            env_devices=env_device_map,
            shuffle=False,
            generator=gen,
            is_train=False,
        )

        num_train_batches = len(shared_train_iter)
        num_test_batches = len(shared_test_iter)
        env_dataloaders = None  # Not using per-env dataloaders (uses shared iterator)
    else:
        # SharedBatchIterator: Single DataLoader with combined batch size
        # Splits batches across environments and moves to correct devices
        # Get raw datasets from task spec
        train_dataset, test_dataset = task_spec.get_datasets()

        # Create generator for reproducible shuffling
        gen = torch.Generator()
        gen.manual_seed(seed)

        # Create shared iterators for train and test
        shared_train_iter = SharedBatchIterator(
            dataset=train_dataset,
            batch_size_per_env=batch_size_per_env,
            n_envs=n_envs,
            env_devices=env_device_map,
            num_workers=effective_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            generator=gen,
        )

        # Test iterator: num_workers=0 because validation is infrequent (once per epoch)
        # No point spawning persistent workers for ~2% of total iteration time
        shared_test_iter = SharedBatchIterator(
            dataset=test_dataset,
            batch_size_per_env=batch_size_per_env,
            n_envs=n_envs,
            env_devices=env_device_map,
            num_workers=0,  # Validation is fast enough without workers
            shuffle=False,
            pin_memory=True,
            drop_last=False,  # BUG-009 fix: include all validation samples
        )

        num_train_batches = len(shared_train_iter)
        num_test_batches = len(shared_test_iter)
        env_dataloaders = None  # Not using per-env dataloaders

        # CRITICAL: Warm up DataLoader to spawn persistent workers BEFORE TUI starts.
        # With persistent_workers=True, workers are spawned on first iter() and reused.
        # If we don't do this here, workers spawn after Textual takes over the terminal,
        # which corrupts file descriptors and causes "bad value(s) in fds_to_keep" errors.
        if effective_workers > 0:
            _warmup_iter = iter(shared_train_iter)
            # Fetch one batch to ensure workers are fully initialized
            try:
                _warmup_batch = next(_warmup_iter)
                del _warmup_batch  # Free memory
            except StopIteration:
                pass  # Empty dataset (shouldn't happen)
            del _warmup_iter  # Iterator done, but workers persist

    # Signal that DataLoaders are ready (workers spawned) - TUI can now safely start
    if ready_event is not None:
        ready_event.set()

    def loss_and_correct(outputs: torch.Tensor, targets: torch.Tensor, criterion: nn.Module):
        """Compute loss and correct counts for classification or LM."""
        if task_spec.task_type == "lm":
            vocab = outputs.size(-1)
            loss = criterion(outputs.view(-1, vocab), targets.view(-1))
            predicted = outputs.argmax(dim=-1)
            correct = predicted.eq(targets).sum()
            total = targets.numel()
        else:
            loss = criterion(outputs, targets)
            _, predicted = outputs.max(1)
            correct = predicted.eq(targets).sum()
            total = targets.size(0)
        return loss, correct, total

    # AMP enabled gate - actual GradScaler created per-env in ParallelEnvState
    # to avoid stream race conditions (GradScaler internal state is not stream-safe)
    amp_enabled = amp and torch.cuda.is_available()

    # Create or resume PPO agent
    start_episode = 0
    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=True)
        agent = PPOAgent.load(resume_path, device=device)

        # Restore observation normalizer state
        metadata = checkpoint.get('metadata', {})
        if 'obs_normalizer_mean' in metadata:
            # Create tensors directly on target device to avoid CPU->GPU transfer
            obs_normalizer.mean = torch.tensor(metadata['obs_normalizer_mean'], device=device)
            obs_normalizer.var = torch.tensor(metadata['obs_normalizer_var'], device=device)
            obs_normalizer._device = device
            # Restore count for correct Welford/EMA continuation
            if 'obs_normalizer_count' in metadata:
                obs_normalizer.count = torch.tensor(metadata['obs_normalizer_count'], device=device)
            # Restore momentum (critical for EMA mode - affects normalization dynamics)
            if 'obs_normalizer_momentum' in metadata:
                obs_normalizer.momentum = metadata['obs_normalizer_momentum']

        # Restore reward normalizer state (P1-6 fix: prevents value function instability on resume)
        if 'reward_normalizer_mean' in metadata:
            reward_normalizer.mean = metadata['reward_normalizer_mean']
            reward_normalizer.m2 = metadata['reward_normalizer_m2']
            reward_normalizer.count = metadata['reward_normalizer_count']

        # Calculate starting episode from checkpoint
        if 'n_episodes' in metadata:
            start_episode = metadata['n_episodes']

        # Emit telemetry for checkpoint resume
        hub.emit(TelemetryEvent(
            event_type=TelemetryEventType.CHECKPOINT_LOADED,
            message=f"Resumed from checkpoint: {resume_path}",
            data={
                "path": str(resume_path),
                "start_episode": start_episode,
                "obs_normalizer_momentum": obs_normalizer.momentum if 'obs_normalizer_mean' in metadata else None,
            },
        ))
    else:
        agent = PPOAgent(
            state_dim=state_dim,
            action_dim=len(ActionEnum),
            lr=lr,
            clip_ratio=clip_ratio,
            entropy_coef=entropy_coef,
            entropy_coef_start=entropy_coef_start,
            entropy_coef_end=entropy_coef_end,
            entropy_coef_min=entropy_coef_min,
            adaptive_entropy_floor=adaptive_entropy_floor,
            entropy_anneal_steps=entropy_anneal_steps,
            gamma=gamma,
            device=device,
            lstm_hidden_dim=lstm_hidden_dim,
            chunk_length=chunk_length,
            # Buffer dimensions must match training loop parameters
            num_envs=n_envs,
            max_steps_per_env=max_epochs,
            slot_config=slot_config,
        )

    # Initialize anomaly detector for automatic diagnostics
    anomaly_detector = AnomalyDetector()

    def make_telemetry_callback(env_idx: int, env_device: str):
        """Create callback that injects env_id before emitting to hub."""

        def callback(event: TelemetryEvent):
            emit_with_env_context(hub, env_idx, env_device, event)

        return callback

    def configure_slot_telemetry(
        env_state: ParallelEnvState,
        *,
        inner_epoch: int | None = None,
        global_epoch: int | None = None,
    ) -> None:
        """Configure slot telemetry/fast_mode based on current telemetry level."""
        ops_telemetry_enabled = use_telemetry and (
            telemetry_config is None or telemetry_config.should_collect("ops_normal")
        )
        apply_slot_telemetry(
            env_state,
            ops_telemetry_enabled=ops_telemetry_enabled,
            lifecycle_only=telemetry_lifecycle_only,
            inner_epoch=inner_epoch,
            global_epoch=global_epoch,
        )

    def create_env_state(env_idx: int, base_seed: int) -> ParallelEnvState:
        """Create environment state with CUDA stream.

        DataLoaders are now shared via SharedBatchIterator, not per-env.
        """
        env_device = env_device_map[env_idx]
        torch.manual_seed(base_seed + env_idx * 1000)
        random.seed(base_seed + env_idx * 1000)

        model = create_model(task=task_spec, device=env_device, slots=slots)

        telemetry_cb = make_telemetry_callback(env_idx, env_device)
        for slot in model.seed_slots.values():
            slot.on_telemetry = telemetry_cb
            # fast_mode toggled per epoch via apply_slot_telemetry (telemetry-enabled by default)
            slot.fast_mode = False
            # Incubator mode gradient isolation: detach host input into the seed path so
            # host gradients remain identical to the host-only model while the seed
            # trickle-learns via STE in TRAINING. The host optimizer still steps
            # every batch; isolation only affects gradients through the seed branch.
            slot.isolate_gradients = True

        # Set host_params baseline for scoreboard via Nissa analytics
        host_params = sum(p.numel() for p in model.host.parameters() if p.requires_grad)
        analytics.set_host_params(env_idx, host_params)

        host_optimizer = torch.optim.SGD(
            model.get_host_parameters(), lr=task_spec.host_lr, momentum=0.9, weight_decay=5e-4
        )

        # Create CUDA stream for this environment
        env_device_obj = torch.device(env_device)
        stream = torch.cuda.Stream(device=env_device_obj) if env_device_obj.type == "cuda" else None

        # Per-env AMP scaler to avoid stream race conditions (GradScaler state is not stream-safe)
        # Use new torch.amp.GradScaler API (torch.cuda.amp.GradScaler deprecated in PyTorch 2.4+)
        env_scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if env_device_obj.type == "cuda" else None

        # Pre-compute autocast decision for hot path (avoids per-batch device type checks)
        autocast_enabled = amp_enabled and env_device_obj.type == "cuda"

        # Determine random guess loss for lobotomy detection
        random_guess_loss = None
        if task_spec.task_type == "classification" and task_spec.num_classes:
            random_guess_loss = math.log(task_spec.num_classes)
        elif task_spec.task_type == "lm" and task_spec.vocab_size:
            random_guess_loss = math.log(task_spec.vocab_size)

        # Create Governor for fail-safe watchdog
        # Conservative settings to avoid false positives during seed blending:
        # - sensitivity=6.0: 6-sigma is very rare for Gaussian
        # - history_window=20: longer window smooths blending transients
        # - min_panics=3: require 3 consecutive anomalies before rollback
        governor = TolariaGovernor(
            model=model,
            sensitivity=DEFAULT_GOVERNOR_SENSITIVITY,
            absolute_threshold=DEFAULT_GOVERNOR_ABSOLUTE_THRESHOLD,
            death_penalty=DEFAULT_GOVERNOR_DEATH_PENALTY,
            history_window=DEFAULT_GOVERNOR_HISTORY_WINDOW,
            min_panics_before_rollback=DEFAULT_MIN_PANICS_BEFORE_ROLLBACK,
            random_guess_loss=random_guess_loss,
        )
        governor.snapshot()  # Ensure rollback is always possible before first panic

        # Create HealthMonitor for system health tracking
        # Uses existing telemetry_cb from create_env_state scope
        health_monitor = HealthMonitor(
            store=None,  # TelemetryStore integration deferred
            emit_callback=telemetry_cb,  # Same callback as slots
        ) if use_telemetry else None

        # Create CounterfactualHelper for Shapley value analysis at episode end
        counterfactual_helper = CounterfactualHelper(
            strategy="auto",  # Full factorial for <=4 slots, Shapley sampling otherwise
            shapley_samples=20,
            emit_events=use_telemetry,
        ) if use_telemetry else None

        env_state = ParallelEnvState(
            model=model,
            host_optimizer=host_optimizer,
            signal_tracker=SignalTracker(env_id=env_idx),
            governor=governor,
            health_monitor=health_monitor,
            counterfactual_helper=counterfactual_helper,
            env_device=env_device,
            stream=stream,
            scaler=env_scaler,
            seeds_created=0,
            episode_rewards=[],
            action_enum=ActionEnum,
            telemetry_cb=telemetry_cb,
            autocast_enabled=autocast_enabled,
        )
        # Pre-allocate accumulators to avoid per-epoch allocation churn
        env_state.init_accumulators(slots)
        configure_slot_telemetry(env_state)
        return env_state

    @torch.compiler.disable
    def _collect_gradient_telemetry_for_batch(
        model: "HostWithSeeds",
        slots_with_active_seeds: list[str],
        env_dev: str,
    ) -> dict[str, dict[str, Any]] | None:
        """Collect gradient telemetry for all active slots.

        Isolated from torch.compile to prevent graph breaks from
        data-dependent slot iteration and conditional logic.

        Args:
            model: The HostWithSeeds model
            slots_with_active_seeds: Pre-filtered list of slots with active seeds
            env_dev: Device string
        """
        from esper.leyline import SeedStage
        from esper.simic.telemetry.gradient_collector import (
            collect_host_gradients_async,
            collect_seed_gradients_only_async,
        )

        slots_needing_grad_telemetry = []
        for slot_id in slots_with_active_seeds:
            # Already filtered to active slots via cache
            seed_state = model.seed_slots[slot_id].state
            if seed_state and seed_state.stage in (SeedStage.TRAINING, SeedStage.BLENDING):
                slots_needing_grad_telemetry.append(slot_id)

        if not slots_needing_grad_telemetry:
            return None

        # Compute host gradient stats ONCE (expensive), then reuse for each seed
        host_stats = collect_host_gradients_async(
            model.get_host_parameters(),
            device=env_dev,
        )

        grad_stats_by_slot = {}
        for slot_id in slots_needing_grad_telemetry:
            seed_stats = collect_seed_gradients_only_async(
                model.get_seed_parameters(slot_id),
                device=env_dev,
            )
            grad_stats_by_slot[slot_id] = {
                **host_stats,
                **seed_stats,
            }

        return grad_stats_by_slot

    def process_train_batch(
        env_state: ParallelEnvState,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        criterion: nn.Module,
        use_telemetry: bool = False,
        slots: list[str] | None = None,
        use_amp: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, int, dict[str, dict] | None]:
        """Process a single training batch for one environment (runs in CUDA stream).

        Returns TENSORS (not floats) to avoid blocking .item() calls inside stream context.
        Call .item() only AFTER synchronizing all streams.

        Args:
            env_state: Parallel environment state (includes per-env scaler)
            inputs: Input tensor
            targets: Target tensor
            criterion: Loss criterion
            use_telemetry: Whether to collect telemetry
            slots: Enabled slot IDs (train all active slots)

        Returns:
            Tuple of (loss_tensor, correct_tensor, total, grad_stats)
            grad_stats maps slot_id -> async dual-gradient stats dict for that slot.
            It is None if use_telemetry=False or no slot needs gradient telemetry.
        """
        if not slots:
            raise ValueError("slots parameter is required and cannot be empty")

        model = env_state.model
        env_dev = env_state.env_device

        # Cache slot activity to avoid repeated dict lookups in hot path
        active_slots = {
            slot_id: model.has_active_seed_in_slot(slot_id)
            for slot_id in slots
        }
        slots_with_active_seeds = [slot_id for slot_id, active in active_slots.items() if active]

        # Use CUDA stream for async execution
        stream_ctx = torch.cuda.stream(env_state.stream) if env_state.stream else nullcontext()

        with stream_ctx:
            # Move data asynchronously
            inputs = inputs.to(env_dev, non_blocking=True)
            targets = targets.to(env_dev, non_blocking=True)
            # SharedBatchIterator may have created these tensors on the device's default stream.
            # record_stream prevents reuse/free while this env stream still consumes them.
            if env_state.stream and inputs.is_cuda:
                inputs.record_stream(env_state.stream)
                targets.record_stream(env_state.stream)

            model.train()

            # Auto-advance GERMINATED → TRAINING before forward so STE training starts immediately.
            # Do this per-slot (multi-seed support).
            for slot_id in slots_with_active_seeds:
                # Already filtered to active slots via cache
                slot_state = model.seed_slots[slot_id].state
                if slot_state and slot_state.stage == SeedStage.GERMINATED:
                    gate_result = model.seed_slots[slot_id].advance_stage(SeedStage.TRAINING)
                    if not gate_result.passed:
                        raise RuntimeError(
                            f"G1 gate failed during TRAINING entry for slot '{slot_id}': {gate_result}"
                        )

            # Ensure per-slot seed optimizers exist for any slot with a live seed.
            # We keep optimizers per-slot to avoid dynamic param-group surgery.
            slots_to_step: list[str] = []
            for slot_id in slots_with_active_seeds:
                # Already filtered to active slots via cache
                seed_state = model.seed_slots[slot_id].state
                if seed_state is None:
                    continue

                # Seeds can continue training through BLENDING/PROBATIONARY/FOSSILIZED.
                slots_to_step.append(slot_id)
                if slot_id not in env_state.seed_optimizers:
                    seed_params = list(model.get_seed_parameters(slot_id))
                    if not seed_params:
                        raise RuntimeError(
                            f"Seed in slot '{slot_id}' has no trainable parameters. "
                            f"Stage: {seed_state.stage.name}, "
                            f"Blueprint: {seed_state.blueprint_id}, "
                            f"Slot.seed: {model.seed_slots[slot_id].seed is not None}"
                        )
                    env_state.seed_optimizers[slot_id] = torch.optim.SGD(
                        seed_params, lr=task_spec.seed_lr, momentum=0.9
                    )

            env_state.host_optimizer.zero_grad(set_to_none=True)
            for slot_id in slots_to_step:
                env_state.seed_optimizers[slot_id].zero_grad(set_to_none=True)

            # Use pre-computed autocast decision from env_state
            autocast_ctx = (
                torch_amp.autocast(device_type="cuda", dtype=torch.float16)  # type: ignore[call-arg]
                if env_state.autocast_enabled
                else nullcontext()
            )
            with autocast_ctx:
                outputs = model(inputs)
                loss, correct_tensor, total = loss_and_correct(outputs, targets, criterion)

            if env_state.autocast_enabled:
                env_state.scaler.scale(loss).backward()
            else:
                loss.backward()
            # Collect gradient telemetry (isolated from torch.compile)
            grad_stats_by_slot = None
            if use_telemetry:
                grad_stats_by_slot = _collect_gradient_telemetry_for_batch(model, slots_with_active_seeds, env_dev)

            if use_amp and env_state.scaler is not None and env_dev.startswith("cuda"):
                # H12: AMP GradScaler stream safety documentation
                # Each env has its own GradScaler (created at line 855) to avoid race conditions.
                # GradScaler's internal state (_scale, _growth_tracker) is NOT stream-safe:
                # - scale() reads _scale without sync
                # - step() may write _found_inf_per_device
                # - update() modifies _scale and _growth_tracker
                #
                # This is safe because:
                # 1. Per-env scaler: No cross-env state sharing
                # 2. Sequential within stream: scale() → step() → update() ordered by stream
                # 3. No cross-batch state: Each env_state.scaler is isolated
                #
                # Note: scale factor update (update()) changes _scale for the NEXT batch,
                # which is fine since each batch fully completes before the next starts.
                env_state.scaler.step(env_state.host_optimizer)
                for slot_id in slots_to_step:
                    env_state.scaler.step(env_state.seed_optimizers[slot_id])
                env_state.scaler.update()
            else:
                env_state.host_optimizer.step()
                for slot_id in slots_to_step:
                    env_state.seed_optimizers[slot_id].step()

            # Return tensors - .item() called after stream sync
            return loss.detach(), correct_tensor, total, grad_stats_by_slot

    def process_val_batch(env_state: ParallelEnvState, inputs: torch.Tensor,
                          targets: torch.Tensor, criterion: nn.Module, slots: list[str] | None = None) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Process a validation batch for one environment.

        Returns TENSORS (not floats) to avoid blocking .item() calls inside stream context.
        Call .item() only AFTER synchronizing all streams.

        Args:
            env_state: Parallel environment state
            inputs: Input tensor
            targets: Target tensor
            criterion: Loss criterion
            slots: List of slot names (not used in this function but kept for API consistency)
        """
        model = env_state.model
        env_dev = env_state.env_device

        stream_ctx = torch.cuda.stream(env_state.stream) if env_state.stream else nullcontext()

        with stream_ctx:
            inputs = inputs.to(env_dev, non_blocking=True)
            targets = targets.to(env_dev, non_blocking=True)
            if env_state.stream and inputs.is_cuda:
                inputs.record_stream(env_state.stream)
                targets.record_stream(env_state.stream)

            model.eval()
            with torch.inference_mode():
                outputs = model(inputs)
                loss, correct_tensor, total = loss_and_correct(outputs, targets, criterion)

            # Return tensors - .item() called after stream sync
            return loss, correct_tensor, total

    history = []
    best_avg_acc = 0.0
    best_state = None
    recent_accuracies = []
    recent_rewards = []
    prev_rolling_avg_acc: float | None = None  # Track previous rolling avg for trend detection

    episodes_completed = start_episode
    total_episodes = n_episodes + start_episode  # Total target including resumed episodes

    batch_idx = 0
    # Gradient EMA tracker for drift detection (P4-9)
    # Persists across batches to track slow degradation
    grad_ema_tracker = GradientEMATracker() if use_telemetry else None

    while episodes_completed < total_episodes:
        # Determine how many envs to run this batch (may be fewer than n_envs for last batch)
        remaining = total_episodes - episodes_completed
        envs_this_batch = min(n_envs, remaining)
        # Monotonic epoch id for all per-batch snapshot events (commit barrier, PPO, analytics).
        # We use "episodes completed after this batch" so resumed runs stay monotonic.
        batch_epoch_id = episodes_completed + envs_this_batch

        # Create fresh environments for this batch
        # DataLoaders are shared via SharedBatchIterator (not per-env)
        base_seed = seed + batch_idx * 10000
        env_states = [
            create_env_state(i, base_seed)
            for i in range(envs_this_batch)
        ]
        criterion = nn.CrossEntropyLoss()

        # Initialize episode for vectorized training
        for env_idx in range(envs_this_batch):
            agent.buffer.start_episode(env_id=env_idx)
            env_states[env_idx].lstm_hidden = None  # Fresh hidden for new episode

        # Per-env accumulators
        env_final_accs = [0.0] * envs_this_batch
        env_total_rewards = [0.0] * envs_this_batch

        throughput_step_time_ms_sum = 0.0
        throughput_dataloader_wait_ms_sum = 0.0
        # GPU-accurate timing (P4-1) - uses CUDA events instead of perf_counter
        step_timer = CUDATimer(env_states[0].env_device)
        reward_summary_accum = [
            {"bounded_attribution": 0.0, "compute_rent": 0.0, "total_reward": 0.0, "count": 0}
            for _ in range(envs_this_batch)
        ]
        mask_hits = {head: 0 for head in HEAD_NAMES}
        mask_total = {head: 0 for head in HEAD_NAMES}

        # Accumulate raw (unnormalized) states for deferred normalizer update.
        # We freeze normalizer stats during rollout to ensure consistent normalization
        # across all states in a batch, then update stats after PPO update.
        raw_states_for_normalizer_update = []

        # Track per-environment rollback (more sample-efficient than batch-level).
        # Only envs that experienced rollback have stale transitions.
        env_rollback_occurred = [False] * envs_this_batch

        # Run epochs with INVERTED CONTROL FLOW
        for epoch in range(1, max_epochs + 1):
            step_timer.start()  # GPU-accurate timing (P4-1)
            dataloader_wait_ms_epoch = 0.0
            if telemetry_config is not None:
                telemetry_config.tick_escalation()
            for env_state in env_states:
                configure_slot_telemetry(env_state, inner_epoch=epoch, global_epoch=batch_epoch_id)
            # Track gradient stats per env for telemetry sync
            env_grad_stats: list[dict[str, dict[Any, Any]] | None] = [None] * envs_this_batch

            # Reset per-epoch metrics by zeroing pre-allocated accumulators (faster than reallocating)
            train_totals = [0] * envs_this_batch
            for env_state in env_states:
                env_state.zero_accumulators()

            # ===== TRAINING: Iterate batches first, launch all envs via CUDA streams =====
            # SharedBatchIterator: single DataLoader, batches pre-split and moved to devices
            # SharedGPUBatchIterator: GPU-resident data, one DataLoader per device

            # Issue one wait_stream per env BEFORE the loop starts (not per-batch).
            # This syncs the accumulator zeroing on default stream before we write.
            # record_stream marks tensors as used by this stream, preventing deallocation.
            for i, env_state in enumerate(env_states):
                if env_state.stream:
                    # Accumulators guaranteed non-None after init_accumulators()
                    env_state.train_loss_accum.record_stream(env_state.stream)  # type: ignore[union-attr]
                    env_state.train_correct_accum.record_stream(env_state.stream)  # type: ignore[union-attr]
                    env_state.stream.wait_stream(torch.cuda.default_stream(torch.device(env_state.env_device)))

            # Iterate training batches using shared iterator (SharedBatchIterator or SharedGPUBatchIterator)
            # Both provide list of (inputs, targets) per environment, already on correct devices
            train_iter = iter(shared_train_iter)
            for batch_step in range(num_train_batches):
                try:
                    fetch_start = time.perf_counter()
                    env_batches = next(train_iter)  # List of (inputs, targets), already on devices
                    dataloader_wait_ms_epoch += (time.perf_counter() - fetch_start) * 1000.0
                except StopIteration:
                    break

                # Launch all environments in their respective CUDA streams (async)
                # Data already moved to correct device by the shared iterator
                for i, env_state in enumerate(env_states):
                    if i >= len(env_batches):
                        continue
                    # CRITICAL: DataLoader collation (torch.stack) runs on the default stream.
                    # We must sync env_state.stream with default stream before using the data,
                    # otherwise we may access partially-transferred data (race condition).
                    # This applies to BOTH SharedBatchIterator (CPU→GPU transfers) and
                    # SharedGPUBatchIterator (collation still uses default stream for GPU data).
                    if env_state.stream:
                        env_state.stream.wait_stream(torch.cuda.default_stream(torch.device(env_state.env_device)))
                    inputs, targets = env_batches[i]
                    loss_tensor, correct_tensor, total, grad_stats = process_train_batch(
                        env_state, inputs, targets, criterion, use_telemetry=use_telemetry, slots=slots,
                        use_amp=amp_enabled,
                    )
                    if grad_stats is not None:
                        env_grad_stats[i] = grad_stats  # Keep last batch's grad stats
                    stream_ctx = torch.cuda.stream(env_state.stream) if env_state.stream else nullcontext()
                    with stream_ctx:
                        env_state.train_loss_accum.add_(loss_tensor)  # type: ignore[union-attr]
                        env_state.train_correct_accum.add_(correct_tensor)  # type: ignore[union-attr]
                    train_totals[i] += total

            # Sync all streams ONCE at epoch end
            for env_state in env_states:
                if env_state.stream:
                    env_state.stream.synchronize()

            # NOW safe to call .item() - all GPU work done
            # Accumulators guaranteed non-None after init_accumulators()
            train_losses = [env_state.train_loss_accum.item() for env_state in env_states]  # type: ignore[union-attr]
            train_corrects = [env_state.train_correct_accum.item() for env_state in env_states]  # type: ignore[union-attr]

            # ===== VALIDATION + COUNTERFACTUAL (FUSED): Single pass over test data =====
            # Instead of iterating test data twice (once for main validation, once for
            # counterfactual), we fuse both into a single loop. For each batch, we run:
            # 1. Main validation (real alpha) - accumulates val_correct_accum
            # 2. Per-slot counterfactual (alpha=0) - accumulates cf_correct_accums[slot_id]
            # This eliminates DataLoader overhead and enables multi-slot reward attribution.
            # Note: val/cf accumulators were already zeroed by zero_accumulators() above

            val_totals = [0] * envs_this_batch

            # Determine which slots need counterfactual BEFORE the loop
            # (seed with alpha > 0 means the seed is contributing to output)
            # slots_needing_counterfactual[env_idx] = set of slot_ids with active seeds
            slots_needing_counterfactual: dict[int, set[str]] = {}
            for i, env_state in enumerate(env_states):
                model = env_state.model
                active_slots: set[str] = set()
                for slot_id in slots:
                    if model.has_active_seed_in_slot(slot_id):
                        seed_state = model.seed_slots[slot_id].state
                        if seed_state and seed_state.alpha > 0:
                            active_slots.add(slot_id)
                if active_slots:
                    slots_needing_counterfactual[i] = active_slots

            # baseline_accs[env_idx][slot_id] = accuracy with that slot's seed disabled
            baseline_accs: list[dict[str, float]] = [{} for _ in range(envs_this_batch)]

            # Issue one wait_stream per env before the loop starts (not per-batch)
            # This syncs the accumulator zeroing on default stream before we write.
            for i, env_state in enumerate(env_states):
                if env_state.stream:
                    # Accumulators guaranteed non-None after init_accumulators()
                    env_state.val_loss_accum.record_stream(env_state.stream)  # type: ignore[union-attr]
                    env_state.val_correct_accum.record_stream(env_state.stream)  # type: ignore[union-attr]
                    env_state.stream.wait_stream(torch.cuda.default_stream(torch.device(env_state.env_device)))
                    # Register per-slot counterfactual accumulators with stream
                    if i in slots_needing_counterfactual:
                        for slot_id in slots_needing_counterfactual[i]:
                            env_state.cf_correct_accums[slot_id].record_stream(env_state.stream)
                        n_slots = len(slots_needing_counterfactual[i])
                        # Register all-disabled accumulator when full factorial is active (2-4 seeds)
                        if 2 <= n_slots <= 4:
                            env_state.cf_all_disabled_accum.record_stream(env_state.stream)  # type: ignore[union-attr]
                        # Register pair accumulators when 3-4 seeds (for 2 seeds, pair = all enabled)
                        if 3 <= n_slots <= 4:
                            for pair_key in env_state.cf_pair_accums:
                                env_state.cf_pair_accums[pair_key].record_stream(env_state.stream)

            # Iterate validation batches using shared iterator (SharedBatchIterator or SharedGPUBatchIterator)
            test_iter = iter(shared_test_iter)
            for batch_step in range(num_test_batches):
                try:
                    fetch_start = time.perf_counter()
                    env_batches = next(test_iter)  # List of (inputs, targets), already on devices
                    dataloader_wait_ms_epoch += (time.perf_counter() - fetch_start) * 1000.0
                except StopIteration:
                    break

                # Launch all environments: MAIN VALIDATION + COUNTERFACTUAL on same batch
                for i, env_state in enumerate(env_states):
                    if i >= len(env_batches):
                        continue
                    # CRITICAL: DataLoader collation (torch.stack) runs on the default stream.
                    # We must sync env_state.stream with default stream before using the data,
                    # otherwise we may access partially-transferred data (race condition).
                    # This applies to BOTH SharedBatchIterator (CPU→GPU transfers) and
                    # SharedGPUBatchIterator (collation still uses default stream for GPU data).
                    if env_state.stream:
                        env_state.stream.wait_stream(torch.cuda.default_stream(torch.device(env_state.env_device)))
                    inputs, targets = env_batches[i]

                    # MAIN VALIDATION (real alpha)
                    loss_tensor, correct_tensor, total = process_val_batch(env_state, inputs, targets, criterion, slots=slots)
                    stream_ctx = torch.cuda.stream(env_state.stream) if env_state.stream else nullcontext()
                    with stream_ctx:
                        env_state.val_loss_accum.add_(loss_tensor)  # type: ignore[union-attr]
                        env_state.val_correct_accum.add_(correct_tensor)  # type: ignore[union-attr]
                    val_totals[i] += total

                    # COUNTERFACTUAL (alpha=0) - SAME BATCH, no DataLoader reload!
                    # Data is already on GPU from the main validation pass.
                    # Compute per-slot counterfactual for multi-slot reward attribution
                    if i in slots_needing_counterfactual:
                        active_slot_list = list(slots_needing_counterfactual[i])
                        n_active = len(active_slot_list)

                        # Per-slot ablation (always computed - needed for Tamiyo)
                        for slot_id in active_slot_list:
                            # Entire counterfactual pass in stream_ctx (PyTorch specialist fix)
                            with stream_ctx:
                                with env_state.model.seed_slots[slot_id].force_alpha(0.0):
                                    _, cf_correct_tensor, cf_total = process_val_batch(
                                        env_state, inputs, targets, criterion, slots=slots
                                    )
                                env_state.cf_correct_accums[slot_id].add_(cf_correct_tensor)
                            env_state.cf_totals[slot_id] += cf_total

                        # "All disabled" pass for full factorial (only when ≤4 seeds to limit 2^n blowup)
                        if 2 <= n_active <= 4:
                            with stream_ctx:
                                # Disable ALL active seeds for true baseline measurement
                                with ExitStack() as stack:
                                    for slot_id in active_slot_list:
                                        stack.enter_context(env_state.model.seed_slots[slot_id].force_alpha(0.0))
                                    _, cf_all_correct, cf_all_total = process_val_batch(
                                        env_state, inputs, targets, criterion, slots=slots
                                    )
                                env_state.cf_all_disabled_accum.add_(cf_all_correct)
                            env_state.cf_all_disabled_total += cf_all_total

                        # Pair passes for 3-4 seeds (enable only pair, disable others)
                        # For 2 seeds, the "all enabled" config IS the pair, so no extra passes needed
                        # NOTE: Use seed_i/seed_j to avoid shadowing outer env loop variable 'i'
                        if 3 <= n_active <= 4:
                            for seed_i in range(n_active):
                                for seed_j in range(seed_i + 1, n_active):
                                    with stream_ctx:
                                        with ExitStack() as stack:
                                            # Disable seeds NOT in the pair
                                            for k, slot_id in enumerate(active_slot_list):
                                                if k != seed_i and k != seed_j:
                                                    stack.enter_context(env_state.model.seed_slots[slot_id].force_alpha(0.0))
                                            _, cf_pair_correct, cf_pair_total = process_val_batch(
                                                env_state, inputs, targets, criterion, slots=slots
                                            )
                                        env_state.cf_pair_accums[(seed_i, seed_j)].add_(cf_pair_correct)
                                    env_state.cf_pair_totals[(seed_i, seed_j)] += cf_pair_total

            # Single sync point at end (not once per pass)
            for env_state in env_states:
                if env_state.stream:
                    env_state.stream.synchronize()

            # NOW safe to call .item()
            # Accumulators guaranteed non-None after init_accumulators()
            val_losses = [env_state.val_loss_accum.item() for env_state in env_states]  # type: ignore[union-attr]
            val_corrects = [env_state.val_correct_accum.item() for env_state in env_states]  # type: ignore[union-attr]

            # Compute per-slot baseline accuracies from counterfactual accumulators
            all_disabled_accs: dict[int, float] = {}  # env_idx -> accuracy with all seeds disabled
            for i in slots_needing_counterfactual:
                for slot_id in slots_needing_counterfactual[i]:
                    cf_total = env_states[i].cf_totals[slot_id]
                    if cf_total > 0:
                        baseline_accs[i][slot_id] = (
                            100.0 * env_states[i].cf_correct_accums[slot_id].item() / cf_total
                        )
                # Compute "all disabled" accuracy if measured
                if env_states[i].cf_all_disabled_total > 0:
                    all_disabled_accs[i] = (
                        100.0 * env_states[i].cf_all_disabled_accum.item() / env_states[i].cf_all_disabled_total
                    )

            # Compute pair accuracies for 3-4 seeds
            pair_accs: dict[int, dict[tuple[int, int], float]] = {}  # env_idx -> {(i,j): accuracy}
            for i in slots_needing_counterfactual:
                if len(slots_needing_counterfactual[i]) >= 3:
                    pair_accs[i] = {}
                    for pair_key, total in env_states[i].cf_pair_totals.items():
                        if total > 0:
                            pair_accs[i][pair_key] = (
                                100.0 * env_states[i].cf_pair_accums[pair_key].item() / total
                            )

            # ===== Compute epoch metrics and get BATCHED actions =====
            # NOTE: Telemetry sync (gradients/counterfactual) happens after record_accuracy()
            # so telemetry reflects the current epoch's metrics.

            # Collect features and action masks from all environments
            all_features = []
            all_masks = []
            all_signals = []
            governor_panic_envs = []  # Track which envs need rollback

            # Number of germinate actions = total actions - 3 (WAIT, FOSSILIZE, CULL)
            for env_idx, env_state in enumerate(env_states):
                model = env_state.model

                train_loss = train_losses[env_idx] / num_train_batches
                train_acc = 100.0 * train_corrects[env_idx] / max(train_totals[env_idx], 1)
                val_loss = val_losses[env_idx] / num_test_batches
                val_acc = 100.0 * val_corrects[env_idx] / max(val_totals[env_idx], 1)

                # Store metrics for later
                env_state.train_loss = train_loss
                env_state.train_acc = train_acc
                env_state.val_loss = val_loss
                env_state.val_acc = val_acc
                # Track maximum accuracy for sparse reward
                env_state.host_max_acc = max(env_state.host_max_acc, env_state.val_acc)

                # Governor watchdog: snapshot when loss is stable (every 5 epochs)
                if epoch % 5 == 0:
                    env_state.governor.snapshot()

                # Governor watchdog: check vital signs after validation
                is_panic = env_state.governor.check_vital_signs(val_loss)
                if is_panic:
                    governor_panic_envs.append(env_idx)

                # Health monitor - check GPU memory warnings
                if env_state.health_monitor is not None:
                    try:
                        if torch.cuda.is_available():
                            gpu_allocated = torch.cuda.memory_allocated(env_state.env_device)
                            gpu_total = torch.cuda.get_device_properties(
                                torch.device(env_state.env_device).index
                            ).total_memory
                            if gpu_total > 0:
                                env_state.health_monitor._check_memory_and_warn(
                                    gpu_utilization=gpu_allocated / gpu_total,
                                    gpu_allocated_gb=gpu_allocated / (1024**3),
                                    gpu_total_gb=gpu_total / (1024**3),
                                )
                    except Exception:
                        pass  # Non-critical monitoring - don't crash training

                # Gather active seeds across ALL enabled slots (multi-seed support)
                active_seeds = []
                for slot_id in slots:
                    if model.has_active_seed_in_slot(slot_id):
                        seed_state = model.seed_slots[slot_id].state
                        if seed_state is not None:
                            active_seeds.append(seed_state)

                # Record accuracy for all active seeds (per-slot stage counters + deltas)
                for seed_state in active_seeds:
                    if seed_state.metrics:
                        seed_state.metrics.record_accuracy(val_acc)

                # Update counterfactual contribution for any slot where we computed a baseline this epoch
                # (used by G5 gate + PROBATIONARY safety auto-cull).
                if baseline_accs[env_idx]:
                    for slot_id, baseline_acc in baseline_accs[env_idx].items():
                        if model.has_active_seed_in_slot(slot_id):
                            seed_state = model.seed_slots[slot_id].state
                            if seed_state and seed_state.metrics:
                                seed_state.metrics.counterfactual_contribution = val_acc - baseline_acc

                    # Emit live counterfactual matrix for Sanctum real-time display
                    # This shows "removing this seed decreases accuracy by X%" - useful for operators
                    # When 2-4 seeds active, we measure full factorial (all-disabled pass included).
                    # Otherwise, ablation-only with estimates.
                    if hub:
                        active_slots = list(baseline_accs[env_idx].keys())
                        if active_slots:
                            # Build matrix from ablation data + measured all-disabled when available
                            configs = []
                            n = len(active_slots)

                            # All disabled - use measured value when available (2-4 seeds),
                            # otherwise fall back to estimate
                            if env_idx in all_disabled_accs:
                                all_disabled_acc = all_disabled_accs[env_idx]
                                strategy = "full_factorial"
                            else:
                                all_disabled_acc = min(baseline_accs[env_idx].values())
                                strategy = "ablation_only"
                            configs.append({
                                "seed_mask": [False] * n,
                                "accuracy": all_disabled_acc,
                            })

                            # Per-slot enabled (invert ablation: if removing A gives baseline_A,
                            # then A's solo contribution ≈ val_acc - baseline_A when A is the ONLY one removed)
                            # Actually, for single-slot enabled, we approximate by reflecting the ablation:
                            # If combined=75% and removing A gives 70%, A contributes 5% marginal
                            # A's "solo" ≈ all_disabled_acc + contribution
                            for i, slot_id in enumerate(active_slots):
                                contribution = val_acc - baseline_accs[env_idx][slot_id]
                                solo_estimate = all_disabled_acc + contribution
                                mask = [j == i for j in range(n)]
                                configs.append({
                                    "seed_mask": mask,
                                    "accuracy": solo_estimate,
                                })

                            # Pair configs for 3-4 seeds (measured, not estimated)
                            # For 2 seeds, "all enabled" IS the pair, so no separate pair config needed
                            if env_idx in pair_accs and n >= 3:
                                for (i, j), pair_acc in pair_accs[env_idx].items():
                                    mask = [k == i or k == j for k in range(n)]
                                    configs.append({
                                        "seed_mask": mask,
                                        "accuracy": pair_acc,
                                    })

                            # All enabled - current accuracy
                            configs.append({
                                "seed_mask": [True] * n,
                                "accuracy": val_acc,
                            })

                            hub.emit(TelemetryEvent(
                                event_type=TelemetryEventType.COUNTERFACTUAL_MATRIX_COMPUTED,
                                data={
                                    "env_id": env_idx,
                                    "slot_ids": active_slots,
                                    "configs": configs,
                                    "strategy": strategy,
                                    "compute_time_ms": 0.0,  # Extra pass only for 2-4 seeds
                                },
                            ))

                # Log counterfactual contribution at terminal epoch (for all active slots)
                if epoch == max_epochs and hub and env_idx in slots_needing_counterfactual:
                    for slot_id in slots_needing_counterfactual[env_idx]:
                        baseline_acc_log = baseline_accs[env_idx].get(slot_id)
                        if baseline_acc_log is None:
                            emit_cf_unavailable(
                                hub,
                                env_id=env_idx,
                                slot_id=slot_id,
                                reason="missing_baseline",
                            )
                            continue
                        hub.emit(TelemetryEvent(
                            event_type=TelemetryEventType.COUNTERFACTUAL_COMPUTED,
                            slot_id=slot_id,
                            data={
                                "env_id": env_idx,
                                "device": env_state.env_device,
                                "available": True,
                                "slot_id": slot_id,
                                "real_accuracy": val_acc,
                                "baseline_accuracy": baseline_acc_log,
                                "contribution": val_acc - baseline_acc_log,
                            },
                        ))

                # Sync gradient telemetry after record_accuracy so telemetry reflects this epoch's metrics.
                grad_stats_for_env = env_grad_stats[env_idx]
                if use_telemetry and grad_stats_for_env is not None:
                    for slot_id, async_stats in grad_stats_for_env.items():
                        if not model.has_active_seed_in_slot(slot_id):
                            continue
                        seed_state = model.seed_slots[slot_id].state
                        if seed_state is None or seed_state.metrics is None:
                            continue

                        dual_stats = materialize_dual_grad_stats(async_stats)
                        current_ratio = dual_stats.normalized_ratio

                        prev_ema = env_state.gradient_ratio_ema.get(slot_id)
                        if prev_ema is None:
                            ema = current_ratio
                        else:
                            ema = 0.9 * prev_ema + 0.1 * current_ratio
                        env_state.gradient_ratio_ema[slot_id] = ema

                        # Sync ratio to SeedMetrics for G2 gate evaluation
                        seed_state.metrics.seed_gradient_norm_ratio = ema

                        # Sync telemetry using seed gradient stats from dual collection
                        seed_state.sync_telemetry(
                            gradient_norm=dual_stats.seed_grad_norm,
                            gradient_health=1.0,  # Simplified: dual stats don't compute health
                            has_vanishing=dual_stats.seed_grad_norm < 1e-7,
                            has_exploding=dual_stats.seed_grad_norm > 100.0,
                            epoch=epoch,
                            max_epochs=max_epochs,
                        )

                if use_telemetry:
                    # Refresh telemetry for ALL active enabled slots each epoch.
                    # If gradient telemetry isn't available for a slot, preserve the last-known
                    # gradient fields but keep stage/alpha/accuracy/epochs_in_stage fresh.
                    for slot_id in slots:
                        if not model.has_active_seed_in_slot(slot_id):
                            continue
                        seed_state = model.seed_slots[slot_id].state
                        if seed_state is None or seed_state.telemetry is None:
                            continue
                        seed_state.sync_telemetry(
                            gradient_norm=seed_state.telemetry.gradient_norm,
                            gradient_health=seed_state.telemetry.gradient_health,
                            has_vanishing=seed_state.telemetry.has_vanishing,
                            has_exploding=seed_state.telemetry.has_exploding,
                            epoch=epoch,
                            max_epochs=max_epochs,
                        )

                slot_reports = model.get_slot_reports()

                # Emit per-env EPOCH_COMPLETED with per-seed telemetry for TUI updates
                if hub and use_telemetry:
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

                    hub.emit(TelemetryEvent(
                        event_type=TelemetryEventType.EPOCH_COMPLETED,
                        epoch=epoch,
                        data={
                            "env_id": env_idx,
                            "inner_epoch": epoch,
                            "val_accuracy": val_acc,
                            "val_loss": val_loss,
                            "train_accuracy": train_acc,
                            "train_loss": train_loss,
                            "seeds": seeds_telemetry,
                        },
                    ))

                # Update signal tracker
                available_slots = sum(
                    1 for slot_id in slots if not model.has_active_seed_in_slot(slot_id)
                )
                signals = env_state.signal_tracker.update(
                    epoch=epoch,
                    global_step=epoch * num_train_batches,
                    train_loss=train_loss,
                    train_accuracy=train_acc,
                    val_loss=val_loss,
                    val_accuracy=val_acc,
                    active_seeds=active_seeds,
                    available_slots=available_slots,
                )
                all_signals.append(signals)

                features = signals_to_features(
                    signals,
                    slot_reports=slot_reports,
                    use_telemetry=use_telemetry,
                    slots=slots,
                    total_params=model.total_params if model else 0,
                    total_seeds=model.total_seeds() if model else 0,
                    max_seeds=effective_max_seeds,
                    slot_config=slot_config,
                )
                all_features.append(features)

                # Compute action mask based on current state (physical constraints only)
                # Build slot states for ALL enabled slots (multi-slot masking)
                ordered = validate_slot_ids(list(slots))
                slot_states = build_slot_states(slot_reports, ordered)
                mask = compute_action_masks(
                    slot_states=slot_states,
                    enabled_slots=ordered,
                    total_seeds=model.total_seeds() if model else 0,
                    max_seeds=effective_max_seeds,
                    slot_config=slot_config,
                    device=torch.device(device),
                    topology=task_spec.topology,
                )
                all_masks.append(mask)

            # Batch all states and masks into tensors
            states_batch = torch.tensor(all_features, dtype=torch.float32, device=device)
            # Stack dict masks into batched dict: {key: [n_envs, head_dim]}
            # Use static HEAD_NAMES for torch.compile compatibility
            masks_batch = {
                key: torch.stack([m[key] for m in all_masks]).to(device)
                for key in HEAD_NAMES
            }

            # Accumulate raw states for deferred normalizer update
            raw_states_for_normalizer_update.append(states_batch.detach())

            # Normalize using FROZEN statistics during rollout collection.
            # IMPORTANT: We do NOT update obs_normalizer here - statistics are updated
            # in _run_ppo_updates() BEFORE the PPO update (C5 FIX). This ensures all
            # states in a rollout batch use identical normalization parameters during
            # collection, preventing the "normalizer drift" bug where states from
            # different steps within the same batch would be normalized with different
            # mean/var, causing PPO ratio calculation errors.
            states_batch_normalized = obs_normalizer.normalize(states_batch)

            # Get BATCHED actions from policy network with action masking (single forward pass!)
            # Tamiyo mode: LSTM with per-head log_probs
            # Batch hidden states together: [num_layers, batch, hidden_dim]
            #
            # IMPORTANT: Capture PRE-STEP hidden states for buffer storage.
            # The hidden state stored with a transition should be the INPUT to the network
            # when selecting the action, not the OUTPUT. This enables proper BPTT during
            # training by reconstructing the exact forward pass.
            pre_step_hiddens: list[tuple[torch.Tensor, torch.Tensor]] = []
            batched_hidden = None
            if env_states[0].lstm_hidden is not None:
                # Concatenate per-env hidden states along batch dimension
                # All envs have same hidden state lifecycle (all None or all set)
                h_list = [env_state.lstm_hidden[0] for env_state in env_states]  # type: ignore[index]
                c_list = [env_state.lstm_hidden[1] for env_state in env_states]  # type: ignore[index]
                # Clone pre-step hidden states for buffer storage
                pre_step_hiddens = [(h.clone(), c.clone()) for h, c in zip(h_list, c_list)]
                batched_h = torch.cat(h_list, dim=1)  # [layers, batch, hidden]
                batched_c = torch.cat(c_list, dim=1)
                batched_hidden = (batched_h, batched_c)
            else:
                # First step of episode - use initial hidden state for all envs
                init_hidden = agent.network.get_initial_hidden(len(env_states), agent.device)
                # Unbatch to per-env for storage
                init_h, init_c = init_hidden
                for env_idx in range(len(env_states)):
                    env_h = init_h[:, env_idx:env_idx+1, :].clone()
                    env_c = init_c[:, env_idx:env_idx+1, :].clone()
                    pre_step_hiddens.append((env_h, env_c))

            # get_action returns GetActionResult dataclass
            # Request op_logits when telemetry is enabled for Decision Snapshot
            action_result = agent.network.get_action(
                states_batch_normalized,
                hidden=batched_hidden,
                slot_mask=masks_batch["slot"],
                blueprint_mask=masks_batch["blueprint"],
                blend_mask=masks_batch["blend"],
                op_mask=masks_batch["op"],
                deterministic=False,
                return_op_logits=use_telemetry,
            )
            actions_dict = action_result.actions
            head_log_probs = action_result.log_probs
            values_tensor = action_result.values
            new_hidden = action_result.hidden
            # op_logits available via action_result.op_logits when use_telemetry=True

            # Unbatch new hidden states back to per-env
            # new_hidden is (h, c) each [num_layers, batch, hidden_dim]
            new_h, new_c = new_hidden
            for env_idx, env_state in enumerate(env_states):
                # Extract this env's hidden: [num_layers, 1, hidden_dim]
                env_h = new_h[:, env_idx:env_idx+1, :].contiguous()
                env_c = new_c[:, env_idx:env_idx+1, :].contiguous()
                env_state.lstm_hidden = (env_h, env_c)

                # Check LSTM health (P4-8) - detect hidden state drift/explosion
                if use_telemetry:
                    lstm_health = compute_lstm_health(env_state.lstm_hidden)
                    if lstm_health is not None and not lstm_health.is_healthy():
                        warnings.warn(
                            f"Env {env_idx}: LSTM unhealthy - h_norm={lstm_health.h_norm:.2f}, "
                            f"c_norm={lstm_health.c_norm:.2f}, nan={lstm_health.has_nan}, "
                            f"inf={lstm_health.has_inf}",
                            RuntimeWarning,
                            stacklevel=2,
                        )

            # Convert to list of dicts for per-env processing
            # Batch transfer actions to CPU (eliminates 16 .item() syncs per epoch)
            actions_cpu = {key: actions_dict[key].cpu().numpy() for key in HEAD_NAMES}
            actions = [
                {key: int(actions_cpu[key][i]) for key in HEAD_NAMES}
                for i in range(len(env_states))
            ]
            # Single CPU transfer - .tolist() is efficient (no per-element sync)
            values = values_tensor.tolist()
            # head_log_probs is dict of tensors {key: [batch]}
            # Keep as-is for tamiyo buffer storage

            # Batch compute mask stats for telemetry (eliminates 16 .item() syncs)
            # "masked" means not all actions are valid for this head
            if hub and use_telemetry:
                masked_batch = {
                    key: ~masks_batch[key].all(dim=-1)  # [num_envs] bool tensor
                    for key in HEAD_NAMES
                }
                masked_cpu = {key: masked_batch[key].cpu().numpy() for key in HEAD_NAMES}
            else:
                masked_cpu = None

            # PHASE 1: Execute actions and collect data for bootstrap computation
            # We collect bootstrap data for all truncated envs, then compute in batch
            bootstrap_data = []
            transitions_data = []  # Store transition data for buffer storage

            for env_idx, env_state in enumerate(env_states):
                model = env_state.model
                signals = all_signals[env_idx]

                # Now Python floats/ints - no GPU sync
                value = values[env_idx]

                # Parse factored action using direct indexing (no object creation)
                action_dict = actions[env_idx]  # {slot: int, blueprint: int, blend: int, tempo: int, op: int}
                slot_idx = action_dict["slot"]
                blueprint_idx = action_dict["blueprint"]
                blend_idx = action_dict["blend"]
                tempo_idx = action_dict["tempo"]
                op_idx = action_dict["op"]

                # DEBUG: Verify direct indexing matches FactoredAction properties
                # This block is stripped by Python when run with -O flag (production)
                if __debug__:
                    _fa = FactoredAction.from_indices(slot_idx, blueprint_idx, blend_idx, tempo_idx, op_idx)
                    assert slot_idx == _fa.slot_idx, f"slot_idx mismatch: {slot_idx} != {_fa.slot_idx}"
                    assert OP_NAMES[op_idx] == _fa.op.name, f"op.name mismatch: {OP_NAMES[op_idx]} != {_fa.op.name}"
                    assert (op_idx == OP_GERMINATE) == _fa.is_germinate, "is_germinate mismatch"
                    assert (op_idx == OP_FOSSILIZE) == _fa.is_fossilize, "is_fossilize mismatch"
                    assert (op_idx == OP_CULL) == _fa.is_cull, "is_cull mismatch"
                    assert BLUEPRINT_IDS[blueprint_idx] == _fa.blueprint_id, f"blueprint_id mismatch"
                    assert BLEND_IDS[blend_idx] == _fa.blend_algorithm_id, f"blend_id mismatch"
                    del _fa  # Don't leak into scope

                # Use the SAMPLED slot as target (multi-slot support)
                # slot_idx is in canonical SlotConfig order, not caller slot list order.
                target_slot, slot_is_enabled = _resolve_target_slot(
                    slot_idx,
                    enabled_slots=slots,
                    slot_config=slot_config,
                )
                seed_state = (
                    model.seed_slots[target_slot].state
                    if slot_is_enabled and model.has_active_seed_in_slot(target_slot)
                    else None
                )
                # Use op name for action counting
                env_state.action_counts[OP_NAMES[op_idx]] = env_state.action_counts.get(OP_NAMES[op_idx], 0) + 1
                # For reward computation, use LifecycleOp (IntEnum compatible)
                action_for_reward = LifecycleOp(op_idx)

                action_success = False

                # Governor rollback: execute if this env panicked
                if env_idx in governor_panic_envs:
                    env_state.governor.execute_rollback(
                        env_id=env_idx,
                        optimizer=env_state.host_optimizer
                    )
                    env_rollback_occurred[env_idx] = True  # Mark only this env as stale

                # Compute reward with cost params
                # Derive cost from CURRENT architecture, not cumulative scoreboard
                # (rent should reflect current extra params, not historical totals)
                scoreboard = analytics._get_scoreboard(env_idx)
                host_params = scoreboard.host_params
                # BUG FIX: Was using active_seed_params (seed only), should be total (host + seed)
                # Without this fix, total_params < host_params is always true, so rent is never charged
                total_params = model.total_params

                # Compute seed_contribution from counterfactual for the SAMPLED slot
                # (multi-slot reward attribution: use the slot the policy chose)
                seed_contribution = None
                if target_slot in baseline_accs[env_idx]:
                    seed_contribution = env_state.val_acc - baseline_accs[env_idx][target_slot]
                    # Store in metrics for telemetry at fossilize/cull
                    if seed_state and seed_state.metrics:
                        seed_state.metrics.counterfactual_contribution = seed_contribution

                emit_reward_components_event = (
                    telemetry_config is not None and telemetry_config.should_collect("debug")
                )
                collect_reward_summary = (
                    telemetry_config is not None and telemetry_config.should_collect("ops_normal")
                )

                seed_params_for_slot = (
                    model.seed_slots[target_slot].active_seed_params
                    if slot_is_enabled
                    else 0
                )

                # Unified reward computation - family selector: contribution vs loss-primary
                reward_components = None
                seed_info = SeedInfo.from_seed_state(seed_state, seed_params_for_slot)
                if reward_family_enum == RewardFamily.CONTRIBUTION:
                    need_reward_components = emit_reward_components_event or collect_reward_summary
                    if need_reward_components:
                        reward, reward_components = compute_reward(
                            action=action_for_reward,
                            seed_contribution=seed_contribution,
                            val_acc=env_state.val_acc,
                            host_max_acc=env_state.host_max_acc,
                            seed_info=seed_info,
                            epoch=epoch,
                            max_epochs=max_epochs,
                            total_params=total_params,
                            host_params=host_params,
                            acc_at_germination=env_state.acc_at_germination.get(target_slot),
                            acc_delta=signals.metrics.accuracy_delta,
                            return_components=True,
                            num_fossilized_seeds=env_state.seeds_fossilized,
                            num_contributing_fossilized=env_state.contributing_fossilized,
                            config=env_reward_configs[env_idx],
                        )
                        if target_slot in baseline_accs[env_idx]:
                            reward_components.host_baseline_acc = baseline_accs[env_idx][target_slot]
                    else:
                        reward = compute_reward(
                            action=action_for_reward,
                            seed_contribution=seed_contribution,
                            val_acc=env_state.val_acc,
                            host_max_acc=env_state.host_max_acc,
                            seed_info=seed_info,
                            epoch=epoch,
                            max_epochs=max_epochs,
                            total_params=total_params,
                            host_params=host_params,
                            acc_at_germination=env_state.acc_at_germination.get(target_slot),
                            acc_delta=signals.metrics.accuracy_delta,
                            num_fossilized_seeds=env_state.seeds_fossilized,
                            num_contributing_fossilized=env_state.contributing_fossilized,
                            config=env_reward_configs[env_idx],
                        )
                else:
                    reward = compute_loss_reward(
                        action=action_for_reward,
                        loss_delta=signals.metrics.loss_delta,
                        val_loss=env_state.val_loss,
                        seed_info=seed_info,
                        epoch=epoch,
                        max_epochs=max_epochs,
                        total_params=total_params,
                        host_params=host_params,
                        config=loss_reward_config,
                    )

                # Governor punishment: inject negative reward if rollback occurred
                if env_idx in governor_panic_envs:
                    punishment = env_state.governor.get_punishment_reward()
                    reward += punishment
                    if hub:
                        hub.emit(TelemetryEvent(
                            event_type=TelemetryEventType.REWARD_COMPUTED,
                            severity="warning",
                            data={
                                "env_id": env_idx,
                                "action_name": "PUNISHMENT",
                                "total_reward": reward,
                                "punishment": punishment,
                                "reason": "governor_rollback",
                            },
                        ))

                # Apply pending auto-cull penalty from previous step_epoch()
                # (DRL Expert review 2025-12-17: prevents degenerate WAIT-spam policies)
                if env_state.pending_auto_cull_penalty != 0.0:
                    reward += env_state.pending_auto_cull_penalty
                    if hub:
                        hub.emit(TelemetryEvent(
                            event_type=TelemetryEventType.REWARD_COMPUTED,
                            severity="info",
                            data={
                                "env_id": env_idx,
                                "action_name": "AUTO_CULL_PENALTY",
                                "total_reward": reward,
                                "penalty": env_state.pending_auto_cull_penalty,
                                "reason": "auto_cull_from_prev_step",
                            },
                        ))
                    env_state.pending_auto_cull_penalty = 0.0  # Clear after application

                if collect_reward_summary and reward_components is not None:
                    summary = reward_summary_accum[env_idx]
                    summary["total_reward"] += reward
                    if reward_components.bounded_attribution is not None:
                        summary["bounded_attribution"] += reward_components.bounded_attribution
                    summary["compute_rent"] += reward_components.compute_rent
                    summary["count"] += 1

                # Execute action using FactoredAction properties
                # Validate sampled slot is in enabled slots (masking should prevent this, but safety check)
                if not slot_is_enabled:
                    # Invalid slot selection - action fails silently
                    # (should not happen if action masking is working correctly)
                    action_success = False

                elif op_idx == OP_GERMINATE:
                    # Germinate in the SAMPLED slot (multi-slot support)
                    if not model.has_active_seed_in_slot(target_slot):
                        env_state.acc_at_germination[target_slot] = env_state.val_acc
                        blueprint_id = BLUEPRINT_IDS[blueprint_idx]
                        blend_algorithm_id = BLEND_IDS[blend_idx]
                        tempo_epochs = TEMPO_TO_EPOCHS[TempoAction(tempo_idx)]
                        seed_id = f"env{env_idx}_seed_{env_state.seeds_created}"
                        model.germinate_seed(
                            blueprint_id,
                            seed_id,
                            slot=target_slot,
                            blend_algorithm_id=blend_algorithm_id,
                            blend_tempo_epochs=tempo_epochs,
                        )
                        env_state.seeds_created += 1
                        env_state.seed_optimizers.pop(target_slot, None)
                        action_success = True

                elif op_idx == OP_FOSSILIZE:
                    # Fossilize the seed in the SAMPLED slot
                    seed_total_improvement = (
                        seed_state.metrics.total_improvement
                        if seed_state and seed_state.metrics else 0.0
                    )
                    # _advance_active_seed handles the actual fossilization
                    action_success = _advance_active_seed(model, target_slot)
                    if action_success:
                        env_state.seeds_fossilized += 1
                        if seed_total_improvement >= DEFAULT_MIN_FOSSILIZE_CONTRIBUTION:
                            env_state.contributing_fossilized += 1
                        env_state.acc_at_germination.pop(target_slot, None)

                elif op_idx == OP_CULL:
                    # Cull the seed in the SAMPLED slot
                    if model.has_active_seed_in_slot(target_slot):
                        model.cull_seed(slot=target_slot)
                        env_state.seed_optimizers.pop(target_slot, None)
                        env_state.acc_at_germination.pop(target_slot, None)
                        action_success = True

                else:
                    # WAIT always succeeds
                    action_success = True

                if action_success:
                    env_state.successful_action_counts[OP_NAMES[op_idx]] = env_state.successful_action_counts.get(OP_NAMES[op_idx], 0) + 1

                if hub and use_telemetry and (
                    telemetry_config is None or telemetry_config.should_collect("ops_normal")
                ):
                    # Use pre-computed batched mask stats (0 GPU syncs - already on CPU)
                    masked_flags = {key: bool(masked_cpu[key][env_idx]) for key in HEAD_NAMES}
                    for head, masked in masked_flags.items():
                        mask_total[head] += 1
                        if masked:
                            mask_hits[head] += 1
                    emit_last_action(
                        env_id=env_idx,
                        epoch=epoch,
                        slot_idx=slot_idx,
                        blueprint_idx=blueprint_idx,
                        blend_idx=blend_idx,
                        tempo_idx=tempo_idx,
                        op_idx=op_idx,
                        slot_id=target_slot,
                        masked=masked_flags,
                        success=action_success,
                    )

                # Emit reward telemetry if collecting (after action execution so we have action_success)
                if emit_reward_components_event and reward_components is not None:
                    reward_components.action_success = action_success

                    # Build Decision Snapshot fields
                    # action_confidence: P(chosen_op) = exp(log_prob)
                    action_confidence = float(
                        head_log_probs["op"][env_idx].exp().item()
                    )

                    # value_estimate: V(s_t) from critic
                    value_estimate = float(value)

                    # slot_states: {slot_id: "Stage N%"} for each slot
                    slot_states_dict: dict[str, str] = {}
                    for slot_id in slots:
                        if model.has_active_seed_in_slot(slot_id):
                            slot_state = model.seed_slots[slot_id].state
                            if slot_state is not None:
                                stage_name = slot_state.stage.name.title()
                                progress = slot_state.metrics.epochs_in_current_stage if slot_state.metrics else 0
                                slot_states_dict[slot_id] = f"{stage_name} {progress}ep"
                            else:
                                slot_states_dict[slot_id] = "Empty"
                        else:
                            slot_states_dict[slot_id] = "Empty"

                    # alternatives: top-2 ops excluding chosen (if op_logits available)
                    alternatives: list[tuple[str, float]] = []
                    if action_result.op_logits is not None:
                        op_probs = F.softmax(action_result.op_logits[env_idx], dim=-1)
                        # Zero out chosen action to find alternatives
                        alt_probs = op_probs.clone()
                        alt_probs[op_idx] = 0.0
                        top_probs, top_indices = alt_probs.topk(k=min(2, len(alt_probs)), dim=-1)
                        alternatives = [
                            (OP_NAMES[int(idx)], float(prob))
                            for idx, prob in zip(top_indices.tolist(), top_probs.tolist())
                            if prob > 0.0
                        ]

                    hub.emit(TelemetryEvent(
                        event_type=TelemetryEventType.REWARD_COMPUTED,
                        seed_id=seed_state.seed_id if seed_state else None,
                        epoch=epoch,
                        data={
                            "env_id": env_idx,
                            "episode": episodes_completed + env_idx,
                            "ab_group": env_reward_configs[env_idx].reward_mode.value,
                            **reward_components.to_dict(),
                            # Decision Snapshot fields
                            "action_confidence": action_confidence,
                            "value_estimate": value_estimate,
                            "slot_states": slot_states_dict,
                            "alternatives": alternatives,
                        },
                        severity="debug",
                    ))

                # Normalize reward for critic stability (prevents value loss explosion)
                # Keep raw reward for episode_rewards display
                raw_reward = reward
                normalized_reward = reward_normalizer.update_and_normalize(reward)

                # Store transition with action mask (use normalized state to match what policy saw)
                # Keep tensors on policy device to avoid CPU round-trip overhead.
                # get_batches() expects tensors on CPU or will move them - since policy device
                # is consistent across all envs, keeping on GPU is more efficient.
                done = (epoch == max_epochs)
                truncated = done  # All episodes end at max_epochs (time limit truncation)

                # Bootstrap value for truncation: use V(s_{t+1}), not V(s_t)
                # For truncated episodes (time limit), we need the value of the POST-action
                # state to correctly estimate returns. Using V(s_t) causes biased advantages.
                # Collect data now, compute in batch after loop to eliminate per-env GPU syncs.
                if truncated:
                    # Get post-action slot reports (seed states changed by action)
                    post_action_slot_reports = model.get_slot_reports()

                    # Gather post-action active seeds
                    post_action_seeds = []
                    for slot_id in slots:
                        if model.has_active_seed_in_slot(slot_id):
                            seed_state = model.seed_slots[slot_id].state
                            if seed_state is not None:
                                post_action_seeds.append(seed_state)

                    # Compute post-action available slots
                    post_action_available = sum(
                        1 for slot_id in slots if not model.has_active_seed_in_slot(slot_id)
                    )

                    # Build post-action signals (epoch/accuracy same, seed states changed)
                    post_action_signals = env_state.signal_tracker.peek(
                        epoch=epoch,
                        global_step=epoch * num_train_batches,
                        train_loss=env_state.train_loss,
                        train_accuracy=env_state.train_acc,
                        val_loss=env_state.val_loss,
                        val_accuracy=env_state.val_acc,
                        active_seeds=post_action_seeds,
                        available_slots=post_action_available,
                    )

                    # Convert to features
                    post_action_features = signals_to_features(
                        post_action_signals,
                        slot_reports=post_action_slot_reports,
                        use_telemetry=use_telemetry,
                        slots=slots,
                        total_params=model.total_params if model else 0,
                        total_seeds=model.total_seeds() if model else 0,
                        max_seeds=effective_max_seeds,
                        slot_config=slot_config,
                    )

                    # H4 FIX: Compute POST-action masks for bootstrap value computation.
                    # After GERMINATE, slot occupancy changes - using pre-action masks causes
                    # incorrect action masking for V(s_{t+1}). The critic needs masks that
                    # reflect the post-action state to correctly estimate continuation value.
                    # NOTE: These are separate from env_masks (pre-action) which go into buffer.
                    ordered = validate_slot_ids(list(slots))
                    post_action_slot_states = build_slot_states(post_action_slot_reports, ordered)
                    bootstrap_masks = compute_action_masks(
                        slot_states=post_action_slot_states,
                        enabled_slots=ordered,
                        total_seeds=model.total_seeds() if model else 0,
                        max_seeds=effective_max_seeds,
                        slot_config=slot_config,
                        device=torch.device(device),
                        topology=task_spec.topology,
                    )

                    # Collect bootstrap data for batched computation (no forward pass yet)
                    bootstrap_data.append({
                        "features": post_action_features,
                        "hidden": env_state.lstm_hidden,  # Updated LSTM state from this step
                        "masks": bootstrap_masks,
                    })

                # Extract pre-action masks for buffer storage (needed for policy evaluation)
                env_masks = {key: masks_batch[key][env_idx] for key in masks_batch}

                # Collect transition data for buffer storage (store after bootstrap computation)
                # Use PRE-STEP hidden states (captured before get_action)
                # This is the hidden state that was INPUT to the network when selecting
                # the action, enabling proper BPTT reconstruction during training.
                hidden_h, hidden_c = pre_step_hiddens[env_idx]

                transitions_data.append({
                    "env_id": env_idx,
                    "state": states_batch_normalized[env_idx],
                    "action_dict": action_dict,
                    "log_probs": {key: head_log_probs[key][env_idx] for key in HEAD_NAMES},
                    "value": value,
                    "reward": normalized_reward,
                    "done": done,
                    "env_masks": env_masks,
                    "hidden_h": hidden_h,
                    "hidden_c": hidden_c,
                    "truncated": truncated,
                })

                env_state.episode_rewards.append(raw_reward)  # Display raw for interpretability

                # Mechanical lifecycle advance (blending/shadowing dwell) AFTER RL transition
                # This ensures state/action/reward alignment - advance happens after the step is recorded
                # Advance ALL enabled slots so non-targeted seeds still progress mechanically.
                # Cleanup per-slot bookkeeping if a seed is auto-culled during step_epoch().
                # Track auto-culls and accumulate penalty for next reward computation.
                for slot_id in slots:
                    was_auto_culled = model.seed_slots[slot_id].step_epoch()
                    if was_auto_culled:
                        # Accumulate auto-cull penalty for next step's reward
                        # (prevents WAIT-spam policies relying on env cleanup)
                        env_state.pending_auto_cull_penalty += reward_config.auto_cull_penalty
                    if not model.has_active_seed_in_slot(slot_id):
                        env_state.seed_optimizers.pop(slot_id, None)
                        env_state.acc_at_germination.pop(slot_id, None)
                        env_state.gradient_ratio_ema.pop(slot_id, None)

                if epoch == max_epochs:
                    env_final_accs[env_idx] = env_state.val_acc
                    env_total_rewards[env_idx] = sum(env_state.episode_rewards)

                    # Compute Shapley contributions at episode end (emits ANALYTICS_SNAPSHOT)
                    if env_state.counterfactual_helper is not None:
                        active_slot_ids = [
                            slot_id for slot_id in slots
                            if model.has_active_seed_in_slot(slot_id)
                            and model.seed_slots[slot_id].alpha > 0
                        ]

                        if active_slot_ids and baseline_accs[env_idx]:
                            # Create evaluate_fn using cached baseline_accs
                            # (avoids expensive re-validation for each counterfactual config)
                            cached_baselines = baseline_accs[env_idx]
                            full_acc = env_state.val_acc
                            full_loss = env_state.val_loss

                            def _make_evaluate_fn() -> Callable[[dict[str, float]], tuple[float, float]]:
                                def evaluate_fn(alpha_settings: dict[str, float]) -> tuple[float, float]:
                                    # All enabled: return full accuracy
                                    if all(a >= 0.99 for a in alpha_settings.values()):
                                        return full_loss, full_acc

                                    # Single slot disabled: use cached baseline
                                    disabled = [s for s, a in alpha_settings.items() if a < 0.01]
                                    if len(disabled) == 1 and disabled[0] in cached_baselines:
                                        return full_loss * 1.1, cached_baselines[disabled[0]]

                                    # Multi-slot disabled: estimate as average of individual ablations
                                    # (not perfect but reasonable for Shapley estimation)
                                    if disabled:
                                        avg_baseline = sum(
                                            cached_baselines.get(s, full_acc) for s in disabled
                                        ) / len(disabled)
                                        return full_loss * 1.2, avg_baseline

                                    return full_loss, full_acc

                                return evaluate_fn

                            try:
                                contributions = env_state.counterfactual_helper.compute_contributions(
                                    slot_ids=active_slot_ids,
                                    evaluate_fn=_make_evaluate_fn(),
                                    epoch=epoch,
                                )
                                _logger.debug(
                                    f"Env {env_idx}: Shapley computed for {len(active_slot_ids)} slots: "
                                    f"{', '.join(f'{s}={c.shapley_mean:.2f}' for s, c in contributions.items())}"
                                )

                                # Emit full counterfactual matrix for Sanctum visualization
                                counterfactual_matrix = env_state.counterfactual_helper.last_matrix
                                if hub and counterfactual_matrix is not None and counterfactual_matrix.configs:
                                    matrix_data = {
                                        "env_id": env_idx,
                                        "slot_ids": list(counterfactual_matrix.configs[0].slot_ids),
                                        "configs": [
                                            {
                                                "seed_mask": list(cfg.config),
                                                "accuracy": cfg.val_accuracy,
                                            }
                                            for cfg in counterfactual_matrix.configs
                                        ],
                                        "strategy": counterfactual_matrix.strategy_used,
                                        "compute_time_ms": counterfactual_matrix.compute_time_seconds * 1000,
                                    }
                                    hub.emit(TelemetryEvent(
                                        event_type=TelemetryEventType.COUNTERFACTUAL_MATRIX_COMPUTED,
                                        data=matrix_data,
                                    ))
                            except Exception as e:
                                _logger.warning(f"Shapley computation failed for env {env_idx}: {e}")

            # PHASE 2: Compute all bootstrap values in single batched forward pass
            # All episodes truncate at max_epochs, so we batch-compute all bootstrap values
            # in one forward pass instead of N separate forward passes (eliminates N-1 GPU syncs)
            # NOTE: bootstrap_data may be empty if no transitions were truncated (all done=True).
            # This is valid - _compute_batched_bootstrap_values returns [] for empty input.
            bootstrap_values = _compute_batched_bootstrap_values(
                agent=agent,
                post_action_data=bootstrap_data,
                obs_normalizer=obs_normalizer,
                device=device,
            )

            # Validate bootstrap value count matches truncated transition count
            truncated_count = sum(1 for t in transitions_data if t["truncated"])
            assert len(bootstrap_values) == truncated_count, (
                f"Bootstrap value count mismatch: expected {truncated_count} values "
                f"for truncated transitions, got {len(bootstrap_values)}. "
                f"Data collection failed in Phase 1."
            )

            # PHASE 3: Store transitions to buffer with pre-computed bootstrap values
            bootstrap_idx = 0
            for transition in transitions_data:
                # Get bootstrap value if this transition was truncated
                if transition["truncated"]:
                    bootstrap_value = bootstrap_values[bootstrap_idx]
                    bootstrap_idx += 1
                else:
                    bootstrap_value = 0.0

                # Store transition to buffer
                action_dict = transition["action_dict"]
                log_probs = transition["log_probs"]
                env_masks = transition["env_masks"]

                agent.buffer.add(
                    env_id=transition["env_id"],
                    state=transition["state"],
                    slot_action=action_dict["slot"],
                    blueprint_action=action_dict["blueprint"],
                    blend_action=action_dict["blend"],
                    tempo_action=action_dict["tempo"],
                    op_action=action_dict["op"],
                    slot_log_prob=log_probs["slot"],
                    blueprint_log_prob=log_probs["blueprint"],
                    blend_log_prob=log_probs["blend"],
                    tempo_log_prob=log_probs["tempo"],
                    op_log_prob=log_probs["op"],
                    value=transition["value"],
                    reward=transition["reward"],
                    done=transition["done"],
                    slot_mask=env_masks["slot"],
                    blueprint_mask=env_masks["blueprint"],
                    blend_mask=env_masks["blend"],
                    tempo_mask=env_masks["tempo"],
                    op_mask=env_masks["op"],
                    hidden_h=transition["hidden_h"],
                    hidden_c=transition["hidden_c"],
                    truncated=transition["truncated"],
                    bootstrap_value=bootstrap_value,
                )

                # Episode boundary: end episode on done
                if transition["done"]:
                    agent.buffer.end_episode(env_id=transition["env_id"])

            throughput_step_time_ms_sum += step_timer.stop()  # GPU-accurate timing (P4-1)
            throughput_dataloader_wait_ms_sum += dataloader_wait_ms_epoch

        # PPO Update after all episodes in batch complete
        # Truncation bootstrapping: Episodes end at max_epochs (time limit), not natural
        # termination. Each transition stores its bootstrap_value (V(s_final)) which GAE
        # uses instead of 0 for truncated episodes. This prevents systematic downward bias
        # in advantage estimates.
        #
        # Multiple updates per batch improves sample efficiency by reusing data.
        # With KL early stopping, the policy won't diverge too far from the
        # data collection distribution even with multiple updates.
        metrics = {}
        ppo_grad_norm: float | None = None
        ppo_update_time_ms: float | None = None

        # If any Governor rollback occurred, clear only the affected env transitions.
        # This is more sample-efficient than discarding the entire batch.
        anomaly_report: AnomalyReport | None = None
        rollback_env_indices = [i for i, occurred in enumerate(env_rollback_occurred) if occurred]

        if rollback_env_indices:
            # Clear only the affected envs (preserves valid transitions from other envs).
            # Note: This must happen BEFORE _run_ppo_updates(), which calls
            # buffer.normalize_advantages(). Clearing sets step_counts[env_id]=0,
            # so normalize_advantages() will exclude cleared envs from the
            # normalization statistics (mean/std computed only on remaining data).

            # Track sample efficiency metrics BEFORE clearing
            transitions_before = len(agent.buffer)
            transitions_in_cleared_envs = sum(
                agent.buffer.step_counts[i] for i in rollback_env_indices
            )

            for env_idx in rollback_env_indices:
                agent.buffer.clear_env(env_idx)

            transitions_after = len(agent.buffer)
            preservation_ratio = transitions_after / max(transitions_before, 1)

            if hub:
                hub.emit(TelemetryEvent(
                    event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
                    epoch=batch_epoch_id,  # Monotonic batch epoch id (post-batch)
                    severity="warning",
                    message=f"Cleared {len(rollback_env_indices)} env(s) due to Governor rollback",
                    data={
                        "reason": "governor_rollback_partial",
                        "cleared_envs": rollback_env_indices,
                        "batch": batch_idx + 1,
                        "episodes_completed": batch_epoch_id,
                        "inner_epoch": epoch,
                        # Sample efficiency telemetry
                        "transitions_before": transitions_before,
                        "transitions_discarded": transitions_in_cleared_envs,
                        "transitions_preserved": transitions_after,
                        "preservation_ratio": preservation_ratio,
                    },
                ))

        # Only skip PPO update if ALL envs were cleared (no valid data left)
        update_skipped = len(agent.buffer) == 0
        if update_skipped:
            if hub:
                hub.emit(TelemetryEvent(
                    event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
                    epoch=batch_epoch_id,
                    severity="warning",
                    message="Buffer empty after rollback clearing - skipping update",
                    data={
                        "reason": "buffer_empty",
                        "skipped": True,
                        "batch": batch_idx + 1,
                    },
                ))
        else:
            update_start = time.perf_counter()
            metrics = _run_ppo_updates(
                agent=agent,
                ppo_updates_per_batch=ppo_updates_per_batch,
                raw_states_for_normalizer_update=raw_states_for_normalizer_update,
                obs_normalizer=obs_normalizer,
                use_amp=amp,
            )
            ppo_update_time_ms = (time.perf_counter() - update_start) * 1000.0
            ppo_grad_norm = compute_grad_norm_surrogate(agent.network)

            # === Anomaly Detection ===
            # Use check_all() for comprehensive anomaly detection
            metric_values = [v for v in metrics.values() if isinstance(v, (int, float))]
            has_nan = any(math.isnan(v) for v in metric_values)
            has_inf = any(math.isinf(v) for v in metric_values)

            anomaly_report = anomaly_detector.check_all(
                ratio_max=metrics.get("ratio_max", 1.0),
                ratio_min=metrics.get("ratio_min", 1.0),
                explained_variance=metrics.get("explained_variance", 0.0),
                has_nan=has_nan,
                has_inf=has_inf,
                current_episode=batch_epoch_id,
                total_episodes=total_episodes,
            )

            # Gradient drift detection (P4-9) - catches slow degradation
            if grad_ema_tracker is not None and ppo_grad_norm is not None:
                # Simple gradient health: 1.0 if norm in [0.01, 100], scales down outside
                grad_health = 1.0 if 0.01 <= ppo_grad_norm <= 100.0 else max(0.0, 1.0 - abs(ppo_grad_norm - 50) / 100)
                has_drift, drift_metrics = grad_ema_tracker.check_drift(ppo_grad_norm, grad_health)
                if has_drift:
                    drift_report = anomaly_detector.check_gradient_drift(
                        norm_drift=drift_metrics["norm_drift"],
                        health_drift=drift_metrics["health_drift"],
                    )
                    if drift_report.has_anomaly:
                        anomaly_report.has_anomaly = True
                        anomaly_report.anomaly_types.extend(drift_report.anomaly_types)
                        anomaly_report.details.update(drift_report.details)

            collect_debug_anomaly = (
                telemetry_config is not None
                and telemetry_config.should_collect("debug")
                and telemetry_config.per_layer_gradients
            )
            _emit_anomaly_diagnostics(
                hub=hub,
                anomaly_report=anomaly_report,
                agent=agent,
                batch_epoch_id=batch_epoch_id,
                batch_idx=batch_idx,
                max_epochs=max_epochs,
                total_episodes=total_episodes,
                collect_debug=collect_debug_anomaly,
                ratio_diagnostic=metrics.get("ratio_diagnostic"),
            )

        # Telemetry escalation countdown happens once per batch
        _handle_telemetry_escalation(anomaly_report, telemetry_config)

        # Track results
        avg_acc = sum(env_final_accs) / len(env_final_accs)
        avg_reward = sum(env_total_rewards) / len(env_total_rewards)

        recent_accuracies.append(avg_acc)
        recent_rewards.append(avg_reward)
        if len(recent_accuracies) > 10:
            recent_accuracies.pop(0)
            recent_rewards.pop(0)

        rolling_avg_acc = sum(recent_accuracies) / len(recent_accuracies)

        episodes_completed += envs_this_batch
        if hub:
            avg_step_time_ms = throughput_step_time_ms_sum / max(max_epochs, 1)
            avg_dataloader_wait_ms = throughput_dataloader_wait_ms_sum / max(max_epochs, 1)
            for env_id in range(envs_this_batch):
                emit_throughput(
                    hub=hub,
                    env_id=env_id,
                    batch_idx=batch_idx + 1,
                    episodes_completed=episodes_completed,
                    step_time_ms=avg_step_time_ms,
                    dataloader_wait_ms=avg_dataloader_wait_ms,
                )
            if telemetry_config is None or telemetry_config.should_collect("ops_normal"):
                for env_id in range(envs_this_batch):
                    summary = reward_summary_accum[env_id]
                    count = summary["count"]
                    if count < 1:
                        continue
                    payload = {
                        "bounded_attribution": summary["bounded_attribution"] / count,
                        "compute_rent": summary["compute_rent"] / count,
                        "total_reward": summary["total_reward"] / count,
                    }
                    emit_reward_summary(
                        hub=hub,
                        env_id=env_id,
                        batch_idx=batch_idx + 1,
                        episodes_completed=episodes_completed,
                        summary=payload,
                    )

        total_actions = {op.name: 0 for op in LifecycleOp}
        successful_actions = {op.name: 0 for op in LifecycleOp}
        for env_state in env_states:
            for a, c in env_state.action_counts.items():
                total_actions[a] += c
            for a, c in env_state.successful_action_counts.items():
                successful_actions[a] += c

        if hub:
            # Emit PPO telemetry (only for non-skipped updates).
            # Note: clip_fraction, ratio_*, explained_variance not available in recurrent path.
            if metrics:
                payload = dict(metrics)
                payload["train_steps"] = agent.train_steps
                payload["entropy_coef"] = agent.get_entropy_coef()
                payload["avg_accuracy"] = avg_acc
                payload["avg_reward"] = avg_reward
                payload["rolling_avg_accuracy"] = rolling_avg_acc
                # Collect layer gradient health for telemetry (Task 1)
                if telemetry_config is None or telemetry_config.should_collect("debug"):
                    try:
                        layer_stats = collect_per_layer_gradients(agent.policy)
                        layer_health = aggregate_layer_gradient_health(layer_stats)
                        payload.update(layer_health)
                    except Exception:
                        pass  # Graceful degradation if collection fails
                emit_ppo_update_event(
                    hub=hub,
                    metrics=payload,
                    episodes_completed=episodes_completed,
                    batch_idx=batch_idx,
                    epoch=epoch,
                    optimizer=agent.optimizer,
                    grad_norm=ppo_grad_norm,
                    update_time_ms=ppo_update_time_ms,
                )
            emit_action_distribution(
                hub=hub,
                batch_idx=batch_idx + 1,
                episodes_completed=episodes_completed,
                action_counts=total_actions,
                success_counts=successful_actions,
            )
            if sum(mask_total.values()) > 0:
                emit_mask_hit_rates(
                    hub=hub,
                    batch_idx=batch_idx + 1,
                    episodes_completed=episodes_completed,
                    mask_hits=mask_hits,
                    mask_total=mask_total,
                )

            # Emit ANALYTICS_SNAPSHOT for dashboard full-state sync (P1-07).
            # Always emitted so dashboards stay in sync even when PPO update is skipped.
            total_seeds_created = sum(es.seeds_created for es in env_states)
            total_seeds_fossilized = sum(es.seeds_fossilized for es in env_states)
            hub.emit(TelemetryEvent(
                event_type=TelemetryEventType.ANALYTICS_SNAPSHOT,
                epoch=episodes_completed,
                data={
                    "inner_epoch": epoch,
                    "batch": batch_idx + 1,
                    "accuracy": rolling_avg_acc,
                    "host_accuracy": avg_acc,  # Per-batch accuracy
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

            # Emit training progress events BEFORE EPOCH_COMPLETED so they are included
            # in the committed snapshot for this batch epoch.
            if prev_rolling_avg_acc is not None:
                rolling_delta = rolling_avg_acc - prev_rolling_avg_acc

                if abs(rolling_delta) < plateau_threshold:  # True plateau - no significant change
                    hub.emit(TelemetryEvent(
                        event_type=TelemetryEventType.PLATEAU_DETECTED,
                        data={
                            "batch": batch_idx + 1,
                            "rolling_delta": rolling_delta,
                            "rolling_avg_accuracy": rolling_avg_acc,
                            "prev_rolling_avg_accuracy": prev_rolling_avg_acc,
                            "episodes_completed": episodes_completed,
                        },
                    ))
                elif rolling_delta < -improvement_threshold:  # Significant degradation
                    hub.emit(TelemetryEvent(
                        event_type=TelemetryEventType.DEGRADATION_DETECTED,
                        data={
                            "batch": batch_idx + 1,
                            "rolling_delta": rolling_delta,
                            "rolling_avg_accuracy": rolling_avg_acc,
                            "prev_rolling_avg_accuracy": prev_rolling_avg_acc,
                            "episodes_completed": episodes_completed,
                        },
                    ))
                elif rolling_delta > improvement_threshold:  # Significant improvement
                    hub.emit(TelemetryEvent(
                        event_type=TelemetryEventType.IMPROVEMENT_DETECTED,
                        data={
                            "batch": batch_idx + 1,
                            "rolling_delta": rolling_delta,
                            "rolling_avg_accuracy": rolling_avg_acc,
                            "prev_rolling_avg_accuracy": prev_rolling_avg_acc,
                            "episodes_completed": episodes_completed,
                        },
                    ))

            # BATCH_EPOCH_COMPLETED: Commit barrier for Karn.
            # This is batch-level (no env_id), distinct from per-env EPOCH_COMPLETED.
            # Must be emitted LAST for each batch epoch.
            total_train_correct = sum(train_corrects)
            total_train_samples = sum(train_totals)
            total_val_correct = sum(val_corrects)
            total_val_samples = sum(val_totals)

            # Detect plateau: accuracy improvement < 0.5% from previous batch
            if prev_rolling_avg_acc is not None:
                accuracy_delta = abs(rolling_avg_acc - prev_rolling_avg_acc)
                plateau_detected = accuracy_delta < 0.5
            else:
                plateau_detected = False  # First batch, no plateau possible

            hub.emit(TelemetryEvent(
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

            prev_rolling_avg_acc = rolling_avg_acc

        # Emit periodic analytics table snapshots via telemetry (no print).
        if not quiet_analytics and episodes_completed % 5 == 0 and len(analytics.stats) > 0:
            summary_table = analytics.summary_table()
            scoreboard_tables = {
                env_idx: analytics.scoreboard_table(env_idx)
                for env_idx in range(n_envs)
                if env_idx in analytics.scoreboards
            }
            message = summary_table
            if scoreboard_tables:
                message = f"{summary_table}\n" + "\n".join(scoreboard_tables.values())

            hub.emit(TelemetryEvent(
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

        history.append({
            'batch': batch_idx + 1,
            'episodes': episodes_completed,
            'env_accuracies': list(env_final_accs),
            'env_rewards': list(env_total_rewards),
            'avg_accuracy': avg_acc,
            'rolling_avg_accuracy': rolling_avg_acc,
            'avg_reward': avg_reward,
            'action_counts': total_actions,
            'entropy_coef': agent.get_entropy_coef(),
            **metrics,
        })

        if rolling_avg_acc > best_avg_acc:
            best_avg_acc = rolling_avg_acc
            # Store on CPU to save GPU memory (checkpoint is rarely loaded)
            best_state = {k: v.cpu().clone() for k, v in agent.network.state_dict().items()}

        batch_idx += 1

    if best_state:
        agent.network.load_state_dict(best_state)
        if hub:
            hub.emit(TelemetryEvent(
                event_type=TelemetryEventType.CHECKPOINT_LOADED,
                message="Loaded best weights",
                data={"source": "best_state", "avg_accuracy": best_avg_acc},
            ))

    if save_path:
        agent.save(save_path, metadata={
            'n_episodes': episodes_completed,  # Total episodes trained (for resume)
            'n_envs': n_envs,
            'max_epochs': max_epochs,
            'best_avg_accuracy': best_avg_acc,
            'use_telemetry': use_telemetry,
            'seed': seed,
            'obs_normalizer_mean': obs_normalizer.mean.tolist(),
            'obs_normalizer_var': obs_normalizer.var.tolist(),
            'obs_normalizer_count': obs_normalizer.count.item(),
            'obs_normalizer_momentum': obs_normalizer.momentum,
            # Reward normalizer state (P1-6 fix: prevents value function instability on resume)
            'reward_normalizer_mean': reward_normalizer.mean,
            'reward_normalizer_m2': reward_normalizer.m2,
            'reward_normalizer_count': reward_normalizer.count,
        })
        if hub:
            hub.emit(TelemetryEvent(
                event_type=TelemetryEventType.CHECKPOINT_SAVED,
                message=f"Model saved to {save_path}",
                data={"path": str(save_path), "avg_accuracy": best_avg_acc},
            ))

    # Add analytics to final history entry
    if history:
        history[-1]["blueprint_analytics"] = analytics.snapshot()

    # A/B Test Summary
    if ab_reward_modes is not None and not quiet_analytics:
        print("\n" + "=" * 60)
        print("A/B TEST RESULTS")
        print("=" * 60)

        # Group episodes by reward mode
        from collections import defaultdict
        ab_groups = defaultdict(list)

        # Iterate through batches and environments to collect per-episode data
        for batch_data in history:
            env_accs = batch_data.get("env_accuracies", [])
            env_rews = batch_data.get("env_rewards", [])

            for env_idx in range(len(env_accs)):
                # Determine which reward mode this environment used
                mode = env_reward_configs[env_idx].reward_mode.value
                ab_groups[mode].append({
                    "episode_reward": env_rews[env_idx] if env_idx < len(env_rews) else 0,
                    "final_accuracy": env_accs[env_idx],
                })

        for mode, episodes in sorted(ab_groups.items()):
            rewards = [ep["episode_reward"] for ep in episodes]
            accuracies = [ep["final_accuracy"] for ep in episodes]
            avg_reward = sum(rewards) / len(rewards) if rewards else 0
            avg_acc = sum(accuracies) / len(accuracies) if accuracies else 0
            min_rwd = min(rewards) if rewards else 0
            max_rwd = max(rewards) if rewards else 0
            print(f"\n{mode.upper()} ({len(episodes)} episodes):")
            print(f"  Avg Episode Reward: {avg_reward:.2f}")
            print(f"  Avg Final Accuracy: {avg_acc:.2f}%")
            print(f"  Reward Range: [{min_rwd:.2f}, {max_rwd:.2f}]")
        print("=" * 60)

    return agent, history


__all__ = [
    "ParallelEnvState",
    "train_ppo_vectorized",
]
