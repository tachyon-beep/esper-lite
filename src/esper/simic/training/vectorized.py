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
import os
import random
import threading
import time
import warnings
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

_logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.amp as torch_amp

# NOTE: get_task_spec imported lazily inside train_ppo_vectorized to avoid circular import:
#   runtime -> simic.rewards -> simic -> simic.training -> vectorized -> runtime
from esper.utils.data import SharedBatchIterator
from esper.leyline import (
    AlphaMode,
    AlphaAlgorithm,
    SeedStage,
    SeedTelemetry,
    TelemetryEvent,
    TelemetryEventType,
    TrainingStartedPayload,
    SlotConfig,
    DEFAULT_GAMMA,
    DEFAULT_GAE_LAMBDA,
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
    AlphaCurveAction,
    AlphaSpeedAction,
    AlphaTargetAction,
    ALPHA_SPEED_TO_STEPS,
    ALPHA_TARGET_VALUES,
    LifecycleOp,
    OP_NAMES,
    BLUEPRINT_IDS,
    STYLE_ALPHA_ALGORITHMS,
    STYLE_BLEND_IDS,
    TempoAction,
    TEMPO_TO_EPOCHS,
    OP_WAIT,
    OP_GERMINATE,
    OP_SET_ALPHA_TARGET,
    OP_PRUNE,
    OP_FOSSILIZE,
    OP_ADVANCE,
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
from esper.tamiyo.policy.features import (
    MULTISLOT_FEATURE_SIZE,
    get_feature_size,
    batch_obs_to_features,
)
from esper.simic.agent import PPOAgent
from esper.tamiyo.policy import create_policy
from esper.simic.rewards import (
    compute_reward,
    compute_loss_reward,
    RewardMode,
    RewardFamily,
    ContributionRewardConfig,
    SeedInfo,
)
from esper.leyline import DEFAULT_MIN_FOSSILIZE_CONTRIBUTION, MIN_PRUNE_AGE
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
    VectorizedEmitter,
)
from .parallel_env_state import ParallelEnvState
from .helpers import compute_rent_and_shock_inputs


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


def _aggregate_ppo_metrics(update_metrics: list[dict]) -> dict:
    """Aggregate metrics across multiple PPO updates for a single batch."""
    if not update_metrics:
        return {}

    aggregated: dict = {}
    keys = {k for metrics in update_metrics for k in metrics.keys()}
    for key in keys:
        values = [
            metrics[key]
            for metrics in update_metrics
            if key in metrics and metrics[key] is not None
        ]
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
    amp_dtype: torch.dtype | None = None,  # None=float16 for backwards compat
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
            # Use provided dtype, default to float16 for backwards compatibility
            dtype = amp_dtype if amp_dtype is not None else torch.float16
            with torch_amp.autocast(device_type="cuda", dtype=dtype):  # type: ignore[call-arg]
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
    group_id: str | None = None,
) -> None:
    """Emit anomaly telemetry, optionally with expensive diagnostics when debug is enabled."""
    if hub is None or anomaly_report is None or not anomaly_report.has_anomaly:
        return

    from esper.leyline import AnomalyDetectedPayload

    event_type_map = {
        "ratio_explosion": TelemetryEventType.RATIO_EXPLOSION_DETECTED,
        "ratio_collapse": TelemetryEventType.RATIO_COLLAPSE_DETECTED,
        "value_collapse": TelemetryEventType.VALUE_COLLAPSE_DETECTED,
        "numerical_instability": TelemetryEventType.NUMERICAL_INSTABILITY_DETECTED,
    }

    gradient_stats = None
    stability_report = None
    if collect_debug:
        gradient_stats = collect_per_layer_gradients(agent.policy._network)
        stability_report = check_numerical_stability(agent.policy._network)

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
                group_id=group_id,
                data=payload,
                severity="debug" if collect_debug else "warning",
            )
        )


def loss_and_correct(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    task_type: str,
    elementwise: bool = False,
):
    """Compute loss and correct counts for classification or LM.

    Compiled module-level function to avoid closure overhead and kernel launch stalls.
    """
    if task_type == "lm":
        vocab = outputs.size(-1)
        loss = criterion(outputs.view(-1, vocab), targets.view(-1))
        predicted = outputs.argmax(dim=-1)
        correct = predicted.eq(targets)
        if not elementwise:
            correct = correct.sum()
        total = targets.numel()
    else:
        loss = criterion(outputs, targets)
        _, predicted = outputs.max(1)
        correct = predicted.eq(targets)
        if not elementwise:
            correct = correct.sum()
        total = targets.size(0)
    return loss, correct, total


# NOTE: torch.compile() of this helper has proven unstable across CUDA/Python
# combinations (observed: TorchInductor device-side asserts during long runs).
# Keep eager execution here; the PPO policy network remains the primary target
# for compilation speedups.
_compiled_loss_and_correct = loss_and_correct


# =============================================================================
# Vectorized PPO Training
# =============================================================================


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
    gae_lambda: float = DEFAULT_GAE_LAMBDA,  # From leyline
    ppo_updates_per_batch: int = 1,
    save_path: str | None = None,
    resume_path: str | None = None,
    seed: int = 42,
    num_workers: int | None = None,
    batch_size_per_env: int | None = None,
    gpu_preload: bool = False,
    amp: bool = False,
    amp_dtype: str = "auto",  # "auto", "float16", "bfloat16", or "off"
    compile_mode: str = "default",  # "default", "max-autotune", "reduce-overhead", "off"
    lstm_hidden_dim: int = DEFAULT_LSTM_HIDDEN_DIM,
    chunk_length: int = DEFAULT_EPISODE_LENGTH,  # Must match max_epochs (from leyline)
    telemetry_config: "TelemetryConfig | None" = None,
    telemetry_lifecycle_only: bool = False,
    plateau_threshold: float = 0.5,
    improvement_threshold: float = 2.0,
    gradient_telemetry_stride: int = 10,
    slots: list[str] | None = None,
    max_seeds: int | None = None,
    reward_mode: str = "shaped",
    param_budget: int = 500_000,
    param_penalty_weight: float = 0.1,
    sparse_reward_scale: float = 1.0,
    reward_family: str = "contribution",
    ab_reward_modes: list[str] | None = None,
    permissive_gates: bool = True,
    quiet_analytics: bool = False,
    telemetry_dir: str | None = None,
    ready_event: "threading.Event | None" = None,
    group_id: str = "default",  # A/B testing group identifier
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
        batch_size_per_env: Optional per-environment batch size override. When unset, uses
            the task defaults (with CIFAR-10 tuned for high throughput).
        plateau_threshold: Rolling average delta threshold below which training is considered
            plateaued (emits PLATEAU_DETECTED event). Compares current vs previous batch's
            rolling average. Scale-dependent: adjust for accuracy scales (e.g., 0-1 vs 0-100).
        improvement_threshold: Rolling average delta threshold above which training shows
            significant improvement/degradation (emits IMPROVEMENT_DETECTED/DEGRADATION_DETECTED).
            Events align with displayed rolling_avg_accuracy trend.

    Returns:
        Tuple of (trained_agent, training_history)
    """
    # NOTE: PyTorch's internal FX passes use tqdm, which attempts to create a
    # multiprocessing lock by default. When Esper runs a Textual TUI (Sanctum/
    # Overwatch), terminal FD manipulation can break multiprocessing spawn and
    # crash torch.compile with errors like "bad value(s) in fds_to_keep".
    #
    # Force tqdm to use a thread-only lock (no multiprocessing) to keep compile
    # stable under TUI mode.
    try:
        from threading import RLock

        from tqdm import tqdm

        tqdm.set_lock(RLock())
    except (ImportError, AttributeError) as e:
        # Best-effort: tqdm isn't required, and compile should still work in
        # environments where multiprocessing locks are healthy.
        _logger.debug("tqdm lock configuration skipped: %s", e)

    from esper.tolaria import create_model
    from esper.tamiyo import SignalTracker

    def _parse_device(device_str: str) -> torch.device:
        """Parse a device string with an actionable error on failure."""
        try:
            return torch.device(device_str)
        except Exception as exc:  # pragma: no cover - torch raises varied exceptions
            raise ValueError(f"Invalid device '{device_str}': {exc}") from exc

    def _validate_cuda_device(device_str: str, *, require_explicit_index: bool) -> None:
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

    # Derive slot_config from host's injection specs, filtered to requested slots
    # Create a temporary model to query the host's injection topology
    temp_device = "cpu"  # Use CPU for temp model to avoid GPU allocation
    temp_model = create_model(
        task=task_spec, device=temp_device, slots=slots, permissive_gates=permissive_gates
    )
    # Filter specs to only include requested slots (not all host injection points)
    enabled_specs = [
        spec for spec in temp_model.host.injection_specs()
        if spec.slot_id in temp_model.seed_slots
    ]
    slot_config = SlotConfig.from_specs(enabled_specs)
    # Calculate host_params while we have the model (constant across all envs)
    host_params_baseline = sum(
        p.numel() for p in temp_model.host.parameters() if p.requires_grad
    )
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
    if (
        reward_family_enum == RewardFamily.LOSS
        and reward_mode_enum != RewardMode.SHAPED
    ):
        raise ValueError(
            "reward_mode applies only to contribution rewards. Use default when reward_family=loss."
        )

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
            {mode: ab_reward_modes.count(mode) for mode in set(ab_reward_modes)},
        )
    else:
        env_reward_configs = [reward_config] * n_envs

    # Map environments to devices in round-robin (needed for SharedBatchIterator)
    env_device_map = [devices[i % len(devices)] for i in range(n_envs)]

    # DataLoader settings (used for SharedBatchIterator + diagnostics).
    default_batch_size_per_env = task_spec.dataloader_defaults.get("batch_size", 128)
    if task_spec.name == "cifar10":
        default_batch_size_per_env = 512  # High-throughput setting for CIFAR
    effective_batch_size_per_env = (
        batch_size_per_env
        if batch_size_per_env is not None
        else default_batch_size_per_env
    )
    if effective_batch_size_per_env < 1:
        raise ValueError(
            f"batch_size_per_env must be >= 1 (got {effective_batch_size_per_env})"
        )
    effective_workers = num_workers if num_workers is not None else 4

    # State dimension: base features (dynamic based on slot count) + telemetry features when enabled.
    # For 3 slots: 23 base + 3*25 slot features = 98, plus 3*17 telemetry = 149.
    base_feature_size = get_feature_size(slot_config)
    telemetry_size = (
        slot_config.num_slots * SeedTelemetry.feature_dim() if use_telemetry else 0
    )
    state_dim = base_feature_size + telemetry_size

    # Use EMA momentum for stable normalization during long training runs
    # (prevents distribution shift that can break PPO ratio calculations)
    obs_normalizer = RunningMeanStd((state_dim,), device=device, momentum=0.99)

    # Reward normalizer for critic stability (prevents value loss explosion)
    # Essential after ransomware fix where reward magnitudes changed significantly
    reward_normalizer = RewardNormalizer(clip=10.0)

    # ==========================================================================
    # Blueprint Analytics + Nissa Hub Wiring
    # ==========================================================================
    hub = get_hub()
    analytics = BlueprintAnalytics(quiet=quiet_analytics)
    hub.add_backend(analytics)

    # Ops-normal telemetry gates (UI snapshots, per-step decisions, etc).
    # NOTE: `use_telemetry` controls whether telemetry features are part of the
    # RL observation vector; it must not be used to gate ops-normal UI emission.
    ops_telemetry_enabled = (
        not telemetry_lifecycle_only
        and (telemetry_config is None or telemetry_config.should_collect("ops_normal"))
    )

    # Optional file-based telemetry logging (for programmatic callers that bypass scripts/train.py)
    if telemetry_dir and use_telemetry:
        hub.add_backend(DirectoryOutput(telemetry_dir))
        _logger.info("Telemetry logging to: %s", telemetry_dir)

    anomaly_detector = AnomalyDetector()
    start_episode = 0

    # Mapping diagnostics: required for multi-GPU sign-off.
    unique_env_devices = list(dict.fromkeys(devices))
    env_device_counts = {dev: 0 for dev in unique_env_devices}
    for mapped_device in env_device_map:
        env_device_counts[mapped_device] += 1

    # Convert episode-based annealing to step-based (respecting multi-update batches)
    entropy_anneal_steps = _calculate_entropy_anneal_steps(
        entropy_anneal_episodes=entropy_anneal_episodes,
        n_envs=n_envs,
        ppo_updates_per_batch=ppo_updates_per_batch,
    )

    # Create per-environment emitters for consolidated telemetry logic
    emitters = [
        VectorizedEmitter(
            env_id=i,
            device=env_device_map[i],
            hub=hub,
            telemetry_config=telemetry_config,
            quiet_analytics=quiet_analytics,
        )
        for i in range(n_envs)
    ]
    batch_emitter = emitters[0]

    # Create or resume PPO agent
    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=True)
        agent = PPOAgent.load(resume_path, device=device)

        # Restore observation normalizer state
        metadata = checkpoint.get("metadata", {})
        if "obs_normalizer_mean" in metadata:
            obs_normalizer.mean = torch.tensor(
                metadata["obs_normalizer_mean"], device=device
            )
            obs_normalizer.var = torch.tensor(
                metadata["obs_normalizer_var"], device=device
            )
            obs_normalizer._device = device
            if "obs_normalizer_count" in metadata:
                obs_normalizer.count = torch.tensor(
                    metadata["obs_normalizer_count"], device=device
                )
            if "obs_normalizer_momentum" in metadata:
                obs_normalizer.momentum = metadata["obs_normalizer_momentum"]

        # Restore reward normalizer state
        if "reward_normalizer_mean" in metadata:
            reward_normalizer.mean = metadata["reward_normalizer_mean"]
            reward_normalizer.m2 = metadata["reward_normalizer_m2"]
            reward_normalizer.count = metadata["reward_normalizer_count"]

        if "n_episodes" in metadata:
            start_episode = metadata["n_episodes"]

        # Emit checkpoint loaded event
        hub.emit(
            TelemetryEvent(
                event_type=TelemetryEventType.CHECKPOINT_LOADED,
                group_id=group_id,
                data={
                    "path": str(resume_path),
                    "start_episode": start_episode,
                    "source": "resume",
                },
            )
        )
    else:
        # TUI mode (Sanctum/Overwatch) is optimized for debuggability. Disable
        # torch.compile here to avoid TorchInductor failures that are difficult
        # to recover from in an interactive session.
        # Determine effective compile mode: quiet_analytics disables compilation
        effective_compile_mode = compile_mode if not quiet_analytics else "off"

        # Create policy via Tamiyo factory
        # IMPORTANT: Pass actual slot_config to ensure action heads/masks align
        # with environment slot ordering (critical for non-default slot layouts)
        policy = create_policy(
            policy_type="lstm",
            state_dim=state_dim,
            slot_config=slot_config,
            device=device,
            compile_mode=effective_compile_mode,
            lstm_hidden_dim=lstm_hidden_dim,
        )

        # Create agent with injected policy
        agent = PPOAgent(
            policy=policy,
            slot_config=slot_config,
            lr=lr,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_ratio=clip_ratio,
            entropy_coef=entropy_coef,
            entropy_coef_start=entropy_coef_start,
            entropy_coef_end=entropy_coef_end,
            entropy_coef_min=entropy_coef_min,
            adaptive_entropy_floor=adaptive_entropy_floor,
            entropy_anneal_steps=entropy_anneal_steps,
            device=device,
            chunk_length=chunk_length,
            num_envs=n_envs,
            max_steps_per_env=max_epochs,
        )

    # Emit TRAINING_STARTED to activate Karn (Sanctum/Overwatch) and capture run config.
    entropy_anneal_summary = None
    if entropy_anneal_episodes > 0:
        entropy_anneal_summary = {
            "start": entropy_coef_start if entropy_coef_start is not None else entropy_coef,
            "end": entropy_coef_end if entropy_coef_end is not None else entropy_coef,
            "episodes": entropy_anneal_episodes,
            "steps": entropy_anneal_steps,
        }

    dataloader_summary = {
        "mode": "gpu_preload" if gpu_preload else "shared_batch_iterator",
        "batch_size_per_env": effective_batch_size_per_env,
        "num_workers": None if gpu_preload else effective_workers,
        "pin_memory": not gpu_preload,
    }

    hub.emit(TelemetryEvent(
        event_type=TelemetryEventType.TRAINING_STARTED,
        group_id=group_id,
        message=(
            f"PPO vectorized training initialized: policy_device={device}, "
            f"env_device_counts={env_device_counts}"
        ),
        data=TrainingStartedPayload(
            n_envs=n_envs,
            max_epochs=max_epochs,
            task=task,
            host_params=host_params_baseline,
            slot_ids=tuple(slot_config.slot_ids),
            seed=seed,
            n_episodes=n_episodes + start_episode,
            lr=lr,
            clip_ratio=clip_ratio,
            entropy_coef=entropy_coef,
            param_budget=param_budget,
            policy_device=device,
            env_devices=tuple(devices),
            # Optional fields
            episode_id=f"ppo_{seed}_{n_episodes}ep",
            resume_path=str(resume_path) if resume_path else "",
            reward_mode=reward_mode,
            start_episode=start_episode,
            entropy_anneal=entropy_anneal_summary,
        ),
    ))

    # Create SharedBatchIterator for parallel data loading
    trainset, testset = task_spec.get_datasets()

    if gpu_preload:
        if "cifar10" not in task_spec.name.lower():
            _logger.warning(
                f"gpu_preload=True is disabled for task '{task_spec.name}' "
                "(SharedGPUBatchIterator supports CIFAR-10 only). "
                "Falling back to standard CPU DataLoader."
            )
            gpu_preload = False

    if gpu_preload:
        from esper.utils.data import SharedGPUBatchIterator

        shared_train_iter = SharedGPUBatchIterator(
            batch_size_per_env=effective_batch_size_per_env,
            n_envs=n_envs,
            env_devices=env_device_map,
            shuffle=True,
            data_root=task_spec.dataloader_defaults.get("data_root", "./data"),
            is_train=True,
        )
        shared_test_iter = SharedGPUBatchIterator(
            batch_size_per_env=effective_batch_size_per_env,
            n_envs=n_envs,
            env_devices=env_device_map,
            shuffle=False,
            data_root=task_spec.dataloader_defaults.get("data_root", "./data"),
            is_train=False,
        )
    else:
        shared_train_iter = SharedBatchIterator(
            trainset,
            batch_size_per_env=effective_batch_size_per_env,
            n_envs=n_envs,
            env_devices=env_device_map,
            num_workers=effective_workers,
            shuffle=True,
        )
        shared_test_iter = SharedBatchIterator(
            testset,
            batch_size_per_env=effective_batch_size_per_env,
            n_envs=n_envs,
            env_devices=env_device_map,
            num_workers=effective_workers,
            shuffle=False,
        )

    num_train_batches = len(shared_train_iter)
    num_test_batches = len(shared_test_iter)

    # Warm up DataLoaders to ensure workers are spawned.
    #
    # CRITICAL: When running a Textual TUI (Sanctum/Overwatch), the main thread
    # must not start the TUI until ALL DataLoader workers have spawned.
    # Textual can modify terminal file descriptors and break multiprocessing
    # spawn. We therefore warm up both train and test loaders here, before we
    # signal `ready_event`.
    # Skip warmup when gpu_preload is True for any CIFAR variant (cifar10, cifar10_deep, cifar10_blind)
    # SharedGPUBatchIterator already has num_workers=0, and iterating it during warmup can cause
    # race conditions when accessing GPU-cached tensors across multiple devices before streams sync.
    is_cifar_task = "cifar10" in task_spec.name.lower()
    if effective_workers > 0 and not (gpu_preload and is_cifar_task):
        _warmup_iter = iter(shared_train_iter)
        try:
            next(_warmup_iter)
        except StopIteration:
            pass
        del _warmup_iter
        _warmup_iter = iter(shared_test_iter)
        try:
            next(_warmup_iter)
        except StopIteration:
            pass
        del _warmup_iter

    # Signal that DataLoaders are ready (workers spawned) - TUI can now safely start
    if ready_event is not None:
        ready_event.set()

    def make_telemetry_callback(
        env_idx: int, device: str
    ) -> Callable[[TelemetryEvent], None]:
        """Create a telemetry callback that injects env_id and device."""
        if not hub:
            return lambda _: None

        def callback(event: TelemetryEvent) -> None:
            emit_with_env_context(hub, env_idx, device, event)

        return callback

    def configure_slot_telemetry(
        env_state: ParallelEnvState,
        inner_epoch: int | None = None,
        global_epoch: int | None = None,
    ) -> None:
        """Configure slot telemetry and fast_mode for an environment."""
        apply_slot_telemetry(
            env_state,
            ops_telemetry_enabled=ops_telemetry_enabled,
            lifecycle_only=telemetry_lifecycle_only,
            inner_epoch=inner_epoch,
            global_epoch=global_epoch,
        )

    # AMP enabled gate - actual GradScaler created per-env in ParallelEnvState
    # to avoid stream race conditions (GradScaler internal state is not stream-safe)
    #
    # Resolve amp_dtype: "auto" detects BF16 support, which eliminates GradScaler overhead.
    # BF16 has same exponent range as FP32, so no loss scaling needed.
    resolved_amp_dtype: torch.dtype | None = None
    use_grad_scaler = False

    if amp and torch.cuda.is_available():
        if amp_dtype == "off":
            # AMP explicitly disabled via dtype
            resolved_amp_dtype = None
        elif amp_dtype == "bfloat16":
            resolved_amp_dtype = torch.bfloat16
            use_grad_scaler = False  # BF16 doesn't need scaler
        elif amp_dtype == "float16":
            resolved_amp_dtype = torch.float16
            use_grad_scaler = True  # FP16 needs GradScaler
        elif amp_dtype == "auto":
            # Auto-detect: use BF16 if supported (Ampere+ GPUs), else FP16
            if torch.cuda.is_bf16_supported():
                resolved_amp_dtype = torch.bfloat16
                use_grad_scaler = False
                _logger.info("AMP auto-detected BF16 support (Ampere+ GPU) - no GradScaler needed")
            else:
                resolved_amp_dtype = torch.float16
                use_grad_scaler = True
                _logger.info("AMP using FP16 with GradScaler (pre-Ampere GPU)")
        else:
            raise ValueError(f"Invalid amp_dtype: {amp_dtype}")

    amp_enabled = resolved_amp_dtype is not None

    def create_env_state(env_idx: int, base_seed: int) -> ParallelEnvState:
        """Create environment state with CUDA stream.

        DataLoaders are now shared via SharedBatchIterator, not per-env.
        """
        env_device = env_device_map[env_idx]
        torch.manual_seed(base_seed + env_idx * 1000)
        random.seed(base_seed + env_idx * 1000)

        model = create_model(
            task=task_spec, device=env_device, slots=slots, permissive_gates=permissive_gates
        )

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
            model.get_host_parameters(),
            lr=task_spec.host_lr,
            momentum=0.9,
            weight_decay=5e-4,
        )

        # Create CUDA stream for this environment
        env_device_obj = torch.device(env_device)
        stream = (
            torch.cuda.Stream(device=env_device_obj)
            if env_device_obj.type == "cuda"
            else None
        )

        # Per-env AMP scaler to avoid stream race conditions (GradScaler state is not stream-safe)
        # Use new torch.amp.GradScaler API (torch.cuda.amp.GradScaler deprecated in PyTorch 2.4+)
        # Note: BF16 doesn't need GradScaler (same exponent range as FP32)
        env_scaler = (
            torch.amp.GradScaler("cuda", enabled=use_grad_scaler)
            if env_device_obj.type == "cuda" and use_grad_scaler
            else None
        )

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
        health_monitor = (
            HealthMonitor(
                store=None,  # TelemetryStore integration deferred
                emit_callback=telemetry_cb,  # Same callback as slots
            )
            if use_telemetry
            else None
        )

        # Create CounterfactualHelper for Shapley value analysis at episode end
        counterfactual_helper = (
            CounterfactualHelper(
                strategy="auto",  # Full factorial for <=4 slots, Shapley sampling otherwise
                shapley_samples=20,
                emit_events=use_telemetry,
            )
            if use_telemetry
            else None
        )

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
        env_state.prev_slot_alphas = {slot_id: 0.0 for slot_id in slots}
        env_state.prev_slot_params = {slot_id: 0 for slot_id in slots}
        # Pre-allocate accumulators to avoid per-epoch allocation churn
        env_state.init_accumulators(slots)
        configure_slot_telemetry(env_state)
        return env_state

    @torch.compiler.disable
    def _collect_gradient_telemetry_for_batch(
        model: "HostWithSeeds",
        slots_with_active_seeds: list[str],
        env_dev: str,
    ) -> dict[str, dict[Any, Any]] | None:
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
            if seed_state and seed_state.stage in (
                SeedStage.TRAINING,
                SeedStage.BLENDING,
            ):
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

    def _parse_sampled_action(
        env_idx: int,
        op_idx: int,
        slot_idx: int,
        style_idx: int,
        alpha_target_idx: int,
        slots: list[str],
        slot_config: SlotConfig,
        model: "SlottedHostProtocol",
    ) -> tuple[str, bool, Any, Any, bool, LifecycleOp, str, Any, float]:
        """Consolidate action derived values and validation logic (Deduplication)."""
        # Use the SAMPLED slot as target (multi-slot support)
        target_slot, slot_is_enabled = _resolve_target_slot(
            slot_idx,
            enabled_slots=slots,
            slot_config=slot_config,
        )

        slot_state = model.seed_slots[target_slot].state if slot_is_enabled else None
        seed_state = (
            slot_state
            if slot_is_enabled and model.has_active_seed_in_slot(target_slot)
            else None
        )

        blend_algorithm_id = STYLE_BLEND_IDS[style_idx]
        alpha_algorithm = STYLE_ALPHA_ALGORITHMS[style_idx]
        alpha_target = ALPHA_TARGET_VALUES[alpha_target_idx]

        action_valid_for_reward = True
        if not slot_is_enabled:
            action_valid_for_reward = False
        elif op_idx == OP_GERMINATE:
            action_valid_for_reward = slot_state is None
        elif op_idx == OP_FOSSILIZE:
            action_valid_for_reward = (
                seed_state is not None and seed_state.stage == SeedStage.HOLDING
            )
        elif op_idx == OP_PRUNE:
            action_valid_for_reward = (
                seed_state is not None
                and seed_state.alpha_controller.alpha_mode == AlphaMode.HOLD
                and seed_state.can_transition_to(SeedStage.PRUNED)
                # BUG-020 fix: enforce MIN_PRUNE_AGE to match masking invariant
                and seed_state.metrics is not None
                and seed_state.metrics.epochs_total >= MIN_PRUNE_AGE
            )
        elif op_idx == OP_SET_ALPHA_TARGET:
            action_valid_for_reward = (
                seed_state is not None
                and seed_state.alpha_controller.alpha_mode == AlphaMode.HOLD
                and seed_state.stage in (SeedStage.BLENDING, SeedStage.HOLDING)
            )
        elif op_idx == OP_ADVANCE:
            action_valid_for_reward = seed_state is not None and seed_state.stage in (
                SeedStage.GERMINATED,
                SeedStage.TRAINING,
                SeedStage.BLENDING,
            )

        action_for_reward = (
            LifecycleOp(op_idx) if action_valid_for_reward else LifecycleOp.WAIT
        )

        return (
            target_slot,
            slot_is_enabled,
            slot_state,
            seed_state,
            action_valid_for_reward,
            action_for_reward,
            blend_algorithm_id,
            alpha_algorithm,
            alpha_target,
        )

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
            slot_id: model.has_active_seed_in_slot(slot_id) for slot_id in slots
        }
        slots_with_active_seeds = [
            slot_id for slot_id, active in active_slots.items() if active
        ]

        # Use CUDA stream for async execution
        stream_ctx = (
            torch.cuda.stream(env_state.stream) if env_state.stream else nullcontext()
        )

        with stream_ctx:
            # Move data asynchronously (no-op if already on device from SharedGPUBatchIterator)
            inputs = inputs.to(env_dev, non_blocking=True)
            targets = targets.to(env_dev, non_blocking=True)

            if env_state.stream and inputs.is_cuda:
                # CRITICAL: SharedGPUBatchIterator returns views on the default stream.
                # Even with wait_stream, passing these views directly to kernels can cause
                # race conditions (Nll_loss assertion failures). Cloning enforces a
                # strong memory dependency on the current stream.
                inputs = inputs.clone()
                targets = targets.clone()
                # Record the new tensors so they aren't freed prematurely
                inputs.record_stream(env_state.stream)
                targets.record_stream(env_state.stream)

            # Ensure per-slot seed optimizers exist for any slot with a live seed.
            # We keep optimizers per-slot to avoid dynamic param-group surgery.
            slots_to_step: list[str] = []
            for slot_id in slots_with_active_seeds:
                # Already filtered to active slots via cache
                seed_state = model.seed_slots[slot_id].state
                if seed_state is None:
                    continue

                # Seeds can continue training through BLENDING/HOLDING/FOSSILIZED.
                slots_to_step.append(slot_id)

                # OPTIMIZATION: Removed expensive parameter-set validation from hot path.
                # Rely on env_state.seed_optimizers.pop() in the action execution block.
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
            # dtype comes from resolved_amp_dtype (BF16 on Ampere+, FP16 otherwise)
            autocast_ctx = (
                torch_amp.autocast(device_type="cuda", dtype=resolved_amp_dtype)  # type: ignore[call-arg]
                if env_state.autocast_enabled and resolved_amp_dtype is not None
                else nullcontext()
            )
            with autocast_ctx:
                outputs = model(inputs)
                loss, correct_tensor, total = _compiled_loss_and_correct(
                    outputs, targets, criterion, task_type=task_spec.task_type
                )

            # Use scaler.scale() only when using FP16 (BF16 doesn't need scaling)
            if env_state.scaler is not None:
                env_state.scaler.scale(loss).backward()
            else:
                loss.backward()
            # Collect gradient telemetry (isolated from torch.compile)
            grad_stats_by_slot = None
            if use_telemetry:
                grad_stats_by_slot = _collect_gradient_telemetry_for_batch(
                    model, slots_with_active_seeds, env_dev
                )

            if env_state.scaler is not None:
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
                    seed_opt = env_state.seed_optimizers[slot_id]
                    # Guard: Only call scaler.step() if optimizer has gradients.
                    # With isolate_gradients=True, seeds may not have grads in the scaled backward.
                    has_grads = any(
                        p.grad is not None for group in seed_opt.param_groups for p in group["params"]
                    )
                    if has_grads:
                        env_state.scaler.step(seed_opt)
                    else:
                        # No grads from scaled backward - step without scaler (no-op if no grads)
                        seed_opt.step()
                env_state.scaler.update()
            else:
                env_state.host_optimizer.step()
                for slot_id in slots_to_step:
                    env_state.seed_optimizers[slot_id].step()

            # Return tensors - .item() called after stream sync
            return loss.detach(), correct_tensor, total, grad_stats_by_slot

    def process_val_batch(
        env_state: ParallelEnvState,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        criterion: nn.Module,
        slots: list[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
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

        stream_ctx = (
            torch.cuda.stream(env_state.stream) if env_state.stream else nullcontext()
        )

        with stream_ctx:
            inputs = inputs.to(env_dev, non_blocking=True)
            targets = targets.to(env_dev, non_blocking=True)

            if env_state.stream and inputs.is_cuda:
                # Resolve race condition (see process_train_batch)
                inputs = inputs.clone()
                targets = targets.clone()
                inputs.record_stream(env_state.stream)
                targets.record_stream(env_state.stream)

            model.eval()
            with torch.inference_mode():
                outputs = model(inputs)
                loss, correct_tensor, total = _compiled_loss_and_correct(
                    outputs, targets, criterion, task_type=task_spec.task_type
                )

            # Return tensors - .item() called after stream sync
            return loss, correct_tensor, total

    def process_fused_val_batch(
        env_state: ParallelEnvState,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        criterion: nn.Module,
        alpha_overrides: dict[str, torch.Tensor],
        num_configs: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Process a fused validation batch with multiple alpha configurations.

        Uses MorphogeneticModel.fused_forward() to saturate GPU and avoid CPU orchestration stalls.

        Args:
            env_state: Parallel environment state
            inputs: Original input tensor [B, ...]
            targets: Original target tensor [B, ...]
            criterion: Loss criterion
            alpha_overrides: Dict mapping slot_id -> override tensor [K*B, 1, 1, 1]
            num_configs: Number of configurations K

        Returns:
            Tuple of (loss_tensor, correct_tensor, total) for the expanded batch.
        """
        model = env_state.model
        env_dev = env_state.env_device
        batch_size = inputs.size(0)

        stream_ctx = (
            torch.cuda.stream(env_state.stream) if env_state.stream else nullcontext()
        )

        with stream_ctx:
            # Move data and expand for all configurations
            inputs = inputs.to(env_dev, non_blocking=True)
            targets = targets.to(env_dev, non_blocking=True)

            # Expand inputs/targets: [K*B, ...]
            # Using repeat() is safe here as data is small and we want contiguous chunks
            fused_inputs = inputs.repeat(num_configs, *([1] * (inputs.dim() - 1)))
            fused_targets = targets.repeat(num_configs, *([1] * (targets.dim() - 1)))
            # PERF: Preserve channels_last memory format after repeat() for CNN throughput.
            # repeat() returns contiguous (NCHW) layout; restore channels_last if input was.
            if inputs.is_contiguous(memory_format=torch.channels_last):
                fused_inputs = fused_inputs.contiguous(memory_format=torch.channels_last)

            if env_state.stream and inputs.is_cuda:
                fused_inputs.record_stream(env_state.stream)
                fused_targets.record_stream(env_state.stream)

            model.eval()
            with torch.inference_mode():
                outputs = model.fused_forward(fused_inputs, alpha_overrides)
                loss, correct_fused, total = _compiled_loss_and_correct(
                    outputs,
                    fused_targets,
                    criterion,
                    task_type=task_spec.task_type,
                    elementwise=True,
                )

            # Sum elementwise correctness per configuration
            # correct_fused shape is [K*B]
            correct_per_config = correct_fused.view(num_configs, batch_size).sum(dim=1)

            return loss, correct_per_config, total

    def batch_signals_to_features(
        batch_signals: list,
        batch_slot_reports: list[dict[str, "SeedStateReport"]],
        use_telemetry: bool,
        max_epochs: int,
        effective_max_seeds: int,
        slot_config: SlotConfig,
        env_states: list[ParallelEnvState],
        device: torch.device,
    ) -> torch.Tensor:
        """Consolidated signals-to-features conversion for all environments."""
        all_total_params = [es.model.total_params for es in env_states]
        all_total_seeds = [es.model.total_seeds() for es in env_states]

        return batch_obs_to_features(
            batch_signals=batch_signals,
            batch_slot_reports=batch_slot_reports,
            use_telemetry=use_telemetry,
            max_epochs=max_epochs,
            total_params=all_total_params,
            total_seeds=all_total_seeds,
            max_seeds=effective_max_seeds,
            slot_config=slot_config,
            device=device,
        )

    history = []
    episode_history = []  # Per-episode tracking for A/B testing
    best_avg_acc = 0.0
    best_state = None
    recent_accuracies = []
    recent_rewards = []
    prev_rolling_avg_acc: float | None = (
        None  # Track previous rolling avg for trend detection
    )

    episodes_completed = start_episode
    total_episodes = (
        n_episodes + start_episode
    )  # Total target including resumed episodes

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
        env_states = [create_env_state(i, base_seed) for i in range(envs_this_batch)]
        criterion = nn.CrossEntropyLoss()

        # Initialize episode for vectorized training
        for env_idx in range(envs_this_batch):
            env_states[env_idx].reset_episode_state(slots)
            agent.buffer.start_episode(env_id=env_idx)

        # Initialize batched LSTM hidden state for all environments
        # (Batched hidden management avoids per-step cat/slice overhead)
        batched_lstm_hidden: tuple[torch.Tensor, torch.Tensor] | None = None

        # Per-env accumulators
        env_final_accs = [0.0] * envs_this_batch
        env_total_rewards = [0.0] * envs_this_batch

        throughput_step_time_ms_sum = 0.0
        throughput_dataloader_wait_ms_sum = 0.0
        # GPU-accurate timing (P4-1) - uses CUDA events instead of perf_counter
        step_timer = CUDATimer(env_states[0].env_device)
        reward_summary_accum = [
            {
                "bounded_attribution": 0.0,
                "compute_rent": 0.0,
                "alpha_shock": 0.0,
                "total_reward": 0.0,
                "count": 0,
            }
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
                configure_slot_telemetry(
                    env_state, inner_epoch=epoch, global_epoch=batch_epoch_id
                )
            # Track gradient stats per env for telemetry sync
            env_grad_stats: list[dict[str, dict[Any, Any]] | None] = [
                None
            ] * envs_this_batch

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
                    env_state.stream.wait_stream(
                        torch.cuda.default_stream(torch.device(env_state.env_device))
                    )

            # Iterate training batches using shared iterator (SharedBatchIterator or SharedGPUBatchIterator)
            # Both provide list of (inputs, targets) per environment, already on correct devices
            train_iter = iter(shared_train_iter)
            for batch_step in range(num_train_batches):
                try:
                    fetch_start = time.perf_counter()
                    env_batches = next(
                        train_iter
                    )  # List of (inputs, targets), already on devices
                    dataloader_wait_ms_epoch += (
                        time.perf_counter() - fetch_start
                    ) * 1000.0
                except StopIteration:
                    break

                # Launch all environments in their respective CUDA streams (async)
                # Data already moved to correct device by the shared iterator
                for i, env_state in enumerate(env_states):
                    if i >= len(env_batches):
                        continue
                    # CRITICAL: DataLoader .to(device, non_blocking=True) runs on the DEFAULT stream.
                    # We must sync env_state.stream with default stream before using the data,
                    # otherwise we may access partially-transferred data (race condition).
                    # BUG FIX: Use default_stream(), NOT current_stream() - the transfer happens
                    # on the default stream regardless of what stream is "current" in this context.
                    if env_state.stream:
                        # Wait for default stream where async .to() transfers are scheduled
                        loader_stream = torch.cuda.default_stream(torch.device(env_state.env_device))
                        env_state.stream.wait_stream(loader_stream)
                    inputs, targets = env_batches[i]

                    # BUG-031: Defensive validation for NLL loss assertion failures
                    # If targets contain values outside [0, n_classes), the NLL loss kernel
                    # will fail with "Assertion t>=0 && t < n_classes failed".
                    # Enable with ESPER_DEBUG_TARGETS=1 to catch the issue with diagnostics.
                    if os.environ.get("ESPER_DEBUG_TARGETS"):
                        if targets.is_cuda:
                            torch.cuda.synchronize(targets.device)
                        target_min = targets.min().item()
                        target_max = targets.max().item()
                        if target_min < 0 or target_max >= 10:  # CIFAR-10 has 10 classes
                            raise RuntimeError(
                                f"BUG-031: Invalid target values detected before loss computation. "
                                f"targets.min()={target_min}, targets.max()={target_max}, "
                                f"targets.device={targets.device}, env_idx={i}, batch_step={batch_step}, "
                                f"inputs.device={inputs.device}, inputs.shape={inputs.shape}, "
                                f"gpu_preload={gpu_preload}"
                            )

                    collect_gradients = use_telemetry and (
                        batch_step % gradient_telemetry_stride == 0
                    )
                    loss_tensor, correct_tensor, total, grad_stats = (
                        process_train_batch(
                            env_state,
                            inputs,
                            targets,
                            criterion,
                            use_telemetry=collect_gradients,
                            slots=slots,
                            use_amp=amp_enabled,
                        )
                    )
                    if grad_stats is not None:
                        env_grad_stats[i] = grad_stats  # Keep last batch's grad stats
                    stream_ctx = (
                        torch.cuda.stream(env_state.stream)
                        if env_state.stream
                        else nullcontext()
                    )
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
            train_losses = [
                env_state.train_loss_accum.item() for env_state in env_states
            ]  # type: ignore[union-attr]
            train_corrects = [
                env_state.train_correct_accum.item() for env_state in env_states
            ]  # type: ignore[union-attr]

            # ===== VALIDATION + COUNTERFACTUAL (FUSED): Single pass over test data =====
            # Instead of iterating test data multiple times or performing sequential
            # forward passes, we stack all configurations into a single fused pass.

            # CRITICAL: Reset scaffolding metrics at START of counterfactual phase.
            # These metrics accumulate per-epoch interaction/topology data.
            # Feature extraction expects per-epoch values, not cross-epoch accumulation.
            for env_state in env_states:
                for slot_id in slots:
                    if env_state.model.has_active_seed_in_slot(slot_id):
                        seed_state = env_state.model.seed_slots[slot_id].state
                        if seed_state and seed_state.metrics:
                            seed_state.metrics.interaction_sum = 0.0
                            seed_state.metrics.boost_received = 0.0
                            seed_state.metrics.upstream_alpha_sum = 0.0
                            seed_state.metrics.downstream_alpha_sum = 0.0

            # 1. Determine configurations per environment
            env_configs: list[list[dict[str, float]]] = []
            for i, env_state in enumerate(env_states):
                model = env_state.model
                active_slot_list = [
                    sid
                    for sid in slots
                    if model.has_active_seed_in_slot(sid)
                    and model.seed_slots[sid].state
                    and model.seed_slots[sid].state.alpha > 0
                ]

                # Config 0: Main (current alphas)
                configs = [{"_kind": "main"}]

                if active_slot_list:
                    # Configs 1..N: Solo ablation (one slot off)
                    for slot_id in active_slot_list:
                        configs.append(
                            {"_kind": "solo", "_slot": slot_id, slot_id: 0.0}
                        )

                    n_active = len(active_slot_list)
                    # Config N+1: All disabled (for 2-4 seeds)
                    if 2 <= n_active <= 4:
                        all_off = {sid: 0.0 for sid in active_slot_list}
                        all_off["_kind"] = "all_off"
                        configs.append(all_off)

                    # Pair configs (for 3-4 seeds)
                    if 3 <= n_active <= 4:
                        for idx_i in range(n_active):
                            for idx_j in range(idx_i + 1, n_active):
                                pair_config = {
                                    sid: 0.0
                                    for k, sid in enumerate(active_slot_list)
                                    if k != idx_i and k != idx_j
                                }
                                pair_config["_kind"] = "pair"
                                pair_config["_pair"] = (idx_i, idx_j)
                                configs.append(pair_config)

                # Inject exact Shapley configurations at episode end
                # This uses the Fused Validation kernel to compute analytics in parallel
                # with the main validation pass, mitigating the O(N!) overhead.
                if active_slot_list and env_state.counterfactual_helper and epoch == max_epochs:
                    required_configs = env_state.counterfactual_helper.get_required_configs(active_slot_list)
                    for config_tuple in required_configs:
                        # Convert tuple[bool] to alpha dict
                        shapley_cfg = {
                            sid: 1.0 if enabled else 0.0
                            for sid, enabled in zip(active_slot_list, config_tuple)
                        }
                        shapley_cfg["_kind"] = "shapley"
                        shapley_cfg["_tuple"] = config_tuple
                        configs.append(shapley_cfg)

                env_configs.append(configs)

            # baseline_accs[env_idx][slot_id] = accuracy with that slot's seed disabled
            baseline_accs: list[dict[str, float]] = [{} for _ in range(envs_this_batch)]
            all_disabled_accs: dict[int, float] = {}
            pair_accs: dict[int, dict[tuple[int, int], float]] = {}
            shapley_results: dict[int, dict[tuple[bool, ...], tuple[float, float]]] = {}
            val_totals = [0] * envs_this_batch

            # Accumulators for fused counts: env_cfg_correct_accums[env_idx] = [K] tensor
            env_cfg_correct_accums: list[torch.Tensor] = []
            for i, configs in enumerate(env_configs):
                env_cfg_correct_accums.append(
                    torch.zeros(len(configs), device=env_states[i].env_device)
                )

            # Iterate validation batches using shared iterator
            test_iter = iter(shared_test_iter)
            for batch_step in range(num_test_batches):
                try:
                    fetch_start = time.perf_counter()
                    env_batches = next(test_iter)
                    dataloader_wait_ms_epoch += (
                        time.perf_counter() - fetch_start
                    ) * 1000.0
                except StopIteration:
                    break

                for i, env_state in enumerate(env_states):
                    if i >= len(env_batches):
                        continue
                    if env_state.stream:
                        loader_stream = torch.cuda.current_stream(env_state.env_device)
                        env_state.stream.wait_stream(loader_stream)
                    inputs, targets = env_batches[i]
                    batch_size = inputs.size(0)
                    configs = env_configs[i]
                    num_configs = len(configs)

                    # Build alpha_overrides tensors for the fused pass: [K*B, 1, 1, 1]
                    #
                    # IMPORTANT: Only pass alpha_override when at least one config
                    # actually overrides that slot's alpha. Passing a no-op override
                    # (e.g., alpha==0.0) forces SeedSlot.forward down the blending path
                    # and bypasses the TRAINING-stage STE shortcut, changing semantics
                    # and creating unnecessary alpha_schedule requirements.
                    alpha_overrides = {}
                    for slot_id in env_state.model._active_slots:
                        needs_override = any(slot_id in cfg for cfg in configs)
                        if not needs_override:
                            continue
                        slot = env_state.model.seed_slots[slot_id]

                        # Enforce Phase 3 contract: alpha_schedule only valid for GATE.
                        if slot.alpha_schedule is not None and (
                            slot.state is None
                            or slot.state.alpha_algorithm != AlphaAlgorithm.GATE
                        ):
                            slot.alpha_schedule = None

                        # P4-FIX: Ensure alpha_schedule exists for GATE algorithm during fused pass.
                        # This can happen if a seed is in HOLD mode and its schedule was cleared.
                        if (
                            slot.state
                            and slot.state.alpha_algorithm == AlphaAlgorithm.GATE
                            and slot.alpha_schedule is None
                        ):
                            from esper.kasmina.blending import BlendCatalog

                            topology = task_spec.topology
                            # Use default tempo steps since it's already in HOLD
                            slot.alpha_schedule = BlendCatalog.create(
                                "gated",
                                channels=slot.channels,
                                topology=topology,
                                total_steps=5,
                            ).to(slot.device)

                        current_alpha = slot.alpha
                        override_vec = torch.full(
                            (num_configs * batch_size, 1, 1, 1),
                            current_alpha,
                            device=env_state.env_device,
                            dtype=inputs.dtype,
                        )

                        for cfg_idx, cfg in enumerate(configs):
                            if slot_id in cfg:
                                start, end = (
                                    cfg_idx * batch_size,
                                    (cfg_idx + 1) * batch_size,
                                )
                                override_vec[start:end].fill_(cfg[slot_id])
                        alpha_overrides[slot_id] = override_vec

                    # Run FUSED validation pass
                    _, correct_per_config, _ = process_fused_val_batch(
                        env_state,
                        inputs,
                        targets,
                        criterion,
                        alpha_overrides,
                        num_configs,
                    )

                    stream_ctx = (
                        torch.cuda.stream(env_state.stream)
                        if env_state.stream
                        else nullcontext()
                    )
                    with stream_ctx:
                        env_cfg_correct_accums[i].add_(correct_per_config)
                    val_totals[i] += batch_size

            # Single sync point at end
            for env_state in env_states:
                if env_state.stream:
                    env_state.stream.synchronize()

            # Process results for each config
            val_losses = [
                0.0
            ] * envs_this_batch  # Placeholder, losses not used for rewards
            val_corrects = [0] * envs_this_batch

            for i, env_state in enumerate(env_states):
                correct_counts = env_cfg_correct_accums[i].tolist()
                configs = env_configs[i]
                total = val_totals[i]

                if total == 0:
                    continue

                for cfg_idx, cfg in enumerate(configs):
                    acc = 100.0 * correct_counts[cfg_idx] / total
                    kind = cfg["_kind"]

                    if kind == "main":
                        val_corrects[i] = int(correct_counts[cfg_idx])
                        env_state.val_acc = acc
                    elif kind == "solo":
                        slot_id = cfg["_slot"]
                        baseline_accs[i][slot_id] = acc
                        # Sync to metrics
                        if env_state.model.has_active_seed_in_slot(slot_id):
                            seed_state = env_state.model.seed_slots[slot_id].state
                            if seed_state and seed_state.metrics:
                                new_contribution = env_state.val_acc - acc
                                # Compute contribution velocity (EMA of delta)
                                prev = seed_state.metrics._prev_contribution
                                if prev is not None:
                                    delta = new_contribution - prev
                                    # EMA with decay 0.7 (responsive to recent changes)
                                    seed_state.metrics.contribution_velocity = (
                                        0.7 * seed_state.metrics.contribution_velocity
                                        + 0.3 * delta
                                    )
                                seed_state.metrics._prev_contribution = new_contribution
                                seed_state.metrics.counterfactual_contribution = new_contribution
                    elif kind == "all_off":
                        all_disabled_accs[i] = acc
                    elif kind == "pair":
                        if i not in pair_accs:
                            pair_accs[i] = {}
                        pair_accs[i][cfg["_pair"]] = acc
                    elif kind == "shapley":
                        if i not in shapley_results:
                            shapley_results[i] = {}
                        # Validation loss approximated as 0.0 since we only track acc here
                        shapley_results[i][cfg["_tuple"]] = (0.0, acc)

                # Consolidate matrix reporting
                # CRITICAL: Sort active_slots for position-based topology computation.
                # Dict.keys() order is NOT guaranteed to match slot positions (r0c0, r0c1, r0c2...).
                # Lexicographic sort on slot IDs ensures correct upstream/downstream alpha sums.
                active_slots = sorted(baseline_accs[i].keys())
                if active_slots:
                    emitters[i].on_counterfactual_matrix(
                        active_slots=active_slots,
                        baseline_accs=baseline_accs[i],
                        val_acc=env_state.val_acc,
                        all_disabled_acc=all_disabled_accs.get(i),
                        pair_accs=pair_accs.get(i),
                    )

                # Compute interaction terms and populate scaffolding metrics
                if len(active_slots) >= 2 and i in pair_accs:
                    all_off_acc = all_disabled_accs.get(i, 0.0)
                    for (slot_a, slot_b), pair_acc in pair_accs[i].items():
                        solo_a = baseline_accs[i].get(slot_a, 0.0)
                        solo_b = baseline_accs[i].get(slot_b, 0.0)
                        # I_ij = f({i,j}) - f({i}) - f({j}) + f(empty)
                        interaction = pair_acc - solo_a - solo_b + all_off_acc

                        # Update metrics for both seeds
                        if env_state.model.has_active_seed_in_slot(slot_a):
                            seed_a = env_state.model.seed_slots[slot_a].state
                            if seed_a and seed_a.metrics:
                                seed_a.metrics.interaction_sum += interaction
                                seed_a.metrics.boost_received = max(
                                    seed_a.metrics.boost_received, interaction
                                )

                        if env_state.model.has_active_seed_in_slot(slot_b):
                            seed_b = env_state.model.seed_slots[slot_b].state
                            if seed_b and seed_b.metrics:
                                seed_b.metrics.interaction_sum += interaction
                                seed_b.metrics.boost_received = max(
                                    seed_b.metrics.boost_received, interaction
                                )

                # Compute topology features (upstream/downstream alpha sums)
                # active_slots is now sorted by position (lexicographic), ensuring correct topology
                for slot_idx, slot_id in enumerate(active_slots):
                    if not env_state.model.has_active_seed_in_slot(slot_id):
                        continue
                    seed_state = env_state.model.seed_slots[slot_id].state
                    if seed_state is None or seed_state.metrics is None:
                        continue

                    upstream_sum = 0.0
                    downstream_sum = 0.0
                    for other_idx, other_id in enumerate(active_slots):
                        if other_id == slot_id:
                            continue
                        if not env_state.model.has_active_seed_in_slot(other_id):
                            continue
                        other_state = env_state.model.seed_slots[other_id].state
                        if other_state is None:
                            continue

                        other_alpha = other_state.metrics.current_alpha if other_state.metrics else 0.0
                        if other_idx < slot_idx:
                            upstream_sum += other_alpha
                        else:
                            downstream_sum += other_alpha

                    seed_state.metrics.upstream_alpha_sum = upstream_sum
                    seed_state.metrics.downstream_alpha_sum = downstream_sum

                # Feed Shapley results to helper
                if i in shapley_results and env_state.counterfactual_helper:
                    try:
                        env_state.counterfactual_helper.compute_contributions_from_results(
                            slot_ids=active_slots,
                            results=shapley_results[i],
                            epoch=epoch,
                        )
                    except Exception as e:
                        _logger.warning(f"Shapley computation failed for env {i}: {e}")

            # ===== Compute epoch metrics and get BATCHED actions =====
            # NOTE: Telemetry sync (gradients/counterfactual) happens after record_accuracy()
            # so telemetry reflects the current epoch's metrics.

            # Collect signals, slot reports and action masks from all environments
            all_signals = []
            all_slot_reports = []
            all_total_params = []
            all_total_seeds = []
            all_masks = []

            # Post-action metadata for batched bootstrap computation
            all_post_action_signals = []
            all_post_action_slot_reports = []
            all_post_action_masks = []

            governor_panic_envs = []  # Track which envs need rollback
            ordered_slots = validate_slot_ids(list(slots))

            for env_idx, env_state in enumerate(env_states):
                model = env_state.model

                train_loss = env_state.train_loss
                train_acc = env_state.train_acc
                val_loss = env_state.val_loss
                val_acc = env_state.val_acc
                # Track maximum accuracy for sparse reward
                env_state.host_max_acc = max(env_state.host_max_acc, env_state.val_acc)

                # Governor watchdog: snapshot when loss is stable (every 5 epochs)
                if epoch % 5 == 0:
                    env_state.governor.snapshot()

                # Governor watchdog: check vital signs after validation
                is_panic = env_state.governor.check_vital_signs(val_loss)
                if is_panic:
                    governor_panic_envs.append(env_idx)

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

                slot_reports = model.get_slot_reports()

                # Consolidate environment-level telemetry emission
                emitters[env_idx].on_epoch_completed(epoch, env_state, slot_reports)

                # Update signal tracker
                # Phase 4: embargo/cooldown stages keep state while seed is removed.
                # Availability for germination is therefore "no state", not merely "no active seed".
                available_slots = sum(
                    1 for slot_id in slots if model.seed_slots[slot_id].state is None
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
                all_slot_reports.append(slot_reports)
                all_total_params.append(model.total_params if model else 0)
                all_total_seeds.append(model.total_seeds() if model else 0)

                # Compute action mask based on current state (physical constraints only)
                # Build slot states for ALL enabled slots (multi-slot masking)
                slot_states = build_slot_states(slot_reports, ordered_slots)
                mask = compute_action_masks(
                    slot_states=slot_states,
                    enabled_slots=ordered_slots,
                    total_seeds=model.total_seeds() if model else 0,
                    max_seeds=effective_max_seeds,
                    slot_config=slot_config,
                    device=torch.device(device),
                    topology=task_spec.topology,
                )
                all_masks.append(mask)

            # OPTIMIZATION: Batched tensor-driven feature extraction
            states_batch = batch_obs_to_features(
                batch_signals=all_signals,
                batch_slot_reports=all_slot_reports,
                use_telemetry=use_telemetry,
                max_epochs=max_epochs,
                total_params=all_total_params,
                total_seeds=all_total_seeds,
                max_seeds=effective_max_seeds,
                slot_config=slot_config,
                device=torch.device(device),
            )

            # Stack dict masks into batched dict: {key: [n_envs, head_dim]}
            # Use static HEAD_NAMES for torch.compile compatibility
            masks_batch = {
                key: torch.stack([m[key] for m in all_masks]).to(device)
                for key in HEAD_NAMES
            }

            # Accumulate raw states for deferred normalizer update
            raw_states_for_normalizer_update.append(states_batch.detach())

            # Normalize using FROZEN statistics during rollout collection.
            states_batch_normalized = obs_normalizer.normalize(states_batch)

            # Get BATCHED actions from policy network with action masking (single forward pass!)
            pre_step_hiddens: list[tuple[torch.Tensor, torch.Tensor]] = []

            if batched_lstm_hidden is not None:
                h_batch, c_batch = batched_lstm_hidden
                for env_idx in range(len(env_states)):
                    env_h = h_batch[:, env_idx : env_idx + 1, :].clone()
                    env_c = c_batch[:, env_idx : env_idx + 1, :].clone()
                    pre_step_hiddens.append((env_h, env_c))
            else:
                batched_lstm_hidden = agent.policy.initial_hidden(
                    len(env_states)
                )
                init_h, init_c = batched_lstm_hidden
                for env_idx in range(len(env_states)):
                    env_h = init_h[:, env_idx : env_idx + 1, :].clone()
                    env_c = init_c[:, env_idx : env_idx + 1, :].clone()
                    pre_step_hiddens.append((env_h, env_c))

            # get_action returns ActionResult dataclass
            action_result = agent.policy.get_action(
                states_batch_normalized,
                masks=masks_batch,
                hidden=batched_lstm_hidden,
                deterministic=False,
            )
            # TODO: [FUTURE FUNCTIONALITY] - op_logits telemetry not yet supported in PolicyBundle interface
            # Would require extending ActionResult to include optional logits field
            op_probs_cpu: list[list[float]] | None = None
            actions_dict = action_result.action
            head_log_probs = action_result.log_prob
            values_tensor = action_result.value

            # OPTIMIZATION: Update batched hidden state directly (eliminates per-env slice/cat)
            batched_lstm_hidden = action_result.hidden

            # Convert to list of dicts for per-env processing
            actions_cpu = {key: actions_dict[key].cpu().numpy() for key in HEAD_NAMES}
            actions = [
                {key: int(actions_cpu[key][i]) for key in HEAD_NAMES}
                for i in range(len(env_states))
            ]
            values = values_tensor.tolist()

            # Batch compute mask stats for telemetry
            if ops_telemetry_enabled:
                masked_batch = {
                    key: ~masks_batch[key].all(dim=-1)  # [num_envs] bool tensor
                    for key in HEAD_NAMES
                }
                masked_cpu = {
                    key: masked_batch[key].cpu().numpy() for key in HEAD_NAMES
                }
            else:
                masked_cpu = None

            # PHASE 1: Execute actions and collect data for bootstrap computation
            bootstrap_data = []
            transitions_data = []  # Store transition data for buffer storage

            for env_idx, env_state in enumerate(env_states):
                model = env_state.model
                signals = all_signals[env_idx]
                value = values[env_idx]

                # Parse sampled action indices and derive values (Deduplication)
                action_dict = actions[env_idx]
                (
                    target_slot,
                    slot_is_enabled,
                    slot_state,
                    seed_state,
                    action_valid_for_reward,
                    action_for_reward,
                    blend_algorithm_id,
                    alpha_algorithm,
                    alpha_target,
                ) = _parse_sampled_action(
                    env_idx,
                    action_dict["op"],
                    action_dict["slot"],
                    action_dict["style"],
                    action_dict["alpha_target"],
                    slots,
                    slot_config,
                    model,
                )

                # Use op name for action counting
                env_state.action_counts[action_for_reward.name] = (
                    env_state.action_counts.get(action_for_reward.name, 0) + 1
                )

                action_success = False

                # Governor rollback
                if env_idx in governor_panic_envs:
                    env_state.governor.execute_rollback(
                        env_id=env_idx, optimizer=env_state.host_optimizer
                    )
                    env_rollback_occurred[env_idx] = True

                # Compute reward
                scoreboard = analytics._get_scoreboard(env_idx)
                host_params = scoreboard.host_params

                effective_seed_params, alpha_delta_sq_sum = (
                    compute_rent_and_shock_inputs(
                        model=model,
                        slot_ids=slots,
                        host_params=host_params,
                        base_slot_rent_ratio=env_reward_configs[
                            env_idx
                        ].base_slot_rent_ratio,
                        prev_slot_alphas=env_state.prev_slot_alphas,
                        prev_slot_params=env_state.prev_slot_params,
                    )
                )

                seed_contribution = None
                if target_slot in baseline_accs[env_idx]:
                    seed_contribution = (
                        env_state.val_acc - baseline_accs[env_idx][target_slot]
                    )

                emit_reward_components_event = (
                    telemetry_config is not None
                    and telemetry_config.should_collect("debug")
                )
                collect_reward_summary = (
                    telemetry_config is not None
                    and telemetry_config.should_collect("ops_normal")
                )

                seed_params_for_slot = (
                    model.seed_slots[target_slot].active_seed_params
                    if slot_is_enabled
                    else 0
                )
                seed_info = SeedInfo.from_seed_state(seed_state, seed_params_for_slot)

                if reward_family_enum == RewardFamily.CONTRIBUTION:
                    reward_args = {
                        "action": action_for_reward,
                        "seed_contribution": seed_contribution,
                        "val_acc": env_state.val_acc,
                        "host_max_acc": env_state.host_max_acc,
                        "seed_info": seed_info,
                        "epoch": epoch,
                        "max_epochs": max_epochs,
                        "total_params": model.total_params,
                        "host_params": host_params,
                        "acc_at_germination": env_state.acc_at_germination.get(
                            target_slot
                        ),
                        "acc_delta": signals.metrics.accuracy_delta,
                        "effective_seed_params": effective_seed_params,
                        "alpha_delta_sq_sum": alpha_delta_sq_sum,
                        "num_fossilized_seeds": env_state.seeds_fossilized,
                        "num_contributing_fossilized": env_state.contributing_fossilized,
                        "config": env_reward_configs[env_idx],
                    }
                    if emit_reward_components_event or collect_reward_summary:
                        reward, reward_components = compute_reward(
                            **reward_args, return_components=True
                        )
                        if target_slot in baseline_accs[env_idx]:
                            reward_components.host_baseline_acc = baseline_accs[
                                env_idx
                            ][target_slot]
                    else:
                        reward = compute_reward(**reward_args)
                else:
                    reward = compute_loss_reward(
                        action=action_for_reward,
                        loss_delta=signals.metrics.loss_delta,
                        val_loss=env_state.val_loss,
                        seed_info=seed_info,
                        epoch=epoch,
                        max_epochs=max_epochs,
                        total_params=model.total_params,
                        host_params=host_params,
                        config=loss_reward_config,
                    )

                reward += env_state.pending_auto_prune_penalty
                env_state.pending_auto_prune_penalty = 0.0

                # Normalize reward for PPO stability (P1-6 fix)
                normalized_reward = reward_normalizer.update_and_normalize(reward)
                env_state.episode_rewards.append(reward)

                if collect_reward_summary and "reward_components" in locals():
                    summary = reward_summary_accum[env_idx]
                    summary["total_reward"] += reward
                    if reward_components.bounded_attribution is not None:
                        summary["bounded_attribution"] += (
                            reward_components.bounded_attribution
                        )
                    summary["compute_rent"] += reward_components.compute_rent
                    summary["alpha_shock"] += reward_components.alpha_shock
                    summary["count"] += 1

                # Execute action
                op_idx = action_dict["op"]
                if slot_is_enabled:
                    if (
                        op_idx == OP_GERMINATE
                        and model.seed_slots[target_slot].state is None
                    ):
                        env_state.acc_at_germination[target_slot] = env_state.val_acc
                        model.germinate_seed(
                            BLUEPRINT_IDS[action_dict["blueprint"]],
                            f"env{env_idx}_seed_{env_state.seeds_created}",
                            slot=target_slot,
                            blend_algorithm_id=blend_algorithm_id,
                            blend_tempo_epochs=TEMPO_TO_EPOCHS[
                                TempoAction(action_dict["tempo"])
                            ],
                            alpha_algorithm=alpha_algorithm,
                            alpha_target=alpha_target,
                        )
                        env_state.seeds_created += 1
                        env_state.seed_optimizers.pop(target_slot, None)
                        action_success = True
                    elif op_idx == OP_FOSSILIZE:
                        action_success = _advance_active_seed(model, target_slot)
                        if action_success:
                            env_state.seeds_fossilized += 1
                            if (
                                seed_info.total_improvement
                                >= DEFAULT_MIN_FOSSILIZE_CONTRIBUTION
                            ):
                                env_state.contributing_fossilized += 1
                            env_state.acc_at_germination.pop(target_slot, None)
                    elif (
                        op_idx == OP_PRUNE
                        and model.has_active_seed_in_slot(target_slot)
                        # BUG-020 fix: enforce MIN_PRUNE_AGE at execution gate
                        and seed_info.seed_age_epochs >= MIN_PRUNE_AGE
                    ):
                        speed_steps = ALPHA_SPEED_TO_STEPS[
                            AlphaSpeedAction(action_dict["alpha_speed"])
                        ]
                        curve = AlphaCurveAction(action_dict["alpha_curve"]).to_curve()
                        if speed_steps <= 0:
                            action_success = model.seed_slots[target_slot].prune(
                                reason="policy_prune", initiator="policy"
                            )
                        else:
                            action_success = model.seed_slots[
                                target_slot
                            ].schedule_prune(
                                steps=speed_steps, curve=curve, initiator="policy"
                            )
                        if action_success:
                            env_state.seed_optimizers.pop(target_slot, None)
                            env_state.acc_at_germination.pop(target_slot, None)
                    elif (
                        op_idx == OP_SET_ALPHA_TARGET
                        and model.has_active_seed_in_slot(target_slot)
                    ):
                        action_success = model.seed_slots[target_slot].set_alpha_target(
                            alpha_target=ALPHA_TARGET_VALUES[
                                action_dict["alpha_target"]
                            ],
                            steps=ALPHA_SPEED_TO_STEPS[
                                AlphaSpeedAction(action_dict["alpha_speed"])
                            ],
                            curve=AlphaCurveAction(
                                action_dict["alpha_curve"]
                            ).to_curve(),
                            alpha_algorithm=alpha_algorithm,
                            initiator="policy",
                        )
                        if action_success:
                            env_state.seed_optimizers.pop(target_slot, None)
                    elif op_idx == OP_ADVANCE and model.has_active_seed_in_slot(
                        target_slot
                    ):
                        gate_result = model.seed_slots[target_slot].advance_stage()
                        action_success = gate_result.passed
                        if action_success:
                            env_state.seed_optimizers.pop(target_slot, None)
                elif op_idx == OP_WAIT:
                    action_success = True

                if action_success:
                    env_state.successful_action_counts[action_for_reward.name] = (
                        env_state.successful_action_counts.get(
                            action_for_reward.name, 0
                        )
                        + 1
                    )

                # Consolidate telemetry via emitter
                if ops_telemetry_enabled and masked_cpu is not None:
                    masked_flags = {k: bool(masked_cpu[k][env_idx]) for k in HEAD_NAMES}
                    for k, m in masked_flags.items():
                        mask_total[k] += 1
                        if m:
                            mask_hits[k] += 1

                    post_slot_state = model.seed_slots[target_slot].state
                    active_algo = (
                        post_slot_state.alpha_algorithm.name
                        if post_slot_state
                        else None
                    )
                    slot_reports_for_decision = all_slot_reports[env_idx]
                    decision_slot_states: dict[str, str] = {}
                    for slot_id in ordered_slots:
                        report = slot_reports_for_decision.get(slot_id)
                        if report is None:
                            decision_slot_states[slot_id] = "Empty"
                            continue
                        stage_label = report.stage.name.title()
                        decision_slot_states[slot_id] = (
                            f"{stage_label} {report.metrics.total_improvement:.0f}%"
                        )

                    action_confidence = None
                    alternatives: list[tuple[str, float]] | None = None
                    decision_entropy = None
                    if op_probs_cpu is not None and env_idx < len(op_probs_cpu):
                        probs = op_probs_cpu[env_idx]
                        chosen_op = int(action_dict["op"])
                        if 0 <= chosen_op < len(probs):
                            action_confidence = float(probs[chosen_op])
                        ranked = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
                        alternatives = [
                            (OP_NAMES[op_idx], float(prob))
                            for op_idx, prob in ranked
                            if op_idx != chosen_op
                        ][:2]
                        # Compute decision entropy: -sum(p * log(p)) for op head
                        entropy_sum = 0.0
                        for p in probs:
                            if p > 1e-8:  # Avoid log(0)
                                entropy_sum -= p * math.log(p)
                        decision_entropy = entropy_sum
                    emitters[env_idx].on_last_action(
                        epoch,
                        action_dict,
                        target_slot,
                        masked_flags,
                        action_success,
                        active_algo,
                        total_reward=reward,
                        value_estimate=value,
                        host_accuracy=env_state.val_acc,
                        slot_states=decision_slot_states,
                        action_confidence=action_confidence,
                        alternatives=alternatives,
                        decision_entropy=decision_entropy,
                    )

                # Store transition
                done = epoch == max_epochs
                truncated = done

                if truncated:
                    pass

                transitions_data.append(
                    {
                        "env_id": env_idx,
                        "state": states_batch[env_idx].detach(),
                        "action_dict": action_dict,
                        "log_probs": {
                            k: v[env_idx].detach() for k, v in head_log_probs.items()
                        },
                        "env_masks": {
                            k: v[env_idx].detach() for k, v in masks_batch.items()
                        },
                        "value": value,
                        "reward": normalized_reward,
                        "done": done,
                        "truncated": truncated,
                        "hidden_h": pre_step_hiddens[env_idx][0].detach(),
                        "hidden_c": pre_step_hiddens[env_idx][1].detach(),
                    }
                )

                if done:
                    agent.buffer.end_episode(env_id=env_idx)
                    if batched_lstm_hidden is not None:
                        # P4-FIX: Inplace update to inference tensor not allowed.
                        # Reset this environment's hidden state in the batch for the next episode.
                        init_h, init_c = agent.policy.initial_hidden(1)

                        # Create new tensors to avoid inplace modification of inference tensors
                        new_h = batched_lstm_hidden[0].clone()
                        new_c = batched_lstm_hidden[1].clone()
                        new_h[:, env_idx : env_idx + 1, :] = init_h
                        new_c[:, env_idx : env_idx + 1, :] = init_c
                        batched_lstm_hidden = (new_h, new_c)

                # Mechanical lifecycle advance
                for slot_id in slots:
                    if model.seed_slots[slot_id].step_epoch():
                        env_state.pending_auto_prune_penalty += (
                            reward_config.auto_prune_penalty
                        )
                    if not model.has_active_seed_in_slot(slot_id):
                        env_state.seed_optimizers.pop(slot_id, None)
                        env_state.acc_at_germination.pop(slot_id, None)
                        env_state.gradient_ratio_ema.pop(slot_id, None)

                # Fix BUG-022: Collect bootstrap state AFTER mechanical advance
                if truncated:
                    all_post_action_signals.append(
                        env_state.signal_tracker.peek(
                            epoch=epoch,
                            global_step=epoch * num_train_batches,
                            train_loss=env_state.train_loss,
                            train_accuracy=env_state.train_acc,
                            val_loss=env_state.val_loss,
                            val_accuracy=env_state.val_acc,
                            active_seeds=[
                                s.state
                                for s in model.seed_slots.values()
                                if s.is_active and s.state
                            ],
                            available_slots=sum(
                                1 for s in model.seed_slots.values() if s.state is None
                            ),
                        )
                    )
                    all_post_action_slot_reports.append(model.get_slot_reports())
                    all_post_action_masks.append(
                        compute_action_masks(
                            slot_states=build_slot_states(
                                all_post_action_slot_reports[-1], ordered_slots
                            ),
                            enabled_slots=ordered_slots,
                            total_seeds=model.total_seeds(),
                            max_seeds=effective_max_seeds,
                            slot_config=slot_config,
                            device=torch.device(device),
                            topology=task_spec.topology,
                        )
                    )

                if epoch == max_epochs:
                    env_final_accs[env_idx] = env_state.val_acc
                    env_total_rewards[env_idx] = sum(env_state.episode_rewards)

                    # Track episode completion for A/B testing
                    episode_history.append({
                        "env_idx": env_idx,
                        "episode_reward": env_total_rewards[env_idx],
                        "final_accuracy": env_final_accs[env_idx],
                    })

                    # Shapley contributions at episode end
                    if (
                        env_state.counterfactual_helper is not None
                        and baseline_accs[env_idx]
                    ):
                        active_slot_ids = [
                            sid
                            for sid in slots
                            if model.has_active_seed_in_slot(sid)
                            and model.seed_slots[sid].alpha > 0
                        ]
                        if active_slot_ids:
                            cached_baselines = baseline_accs[env_idx]

                            def eval_fn(
                                alpha_settings: dict[str, float],
                            ) -> tuple[float, float]:
                                if all(a >= 0.99 for a in alpha_settings.values()):
                                    return env_state.val_loss, env_state.val_acc
                                disabled = [
                                    s for s, a in alpha_settings.items() if a < 0.01
                                ]
                                if (
                                    len(disabled) == 1
                                    and disabled[0] in cached_baselines
                                ):
                                    return env_state.val_loss * 1.1, cached_baselines[
                                        disabled[0]
                                    ]
                                if disabled:
                                    return env_state.val_loss * 1.2, sum(
                                        cached_baselines.get(s, env_state.val_acc)
                                        for s in disabled
                                    ) / len(disabled)
                                return env_state.val_loss, env_state.val_acc

                            try:
                                env_state.counterfactual_helper.compute_contributions(
                                    slot_ids=active_slot_ids,
                                    evaluate_fn=eval_fn,
                                    epoch=epoch,
                                )
                            except Exception as e:
                                _logger.warning(
                                    f"Shapley failed for env {env_idx}: {e}"
                                )

            # PHASE 2: Compute all bootstrap values in single batched forward pass
            bootstrap_values = []
            if all_post_action_signals:
                post_action_features_batch = batch_signals_to_features(
                    batch_signals=all_post_action_signals,
                    batch_slot_reports=all_post_action_slot_reports,
                    use_telemetry=use_telemetry,
                    max_epochs=max_epochs,
                    effective_max_seeds=effective_max_seeds,
                    slot_config=slot_config,
                    env_states=env_states,
                    device=torch.device(device),
                )
                post_action_features_normalized = obs_normalizer.normalize(
                    post_action_features_batch
                )
                post_masks_batch = {
                    k: torch.stack([m[k] for m in all_post_action_masks]).to(device)
                    for k in HEAD_NAMES
                }

                with torch.inference_mode():
                    bootstrap_result = agent.policy.get_action(
                        post_action_features_normalized,
                        masks=post_masks_batch,
                        hidden=batched_lstm_hidden,
                        deterministic=True,
                    )
                bootstrap_values = bootstrap_result.value.tolist()

            # PHASE 3: Store transitions
            bootstrap_idx = 0
            for transition in transitions_data:
                bootstrap_val = (
                    bootstrap_values[bootstrap_idx] if transition["truncated"] else 0.0
                )
                if transition["truncated"]:
                    bootstrap_idx += 1

                agent.buffer.add(
                    env_id=transition["env_id"],
                    state=transition["state"],
                    slot_action=transition["action_dict"]["slot"],
                    blueprint_action=transition["action_dict"]["blueprint"],
                    style_action=transition["action_dict"]["style"],
                    tempo_action=transition["action_dict"]["tempo"],
                    alpha_target_action=transition["action_dict"]["alpha_target"],
                    alpha_speed_action=transition["action_dict"]["alpha_speed"],
                    alpha_curve_action=transition["action_dict"]["alpha_curve"],
                    op_action=transition["action_dict"]["op"],
                    slot_log_prob=transition["log_probs"]["slot"],
                    blueprint_log_prob=transition["log_probs"]["blueprint"],
                    style_log_prob=transition["log_probs"]["style"],
                    tempo_log_prob=transition["log_probs"]["tempo"],
                    alpha_target_log_prob=transition["log_probs"]["alpha_target"],
                    alpha_speed_log_prob=transition["log_probs"]["alpha_speed"],
                    alpha_curve_log_prob=transition["log_probs"]["alpha_curve"],
                    op_log_prob=transition["log_probs"]["op"],
                    value=transition["value"],
                    reward=transition["reward"],
                    done=transition["done"],
                    slot_mask=transition["env_masks"]["slot"],
                    blueprint_mask=transition["env_masks"]["blueprint"],
                    style_mask=transition["env_masks"]["style"],
                    tempo_mask=transition["env_masks"]["tempo"],
                    alpha_target_mask=transition["env_masks"]["alpha_target"],
                    alpha_speed_mask=transition["env_masks"]["alpha_speed"],
                    alpha_curve_mask=transition["env_masks"]["alpha_curve"],
                    op_mask=transition["env_masks"]["op"],
                    hidden_h=transition["hidden_h"],
                    hidden_c=transition["hidden_c"],
                    truncated=transition["truncated"],
                    bootstrap_value=bootstrap_val,
                )

        throughput_step_time_ms_sum += step_timer.stop()
        throughput_dataloader_wait_ms_sum += dataloader_wait_ms_epoch

        # PPO Update
        metrics: dict[str, Any] = {}
        ppo_grad_norm, ppo_update_time_ms = None, None
        rollback_env_indices = [
            i for i, occurred in enumerate(env_rollback_occurred) if occurred
        ]
        if rollback_env_indices:
            for env_idx in rollback_env_indices:
                agent.buffer.clear_env(env_idx)

        update_skipped = len(agent.buffer) == 0
        if not update_skipped:
            update_start = time.perf_counter()
            metrics = _run_ppo_updates(
                agent=agent,
                ppo_updates_per_batch=ppo_updates_per_batch,
                raw_states_for_normalizer_update=raw_states_for_normalizer_update,
                obs_normalizer=obs_normalizer,
                use_amp=amp,
                amp_dtype=resolved_amp_dtype,
            )
            ppo_update_time_ms = (time.perf_counter() - update_start) * 1000.0
            ppo_grad_norm = compute_grad_norm_surrogate(agent.policy._network)

            metric_values = [v for v in metrics.values() if isinstance(v, (int, float))]
            anomaly_report = anomaly_detector.check_all(
                ratio_max=metrics.get("ratio_max", 1.0),
                ratio_min=metrics.get("ratio_min", 1.0),
                explained_variance=metrics.get("explained_variance", 0.0),
                has_nan=any(math.isnan(v) for v in metric_values),
                has_inf=any(math.isinf(v) for v in metric_values),
                current_episode=batch_epoch_id,
                total_episodes=total_episodes,
            )
            _handle_telemetry_escalation(anomaly_report, telemetry_config)
            _emit_anomaly_diagnostics(
                hub,
                anomaly_report,
                agent,
                batch_epoch_id,
                batch_idx,
                max_epochs,
                total_episodes,
                False,
                group_id=group_id,
            )

        # Track results and aggregate batch-level metrics
        avg_acc = sum(env_final_accs) / len(env_final_accs)
        avg_reward = sum(env_total_rewards) / len(env_total_rewards)

        recent_accuracies.append(avg_acc)
        recent_rewards.append(avg_reward)
        if len(recent_accuracies) > 10:
            recent_accuracies.pop(0)
            recent_rewards.pop(0)

        rolling_avg_acc = sum(recent_accuracies) / len(recent_accuracies)

        if hub:
            if not update_skipped:
                batch_emitter.on_ppo_update(
                    metrics=metrics,
                    episodes_completed=batch_epoch_id,
                    batch_idx=batch_idx,
                    epoch=epoch,
                    agent=agent,
                    ppo_grad_norm=ppo_grad_norm,
                    ppo_update_time_ms=ppo_update_time_ms,
                    avg_acc=avg_acc,
                    avg_reward=avg_reward,
                    rolling_avg_acc=rolling_avg_acc,
                )

            # Aggregate per-environment metrics for the BATCH_EPOCH_COMPLETED event
            batch_train_losses = [es.train_loss for es in env_states]
            batch_train_corrects = [
                int(es.train_acc * 100) for es in env_states
            ]  # Placeholder, needs real count
            # Use sum of transition counts per env
            batch_train_totals = [
                agent.buffer.step_counts[i] for i in range(envs_this_batch)
            ]

            batch_val_losses = [0.0] * envs_this_batch
            batch_val_corrects = val_corrects
            batch_val_totals = val_totals

            batch_emitter.on_batch_completed(
                batch_idx=batch_idx,
                episodes_completed=episodes_completed + envs_this_batch,
                rolling_avg_acc=rolling_avg_acc,
                avg_acc=avg_acc,
                metrics=metrics,
                env_states=env_states,
                update_skipped=update_skipped,
                plateau_threshold=plateau_threshold,
                improvement_threshold=improvement_threshold,
                prev_rolling_avg_acc=prev_rolling_avg_acc,
                total_episodes=total_episodes,
                start_episode=start_episode,
                n_episodes=n_episodes,
                env_final_accs=env_final_accs,
                avg_reward=avg_reward,
                train_losses=batch_train_losses,
                train_corrects=batch_train_corrects,
                train_totals=batch_train_totals,
                val_losses=batch_val_losses,
                val_corrects=batch_val_corrects,
                val_totals=batch_val_totals,
                num_train_batches=num_train_batches,
                num_test_batches=num_test_batches,
                analytics=analytics,
                epoch=epoch,
            )
            prev_rolling_avg_acc = rolling_avg_acc

        history.append(
            {
                "batch": batch_idx + 1,
                "episodes": episodes_completed + envs_this_batch,
                "avg_accuracy": avg_acc,
                "rolling_avg_accuracy": rolling_avg_acc,
                **metrics,
            }
        )

        if rolling_avg_acc > best_avg_acc:
            best_avg_acc = rolling_avg_acc
            best_state = {
                k: v.cpu().clone() for k, v in agent.policy.state_dict().items()
            }

        episodes_completed += envs_this_batch
        batch_idx += 1

    if best_state:
        agent.policy.load_state_dict(best_state)

    if save_path:
        agent.save(save_path)

    # A/B Test Summary
    if ab_reward_modes is not None:
        print("\n" + "=" * 60)
        print("A/B TEST RESULTS")
        print("=" * 60)

        # Group episodes by reward mode
        from collections import defaultdict
        ab_groups = defaultdict(list)
        for ep_data in episode_history:
            env_idx = ep_data["env_idx"]
            mode = env_reward_configs[env_idx].reward_mode.value
            ab_groups[mode].append(ep_data)

        for mode, episodes in sorted(ab_groups.items()):
            rewards = [ep.get("episode_reward", 0) for ep in episodes]
            accuracies = [ep.get("final_accuracy", 0) for ep in episodes]
            avg_reward = sum(rewards) / len(rewards) if rewards else 0
            avg_acc = sum(accuracies) / len(accuracies) if accuracies else 0
            print(f"\n{mode.upper()} ({len(episodes)} episodes):")
            print(f"  Avg Episode Reward: {avg_reward:.2f}")
            print(f"  Avg Final Accuracy: {avg_acc:.2f}%")
            print(f"  Reward Range: [{min(rewards):.2f}, {max(rewards):.2f}]")
        print("=" * 60)

    return agent, history


__all__ = ["ParallelEnvState", "train_ppo_vectorized"]
