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
    from esper.simic.training.vectorized import train_ppo_vectorized

    agent, history = train_ppo_vectorized(
        n_episodes=100,
        n_envs=4,
        devices=["cuda:0", "cuda:1"],
    )
"""

from __future__ import annotations

import dataclasses
import logging
import math
import os
import threading
import time

import numpy as np
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn as nn
import torch.amp as torch_amp

if TYPE_CHECKING:
    from esper.leyline.reports import SeedStateReport
    from esper.simic.rewards.reward_telemetry import RewardComponentsTelemetry

# NOTE: get_task_spec imported lazily inside train_ppo_vectorized to avoid circular import:
#   runtime -> simic.rewards -> simic -> simic.training -> vectorized -> runtime
from esper.utils.data import SharedBatchIterator, augment_cifar10_batch
from esper.leyline import (
    ALPHA_SPEED_TO_STEPS,
    AlphaAlgorithm,
    GateLevel,
    SeedSlotProtocol,
    SeedStateProtocol,
    AlphaCurveAction,
    AlphaMode,
    AlphaSpeedAction,
    BLUEPRINT_IDS,
    DEFAULT_CLIP_RATIO,
    DEFAULT_ENTROPY_COEF,
    DEFAULT_ENTROPY_COEF_MIN,
    DEFAULT_EPISODE_LENGTH,
    DEFAULT_GAE_LAMBDA,
    DEFAULT_GAMMA,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LSTM_HIDDEN_DIM,
    DEFAULT_N_ENVS,
    CheckpointLoadedPayload,
    EpisodeOutcomePayload,
    HEAD_NAMES,
    HeadTelemetry,
    LifecycleOp,
    OP_ADVANCE,
    OP_FOSSILIZE,
    OP_GERMINATE,
    OP_NAMES,
    OP_PRUNE,
    OP_SET_ALPHA_TARGET,
    OP_WAIT,
    SeedStage,
    SlotConfig,
    TEMPO_TO_EPOCHS,
    TelemetryEvent,
    TelemetryEventType,
    TempoAction,
    TrainingStartedPayload,
)
from esper.tamiyo.policy.action_masks import build_slot_states, compute_action_masks
from esper.leyline.slot_id import validate_slot_ids
from esper.simic.telemetry import (
    AnomalyDetector,
    AnomalyReport,
    check_numerical_stability,
    collect_per_layer_gradients,
    materialize_dual_grad_stats,
    materialize_grad_stats,
    TelemetryConfig,
    GradientEMATracker,  # P4-9
    training_profiler,
    compute_lstm_health,  # B7-DRL-04
    compute_observation_stats,  # TELE-OBS: Observation space health
)
from esper.simic.control import RunningMeanStd, RewardNormalizer
from esper.tamiyo.policy.features import (
    get_feature_size,
    batch_obs_to_features,
)
from esper.simic.agent import PPOAgent
from esper.simic.agent.types import PPOUpdateMetrics
from esper.tamiyo.policy import create_policy
from esper.simic.rewards import (
    compute_reward,
    compute_loss_reward,
    compute_scaffold_hindsight_credit,
    RewardMode,
    RewardFamily,
    ContributionRewardConfig,
    SeedInfo,
    STAGE_POTENTIALS,
)
from esper.leyline import (
    DEFAULT_BATCH_SIZE_TRAINING,
    DEFAULT_MIN_FOSSILIZE_CONTRIBUTION,
    EpisodeOutcome,  # Cross-subsystem Pareto analysis
    HINDSIGHT_CREDIT_WEIGHT,
    MAX_HINDSIGHT_CREDIT,
    MIN_PRUNE_AGE,
)
from esper.nissa import get_hub, BlueprintAnalytics, DirectoryOutput
from esper.simic.telemetry.emitters import (
    check_performance_degradation,
    compute_grad_norm_surrogate,
    VectorizedEmitter,
)
from .action_execution import parse_sampled_action
from .batch_ops import process_fused_val_batch, process_train_batch
from .counterfactual_eval import build_env_configs, reset_scaffolding_metrics
from .env_factory import EnvFactory
from .feature_ops import batch_signals_to_features
from .helpers import compute_rent_and_shock_inputs
from .parallel_env_state import ParallelEnvState
from .vectorized_helpers import (
    _advance_active_seed,
    _aggregate_ppo_metrics,
    _calculate_entropy_anneal_steps,
    _handle_telemetry_escalation,
    _resolve_target_slot,
    _run_ppo_updates,
)
from .vectorized_trainer import VectorizedPPOTrainer
from . import vectorized_helpers

# PERF: Static head indices (optimized for high env counts).
_HEAD_NAME_TO_IDX: dict[str, int] = {name: idx for idx, name in enumerate(HEAD_NAMES)}
_HEAD_SLOT_IDX = _HEAD_NAME_TO_IDX["slot"]
_HEAD_BLUEPRINT_IDX = _HEAD_NAME_TO_IDX["blueprint"]
_HEAD_STYLE_IDX = _HEAD_NAME_TO_IDX["style"]
_HEAD_TEMPO_IDX = _HEAD_NAME_TO_IDX["tempo"]
_HEAD_ALPHA_TARGET_IDX = _HEAD_NAME_TO_IDX["alpha_target"]
_HEAD_ALPHA_SPEED_IDX = _HEAD_NAME_TO_IDX["alpha_speed"]
_HEAD_ALPHA_CURVE_IDX = _HEAD_NAME_TO_IDX["alpha_curve"]
_HEAD_OP_IDX = _HEAD_NAME_TO_IDX["op"]

_logger = logging.getLogger(__name__)


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
            self.start_event = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
            self.end_event = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        else:
            self.start_time = 0.0

    def start(self) -> None:
        """Record start time."""
        if self.use_cuda:
            self.start_event.record()  # type: ignore[no-untyped-call]
        else:
            self.start_time = time.perf_counter()

    def stop(self) -> float:
        """Record end time and return elapsed milliseconds.

        For CUDA: synchronizes to ensure GPU work is complete before measuring.
        """
        if self.use_cuda:
            self.end_event.record()  # type: ignore[no-untyped-call]
            self.end_event.synchronize()
            return self.start_event.elapsed_time(self.end_event)  # type: ignore[no-untyped-call,no-any-return]
        else:
            return (time.perf_counter() - self.start_time) * 1000.0


def loss_and_correct(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    task_type: str,
    elementwise: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Compute loss and correct counts for classification or LM.

    Compiled module-level function to avoid closure overhead and kernel launch stalls.

    Returns:
        Tuple of (loss tensor, correct tensor, total count).
        When elementwise=False, correct is a 0-dimensional tensor (.sum() of booleans).
        This always returns a tensor, not int - .sum() on tensors yields tensors.
    """
    if task_type == "lm":
        vocab = outputs.size(-1)
        # Use reshape instead of view - handles non-contiguous tensors safely
        loss = criterion(outputs.reshape(-1, vocab), targets.reshape(-1))
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
) -> None:
    vectorized_helpers._emit_anomaly_diagnostics(
        hub=hub,
        anomaly_report=anomaly_report,
        agent=agent,
        batch_epoch_id=batch_epoch_id,
        batch_idx=batch_idx,
        max_epochs=max_epochs,
        total_episodes=total_episodes,
        collect_debug=collect_debug,
        ratio_diagnostic=ratio_diagnostic,
        group_id=group_id,
        collect_per_layer_gradients_fn=collect_per_layer_gradients,
        check_numerical_stability_fn=check_numerical_stability,
    )


# NOTE: torch.compile() of this helper has proven unstable across CUDA/Python
# combinations (observed: TorchInductor device-side asserts during long runs).
# Keep eager execution here; the PPO policy network remains the primary target
# =============================================================================
# Vectorized PPO Training
# =============================================================================


def _train_ppo_vectorized_impl(
    n_episodes: int = 100,
    n_envs: int = DEFAULT_N_ENVS,
    max_epochs: int = DEFAULT_EPISODE_LENGTH,
    device: str = "cuda:0",
    devices: list[str] | None = None,
    task: str = "cifar_baseline",
    use_telemetry: bool = True,
    lr: float = DEFAULT_LEARNING_RATE,
    clip_ratio: float = DEFAULT_CLIP_RATIO,
    entropy_coef: float = DEFAULT_ENTROPY_COEF,  # From leyline
    entropy_coef_start: float | None = None,
    entropy_coef_end: float | None = None,
    entropy_coef_min: float = DEFAULT_ENTROPY_COEF_MIN,  # From leyline
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
    experimental_gpu_preload_gather: bool = False,
    gpu_preload_augment: bool = False,
    gpu_preload_precompute_augment: bool = False,
    amp: bool = False,
    amp_dtype: str = "auto",  # "auto", "float16", "bfloat16", or "off"
    max_grad_norm: float | None = None,  # Gradient clipping max norm (None disables)
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
    rent_host_params_floor: int = 200,
    reward_family: str = "contribution",
    permissive_gates: bool = True,
    auto_forward_g1: bool = False,
    auto_forward_g2: bool = False,
    auto_forward_g3: bool = False,
    # Ablation flags for systematic reward function experiments
    disable_pbrs: bool = False,  # Disable PBRS stage advancement shaping
    disable_terminal_reward: bool = False,  # Disable terminal accuracy bonus
    disable_anti_gaming: bool = False,  # Disable ratio_penalty and alpha_shock
    quiet_analytics: bool = False,
    force_compile: bool = False,
    telemetry_dir: str | None = None,
    ready_event: "threading.Event | None" = None,
    shutdown_event: "threading.Event | None" = None,
    group_id: str = "default",  # A/B testing group identifier
    torch_profiler: bool = False,
    torch_profiler_dir: str = "./profiler_traces",
    torch_profiler_wait: int = 1,
    torch_profiler_warmup: int = 1,
    torch_profiler_active: int = 3,
    torch_profiler_repeat: int = 1,
    torch_profiler_record_shapes: bool = False,
    torch_profiler_profile_memory: bool = False,
    torch_profiler_with_stack: bool = False,
    torch_profiler_summary: bool = False,
) -> tuple[PPOAgent, list[dict[str, Any]]]:
    """Train PPO with vectorized environments using INVERTED CONTROL FLOW.

    Key architecture: Instead of iterating environments then dataloaders,
    we iterate dataloader batches FIRST, then run all environments in parallel
    using CUDA streams. This ensures both GPUs are working simultaneously.

    Args:
        n_episodes: Total PPO update rounds (batches). Each round runs n_envs episodes.
        n_envs: Number of parallel environments per round
        max_epochs: Max epochs per episode (full train-loader passes). Also the
            LSTM sequence length because chunk_length == max_epochs.
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
    _compiled_loss_and_correct = loss_and_correct
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

    from esper.tolaria import create_model, validate_device

    if not slots:
        raise ValueError("slots parameter is required and cannot be empty")

    auto_forward_gates: frozenset[GateLevel] = frozenset(
        gate
        for gate, enabled in (
            (GateLevel.G1, auto_forward_g1),
            (GateLevel.G2, auto_forward_g2),
            (GateLevel.G3, auto_forward_g3),
        )
        if enabled
    )
    disable_advance: bool = auto_forward_g1 and auto_forward_g2 and auto_forward_g3

    # Get task spec early (needed for model creation to derive slot_config)
    # Lazy import to avoid circular dependency
    from esper.runtime import get_task_spec

    task_spec = get_task_spec(task)
    ActionEnum = task_spec.action_enum

    # Derive slot_config from host's injection specs, filtered to requested slots
    # Create a temporary model to query the host's injection topology
    from esper.kasmina.host import MorphogeneticModel

    temp_device = "cpu"  # Use CPU for temp model to avoid GPU allocation
    temp_model_raw = create_model(
        task=task_spec,
        device=temp_device,
        slots=slots,
        permissive_gates=permissive_gates,
    )
    # Type assertion: create_model returns MorphogeneticModel
    assert isinstance(temp_model_raw, MorphogeneticModel)
    temp_model: MorphogeneticModel = temp_model_raw

    # Filter specs to only include requested slots (not all host injection points)
    enabled_specs = [
        spec
        for spec in temp_model.host.injection_specs()
        if spec.slot_id in temp_model.seed_slots
    ]
    slot_config = SlotConfig.from_specs(enabled_specs)
    # Calculate host_params while we have the model (constant across all envs)
    host_params_baseline = sum(
        p.numel() for p in temp_model.get_host_parameters() if p.requires_grad
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
    validate_device(device, require_explicit_index=False)
    for env_device in devices:
        validate_device(env_device, require_explicit_index=True)

    if len(devices) > n_envs:
        raise ValueError(
            f"n_envs={n_envs} must be >= len(devices)={len(devices)} so every requested device "
            "runs at least one environment."
        )

    if gpu_preload_augment and gpu_preload_precompute_augment:
        raise ValueError(
            "gpu_preload_augment and gpu_preload_precompute_augment are mutually exclusive"
        )
    if (gpu_preload_augment or gpu_preload_precompute_augment) and not gpu_preload:
        raise ValueError(
            "gpu_preload_augment/precompute requires gpu_preload=True"
        )
    if (
        gpu_preload_augment
        or gpu_preload_precompute_augment
    ) and not task_spec.name.startswith("cifar_"):
        raise ValueError(
            "CIFAR GPU augmentations are supported only for CIFAR tasks"
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
    if rent_host_params_floor < 1:
        raise ValueError(
            f"rent_host_params_floor must be >= 1 (got {rent_host_params_floor})"
        )

    reward_config = ContributionRewardConfig(
        reward_mode=reward_mode_enum,
        param_budget=param_budget,
        param_penalty_weight=param_penalty_weight,
        sparse_reward_scale=sparse_reward_scale,
        rent_host_params_floor=rent_host_params_floor,
        disable_pbrs=disable_pbrs,
        disable_terminal_reward=disable_terminal_reward,
        disable_anti_gaming=disable_anti_gaming,
    )
    loss_reward_config = task_spec.loss_reward_config

    # Per-environment reward configs (single reward mode per run).
    env_reward_configs = [reward_config] * n_envs

    # Map environments to devices in round-robin (needed for SharedBatchIterator)
    env_device_map = [devices[i % len(devices)] for i in range(n_envs)]

    # DataLoader settings (used for SharedBatchIterator + diagnostics).
    # MED-03 fix: Log warning if batch_size not specified (could indicate schema drift)
    if "batch_size" not in task_spec.dataloader_defaults:
        _logger.debug(
            "batch_size not in task_spec.dataloader_defaults, using default 128"
        )
    default_batch_size_per_env = task_spec.dataloader_defaults.get(
        "batch_size", DEFAULT_BATCH_SIZE_TRAINING
    )
    if (
        task_spec.name.startswith("cifar_")
        and batch_size_per_env is None
        and default_batch_size_per_env == DEFAULT_BATCH_SIZE_TRAINING
    ):
        # High-throughput setting for CIFAR tasks (unless caller overrides batch size).
        default_batch_size_per_env = 512
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

    # State dimension: Obs V3 features from batch_obs_to_features().
    # For 3 slots: 23 base + 3*31 slot features = 116 dims.
    # NOTE: Telemetry features are now MERGED into slot features (31 per slot),
    # so we no longer add separate SeedTelemetry.feature_dim() per slot.
    # Blueprint embeddings (4 × num_slots) are added inside the network.
    state_dim = get_feature_size(slot_config)

    # Use EMA momentum for stable normalization during long training runs
    # (prevents distribution shift that can break PPO ratio calculations)
    obs_normalizer = RunningMeanStd((state_dim,), device=device, momentum=0.99)
    # TELE-OBS: Capture initial normalizer mean for drift detection
    initial_obs_normalizer_mean = obs_normalizer.mean.clone()

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
    ops_telemetry_enabled = not telemetry_lifecycle_only and (
        telemetry_config is None or telemetry_config.should_collect("ops_normal")
    )

    # Optional file-based telemetry logging (for programmatic callers that bypass scripts/train.py)
    if telemetry_dir and use_telemetry:
        hub.add_backend(DirectoryOutput(telemetry_dir))
        _logger.info("Telemetry logging to: %s", telemetry_dir)

    anomaly_detector = AnomalyDetector()
    start_episode = 0
    start_batch = 0

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
        metadata = checkpoint["metadata"]
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

        start_batch = metadata["batches_completed"]
        checkpoint_envs = metadata["n_envs"]
        if checkpoint_envs != n_envs:
            raise ValueError(
                f"Checkpoint n_envs={checkpoint_envs} does not match current n_envs={n_envs}"
            )
        start_episode = start_batch * n_envs

        # Emit checkpoint loaded event with typed payload
        hub.emit(
            TelemetryEvent(
                event_type=TelemetryEventType.CHECKPOINT_LOADED,
                group_id=group_id,
                data=CheckpointLoadedPayload(
                    path=str(resume_path),
                    start_episode=start_episode,
                ),
            )
        )
    else:
        # TUI mode (Sanctum/Overwatch) is optimized for debuggability. Disable
        # torch.compile here to avoid TorchInductor failures that are difficult
        # to recover from in an interactive session.
        # Determine effective compile mode: quiet_analytics disables compilation
        # unless force_compile is set (for testing compilation with TUI).
        if force_compile:
            effective_compile_mode = compile_mode
        else:
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
            entropy_anneal_steps=entropy_anneal_steps,
            device=device,
            chunk_length=chunk_length,
            num_envs=n_envs,
            max_steps_per_env=max_epochs,
            compile_mode=effective_compile_mode,  # Persisted for checkpoint resume
        )

    # Emit TRAINING_STARTED to activate Karn (Sanctum/Overwatch) and capture run config.
    entropy_anneal_summary = None
    if entropy_anneal_episodes > 0:
        entropy_anneal_summary = {
            "start": entropy_coef_start
            if entropy_coef_start is not None
            else entropy_coef,
            "end": entropy_coef_end if entropy_coef_end is not None else entropy_coef,
            "episodes": entropy_anneal_episodes,
            "steps": entropy_anneal_steps,
        }

    total_batches = n_episodes + start_batch
    total_env_episodes = total_batches * n_envs

    # Emit TRAINING_STARTED (max_batches = total PPO update rounds)
    hub.emit(
        TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            group_id=group_id,
            message=(
                f"PPO vectorized training initialized: policy_device={device}, "
                f"env_device_counts={env_device_counts}"
            ),
            data=TrainingStartedPayload(
                n_envs=n_envs,
                max_epochs=max_epochs,
                max_batches=total_batches,
                task=task,
                host_params=host_params_baseline,
                slot_ids=tuple(slot_config.slot_ids),
                seed=seed,
                n_episodes=total_env_episodes,
                lr=lr,
                clip_ratio=clip_ratio,
                entropy_coef=entropy_coef,
                param_budget=param_budget,
                policy_device=device,
                env_devices=tuple(devices),
                # Optional fields
                episode_id=f"ppo_{seed}_{total_env_episodes}ep",
                resume_path=str(resume_path) if resume_path else "",
                reward_mode=reward_mode,
                start_episode=start_episode,
                entropy_anneal=entropy_anneal_summary,
                # torch.compile status (wired to ValueDiagnosticsPanel)
                compile_enabled=(agent.compile_mode != "off"),
                compile_backend="inductor" if agent.compile_mode != "off" else None,
                compile_mode=agent.compile_mode
                if agent.compile_mode != "off"
                else None,
            ),
        )
    )

    # Create SharedBatchIterator for parallel data loading
    trainset, testset = task_spec.get_datasets()

    # Type annotation for shared iterators (union type to handle both branches)
    shared_train_iter: SharedBatchIterator | Any
    shared_test_iter: SharedBatchIterator | Any

    if gpu_preload:
        if not task_spec.name.startswith("cifar_"):
            _logger.warning(
                f"gpu_preload=True is disabled for task '{task_spec.name}' "
                "(SharedGPUBatchIterator supports CIFAR tasks only). "
                "Falling back to standard CPU DataLoader."
            )
            gpu_preload = False
    if experimental_gpu_preload_gather and not gpu_preload:
        raise ValueError(
            "experimental_gpu_preload_gather requires gpu_preload=True for CIFAR-10 tasks"
        )
    if experimental_gpu_preload_gather:
        unique_devices = list(dict.fromkeys(devices))
        if len(unique_devices) != len(devices):
            raise ValueError(
                "experimental_gpu_preload_gather requires a unique devices list (no duplicates)"
            )
        if n_envs % len(unique_devices) != 0:
            raise ValueError(
                "experimental_gpu_preload_gather requires n_envs to be divisible by number of devices"
            )

    if gpu_preload:
        if experimental_gpu_preload_gather:
            from esper.utils.data import SharedGPUGatherBatchIterator

            shared_train_iter = SharedGPUGatherBatchIterator(
                batch_size_per_env=effective_batch_size_per_env,
                n_envs=n_envs,
                env_devices=env_device_map,
                shuffle=True,
                data_root=task_spec.dataloader_defaults.get("data_root", "./data"),
                is_train=True,
                seed=seed,
                cifar_precompute_aug=gpu_preload_precompute_augment,
            )
            shared_test_iter = SharedGPUGatherBatchIterator(
                batch_size_per_env=effective_batch_size_per_env,
                n_envs=n_envs,
                env_devices=env_device_map,
                shuffle=False,
                data_root=task_spec.dataloader_defaults.get("data_root", "./data"),
                is_train=False,
                seed=seed,
                cifar_precompute_aug=gpu_preload_precompute_augment,
            )
        else:
            from esper.utils.data import SharedGPUBatchIterator

            shared_train_iter = SharedGPUBatchIterator(
                batch_size_per_env=effective_batch_size_per_env,
                n_envs=n_envs,
                env_devices=env_device_map,
                shuffle=True,
                data_root=task_spec.dataloader_defaults.get("data_root", "./data"),
                is_train=True,
                cifar_precompute_aug=gpu_preload_precompute_augment,
                seed=seed,
            )
            shared_test_iter = SharedGPUBatchIterator(
                batch_size_per_env=effective_batch_size_per_env,
                n_envs=n_envs,
                env_devices=env_device_map,
                shuffle=False,
                data_root=task_spec.dataloader_defaults.get("data_root", "./data"),
                is_train=False,
                cifar_precompute_aug=gpu_preload_precompute_augment,
                seed=seed,
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

    if num_train_batches < 1:
        message = (
            "No training batches available (num_train_batches=0). "
            "Reduce n_envs or batch_size_per_env, or increase available data."
        )
        _logger.warning(
            "%s n_envs=%s batch_size_per_env=%s devices=%s",
            message,
            n_envs,
            effective_batch_size_per_env,
            devices,
        )
        raise ValueError(message)

    # Warm up DataLoaders to ensure workers are spawned.
    #
    # CRITICAL: When running a Textual TUI (Sanctum/Overwatch), the main thread
    # must not start the TUI until ALL DataLoader workers have spawned.
    # Textual can modify terminal file descriptors and break multiprocessing
    # spawn. We therefore warm up both train and test loaders here, before we
    # signal `ready_event`.
    # Skip warmup when gpu_preload is True for any CIFAR variant (cifar_baseline, cifar_scale, etc.)
    # SharedGPUBatchIterator already has num_workers=0, and iterating it during warmup can cause
    # race conditions when accessing GPU-cached tensors across multiple devices before streams sync.
    is_cifar_task = task_spec.name.startswith("cifar_")
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
                _logger.info(
                    "AMP auto-detected BF16 support (Ampere+ GPU) - no GradScaler needed"
                )
            else:
                resolved_amp_dtype = torch.float16
                use_grad_scaler = True
                _logger.info("AMP using FP16 with GradScaler (pre-Ampere GPU)")
        else:
            raise ValueError(f"Invalid amp_dtype: {amp_dtype}")

    amp_enabled = resolved_amp_dtype is not None

    env_factory = EnvFactory(
        env_device_map=env_device_map,
        task_spec=task_spec,
        slots=slots,
        permissive_gates=permissive_gates,
        auto_forward_gates=auto_forward_gates,
        analytics=analytics,
        use_telemetry=use_telemetry,
        ops_telemetry_enabled=ops_telemetry_enabled,
        telemetry_lifecycle_only=telemetry_lifecycle_only,
        hub=hub,
        create_model=create_model,
        action_enum=ActionEnum,
        gpu_preload_augment=gpu_preload_augment,
        use_grad_scaler=use_grad_scaler,
        amp_enabled=amp_enabled,
        resolved_amp_dtype=resolved_amp_dtype,
    )

    profiler_cm = training_profiler(
        output_dir=torch_profiler_dir,
        enabled=torch_profiler,
        wait=torch_profiler_wait,
        warmup=torch_profiler_warmup,
        active=torch_profiler_active,
        repeat=torch_profiler_repeat,
        record_shapes=torch_profiler_record_shapes,
        profile_memory=torch_profiler_profile_memory,
        with_stack=torch_profiler_with_stack,
    )
    prof = profiler_cm.__enter__()
    prof_steps = 0

    try:
        history: list[dict[str, Any]] = []
        episode_history = []  # Per-episode tracking for A/B testing
        episode_outcomes: list[EpisodeOutcome] = []  # Pareto analysis outcomes
        best_avg_acc = 0.0
        best_state = None
        recent_accuracies = []
        recent_rewards = []
        consecutive_finiteness_failures = 0  # Track PPO updates with all epochs skipped
        prev_rolling_avg_acc: float | None = (
            None  # Track previous rolling avg for trend detection
        )

        episodes_completed = start_episode  # Env-episodes for telemetry/analytics
        batch_idx = start_batch  # PPO update rounds completed
        # Gradient EMA tracker for drift detection (P4-9)
        # Persists across batches to track slow degradation
        grad_ema_tracker = GradientEMATracker() if use_telemetry else None

        while batch_idx < total_batches:
            # One PPO update per full batch of environments.
            envs_this_batch = n_envs
            # Monotonic epoch id for all per-batch snapshot events (commit barrier, PPO, analytics).
            # We use "episodes completed after this batch" so resumed runs stay monotonic.
            batch_epoch_id = episodes_completed + envs_this_batch

            # Create fresh environments for this batch
            # DataLoaders are shared via SharedBatchIterator (not per-env)
            base_seed = seed + batch_idx * 10000
            env_states = [
                env_factory.create_env_state(i, base_seed)
                for i in range(envs_this_batch)
            ]
            criterion = nn.CrossEntropyLoss()
            # Per-sample loss for fused validation - enables separating main config
            # from ablations for Governor telemetry (fixes ablation signal contamination)
            val_criterion = nn.CrossEntropyLoss(reduction="none")

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
                    "hindsight_credit": 0.0,
                    "total_reward": 0.0,
                    "count": 0,
                    # Scaffold hindsight credit debugging fields (Phase 3.2)
                    "scaffold_count": 0,
                    "scaffold_delay_total": 0.0,
                }
                for _ in range(envs_this_batch)
            ]

            # Accumulate raw (unnormalized) states for deferred normalizer update.
            # We freeze normalizer stats during rollout to ensure consistent normalization
            # across all states in a batch, then update stats after PPO update.
            raw_states_for_normalizer_update = []

            # Track per-environment rollback (more sample-efficient than batch-level).
            # Only envs that experienced rollback have stale transitions.
            env_rollback_occurred = [False] * envs_this_batch

            # Pre-compute ordered slots once per batch (not per-epoch)
            # validate_slot_ids parses/sorts slot IDs - expensive to repeat 25x per episode
            ordered_slots = validate_slot_ids(list(slots))

            # Run epochs with INVERTED CONTROL FLOW
            for epoch in range(1, max_epochs + 1):
                step_timer.start()  # GPU-accurate timing (P4-1)
                dataloader_wait_ms_epoch = 0.0
                if telemetry_config is not None:
                    telemetry_config.tick_escalation()
                for env_state in env_states:
                    env_factory.configure_slot_telemetry(
                        env_state, inner_epoch=epoch, global_epoch=batch_epoch_id
                    )
                # Track gradient stats per env for telemetry sync
                env_grad_stats: list[dict[str, dict[Any, Any]] | None] = [
                    None
                ] * envs_this_batch

                # Reset per-epoch metrics by zeroing pre-allocated accumulators (faster than reallocating)
                train_totals = [0] * envs_this_batch
                train_batch_counts = [
                    0
                ] * envs_this_batch  # Track batch count for correct loss averaging
                for env_state in env_states:
                    env_state.zero_accumulators()

                # Ensure models are in training mode before training phase.
                # CRITICAL: process_val_batch/process_fused_val_batch call model.eval(), and without
                # this explicit model.train() call, all epochs after the first validation would run
                # with eval-mode semantics (frozen BatchNorm stats, disabled Dropout).
                for env_state in env_states:
                    env_state.model.train()

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
                            torch.cuda.default_stream(
                                torch.device(env_state.env_device)
                            )
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
                            loader_stream = torch.cuda.default_stream(
                                torch.device(env_state.env_device)
                            )
                            env_state.stream.wait_stream(loader_stream)
                        inputs, targets = env_batches[i]
                        if gpu_preload_augment:
                            assert env_state.augment_generator is not None
                            if env_state.stream:
                                with torch.cuda.stream(env_state.stream):
                                    inputs = augment_cifar10_batch(
                                        inputs,
                                        generator=env_state.augment_generator,
                                    )
                            else:
                                inputs = augment_cifar10_batch(
                                    inputs,
                                    generator=env_state.augment_generator,
                                )

                        # BUG-031: Defensive validation for NLL loss assertion failures
                        # If targets contain values outside [0, n_classes), the NLL loss kernel
                        # will fail with "Assertion t>=0 && t < n_classes failed".
                        # Enable with ESPER_DEBUG_TARGETS=1 to catch the issue with diagnostics.
                        if os.environ.get("ESPER_DEBUG_TARGETS"):
                            if targets.is_cuda:
                                torch.cuda.synchronize(targets.device)
                            target_min = targets.min().item()
                            target_max = targets.max().item()
                            if target_min < 0 or target_max >= task_spec.num_classes:
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
                                task_spec=task_spec,
                                resolved_amp_dtype=resolved_amp_dtype,
                                compiled_loss_and_correct=_compiled_loss_and_correct,
                                use_telemetry=collect_gradients,
                                slots=slots,
                                use_amp=amp_enabled,
                                max_grad_norm=max_grad_norm,
                            )
                        )
                        if grad_stats is not None:
                            env_grad_stats[i] = (
                                grad_stats  # Keep last batch's grad stats
                            )
                        stream_ctx = (
                            torch.cuda.stream(env_state.stream)
                            if env_state.stream
                            else nullcontext()
                        )
                        with stream_ctx:
                            env_state.train_loss_accum.add_(loss_tensor)  # type: ignore[union-attr]
                            env_state.train_correct_accum.add_(correct_tensor)  # type: ignore[union-attr]
                        train_totals[i] += total
                        train_batch_counts[i] += 1

                # Sync all streams ONCE at epoch end
                for env_state in env_states:
                    if env_state.stream:
                        env_state.stream.synchronize()

                # NOW safe to call .item() - all GPU work done
                # Accumulators guaranteed non-None after init_accumulators()
                train_losses = [
                    env_state.train_loss_accum.item()
                    if env_state.train_loss_accum is not None
                    else 0.0
                    for env_state in env_states
                ]
                train_corrects = [
                    env_state.train_correct_accum.item()
                    if env_state.train_correct_accum is not None
                    else 0.0
                    for env_state in env_states
                ]

                # Sync train metrics to env_state for telemetry (Sanctum TUI display)
                # NOTE: Loss is sum of batch means, so divide by batch count (not sample count).
                # Accuracy is sum of correct samples, so divide by sample count.
                for i, env_state in enumerate(env_states):
                    env_state.train_loss = train_losses[i] / max(
                        1, train_batch_counts[i]
                    )
                    env_state.train_acc = (
                        100.0 * train_corrects[i] / max(1, train_totals[i])
                    )

                # ===== VALIDATION + COUNTERFACTUAL (FUSED): Single pass over test data =====
                # Instead of iterating test data multiple times or performing sequential
                # forward passes, we stack all configurations into a single fused pass.

                reset_scaffolding_metrics(env_states, slots)

                # 1. Determine configurations per environment
                env_configs = build_env_configs(env_states, slots, epoch, max_epochs)

                # baseline_accs[env_idx][slot_id] = accuracy with that slot's seed disabled
                baseline_accs: list[dict[str, Any]] = [
                    {} for _ in range(envs_this_batch)
                ]
                all_disabled_accs: dict[int, float] = {}
                pair_accs: dict[int, dict[tuple[int, int], float]] = {}
                shapley_results: dict[
                    int, dict[tuple[bool, ...], tuple[float, float]]
                ] = {}
                val_totals = [0] * envs_this_batch
                val_batch_counts = [
                    0
                ] * envs_this_batch  # Track batch count for correct loss averaging

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
                            # CRITICAL: DataLoader .to(device, non_blocking=True) runs on the DEFAULT stream.
                            # We must sync env_state.stream with default stream before using the data,
                            # otherwise we may access partially-transferred data (race condition).
                            # BUG FIX: Use default_stream(), NOT current_stream() - the transfer happens
                            # on the default stream regardless of what stream is "current" in this context.
                            loader_stream = torch.cuda.default_stream(
                                torch.device(env_state.env_device)
                            )
                            env_state.stream.wait_stream(loader_stream)
                        inputs, targets = env_batches[i]

                        # SharedGPUBatchIterator now returns clones (not views) to fix race conditions.
                        # We still need record_stream to prevent premature deallocation when the
                        # tensor is used asynchronously by env_state.stream.
                        if env_state.stream and inputs.is_cuda:
                            inputs.record_stream(env_state.stream)
                            targets.record_stream(env_state.stream)

                        batch_size = inputs.size(0)
                        configs = env_configs[i]
                        num_configs = len(configs)

                        # Build alpha_overrides tensors for the fused pass
                        # Shape is topology-aware: [K*B, 1, 1, 1] for CNN, [K*B, 1, 1] for transformer
                        #
                        # IMPORTANT: Only pass alpha_override when at least one config
                        # actually overrides that slot's alpha. Passing a no-op override
                        # (e.g., alpha==0.0) forces SeedSlot.forward down the blending path
                        # and bypasses the TRAINING-stage STE shortcut, changing semantics
                        # and creating unnecessary alpha_schedule requirements.
                        alpha_overrides: dict[str, torch.Tensor] = {}
                        for slot_id in env_state.model._active_slots:
                            needs_override = any(slot_id in cfg for cfg in configs)
                            if not needs_override:
                                continue
                            slot = cast(
                                SeedSlotProtocol, env_state.model.seed_slots[slot_id]
                            )
                            # Access concrete SeedSlot for alpha_schedule assignment
                            from esper.kasmina.slot import SeedSlot

                            assert isinstance(slot, SeedSlot), (
                                "Expected SeedSlot for alpha_schedule manipulation"
                            )
                            slot_concrete: SeedSlot = slot

                            # Enforce Phase 3 contract: alpha_schedule only valid for GATE.
                            if slot_concrete.alpha_schedule is not None and (
                                slot.state is None
                                or slot.state.alpha_algorithm != AlphaAlgorithm.GATE
                            ):
                                slot_concrete.alpha_schedule = None

                            # P4-FIX: Ensure alpha_schedule exists for GATE algorithm during fused pass.
                            # This can happen if a seed is in HOLD mode and its schedule was cleared.
                            if (
                                slot.state
                                and slot.state.alpha_algorithm == AlphaAlgorithm.GATE
                                and slot_concrete.alpha_schedule is None
                            ):
                                from esper.kasmina.blending import BlendCatalog

                                topology = task_spec.topology
                                # Use default tempo steps since it's already in HOLD
                                slot_concrete.alpha_schedule = BlendCatalog.create(
                                    "gated",
                                    channels=slot_concrete.channels,
                                    topology=topology,
                                    total_steps=5,
                                ).to(slot_concrete.device)

                            current_alpha = slot.alpha
                            # Topology-aware shape for alpha_overrides
                            if task_spec.topology == "cnn":
                                alpha_shape = (num_configs * batch_size, 1, 1, 1)
                            else:  # transformer
                                alpha_shape = (num_configs * batch_size, 1, 1)
                            override_vec = torch.full(
                                alpha_shape,
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
                                    alpha_value = cfg[slot_id]
                                    assert isinstance(alpha_value, (int, float))
                                    override_vec[start:end].fill_(alpha_value)
                            alpha_overrides[slot_id] = override_vec

                        # Run FUSED validation pass with per-sample loss criterion
                        loss_per_config, correct_per_config, _ = (
                            process_fused_val_batch(
                                env_state,
                                inputs,
                                targets,
                                val_criterion,
                                alpha_overrides,
                                num_configs,
                                task_spec=task_spec,
                                compiled_loss_and_correct=_compiled_loss_and_correct,
                            )
                        )

                        stream_ctx = (
                            torch.cuda.stream(env_state.stream)
                            if env_state.stream
                            else nullcontext()
                        )
                        with stream_ctx:
                            env_cfg_correct_accums[i].add_(correct_per_config)
                            # Accumulate ONLY main config loss (idx 0) for Governor telemetry.
                            # Ablation losses would contaminate the signal since they're
                            # intentionally worse - measuring seed contribution not model health.
                            if env_state.val_loss_accum is not None:
                                env_state.val_loss_accum.add_(loss_per_config[0])
                        val_totals[i] += batch_size
                        val_batch_counts[i] += 1

                # Single sync point at end
                for env_state in env_states:
                    if env_state.stream:
                        env_state.stream.synchronize()

                # PERF: Batch GPU→CPU transfer before iterating
                # Moving tensors to CPU after sync is ~free (data already computed).
                # But .tolist() on GPU tensor would force per-tensor sync without this.
                env_cfg_correct_accums_cpu = [
                    accum.cpu() for accum in env_cfg_correct_accums
                ]

                # Sync val_loss to env_state (for Sanctum TUI display)
                # NOTE: Loss is sum of batch means, so divide by batch count (not sample count).
                for i, env_state in enumerate(env_states):
                    if env_state.val_loss_accum is not None and val_batch_counts[i] > 0:
                        env_state.val_loss = (
                            env_state.val_loss_accum.item() / val_batch_counts[i]
                        )
                    else:
                        env_state.val_loss = 0.0

                # Process results for each config
                val_corrects = [0] * envs_this_batch

                for i, env_state in enumerate(env_states):
                    correct_counts = env_cfg_correct_accums_cpu[i].tolist()
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
                            env_state.committed_val_acc = acc
                        elif kind == "solo":
                            slot_id = cfg["_slot"]
                            baseline_accs[i][slot_id] = acc
                            # Sync to metrics
                            if env_state.model.has_active_seed_in_slot(slot_id):
                                slot_for_state = cast(
                                    SeedSlotProtocol,
                                    env_state.model.seed_slots[slot_id],
                                )
                                seed_state = slot_for_state.state
                                if seed_state and seed_state.metrics:
                                    new_contribution = env_state.val_acc - acc
                                    # Compute contribution velocity (EMA of delta)
                                    prev = seed_state.metrics._prev_contribution
                                    if prev is not None:
                                        delta = new_contribution - prev
                                        # EMA with decay 0.7 (responsive to recent changes)
                                        seed_state.metrics.contribution_velocity = (
                                            0.7
                                            * seed_state.metrics.contribution_velocity
                                            + 0.3 * delta
                                        )
                                    seed_state.metrics._prev_contribution = (
                                        new_contribution
                                    )
                                    seed_state.metrics.counterfactual_contribution = (
                                        new_contribution
                                    )
                                    # Obs V3: Reset counterfactual staleness tracker on fresh measurement
                                    env_state.epochs_since_counterfactual[slot_id] = 0
                        elif kind == "all_off":
                            all_disabled_accs[i] = acc
                        elif kind == "pair":
                            if i not in pair_accs:
                                pair_accs[i] = {}
                            pair_key = cfg["_pair"]
                            assert isinstance(pair_key, tuple)
                            pair_accs[i][pair_key] = acc
                        elif kind == "shapley":
                            if i not in shapley_results:
                                shapley_results[i] = {}
                            # Validation loss approximated as 0.0 since we only track acc here
                            shapley_tuple = cfg["_tuple"]
                            assert isinstance(shapley_tuple, tuple)
                            shapley_results[i][shapley_tuple] = (0.0, acc)
                        elif kind == "committed":
                            env_state.committed_val_acc = acc

                    env_state.committed_acc_history.append(env_state.committed_val_acc)

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
                            all_disabled_acc=all_disabled_accs.get(
                                i
                            ),  # None triggers emitter fallback
                            pair_accs=pair_accs.get(i, {}),
                        )

                    # Compute interaction terms and populate scaffolding metrics
                    if len(active_slots) >= 2 and i in pair_accs:
                        # Use solo ablation fallback for single-seed: min(baseline_accs) = host-only acc
                        # Explicit None check: 0.0 is a valid baseline accuracy (model predicts nothing)
                        all_off_acc = all_disabled_accs.get(i)
                        if all_off_acc is None:
                            all_off_acc = min(baseline_accs[i].values())
                        for (idx_a, idx_b), pair_acc in pair_accs[i].items():
                            # Map indices to slot IDs
                            slot_a = active_slots[idx_a]
                            slot_b = active_slots[idx_b]
                            # Solo accuracies MUST exist - active_slots derived from baseline_accs keys
                            solo_a = baseline_accs[i][slot_a]
                            solo_b = baseline_accs[i][slot_b]
                            # I_ij = f({i,j}) - f({i}) - f({j}) + f(empty)
                            interaction = pair_acc - solo_a - solo_b + all_off_acc

                            # Track positive synergy in scaffold boost ledger for hindsight credit
                            if interaction > 0:
                                # Seed A boosted Seed B (symmetric relationship)
                                env_state.scaffold_boost_ledger[slot_a].append(
                                    (interaction, slot_b, epoch)
                                )
                                # Seed B boosted Seed A
                                env_state.scaffold_boost_ledger[slot_b].append(
                                    (interaction, slot_a, epoch)
                                )

                            # Update metrics for both seeds
                            if env_state.model.has_active_seed_in_slot(slot_a):
                                slot_obj_a = cast(
                                    SeedSlotProtocol, env_state.model.seed_slots[slot_a]
                                )
                                seed_a = slot_obj_a.state
                                if seed_a and seed_a.metrics:
                                    seed_a.metrics.interaction_sum += interaction
                                    seed_a.metrics.boost_received = max(
                                        seed_a.metrics.boost_received, interaction
                                    )

                            if env_state.model.has_active_seed_in_slot(slot_b):
                                slot_obj_b = cast(
                                    SeedSlotProtocol, env_state.model.seed_slots[slot_b]
                                )
                                seed_b = slot_obj_b.state
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
                        slot_obj = cast(
                            SeedSlotProtocol, env_state.model.seed_slots[slot_id]
                        )
                        seed_state = slot_obj.state
                        if seed_state is None or seed_state.metrics is None:
                            continue

                        upstream_sum = 0.0
                        downstream_sum = 0.0
                        for other_idx, other_id in enumerate(active_slots):
                            if other_id == slot_id:
                                continue
                            if not env_state.model.has_active_seed_in_slot(other_id):
                                continue
                            other_slot_obj = cast(
                                SeedSlotProtocol, env_state.model.seed_slots[other_id]
                            )
                            other_state = other_slot_obj.state
                            if other_state is None:
                                continue

                            other_alpha = (
                                other_state.metrics.current_alpha
                                if other_state.metrics
                                else 0.0
                            )
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
                                epoch=batch_idx + 1,
                            )
                        except (KeyError, ZeroDivisionError, ValueError) as e:
                            # HIGH-01 fix: Narrow to expected failures in Shapley computation
                            _logger.warning(
                                f"Shapley computation failed for env {i}: {e}"
                            )

                # ===== Compute epoch metrics and get BATCHED actions =====
                # NOTE: Telemetry sync (gradients/counterfactual) happens after record_accuracy()
                # so telemetry reflects the current epoch's metrics.

                # Collect signals, slot reports and action masks from all environments
                all_signals = []
                all_slot_reports = []
                all_masks = []

                # Post-action metadata for batched bootstrap computation
                all_post_action_signals = []
                all_post_action_slot_reports = []
                all_post_action_masks = []

                governor_panic_envs = []  # Track which envs need rollback

                for env_idx, env_state in enumerate(env_states):
                    model = env_state.model

                    train_loss = env_state.train_loss
                    train_acc = env_state.train_acc
                    val_loss = env_state.val_loss
                    val_acc = env_state.val_acc
                    # Track maximum accuracy for sparse reward
                    env_state.host_max_acc = max(
                        env_state.host_max_acc, env_state.val_acc
                    )

                    # Governor watchdog: snapshot when loss is stable (every 5 epochs)
                    # Also snapshot immediately after fossilization to prevent incoherent rollback
                    # (see BUG FIX comment in OP_FOSSILIZE handling above)
                    if epoch % 5 == 0 or env_state.needs_governor_snapshot:
                        env_state.governor.snapshot()
                        env_state.needs_governor_snapshot = False

                    # Governor watchdog: check vital signs after validation
                    is_panic = env_state.governor.check_vital_signs(val_loss)
                    if is_panic:
                        governor_panic_envs.append(env_idx)

                    # Gather active seeds across ALL enabled slots (multi-seed support)
                    active_seeds = []
                    for slot_id in slots:
                        if model.has_active_seed_in_slot(slot_id):
                            slot_obj = cast(SeedSlotProtocol, model.seed_slots[slot_id])
                            seed_state = slot_obj.state
                            if seed_state is not None:
                                active_seeds.append(seed_state)

                    # Record accuracy for all active seeds (per-slot stage counters + deltas)
                    for seed_state in active_seeds:
                        if seed_state.metrics:
                            seed_state.metrics.record_accuracy(val_acc)

                    # Sync gradient telemetry after record_accuracy so telemetry reflects this epoch's metrics.
                    grad_stats_for_env = env_grad_stats[env_idx]
                    synced_slot_ids: set[str] = set()
                    if use_telemetry and grad_stats_for_env is not None:
                        for slot_id, async_stats in grad_stats_for_env.items():
                            if not model.has_active_seed_in_slot(slot_id):
                                continue
                            slot_obj_for_grad = cast(
                                SeedSlotProtocol, model.seed_slots[slot_id]
                            )
                            seed_state = slot_obj_for_grad.state
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

                            # Materialize health stats for gradient health telemetry
                            health_stats = materialize_grad_stats(
                                async_stats["_health_stats"]
                            )

                            # Sync telemetry using real gradient health from collect_seed_gradients_async
                            seed_state.sync_telemetry(
                                gradient_norm=health_stats["gradient_norm"],
                                gradient_health=health_stats["gradient_health"],
                                has_vanishing=health_stats["has_vanishing"],
                                has_exploding=health_stats["has_exploding"],
                                epoch=epoch,
                                max_epochs=max_epochs,
                            )
                            synced_slot_ids.add(slot_id)

                    # Fallback: sync telemetry for active seeds that didn't get gradient stats
                    # This ensures accuracy_delta is always populated from metrics.improvement_since_stage_start
                    # Gradient parameters are omitted - sync_telemetry leaves gradient fields at defaults
                    for slot_id in slots:
                        if slot_id in synced_slot_ids:
                            continue
                        if not model.has_active_seed_in_slot(slot_id):
                            continue
                        slot_obj_fallback = cast(
                            SeedSlotProtocol, model.seed_slots[slot_id]
                        )
                        seed_state_fallback = slot_obj_fallback.state
                        if seed_state_fallback is None:
                            continue
                        # Only sync accuracy/stage telemetry - no gradient data available
                        seed_state_fallback.sync_telemetry(
                            epoch=epoch, max_epochs=max_epochs
                        )

                    slot_reports = model.get_slot_reports()

                    # Consolidate environment-level telemetry emission
                    emitters[env_idx].on_epoch_completed(epoch, env_state, slot_reports)

                    # Update signal tracker
                    # Phase 4: embargo/cooldown stages keep state while seed is removed.
                    # Availability for germination is therefore "no state", not merely "no active seed".
                    available_slots = sum(
                        1
                        for slot_id in slots
                        if model.seed_slots[slot_id].state is None
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
                    # Cache total_seeds for this env (used in action masking)
                    env_total_seeds = model.total_seeds() if model else 0

                    # Compute action mask based on current state (physical constraints only)
                    # Build slot states for ALL enabled slots (multi-slot masking)
                    slot_states = build_slot_states(slot_reports, ordered_slots)
                    mask = compute_action_masks(
                        slot_states=slot_states,
                        enabled_slots=ordered_slots,
                        total_seeds=env_total_seeds,
                        max_seeds=effective_max_seeds,
                        slot_config=slot_config,
                        device=torch.device(device),
                        topology=task_spec.topology,
                        disable_advance=disable_advance,
                    )
                    all_masks.append(mask)

                # OPTIMIZATION: Batched tensor-driven feature extraction (Obs V3)
                # Returns tuple: (obs [batch, obs_dim], blueprint_indices [batch, num_slots])
                states_batch, blueprint_indices_batch = batch_obs_to_features(
                    batch_signals=all_signals,
                    batch_slot_reports=all_slot_reports,
                    batch_env_states=env_states,
                    slot_config=slot_config,
                    device=torch.device(device),
                    max_epochs=max_epochs,
                )
                # NOTE: blueprint_indices_batch is passed to get_action() for op-conditioned value (Phase 4)

                # Stack dict masks into batched dict: {key: [n_envs, head_dim]}
                # Use static HEAD_NAMES for torch.compile compatibility
                masks_batch = {
                    key: torch.stack([m[key] for m in all_masks]).to(device)
                    for key in HEAD_NAMES
                }
                masks_batch["slot_by_op"] = torch.stack(
                    [m["slot_by_op"] for m in all_masks]
                ).to(device)

                # Accumulate raw states for deferred normalizer update
                raw_states_for_normalizer_update.append(states_batch.detach())

                # Normalize using FROZEN statistics during rollout collection.
                states_batch_normalized = obs_normalizer.normalize(states_batch)

                # TELE-OBS: Compute observation stats once per step (for Sanctum ObservationStats panel)
                # Only computed when ops telemetry is enabled to avoid overhead
                step_obs_stats = None
                if ops_telemetry_enabled:
                    step_obs_stats = compute_observation_stats(
                        states_batch,
                        normalized_obs_tensor=states_batch_normalized,
                        clip=10.0,
                        normalizer_mean=obs_normalizer.mean,
                        normalizer_var=obs_normalizer.var,
                        initial_normalizer_mean=initial_obs_normalizer_mean,
                    )

                # Get BATCHED actions from policy network with action masking (single forward pass!)
                pre_step_hiddens: list[tuple[torch.Tensor, torch.Tensor]] = []

                if batched_lstm_hidden is not None:
                    h_batch, c_batch = batched_lstm_hidden
                    for env_idx in range(len(env_states)):
                        env_h = h_batch[:, env_idx : env_idx + 1, :].clone()
                        env_c = c_batch[:, env_idx : env_idx + 1, :].clone()
                        pre_step_hiddens.append((env_h, env_c))
                else:
                    batched_lstm_hidden = agent.policy.initial_hidden(len(env_states))
                    if batched_lstm_hidden is not None:
                        init_h, init_c = batched_lstm_hidden
                        for env_idx in range(len(env_states)):
                            env_h = init_h[:, env_idx : env_idx + 1, :].clone()
                            env_c = init_c[:, env_idx : env_idx + 1, :].clone()
                            pre_step_hiddens.append((env_h, env_c))

                # get_action returns ActionResult dataclass
                action_result = agent.policy.get_action(
                    states_batch_normalized,
                    blueprint_indices=blueprint_indices_batch,
                    masks=masks_batch,
                    hidden=batched_lstm_hidden,
                    deterministic=False,
                )
                actions_dict = action_result.action
                head_log_probs = action_result.log_prob
                values_tensor = action_result.value

                # OPTIMIZATION: Update batched hidden state directly (eliminates per-env slice/cat)
                batched_lstm_hidden = action_result.hidden

                # Convert to list of dicts for per-env processing
                # PERF NOTE: Consolidate action head transfers into a single D2H copy.
                # This matters for larger env counts (16+), where per-head transfers and
                # per-env Python dict construction become a scaling bottleneck.
                actions_stacked = torch.stack(
                    [actions_dict[name] for name in HEAD_NAMES]
                )
                actions_np = actions_stacked.cpu().numpy()  # [num_heads, num_envs]
                values = values_tensor.cpu().tolist()  # .tolist() on CPU tensor is free

                # Batch compute mask stats for telemetry
                masked_np: np.ndarray | None = None  # [num_heads, num_envs]
                if ops_telemetry_enabled:
                    masked_batch = {
                        key: ~masks_batch[key].all(dim=-1)  # [num_envs] bool tensor
                        for key in HEAD_NAMES
                    }
                    masked_stacked = torch.stack(
                        [masked_batch[name] for name in HEAD_NAMES]
                    )
                    masked_np = masked_stacked.cpu().numpy()
                else:
                    masked_np = None

                # PERF: Pre-compute op_probs for telemetry ONCE before env loop.
                # Previous code called .cpu() per-env inside the loop, causing N GPU syncs
                # per step instead of 1. This was the root cause of 90% throughput drop.
                op_probs_cpu: np.ndarray | None = None
                if ops_telemetry_enabled and action_result.op_logits is not None:
                    # Batch softmax over all envs, single GPU->CPU transfer
                    op_probs_all = torch.softmax(action_result.op_logits, dim=-1)
                    op_probs_cpu = op_probs_all.cpu().numpy()

                # PERF: Pre-compute per-head confidences AND entropy for telemetry.
                # Uses batched GPU->CPU transfer: stack all heads, single transfer.
                #
                # Confidence = exp(log_prob) = P(chosen_action | valid_mask)
                # This properly handles masking via MaskedCategorical.
                #
                # Head names in order matching HeadTelemetry field positions.
                # We stack log_probs in this order, then index [0..7] to get each head's value.
                _HEAD_NAMES_FOR_TELEM = (
                    "op",
                    "slot",
                    "blueprint",
                    "style",
                    "tempo",
                    "alpha_target",
                    "alpha_speed",
                    "alpha_curve",
                )
                head_confidences_cpu: np.ndarray | None = None  # [8, num_envs]

                # NOTE: Entropy is not available during action sampling (only during PPO evaluation).
                # All entropy fields will be 0.0 until we add entropy computation to get_action().
                head_entropies_cpu: np.ndarray | None = None

                if ops_telemetry_enabled and head_log_probs:
                    # Stack all head log probs: [8, num_envs]
                    stacked_log_probs = torch.stack(
                        [head_log_probs[h] for h in _HEAD_NAMES_FOR_TELEM]
                    )
                    # Single exp + detach + transfer
                    head_confidences_cpu = (
                        torch.exp(stacked_log_probs).detach().cpu().numpy()
                    )

                # PHASE 1: Execute actions and store transitions (bootstrap patched after)
                truncated_bootstrap_targets: list[tuple[int, int]] = []

                # Cache per-head tensors for buffer writes (avoid dict lookups in hot loop)
                slot_log_probs_batch = head_log_probs["slot"]
                blueprint_log_probs_batch = head_log_probs["blueprint"]
                style_log_probs_batch = head_log_probs["style"]
                tempo_log_probs_batch = head_log_probs["tempo"]
                alpha_target_log_probs_batch = head_log_probs["alpha_target"]
                alpha_speed_log_probs_batch = head_log_probs["alpha_speed"]
                alpha_curve_log_probs_batch = head_log_probs["alpha_curve"]
                op_log_probs_batch = head_log_probs["op"]

                slot_by_op_masks_batch = masks_batch["slot_by_op"]
                blueprint_masks_batch = masks_batch["blueprint"]
                style_masks_batch = masks_batch["style"]
                tempo_masks_batch = masks_batch["tempo"]
                alpha_target_masks_batch = masks_batch["alpha_target"]
                alpha_speed_masks_batch = masks_batch["alpha_speed"]
                alpha_curve_masks_batch = masks_batch["alpha_curve"]
                op_masks_batch = masks_batch["op"]

                for env_idx, env_state in enumerate(env_states):
                    model = env_state.model
                    signals = all_signals[env_idx]
                    value = values[env_idx]

                    # Parse sampled action indices and derive values (Deduplication)
                    slot_action = int(actions_np[_HEAD_SLOT_IDX, env_idx])
                    blueprint_action = int(actions_np[_HEAD_BLUEPRINT_IDX, env_idx])
                    style_action = int(actions_np[_HEAD_STYLE_IDX, env_idx])
                    tempo_action = int(actions_np[_HEAD_TEMPO_IDX, env_idx])
                    alpha_target_action = int(
                        actions_np[_HEAD_ALPHA_TARGET_IDX, env_idx]
                    )
                    alpha_speed_action = int(actions_np[_HEAD_ALPHA_SPEED_IDX, env_idx])
                    alpha_curve_action = int(actions_np[_HEAD_ALPHA_CURVE_IDX, env_idx])
                    op_action = int(actions_np[_HEAD_OP_IDX, env_idx])

                    action_dict: dict[str, int] | None = None
                    if ops_telemetry_enabled:
                        action_dict = {
                            "slot": slot_action,
                            "blueprint": blueprint_action,
                            "style": style_action,
                            "tempo": tempo_action,
                            "alpha_target": alpha_target_action,
                            "alpha_speed": alpha_speed_action,
                            "alpha_curve": alpha_curve_action,
                            "op": op_action,
                        }
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
                    ) = parse_sampled_action(
                        env_idx,
                        op_action,
                        slot_action,
                        style_action,
                        alpha_target_action,
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
                        # Stream safety: rollback mutates model tensors; ensure it runs on the
                        # per-env CUDA stream to avoid default-stream leakage and races.
                        rollback_ctx = (
                            torch.cuda.stream(env_state.stream)
                            if env_state.stream
                            else nullcontext()
                        )
                        with rollback_ctx:
                            env_state.governor.execute_rollback(env_id=env_idx)
                        env_rollback_occurred[env_idx] = True

                        # CRITICAL: Clear optimizer momentum after rollback.
                        # PyTorch's load_state_dict() copies weights IN-PLACE, so
                        # Parameter objects retain their identity (same id()). The
                        # optimizer's state dict is keyed by Parameter objects, so
                        # momentum/variance buffers SURVIVE the rollback. Without
                        # clearing, SGD momentum continues pushing toward the
                        # diverged state that caused the panic, risking immediate
                        # re-divergence. See B1-PT-01 correction notes.
                        env_state.host_optimizer.state.clear()
                        for seed_opt in env_state.seed_optimizers.values():
                            seed_opt.state.clear()

                    # Compute reward
                    scoreboard = analytics._get_scoreboard(env_idx)
                    host_params = scoreboard.host_params

                    effective_seed_params, alpha_delta_sq_sum = (
                        compute_rent_and_shock_inputs(
                            model=model,
                            slot_ids=slots,
                            host_params=host_params,
                            host_params_floor=env_reward_configs[
                                env_idx
                            ].rent_host_params_floor,
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
                    # Match ops_telemetry_enabled logic: default to True when no config
                    # This ensures reward_components are computed when on_last_action is called
                    collect_reward_summary = (
                        telemetry_config is None
                        or telemetry_config.should_collect("ops_normal")
                    )

                    seed_params_for_slot = (
                        cast(
                            SeedSlotProtocol, model.seed_slots[target_slot]
                        ).active_seed_params
                        if slot_is_enabled
                        else 0
                    )
                    seed_info = SeedInfo.from_seed_state(
                        seed_state, seed_params_for_slot
                    )

                    # Initialize reward_components to None (only populated for CONTRIBUTION family)
                    reward_components: RewardComponentsTelemetry | None = None

                    if reward_family_enum == RewardFamily.CONTRIBUTION:
                        stable_val_acc = None
                        if env_reward_configs[env_idx].reward_mode == RewardMode.ESCROW:
                            window = env_reward_configs[env_idx].escrow_stable_window
                            if window <= 0:
                                raise ValueError(
                                    f"escrow_stable_window must be positive, got {window}"
                                )
                            acc_history = signals.accuracy_history
                            if not acc_history:
                                raise RuntimeError(
                                    "ESCROW stable accuracy requested before any accuracy history exists"
                                )
                            k = window if window <= len(acc_history) else len(acc_history)
                            stable_val_acc = min(acc_history[-k:])
                        escrow_credit_prev = env_state.escrow_credit[target_slot]
                        fossilized_seed_params = 0
                        for slot_id in slots:
                            slot_obj = cast(
                                SeedSlotProtocol, model.seed_slots[slot_id]
                            )
                            slot_seed_state = slot_obj.state
                            if slot_seed_state is None or slot_seed_state.metrics is None:
                                continue
                            if slot_seed_state.stage == SeedStage.FOSSILIZED:
                                fossilized_seed_params += int(
                                    slot_seed_state.metrics.seed_param_count
                                )
                                if slot_obj.alpha_schedule is not None:
                                    fossilized_seed_params += sum(
                                        p.numel()
                                        for p in slot_obj.alpha_schedule.parameters()
                                    )
                        acc_at_germination = (
                            env_state.acc_at_germination[target_slot]
                            if target_slot in env_state.acc_at_germination
                            else None
                        )
                        seed_id = seed_state.seed_id if seed_state is not None else None
                        force_reward_components = (
                            env_reward_configs[env_idx].reward_mode == RewardMode.ESCROW
                        )
                        if (
                            emit_reward_components_event
                            or collect_reward_summary
                            or force_reward_components
                        ):
                            reward, reward_components = cast(
                                tuple[float, Any],
                                compute_reward(
                                    action=action_for_reward,
                                    seed_contribution=seed_contribution,
                                    val_acc=env_state.val_acc,
                                    seed_info=seed_info,
                                    epoch=epoch,
                                    max_epochs=max_epochs,
                                    total_params=model.total_params,
                                    host_params=host_params,
                                    acc_at_germination=acc_at_germination,
                                    acc_delta=signals.metrics.accuracy_delta,
                                    committed_val_acc=env_state.committed_val_acc,
                                    fossilized_seed_params=fossilized_seed_params,
                                    num_fossilized_seeds=env_state.seeds_fossilized,
                                    num_contributing_fossilized=env_state.contributing_fossilized,
                                    config=env_reward_configs[env_idx],
                                    return_components=True,
                                    effective_seed_params=effective_seed_params,
                                    alpha_delta_sq_sum=alpha_delta_sq_sum,
                                    stable_val_acc=stable_val_acc,
                                    escrow_credit_prev=escrow_credit_prev,
                                    slot_id=target_slot,
                                    seed_id=seed_id,
                                ),
                            )
                            if target_slot in baseline_accs[env_idx]:
                                reward_components.host_baseline_acc = baseline_accs[
                                    env_idx
                                ][target_slot]
                            if (
                                force_reward_components
                                and reward_components is not None
                            ):
                                env_state.escrow_credit[target_slot] = (
                                    reward_components.escrow_credit_next
                                )
                        else:
                            reward = cast(
                                float,
                                compute_reward(
                                    action=action_for_reward,
                                    seed_contribution=seed_contribution,
                                    val_acc=env_state.val_acc,
                                    seed_info=seed_info,
                                    epoch=epoch,
                                    max_epochs=max_epochs,
                                    total_params=model.total_params,
                                    host_params=host_params,
                                    acc_at_germination=acc_at_germination,
                                    acc_delta=signals.metrics.accuracy_delta,
                                    committed_val_acc=env_state.committed_val_acc,
                                    fossilized_seed_params=fossilized_seed_params,
                                    num_fossilized_seeds=env_state.seeds_fossilized,
                                    num_contributing_fossilized=env_state.contributing_fossilized,
                                    config=env_reward_configs[env_idx],
                                    effective_seed_params=effective_seed_params,
                                    alpha_delta_sq_sum=alpha_delta_sq_sum,
                                    stable_val_acc=stable_val_acc,
                                    escrow_credit_prev=escrow_credit_prev,
                                    slot_id=target_slot,
                                    seed_id=seed_id,
                                ),
                            )
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

                    if (
                        env_reward_configs[env_idx].reward_mode == RewardMode.ESCROW
                        and epoch == max_epochs
                    ):
                        assert reward_components is not None, (
                            "RewardMode.ESCROW requires return_components=True"
                        )
                        escrow_forfeit = 0.0
                        for slot_id in slots:
                            slot_obj = cast(
                                SeedSlotProtocol, model.seed_slots[slot_id]
                            )
                            slot_seed_state = slot_obj.state
                            if slot_seed_state is None:
                                continue
                            if slot_seed_state.stage != SeedStage.FOSSILIZED:
                                escrow_forfeit += env_state.escrow_credit[slot_id]
                        if escrow_forfeit != 0.0:
                            reward -= escrow_forfeit
                            reward_components.escrow_forfeit = -escrow_forfeit

                    # Germination deposit clawback: seeds that never reached BLENDING must
                    # repay the one-time PBRS germination bonus. This encourages completing
                    # the scaffolding loop (GERMINATE → BLEND) instead of farming last-minute
                    # germinations that never contribute.
                    if (
                        reward_family_enum == RewardFamily.CONTRIBUTION
                        and epoch == max_epochs
                        and env_reward_configs[env_idx].reward_mode
                        in (RewardMode.SHAPED, RewardMode.ESCROW)
                        and not env_reward_configs[env_idx].disable_pbrs
                    ):
                        phi_germinated = STAGE_POTENTIALS[SeedStage.GERMINATED]
                        germination_bonus = (
                            env_reward_configs[env_idx].pbrs_weight
                            * (env_reward_configs[env_idx].gamma * phi_germinated)
                        )
                        germination_forfeit = 0.0
                        for slot_id in slots:
                            slot_obj = cast(SeedSlotProtocol, model.seed_slots[slot_id])
                            slot_seed_state = slot_obj.state
                            if slot_seed_state is None:
                                continue
                            if slot_seed_state.stage not in (SeedStage.GERMINATED, SeedStage.TRAINING):
                                continue
                            seed_age_epochs = slot_seed_state.metrics.epochs_total
                            discount = env_reward_configs[env_idx].gamma**seed_age_epochs
                            if discount <= 0.0:
                                raise ValueError(
                                    "Invalid gamma discount for germination clawback: "
                                    f"gamma={env_reward_configs[env_idx].gamma} seed_age_epochs={seed_age_epochs}"
                                )
                            # Discount-corrected refund: ensures the terminal clawback cancels
                            # the earlier germination bonus in discounted return space.
                            germination_forfeit += germination_bonus / discount
                        if germination_forfeit != 0.0:
                            reward -= germination_forfeit
                            if reward_components is not None:
                                reward_components.action_shaping -= germination_forfeit

                    reward += env_state.pending_auto_prune_penalty
                    env_state.pending_auto_prune_penalty = 0.0

                    # Add any pending hindsight credit BEFORE normalization
                    # (DRL Specialist review: credit should go through normalizer for scale consistency)
                    hindsight_credit_applied = 0.0
                    if env_state.pending_hindsight_credit > 0:
                        hindsight_credit_applied = env_state.pending_hindsight_credit
                        reward += hindsight_credit_applied
                        env_state.pending_hindsight_credit = 0.0
                        # Populate RewardComponentsTelemetry for shaped_reward_ratio calculation
                        if collect_reward_summary and reward_components is not None:
                            reward_components.hindsight_credit = (
                                hindsight_credit_applied
                            )

                    if reward_components is not None:
                        reward_components.total_reward = reward

                    # Normalize reward for PPO stability (P1-6 fix)
                    normalized_reward = reward_normalizer.update_and_normalize(reward)
                    # B11-CR-03 fix: Store RAW rewards for telemetry interpretability
                    # PPO buffer uses normalized_reward (for training stability)
                    # Telemetry uses raw reward (for cross-run comparability)
                    env_state.episode_rewards.append(reward)

                    if collect_reward_summary and reward_components is not None:
                        summary = reward_summary_accum[env_idx]
                        summary["total_reward"] += reward
                        if reward_components.bounded_attribution is not None:
                            summary["bounded_attribution"] += (
                                reward_components.bounded_attribution
                            )
                        summary["compute_rent"] += reward_components.compute_rent
                        summary["alpha_shock"] += reward_components.alpha_shock
                        summary["hindsight_credit"] += hindsight_credit_applied
                        summary["count"] += 1

                    # Execute action
                    # Stream safety: lifecycle ops can create/move CUDA tensors (germination
                    # validation probes, module moves, etc). Run them on env_state.stream.
                    lifecycle_ctx = (
                        torch.cuda.stream(env_state.stream)
                        if env_state.stream
                        else nullcontext()
                    )
                    with lifecycle_ctx:
                        if slot_is_enabled:
                            if (
                                op_action == OP_GERMINATE
                                and model.seed_slots[target_slot].state is None
                            ):
                                env_state.acc_at_germination[target_slot] = (
                                    env_state.val_acc
                                )
                                env_state.escrow_credit[target_slot] = 0.0
                                blueprint_id = BLUEPRINT_IDS[blueprint_action]
                                assert blueprint_id is not None, (
                                    "NULL blueprint should not reach germination"
                                )
                                model.germinate_seed(
                                    blueprint_id,
                                    f"ep{episodes_completed + env_idx}_env{env_idx}_seed_{env_state.seeds_created}",
                                    slot=target_slot,
                                    blend_algorithm_id=blend_algorithm_id,
                                    blend_tempo_epochs=TEMPO_TO_EPOCHS[
                                        TempoAction(tempo_action)
                                    ],
                                    alpha_algorithm=alpha_algorithm,
                                    alpha_target=alpha_target,
                                )
                                env_state.init_obs_v3_slot_tracking(target_slot)
                                env_state.seeds_created += 1
                                env_state.germinate_count += 1  # TELE-610
                                env_state.seed_optimizers.pop(target_slot, None)
                                action_success = True
                            elif op_action == OP_FOSSILIZE:
                                action_success = _advance_active_seed(
                                    model, target_slot
                                )
                                if action_success:
                                    env_state.seeds_fossilized += 1
                                    env_state.fossilize_count += 1  # TELE-610
                                    if seed_info is not None and (
                                        seed_info.total_improvement
                                        >= DEFAULT_MIN_FOSSILIZE_CONTRIBUTION
                                    ):
                                        env_state.contributing_fossilized += 1
                                    env_state.acc_at_germination.pop(target_slot, None)

                                    # Compute temporally-discounted hindsight credit for scaffolds
                                    beneficiary_improvement = (
                                        seed_info.total_improvement
                                        if seed_info
                                        else 0.0
                                    )
                                    if beneficiary_improvement > 0:
                                        # Use outer loop epoch variable (not per-env counter)
                                        current_epoch = epoch
                                        total_credit = 0.0
                                        scaffold_count = 0
                                        total_delay = 0

                                        # Find all scaffolds that boosted this beneficiary
                                        for (
                                            scaffold_slot,
                                            boosts,
                                        ) in env_state.scaffold_boost_ledger.items():
                                            for (
                                                boost_given,
                                                beneficiary_slot,
                                                epoch_of_boost,
                                            ) in boosts:
                                                if (
                                                    beneficiary_slot == target_slot
                                                    and boost_given > 0
                                                ):
                                                    # Temporal discount: credit decays with distance
                                                    delay = (
                                                        current_epoch - epoch_of_boost
                                                    )
                                                    discount = DEFAULT_GAMMA**delay

                                                    # Compute discounted hindsight credit
                                                    raw_credit = compute_scaffold_hindsight_credit(
                                                        boost_given=boost_given,
                                                        beneficiary_improvement=beneficiary_improvement,
                                                        credit_weight=HINDSIGHT_CREDIT_WEIGHT,
                                                    )
                                                    total_credit += (
                                                        raw_credit * discount
                                                    )
                                                    scaffold_count += 1
                                                    total_delay += delay

                                        # Cap total credit to prevent runaway values
                                        total_credit = min(
                                            total_credit, MAX_HINDSIGHT_CREDIT
                                        )

                                        env_state.pending_hindsight_credit += (
                                            total_credit
                                        )

                                        # Track scaffold metrics for telemetry (per-environment)
                                        if collect_reward_summary:
                                            summary = reward_summary_accum[env_idx]
                                            summary["scaffold_count"] += scaffold_count
                                            summary["scaffold_delay_total"] += (
                                                total_delay
                                            )

                                        # Clear this beneficiary from all ledgers (it's now fossilized)
                                        for scaffold_slot in list(
                                            env_state.scaffold_boost_ledger.keys()
                                        ):
                                            env_state.scaffold_boost_ledger[
                                                scaffold_slot
                                            ] = [
                                                (b, ben, e)
                                                for (
                                                    b,
                                                    ben,
                                                    e,
                                                ) in env_state.scaffold_boost_ledger[
                                                    scaffold_slot
                                                ]
                                                if ben != target_slot
                                            ]
                                            if not env_state.scaffold_boost_ledger[
                                                scaffold_slot
                                            ]:
                                                del env_state.scaffold_boost_ledger[
                                                    scaffold_slot
                                                ]

                                    # B8-DRL-02 FIX: Clean up seed optimizer after fossilization
                                    # (was missing - memory leak for fossilized seed optimizers)
                                    env_state.seed_optimizers.pop(target_slot, None)

                                    # BUG FIX: Trigger governor snapshot after fossilization
                                    # Without this, a rollback between fossilization and the next
                                    # periodic snapshot (every 5 epochs) produces an incoherent state:
                                    # - Fossilized seed weights not in snapshot (excluded when TRAINING)
                                    # - Fossilized seeds can't be pruned during rollback
                                    # - Result: host reverts but fossilized seed keeps stale weights
                                    # The snapshot must be taken OUTSIDE the CUDA stream context,
                                    # so we set a flag here and take the snapshot later.
                                    env_state.needs_governor_snapshot = True
                            elif (
                                op_action == OP_PRUNE
                                and model.has_active_seed_in_slot(target_slot)
                                and seed_state is not None
                                and seed_state.stage
                                in (
                                    SeedStage.GERMINATED,
                                    SeedStage.TRAINING,
                                    SeedStage.BLENDING,
                                    SeedStage.HOLDING,
                                )
                                # BUG-020 fix: enforce MIN_PRUNE_AGE at execution gate
                                and seed_info is not None
                                and seed_info.seed_age_epochs >= MIN_PRUNE_AGE
                            ):
                                speed_steps = ALPHA_SPEED_TO_STEPS[
                                    AlphaSpeedAction(alpha_speed_action)
                                ]
                                curve_action_obj = AlphaCurveAction(alpha_curve_action)
                                curve = curve_action_obj.to_curve()
                                steepness = curve_action_obj.to_steepness()
                                target_slot_obj = cast(
                                    SeedSlotProtocol, model.seed_slots[target_slot]
                                )
                                # Access concrete SeedSlot for schedule_prune/prune
                                from esper.kasmina.slot import SeedSlot

                                assert isinstance(target_slot_obj, SeedSlot)
                                if speed_steps <= 0:
                                    action_success = target_slot_obj.prune(
                                        reason="policy_prune", initiator="policy"
                                    )
                                else:
                                    action_success = target_slot_obj.schedule_prune(
                                        steps=speed_steps,
                                        curve=curve,
                                        steepness=steepness,
                                        initiator="policy",
                                    )
                                if action_success:
                                    env_state.prune_count += 1  # TELE-610
                                    env_state.seed_optimizers.pop(target_slot, None)
                                    env_state.acc_at_germination.pop(target_slot, None)
                            elif (
                                op_action == OP_SET_ALPHA_TARGET
                                and model.has_active_seed_in_slot(target_slot)
                            ):
                                target_slot_obj_alpha = cast(
                                    SeedSlotProtocol, model.seed_slots[target_slot]
                                )
                                from esper.kasmina.slot import SeedSlot

                                assert isinstance(target_slot_obj_alpha, SeedSlot)
                                curve_action_alpha = AlphaCurveAction(
                                    alpha_curve_action
                                )
                                action_success = target_slot_obj_alpha.set_alpha_target(
                                    alpha_target=alpha_target,
                                    steps=ALPHA_SPEED_TO_STEPS[
                                        AlphaSpeedAction(alpha_speed_action)
                                    ],
                                    curve=curve_action_alpha.to_curve(),
                                    steepness=curve_action_alpha.to_steepness(),
                                    alpha_algorithm=alpha_algorithm,
                                    initiator="policy",
                                )
                                # B8-DRL-02 FIX: Removed incorrect seed_optimizers.pop() here.
                                # SET_ALPHA_TARGET doesn't terminate the seed - it's still active.
                            elif (
                                op_action == OP_ADVANCE
                                and model.has_active_seed_in_slot(target_slot)
                            ):
                                target_slot_obj_advance = cast(
                                    SeedSlotProtocol, model.seed_slots[target_slot]
                                )
                                gate_result = target_slot_obj_advance.advance_stage()
                                action_success = gate_result.passed
                                # B8-DRL-02 FIX: Only pop optimizer if seed terminated.
                                # ADVANCE can move to non-terminal stages (TRAINING, BLENDING, etc.)
                                # where the seed is still active and needs its optimizer.
                                if action_success and not model.has_active_seed_in_slot(
                                    target_slot
                                ):
                                    env_state.seed_optimizers.pop(target_slot, None)
                            elif op_action == OP_WAIT:
                                # WAIT is always a valid no-op for enabled slots.
                                action_success = True
                        elif op_action == OP_WAIT:
                            action_success = True

                    if action_success:
                        env_state.successful_action_counts[action_for_reward.name] = (
                            env_state.successful_action_counts.get(
                                action_for_reward.name, 0
                            )
                            + 1
                        )

                    # Obs V3: Update action feedback state for next timestep's feature extraction
                    env_state.last_action_success = action_success
                    env_state.last_action_op = op_action

                    # Obs V3: Update gradient health history for LSTM trend detection
                    # Use slot reports from current timestep (before action) as "prev" for next timestep
                    slot_reports_for_env = all_slot_reports[env_idx]
                    for slot_id in ordered_slots:
                        if slot_id in slot_reports_for_env:
                            report = slot_reports_for_env[slot_id]
                            if report.telemetry is not None:
                                health_val = report.telemetry.gradient_health
                                # Fail-fast if gradient_health contains NaN/inf
                                # This would poison observation features and crash get_action()
                                if not math.isfinite(health_val):
                                    raise ValueError(
                                        f"NaN/inf gradient_health from telemetry for slot {slot_id}: "
                                        f"{health_val}. Check materialize_grad_stats() or sync_telemetry()."
                                    )
                                env_state.gradient_health_prev[slot_id] = health_val

                            # Obs V3: Increment epochs since last counterfactual measurement
                            # This is reset to 0 when counterfactual_contribution is updated (see line ~2191)
                            if slot_id in env_state.epochs_since_counterfactual:
                                env_state.epochs_since_counterfactual[slot_id] += 1
                            else:
                                # Initialize tracking for new slots
                                env_state.epochs_since_counterfactual[slot_id] = 0

                    # Consolidate telemetry via emitter
                    if ops_telemetry_enabled and masked_np is not None:
                        assert action_dict is not None
                        masked_flags = {
                            head: bool(masked_np[head_idx, env_idx])
                            for head_idx, head in enumerate(HEAD_NAMES)
                        }

                        post_slot_obj = cast(
                            SeedSlotProtocol, model.seed_slots[target_slot]
                        )
                        post_slot_state = post_slot_obj.state
                        active_algo = (
                            post_slot_state.alpha_algorithm.name
                            if post_slot_state
                            else None
                        )
                        slot_reports_for_decision = all_slot_reports[env_idx]
                        decision_slot_states: dict[str, str] = {}
                        for slot_id in ordered_slots:
                            if slot_id not in slot_reports_for_decision:
                                decision_slot_states[slot_id] = "Empty"
                                continue
                            slot_report = slot_reports_for_decision[slot_id]
                            stage_label = slot_report.stage.name.title()
                            decision_slot_states[slot_id] = (
                                f"{stage_label} {slot_report.metrics.total_improvement:.0f}%"
                            )

                        # Compute action_confidence, alternatives, and decision_entropy from op_logits
                        # PolicyBundle returns op_logits for telemetry when available
                        # PERF: op_probs_cpu is pre-computed BEFORE the env loop to avoid N GPU syncs
                        action_confidence = None
                        alternatives: list[tuple[str, float]] | None = None
                        decision_entropy = None
                        if op_probs_cpu is not None and env_idx < op_probs_cpu.shape[0]:
                            probs_cpu = op_probs_cpu[env_idx]
                            chosen_op = op_action

                            # action_confidence = P(chosen_op)
                            if 0 <= chosen_op < len(probs_cpu):
                                action_confidence = float(probs_cpu[chosen_op])

                            # alternatives = top-2 ops excluding chosen
                            ranked = sorted(
                                enumerate(probs_cpu), key=lambda x: x[1], reverse=True
                            )
                            alternatives = [
                                (OP_NAMES[op_idx], float(prob))
                                for op_idx, prob in ranked
                                if op_idx != chosen_op
                            ][:2]

                            # decision_entropy = -sum(p * log(p)) for op head
                            entropy_sum = 0.0
                            for p in probs_cpu:
                                if p > 1e-8:  # Avoid log(0)
                                    entropy_sum -= p * math.log(p)
                            decision_entropy = entropy_sum

                        # Build HeadTelemetry for this env (typed dataclass, not raw dict)
                        head_telem: HeadTelemetry | None = None
                        if head_confidences_cpu is not None:
                            head_telem = HeadTelemetry(
                                op_confidence=float(head_confidences_cpu[0, env_idx]),
                                slot_confidence=float(head_confidences_cpu[1, env_idx]),
                                blueprint_confidence=float(
                                    head_confidences_cpu[2, env_idx]
                                ),
                                style_confidence=float(
                                    head_confidences_cpu[3, env_idx]
                                ),
                                tempo_confidence=float(
                                    head_confidences_cpu[4, env_idx]
                                ),
                                alpha_target_confidence=float(
                                    head_confidences_cpu[5, env_idx]
                                ),
                                alpha_speed_confidence=float(
                                    head_confidences_cpu[6, env_idx]
                                ),
                                curve_confidence=float(
                                    head_confidences_cpu[7, env_idx]
                                ),
                                # Entropy (0.0 if not available)
                                op_entropy=float(head_entropies_cpu[0, env_idx])
                                if head_entropies_cpu is not None
                                else 0.0,
                                slot_entropy=float(head_entropies_cpu[1, env_idx])
                                if head_entropies_cpu is not None
                                else 0.0,
                                blueprint_entropy=float(head_entropies_cpu[2, env_idx])
                                if head_entropies_cpu is not None
                                else 0.0,
                                style_entropy=float(head_entropies_cpu[3, env_idx])
                                if head_entropies_cpu is not None
                                else 0.0,
                                tempo_entropy=float(head_entropies_cpu[4, env_idx])
                                if head_entropies_cpu is not None
                                else 0.0,
                                alpha_target_entropy=float(
                                    head_entropies_cpu[5, env_idx]
                                )
                                if head_entropies_cpu is not None
                                else 0.0,
                                alpha_speed_entropy=float(
                                    head_entropies_cpu[6, env_idx]
                                )
                                if head_entropies_cpu is not None
                                else 0.0,
                                curve_entropy=float(head_entropies_cpu[7, env_idx])
                                if head_entropies_cpu is not None
                                else 0.0,
                            )

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
                            reward_components=reward_components,  # Pass directly (may be None for LOSS family)
                            head_telemetry=head_telem,
                            # TELE-OBS: Only pass for env 0 to avoid redundant data (batch-level stat)
                            observation_stats=step_obs_stats if env_idx == 0 else None,
                        )

                    # Store transition directly into rollout buffer.
                    done = epoch == max_epochs
                    truncated = done
                    effective_op_action = int(action_for_reward)

                    step_idx = agent.buffer.step_counts[env_idx]
                    agent.buffer.add(
                        env_id=env_idx,
                        state=states_batch_normalized[env_idx].detach(),
                        blueprint_indices=blueprint_indices_batch[env_idx].detach(),
                        slot_action=slot_action,
                        blueprint_action=blueprint_action,
                        style_action=style_action,
                        tempo_action=tempo_action,
                        alpha_target_action=alpha_target_action,
                        alpha_speed_action=alpha_speed_action,
                        alpha_curve_action=alpha_curve_action,
                        op_action=op_action,
                        effective_op_action=effective_op_action,
                        slot_log_prob=slot_log_probs_batch[env_idx],
                        blueprint_log_prob=blueprint_log_probs_batch[env_idx],
                        style_log_prob=style_log_probs_batch[env_idx],
                        tempo_log_prob=tempo_log_probs_batch[env_idx],
                        alpha_target_log_prob=alpha_target_log_probs_batch[env_idx],
                        alpha_speed_log_prob=alpha_speed_log_probs_batch[env_idx],
                        alpha_curve_log_prob=alpha_curve_log_probs_batch[env_idx],
                        op_log_prob=op_log_probs_batch[env_idx],
                        value=value,
                        reward=normalized_reward,
                        done=done,
                        slot_mask=slot_by_op_masks_batch[env_idx, op_action],
                        blueprint_mask=blueprint_masks_batch[env_idx],
                        style_mask=style_masks_batch[env_idx],
                        tempo_mask=tempo_masks_batch[env_idx],
                        alpha_target_mask=alpha_target_masks_batch[env_idx],
                        alpha_speed_mask=alpha_speed_masks_batch[env_idx],
                        alpha_curve_mask=alpha_curve_masks_batch[env_idx],
                        op_mask=op_masks_batch[env_idx],
                        hidden_h=pre_step_hiddens[env_idx][0].detach(),
                        hidden_c=pre_step_hiddens[env_idx][1].detach(),
                        truncated=truncated,
                        bootstrap_value=0.0,
                    )
                    if truncated:
                        truncated_bootstrap_targets.append((env_idx, step_idx))

                    if done:
                        agent.buffer.end_episode(env_id=env_idx)
                        # NOTE: Do NOT reset batched_lstm_hidden here. The bootstrap value computation
                        # (after the epoch loop) requires the carried episode hidden state to correctly
                        # estimate V(s_{t+1}) for truncated episodes. Resetting to initial_hidden() would
                        # bias the GAE computation by computing V(s_{t+1}) with a "memory-wiped" agent.
                        # The next rollout will initialize fresh hidden states anyway (line 1747).

                    # Mechanical lifecycle advance
                    for slot_id in slots:
                        slot_for_step = cast(
                            SeedSlotProtocol, model.seed_slots[slot_id]
                        )

                        # Advance lifecycle (may set auto_pruned if scheduled prune completes)
                        slot_for_step.step_epoch()

                        # Check auto-prune flag AFTER step_epoch to catch both:
                        # - Governor prunes (set flag outside step_epoch, caught on next check)
                        # - Scheduled prune completions (set flag inside step_epoch, caught immediately)
                        if (
                            slot_for_step.state
                            and slot_for_step.state.metrics.auto_pruned
                        ):
                            env_state.pending_auto_prune_penalty += (
                                reward_config.auto_prune_penalty
                            )
                            # Clear one-shot flag after reading
                            slot_for_step.state.metrics.auto_pruned = False

                        if not model.has_active_seed_in_slot(slot_id):
                            env_state.seed_optimizers.pop(slot_id, None)
                            env_state.acc_at_germination.pop(slot_id, None)
                            env_state.gradient_ratio_ema.pop(slot_id, None)
                            env_state.escrow_credit[slot_id] = 0.0
                        if slot_for_step.state is None:
                            env_state.clear_obs_v3_slot_tracking(slot_id)

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
                                active_seeds=cast(
                                    list[SeedStateProtocol],
                                    [
                                        s.state
                                        for s in model.seed_slots.values()
                                        if s.is_active and s.state
                                    ],
                                ),
                                available_slots=sum(
                                    1
                                    for s in model.seed_slots.values()
                                    if s.state is None
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
                                disable_advance=disable_advance,
                            )
                        )

                    if epoch == max_epochs:
                        env_final_accs[env_idx] = env_state.val_acc
                        env_total_rewards[env_idx] = sum(env_state.episode_rewards)

                        # Track episode completion for A/B testing
                        episode_history.append(
                            {
                                "env_id": env_idx,
                                "episode_reward": env_total_rewards[env_idx],
                                "final_accuracy": env_final_accs[env_idx],
                            }
                        )

                        # Compute stability score from reward variance
                        recent_ep_rewards = (
                            env_state.episode_rewards[-20:]
                            if len(env_state.episode_rewards) >= 20
                            else env_state.episode_rewards
                        )
                        if len(recent_ep_rewards) > 1:
                            reward_var = float(np.var(recent_ep_rewards))
                            stability = 1.0 / (1.0 + reward_var)
                        else:
                            stability = 1.0  # Default if insufficient data

                        # Create EpisodeOutcome for Pareto analysis
                        episode_outcome = EpisodeOutcome(
                            env_id=env_idx,
                            episode_idx=episodes_completed + env_idx,
                            final_accuracy=env_state.val_acc,
                            param_ratio=(model.total_params - host_params_baseline)
                            / max(1, host_params_baseline),
                            num_fossilized=env_state.seeds_fossilized,
                            num_contributing_fossilized=env_state.contributing_fossilized,
                            episode_reward=env_total_rewards[env_idx],
                            stability_score=stability,
                            reward_mode=env_reward_configs[env_idx].reward_mode.value,
                        )
                        episode_outcomes.append(episode_outcome)

                        # Emit EPISODE_OUTCOME telemetry for Pareto analysis
                        # B11-CR-04 fix: Skip emission for rollback episodes (will emit corrected outcome later)
                        if (
                            env_state.telemetry_cb
                            and not env_rollback_occurred[env_idx]
                        ):
                            # TELE-610: Classify episode outcome
                            # SUCCESS_THRESHOLD is configurable; 0.8 = 80% accuracy considered "success"
                            SUCCESS_THRESHOLD = 0.8
                            if env_state.val_acc >= SUCCESS_THRESHOLD:
                                outcome_type = "success"
                            else:
                                outcome_type = "timeout"  # Fixed-length episodes that don't hit goal

                            env_state.telemetry_cb(
                                TelemetryEvent(
                                    event_type=TelemetryEventType.EPISODE_OUTCOME,
                                    epoch=episodes_completed + env_idx,
                                    data=EpisodeOutcomePayload(
                                        env_id=env_idx,
                                        episode_idx=episode_outcome.episode_idx,
                                        final_accuracy=episode_outcome.final_accuracy,
                                        param_ratio=episode_outcome.param_ratio,
                                        num_fossilized=episode_outcome.num_fossilized,
                                        num_contributing_fossilized=episode_outcome.num_contributing_fossilized,
                                        episode_reward=episode_outcome.episode_reward,
                                        stability_score=episode_outcome.stability_score,
                                        reward_mode=episode_outcome.reward_mode,
                                        # TELE-610: Episode diagnostics
                                        episode_length=epoch,  # Current epoch = episode length
                                        outcome_type=outcome_type,
                                        germinate_count=env_state.action_counts[
                                            "GERMINATE"
                                        ],
                                        prune_count=env_state.action_counts["PRUNE"],
                                        fossilize_count=env_state.action_counts[
                                            "FOSSILIZE"
                                        ],
                                    ),
                                )
                            )

                        # Shapley contributions at episode end
                        if (
                            env_state.counterfactual_helper is not None
                            and baseline_accs[env_idx]
                        ):
                            active_slot_ids = [
                                sid
                                for sid in slots
                                if model.has_active_seed_in_slot(sid)
                                and cast(SeedSlotProtocol, model.seed_slots[sid]).alpha
                                > 0
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
                                        return (
                                            env_state.val_loss * 1.1,
                                            cached_baselines[disabled[0]],
                                        )
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
                                        epoch=batch_idx + 1,
                                    )
                                except (KeyError, ZeroDivisionError, ValueError) as e:
                                    # HIGH-01 fix: Narrow to expected failures in Shapley computation
                                    _logger.warning(
                                        f"Shapley failed for env {env_idx}: {e}"
                                    )

                # PHASE 2: Compute all bootstrap values in single batched forward pass
                bootstrap_values: list[float] = []
                if all_post_action_signals:
                    # Unpack Obs V3 tuple (obs, blueprint_indices)
                    post_action_features_batch, post_action_bp_indices = (
                        batch_signals_to_features(
                            batch_signals=all_post_action_signals,
                            batch_slot_reports=all_post_action_slot_reports,
                            slot_config=slot_config,
                            env_states=env_states,
                            device=torch.device(device),
                            max_epochs=max_epochs,
                        )
                    )
                    post_action_features_normalized = obs_normalizer.normalize(
                        post_action_features_batch
                    )
                    post_masks_batch = {
                        k: torch.stack([m[k] for m in all_post_action_masks]).to(device)
                        for k in HEAD_NAMES
                    }
                    post_masks_batch["slot_by_op"] = torch.stack(
                        [m["slot_by_op"] for m in all_post_action_masks]
                    ).to(device)

                    with torch.inference_mode():
                        bootstrap_result = agent.policy.get_action(
                            post_action_features_normalized,
                            blueprint_indices=post_action_bp_indices,
                            masks=post_masks_batch,
                            hidden=batched_lstm_hidden,
                            deterministic=True,
                        )
                    # PERF: Move to CPU before .tolist() to avoid per-value GPU sync
                    bootstrap_values = bootstrap_result.value.cpu().tolist()

                if truncated_bootstrap_targets:
                    if not bootstrap_values:
                        raise RuntimeError(
                            "Missing bootstrap values for truncated transitions."
                        )
                    for (env_id, step_idx), bootstrap_val in zip(
                        truncated_bootstrap_targets, bootstrap_values, strict=True
                    ):
                        agent.buffer.bootstrap_values[env_id, step_idx] = bootstrap_val

                # Check for graceful shutdown at end of each epoch (not just batch end)
                # This gives user faster response (~seconds) instead of waiting for full batch
                if shutdown_event is not None and shutdown_event.is_set():
                    print(
                        f"\n[Shutdown requested] Stopping at epoch {epoch}/{max_epochs} "
                        f"(batch {batch_idx + 1}, {episodes_completed}/{total_env_episodes} episodes)"
                    )
                    break  # Exit epoch loop; batch-level break below will handle cleanup
                if prof is not None:
                    prof.step()
                    prof_steps += 1

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
                    # B1-DRL-01 fix: Inject death penalty so PPO learns to avoid
                    # catastrophic actions. Previously get_punishment_reward() was dead code.
                    # P1-NORM fix: Normalize penalty to match other rewards' scale.
                    # Use normalize_only to avoid polluting running stats with rare outliers.
                    penalty = env_states[env_idx].governor.get_punishment_reward()
                    normalized_penalty = reward_normalizer.normalize_only(penalty)
                    agent.buffer.mark_terminal_with_penalty(env_idx, normalized_penalty)
                    # B11-CR-03 fix: OVERWRITE last reward with RAW penalty (for telemetry interpretability).
                    # Buffer gets normalized_penalty (for PPO training stability).
                    # Telemetry gets raw penalty (for cross-run comparability).
                    if env_states[env_idx].episode_rewards:
                        env_states[env_idx].episode_rewards[-1] = penalty

                # B11-CR-02 fix: Recompute metrics after penalty injection
                # Metrics were computed in the epoch loop (lines 3173-3214) BEFORE penalty was applied.
                # This caused EpisodeOutcome, episode_history, and stability to reflect PRE-PENALTY
                # rewards, making rollback episodes appear ~2x more rewarding and ~1.6x more stable.
                if rollback_env_indices:
                    for env_idx in rollback_env_indices:
                        env_state = env_states[env_idx]

                        # 1. Recompute total reward from post-penalty episode_rewards
                        env_total_rewards[env_idx] = sum(env_state.episode_rewards)

                        # 2. Update episode_history entry for this env
                        for entry in reversed(episode_history):
                            if entry["env_id"] == env_idx:
                                entry["episode_reward"] = env_total_rewards[env_idx]
                                break

                        # 3. Recompute stability from post-penalty variance
                        recent_ep_rewards = (
                            env_state.episode_rewards[-20:]
                            if len(env_state.episode_rewards) >= 20
                            else env_state.episode_rewards
                        )
                        if len(recent_ep_rewards) > 1:
                            reward_var = float(np.var(recent_ep_rewards))
                            stability = 1.0 / (1.0 + reward_var)
                        else:
                            stability = 1.0

                        # 4. Find and replace EpisodeOutcome for this env
                        # EpisodeOutcome is frozen dataclass, use dataclasses.replace()
                        for i, outcome in enumerate(episode_outcomes):
                            if outcome.env_id == env_idx:
                                corrected_outcome = dataclasses.replace(
                                    outcome,
                                    episode_reward=env_total_rewards[env_idx],
                                    stability_score=stability,
                                )
                                episode_outcomes[i] = corrected_outcome

                                # 5. Emit corrected EPISODE_OUTCOME telemetry
                                # B11-CR-04 fix: First emission was suppressed for rollback episodes
                                # (see line 3213), so we emit the corrected outcome here (one event total).
                                if env_state.telemetry_cb:
                                    # TELE-610: Classify rollback episode outcome
                                    # Use same threshold as main path (line 3381)
                                    SUCCESS_THRESHOLD = 0.8
                                    if (
                                        corrected_outcome.final_accuracy
                                        >= SUCCESS_THRESHOLD
                                    ):
                                        rollback_outcome_type = "success"
                                    else:
                                        rollback_outcome_type = "timeout"

                                    env_state.telemetry_cb(
                                        TelemetryEvent(
                                            event_type=TelemetryEventType.EPISODE_OUTCOME,
                                            epoch=corrected_outcome.episode_idx,
                                            data=EpisodeOutcomePayload(
                                                env_id=env_idx,
                                                episode_idx=corrected_outcome.episode_idx,
                                                final_accuracy=corrected_outcome.final_accuracy,
                                                param_ratio=corrected_outcome.param_ratio,
                                                num_fossilized=corrected_outcome.num_fossilized,
                                                num_contributing_fossilized=corrected_outcome.num_contributing_fossilized,
                                                episode_reward=corrected_outcome.episode_reward,
                                                stability_score=corrected_outcome.stability_score,
                                                reward_mode=corrected_outcome.reward_mode,
                                                # TELE-610: Episode diagnostics (was missing for rollback path)
                                                episode_length=epoch,
                                                outcome_type=rollback_outcome_type,
                                                germinate_count=env_state.action_counts[
                                                    "GERMINATE"
                                                ],
                                                prune_count=env_state.action_counts[
                                                    "PRUNE"
                                                ],
                                                fossilize_count=env_state.action_counts[
                                                    "FOSSILIZE"
                                                ],
                                            ),
                                        )
                                    )
                                break

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
                ppo_grad_norm = compute_grad_norm_surrogate(agent.policy.network)

                # B7-DRL-01: Wire up gradient EMA tracking for drift detection
                # Compute gradient health heuristic from norm (vanishing/exploding detection)
                drift_metrics: dict[str, float] | None = None
                if grad_ema_tracker is not None:
                    if ppo_grad_norm < 1e-7:
                        grad_health = 0.3  # Vanishing gradients
                    elif ppo_grad_norm > 100.0:
                        grad_health = 0.3  # Exploding gradients
                    else:
                        grad_health = 1.0  # Healthy range
                    drift_metrics = grad_ema_tracker.update(ppo_grad_norm, grad_health)

                # FINITENESS GATE CONTRACT: Check if PPO update actually occurred
                if not metrics.get("ppo_update_performed", True):
                    # All epochs skipped due to non-finite values
                    skip_count = metrics.get("finiteness_gate_skip_count", 0)
                    consecutive_finiteness_failures += 1
                    _logger.warning(
                        f"PPO update skipped (all {skip_count} epochs hit finiteness gate). "
                        f"Consecutive failures: {consecutive_finiteness_failures}/3"
                    )

                    # Escalate after 3 consecutive failures (DRL best practice)
                    if consecutive_finiteness_failures >= 3:
                        raise RuntimeError(
                            f"PPO training failed: {consecutive_finiteness_failures} consecutive updates "
                            "skipped due to non-finite values. Check policy/value network outputs for NaN. "
                            f"Last failure: {metrics.get('finiteness_gate_failures', 'unknown')}"
                        )
                    # Skip anomaly detection for this batch - metrics are NaN
                    continue

                # Reset counter on successful update
                consecutive_finiteness_failures = 0

                metric_values = [
                    v for v in metrics.values() if isinstance(v, (int, float))
                ]
                anomaly_report = anomaly_detector.check_all(
                    # MANDATORY metrics after PPO update - fail loudly if missing
                    ratio_max=metrics["ratio_max"],
                    ratio_min=metrics["ratio_min"],
                    explained_variance=metrics.get(
                        "explained_variance", 0.0
                    ),  # Optional: computed once
                    has_nan=any(math.isnan(v) for v in metric_values),
                    has_inf=any(math.isinf(v) for v in metric_values),
                    current_episode=batch_epoch_id,
                    total_episodes=total_env_episodes,
                )

                # B7-DRL-01: Check gradient drift and merge into anomaly report
                if drift_metrics is not None:
                    drift_report = anomaly_detector.check_gradient_drift(
                        norm_drift=drift_metrics["norm_drift"],
                        health_drift=drift_metrics["health_drift"],
                    )
                    if drift_report.has_anomaly:
                        anomaly_report.has_anomaly = True
                        anomaly_report.anomaly_types.extend(drift_report.anomaly_types)
                        anomaly_report.details.update(drift_report.details)

                # B7-DRL-04: Check LSTM hidden state health after PPO update
                # LSTM hidden states can become corrupted during BPTT - monitor for
                # explosion/saturation (RMS > threshold), vanishing (RMS < 1e-6), or NaN/Inf.
                lstm_health = compute_lstm_health(batched_lstm_hidden)
                if lstm_health is not None:
                    lstm_report = anomaly_detector.check_lstm_health(
                        h_rms=lstm_health.h_rms,
                        c_rms=lstm_health.c_rms,
                        h_env_rms_max=lstm_health.h_env_rms_max,
                        c_env_rms_max=lstm_health.c_env_rms_max,
                        has_nan=lstm_health.has_nan,
                        has_inf=lstm_health.has_inf,
                    )
                    if lstm_report.has_anomaly:
                        anomaly_report.has_anomaly = True
                        anomaly_report.anomaly_types.extend(lstm_report.anomaly_types)
                        anomaly_report.details.update(lstm_report.details)
                    # Add LSTM health to metrics for telemetry display in Sanctum
                    metrics.update(lstm_health.to_dict())

                _handle_telemetry_escalation(anomaly_report, telemetry_config)
                _emit_anomaly_diagnostics(
                    hub,
                    anomaly_report,
                    agent,
                    batch_epoch_id,
                    batch_idx,
                    max_epochs,
                    total_env_episodes,
                    False,
                    group_id=group_id,
                )

            # If the epoch loop exited early (e.g. graceful shutdown), ensure the batch
            # summary reflects the partial episode outcomes instead of the default zeros.
            if epoch < max_epochs:
                for env_idx, env_state in enumerate(env_states):
                    env_final_accs[env_idx] = env_state.val_acc
                    env_total_rewards[env_idx] = sum(env_state.episode_rewards)

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
                    # Assert non-None: values assigned in same `if not update_skipped` block above
                    assert ppo_grad_norm is not None and ppo_update_time_ms is not None
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

                batch_val_losses = [es.val_loss for es in env_states]
                batch_val_corrects = val_corrects
                batch_val_totals = val_totals

                batch_emitter.on_batch_completed(
                    batch_idx=batch_idx,
                    episodes_completed=batch_epoch_id,
                    rolling_avg_acc=rolling_avg_acc,
                    avg_acc=avg_acc,
                    metrics=metrics,
                    env_states=env_states,
                    update_skipped=update_skipped,
                    plateau_threshold=plateau_threshold,
                    improvement_threshold=improvement_threshold,
                    prev_rolling_avg_acc=prev_rolling_avg_acc,
                    total_episodes=total_env_episodes,
                    start_episode=start_episode,
                    n_episodes=total_env_episodes,
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

                # B7-DRL-02: Check for performance degradation (was previously unwired)
                # Detects catastrophic forgetting, reward hacking, and training decay
                training_progress = batch_epoch_id / total_env_episodes
                check_performance_degradation(
                    hub,
                    current_acc=avg_acc,
                    rolling_avg_acc=rolling_avg_acc,
                    env_id=0,  # Aggregate metric across all envs
                    training_progress=training_progress,
                )

            history.append(
                {
                    "batch": batch_idx + 1,
                    "episodes": batch_epoch_id,
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

            episodes_completed = batch_epoch_id
            batch_idx += 1

            # Check for graceful shutdown request (e.g., user quit TUI)
            # Per-epoch check already printed progress; just break here
            if shutdown_event is not None and shutdown_event.is_set():
                break

    finally:
        # Ensure profiler context is always closed, even on exceptions
        profiler_cm.__exit__(None, None, None)
        if torch_profiler_summary and prof is not None:
            print("\n=== torch.profiler: CUDA time (top 30) ===")
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
            print("\n=== torch.profiler: CPU time (top 30) ===")
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))
        if torch_profiler:
            min_steps_for_trace = (
                torch_profiler_wait + torch_profiler_warmup + torch_profiler_active
            )
            if prof_steps < min_steps_for_trace:
                print(
                    f"\n[torch.profiler] No trace captured (ran {prof_steps} steps; "
                    f"need >= {min_steps_for_trace} for wait={torch_profiler_wait} "
                    f"warmup={torch_profiler_warmup} active={torch_profiler_active}). "
                    "Run longer or reduce --torch-profiler-wait/--torch-profiler-warmup."
                )

    if best_state:
        agent.policy.load_state_dict(best_state)

    if save_path:
        # B5-PT-02 FIX: Save normalizer state for correct training resume.
        # Resume expects these keys in metadata.
        checkpoint_metadata = {
            # Observation normalizer (RunningMeanStd)
            "obs_normalizer_mean": obs_normalizer.mean.tolist(),
            "obs_normalizer_var": obs_normalizer.var.tolist(),
            "obs_normalizer_count": obs_normalizer.count.item(),
            "obs_normalizer_momentum": obs_normalizer.momentum,
            # Reward normalizer (RewardNormalizer)
            "reward_normalizer_mean": reward_normalizer.mean,
            "reward_normalizer_m2": reward_normalizer.m2,
            "reward_normalizer_count": reward_normalizer.count,
            # Resume counters
            "batches_completed": batch_idx,
            "n_envs": n_envs,
        }
        agent.save(save_path, metadata=checkpoint_metadata)

    return agent, history


def train_ppo_vectorized(
    n_episodes: int = 100,
    n_envs: int = DEFAULT_N_ENVS,
    max_epochs: int = DEFAULT_EPISODE_LENGTH,
    device: str = "cuda:0",
    devices: list[str] | None = None,
    task: str = "cifar_baseline",
    use_telemetry: bool = True,
    lr: float = DEFAULT_LEARNING_RATE,
    clip_ratio: float = DEFAULT_CLIP_RATIO,
    entropy_coef: float = DEFAULT_ENTROPY_COEF,  # From leyline
    entropy_coef_start: float | None = None,
    entropy_coef_end: float | None = None,
    entropy_coef_min: float = DEFAULT_ENTROPY_COEF_MIN,  # From leyline
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
    experimental_gpu_preload_gather: bool = False,
    gpu_preload_augment: bool = False,
    gpu_preload_precompute_augment: bool = False,
    amp: bool = False,
    amp_dtype: str = "auto",  # "auto", "float16", "bfloat16", or "off"
    max_grad_norm: float | None = None,  # Gradient clipping max norm (None disables)
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
    rent_host_params_floor: int = 200,
    reward_family: str = "contribution",
    permissive_gates: bool = True,
    auto_forward_g1: bool = False,
    auto_forward_g2: bool = False,
    auto_forward_g3: bool = False,
    # Ablation flags for systematic reward function experiments
    disable_pbrs: bool = False,  # Disable PBRS stage advancement shaping
    disable_terminal_reward: bool = False,  # Disable terminal accuracy bonus
    disable_anti_gaming: bool = False,  # Disable ratio_penalty and alpha_shock
    quiet_analytics: bool = False,
    force_compile: bool = False,
    telemetry_dir: str | None = None,
    ready_event: "threading.Event | None" = None,
    shutdown_event: "threading.Event | None" = None,
    group_id: str = "default",  # A/B testing group identifier
    torch_profiler: bool = False,
    torch_profiler_dir: str = "./profiler_traces",
    torch_profiler_wait: int = 1,
    torch_profiler_warmup: int = 1,
    torch_profiler_active: int = 3,
    torch_profiler_repeat: int = 1,
    torch_profiler_record_shapes: bool = False,
    torch_profiler_profile_memory: bool = False,
    torch_profiler_with_stack: bool = False,
    torch_profiler_summary: bool = False,
) -> tuple[PPOAgent, list[dict[str, Any]]]:
    """Train PPO with vectorized environments using INVERTED CONTROL FLOW."""
    trainer = VectorizedPPOTrainer(_train_ppo_vectorized_impl)
    return trainer.run(
        n_episodes,
        n_envs,
        max_epochs,
        device,
        devices,
        task,
        use_telemetry,
        lr,
        clip_ratio,
        entropy_coef,
        entropy_coef_start,
        entropy_coef_end,
        entropy_coef_min,
        entropy_anneal_episodes,
        gamma,
        gae_lambda,
        ppo_updates_per_batch,
        save_path,
        resume_path,
        seed,
        num_workers,
        batch_size_per_env,
        gpu_preload,
        experimental_gpu_preload_gather,
        gpu_preload_augment,
        gpu_preload_precompute_augment,
        amp,
        amp_dtype,
        max_grad_norm,
        compile_mode,
        lstm_hidden_dim,
        chunk_length,
        telemetry_config,
        telemetry_lifecycle_only,
        plateau_threshold,
        improvement_threshold,
        gradient_telemetry_stride,
        slots,
        max_seeds,
        reward_mode,
        param_budget,
        param_penalty_weight,
        sparse_reward_scale,
        rent_host_params_floor,
        reward_family,
        permissive_gates,
        auto_forward_g1,
        auto_forward_g2,
        auto_forward_g3,
        disable_pbrs,
        disable_terminal_reward,
        disable_anti_gaming,
        quiet_analytics,
        force_compile,
        telemetry_dir,
        ready_event,
        shutdown_event,
        group_id,
        torch_profiler,
        torch_profiler_dir,
        torch_profiler_wait,
        torch_profiler_warmup,
        torch_profiler_active,
        torch_profiler_repeat,
        torch_profiler_record_shapes,
        torch_profiler_profile_memory,
        torch_profiler_with_stack,
        torch_profiler_summary,
    )


__all__ = ["ParallelEnvState", "train_ppo_vectorized"]
