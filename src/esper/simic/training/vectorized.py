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
import threading
from typing import Any, cast

import torch
import torch.amp as torch_amp
import torch.nn as nn

# NOTE: get_task_spec imported lazily inside train_ppo_vectorized to avoid circular import:
#   runtime -> simic.rewards -> simic -> simic.training -> vectorized -> runtime
from esper.utils.data import SharedBatchIterator
from esper.leyline import (
    CheckpointLoadedPayload,
    DEFAULT_BATCH_SIZE_TRAINING,
    DEFAULT_CLIP_RATIO,
    DEFAULT_ENTROPY_COEF,
    DEFAULT_ENTROPY_COEF_MIN,
    DEFAULT_EPISODE_LENGTH,
    DEFAULT_GAE_LAMBDA,
    DEFAULT_GAMMA,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LSTM_HIDDEN_DIM,
    DEFAULT_N_ENVS,
    GateLevel,
    OP_NAMES,
    OP_PRUNE,
    SeedSlotProtocol,
    SeedStage,
    SlotConfig,
    TelemetryEvent,
    TelemetryEventType,
    TrainingStartedPayload,
)
from esper.simic.telemetry import (
    AnomalyDetector,
    AnomalyReport,
    TelemetryConfig,
    check_numerical_stability,
    collect_per_layer_gradients,
)
from esper.simic.control import RunningMeanStd, RewardNormalizer
from esper.tamiyo.policy.features import get_feature_size
from esper.simic.agent import PPOAgent
from esper.simic.agent.types import PPOUpdateMetrics
from esper.tamiyo.policy import create_policy
from esper.simic.rewards import (
    ContributionRewardConfig,
    RewardFamily,
    RewardMode,
)
from esper.nissa import get_hub, BlueprintAnalytics, DirectoryOutput
from esper.simic.telemetry.emitters import VectorizedEmitter
from .env_factory import EnvFactoryContext
from .parallel_env_state import ParallelEnvState
from .vectorized_trainer import VectorizedPPOTrainer

_logger = logging.getLogger(__name__)


# =============================================================================
# Seed Management Helpers
# =============================================================================


def _fossilize_active_seed(model: Any, slot_id: str) -> bool:
    """Fossilize the active seed in the specified slot (HOLDING only).

    Args:
        model: MorphogeneticModel instance
        slot_id: Target slot ID (e.g., "r0c0", "r0c1", "r0c2")

    Returns:
        True if the lifecycle transition succeeds, False otherwise.
    """
    if not model.has_active_seed_in_slot(slot_id):
        return False

    slot = cast(SeedSlotProtocol, model.seed_slots[slot_id])
    seed_state = slot.state
    if seed_state is None:
        return False
    current_stage = seed_state.stage

    if current_stage != SeedStage.HOLDING:
        return False

    gate_result = slot.advance_stage(SeedStage.FOSSILIZED)
    if gate_result.passed:
        slot.set_alpha(1.0)
        return True
    # Gate check failure is normal; reward shaping will penalize
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


def _calculate_value_warmup_steps(
    value_warmup_batches: int,
    ppo_updates_per_batch: int,
) -> int:
    """Convert batch-based value warmup to PPO update steps.

    Value warmup starts the critic with a low coefficient (e.g., 0.1 * target) and
    ramps up over the warmup period. This prevents critic collapse when early
    returns have low variance (before policy discovers high-value strategies).

    Expressed in batches (not episodes) so it scales correctly with n_envs.
    """
    if value_warmup_batches <= 0:
        return 0
    return value_warmup_batches * max(1, ppo_updates_per_batch)


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
        elif isinstance(values[0], (int, float)):
            aggregated[key] = sum(values) / len(values)
        else:
            aggregated[key] = values[0]
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
        gradient_stats = collect_per_layer_gradients(agent.policy.network)
        stability_report = check_numerical_stability(agent.policy.network)

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
    task: str = "cifar_baseline",
    use_telemetry: bool = True,
    lr: float = DEFAULT_LEARNING_RATE,
    clip_ratio: float = DEFAULT_CLIP_RATIO,
    entropy_coef: float = DEFAULT_ENTROPY_COEF,  # From leyline
    entropy_coef_start: float | None = None,
    entropy_coef_end: float | None = None,
    entropy_coef_min: float = DEFAULT_ENTROPY_COEF_MIN,  # From leyline
    entropy_anneal_episodes: int = 0,
    entropy_coef_per_head: dict[str, float] | None = None,  # Per-head multipliers
    value_coef: float = 0.5,  # Value loss coefficient (lower reduces critic dominance)
    value_warmup_batches: int = 0,  # Batches to ramp up value_coef (0 = no warmup)
    value_coef_start: float | None = None,  # Starting value_coef (default: 0.1 * value_coef)
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
    chunk_length: int | None = None,  # None = auto-match max_epochs
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
    basic_acc_delta_weight: float = 5.0,
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
    ready_event: threading.Event | None = None,
    shutdown_event: threading.Event | None = None,
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
    from esper.tamiyo import SignalTracker

    if not slots:
        raise ValueError("slots parameter is required and cannot be empty")
    if chunk_length is None:
        chunk_length = max_epochs
    if chunk_length != max_epochs:
        raise ValueError(
            "chunk_length must match max_epochs for the current training loop"
        )

    auto_forward_gates: frozenset[GateLevel] = frozenset(
        gate
        for gate, enabled in (
            (GateLevel.G1, auto_forward_g1),
            (GateLevel.G2, auto_forward_g2),
            (GateLevel.G3, auto_forward_g3),
        )
        if enabled
    )
    # Manual ADVANCE is only disabled when all auto-forward gates are enabled.
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
        basic_acc_delta_weight=basic_acc_delta_weight,
        disable_pbrs=disable_pbrs,
        disable_terminal_reward=disable_terminal_reward,
        disable_anti_gaming=disable_anti_gaming,
    )
    loss_reward_config = task_spec.loss_reward_config

    # Per-environment reward configs (single reward mode per run).
    env_reward_configs = [
        dataclasses.replace(reward_config) for _ in range(n_envs)
    ]

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

    # Convert batch-based value warmup to step-based
    value_warmup_steps = _calculate_value_warmup_steps(
        value_warmup_batches=value_warmup_batches,
        ppo_updates_per_batch=ppo_updates_per_batch,
    )

    # Create per-environment emitters for consolidated telemetry logic
    emitters = [
        VectorizedEmitter(
            env_id=i,
            device=env_device_map[i],
            group_id=group_id,
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
            entropy_coef_per_head=entropy_coef_per_head,
            value_coef=value_coef,
            value_coef_start=value_coef_start,
            value_warmup_steps=value_warmup_steps,
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

    env_factory = EnvFactoryContext(
        env_device_map=env_device_map,
        create_model=create_model,
        task_spec=task_spec,
        slots=slots,
        permissive_gates=permissive_gates,
        auto_forward_gates=auto_forward_gates,
        analytics=analytics,
        action_enum=ActionEnum,
        use_telemetry=use_telemetry,
        amp_enabled=amp_enabled,
        resolved_amp_dtype=resolved_amp_dtype,
        use_grad_scaler=use_grad_scaler,
        gpu_preload_augment=gpu_preload_augment,
        ops_telemetry_enabled=ops_telemetry_enabled,
        telemetry_lifecycle_only=telemetry_lifecycle_only,
        hub=hub,
        signal_tracker_cls=SignalTracker,
        group_id=group_id,
    )

    trainer = VectorizedPPOTrainer(
        agent=agent,
        task_spec=task_spec,
        slots=slots,
        slot_config=slot_config,
        n_envs=n_envs,
        max_epochs=max_epochs,
        ppo_updates_per_batch=ppo_updates_per_batch,
        total_batches=total_batches,
        total_env_episodes=total_env_episodes,
        start_episode=start_episode,
        start_batch=start_batch,
        save_path=save_path,
        seed=seed,
        env_device_map=env_device_map,
        shared_train_iter=shared_train_iter,
        shared_test_iter=shared_test_iter,
        num_train_batches=num_train_batches,
        num_test_batches=num_test_batches,
        env_reward_configs=env_reward_configs,
        reward_family_enum=reward_family_enum,
        reward_config=reward_config,
        loss_reward_config=loss_reward_config,
        reward_normalizer=reward_normalizer,
        obs_normalizer=obs_normalizer,
        initial_obs_normalizer_mean=initial_obs_normalizer_mean,
        telemetry_config=telemetry_config,
        telemetry_lifecycle_only=telemetry_lifecycle_only,
        ops_telemetry_enabled=ops_telemetry_enabled,
        use_telemetry=use_telemetry,
        gradient_telemetry_stride=gradient_telemetry_stride,
        max_grad_norm=max_grad_norm,
        plateau_threshold=plateau_threshold,
        improvement_threshold=improvement_threshold,
        anomaly_detector=anomaly_detector,
        hub=hub,
        analytics=analytics,
        emitters=emitters,
        batch_emitter=batch_emitter,
        shutdown_event=shutdown_event,
        group_id=group_id,
        torch_profiler=torch_profiler,
        torch_profiler_dir=torch_profiler_dir,
        torch_profiler_wait=torch_profiler_wait,
        torch_profiler_warmup=torch_profiler_warmup,
        torch_profiler_active=torch_profiler_active,
        torch_profiler_repeat=torch_profiler_repeat,
        torch_profiler_record_shapes=torch_profiler_record_shapes,
        torch_profiler_profile_memory=torch_profiler_profile_memory,
        torch_profiler_with_stack=torch_profiler_with_stack,
        torch_profiler_summary=torch_profiler_summary,
        gpu_preload_augment=gpu_preload_augment,
        amp_enabled=amp_enabled,
        resolved_amp_dtype=resolved_amp_dtype,
        env_factory=env_factory,
        compiled_loss_and_correct=_compiled_loss_and_correct,
        run_ppo_updates=_run_ppo_updates,
        aggregate_ppo_metrics=_aggregate_ppo_metrics,
        handle_telemetry_escalation=_handle_telemetry_escalation,
        emit_anomaly_diagnostics=_emit_anomaly_diagnostics,
        fossilize_active_seed=_fossilize_active_seed,
        resolve_target_slot=_resolve_target_slot,
        host_params_baseline=host_params_baseline,
        disable_advance=disable_advance,
        effective_max_seeds=effective_max_seeds,
        device=device,
        logger=_logger,
    )

    history = trainer.run()

    return agent, history


__all__ = [ParallelEnvState.__name__, "train_ppo_vectorized", "OP_NAMES", "OP_PRUNE"]
