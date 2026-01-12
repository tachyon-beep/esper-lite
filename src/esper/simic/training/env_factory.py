from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, cast

import torch
import torch.amp as torch_amp

from esper.karn.health import HealthMonitor
from esper.kasmina.host import MorphogeneticModel
from esper.utils.data import AugmentationBuffers
from esper.leyline import (
    DEFAULT_GOVERNOR_ABSOLUTE_THRESHOLD,
    DEFAULT_GOVERNOR_DEATH_PENALTY,
    DEFAULT_GOVERNOR_HISTORY_WINDOW,
    DEFAULT_GOVERNOR_SENSITIVITY,
    DEFAULT_MIN_PANICS_BEFORE_ROLLBACK,
    GateLevel,
    SeedSlotProtocol,
    TelemetryEvent,
)
from esper.nissa import BlueprintAnalytics
from esper.simic.attribution import CounterfactualHelper
from esper.simic.telemetry.emitters import apply_slot_telemetry, emit_with_env_context
from esper.tolaria import TolariaGovernor

from .parallel_env_state import ParallelEnvState

if TYPE_CHECKING:
    from esper.tamiyo.tracker import SignalTracker
    from esper.runtime.tasks import TaskSpec


class EpisodeContext:
    """Mutable holder for episode-level context needed by telemetry callbacks.

    Callbacks are created early (before env_state exists) but need access to
    episode_idx which is set later. This holder bridges that gap.
    """

    __slots__ = ("episode_idx",)

    def __init__(self) -> None:
        self.episode_idx: int | None = None


def make_telemetry_callback(
    env_idx: int,
    device: str,
    hub: Any,
    group_id: str,
    episode_context: EpisodeContext | None = None,
) -> Callable[[TelemetryEvent], None]:
    """Create a telemetry callback that injects env_id, device, group_id, and episode_idx."""
    if not hub:
        return lambda _: None

    def callback(event: TelemetryEvent) -> None:
        episode_idx = episode_context.episode_idx if episode_context else None
        emit_with_env_context(hub, env_idx, device, event, group_id, episode_idx=episode_idx)

    return callback


def configure_slot_telemetry(
    env_state: ParallelEnvState,
    *,
    ops_telemetry_enabled: bool,
    telemetry_lifecycle_only: bool,
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


@dataclass(frozen=True)
class EnvFactoryContext:
    env_device_map: list[str]
    create_model: Callable[..., Any]
    task_spec: TaskSpec
    slots: list[str]
    permissive_gates: bool
    auto_forward_gates: frozenset[GateLevel]
    analytics: BlueprintAnalytics
    action_enum: Any
    use_telemetry: bool
    amp_enabled: bool
    resolved_amp_dtype: torch.dtype | None
    use_grad_scaler: bool
    gpu_preload_augment: bool
    ops_telemetry_enabled: bool
    telemetry_lifecycle_only: bool
    hub: Any
    signal_tracker_cls: type[SignalTracker]
    group_id: str


def create_env_state(
    env_idx: int,
    base_seed: int,
    context: EnvFactoryContext,
) -> ParallelEnvState:
    """Create environment state with CUDA stream.

    DataLoaders are now shared via SharedBatchIterator, not per-env.
    """
    env_device = context.env_device_map[env_idx]
    torch.manual_seed(base_seed + env_idx * 1000)
    random.seed(base_seed + env_idx * 1000)

    model_raw = context.create_model(
        task=context.task_spec,
        device=env_device,
        slots=context.slots,
        permissive_gates=context.permissive_gates,
    )
    # Type assertion: create_model returns MorphogeneticModel
    assert isinstance(model_raw, MorphogeneticModel)
    model: MorphogeneticModel = model_raw

    # Convert host model to channels-last for optimal conv performance.
    # This matches the channels-last data format from SharedGPUGatherBatchIterator,
    # avoiding runtime layout permutations in conv layers.
    model = model.to(memory_format=torch.channels_last)

    # Create episode context for telemetry (training loop updates episode_idx at episode start)
    episode_context = EpisodeContext() if context.use_telemetry else None
    telemetry_cb = make_telemetry_callback(
        env_idx, env_device, context.hub, context.group_id, episode_context
    )
    for slot_module in model.seed_slots.values():
        slot = cast(SeedSlotProtocol, slot_module)
        slot.on_telemetry = telemetry_cb
        # fast_mode toggled per epoch via apply_slot_telemetry (telemetry-enabled by default)
        slot.fast_mode = False
        slot.auto_forward_gates = context.auto_forward_gates
        # Incubator mode gradient isolation: detach host input into the seed path so
        # host gradients remain identical to the host-only model while the seed
        # trickle-learns via STE in TRAINING. The host optimizer still steps
        # every batch; isolation only affects gradients through the seed branch.
        slot.isolate_gradients = True

    # Set host_params baseline for scoreboard via Nissa analytics
    host_params = sum(
        p.numel() for p in model.get_host_parameters() if p.requires_grad
    )
    context.analytics.set_host_params(env_idx, host_params)

    host_optimizer = torch.optim.SGD(
        model.get_host_parameters(),
        lr=context.task_spec.host_lr,
        momentum=0.9,
        weight_decay=5e-4,
    )

    # Create CUDA stream for this environment
    env_device_obj = torch.device(env_device)
    stream = (
        torch.cuda.Stream(device=env_device_obj)  # type: ignore[no-untyped-call]
        if env_device_obj.type == "cuda"
        else None
    )

    augment_generator = None
    augment_buffers = None
    if context.gpu_preload_augment:
        augment_generator = torch.Generator(device=env_device)
        augment_generator.manual_seed(base_seed + env_idx * 1009)
        # Pre-allocate buffers to reduce memory fragmentation during augmentation.
        # The ensure_capacity() method handles resizing if batch size changes.
        augment_buffers = AugmentationBuffers(device=env_device)

    # Per-env AMP scaler to avoid stream race conditions (GradScaler state is not stream-safe)
    # Use new torch.amp.GradScaler API (torch.cuda.amp.GradScaler deprecated in PyTorch 2.4+)
    # Note: BF16 doesn't need GradScaler (same exponent range as FP32)
    env_scaler = (
        torch_amp.GradScaler("cuda", enabled=context.use_grad_scaler)  # type: ignore[attr-defined]
        if env_device_obj.type == "cuda" and context.use_grad_scaler
        else None
    )

    # Pre-compute autocast decision for hot path (avoids per-batch device type checks)
    autocast_enabled = context.amp_enabled and env_device_obj.type == "cuda"

    # Determine random guess loss for lobotomy detection
    random_guess_loss = None
    if (
        context.task_spec.task_type == "classification"
        and context.task_spec.num_classes
    ):
        random_guess_loss = math.log(context.task_spec.num_classes)
    elif context.task_spec.task_type == "lm" and context.task_spec.vocab_size:
        random_guess_loss = math.log(context.task_spec.vocab_size)

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
        if context.use_telemetry
        else None
    )

    # Create CounterfactualHelper for Shapley value analysis at episode end
    # Use base_seed for reproducible Shapley permutation sampling (B5-CR-01)
    # Pass telemetry_cb for per-env context (same pattern as HealthMonitor)
    counterfactual_helper = (
        CounterfactualHelper(
            strategy="auto",  # Full factorial for <=4 slots, Shapley sampling otherwise
            shapley_samples=20,
            emit_callback=telemetry_cb,  # Pre-wired with env context
            seed=base_seed,
        )
        if context.use_telemetry
        else None
    )

    env_state = ParallelEnvState(
        model=model,
        host_optimizer=host_optimizer,
        signal_tracker=context.signal_tracker_cls(env_id=env_idx),
        governor=governor,
        health_monitor=health_monitor,
        counterfactual_helper=counterfactual_helper,
        env_device=env_device,
        stream=stream,
        augment_generator=augment_generator,
        augment_buffers=augment_buffers,
        scaler=env_scaler,
        seeds_created=0,
        episode_context=episode_context,
        episode_rewards=[],
        action_enum=context.action_enum,
        telemetry_cb=telemetry_cb,
        autocast_enabled=autocast_enabled,
    )
    env_state.prev_slot_alphas = {slot_id: 0.0 for slot_id in context.slots}
    env_state.prev_slot_params = {slot_id: 0 for slot_id in context.slots}
    # Pre-allocate accumulators to avoid per-epoch allocation churn
    env_state.init_accumulators(context.slots)
    configure_slot_telemetry(
        env_state,
        ops_telemetry_enabled=context.ops_telemetry_enabled,
        telemetry_lifecycle_only=context.telemetry_lifecycle_only,
    )
    return env_state
