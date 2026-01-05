"""Training loops for PPO.

This module contains the main training functions extracted from ppo.py.
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, Any, Callable, Iterator, Protocol, cast

import torch
import torch.nn as nn

from esper.leyline import (
    AnalyticsSnapshotPayload,
    EpochCompletedPayload,
    FactoredAction,
    GERMINATE_PREFIX,
    LifecycleOp,
    SeedStage,
    TelemetryEvent,
    TelemetryEventType,
    TrainingStartedPayload,
    is_germinate_action_name,
)

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from esper.runtime import TaskSpec
    from esper.tamiyo.heuristic import HeuristicTamiyo
# NOTE: get_task_spec imported lazily inside functions to avoid circular import:
#   runtime -> simic.rewards -> simic -> simic.training -> helpers -> runtime
from esper.simic.rewards import compute_contribution_reward, ContributionRewardConfig, SeedInfo
from esper.simic.telemetry import (
    collect_seed_gradients_async,
    materialize_grad_stats,
    TelemetryConfig,
)
from esper.simic.telemetry.gradient_collector import GradientHealthStats
from esper.leyline import SlottedHostProtocol
from esper.leyline.slot_config import SlotConfig
from esper.leyline.slot_id import validate_slot_ids
from esper.nissa import get_hub
from esper.utils.loss import compute_task_loss_with_metrics

logger = logging.getLogger(__name__)


class _HasSeedParameters(Protocol):
    """Protocol for models that have seed parameters (e.g., HostModel)."""

    def get_seed_parameters(self, slot: str | None = None) -> Iterator[torch.nn.Parameter]:
        """Yield seed parameters for gradient collection."""
        ...


def compute_rent_and_shock_inputs(
    *,
    model: SlottedHostProtocol,
    slot_ids: list[str],
    host_params: int,
    host_params_floor: int = 1,
    base_slot_rent_ratio: float,
    prev_slot_alphas: dict[str, float],
    prev_slot_params: dict[str, int],
) -> tuple[float, float]:
    """Compute effective params and convex alpha-shock inputs for reward shaping.

    Phase 5 contract:
    - BaseSlotRent applies only from BLENDING onward (cooldown/training pays no rent).
    - Param counts are invariant to BLEND_OUT freeze (requires_grad toggles).
    - Gate network params (alpha_schedule) count toward overhead when present.

    Updates prev_slot_alphas/prev_slot_params in place.
    """
    base_slot_rent_params = base_slot_rent_ratio * host_params if host_params > 0 else 0.0
    if host_params > 0 and host_params_floor < 1:
        raise ValueError(f"host_params_floor must be >= 1 (got {host_params_floor})")
    denom_host_params = max(host_params, host_params_floor) if host_params > 0 else 0

    effective_seed_params = 0.0
    alpha_delta_sq_sum = 0.0

    for slot_id in slot_ids:
        slot = model.seed_slots[slot_id]
        has_active_seed = model.has_active_seed_in_slot(slot_id)
        current_alpha = slot.alpha if has_active_seed else 0.0

        slot_param_count = 0
        if has_active_seed and slot.state is not None:
            # Use cached counts to avoid per-step parameter iteration and to be
            # invariant to BLEND_OUT requires_grad freezing.
            # seed_param_count is a required field on SeedMetrics (slot.py line 148)
            slot_param_count = int(slot.state.metrics.seed_param_count)
            if slot_param_count <= 0 and slot.seed is not None:
                slot_param_count = sum(p.numel() for p in slot.seed.parameters())

            # Gate network params (GATE alpha_algorithm) must also count as overhead.
            if slot.alpha_schedule is not None:
                slot_param_count += sum(p.numel() for p in slot.alpha_schedule.parameters())

            stage = slot.state.stage
            if stage in (SeedStage.BLENDING, SeedStage.HOLDING, SeedStage.FOSSILIZED):
                effective_seed_params += base_slot_rent_params
                effective_seed_params += current_alpha * slot_param_count

        prev_alpha = prev_slot_alphas[slot_id]
        prev_params = prev_slot_params[slot_id]
        if host_params > 0 and prev_params > 0:
            delta = current_alpha - prev_alpha
            alpha_delta_sq_sum += (delta * delta) * (prev_params / denom_host_params)

        prev_slot_alphas[slot_id] = current_alpha
        prev_slot_params[slot_id] = slot_param_count if has_active_seed else 0

    return effective_seed_params, alpha_delta_sq_sum


# =============================================================================
# Compiled Training Step
# =============================================================================


def _train_step_impl(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inner training step - forward pass and loss computation.

    This is the compilable core that can be optimized with torch.compile.
    Control flow (optimizer steps, seed handling) stays OUTSIDE this function.

    Args:
        model: The model to train
        inputs: Input batch tensor
        targets: Target batch tensor
        criterion: Loss function (CrossEntropyLoss)

    Returns:
        Tuple of (loss tensor, output logits)
    """
    outputs = model(inputs)
    # Reshape for criterion if needed (handles both LM and classification)
    if outputs.dim() == 3:  # LM: (batch, seq, vocab)
        vocab = outputs.size(-1)
        # Use reshape instead of view - handles non-contiguous tensors safely
        loss = criterion(outputs.reshape(-1, vocab), targets.reshape(-1))
    else:  # Classification: (batch, classes)
        loss = criterion(outputs, targets)
    return loss, outputs


# Module-level cache for successful compilation only.
# Failures are NOT cached, allowing retry on subsequent calls.
_compiled_train_step_cache: Callable[
    [nn.Module, torch.Tensor, torch.Tensor, nn.Module],
    tuple[torch.Tensor, torch.Tensor]
] | None = None


def _get_compiled_train_step(use_compile: bool = True) -> Callable[
    [nn.Module, torch.Tensor, torch.Tensor, nn.Module],
    tuple[torch.Tensor, torch.Tensor]
]:
    """Get train step function, optionally compiled.

    Only caches SUCCESSFUL compilations. If compilation fails (e.g., due to
    transient GPU memory pressure), subsequent calls will retry compilation
    instead of being stuck with the uncompiled fallback for the process lifetime.

    Note: Uses mode="default" instead of "reduce-overhead" because the model
    parameter varies across calls. CUDA graphs (reduce-overhead) capture memory
    addresses, so different model instances would cause repeated graph recapture.

    Thread-safety: Worst-case race is two threads both compiling successfully;
    one wins, no correctness issue. Failure paths are not cached.

    Args:
        use_compile: If True, attempt to compile; if False, use uncompiled version

    Returns:
        The train step function (compiled or uncompiled)
    """
    global _compiled_train_step_cache

    if not use_compile:
        return _train_step_impl

    # Return cached compiled version if available
    if _compiled_train_step_cache is not None:
        return _compiled_train_step_cache

    try:
        # M22: dynamic=True handles varying batch sizes without recompilation.
        # mode="default" is safest (reduce-overhead uses CUDA graphs which
        # capture memory addresses and break with varying model instances).
        compiled = torch.compile(_train_step_impl, mode="default", dynamic=True)
        _compiled_train_step_cache = compiled  # Cache only on success
        return compiled
    except Exception as e:
        # Log but DON'T cache - next call will retry compilation
        logger.warning(
            "torch.compile failed (will retry on next call), "
            "falling back to uncompiled train_step: %s",
            e,
        )
        return _train_step_impl


def compiled_train_step(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    use_compile: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Training step with optional torch.compile optimization.

    Uses lazy-initialized compiled version if use_compile=True and compilation
    succeeds, otherwise falls back to uncompiled implementation.

    Args:
        model: The model to train
        inputs: Input batch tensor
        targets: Target batch tensor
        criterion: Loss function (CrossEntropyLoss)
        use_compile: Whether to attempt compilation (default True)

    Returns:
        Tuple of (loss tensor, output logits)
    """
    fn = _get_compiled_train_step(use_compile)
    return fn(model, inputs, targets, criterion)


# =============================================================================
# PPO helpers
# =============================================================================


def _train_one_epoch(
    model: nn.Module,
    trainloader: "DataLoader[tuple[torch.Tensor, torch.Tensor]]",
    criterion: nn.Module,
    host_optimizer: torch.optim.Optimizer,
    seed_optimizer: torch.optim.Optimizer | None,
    device: str,
    task_type: str,
    collect_gradients: bool = False,
) -> tuple[float, float, int, GradientHealthStats | None]:
    """Unified training loop for all seed stages.

    This function extracts the repeated inline loop pattern. Callers use
    returned values to compute metrics:
        train_loss = running_loss / len(trainloader)
        train_acc = 100.0 * correct / total

    Args:
        model: The model to train
        trainloader: Training data loader
        criterion: Loss function
        host_optimizer: Optimizer for host parameters
        seed_optimizer: Optimizer for seed parameters (optional)
        device: Device to train on
        task_type: "classification" or "lm"
        collect_gradients: If True, collect gradient stats for telemetry

    Returns:
        Tuple of (running_loss, correct_count, total_count, grad_stats)
        - running_loss: Sum of loss values across batches (float)
        - correct_count: Sum of correct predictions (float/int)
        - total_count: Total samples processed (int)
        - grad_stats: GradientHealthStats if collect_gradients=True, else None

    Note:
        Uses tensor accumulation internally with a single .item() sync at epoch end
        to avoid CUDA synchronization overhead in the hot path.
    """
    model.train()

    # Pre-allocate accumulators on device to avoid .item() sync per batch
    # This is the key optimization: accumulate as tensors, sync once at epoch end
    running_loss = torch.zeros(1, device=device)
    running_correct = torch.zeros(1, device=device, dtype=torch.long)
    total = 0
    grad_stats = None

    for inputs, targets in trainloader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        host_optimizer.zero_grad(set_to_none=True)
        if seed_optimizer:
            seed_optimizer.zero_grad(set_to_none=True)

        loss, outputs = compiled_train_step(model, inputs, targets, criterion)
        # Compute metrics from outputs (compiled_train_step already computed loss)
        if task_type == "classification":
            _, predicted = outputs.max(1)
            correct_batch = predicted.eq(targets).sum()
            batch_total = targets.size(0)
        else:  # LM task
            # Use zeros with dtype=long to match classification branch's .sum() output
            correct_batch = torch.zeros((), device=outputs.device, dtype=torch.long)
            batch_total = targets.numel()
        loss.backward()  # type: ignore[no-untyped-call]

        # Collect gradient stats as tensors (async-safe, no .item() sync)
        # Overwrites each batch; final value materialized after loop
        if collect_gradients:
            # cast() needed because nn.Module doesn't expose get_seed_parameters/seed_slots in stubs
            slotted_model = cast(SlottedHostProtocol, model)
            grad_stats = collect_seed_gradients_async(slotted_model.get_seed_parameters())
            # Keep Kasmina's per-seed G2 metric fresh in strict gate runs.
            # This uses SeedSlot's async capture (no .item() sync in hot path).
            for slot in slotted_model.seed_slots.values():
                slot.capture_gradient_telemetry_async()

        host_optimizer.step()
        if seed_optimizer:
            seed_optimizer.step()

        # Accumulate on device - no .item() sync in hot path
        running_loss.add_(loss.detach())
        running_correct.add_(correct_batch)
        total += batch_total

    # Single sync at epoch end (forces all CUDA ops to complete)
    epoch_loss = running_loss.item()
    epoch_correct = running_correct.item()

    # Finalize per-slot gradient stats AFTER the sync above (safe to call .item()).
    if collect_gradients:
        slotted_model = cast(SlottedHostProtocol, model)
        for slot in slotted_model.seed_slots.values():
            slot.finalize_gradient_telemetry()

    # Now safe to materialize gradient tensors (after implicit sync above)
    materialized_grad_stats: GradientHealthStats | None = None
    if grad_stats is not None and not grad_stats.get('_empty', False):
        materialized_grad_stats = materialize_grad_stats(grad_stats)

    return epoch_loss, epoch_correct, total, materialized_grad_stats


# =============================================================================
# Heuristic Training
# =============================================================================

def _convert_flat_to_factored(action: Any, topology: str = "cnn") -> FactoredAction:
    """Convert flat action enum to FactoredAction for heuristic path.

    Maps flat action names to factored action components.

    Raises:
        ValueError: If action name is not recognized (fail-fast on misconfiguration)
    """
    from esper.leyline import AlphaTargetAction, BlueprintAction

    action_name = action.name

    if is_germinate_action_name(action_name):
        # Extract blueprint from action name like "GERMINATE_CONV_LIGHT"
        blueprint_name_upper = action_name[len(GERMINATE_PREFIX):]
        # Look up BlueprintAction by name - fail fast on unknown blueprints
        try:
            blueprint = BlueprintAction[blueprint_name_upper]
        except KeyError:
            raise ValueError(
                f"Unknown blueprint in action name: {action_name!r}. "
                f"Expected format: GERMINATE_<BLUEPRINT> where BLUEPRINT is one of {[b.name for b in BlueprintAction]}"
            )
        return FactoredAction.from_indices(
            slot_idx=0,  # Default to first slot
            blueprint_idx=blueprint.value,
            style_idx=0,  # Default style
            tempo_idx=0,
            alpha_target_idx=AlphaTargetAction.FULL.value,
            alpha_speed_idx=0,
            alpha_curve_idx=0,
            op_idx=LifecycleOp.GERMINATE,
        )
    elif action_name == "FOSSILIZE":
        return FactoredAction.from_indices(
            slot_idx=0,
            blueprint_idx=0,
            style_idx=0,
            tempo_idx=0,
            alpha_target_idx=AlphaTargetAction.FULL.value,
            alpha_speed_idx=0,
            alpha_curve_idx=0,
            op_idx=LifecycleOp.FOSSILIZE,
        )
    elif action_name == "PRUNE":
        return FactoredAction.from_indices(
            slot_idx=0,
            blueprint_idx=0,
            style_idx=0,
            tempo_idx=0,
            alpha_target_idx=AlphaTargetAction.FULL.value,
            alpha_speed_idx=0,
            alpha_curve_idx=0,
            op_idx=LifecycleOp.PRUNE,
        )
    elif action_name == "ADVANCE":
        return FactoredAction.from_indices(
            slot_idx=0,
            blueprint_idx=0,
            style_idx=0,
            tempo_idx=0,
            alpha_target_idx=AlphaTargetAction.FULL.value,
            alpha_speed_idx=0,
            alpha_curve_idx=0,
            op_idx=LifecycleOp.ADVANCE,
        )
    elif action_name == "WAIT":
        return FactoredAction.from_indices(
            slot_idx=0,
            blueprint_idx=0,
            style_idx=0,
            tempo_idx=0,
            alpha_target_idx=AlphaTargetAction.FULL.value,
            alpha_speed_idx=0,
            alpha_curve_idx=0,
            op_idx=LifecycleOp.WAIT,
        )
    else:
        # Fail fast on unknown actions - don't silently map to WAIT
        raise ValueError(
            f"Unknown action name: {action_name!r}. "
            f"Expected one of: WAIT, ADVANCE, FOSSILIZE, PRUNE, or GERMINATE_<BLUEPRINT>"
        )


def run_heuristic_episode(
    policy: "HeuristicTamiyo",
    trainloader: "DataLoader[tuple[torch.Tensor, torch.Tensor]]",
    testloader: "DataLoader[tuple[torch.Tensor, torch.Tensor]]",
    max_epochs: int = 75,
    max_batches: int | None = None,
    base_seed: int = 42,
    device: str = "cuda:0",
    task_spec: "TaskSpec | None" = None,
    slots: list[str] | None = None,
    telemetry_config: TelemetryConfig | None = None,
    telemetry_lifecycle_only: bool = False,
    gradient_telemetry_stride: int = 10,
) -> tuple[float, dict[str, int], list[Any]]:
    """Run a single training episode with heuristic policy.

    Args:
        policy: HeuristicTamiyo instance
        trainloader: Training data loader
        testloader: Test data loader
        max_epochs: Maximum epochs per episode
        max_batches: Limit batches per epoch (None = all)
        base_seed: Random seed
        device: Device to use
        task_spec: Task specification

    Returns:
        (final_accuracy, action_counts, episode_rewards)
    """
    from esper.leyline import SeedStage
    from esper.tolaria import create_model
    from esper.tamiyo import SignalTracker

    if task_spec is None:
        from esper.runtime import get_task_spec
        task_spec = get_task_spec("cifar_baseline")
    task_type = task_spec.task_type

    if slots is None:
        slots = list(SlotConfig.default().slot_ids)
    if not slots:
        raise ValueError("slots parameter is required and cannot be empty")

    torch.manual_seed(base_seed)
    random.seed(base_seed)

    episode_id = f"heur_{base_seed}"
    model = cast(SlottedHostProtocol, create_model(task=task_spec, device=device, slots=slots))

    # Note: create_model() already validates for duplicate slots, no need to check here
    enabled_slots = validate_slot_ids(list(slots))

    # Wire telemetry
    hub = get_hub()

    ops_telemetry_enabled = telemetry_config is None or telemetry_config.should_collect("ops_normal")

    def telemetry_callback(event: TelemetryEvent) -> None:
        # Note: Slot telemetry events already have properly typed payloads,
        # we just emit them as-is. The env_id/device are set when creating
        # EPOCH_COMPLETED and TRAINING_STARTED events.
        hub.emit(event)

    for slot_id in enabled_slots:
        slot = model.seed_slots[slot_id]
        slot.fast_mode = not ops_telemetry_enabled
        slot.telemetry_lifecycle_only = telemetry_lifecycle_only and not ops_telemetry_enabled
        slot.on_telemetry = (
            telemetry_callback
            if ops_telemetry_enabled or telemetry_lifecycle_only
            else None
        )
        slot.isolate_gradients = True

    # Calculate host_params before emitting (needed for Karn TUI)
    host_params = sum(p.numel() for p in model.get_host_parameters() if p.requires_grad)
    reward_config = ContributionRewardConfig()
    prev_slot_alphas = {slot_id: 0.0 for slot_id in enabled_slots}
    prev_slot_params = {slot_id: 0 for slot_id in enabled_slots}

    # Emit TRAINING_STARTED to activate Karn (P1 fix)
    # Use max_batches if provided, otherwise use len(trainloader)
    batches_per_epoch = max_batches if max_batches is not None else len(trainloader)

    hub.emit(TelemetryEvent(
        event_type=TelemetryEventType.TRAINING_STARTED,
        data=TrainingStartedPayload(
            n_envs=1,
            max_epochs=max_epochs,
            max_batches=batches_per_epoch,
            task=task_spec.name,
            host_params=host_params,
            slot_ids=tuple(enabled_slots),
            seed=base_seed,
            n_episodes=1,
            lr=task_spec.host_lr,
            clip_ratio=0.2,
            entropy_coef=0.01,
            param_budget=0,
            policy_device="cpu",  # Heuristic policy runs on CPU
            env_devices=(device,),  # Use actual training device, not hardcoded "cpu"
            episode_id=episode_id,
            reward_mode="shaped",  # Heuristic mode uses shaped rewards
        ),
    ))

    criterion = nn.CrossEntropyLoss()
    host_optimizer = torch.optim.SGD(
        model.get_host_parameters(), lr=task_spec.host_lr, momentum=0.9, weight_decay=5e-4
    )
    seed_optimizer = None
    signal_tracker = SignalTracker()

    seeds_created = 0
    # Track ops by LifecycleOp enum value
    action_counts = {op.name: 0 for op in LifecycleOp}
    episode_rewards = []

    # Pre-allocate accumulators ONCE before epoch loop (avoid O(max_epochs) allocations)
    running_loss = torch.zeros(1, device=device, dtype=torch.float32)
    running_correct = torch.zeros(1, device=device, dtype=torch.long)
    val_loss_accum = torch.zeros(1, device=device, dtype=torch.float32)
    val_correct_accum = torch.zeros(1, device=device, dtype=torch.long)

    for epoch in range(1, max_epochs + 1):
        for slot_id in enabled_slots:
            slot = model.seed_slots[slot_id]
            slot.telemetry_inner_epoch = epoch
            slot.telemetry_global_epoch = epoch

        # Determine if we should collect gradient telemetry this epoch
        collect_gradients = (
            ops_telemetry_enabled
            and gradient_telemetry_stride > 0
            and epoch % gradient_telemetry_stride == 0
        )
        grad_async: dict[str, Any] | None = None

        # Training phase - use tensor accumulation for deferred sync
        model.train()
        running_loss.zero_()
        running_correct.zero_()
        total = 0
        batch_count = 0

        for inputs, targets in trainloader:
            if max_batches and batch_count >= max_batches:
                break
            batch_count += 1

            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            host_optimizer.zero_grad(set_to_none=True)
            if seed_optimizer:
                seed_optimizer.zero_grad(set_to_none=True)

            outputs = model(inputs)
            loss, correct_batch, batch_total = compute_task_loss_with_metrics(outputs, targets, criterion, task_type)
            loss.backward()  # type: ignore[no-untyped-call]

            # Collect gradient stats (async-safe, overwrites each batch; final value materialized after loop)
            if collect_gradients:
                grad_async = collect_seed_gradients_async(model.get_seed_parameters())
                # Keep Kasmina's per-seed G2 metric fresh in strict gate runs.
                for slot_id in enabled_slots:
                    model.seed_slots[slot_id].capture_gradient_telemetry_async()

            host_optimizer.step()
            if seed_optimizer:
                seed_optimizer.step()

            # Accumulate on device - no .item() sync in hot path
            running_loss.add_(loss.detach())
            running_correct.add_(correct_batch)
            total += batch_total

        # Single sync at end of training
        train_loss = running_loss.item() / max(1, batch_count)
        train_acc = 100.0 * running_correct.item() / total if total > 0 else 0.0

        # Finalize per-slot gradient stats AFTER the sync above (safe to call .item()).
        if collect_gradients:
            for slot_id in enabled_slots:
                model.seed_slots[slot_id].finalize_gradient_telemetry()

        # Materialize gradient stats after training sync (safe to access .item() now)
        epoch_grad_stats: GradientHealthStats | None = None
        if grad_async is not None and not grad_async.get("_empty", False):
            epoch_grad_stats = materialize_grad_stats(grad_async)

        # Validation - use tensor accumulation for deferred sync
        model.eval()
        val_loss_accum.zero_()
        val_correct_accum.zero_()
        total = 0
        batch_count = 0

        with torch.inference_mode():
            for inputs, targets in testloader:
                if max_batches and batch_count >= max_batches:
                    break
                batch_count += 1

                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss, correct_batch, batch_total = compute_task_loss_with_metrics(outputs, targets, criterion, task_type)
                val_loss_accum.add_(loss)
                val_correct_accum.add_(correct_batch)
                total += batch_total

        # Single sync at end of validation
        val_loss = val_loss_accum.item() / max(1, batch_count)
        val_acc = 100.0 * val_correct_accum.item() / total if total > 0 else 0.0

        # Gather active seeds across ALL enabled slots (multi-slot support).
        # Assert seed_id uniqueness - duplicate IDs would make target resolution ambiguous.
        active_seeds: list[Any] = []  # List of SeedState (protocol doesn't match concrete type)
        seed_ids: set[str] = set()
        for slot_id in enabled_slots:
            if not model.has_active_seed_in_slot(slot_id):
                continue
            state = model.seed_slots[slot_id].state
            if state is None:
                continue
            if state.seed_id in seed_ids:
                raise RuntimeError(f"Duplicate seed_id '{state.seed_id}' across slots in one env")
            seed_ids.add(state.seed_id)
            active_seeds.append(state)

        # Record accuracy in seed metrics (per-slot counters + deltas).
        for seed_state in active_seeds:
            if seed_state.metrics:
                seed_state.metrics.record_accuracy(val_acc)

        # Update signal tracker
        # Phase 4: embargo/cooldown stages keep state while seed is physically removed.
        # Availability for germination is therefore "no state", not merely "no active seed".
        available_slots = sum(
            1 for slot_id in enabled_slots if model.seed_slots[slot_id].state is None
        )
        signals = signal_tracker.update(
            epoch=epoch,
            global_step=epoch * len(trainloader),
            train_loss=train_loss,
            train_accuracy=train_acc,
            val_loss=val_loss,
            val_accuracy=val_acc,
            active_seeds=active_seeds,
            available_slots=available_slots,
        )

        acc_delta = signals.metrics.accuracy_delta

        # Mechanical lifecycle advance (alpha ticking + cooldown pipeline)
        for slot_id in enabled_slots:
            model.seed_slots[slot_id].step_epoch()

        def resolve_slot_for_seed_id(seed_id: str) -> str:
            slot_matches = [
                slot_id for slot_id in enabled_slots
                if (state := model.seed_slots[slot_id].state) is not None
                and state.seed_id == seed_id
            ]
            if len(slot_matches) != 1:
                raise RuntimeError(
                    f"target_seed_id '{seed_id}' expected in exactly 1 slot, found {slot_matches}"
                )
            return slot_matches[0]

        # Get heuristic decision and convert to factored action
        decision = policy.decide(signals, active_seeds)
        flat_action = decision.action
        factored_action = _convert_flat_to_factored(flat_action, task_spec.topology)
        action_counts[factored_action.op.name] += 1

        # Compute reward (for comparison with PPO)
        total_params = model.active_seed_params
        reward_seed_state = None
        reward_seed_params = 0
        if decision.target_seed_id:
            target_slot = resolve_slot_for_seed_id(decision.target_seed_id)
            reward_seed_state = model.seed_slots[target_slot].state
            reward_seed_params = model.seed_slots[target_slot].active_seed_params
        effective_seed_params, alpha_delta_sq_sum = compute_rent_and_shock_inputs(
            model=model,
            slot_ids=enabled_slots,
            host_params=host_params,
            host_params_floor=reward_config.rent_host_params_floor,
            base_slot_rent_ratio=reward_config.base_slot_rent_ratio,
            prev_slot_alphas=prev_slot_alphas,
            prev_slot_params=prev_slot_params,
        )
        reward = compute_contribution_reward(
            action=factored_action.op,  # Pass the LifecycleOp enum
            seed_contribution=None,  # No counterfactual in heuristic path
            val_acc=val_acc,
            seed_info=SeedInfo.from_seed_state(
                reward_seed_state,
                reward_seed_params,
            ),
            epoch=epoch,
            max_epochs=max_epochs,
            total_params=total_params,
            host_params=host_params,
            acc_delta=acc_delta,  # Used as proxy signal
            effective_seed_params=effective_seed_params,
            alpha_delta_sq_sum=alpha_delta_sq_sum,
            config=reward_config,
        )
        episode_rewards.append(reward)

        germinate_slot = next(
            # Phase 4: EMBARGOED/RESETTING retain state, so only state==None is available.
            (slot_id for slot_id in enabled_slots if model.seed_slots[slot_id].state is None),
            None,
        )

        # Execute action using FactoredAction properties
        if factored_action.is_germinate:
            if germinate_slot is not None:
                blueprint_id = factored_action.blueprint_id
                if blueprint_id is not None:
                    seed_id = f"seed_{seeds_created}"
                    model.germinate_seed(blueprint_id, seed_id, slot=germinate_slot)
                    seeds_created += 1
                    seed_optimizer = torch.optim.SGD(
                        model.get_seed_parameters(), lr=task_spec.seed_lr, momentum=0.9
                    )

        elif factored_action.is_fossilize:
            if decision.target_seed_id:
                target_slot = resolve_slot_for_seed_id(decision.target_seed_id)
                slot_state = model.seed_slots[target_slot].state
                if slot_state is not None and slot_state.stage == SeedStage.HOLDING:
                    slot = model.seed_slots[target_slot]
                    gate_result = slot.advance_stage(SeedStage.FOSSILIZED)
                    if gate_result.passed:
                        slot.set_alpha(1.0)

        elif factored_action.is_prune:
            if decision.target_seed_id:
                target_slot = resolve_slot_for_seed_id(decision.target_seed_id)
                model.prune_seed(slot=target_slot)
                seed_optimizer = (
                    torch.optim.SGD(model.get_seed_parameters(), lr=task_spec.seed_lr, momentum=0.9)
                    if model.has_active_seed else None
                )
        elif factored_action.op == LifecycleOp.ADVANCE:
            if decision.target_seed_id:
                target_slot = resolve_slot_for_seed_id(decision.target_seed_id)
                slot = model.seed_slots[target_slot]
                gate_result = slot.advance_stage()
                if not gate_result.passed:
                    pass

        # Build seeds telemetry dict if gradient stats were collected
        seeds_telemetry: dict[str, dict[str, Any]] | None = None
        if epoch_grad_stats is not None:
            # Aggregate gradient health across all seeds for this epoch
            # Access TypedDict fields directly - GradientHealthStats guarantees these keys exist
            seeds_telemetry = {
                "__aggregate__": {
                    "gradient_norm": epoch_grad_stats["gradient_norm"],
                    "gradient_health": epoch_grad_stats["gradient_health"],
                    "has_vanishing": epoch_grad_stats["has_vanishing"],
                    "has_exploding": epoch_grad_stats["has_exploding"],
                }
            }

        hub.emit(TelemetryEvent(
            event_type=TelemetryEventType.EPOCH_COMPLETED,
            epoch=epoch,
            seed_id=decision.target_seed_id,
            data=EpochCompletedPayload(
                env_id=0,
                val_accuracy=val_acc,
                val_loss=val_loss,
                inner_epoch=epoch,
                train_loss=train_loss,
                train_accuracy=train_acc,
                seeds=seeds_telemetry,
            ),
        ))

    return val_acc, action_counts, episode_rewards


def train_heuristic(
    n_episodes: int = 1,
    max_epochs: int = 75,
    max_batches: int | None = 50,
    device: str = "cuda:0",
    task: str = "cifar_baseline",
    seed: int = 42,
    slots: list[str] | None = None,
    telemetry_config: TelemetryConfig | None = None,
    telemetry_lifecycle_only: bool = False,
    min_fossilize_improvement: float | None = None,
    gradient_telemetry_stride: int = 10,
) -> list[dict[str, Any]]:
    """Train with heuristic policy.

    Args:
        n_episodes: Number of episodes to run
        max_epochs: Maximum epochs per episode
        max_batches: Limit batches per epoch (None = all, 50 = fast mode)
        device: Device to use
        task: Task preset (cifar10 or tinystories)
        seed: Random seed
        slots: List of slot IDs to use
        telemetry_config: Telemetry configuration
        telemetry_lifecycle_only: If True, only emit lifecycle events
        min_fossilize_improvement: Minimum improvement (%) required to fossilize a seed.
            If None, uses leyline default (0.5%). Lower values risk reward hacking.
    """
    from esper.tamiyo import HeuristicTamiyo
    from esper.tamiyo.heuristic import HeuristicPolicyConfig
    from esper.runtime import get_task_spec  # Lazy import to avoid circular dependency

    if slots is None:
        slots = list(SlotConfig.default().slot_ids)
    if not slots:
        raise ValueError("slots parameter is required and cannot be empty")

    task_spec = get_task_spec(task)

    hub = get_hub()
    ops_telemetry_enabled = telemetry_config is None or telemetry_config.should_collect("ops_normal")
    if telemetry_lifecycle_only and not ops_telemetry_enabled:
        hub.emit(TelemetryEvent(
            event_type=TelemetryEventType.ANALYTICS_SNAPSHOT,
            severity="warning",
            message="Ops telemetry disabled; emitting lifecycle-only seed telemetry",
            data=AnalyticsSnapshotPayload(
                kind="heuristic_warning",
                env_id=0,
                mode="heuristic",
                task=task_spec.name,
                device=device,
                telemetry_lifecycle_only=True,
                telemetry_level=telemetry_config.level.name if telemetry_config is not None else None,
            ),
        ))
    hub.emit(TelemetryEvent(
        event_type=TelemetryEventType.ANALYTICS_SNAPSHOT,
        message="Heuristic training run configuration",
        data=AnalyticsSnapshotPayload(
            kind="heuristic_config",
            env_id=0,
            mode="heuristic",
            task=task_spec.name,
            topology=task_spec.topology,
            episodes=n_episodes,
            max_epochs=max_epochs,
            max_batches=max_batches,
            device=device,
            slots=tuple(slots),
            min_fossilize_improvement=min_fossilize_improvement,
        ),
    ))

    trainloader, testloader = task_spec.create_dataloaders()

    # Create policy config if custom parameters are specified
    policy_config = None
    if min_fossilize_improvement is not None:
        policy_config = HeuristicPolicyConfig(
            min_improvement_to_fossilize=min_fossilize_improvement,
        )

    policy = HeuristicTamiyo(topology=task_spec.topology, config=policy_config)
    history = []

    for ep in range(1, n_episodes + 1):
        policy.reset()
        base_seed = seed + ep * 1000
        episode_id = f"heur_{base_seed}"

        final_acc, action_counts, rewards = run_heuristic_episode(
            policy=policy,
            trainloader=trainloader,
            testloader=testloader,
            max_epochs=max_epochs,
            max_batches=max_batches,
            base_seed=base_seed,
            device=device,
            task_spec=task_spec,
            slots=slots,
            telemetry_config=telemetry_config,
            telemetry_lifecycle_only=telemetry_lifecycle_only,
            gradient_telemetry_stride=gradient_telemetry_stride,
        )

        total_reward = sum(rewards)
        hub.emit(TelemetryEvent(
            event_type=TelemetryEventType.ANALYTICS_SNAPSHOT,
            message="Heuristic episode completed",
            data=AnalyticsSnapshotPayload(
                kind="heuristic_episode",
                env_id=0,
                mode="heuristic",
                task=task_spec.name,
                episode_id=episode_id,
                episode=ep,
                episodes_total=n_episodes,
                base_seed=base_seed,
                final_accuracy=final_acc,
                total_reward=total_reward,
                action_counts={str(k): int(v) for k, v in action_counts.items()},
            ),
        ))

        history.append({
            'episode': ep,
            'accuracy': final_acc,
            'total_reward': total_reward,
            'action_counts': dict(action_counts),
        })

    return history


__all__ = [
    "run_heuristic_episode",
    "train_heuristic",
]
