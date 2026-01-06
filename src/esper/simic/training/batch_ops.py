from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Callable, cast

import torch
import torch.amp as torch_amp
import torch.nn as nn

from esper.leyline import SeedSlotProtocol
from esper.simic.telemetry import (
    collect_host_gradients_async,
    collect_seed_gradients_async,
    collect_seed_gradients_only_async,
)
from esper.tamiyo.policy.features import batch_obs_to_features

from .parallel_env_state import ParallelEnvState


@torch.compiler.disable  # type: ignore[untyped-decorator]
def _collect_gradient_telemetry_for_batch(
    model: Any,
    slots_with_active_seeds: list[str],
    env_dev: str,
) -> dict[str, dict[Any, Any]] | None:
    """Collect gradient telemetry for all active slots.

    Isolated from torch.compile to prevent graph breaks from
    data-dependent slot iteration and conditional logic.

    Args:
        model: The model with seed slots
        slots_with_active_seeds: Pre-filtered list of slots with active seeds
        env_dev: Device string
    """
    from esper.leyline import SeedStage

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
        # Dual stats for G2 gate (host/seed ratio)
        seed_stats = collect_seed_gradients_only_async(
            model.get_seed_parameters(slot_id),
            device=env_dev,
        )
        # Health stats for gradient telemetry (vanishing/exploding detection)
        # Note: get_seed_parameters returns a generator, call it again
        health_stats = collect_seed_gradients_async(
            model.get_seed_parameters(slot_id),
        )
        grad_stats_by_slot[slot_id] = {
            **host_stats,
            **seed_stats,
            "_health_stats": health_stats,  # Nested to avoid key conflicts
        }

    return grad_stats_by_slot


def process_train_batch(
    env_state: ParallelEnvState,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    *,
    use_telemetry: bool = False,
    slots: list[str] | None = None,
    max_grad_norm: float | None = None,
    task_spec: Any,
    resolved_amp_dtype: torch.dtype | None,
    loss_and_correct_fn: Callable[..., tuple[torch.Tensor, torch.Tensor, int]],
) -> tuple[torch.Tensor, torch.Tensor, int, dict[str, dict[str, Any]] | None]:
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
        max_grad_norm: Maximum gradient norm for clipping. None disables clipping.

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
            # Record stream usage for proper CUDA synchronization.
            # Both SharedBatchIterator and SharedGPUBatchIterator now clone
            # their tensors, so we don't need to clone here - just record.
            inputs.record_stream(env_state.stream)
            targets.record_stream(env_state.stream)

        # Ensure per-slot seed optimizers exist for any slot with a live seed.
        # We keep optimizers per-slot to avoid dynamic param-group surgery.
        slots_to_step: list[str] = []
        for slot_id in slots_with_active_seeds:
            # Already filtered to active slots via cache
            slot = cast(SeedSlotProtocol, model.seed_slots[slot_id])
            seed_state = slot.state
            if seed_state is None:
                continue

            # Seeds can continue training through BLENDING/HOLDING/FOSSILIZED.
            slots_to_step.append(slot_id)

            # OPTIMIZATION: Removed expensive parameter-set validation from hot path.
            # Rely on env_state.seed_optimizers.pop() in the action execution block.
            if slot_id not in env_state.seed_optimizers:
                seed_params = list(model.get_seed_parameters(slot_id))
                if not seed_params:
                    slot_for_debug = cast(
                        SeedSlotProtocol, model.seed_slots[slot_id]
                    )
                    raise RuntimeError(
                        f"Seed in slot '{slot_id}' has no trainable parameters. "
                        f"Stage: {seed_state.stage.name}, "
                        f"Blueprint: {seed_state.blueprint_id}, "
                        f"Slot.seed: {slot_for_debug.seed is not None}"
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
            torch_amp.autocast(device_type="cuda", dtype=resolved_amp_dtype)  # type: ignore[attr-defined]
            if env_state.autocast_enabled and resolved_amp_dtype is not None
            else nullcontext()
        )
        with autocast_ctx:
            outputs = model(inputs)
            loss, correct_result, total = loss_and_correct_fn(
                outputs, targets, criterion, task_type=task_spec.task_type
            )
            # loss_and_correct always returns a tensor for correct (see docstring)
            # .sum() on boolean tensor yields 0-dim tensor, never Python int
            correct_tensor = correct_result

        # AMP backward path depends on dtype:
        # - FP16 (scaler != None): Requires GradScaler to handle underflow.
        #   scale() multiplies loss, unscale() before step() inverts it.
        # - BF16 (scaler == None): Has 8 exponent bits (vs FP16's 5), so the
        #   dynamic range matches FP32. No scaling needed; direct backward().
        # Note: backward() is untyped in PyTorch stubs
        if env_state.scaler is not None:
            env_state.scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
        else:
            loss.backward()  # type: ignore[no-untyped-call]
        # Collect gradient telemetry (isolated from torch.compile)
        grad_stats_by_slot = None
        if use_telemetry:
            grad_stats_by_slot = _collect_gradient_telemetry_for_batch(
                model, slots_with_active_seeds, env_dev
            )

        # Compute grad presence once for each seed optimizer (avoid redundant checks)
        # Guard: Only call scaler.step() if optimizer has gradients.
        # With isolate_gradients=True, seeds may not have grads in the scaled backward.
        seed_opts_with_grads: dict[str, tuple[torch.optim.Optimizer, bool]] = {}
        for slot_id in slots_to_step:
            seed_opt = env_state.seed_optimizers[slot_id]
            has_grads = any(
                p.grad is not None
                for group in seed_opt.param_groups
                for p in group["params"]
            )
            seed_opts_with_grads[slot_id] = (seed_opt, has_grads)

        # Gradient clipping (AMP-safe)
        # AMP ordering: scale() -> backward() -> unscale_() -> clip_grad_norm_() -> step() -> update()
        if max_grad_norm is not None and max_grad_norm > 0:
            if env_state.scaler is not None:
                # Unscale before clipping (required for correct FP32 magnitude)
                env_state.scaler.unscale_(env_state.host_optimizer)
                for slot_id, (seed_opt, has_grads) in seed_opts_with_grads.items():
                    if has_grads:
                        env_state.scaler.unscale_(seed_opt)

            # Clip host and each seed independently (preserves gradient isolation)
            # Joint clipping would allow large host gradients to reduce seed budget and vice versa
            host_params = list(model.get_host_parameters())
            if host_params:
                torch.nn.utils.clip_grad_norm_(host_params, max_grad_norm)

            for slot_id in slots_to_step:
                seed_params = list(model.get_seed_parameters(slot_id))
                if seed_params:
                    torch.nn.utils.clip_grad_norm_(seed_params, max_grad_norm)

        # Optimizer step (reuses has_grads computation)
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
        if env_state.scaler is not None:
            env_state.scaler.step(env_state.host_optimizer)
            for slot_id, (seed_opt, has_grads) in seed_opts_with_grads.items():
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
    *,
    slots: list[str] | None = None,
    task_spec: Any,
    loss_and_correct_fn: Callable[..., tuple[torch.Tensor, torch.Tensor, int]],
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
            # Record stream usage for proper CUDA synchronization.
            # Both SharedBatchIterator and SharedGPUBatchIterator now clone
            # their tensors, so we don't need to clone here - just record.
            inputs.record_stream(env_state.stream)
            targets.record_stream(env_state.stream)

        model.eval()
        with torch.inference_mode():
            outputs = model(inputs)
            loss, correct_result, total = loss_and_correct_fn(
                outputs, targets, criterion, task_type=task_spec.task_type
            )
            # loss_and_correct always returns a tensor for correct (see docstring)
            correct_tensor = correct_result

        # Return tensors - .item() called after stream sync
        return loss, correct_tensor, total


def batch_signals_to_features(
    batch_signals: list[Any],
    batch_slot_reports: list[dict[str, Any]],
    slot_config: Any,
    env_states: list[ParallelEnvState],
    device: torch.device,
    *,
    max_epochs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Consolidated signals-to-features conversion for all environments.

    Returns:
        obs: [batch, obs_dim] - observation features (Obs V3: 116 dims for 3 slots)
        blueprint_indices: [batch, num_slots] - blueprint indices for embedding lookup (int64)
    """
    return batch_obs_to_features(
        batch_signals=batch_signals,
        batch_slot_reports=batch_slot_reports,
        batch_env_states=env_states,
        slot_config=slot_config,
        device=device,
        max_epochs=max_epochs,
    )
