from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Callable

import torch
import torch.amp as torch_amp
import torch.nn as nn

from esper.leyline import SeedSlotProtocol, SeedStage
from esper.simic.telemetry import (
    collect_host_gradients_async,
    collect_seed_gradients_async,
    collect_seed_gradients_only_async,
)

from .parallel_env_state import ParallelEnvState


@torch.compiler.disable  # type: ignore[untyped-decorator]
def collect_gradient_telemetry_for_batch(
    model: Any,
    slots_with_active_seeds: list[str],
    env_dev: str,
) -> dict[str, dict[Any, Any]] | None:
    """Collect gradient telemetry for all active slots."""
    slots_needing_grad_telemetry = []
    for slot_id in slots_with_active_seeds:
        seed_state = model.seed_slots[slot_id].state
        if seed_state and seed_state.stage in (SeedStage.TRAINING, SeedStage.BLENDING):
            slots_needing_grad_telemetry.append(slot_id)

    if not slots_needing_grad_telemetry:
        return None

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
        health_stats = collect_seed_gradients_async(
            model.get_seed_parameters(slot_id),
        )
        grad_stats_by_slot[slot_id] = {
            **host_stats,
            **seed_stats,
            "_health_stats": health_stats,
        }

    return grad_stats_by_slot


def process_train_batch(
    env_state: ParallelEnvState,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    *,
    task_spec: Any,
    resolved_amp_dtype: torch.dtype | None,
    compiled_loss_and_correct: Callable[..., tuple[torch.Tensor, torch.Tensor, int]],
    use_telemetry: bool = False,
    slots: list[str] | None = None,
    use_amp: bool = False,
    max_grad_norm: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int, dict[str, dict[str, Any]] | None]:
    """Process a single training batch for one environment (runs in CUDA stream)."""
    if not slots:
        raise ValueError("slots parameter is required and cannot be empty")

    model = env_state.model
    env_dev = env_state.env_device

    active_slots = {slot_id: model.has_active_seed_in_slot(slot_id) for slot_id in slots}
    slots_with_active_seeds = [slot_id for slot_id, active in active_slots.items() if active]

    stream_ctx = (
        torch.cuda.stream(env_state.stream) if env_state.stream else nullcontext()
    )

    with stream_ctx:
        inputs = inputs.to(env_dev, non_blocking=True)
        targets = targets.to(env_dev, non_blocking=True)

        if env_state.stream and inputs.is_cuda:
            inputs.record_stream(env_state.stream)
            targets.record_stream(env_state.stream)

        slots_to_step: list[str] = []
        for slot_id in slots_with_active_seeds:
            slot = env_state.model.seed_slots[slot_id]
            seed_state = slot.state
            if seed_state is None:
                continue

            slots_to_step.append(slot_id)

            if slot_id not in env_state.seed_optimizers:
                seed_params = list(model.get_seed_parameters(slot_id))
                if not seed_params:
                    slot_for_debug = model.seed_slots[slot_id]
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

        autocast_ctx = (
            torch_amp.autocast(device_type="cuda", dtype=resolved_amp_dtype)  # type: ignore[attr-defined]
            if env_state.autocast_enabled and resolved_amp_dtype is not None
            else nullcontext()
        )
        with autocast_ctx:
            outputs = model(inputs)
            loss, correct_result, total = compiled_loss_and_correct(
                outputs, targets, criterion, task_type=task_spec.task_type
            )
            correct_tensor = correct_result

        if env_state.scaler is not None:
            env_state.scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
        else:
            loss.backward()  # type: ignore[no-untyped-call]

        grad_stats_by_slot = None
        if use_telemetry:
            grad_stats_by_slot = collect_gradient_telemetry_for_batch(
                model, slots_with_active_seeds, env_dev
            )

        seed_opts_with_grads: dict[str, tuple[torch.optim.Optimizer, bool]] = {}
        for slot_id in slots_to_step:
            seed_opt = env_state.seed_optimizers[slot_id]
            has_grads = any(
                p.grad is not None
                for group in seed_opt.param_groups
                for p in group["params"]
            )
            seed_opts_with_grads[slot_id] = (seed_opt, has_grads)

        if max_grad_norm is not None and max_grad_norm > 0:
            if env_state.scaler is not None:
                env_state.scaler.unscale_(env_state.host_optimizer)
                for slot_id, (seed_opt, has_grads) in seed_opts_with_grads.items():
                    if has_grads:
                        env_state.scaler.unscale_(seed_opt)

            host_params = list(model.get_host_parameters())
            if host_params:
                torch.nn.utils.clip_grad_norm_(host_params, max_grad_norm)

            for slot_id in slots_to_step:
                seed_params = list(model.get_seed_parameters(slot_id))
                if seed_params:
                    torch.nn.utils.clip_grad_norm_(seed_params, max_grad_norm)

        if env_state.scaler is not None:
            env_state.scaler.step(env_state.host_optimizer)
            for slot_id, (seed_opt, has_grads) in seed_opts_with_grads.items():
                if has_grads:
                    env_state.scaler.step(seed_opt)
                else:
                    seed_opt.step()
            env_state.scaler.update()
        else:
            env_state.host_optimizer.step()
            for slot_id in slots_to_step:
                env_state.seed_optimizers[slot_id].step()

        return loss.detach(), correct_tensor, total, grad_stats_by_slot


def process_val_batch(
    env_state: ParallelEnvState,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    *,
    task_spec: Any,
    compiled_loss_and_correct: Callable[..., tuple[torch.Tensor, torch.Tensor, int]],
    slots: list[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Process a validation batch for one environment."""
    model = env_state.model
    env_dev = env_state.env_device

    stream_ctx = (
        torch.cuda.stream(env_state.stream) if env_state.stream else nullcontext()
    )

    with stream_ctx:
        inputs = inputs.to(env_dev, non_blocking=True)
        targets = targets.to(env_dev, non_blocking=True)

        if env_state.stream and inputs.is_cuda:
            inputs.record_stream(env_state.stream)
            targets.record_stream(env_state.stream)

        model.eval()
        with torch.inference_mode():
            outputs = model(inputs)
            loss, correct_result, total = compiled_loss_and_correct(
                outputs, targets, criterion, task_type=task_spec.task_type
            )
            correct_tensor = correct_result

        return loss, correct_tensor, total


def process_fused_val_batch(
    env_state: ParallelEnvState,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    alpha_overrides: dict[str, torch.Tensor],
    num_configs: int,
    *,
    task_spec: Any,
    compiled_loss_and_correct: Callable[..., tuple[torch.Tensor, torch.Tensor, int]],
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Process a fused validation batch with multiple alpha configurations."""
    model = env_state.model
    env_dev = env_state.env_device

    stream_ctx = (
        torch.cuda.stream(env_state.stream) if env_state.stream else nullcontext()
    )

    with stream_ctx:
        inputs = inputs.to(env_dev, non_blocking=True)
        targets = targets.to(env_dev, non_blocking=True)

        if env_state.stream and inputs.is_cuda:
            inputs.record_stream(env_state.stream)
            targets.record_stream(env_state.stream)

        fused_inputs = inputs.repeat(num_configs, *([1] * (inputs.dim() - 1)))
        fused_targets = targets.repeat(num_configs, *([1] * (targets.dim() - 1)))
        if inputs.is_contiguous(memory_format=torch.channels_last):
            fused_inputs = fused_inputs.contiguous(memory_format=torch.channels_last)

        if env_state.stream and inputs.is_cuda:
            fused_inputs.record_stream(env_state.stream)
            fused_targets.record_stream(env_state.stream)

        model.eval()
        with torch.inference_mode():
            outputs = model.fused_forward(fused_inputs, alpha_overrides)
            loss, correct_fused, total = compiled_loss_and_correct(
                outputs,
                fused_targets,
                criterion,
                task_type=task_spec.task_type,
                elementwise=True,
            )

        assert isinstance(correct_fused, torch.Tensor)
        correct_per_config = correct_fused.view(num_configs, -1).sum(dim=1)

        if loss.dim() > 0 and loss.numel() > 1:
            loss_per_config = loss.view(num_configs, -1).mean(dim=1)
        else:
            loss_per_config = loss.unsqueeze(0).expand(num_configs)

        return loss_per_config, correct_per_config, total
