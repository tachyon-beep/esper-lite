from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Callable

import torch
import torch.nn as nn

from .parallel_env_state import ParallelEnvState


def process_fused_val_batch(
    env_state: ParallelEnvState,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    alpha_overrides: dict[str, torch.Tensor],
    num_configs: int,
    *,
    task_spec: Any,
    loss_and_correct_fn: Callable[..., tuple[torch.Tensor, torch.Tensor, int]],
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Process a fused validation batch with multiple alpha configurations.

    Uses MorphogeneticModel.fused_forward() to saturate GPU and avoid CPU orchestration stalls.

    Args:
        env_state: Parallel environment state
        inputs: Original input tensor [B, ...]
        targets: Original target tensor [B, ...]
        criterion: Loss criterion
        alpha_overrides: Dict mapping slot_id -> override tensor [K*B, 1, 1, 1] (CNN) or [K*B, 1, 1] (transformer)
        num_configs: Number of configurations K

    Returns:
        Tuple of (loss_tensor, correct_tensor, total_per_config) for the expanded batch.
    """
    model = env_state.model
    env_dev = env_state.env_device

    stream_ctx = (
        torch.cuda.stream(env_state.stream) if env_state.stream else nullcontext()
    )

    with stream_ctx:
        # Move data and expand for all configurations
        inputs = inputs.to(env_dev, non_blocking=True)
        targets = targets.to(env_dev, non_blocking=True)

        # B8-PT-01 FIX: Protect source tensors from premature deallocation.
        # The .to() may create new tensors via async copy. Without record_stream(),
        # the allocator might reclaim this memory before repeat() finishes reading.
        # When tensors are already on env_dev (SharedGPUBatchIterator path), .to() is
        # a no-op returning the same tensor; record_stream() is harmless but redundant
        # since the caller already protected them.
        if env_state.stream and inputs.is_cuda:
            inputs.record_stream(env_state.stream)
            targets.record_stream(env_state.stream)

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
            loss, correct_fused, total = loss_and_correct_fn(
                outputs,
                fused_targets,
                criterion,
                task_type=task_spec.task_type,
                elementwise=True,
            )

        # Sum elementwise correctness per configuration
        # correct_fused shape: [K*B] for classification, [K*B, T] for LM
        # Use view(K, -1) to handle both shapes uniformly (same pattern as loss below)
        # elementwise=True guarantees correct_fused is a Tensor
        assert isinstance(correct_fused, torch.Tensor)
        correct_per_config = correct_fused.view(num_configs, -1).sum(dim=1)

        # Compute per-config loss when using reduction='none' criterion
        # loss shape: [K*B] for classification, [K*B*T] for LM
        # Reshape to [K, -1] and mean to get per-config loss: [K]
        # This separates main config (idx 0) from ablations for clean telemetry
        if loss.dim() > 0 and loss.numel() > 1:
            loss_per_config = loss.view(num_configs, -1).mean(dim=1)
        else:
            # Fallback for scalar loss (regular criterion with reduction='mean')
            loss_per_config = loss.unsqueeze(0).expand(num_configs)

        # M1 fix: Guard against edge case where batch_size < num_configs
        # This can happen during debugging or with very small batches
        total_per_config = total // num_configs
        if total_per_config == 0:
            total_per_config = 1  # Prevent divide-by-zero downstream
        return loss_per_config, correct_per_config, total_per_config
