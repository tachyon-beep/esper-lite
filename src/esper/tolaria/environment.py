"""Tolaria Environment - Model factory for Model Alpha.

This module provides the model factory for creating Model Alpha instances.
Dataset loading is handled separately by esper.utils.data.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

# NOTE: TaskSpec imported under TYPE_CHECKING to avoid circular import:
#   runtime -> simic -> simic.training -> vectorized -> tolaria -> environment -> runtime
# get_task_spec imported lazily inside create_model at runtime.
if TYPE_CHECKING:
    from esper.runtime import TaskSpec


def _validate_device(device: str) -> None:
    """Validate requested device, especially for multi-GPU setups.

    Raises a clear error when a specific CUDA index is unavailable instead of
    failing later during model construction.
    """
    if not device.startswith("cuda"):
        return

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device '{device}' requested but CUDA is not available. "
            f"Use device='cpu' or check your CUDA installation."
        )

    # torch.device handles bare "cuda" (index=None) which uses the current default
    requested_index = torch.device(device).index
    if requested_index is None:
        return

    available = torch.cuda.device_count()
    if requested_index >= available:
        raise RuntimeError(
            f"CUDA device '{device}' requested but only {available} device(s) are available."
        )


def create_model(task: TaskSpec | str = "cifar10", device: str = "cuda", slots: list[str] | None = None) -> torch.nn.Module:
    """Create a MorphogeneticModel for the given task on device.

    Args:
        task: Task specification or name.
        device: Target device for the model.
        slots: Seed slots to enable. Required and cannot be empty.
    """
    if isinstance(task, str):
        from esper.runtime import get_task_spec  # Lazy import to avoid circular dependency
        task_spec = get_task_spec(task)
    else:
        task_spec = task

    _validate_device(device)

    if not slots:
        raise ValueError("slots parameter is required and cannot be empty")

    return task_spec.create_model(device=device, slots=slots)
