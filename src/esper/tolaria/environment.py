"""Tolaria Environment - Model factory for Model Alpha.

This module provides the model factory for creating Model Alpha instances.
Dataset loading is handled separately by esper.utils.data.
"""

from __future__ import annotations

import torch

from esper.runtime import TaskSpec, get_task_spec


def create_model(task: TaskSpec | str = "cifar10", device: str = "cuda", slots: list[str] | None = None) -> torch.nn.Module:
    """Create a MorphogeneticModel for the given task on device.

    Args:
        task: Task specification or name.
        device: Target device for the model.
        slots: Seed slots to enable. If None, defaults to ["mid"] for convenience.
    """
    if isinstance(task, str):
        task_spec = get_task_spec(task)
    else:
        task_spec = task

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device '{device}' requested but CUDA is not available. "
            f"Use device='cpu' or check your CUDA installation."
        )

    if slots is None:
        slots = ["mid"]

    return task_spec.create_model(device=device, slots=slots)
