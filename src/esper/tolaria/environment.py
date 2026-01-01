"""Tolaria Environment - Model factory for Model Alpha.

This module provides the model factory for creating Model Alpha instances.
Dataset loading is handled separately by esper.utils.data.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

# NOTE: TaskSpec and MorphogeneticModel imported under TYPE_CHECKING to avoid circular imports:
#   runtime -> simic -> simic.training -> vectorized -> tolaria -> environment -> runtime
#   kasmina.host -> ... -> tolaria -> environment
# get_task_spec imported lazily inside create_model at runtime.
if TYPE_CHECKING:
    from esper.kasmina.host import MorphogeneticModel
    from esper.runtime import TaskSpec


def parse_device(device: str) -> torch.device:
    """Parse a device string to a torch.device, with clear error on invalid format.

    Args:
        device: Device string like "cpu", "cuda", "cuda:0", "cuda:1".

    Returns:
        Parsed torch.device object.

    Raises:
        ValueError: If the device string is malformed (e.g., "cuda0" instead of "cuda:0").
    """
    try:
        return torch.device(device)
    except RuntimeError as exc:
        raise ValueError(f"Invalid device '{device}': {exc}") from exc


def validate_device(device: str, *, require_explicit_index: bool = False) -> torch.device:
    """Validate and parse a device string.

    This is the canonical device validation for Esper. Use this instead of
    implementing device checks inline.

    Args:
        device: Device string like "cpu", "cuda", "cuda:0".
        require_explicit_index: If True, require explicit CUDA index (e.g., "cuda:0"
            instead of bare "cuda"). Use for multi-GPU training where device
            assignment must be unambiguous.

    Returns:
        Parsed torch.device object.

    Raises:
        ValueError: If device string is malformed or violates require_explicit_index.
        RuntimeError: If CUDA is requested but unavailable, or if CUDA index is
            out of range.

    Example:
        >>> validate_device("cuda:0")  # OK
        >>> validate_device("cuda", require_explicit_index=True)  # Raises ValueError
        >>> validate_device("cuda:999")  # Raises RuntimeError if only 2 GPUs
    """
    dev = parse_device(device)

    if dev.type != "cuda":
        return dev

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device '{device}' requested but CUDA is not available. "
            f"Use device='cpu' or check your CUDA installation."
        )

    if require_explicit_index and dev.index is None:
        raise ValueError(
            f"CUDA device '{device}' must include an explicit index like 'cuda:0'. "
            "Bare 'cuda' is ambiguous in multi-GPU contexts."
        )

    if dev.index is None:
        return dev  # Bare "cuda" uses the current default device

    available = torch.cuda.device_count()
    if dev.index >= available:
        raise RuntimeError(
            f"CUDA device '{device}' requested but only {available} device(s) are available. "
            f"Valid indices: cuda:0 through cuda:{available - 1}."
        )

    return dev


def _validate_device(device: str) -> None:
    """Validate requested device (backward-compatible wrapper).

    Deprecated: Use validate_device() directly for new code.
    """
    validate_device(device, require_explicit_index=False)


def create_model(
    task: "TaskSpec | str" = "cifar_baseline",
    device: str = "cuda",
    *,
    slots: list[str],
    permissive_gates: bool = True,
) -> "MorphogeneticModel":
    """Create a MorphogeneticModel for the given task on device.

    Args:
        task: Task specification or name.
        device: Target device for the model.
        slots: Seed slots to enable. Required and cannot be empty.
        permissive_gates: If True, quality gates only check structural requirements
            and let Tamiyo learn quality thresholds through reward signals.
    """
    if isinstance(task, str):
        from esper.runtime import get_task_spec  # Lazy import to avoid circular dependency
        task_spec = get_task_spec(task)
    else:
        task_spec = task

    _validate_device(device)

    if not slots:
        raise ValueError("slots cannot be empty")

    return task_spec.create_model(device=device, slots=slots, permissive_gates=permissive_gates)
