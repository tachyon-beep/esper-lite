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
        device: Device string like "cpu", "cuda", "cuda:0", "mps".
        require_explicit_index: If True, require explicit CUDA index (e.g., "cuda:0"
            instead of bare "cuda"). Use for multi-GPU training where device
            assignment must be unambiguous.

    Returns:
        Parsed torch.device object.

    Raises:
        ValueError: If device string is malformed, unsupported device type,
            or violates require_explicit_index.
        RuntimeError: If CUDA/MPS is requested but unavailable, or if CUDA index is
            out of range.

    Example:
        >>> validate_device("cuda:0")  # OK
        >>> validate_device("cuda", require_explicit_index=True)  # Raises ValueError
        >>> validate_device("cuda:999")  # Raises RuntimeError if only 2 GPUs
        >>> validate_device("mps")  # OK on Apple Silicon, error otherwise
        >>> validate_device("mps:0")  # OK - explicit index 0 is valid
        >>> validate_device("mps:1")  # Raises ValueError - MPS only has one device
        >>> validate_device("meta")  # Raises ValueError - unsupported device type
    """
    from esper.leyline import SUPPORTED_DEVICE_TYPES

    dev = parse_device(device)
    explicit_index: int | None = None
    if ":" in device:
        _, maybe_index = device.split(":", 1)
        if maybe_index.isdigit():
            explicit_index = int(maybe_index)

    # Fail-fast on unsupported device types (meta, xla, xpu, hpu, etc.)
    # These are valid PyTorch devices but not supported for Esper training.
    if dev.type not in SUPPORTED_DEVICE_TYPES:
        raise ValueError(
            f"Unsupported device type '{dev.type}' (from '{device}'). "
            f"Esper supports: {', '.join(sorted(SUPPORTED_DEVICE_TYPES))}. "
            f"Device types like 'meta', 'xla', 'xpu' are not supported for training."
        )

    # MPS validation (Apple Silicon)
    # Unlike CUDA, MPS only supports a single device (the Apple Silicon GPU).
    # PyTorch accepts "mps:1" syntactically but it's not a valid device.
    if dev.type == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                f"MPS device '{device}' requested but MPS is not available. "
                f"Use device='cpu' or check your Apple Silicon configuration."
            )
        if explicit_index not in (None, 0):
            raise ValueError(
                f"Invalid MPS device index in '{device}'. "
                f"MPS only supports a single device: use 'mps' or 'mps:0'."
            )
        return dev

    if dev.type != "cuda":
        return dev

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device '{device}' requested but CUDA is not available. "
            f"Use device='cpu' or check your CUDA installation."
        )

    if require_explicit_index and explicit_index is None:
        raise ValueError(
            f"CUDA device '{device}' must include an explicit index like 'cuda:0'. "
            "Bare 'cuda' is ambiguous in multi-GPU contexts."
        )

    if explicit_index is None:
        return dev  # Bare "cuda" uses the current default device

    available = torch.cuda.device_count()
    if explicit_index >= available:
        raise RuntimeError(
            f"CUDA device '{device}' requested but only {available} device(s) are available. "
            f"Valid indices: cuda:0 through cuda:{available - 1}."
        )

    # PyTorch device indices use a DeviceIndex type (int16 in PyTorch 2.x, was int8
    # in older versions) with -1 sentinel for "no index". Very large indices can
    # overflow and appear negative or wrap around, which would bypass range checks
    # if we only trust dev.index. This explicit comparison catches such overflows.
    if dev.index != explicit_index:
        raise ValueError(
            f"Invalid CUDA device '{device}': parsed as {dev!s}. "
            "This indicates index overflow in torch.device parsing."
        )

    return dev


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

    validated_device = validate_device(device, require_explicit_index=False)

    if not slots:
        raise ValueError("slots cannot be empty")

    if len(slots) != len(set(slots)):
        duplicates = [s for s in slots if slots.count(s) > 1]
        raise ValueError(f"slots contains duplicates: {sorted(set(duplicates))}")

    # Pass the validated device as a string for consistency with TaskSpec.create_model signature
    return task_spec.create_model(
        device=str(validated_device), slots=slots, permissive_gates=permissive_gates
    )
