"""InjectionSpec - Describes a host injection point for seeds.

This is the contract between hosts (which define injection points) and
the rest of the system (which needs to know about available slots).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class SurfaceType(IntEnum):
    """Type of injection surface in the network.

    Surfaces define semantically meaningful boundaries where seeds can attach.
    The enum values are stable and determine action space ordering when
    combined with block/layer indices.

    CNN surfaces:
        BLOCK_END: Legacy/default - after entire block (conv + pool)
        PRE_POOL: After convolution, before pooling (high-resolution features)
        POST_POOL: After pooling (reduced-resolution features)

    Transformer surfaces (Phase 1):
        POST_ATTN: After attention sublayer
        POST_MLP: After MLP sublayer
    """

    BLOCK_END = 0  # Legacy: after entire block (backward compatible)
    PRE_POOL = 1  # CNN: after conv, before pool
    POST_POOL = 2  # CNN: after pool
    POST_ATTN = 3  # Transformer: after attention
    POST_MLP = 4  # Transformer: after MLP


@dataclass(frozen=True)
class InjectionSpec:
    """Specification for a single injection point in a host network.

    Attributes:
        slot_id: Canonical slot ID (e.g., "r0c0", "r0c1")
        channels: Channel/embedding dimension at this injection point
        position: Relative position in network (0.0 = input, 1.0 = output).
            Used for visualization only; sorting uses `order`.
        layer_range: Tuple of (start_layer, end_layer) this slot covers
        surface: Type of surface at this injection point (semantic meaning)
        order: Strictly increasing integer for action index stability. Sorting by order
            produces row-major slot ordering (all row 0 slots first, then row 1, etc.).
            This is used by SlotConfig to derive stable action indices.
            For forward pass execution order, use HostProtocol.execution_order().
        row: Explicit grid row coordinate (derived from slot_id if not provided)
        col: Explicit grid column coordinate (derived from slot_id if not provided)

    Note on spatial dimensions:
        This spec intentionally does NOT include spatial dimensions (H, W).
        Blueprint factories MUST create seeds that are resolution-agnostic
        (e.g., using fully-convolutional or global-pooling architectures).
        PRE_POOL and POST_POOL injection points at the same block will have
        the same channels but different spatial dimensions.
    """

    slot_id: str
    channels: int
    position: float
    layer_range: tuple[int, int]
    surface: SurfaceType = SurfaceType.BLOCK_END
    order: int = 0
    row: int | None = None
    col: int | None = None

    def __post_init__(self) -> None:
        if self.channels <= 0:
            raise ValueError(f"channels must be positive, got {self.channels}")
        if not (0.0 <= self.position <= 1.0):
            raise ValueError(f"position must be between 0 and 1, got {self.position}")
        start, end = self.layer_range
        if start > end:
            raise ValueError(f"layer_range start ({start}) must be <= end ({end})")
        if start < 0:
            raise ValueError(f"layer_range start must be non-negative, got {start}")
        if self.order < 0:
            raise ValueError(f"order must be non-negative, got {self.order}")


__all__ = ["InjectionSpec", "SurfaceType"]
