"""InjectionSpec - Describes a host injection point for seeds.

This is the contract between hosts (which define injection points) and
the rest of the system (which needs to know about available slots).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InjectionSpec:
    """Specification for a single injection point in a host network.

    Attributes:
        slot_id: Canonical slot ID (e.g., "r0c0", "r0c1")
        channels: Channel/embedding dimension at this injection point
        position: Relative position in network (0.0 = input, 1.0 = output)
        layer_range: Tuple of (start_layer, end_layer) this slot covers
    """

    slot_id: str
    channels: int
    position: float
    layer_range: tuple[int, int]

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


__all__ = ["InjectionSpec"]
