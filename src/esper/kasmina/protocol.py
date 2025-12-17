"""Kasmina Protocol - Structural typing for pluggable hosts.

Hosts are pure backbone networks that provide segment routing.
Slot management is handled by MorphogeneticModel, not hosts directly.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import Tensor


@runtime_checkable
class HostProtocol(Protocol):
    """Contract for graftable host networks.

    Hosts are pure backbone networks that provide:
    - injection_points: Available segment boundaries for seed attachment
    - segment_channels: Channel dimensions at each boundary
    - forward_to_segment/forward_from_segment: Segment routing for MorphogeneticModel
    - forward: Standard backbone forward pass (no slot application)

    Slot management is handled by MorphogeneticModel, not hosts directly.
    """

    @property
    def injection_points(self) -> dict[str, int]:
        """Map of slot_id -> channel/embedding dimension."""
        ...

    @property
    def segment_channels(self) -> dict[str, int]:
        """Map of canonical slot_id -> channel dimension."""
        ...

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through backbone (no slot application)."""
        ...

    def forward_to_segment(self, segment: str, x: Tensor, from_segment: str | None = None) -> Tensor:
        """Forward from input or segment to target segment boundary."""
        ...

    def forward_from_segment(self, segment: str, x: Tensor) -> Tensor:
        """Forward from segment boundary to output."""
        ...


__all__ = ["HostProtocol"]
