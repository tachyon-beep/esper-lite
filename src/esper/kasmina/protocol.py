"""Kasmina Protocol - Structural typing for pluggable hosts.

Hosts declare where seeds can be planted (injection_points),
accept seed modules (register_slot), and handle their own
forward pass including calling any attached slots.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn
from torch import Tensor


@runtime_checkable
class HostProtocol(Protocol):
    """Contract for graftable host networks."""

    @property
    def injection_points(self) -> dict[str, int]:
        """Map of slot_id -> channel/embedding dimension."""
        ...

    def register_slot(self, slot_id: str, slot: nn.Module) -> None:
        """Attach a seed module at the specified injection point."""
        ...

    def unregister_slot(self, slot_id: str) -> None:
        """Remove a seed module from the specified injection point."""
        ...

    def forward(self, x: Tensor) -> Tensor:
        """Full forward pass, including any attached slots."""
        ...


__all__ = ["HostProtocol"]
