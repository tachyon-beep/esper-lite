"""Shared contracts and protocols for Simic subsystem.

Defines protocols to decouple Simic from Kasmina implementation details
while enabling proper type checking for seed slot operations.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Iterator, Protocol, runtime_checkable

import torch.nn as nn

if TYPE_CHECKING:
    from esper.leyline import GateResult, SeedStage


class SeedStateProtocol(Protocol):
    """Protocol for SeedState-like objects.

    Captures the interface used by Simic when accessing seed state.
    """

    @property
    def stage(self) -> "SeedStage":
        """Current lifecycle stage of the seed."""
        ...


@runtime_checkable
class SeedSlotProtocol(Protocol):
    """Protocol for SeedSlot-like objects.

    Captures the interface used by Simic when accessing seed slots
    through model.seed_slots[slot_id]. Decouples Simic from the
    concrete kasmina.SeedSlot implementation.

    This enables proper type checking without circular imports.
    """

    @property
    def state(self) -> SeedStateProtocol | None:
        """Current seed state, or None if slot is empty."""
        ...

    @property
    def seed(self) -> nn.Module | None:
        """Active seed module, or None if slot is empty."""
        ...

    @property
    def active_seed_params(self) -> int:
        """Number of trainable parameters in active seed, or 0 if no seed."""
        ...

    def advance_stage(self, target_stage: "SeedStage | None" = None) -> "GateResult":
        """Advance seed to next stage (or specific target stage).

        Returns gate result indicating success/failure.
        """
        ...

    def step_epoch(self) -> bool:
        """Advance lifecycle mechanically once per epoch.

        Returns:
            True if an auto-prune occurred, False otherwise.
        """
        ...

    @contextmanager
    def force_alpha(self, value: float) -> Iterator[None]:
        """Temporarily override alpha for counterfactual evaluation.

        Used for differential validation to measure true seed contribution.
        """
        ...


class SlottedHostProtocol(Protocol):
    """Protocol for host models with seed slots.

    Captures the interface needed by Simic training for models
    that support seed slot operations (like kasmina.SlottedHost).
    """

    @property
    def seed_slots(self) -> dict[str, SeedSlotProtocol]:
        """Dictionary of slot_id -> SeedSlot for all configured slots."""
        ...

    def __call__(self, x: Any) -> Any:
        """Forward pass through the model."""
        ...

    def parameters(self, recurse: bool = True) -> Any:
        """Return model parameters."""
        ...

    def to(self, *args: Any, **kwargs: Any) -> "SlottedHostProtocol":
        """Move model to device."""
        ...

    def train(self, mode: bool = True) -> "SlottedHostProtocol":
        """Set training mode."""
        ...

    def eval(self) -> "SlottedHostProtocol":
        """Set evaluation mode."""
        ...


__all__ = [
    "SeedStateProtocol",
    "SeedSlotProtocol",
    "SlottedHostProtocol",
]
