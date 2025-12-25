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

    seed_id: str
    metrics: Any  # SeedMetrics
    blueprint_id: str
    alpha_controller: Any  # AlphaController
    alpha_algorithm: Any  # AlphaAlgorithm

    @property
    def stage(self) -> "SeedStage":
        """Current lifecycle stage of the seed."""
        ...

    def can_transition_to(self, new_stage: "SeedStage") -> bool:
        """Check if transition to new stage is valid."""
        ...

    def sync_telemetry(self) -> None:
        """Synchronize internal state to telemetry fields."""
        ...


@runtime_checkable
class SeedSlotProtocol(Protocol):
    """Protocol for SeedSlot-like objects.

    Captures the interface used by Simic when accessing seed slots
    through model.seed_slots[slot_id]. Decouples Simic from the
    concrete kasmina.SeedSlot implementation.

    This enables proper type checking without circular imports.
    """

    # Mutable attributes for telemetry configuration
    fast_mode: bool
    telemetry_lifecycle_only: bool
    on_telemetry: Any  # Callable[[TelemetryEvent], None] | None
    isolate_gradients: bool
    telemetry_inner_epoch: int
    telemetry_global_epoch: int

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

    @property
    def alpha(self) -> float:
        """Current alpha (blend weight) value."""
        ...

    @property
    def alpha_schedule(self) -> nn.Module | None:
        """Alpha schedule network (for GATE algorithm), or None."""
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

    def set_alpha(self, value: float) -> None:
        """Set the alpha value directly."""
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
    def seed_slots(self) -> Any:
        """Mapping of slot_id -> SeedSlot (ModuleDict or dict)."""
        ...

    @property
    def active_seed_params(self) -> int:
        """Total number of trainable parameters in all active seeds."""
        ...

    @property
    def has_active_seed(self) -> bool:
        """Whether any slot has an active seed."""
        ...

    def has_active_seed_in_slot(self, slot_id: str) -> bool:
        """Check if a specific slot has an active seed."""
        ...

    def germinate_seed(
        self,
        blueprint_id: str,
        seed_id: str,
        *,
        slot: str,
        blend_algorithm_id: str = ...,
        blend_tempo_epochs: int = ...,
        alpha_algorithm: Any = ...,  # AlphaAlgorithm
        alpha_target: float | None = ...,
    ) -> None:
        """Create a new seed in the specified slot."""
        ...

    def prune_seed(self, *, slot: str) -> None:
        """Remove the seed from the specified slot."""
        ...

    def get_host_parameters(self) -> Iterator[nn.Parameter]:
        """Get host parameters (excluding seed parameters)."""
        ...

    def get_seed_parameters(self, slot: str | None = None) -> Iterator[nn.Parameter]:
        """Get seed parameters, optionally filtered by slot."""
        ...

    def __call__(self, x: Any) -> Any:
        """Forward pass through the model."""
        ...

    def parameters(self, recurse: bool = True) -> Any:
        """Return model parameters."""
        ...

    def to(self, *args: Any, **kwargs: Any) -> Any:
        """Move model to device."""
        ...

    def train(self, mode: bool = True) -> Any:
        """Set training mode."""
        ...

    def eval(self) -> Any:
        """Set evaluation mode."""
        ...


__all__ = [
    "SeedStateProtocol",
    "SeedSlotProtocol",
    "SlottedHostProtocol",
]
