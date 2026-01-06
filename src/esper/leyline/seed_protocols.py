"""Leyline Seed Protocols - Contracts for seed lifecycle operations.

These protocols decouple training infrastructure (simic) from seed
implementation details (kasmina) while enabling proper type checking
for seed slot operations.

The protocols capture the interfaces needed for:
- SeedStateProtocol: Individual seed lifecycle state
- SeedSlotProtocol: Slot container with active seed
- SlottedHostProtocol: Host model with seed slots
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Iterator, Protocol, runtime_checkable

if TYPE_CHECKING:
    import torch.nn as nn

    from esper.leyline.alpha import AlphaAlgorithm, AlphaCurve
    from esper.leyline.schemas import GateResult
    from esper.leyline.schemas import GateLevel
    from esper.leyline.stages import SeedStage


class SeedStateProtocol(Protocol):
    """Protocol for SeedState-like objects.

    Captures the interface used by training infrastructure when accessing
    seed lifecycle state. This enables simic to work with seed state without
    importing kasmina's concrete SeedState class.

    Implementations:
        - kasmina.slot.SeedState: Concrete implementation with full state management
    """

    seed_id: str
    metrics: Any  # SeedMetrics
    blueprint_id: str
    alpha_controller: Any  # AlphaController
    alpha_algorithm: Any  # AlphaAlgorithm
    alpha: float  # Current blend weight

    @property
    def stage(self) -> "SeedStage":
        """Current lifecycle stage of the seed."""
        ...

    @property
    def epochs_in_stage(self) -> int:
        """Number of epochs in the current lifecycle stage."""
        ...

    def can_transition_to(self, new_stage: "SeedStage") -> bool:
        """Check if transition to new stage is valid."""
        ...

    def sync_telemetry(
        self,
        gradient_norm: float | None = None,
        gradient_health: float | None = None,
        has_vanishing: bool | None = None,
        has_exploding: bool | None = None,
        epoch: int = 0,
        max_epochs: int = 25,
    ) -> None:
        """Synchronize internal state to telemetry fields.

        Call this once per epoch after validation to update telemetry.
        SeedMetrics remains the source of truth for accuracy/epoch data.

        Args:
            gradient_norm: Optional gradient norm from gradient stats collection.
            gradient_health: Optional gradient health metric (0-1, higher = healthier).
            has_vanishing: Optional flag indicating vanishing gradients detected.
            has_exploding: Optional flag indicating exploding gradients detected.
            epoch: Current epoch number.
            max_epochs: Maximum epochs for the training run.

        When gradient parameters are None, gradient-related telemetry fields
        are left at their default values (no gradient data available).
        """
        ...


@runtime_checkable
class SeedSlotProtocol(Protocol):
    """Protocol for SeedSlot-like objects.

    Captures the interface used when accessing seed slots through
    model.seed_slots[slot_id]. Decouples training infrastructure from
    the concrete kasmina.SeedSlot implementation.

    This enables proper type checking without circular imports.

    Implementations:
        - kasmina.slot.SeedSlot: Concrete slot with seed lifecycle management
    """

    # Mutable attributes for telemetry configuration
    fast_mode: bool
    telemetry_lifecycle_only: bool
    on_telemetry: Any  # Callable[[TelemetryEvent], None] | None
    isolate_gradients: bool
    telemetry_inner_epoch: int
    telemetry_global_epoch: int
    auto_forward_gates: frozenset["GateLevel"]

    @property
    def state(self) -> SeedStateProtocol | None:
        """Current seed state, or None if slot is empty."""
        ...

    @property
    def seed(self) -> "nn.Module | None":
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
    def alpha_schedule(self) -> "nn.Module | None":
        """Alpha schedule network (for GATE algorithm), or None."""
        ...

    def advance_stage(
        self, target_stage: "SeedStage | None" = None
    ) -> "GateResult":
        """Advance seed to next stage (or specific target stage).

        Returns gate result indicating success/failure.
        """
        ...

    def step_epoch(self) -> None:
        """Advance lifecycle mechanically once per epoch.

        Auto-prune events are signaled via `state.metrics.auto_pruned`.
        Callers should check (and clear) this flag immediately AFTER
        calling `step_epoch()` to catch both governor prunes and scheduled
        prune completion.

        Implementations may also auto-forward configured gated transitions
        as part of the per-epoch lifecycle tick.
        """
        ...

    def set_alpha(self, value: float) -> None:
        """Set the alpha value directly."""
        ...

    def schedule_prune(
        self,
        *,
        steps: int,
        curve: "AlphaCurve | None" = None,
        steepness: float = 12.0,
        reason: str = "",
        initiator: str = "policy",
    ) -> bool:
        """Schedule a prune by ramping alpha down to 0."""
        ...

    def set_alpha_target(
        self,
        *,
        alpha_target: float,
        steps: int,
        curve: "AlphaCurve | None" = None,
        steepness: float = 12.0,
        alpha_algorithm: "AlphaAlgorithm | None" = None,
        initiator: str = "policy",
    ) -> bool:
        """Retarget alpha to a non-zero target from HOLD mode."""
        ...

    @contextmanager
    def force_alpha(self, value: float) -> Iterator[None]:
        """Temporarily override alpha for counterfactual evaluation.

        Used for differential validation to measure true seed contribution.
        """
        ...


class SlottedHostProtocol(Protocol):
    """Protocol for host models with seed slots.

    Captures the interface needed by training infrastructure for models
    that support seed slot operations (like kasmina.MorphogeneticModel).

    This enables simic to work with slotted hosts without importing
    the concrete MorphogeneticModel class.

    Implementations:
        - kasmina.host.MorphogeneticModel: Concrete host with slot management
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

    def get_host_parameters(self) -> "Iterator[nn.Parameter]":
        """Get host parameters (excluding seed parameters)."""
        ...

    def get_seed_parameters(
        self, slot: str | None = None
    ) -> "Iterator[nn.Parameter]":
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
