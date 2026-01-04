"""Kasmina - Seed mechanics for Esper.

Kasmina manages the lifecycle of seed modules:
- Germination: Creating new seeds from blueprints
- Training: Growing seeds with gradient flow
- Blending: Integrating seeds with the host
- Fossilization: Permanent integration

Named after Kasmina, Planeswalker of Secrets - master of hidden knowledge
and the art of subtle manipulation.

NOTE: This module uses PEP 562 lazy imports. Heavy modules (slot, blueprints,
isolation, host with torch dependencies) are only loaded when accessed.
"""

from typing import TYPE_CHECKING, Any

_LEYLINE_EXPORTS = (
    "SeedStage",
    "VALID_TRANSITIONS",
    "is_valid_transition",
    "is_terminal_stage",
    "is_active_stage",
    "is_failure_stage",
    "GateLevel",
    "GateResult",
)
_SLOT_EXPORTS = ("SeedMetrics", "SeedState", "QualityGates", "SeedSlot")
_BLUEPRINT_EXPORTS = ("BlueprintRegistry", "BlueprintSpec", "ConvBlock")
_ISOLATION_EXPORTS = ("blend_with_isolation", "GradientHealthMonitor")
_HOST_EXPORTS = ("CNNHost", "TransformerHost", "TransformerBlock", "MorphogeneticModel")
_PROTOCOL_EXPORTS = ("HostProtocol",)
_ALPHA_EXPORTS = ("AlphaController",)

__all__ = [
    *_LEYLINE_EXPORTS,
    *_SLOT_EXPORTS,
    *_BLUEPRINT_EXPORTS,
    *_ISOLATION_EXPORTS,
    *_PROTOCOL_EXPORTS,
    *_HOST_EXPORTS,
    *_ALPHA_EXPORTS,
]


# TYPE_CHECKING imports for static analysis (mypy, IDE navigation).
# These are never executed at runtime, preserving lazy import semantics.
if TYPE_CHECKING:
    # Leyline re-exports (lightweight at runtime too)
    from esper.leyline import (
        GateLevel as GateLevel,
        GateResult as GateResult,
        SeedStage as SeedStage,
        VALID_TRANSITIONS as VALID_TRANSITIONS,
        is_active_stage as is_active_stage,
        is_failure_stage as is_failure_stage,
        is_terminal_stage as is_terminal_stage,
        is_valid_transition as is_valid_transition,
    )

    # Slot management (HEAVY at runtime - loads torch)
    from esper.kasmina.slot import (
        QualityGates as QualityGates,
        SeedMetrics as SeedMetrics,
        SeedSlot as SeedSlot,
        SeedState as SeedState,
    )

    # Blueprints (HEAVY at runtime - loads torch)
    from esper.kasmina.blueprints import (
        BlueprintRegistry as BlueprintRegistry,
        BlueprintSpec as BlueprintSpec,
        ConvBlock as ConvBlock,
    )

    # Isolation (HEAVY at runtime - loads torch)
    from esper.kasmina.isolation import (
        GradientHealthMonitor as GradientHealthMonitor,
        blend_with_isolation as blend_with_isolation,
    )

    # Host (HEAVY at runtime - loads torch)
    from esper.kasmina.host import (
        CNNHost as CNNHost,
        MorphogeneticModel as MorphogeneticModel,
        TransformerBlock as TransformerBlock,
        TransformerHost as TransformerHost,
    )

    # Protocol & Alpha (lightweight at runtime)
    from esper.kasmina.alpha_controller import AlphaController as AlphaController
    from esper.leyline import HostProtocol as HostProtocol


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))


def __getattr__(name: str) -> Any:
    """Lazy import using PEP 562.

    Heavy modules (slot, blueprints, isolation, host with torch) are only
    loaded when accessed, not at package import time.
    """
    # Leyline re-exports (lightweight)
    if name in _LEYLINE_EXPORTS:
        from esper.leyline import (
            SeedStage,
            VALID_TRANSITIONS,
            is_valid_transition,
            is_terminal_stage,
            is_active_stage,
            is_failure_stage,
            GateLevel,
            GateResult,
        )
        mapping: dict[str, Any] = {
            "SeedStage": SeedStage,
            "VALID_TRANSITIONS": VALID_TRANSITIONS,
            "is_valid_transition": is_valid_transition,
            "is_terminal_stage": is_terminal_stage,
            "is_active_stage": is_active_stage,
            "is_failure_stage": is_failure_stage,
            "GateLevel": GateLevel,
            "GateResult": GateResult,
        }
        globals().update(mapping)
        return mapping[name]

    # Slot (HEAVY - loads torch)
    if name in _SLOT_EXPORTS:
        from esper.kasmina.slot import SeedMetrics, SeedState, QualityGates, SeedSlot
        slot_exports: dict[str, Any] = {
            "SeedMetrics": SeedMetrics,
            "SeedState": SeedState,
            "QualityGates": QualityGates,
            "SeedSlot": SeedSlot,
        }
        globals().update(slot_exports)
        return slot_exports[name]

    # Blueprints (HEAVY - loads torch)
    if name in _BLUEPRINT_EXPORTS:
        from esper.kasmina.blueprints import BlueprintRegistry, BlueprintSpec, ConvBlock
        blueprint_exports: dict[str, Any] = {
            "BlueprintRegistry": BlueprintRegistry,
            "BlueprintSpec": BlueprintSpec,
            "ConvBlock": ConvBlock,
        }
        globals().update(blueprint_exports)
        return blueprint_exports[name]

    # Isolation (HEAVY - loads torch)
    if name in _ISOLATION_EXPORTS:
        from esper.kasmina.isolation import blend_with_isolation, GradientHealthMonitor
        isolation_exports: dict[str, Any] = {
            "blend_with_isolation": blend_with_isolation,
            "GradientHealthMonitor": GradientHealthMonitor,
        }
        globals().update(isolation_exports)
        return isolation_exports[name]

    # Protocol (lightweight - now from leyline)
    if name in _PROTOCOL_EXPORTS:
        from esper.leyline import HostProtocol
        globals()["HostProtocol"] = HostProtocol
        return HostProtocol

    # Host (HEAVY - loads torch)
    if name in _HOST_EXPORTS:
        from esper.kasmina.host import CNNHost, TransformerHost, TransformerBlock, MorphogeneticModel
        host_exports: dict[str, Any] = {
            "CNNHost": CNNHost,
            "TransformerHost": TransformerHost,
            "TransformerBlock": TransformerBlock,
            "MorphogeneticModel": MorphogeneticModel,
        }
        globals().update(host_exports)
        return host_exports[name]

    # Alpha (lightweight)
    if name in _ALPHA_EXPORTS:
        from esper.kasmina.alpha_controller import AlphaController
        globals()["AlphaController"] = AlphaController
        return AlphaController

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
