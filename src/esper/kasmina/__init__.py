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

__all__ = [
    # Re-exported Leyline types
    "SeedStage",
    "VALID_TRANSITIONS",
    "is_valid_transition",
    "is_terminal_stage",
    "is_active_stage",
    "is_failure_stage",
    "GateLevel",
    "GateResult",
    # Slot management
    "SeedMetrics",
    "SeedState",
    "QualityGates",
    "SeedSlot",
    # Blueprints / registry
    "BlueprintRegistry",
    "BlueprintSpec",
    "ConvBlock",
    # Isolation
    "blend_with_isolation",
    "GradientHealthMonitor",
    # Host
    "HostProtocol",
    "CNNHost",
    "TransformerHost",
    "TransformerBlock",
    "MorphogeneticModel",
    # Alpha scheduling
    "AlphaController",
]


from typing import Any


def __getattr__(name: str) -> Any:
    """Lazy import using PEP 562.

    Heavy modules (slot, blueprints, isolation, host with torch) are only
    loaded when accessed, not at package import time.
    """
    # Leyline re-exports (lightweight)
    if name in ("SeedStage", "VALID_TRANSITIONS", "is_valid_transition",
                "is_terminal_stage", "is_active_stage", "is_failure_stage",
                "GateLevel", "GateResult"):
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
        return mapping[name]

    # Slot (HEAVY - loads torch)
    if name in ("SeedMetrics", "SeedState", "QualityGates", "SeedSlot"):
        from esper.kasmina.slot import SeedMetrics, SeedState, QualityGates, SeedSlot
        return {"SeedMetrics": SeedMetrics, "SeedState": SeedState,
                "QualityGates": QualityGates, "SeedSlot": SeedSlot}[name]

    # Blueprints (HEAVY - loads torch)
    if name in ("BlueprintRegistry", "BlueprintSpec", "ConvBlock"):
        from esper.kasmina.blueprints import BlueprintRegistry, BlueprintSpec, ConvBlock
        return {"BlueprintRegistry": BlueprintRegistry, "BlueprintSpec": BlueprintSpec,
                "ConvBlock": ConvBlock}[name]

    # Isolation (HEAVY - loads torch)
    if name in ("blend_with_isolation", "GradientHealthMonitor"):
        from esper.kasmina.isolation import blend_with_isolation, GradientHealthMonitor
        return {"blend_with_isolation": blend_with_isolation,
                "GradientHealthMonitor": GradientHealthMonitor}[name]

    # Host (HEAVY - loads torch)
    if name in ("CNNHost", "TransformerHost", "TransformerBlock", "MorphogeneticModel"):
        from esper.kasmina.host import CNNHost, TransformerHost, TransformerBlock, MorphogeneticModel
        return {"CNNHost": CNNHost, "TransformerHost": TransformerHost,
                "TransformerBlock": TransformerBlock, "MorphogeneticModel": MorphogeneticModel}[name]

    # Protocol & Alpha (lightweight)
    if name == "HostProtocol":
        from esper.kasmina.protocol import HostProtocol
        return HostProtocol

    if name == "AlphaController":
        from esper.kasmina.alpha_controller import AlphaController
        return AlphaController

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
