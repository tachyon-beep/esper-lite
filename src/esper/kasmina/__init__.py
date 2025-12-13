"""Kasmina - Seed mechanics for Esper.

Kasmina manages the lifecycle of seed modules:
- Germination: Creating new seeds from blueprints
- Training: Growing seeds with gradient flow
- Blending: Integrating seeds with the host
- Fossilization: Permanent integration

Named after Kasmina, Planeswalker of Secrets - master of hidden knowledge
and the art of subtle manipulation.
"""

# Re-export Leyline types that Kasmina uses
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

# Slot management
from esper.kasmina.slot import SeedMetrics, SeedState, QualityGates, SeedSlot

# Blueprints / registry
from esper.kasmina.blueprints import BlueprintRegistry, BlueprintSpec, ConvBlock

# Isolation
from esper.kasmina.isolation import blend_with_isolation, GradientIsolationMonitor

# Host
from esper.kasmina.protocol import HostProtocol
from esper.kasmina.host import CNNHost, TransformerHost, TransformerBlock, MorphogeneticModel

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
    "GradientIsolationMonitor",
    # Host
    "HostProtocol",
    "CNNHost",
    "TransformerHost",
    "TransformerBlock",
    "MorphogeneticModel",
]
