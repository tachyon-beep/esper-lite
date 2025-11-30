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
from esper.kasmina.slot import (
    SeedMetrics,
    SeedState,
    QualityGates,
    SeedSlot,
)

# Blueprints (v1 - will be replaced by registry in Phase 2)
from esper.kasmina._blueprints_v1 import (
    ConvBlock,
    ConvEnhanceSeed,
    AttentionSeed,
    NormSeed,
    DepthwiseSeed,
    BlueprintCatalog,
)

# Isolation
from esper.kasmina.isolation import (
    AlphaSchedule,
    blend_with_isolation,
    GradientIsolationMonitor,
)

# Host
from esper.kasmina.host import (
    HostCNN,
    MorphogeneticModel,
)

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
    # Blueprints
    "ConvBlock",
    "ConvEnhanceSeed",
    "AttentionSeed",
    "NormSeed",
    "DepthwiseSeed",
    "BlueprintCatalog",
    # Isolation
    "AlphaSchedule",
    "blend_with_isolation",
    "GradientIsolationMonitor",
    # Host
    "HostCNN",
    "MorphogeneticModel",
]
