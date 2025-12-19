"""Factored Action Space for Multi-Slot Control.

The action space is factored into:
- slot_idx: which slot to target (integer index)
- BlueprintAction: what blueprint to germinate
- BlendAction: which blending algorithm to use
- LifecycleOp: what operation to perform
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class BlueprintAction(IntEnum):
    """Blueprint selection for germination."""
    NOOP = 0
    CONV_LIGHT = 1  # Matches registry name "conv_light"
    ATTENTION = 2
    NORM = 3
    DEPTHWISE = 4
    BOTTLENECK = 5
    CONV_SMALL = 6
    CONV_HEAVY = 7
    LORA = 8
    LORA_LARGE = 9
    MLP_SMALL = 10
    MLP = 11
    FLEX_ATTENTION = 12

    def to_blueprint_id(self) -> str | None:
        """Map to registered blueprint name."""
        mapping = {
            0: "noop",
            1: "conv_light",
            2: "attention",
            3: "norm",
            4: "depthwise",
            5: "bottleneck",
            6: "conv_small",
            7: "conv_heavy",
            8: "lora",
            9: "lora_large",
            10: "mlp_small",
            11: "mlp",
            12: "flex_attention",
        }
        return mapping.get(self.value)


class BlendAction(IntEnum):
    """Blending algorithm selection."""
    LINEAR = 0
    SIGMOID = 1
    GATED = 2

    def to_algorithm_id(self) -> str:
        return ["linear", "sigmoid", "gated"][self.value]


class LifecycleOp(IntEnum):
    """Lifecycle operation."""
    WAIT = 0
    GERMINATE = 1
    CULL = 2       # was 3
    FOSSILIZE = 3  # was 4


# =============================================================================
# Lookup Tables for Hot Path Optimization
# =============================================================================
# These tables enable direct indexing without creating FactoredAction objects.
# CRITICAL: These must stay in sync with enum definitions above.
# Module-level assertions validate sync at import time.

# Operation name lookup (matches LifecycleOp enum order)
OP_NAMES: tuple[str, ...] = tuple(op.name for op in LifecycleOp)

# Blueprint ID lookup (matches BlueprintAction.to_blueprint_id())
BLUEPRINT_IDS: tuple[str | None, ...] = tuple(bp.to_blueprint_id() for bp in BlueprintAction)

# Blend algorithm ID lookup (matches BlendAction.to_algorithm_id())
BLEND_IDS: tuple[str, ...] = tuple(blend.to_algorithm_id() for blend in BlendAction)

# Operation index constants for direct comparison (avoids enum construction)
OP_WAIT: int = LifecycleOp.WAIT.value
OP_GERMINATE: int = LifecycleOp.GERMINATE.value
OP_CULL: int = LifecycleOp.CULL.value
OP_FOSSILIZE: int = LifecycleOp.FOSSILIZE.value

# Module-level validation: catch enum drift at import time
assert OP_NAMES == tuple(op.name for op in LifecycleOp), (
    "OP_NAMES out of sync with LifecycleOp enum - this is a bug"
)
assert len(BLUEPRINT_IDS) == len(BlueprintAction), (
    "BLUEPRINT_IDS length mismatch with BlueprintAction enum"
)
assert len(BLEND_IDS) == len(BlendAction), (
    "BLEND_IDS length mismatch with BlendAction enum"
)


@dataclass(frozen=True, slots=True)
class FactoredAction:
    """Composed action from factored components."""
    slot_idx: int
    blueprint: BlueprintAction
    blend: BlendAction
    op: LifecycleOp

    @property
    def is_germinate(self) -> bool:
        return self.op == LifecycleOp.GERMINATE

    @property
    def is_cull(self) -> bool:
        return self.op == LifecycleOp.CULL

    @property
    def is_fossilize(self) -> bool:
        return self.op == LifecycleOp.FOSSILIZE

    @property
    def is_wait(self) -> bool:
        return self.op == LifecycleOp.WAIT

    @property
    def blueprint_id(self) -> str | None:
        return self.blueprint.to_blueprint_id()

    @property
    def blend_algorithm_id(self) -> str:
        return self.blend.to_algorithm_id()

    @classmethod
    def from_indices(
        cls,
        slot_idx: int,
        blueprint_idx: int,
        blend_idx: int,
        op_idx: int,
    ) -> "FactoredAction":
        return cls(
            slot_idx=slot_idx,
            blueprint=BlueprintAction(blueprint_idx),
            blend=BlendAction(blend_idx),
            op=LifecycleOp(op_idx),
        )


# Dimension sizes for policy network
NUM_BLUEPRINTS = len(BlueprintAction)
NUM_BLENDS = len(BlendAction)
NUM_OPS = len(LifecycleOp)


# Valid blueprints per topology (for action masking)
CNN_BLUEPRINTS = frozenset({
    BlueprintAction.NOOP,
    BlueprintAction.NORM,
    BlueprintAction.ATTENTION,
    BlueprintAction.CONV_LIGHT,
    BlueprintAction.DEPTHWISE,
    BlueprintAction.BOTTLENECK,
    BlueprintAction.CONV_SMALL,
    BlueprintAction.CONV_HEAVY,
})

TRANSFORMER_BLUEPRINTS = frozenset({
    BlueprintAction.NOOP,
    BlueprintAction.NORM,
    BlueprintAction.ATTENTION,
    BlueprintAction.LORA,
    BlueprintAction.LORA_LARGE,
    BlueprintAction.MLP_SMALL,
    BlueprintAction.MLP,
    BlueprintAction.FLEX_ATTENTION,
})


__all__ = [
    "BlueprintAction",
    "BlendAction",
    "LifecycleOp",
    "FactoredAction",
    "NUM_BLUEPRINTS",
    "NUM_BLENDS",
    "NUM_OPS",
    "CNN_BLUEPRINTS",
    "TRANSFORMER_BLUEPRINTS",
    # Lookup tables for hot path optimization
    "OP_NAMES",
    "BLUEPRINT_IDS",
    "BLEND_IDS",
    "OP_WAIT",
    "OP_GERMINATE",
    "OP_CULL",
    "OP_FOSSILIZE",
]
