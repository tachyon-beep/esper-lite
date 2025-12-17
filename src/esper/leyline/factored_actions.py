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
]
