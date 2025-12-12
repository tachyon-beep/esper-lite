"""Factored Action Space for Multi-Slot Control.

The action space is factored into:
- SlotAction: which slot to target
- BlueprintAction: what blueprint to germinate
- BlendAction: which blending algorithm to use
- LifecycleOp: what operation to perform
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class SlotAction(IntEnum):
    """Target slot selection."""
    EARLY = 0
    MID = 1
    LATE = 2

    def to_slot_id(self) -> str:
        return ["early", "mid", "late"][self.value]


class BlueprintAction(IntEnum):
    """Blueprint selection for germination."""
    NOOP = 0
    CONV_ENHANCE = 1
    ATTENTION = 2
    NORM = 3
    DEPTHWISE = 4

    def to_blueprint_id(self) -> str | None:
        """Map to registered blueprint name.

        CONV_ENHANCE maps to "conv_light" (the registered blueprint).
        """
        mapping = {
            0: "noop",
            1: "conv_light",  # CONV_ENHANCE → conv_light (registered name)
            2: "attention",
            3: "norm",
            4: "depthwise",
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
    ADVANCE = 2
    CULL = 3
    FOSSILIZE = 4


@dataclass(frozen=True, slots=True)
class FactoredAction:
    """Composed action from factored components."""
    slot: SlotAction
    blueprint: BlueprintAction
    blend: BlendAction
    op: LifecycleOp

    @property
    def is_germinate(self) -> bool:
        return self.op == LifecycleOp.GERMINATE

    @property
    def is_advance(self) -> bool:
        return self.op == LifecycleOp.ADVANCE

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
    def slot_id(self) -> str:
        return self.slot.to_slot_id()

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
            slot=SlotAction(slot_idx),
            blueprint=BlueprintAction(blueprint_idx),
            blend=BlendAction(blend_idx),
            op=LifecycleOp(op_idx),
        )


# Dimension sizes for policy network
NUM_SLOTS = len(SlotAction)
NUM_BLUEPRINTS = len(BlueprintAction)
NUM_BLENDS = len(BlendAction)
NUM_OPS = len(LifecycleOp)


__all__ = [
    "SlotAction",
    "BlueprintAction",
    "BlendAction",
    "LifecycleOp",
    "FactoredAction",
    "NUM_SLOTS",
    "NUM_BLUEPRINTS",
    "NUM_BLENDS",
    "NUM_OPS",
]
