"""Factored Action Space for Multi-Slot Control.

The action space is factored into:
- slot_idx: which slot to target (integer index)
- BlueprintAction: what blueprint to germinate
- GerminationStyle: which germination style (blend + alpha algorithm) to use
- LifecycleOp: what operation to perform
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from esper.leyline.slot_config import SlotConfig
from esper.leyline.alpha import AlphaAlgorithm, AlphaCurve


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


class GerminationStyle(IntEnum):
    """Germination style (composite decision: blend + alpha algorithm).

    This eliminates invalid blend/alpha_algorithm combinations by making them
    unrepresentable at the action space level.
    """

    LINEAR_ADD = 0
    LINEAR_MULTIPLY = 1
    SIGMOID_ADD = 2
    GATED_GATE = 3


STYLE_TO_KASMINA: dict[GerminationStyle, tuple[str, AlphaAlgorithm]] = {
    GerminationStyle.LINEAR_ADD: ("linear", AlphaAlgorithm.ADD),
    GerminationStyle.LINEAR_MULTIPLY: ("linear", AlphaAlgorithm.MULTIPLY),
    GerminationStyle.SIGMOID_ADD: ("sigmoid", AlphaAlgorithm.ADD),
    GerminationStyle.GATED_GATE: ("gated", AlphaAlgorithm.GATE),
}


class TempoAction(IntEnum):
    """Blending tempo selection.

    Controls how many epochs the alpha ramp takes during BLENDING stage.
    Selected at GERMINATE time, stored in SeedState, used by start_blending().

    Design rationale:
    - FAST: Rapid integration, quick signal, higher risk of instability
    - STANDARD: Current default (5 epochs), balanced approach
    - SLOW: Gradual integration, better stability assessment, longer investment
    """
    FAST = 0      # 3 epochs
    STANDARD = 1  # 5 epochs (current default)
    SLOW = 2      # 8 epochs


# Mapping from enum to actual epoch count
TEMPO_TO_EPOCHS: dict[TempoAction, int] = {
    TempoAction.FAST: 3,
    TempoAction.STANDARD: 5,
    TempoAction.SLOW: 8,
}

# Module-level constant for action space sizing (follows NUM_BLUEPRINTS pattern)
NUM_TEMPO: int = len(TempoAction)


class LifecycleOp(IntEnum):
    """Lifecycle operation."""
    WAIT = 0
    GERMINATE = 1
    SET_ALPHA_TARGET = 2
    PRUNE = 3
    FOSSILIZE = 4
    ADVANCE = 5


class AlphaTargetAction(IntEnum):
    """Discrete alpha target selection (non-zero)."""
    HALF = 0
    SEVENTY = 1
    FULL = 2

    def to_target(self) -> float:
        return ALPHA_TARGET_VALUES[self.value]


class AlphaSpeedAction(IntEnum):
    """Alpha schedule speed selection."""
    INSTANT = 0
    FAST = 1
    MEDIUM = 2
    SLOW = 3

    def to_steps(self) -> int:
        return ALPHA_SPEED_TO_STEPS[self]


class AlphaCurveAction(IntEnum):
    """Alpha schedule curve selection.

    Sigmoid variants control steepness (transition sharpness):
    - SIGMOID_GENTLE (steepness=6): Gradual S-curve, smooth transition
    - SIGMOID (steepness=12): Standard S-curve (default)
    - SIGMOID_SHARP (steepness=24): Steep S-curve, near-step transition
    """
    LINEAR = 0
    COSINE = 1
    SIGMOID_GENTLE = 2
    SIGMOID = 3
    SIGMOID_SHARP = 4

    def to_curve(self) -> AlphaCurve:
        """Return the underlying AlphaCurve enum value."""
        return ALPHA_CURVE_TO_CURVE[self]

    def to_steepness(self) -> float:
        """Return sigmoid steepness (only meaningful for SIGMOID variants)."""
        return ALPHA_CURVE_TO_STEEPNESS[self]


# Alpha target values (non-zero targets only; removal uses PRUNE)
ALPHA_TARGET_VALUES: tuple[float, ...] = (0.5, 0.7, 1.0)

# Alpha speed mapping (controller ticks)
ALPHA_SPEED_TO_STEPS: dict[AlphaSpeedAction, int] = {
    AlphaSpeedAction.INSTANT: 0,
    AlphaSpeedAction.FAST: 3,
    AlphaSpeedAction.MEDIUM: 5,
    AlphaSpeedAction.SLOW: 8,
}

# Alpha curve mapping (action -> AlphaCurve enum)
ALPHA_CURVE_TO_CURVE: dict[AlphaCurveAction, AlphaCurve] = {
    AlphaCurveAction.LINEAR: AlphaCurve.LINEAR,
    AlphaCurveAction.COSINE: AlphaCurve.COSINE,
    AlphaCurveAction.SIGMOID_GENTLE: AlphaCurve.SIGMOID,
    AlphaCurveAction.SIGMOID: AlphaCurve.SIGMOID,
    AlphaCurveAction.SIGMOID_SHARP: AlphaCurve.SIGMOID,
}

# Alpha curve steepness mapping (only meaningful for SIGMOID)
ALPHA_CURVE_TO_STEEPNESS: dict[AlphaCurveAction, float] = {
    AlphaCurveAction.LINEAR: 12.0,  # Unused, but consistent default
    AlphaCurveAction.COSINE: 12.0,  # Unused, but consistent default
    AlphaCurveAction.SIGMOID_GENTLE: 6.0,
    AlphaCurveAction.SIGMOID: 12.0,
    AlphaCurveAction.SIGMOID_SHARP: 24.0,
}

# Alpha curve display glyphs for TUI/dashboard rendering.
# Single source of truth - UI components should import from here.
# Glyph design rationale:
#   LINEAR: diagonal line = constant rate ramp
#   COSINE: wave = ease-in/ease-out oscillation
#   SIGMOID family: arc progression shows transition sharpness
#     - GENTLE: wide top arc (⌒) = slow start/end
#     - STANDARD: narrow bottom arc (⌢) = moderate S-curve
#     - SHARP: squared bracket (⊐) = near-step function
ALPHA_CURVE_GLYPHS: dict[str, str] = {
    "LINEAR": "╱",
    "COSINE": "∿",
    "SIGMOID_GENTLE": "⌒",
    "SIGMOID": "⌢",
    "SIGMOID_SHARP": "⊐",
}


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

# Reverse mapping: blueprint_id string -> embedding index
# Used by SeedStateReport.blueprint_index for efficient embedding lookup.
# Maps lowercase strings (e.g., "conv_heavy") to BlueprintAction enum values (0-12).
# Returns -1 for unknown/empty blueprint_id.
BLUEPRINT_ID_TO_INDEX: dict[str, int] = {
    bid: idx for idx, bid in enumerate(BLUEPRINT_IDS) if bid is not None
}

STYLE_NAMES: tuple[str, ...] = tuple(style.name for style in GerminationStyle)
STYLE_BLEND_IDS: tuple[str, ...] = tuple(STYLE_TO_KASMINA[style][0] for style in GerminationStyle)
STYLE_ALPHA_ALGORITHMS: tuple[AlphaAlgorithm, ...] = tuple(
    STYLE_TO_KASMINA[style][1] for style in GerminationStyle
)

# Tempo name lookup (matches TempoAction enum order)
TEMPO_NAMES: tuple[str, ...] = tuple(t.name for t in TempoAction)

# Alpha target name lookup (matches AlphaTargetAction enum order)
ALPHA_TARGET_NAMES: tuple[str, ...] = tuple(t.name for t in AlphaTargetAction)

# Alpha speed name lookup (matches AlphaSpeedAction enum order)
ALPHA_SPEED_NAMES: tuple[str, ...] = tuple(s.name for s in AlphaSpeedAction)

# Alpha curve name lookup (matches AlphaCurveAction enum order)
ALPHA_CURVE_NAMES: tuple[str, ...] = tuple(c.name for c in AlphaCurveAction)

# Operation index constants for direct comparison (avoids enum construction)
OP_WAIT: int = LifecycleOp.WAIT.value
OP_GERMINATE: int = LifecycleOp.GERMINATE.value
OP_SET_ALPHA_TARGET: int = LifecycleOp.SET_ALPHA_TARGET.value
OP_PRUNE: int = LifecycleOp.PRUNE.value
OP_FOSSILIZE: int = LifecycleOp.FOSSILIZE.value
OP_ADVANCE: int = LifecycleOp.ADVANCE.value

# Module-level validation: catch enum drift at import time
assert OP_NAMES == tuple(op.name for op in LifecycleOp), (
    "OP_NAMES out of sync with LifecycleOp enum - this is a bug"
)
assert len(BLUEPRINT_IDS) == len(BlueprintAction), (
    "BLUEPRINT_IDS length mismatch with BlueprintAction enum"
)
assert len(BLUEPRINT_ID_TO_INDEX) == len(BlueprintAction), (
    "BLUEPRINT_ID_TO_INDEX length mismatch with BlueprintAction enum"
)
assert len(STYLE_NAMES) == len(GerminationStyle), (
    "STYLE_NAMES length mismatch with GerminationStyle enum"
)
assert len(STYLE_BLEND_IDS) == len(GerminationStyle), (
    "STYLE_BLEND_IDS length mismatch with GerminationStyle enum"
)
assert len(STYLE_ALPHA_ALGORITHMS) == len(GerminationStyle), (
    "STYLE_ALPHA_ALGORITHMS length mismatch with GerminationStyle enum"
)
assert len(TEMPO_NAMES) == len(TempoAction), (
    "TEMPO_NAMES length mismatch with TempoAction enum"
)
assert len(ALPHA_TARGET_NAMES) == len(AlphaTargetAction), (
    "ALPHA_TARGET_NAMES length mismatch with AlphaTargetAction enum"
)
assert len(ALPHA_SPEED_NAMES) == len(AlphaSpeedAction), (
    "ALPHA_SPEED_NAMES length mismatch with AlphaSpeedAction enum"
)
assert len(ALPHA_CURVE_NAMES) == len(AlphaCurveAction), (
    "ALPHA_CURVE_NAMES length mismatch with AlphaCurveAction enum"
)


@dataclass(frozen=True, slots=True)
class FactoredAction:
    """Factored action representation for multi-slot morphogenetic control.

    8 action heads:
    - slot_idx: Which slot to target (0-2)
    - blueprint: Which blueprint to germinate
    - style: Germination style (blend + alpha algorithm)
    - tempo: How fast to blend
    - alpha_target: Target alpha amplitude (non-zero)
    - alpha_speed: Schedule speed (controller ticks)
    - alpha_curve: Schedule curve
    - op: Lifecycle operation
    """
    slot_idx: int
    blueprint: BlueprintAction
    style: GerminationStyle
    tempo: TempoAction
    alpha_target: AlphaTargetAction
    alpha_speed: AlphaSpeedAction
    alpha_curve: AlphaCurveAction
    op: LifecycleOp

    @property
    def is_germinate(self) -> bool:
        return self.op == LifecycleOp.GERMINATE

    @property
    def is_prune(self) -> bool:
        return self.op == LifecycleOp.PRUNE

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
        return STYLE_TO_KASMINA[self.style][0]

    @property
    def alpha_algorithm_value(self) -> AlphaAlgorithm:
        return STYLE_TO_KASMINA[self.style][1]

    @property
    def alpha_target_value(self) -> float:
        return self.alpha_target.to_target()

    @property
    def alpha_speed_steps(self) -> int:
        return self.alpha_speed.to_steps()

    @property
    def alpha_curve_value(self) -> AlphaCurve:
        return self.alpha_curve.to_curve()

    @classmethod
    def from_indices(
        cls,
        slot_idx: int,
        blueprint_idx: int,
        style_idx: int,
        tempo_idx: int,
        alpha_target_idx: int,
        alpha_speed_idx: int,
        alpha_curve_idx: int,
        op_idx: int,
    ) -> "FactoredAction":
        """Create from integer indices (used by network output)."""
        return cls(
            slot_idx=slot_idx,
            blueprint=BlueprintAction(blueprint_idx),
            style=GerminationStyle(style_idx),
            tempo=TempoAction(tempo_idx),
            alpha_target=AlphaTargetAction(alpha_target_idx),
            alpha_speed=AlphaSpeedAction(alpha_speed_idx),
            alpha_curve=AlphaCurveAction(alpha_curve_idx),
            op=LifecycleOp(op_idx),
        )

    def to_indices(self) -> tuple[int, int, int, int, int, int, int, int]:
        """Convert to integer indices for network input."""
        return (
            self.slot_idx,
            self.blueprint.value,
            self.style.value,
            self.tempo.value,
            self.alpha_target.value,
            self.alpha_speed.value,
            self.alpha_curve.value,
            self.op.value,
        )


# Dimension sizes for policy network
NUM_BLUEPRINTS = len(BlueprintAction)
NUM_STYLES = len(GerminationStyle)
NUM_OPS = len(LifecycleOp)
NUM_ALPHA_TARGETS = len(AlphaTargetAction)
NUM_ALPHA_SPEEDS = len(AlphaSpeedAction)
NUM_ALPHA_CURVES = len(AlphaCurveAction)


@dataclass(frozen=True, slots=True)
class ActionHeadSpec:
    """Contract for a single action head (order + size + enum mapping)."""

    name: str
    enum: type[IntEnum] | None
    slot_dependent: bool = False

    def size(self, slot_config: SlotConfig) -> int:
        """Resolve the action count for this head."""
        if self.slot_dependent:
            return slot_config.num_slots
        if self.enum is None:
            raise ValueError(f"ActionHeadSpec '{self.name}' missing enum")
        return len(self.enum)

    def names(self, slot_config: SlotConfig) -> tuple[str, ...]:
        """Resolve the action names in enum order (slot IDs for slot head)."""
        if self.slot_dependent:
            return slot_config.slot_ids
        if self.enum is None:
            raise ValueError(f"ActionHeadSpec '{self.name}' missing enum")
        return tuple(item.name for item in self.enum)


ACTION_HEAD_SPECS: tuple[ActionHeadSpec, ...] = (
    ActionHeadSpec(name="slot", enum=None, slot_dependent=True),
    ActionHeadSpec(name="blueprint", enum=BlueprintAction),
    ActionHeadSpec(name="style", enum=GerminationStyle),
    ActionHeadSpec(name="tempo", enum=TempoAction),
    ActionHeadSpec(name="alpha_target", enum=AlphaTargetAction),
    ActionHeadSpec(name="alpha_speed", enum=AlphaSpeedAction),
    ActionHeadSpec(name="alpha_curve", enum=AlphaCurveAction),
    ActionHeadSpec(name="op", enum=LifecycleOp),
)

ACTION_HEAD_NAMES: tuple[str, ...] = tuple(spec.name for spec in ACTION_HEAD_SPECS)


def get_action_head_sizes(slot_config: SlotConfig) -> dict[str, int]:
    """Return per-head action sizes keyed by head name."""
    return {spec.name: spec.size(slot_config) for spec in ACTION_HEAD_SPECS}


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
    "GerminationStyle",
    "STYLE_TO_KASMINA",
    "TempoAction",
    "AlphaTargetAction",
    "AlphaSpeedAction",
    "AlphaCurveAction",
    "LifecycleOp",
    "FactoredAction",
    "NUM_BLUEPRINTS",
    "NUM_STYLES",
    "NUM_TEMPO",
    "NUM_OPS",
    "NUM_ALPHA_TARGETS",
    "NUM_ALPHA_SPEEDS",
    "NUM_ALPHA_CURVES",
    "ActionHeadSpec",
    "ACTION_HEAD_SPECS",
    "ACTION_HEAD_NAMES",
    "get_action_head_sizes",
    "CNN_BLUEPRINTS",
    "TRANSFORMER_BLUEPRINTS",
    # Lookup tables for hot path optimization
    "OP_NAMES",
    "BLUEPRINT_IDS",
    "BLUEPRINT_ID_TO_INDEX",
    "STYLE_NAMES",
    "STYLE_BLEND_IDS",
    "STYLE_ALPHA_ALGORITHMS",
    "TEMPO_NAMES",
    "ALPHA_TARGET_NAMES",
    "ALPHA_SPEED_NAMES",
    "ALPHA_CURVE_NAMES",
    "ALPHA_CURVE_GLYPHS",
    "TEMPO_TO_EPOCHS",
    "ALPHA_TARGET_VALUES",
    "ALPHA_SPEED_TO_STEPS",
    "OP_WAIT",
    "OP_GERMINATE",
    "OP_SET_ALPHA_TARGET",
    "OP_PRUNE",
    "OP_FOSSILIZE",
    "OP_ADVANCE",
]
