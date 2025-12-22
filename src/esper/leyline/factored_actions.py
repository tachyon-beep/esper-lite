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


class BlendAction(IntEnum):
    """Blending algorithm selection."""
    LINEAR = 0
    SIGMOID = 1
    GATED = 2

    def to_algorithm_id(self) -> str:
        return ["linear", "sigmoid", "gated"][self.value]


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
    """Alpha schedule curve selection."""
    LINEAR = 0
    COSINE = 1
    SIGMOID = 2

    def to_curve(self) -> AlphaCurve:
        return {
            AlphaCurveAction.LINEAR: AlphaCurve.LINEAR,
            AlphaCurveAction.COSINE: AlphaCurve.COSINE,
            AlphaCurveAction.SIGMOID: AlphaCurve.SIGMOID,
        }[self]


class AlphaAlgorithmAction(IntEnum):
    """Blend composition / gating algorithm selection."""
    ADD = 0
    MULTIPLY = 1
    GATE = 2

    def to_algorithm(self) -> AlphaAlgorithm:
        return {
            AlphaAlgorithmAction.ADD: AlphaAlgorithm.ADD,
            AlphaAlgorithmAction.MULTIPLY: AlphaAlgorithm.MULTIPLY,
            AlphaAlgorithmAction.GATE: AlphaAlgorithm.GATE,
        }[self]


def is_valid_blend_alpha_combo(
    blend: BlendAction,
    alpha_algorithm: AlphaAlgorithmAction,
) -> bool:
    """Return True when blend/alpha algorithm pairing is compatible."""
    if blend == BlendAction.GATED:
        return alpha_algorithm == AlphaAlgorithmAction.GATE
    return alpha_algorithm != AlphaAlgorithmAction.GATE


# Alpha target values (non-zero targets only; removal uses PRUNE)
ALPHA_TARGET_VALUES: tuple[float, ...] = (0.5, 0.7, 1.0)

# Alpha speed mapping (controller ticks)
ALPHA_SPEED_TO_STEPS: dict[AlphaSpeedAction, int] = {
    AlphaSpeedAction.INSTANT: 0,
    AlphaSpeedAction.FAST: 3,
    AlphaSpeedAction.MEDIUM: 5,
    AlphaSpeedAction.SLOW: 8,
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

# Blend algorithm ID lookup (matches BlendAction.to_algorithm_id())
BLEND_IDS: tuple[str, ...] = tuple(blend.to_algorithm_id() for blend in BlendAction)

# Tempo name lookup (matches TempoAction enum order)
TEMPO_NAMES: tuple[str, ...] = tuple(t.name for t in TempoAction)

# Alpha target name lookup (matches AlphaTargetAction enum order)
ALPHA_TARGET_NAMES: tuple[str, ...] = tuple(t.name for t in AlphaTargetAction)

# Alpha speed name lookup (matches AlphaSpeedAction enum order)
ALPHA_SPEED_NAMES: tuple[str, ...] = tuple(s.name for s in AlphaSpeedAction)

# Alpha curve name lookup (matches AlphaCurveAction enum order)
ALPHA_CURVE_NAMES: tuple[str, ...] = tuple(c.name for c in AlphaCurveAction)

# Alpha algorithm name lookup (matches AlphaAlgorithmAction enum order)
ALPHA_ALGORITHM_NAMES: tuple[str, ...] = tuple(a.name for a in AlphaAlgorithmAction)

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
assert len(BLEND_IDS) == len(BlendAction), (
    "BLEND_IDS length mismatch with BlendAction enum"
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
assert len(ALPHA_ALGORITHM_NAMES) == len(AlphaAlgorithmAction), (
    "ALPHA_ALGORITHM_NAMES length mismatch with AlphaAlgorithmAction enum"
)


@dataclass(frozen=True, slots=True)
class FactoredAction:
    """Factored action representation for multi-slot morphogenetic control.

    9 action heads:
    - slot_idx: Which slot to target (0-2)
    - blueprint: Which blueprint to germinate
    - blend: Which blending algorithm
    - tempo: How fast to blend
    - alpha_target: Target alpha amplitude (non-zero)
    - alpha_speed: Schedule speed (controller ticks)
    - alpha_curve: Schedule curve
    - alpha_algorithm: Blend composition / gating mode
    - op: Lifecycle operation
    """
    slot_idx: int
    blueprint: BlueprintAction
    blend: BlendAction
    tempo: TempoAction
    alpha_target: AlphaTargetAction
    alpha_speed: AlphaSpeedAction
    alpha_curve: AlphaCurveAction
    alpha_algorithm: AlphaAlgorithmAction
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
        return self.blend.to_algorithm_id()

    @property
    def alpha_target_value(self) -> float:
        return self.alpha_target.to_target()

    @property
    def alpha_speed_steps(self) -> int:
        return self.alpha_speed.to_steps()

    @property
    def alpha_curve_value(self) -> AlphaCurve:
        return self.alpha_curve.to_curve()

    @property
    def alpha_algorithm_value(self) -> AlphaAlgorithm:
        return self.alpha_algorithm.to_algorithm()

    @classmethod
    def from_indices(
        cls,
        slot_idx: int,
        blueprint_idx: int,
        blend_idx: int,
        tempo_idx: int,
        alpha_target_idx: int,
        alpha_speed_idx: int,
        alpha_curve_idx: int,
        alpha_algorithm_idx: int,
        op_idx: int,
    ) -> "FactoredAction":
        """Create from integer indices (used by network output)."""
        return cls(
            slot_idx=slot_idx,
            blueprint=BlueprintAction(blueprint_idx),
            blend=BlendAction(blend_idx),
            tempo=TempoAction(tempo_idx),
            alpha_target=AlphaTargetAction(alpha_target_idx),
            alpha_speed=AlphaSpeedAction(alpha_speed_idx),
            alpha_curve=AlphaCurveAction(alpha_curve_idx),
            alpha_algorithm=AlphaAlgorithmAction(alpha_algorithm_idx),
            op=LifecycleOp(op_idx),
        )

    def to_indices(self) -> tuple[int, int, int, int, int, int, int, int, int]:
        """Convert to integer indices for network input."""
        return (
            self.slot_idx,
            self.blueprint.value,
            self.blend.value,
            self.tempo.value,
            self.alpha_target.value,
            self.alpha_speed.value,
            self.alpha_curve.value,
            self.alpha_algorithm.value,
            self.op.value,
        )


# Dimension sizes for policy network
NUM_BLUEPRINTS = len(BlueprintAction)
NUM_BLENDS = len(BlendAction)
NUM_OPS = len(LifecycleOp)
NUM_ALPHA_TARGETS = len(AlphaTargetAction)
NUM_ALPHA_SPEEDS = len(AlphaSpeedAction)
NUM_ALPHA_CURVES = len(AlphaCurveAction)
NUM_ALPHA_ALGORITHMS = len(AlphaAlgorithmAction)


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
    ActionHeadSpec(name="blend", enum=BlendAction),
    ActionHeadSpec(name="tempo", enum=TempoAction),
    ActionHeadSpec(name="alpha_target", enum=AlphaTargetAction),
    ActionHeadSpec(name="alpha_speed", enum=AlphaSpeedAction),
    ActionHeadSpec(name="alpha_curve", enum=AlphaCurveAction),
    ActionHeadSpec(name="alpha_algorithm", enum=AlphaAlgorithmAction),
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
    "BlendAction",
    "TempoAction",
    "AlphaTargetAction",
    "AlphaSpeedAction",
    "AlphaCurveAction",
    "AlphaAlgorithmAction",
    "LifecycleOp",
    "FactoredAction",
    "NUM_BLUEPRINTS",
    "NUM_BLENDS",
    "NUM_TEMPO",
    "NUM_OPS",
    "NUM_ALPHA_TARGETS",
    "NUM_ALPHA_SPEEDS",
    "NUM_ALPHA_CURVES",
    "NUM_ALPHA_ALGORITHMS",
    "is_valid_blend_alpha_combo",
    "ActionHeadSpec",
    "ACTION_HEAD_SPECS",
    "ACTION_HEAD_NAMES",
    "get_action_head_sizes",
    "CNN_BLUEPRINTS",
    "TRANSFORMER_BLUEPRINTS",
    # Lookup tables for hot path optimization
    "OP_NAMES",
    "BLUEPRINT_IDS",
    "BLEND_IDS",
    "TEMPO_NAMES",
    "ALPHA_TARGET_NAMES",
    "ALPHA_SPEED_NAMES",
    "ALPHA_CURVE_NAMES",
    "ALPHA_ALGORITHM_NAMES",
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
