# tests/leyline/test_factored_actions.py
# Removed: test_slot_action_enum - SlotAction enum deleted per No Legacy Code Policy

from hypothesis import given, strategies as st

from esper.leyline import (
    NUM_ALPHA_CURVES,
    NUM_ALPHA_SPEEDS,
    NUM_ALPHA_TARGETS,
    NUM_BLUEPRINTS,
    NUM_OPS,
    NUM_STYLES,
    NUM_TEMPO,
)


def test_blueprint_action_enum():
    """BlueprintAction should enumerate blueprint choices."""
    from esper.leyline import BlueprintAction

    assert BlueprintAction.NOOP.value == 0
    assert BlueprintAction.CONV_LIGHT.value == 1
    assert len(BlueprintAction) >= 5  # noop + 4 blueprints


def test_germination_style_enum():
    """GerminationStyle should enumerate valid germination styles."""
    from esper.leyline import GerminationStyle

    assert GerminationStyle.LINEAR_ADD.value == 0
    assert GerminationStyle.LINEAR_MULTIPLY.value == 1
    assert GerminationStyle.SIGMOID_ADD.value == 2
    assert GerminationStyle.GATED_GATE.value == 3


def test_lifecycle_op_enum():
    """LifecycleOp should enumerate lifecycle operations."""
    from esper.leyline import LifecycleOp

    assert LifecycleOp.WAIT.value == 0
    assert LifecycleOp.GERMINATE.value == 1
    assert LifecycleOp.SET_ALPHA_TARGET.value == 2
    assert LifecycleOp.PRUNE.value == 3
    assert LifecycleOp.FOSSILIZE.value == 4
    assert LifecycleOp.ADVANCE.value == 5
    assert len(LifecycleOp) == 6


def test_factored_action_composition():
    """FactoredAction should compose slot_idx, blueprint, style, tempo, alpha heads, op."""
    from esper.leyline import (
        AlphaCurveAction,
        AlphaSpeedAction,
        AlphaTargetAction,
        BlueprintAction,
        FactoredAction,
        GerminationStyle,
        LifecycleOp,
        TempoAction,
    )

    action = FactoredAction(
        slot_idx=1,  # Was SlotAction.MID
        blueprint=BlueprintAction.CONV_LIGHT,
        style=GerminationStyle.SIGMOID_ADD,
        tempo=TempoAction.STANDARD,
        alpha_target=AlphaTargetAction.FULL,
        alpha_speed=AlphaSpeedAction.MEDIUM,
        alpha_curve=AlphaCurveAction.SIGMOID,
        op=LifecycleOp.GERMINATE,
    )

    assert action.slot_idx == 1
    assert action.blueprint == BlueprintAction.CONV_LIGHT
    assert action.is_germinate
    # CONV_LIGHT maps to "conv_light" (the registered blueprint name)
    assert action.blueprint_id == "conv_light"


def test_factored_action_execution_properties():
    """FactoredAction properties should provide everything needed for execution."""
    from esper.leyline import (
        AlphaCurveAction,
        AlphaSpeedAction,
        AlphaTargetAction,
        BlueprintAction,
        FactoredAction,
        GerminationStyle,
        LifecycleOp,
        TempoAction,
    )

    # GERMINATE action has all info for execution
    germ = FactoredAction(
        slot_idx=0,
        blueprint=BlueprintAction.CONV_LIGHT,
        style=GerminationStyle.SIGMOID_ADD,
        tempo=TempoAction.FAST,
        alpha_target=AlphaTargetAction.FULL,
        alpha_speed=AlphaSpeedAction.FAST,
        alpha_curve=AlphaCurveAction.LINEAR,
        op=LifecycleOp.GERMINATE,
    )
    assert germ.is_germinate
    assert germ.slot_idx == 0
    assert germ.blueprint_id == "conv_light"  # CONV_LIGHT maps to registered name
    assert germ.blend_algorithm_id == "sigmoid"
    assert germ.alpha_target_value == 1.0
    assert germ.alpha_speed_steps == 3

    # PRUNE action
    prune = FactoredAction(
        slot_idx=1,
        blueprint=BlueprintAction.NOOP,
        style=GerminationStyle.SIGMOID_ADD,
        tempo=TempoAction.STANDARD,
        alpha_target=AlphaTargetAction.FULL,
        alpha_speed=AlphaSpeedAction.SLOW,
        alpha_curve=AlphaCurveAction.SIGMOID,
        op=LifecycleOp.PRUNE,
    )
    assert prune.is_prune
    assert prune.slot_idx == 1
    assert prune.alpha_speed_steps == 8

    # FOSSILIZE action
    fossilize = FactoredAction(
        slot_idx=2,
        blueprint=BlueprintAction.NOOP,
        style=GerminationStyle.SIGMOID_ADD,
        tempo=TempoAction.SLOW,
        alpha_target=AlphaTargetAction.FULL,
        alpha_speed=AlphaSpeedAction.MEDIUM,
        alpha_curve=AlphaCurveAction.LINEAR,
        op=LifecycleOp.FOSSILIZE,
    )
    assert fossilize.is_fossilize
    assert fossilize.slot_idx == 2

    # WAIT action
    wait = FactoredAction(
        slot_idx=1,
        blueprint=BlueprintAction.NOOP,
        style=GerminationStyle.SIGMOID_ADD,
        tempo=TempoAction.STANDARD,
        alpha_target=AlphaTargetAction.FULL,
        alpha_speed=AlphaSpeedAction.MEDIUM,
        alpha_curve=AlphaCurveAction.LINEAR,
        op=LifecycleOp.WAIT,
    )
    assert wait.is_wait


def test_tempo_action_enum():
    """TempoAction has expected values."""
    from esper.leyline import TempoAction

    assert len(TempoAction) == 3
    assert TempoAction.FAST.value == 0
    assert TempoAction.STANDARD.value == 1
    assert TempoAction.SLOW.value == 2


def test_style_to_kasmina_mapping():
    """GerminationStyle mapping should only encode valid Kasmina germinate combos."""
    from esper.leyline import GerminationStyle, STYLE_TO_KASMINA
    from esper.leyline.alpha import AlphaAlgorithm

    assert STYLE_TO_KASMINA[GerminationStyle.LINEAR_ADD] == ("linear", AlphaAlgorithm.ADD)
    assert STYLE_TO_KASMINA[GerminationStyle.LINEAR_MULTIPLY] == ("linear", AlphaAlgorithm.MULTIPLY)
    assert STYLE_TO_KASMINA[GerminationStyle.SIGMOID_ADD] == ("sigmoid", AlphaAlgorithm.ADD)
    assert STYLE_TO_KASMINA[GerminationStyle.GATED_GATE] == ("gated", AlphaAlgorithm.GATE)


def test_num_tempo_constant():
    """NUM_TEMPO matches enum length."""
    from esper.leyline import NUM_TEMPO, TempoAction

    assert NUM_TEMPO == len(TempoAction)


def test_tempo_to_epochs_mapping():
    """TEMPO_TO_EPOCHS maps correctly."""
    from esper.leyline import TEMPO_TO_EPOCHS, TempoAction

    assert TEMPO_TO_EPOCHS[TempoAction.FAST] == 3
    assert TEMPO_TO_EPOCHS[TempoAction.STANDARD] == 5
    assert TEMPO_TO_EPOCHS[TempoAction.SLOW] == 8


def test_tempo_names_lookup():
    """TEMPO_NAMES enables hot-path lookups."""
    from esper.leyline import TEMPO_NAMES, TempoAction

    assert TEMPO_NAMES == ("FAST", "STANDARD", "SLOW")
    assert TEMPO_NAMES[TempoAction.STANDARD.value] == "STANDARD"


@given(
    slot=st.integers(0, 2),
    blueprint=st.integers(0, NUM_BLUEPRINTS - 1),
    style=st.integers(0, NUM_STYLES - 1),
    tempo=st.integers(0, NUM_TEMPO - 1),
    alpha_target=st.integers(0, NUM_ALPHA_TARGETS - 1),
    alpha_speed=st.integers(0, NUM_ALPHA_SPEEDS - 1),
    alpha_curve=st.integers(0, NUM_ALPHA_CURVES - 1),
    op=st.integers(0, NUM_OPS - 1),
)
def test_factored_action_roundtrip(
    slot,
    blueprint,
    style,
    tempo,
    alpha_target,
    alpha_speed,
    alpha_curve,
    op,
):
    """FactoredAction survives index conversion."""
    from esper.leyline import FactoredAction, TempoAction

    action = FactoredAction.from_indices(
        slot,
        blueprint,
        style,
        tempo,
        alpha_target,
        alpha_speed,
        alpha_curve,
        op,
    )
    indices = action.to_indices()
    assert indices == (
        slot,
        blueprint,
        style,
        tempo,
        alpha_target,
        alpha_speed,
        alpha_curve,
        op,
    )

    # Verify types
    assert isinstance(action.tempo, TempoAction)
