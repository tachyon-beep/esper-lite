# tests/leyline/test_factored_actions.py
# Removed: test_slot_action_enum - SlotAction enum deleted per No Legacy Code Policy

import pytest
from hypothesis import given, strategies as st


def test_blueprint_action_enum():
    """BlueprintAction should enumerate blueprint choices."""
    from esper.leyline.factored_actions import BlueprintAction

    assert BlueprintAction.NOOP.value == 0
    assert BlueprintAction.CONV_LIGHT.value == 1
    assert len(BlueprintAction) >= 5  # noop + 4 blueprints


def test_blend_action_enum():
    """BlendAction should enumerate blending algorithm choices."""
    from esper.leyline.factored_actions import BlendAction

    assert BlendAction.LINEAR.value == 0
    assert BlendAction.SIGMOID.value == 1
    assert BlendAction.GATED.value == 2


def test_lifecycle_op_enum():
    """LifecycleOp should enumerate lifecycle operations."""
    from esper.leyline.factored_actions import LifecycleOp

    assert LifecycleOp.WAIT.value == 0
    assert LifecycleOp.GERMINATE.value == 1
    assert LifecycleOp.PRUNE.value == 2
    assert LifecycleOp.FOSSILIZE.value == 3
    assert len(LifecycleOp) == 4


def test_factored_action_composition():
    """FactoredAction should compose slot_idx, blueprint, blend, tempo, op."""
    from esper.leyline.factored_actions import (
        FactoredAction, BlueprintAction, BlendAction, TempoAction, LifecycleOp
    )

    action = FactoredAction(
        slot_idx=1,  # Was SlotAction.MID
        blueprint=BlueprintAction.CONV_LIGHT,
        blend=BlendAction.LINEAR,
        tempo=TempoAction.STANDARD,
        op=LifecycleOp.GERMINATE,
    )

    assert action.slot_idx == 1
    assert action.blueprint == BlueprintAction.CONV_LIGHT
    assert action.is_germinate
    # CONV_LIGHT maps to "conv_light" (the registered blueprint name)
    assert action.blueprint_id == "conv_light"


def test_factored_action_execution_properties():
    """FactoredAction properties should provide everything needed for execution."""
    from esper.leyline.factored_actions import (
        FactoredAction, BlueprintAction, BlendAction, TempoAction, LifecycleOp,
    )

    # GERMINATE action has all info for execution
    germ = FactoredAction(slot_idx=0, blueprint=BlueprintAction.CONV_LIGHT, blend=BlendAction.SIGMOID, tempo=TempoAction.FAST, op=LifecycleOp.GERMINATE)
    assert germ.is_germinate
    assert germ.slot_idx == 0
    assert germ.blueprint_id == "conv_light"  # CONV_LIGHT maps to registered name
    assert germ.blend_algorithm_id == "sigmoid"

    # PRUNE action
    prune = FactoredAction(slot_idx=1, blueprint=BlueprintAction.NOOP, blend=BlendAction.LINEAR, tempo=TempoAction.STANDARD, op=LifecycleOp.PRUNE)
    assert prune.is_prune
    assert prune.slot_idx == 1

    # FOSSILIZE action
    fossilize = FactoredAction(slot_idx=2, blueprint=BlueprintAction.NOOP, blend=BlendAction.LINEAR, tempo=TempoAction.SLOW, op=LifecycleOp.FOSSILIZE)
    assert fossilize.is_fossilize
    assert fossilize.slot_idx == 2

    # WAIT action
    wait = FactoredAction(slot_idx=1, blueprint=BlueprintAction.NOOP, blend=BlendAction.LINEAR, tempo=TempoAction.STANDARD, op=LifecycleOp.WAIT)
    assert wait.is_wait


def test_tempo_action_enum():
    """TempoAction has expected values."""
    from esper.leyline.factored_actions import TempoAction

    assert len(TempoAction) == 3
    assert TempoAction.FAST.value == 0
    assert TempoAction.STANDARD.value == 1
    assert TempoAction.SLOW.value == 2


def test_num_tempo_constant():
    """NUM_TEMPO matches enum length."""
    from esper.leyline.factored_actions import NUM_TEMPO, TempoAction

    assert NUM_TEMPO == len(TempoAction)


def test_tempo_to_epochs_mapping():
    """TEMPO_TO_EPOCHS maps correctly."""
    from esper.leyline.factored_actions import TEMPO_TO_EPOCHS, TempoAction

    assert TEMPO_TO_EPOCHS[TempoAction.FAST] == 3
    assert TEMPO_TO_EPOCHS[TempoAction.STANDARD] == 5
    assert TEMPO_TO_EPOCHS[TempoAction.SLOW] == 8


def test_tempo_names_lookup():
    """TEMPO_NAMES enables hot-path lookups."""
    from esper.leyline.factored_actions import TEMPO_NAMES, TempoAction

    assert TEMPO_NAMES == ("FAST", "STANDARD", "SLOW")
    assert TEMPO_NAMES[TempoAction.STANDARD.value] == "STANDARD"


@given(
    slot=st.integers(0, 2),
    blueprint=st.integers(0, 4),  # NUM_BLUEPRINTS - 1 (currently 5 blueprints)
    blend=st.integers(0, 2),
    tempo=st.integers(0, 2),  # NUM_TEMPO - 1
    op=st.integers(0, 3),
)
def test_factored_action_roundtrip(slot, blueprint, blend, tempo, op):
    """FactoredAction survives index conversion."""
    from esper.leyline.factored_actions import FactoredAction, TempoAction

    action = FactoredAction.from_indices(slot, blueprint, blend, tempo, op)
    indices = action.to_indices()
    assert indices == (slot, blueprint, blend, tempo, op)

    # Verify types
    assert isinstance(action.tempo, TempoAction)
