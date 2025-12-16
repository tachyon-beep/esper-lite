# tests/leyline/test_factored_actions.py
# Removed: test_slot_action_enum - SlotAction enum deleted per No Legacy Code Policy


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
    assert LifecycleOp.CULL.value == 2
    assert LifecycleOp.FOSSILIZE.value == 3
    assert len(LifecycleOp) == 4


def test_factored_action_composition():
    """FactoredAction should compose slot_idx, blueprint, blend, op."""
    from esper.leyline.factored_actions import (
        FactoredAction, BlueprintAction, BlendAction, LifecycleOp
    )

    action = FactoredAction(
        slot_idx=1,  # Was SlotAction.MID
        blueprint=BlueprintAction.CONV_LIGHT,
        blend=BlendAction.LINEAR,
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
        FactoredAction, BlueprintAction, BlendAction, LifecycleOp,
    )

    # GERMINATE action has all info for execution
    germ = FactoredAction(slot_idx=0, blueprint=BlueprintAction.CONV_LIGHT, blend=BlendAction.SIGMOID, op=LifecycleOp.GERMINATE)
    assert germ.is_germinate
    assert germ.slot_idx == 0
    assert germ.blueprint_id == "conv_light"  # CONV_LIGHT maps to registered name
    assert germ.blend_algorithm_id == "sigmoid"

    # CULL action
    cull = FactoredAction(slot_idx=1, blueprint=BlueprintAction.NOOP, blend=BlendAction.LINEAR, op=LifecycleOp.CULL)
    assert cull.is_cull
    assert cull.slot_idx == 1

    # FOSSILIZE action
    fossilize = FactoredAction(slot_idx=2, blueprint=BlueprintAction.NOOP, blend=BlendAction.LINEAR, op=LifecycleOp.FOSSILIZE)
    assert fossilize.is_fossilize
    assert fossilize.slot_idx == 2

    # WAIT action
    wait = FactoredAction(slot_idx=1, blueprint=BlueprintAction.NOOP, blend=BlendAction.LINEAR, op=LifecycleOp.WAIT)
    assert wait.is_wait
