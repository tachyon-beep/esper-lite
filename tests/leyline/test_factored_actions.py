# tests/leyline/test_factored_actions.py
def test_slot_action_enum():
    """SlotAction should enumerate slot choices."""
    from esper.leyline.factored_actions import SlotAction

    assert SlotAction.EARLY.value == 0
    assert SlotAction.MID.value == 1
    assert SlotAction.LATE.value == 2
    assert len(SlotAction) == 3


def test_blueprint_action_enum():
    """BlueprintAction should enumerate blueprint choices."""
    from esper.leyline.factored_actions import BlueprintAction

    assert BlueprintAction.NOOP.value == 0
    assert BlueprintAction.CONV_ENHANCE.value == 1
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
    """FactoredAction should compose slot, blueprint, blend, op."""
    from esper.leyline.factored_actions import (
        FactoredAction, SlotAction, BlueprintAction, BlendAction, LifecycleOp
    )

    action = FactoredAction(
        slot=SlotAction.MID,
        blueprint=BlueprintAction.CONV_ENHANCE,
        blend=BlendAction.LINEAR,
        op=LifecycleOp.GERMINATE,
    )

    assert action.slot == SlotAction.MID
    assert action.blueprint == BlueprintAction.CONV_ENHANCE
    assert action.is_germinate
    # CONV_ENHANCE maps to "conv_light" (the registered blueprint name)
    assert action.blueprint_id == "conv_light"


def test_factored_action_execution_properties():
    """FactoredAction properties should provide everything needed for execution."""
    from esper.leyline.factored_actions import (
        FactoredAction, SlotAction, BlueprintAction, BlendAction, LifecycleOp,
    )

    # GERMINATE action has all info for execution
    germ = FactoredAction(SlotAction.EARLY, BlueprintAction.CONV_ENHANCE, BlendAction.SIGMOID, LifecycleOp.GERMINATE)
    assert germ.is_germinate
    assert germ.slot_id == "early"
    assert germ.blueprint_id == "conv_light"  # CONV_ENHANCE maps to registered name
    assert germ.blend_algorithm_id == "sigmoid"

    # CULL action
    cull = FactoredAction(SlotAction.MID, BlueprintAction.NOOP, BlendAction.LINEAR, LifecycleOp.CULL)
    assert cull.is_cull
    assert cull.slot_id == "mid"

    # FOSSILIZE action
    fossilize = FactoredAction(SlotAction.LATE, BlueprintAction.NOOP, BlendAction.LINEAR, LifecycleOp.FOSSILIZE)
    assert fossilize.is_fossilize
    assert fossilize.slot_id == "late"

    # WAIT action
    wait = FactoredAction(SlotAction.MID, BlueprintAction.NOOP, BlendAction.LINEAR, LifecycleOp.WAIT)
    assert wait.is_wait
