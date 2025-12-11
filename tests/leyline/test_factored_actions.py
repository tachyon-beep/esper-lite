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
    assert LifecycleOp.ADVANCE.value == 2
    assert LifecycleOp.CULL.value == 3


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
    assert action.blueprint_id == "conv_enhance"
