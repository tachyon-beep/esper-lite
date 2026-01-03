"""Tests for per-topology action enums.

Note: build_action_enum was moved from leyline.actions to tamiyo.action_enums
to break the Leyline <-> Kasmina import cycle. These tests remain here for
continuity but import from the new location.
"""

from enum import IntEnum

from esper.leyline.actions import get_blueprint_from_action_name


def test_build_action_enum_cnn():
    """build_action_enum creates CNN action enum."""
    from esper.tamiyo.action_enums import build_action_enum

    CNNAction = build_action_enum("cnn")

    assert issubclass(CNNAction, IntEnum)
    assert CNNAction.WAIT.value == 0
    assert hasattr(CNNAction, "GERMINATE_NORM")
    assert hasattr(CNNAction, "FOSSILIZE")
    assert hasattr(CNNAction, "PRUNE")
    assert hasattr(CNNAction, "ADVANCE")


def test_build_action_enum_transformer():
    """build_action_enum creates Transformer action enum."""
    from esper.tamiyo.action_enums import build_action_enum

    TransformerAction = build_action_enum("transformer")

    assert issubclass(TransformerAction, IntEnum)
    assert TransformerAction.WAIT.value == 0
    assert hasattr(TransformerAction, "GERMINATE_LORA")
    assert hasattr(TransformerAction, "FOSSILIZE")
    assert hasattr(TransformerAction, "PRUNE")
    assert hasattr(TransformerAction, "ADVANCE")


def test_action_enum_values_sequential():
    """Action values are sequential integers."""
    from esper.tamiyo.action_enums import build_action_enum

    Action = build_action_enum("cnn")
    values = [a.value for a in Action]

    assert values == list(range(len(Action)))


def test_action_enum_advance_is_last():
    """ADVANCE is always the last action."""
    from esper.tamiyo.action_enums import build_action_enum

    Action = build_action_enum("cnn")

    assert Action.ADVANCE.value == len(Action) - 1


def test_get_blueprint_from_action():
    """Can get blueprint name from germinate action."""
    from esper.tamiyo.action_enums import build_action_enum

    Action = build_action_enum("cnn")

    blueprint = get_blueprint_from_action_name(Action.GERMINATE_NORM.name)
    assert blueprint == "norm"

    assert get_blueprint_from_action_name(Action.WAIT.name) is None
    assert get_blueprint_from_action_name(Action.FOSSILIZE.name) is None
    assert get_blueprint_from_action_name(Action.PRUNE.name) is None
