"""Tests for per-topology action enums."""

from enum import IntEnum


def test_build_action_enum_cnn():
    """build_action_enum creates CNN action enum."""
    from esper.leyline.actions import build_action_enum

    CNNAction = build_action_enum("cnn")

    assert issubclass(CNNAction, IntEnum)
    assert CNNAction.WAIT.value == 0
    assert hasattr(CNNAction, "GERMINATE_NORM")
    assert hasattr(CNNAction, "FOSSILIZE")
    assert hasattr(CNNAction, "CULL")


def test_build_action_enum_transformer():
    """build_action_enum creates Transformer action enum."""
    from esper.leyline.actions import build_action_enum

    TransformerAction = build_action_enum("transformer")

    assert issubclass(TransformerAction, IntEnum)
    assert TransformerAction.WAIT.value == 0
    assert hasattr(TransformerAction, "GERMINATE_LORA")
    assert hasattr(TransformerAction, "FOSSILIZE")
    assert hasattr(TransformerAction, "CULL")


def test_action_enum_values_sequential():
    """Action values are sequential integers."""
    from esper.leyline.actions import build_action_enum

    Action = build_action_enum("cnn")
    values = [a.value for a in Action]

    assert values == list(range(len(Action)))


def test_action_enum_cull_is_last():
    """CULL is always the last action."""
    from esper.leyline.actions import build_action_enum

    Action = build_action_enum("cnn")

    assert Action.CULL.value == len(Action) - 1


def test_get_blueprint_from_action():
    """Can get blueprint name from germinate action."""
    from esper.leyline.actions import build_action_enum, get_blueprint_from_action

    Action = build_action_enum("cnn")

    blueprint = get_blueprint_from_action(Action.GERMINATE_NORM)
    assert blueprint == "norm"

    assert get_blueprint_from_action(Action.WAIT) is None
    assert get_blueprint_from_action(Action.FOSSILIZE) is None
    assert get_blueprint_from_action(Action.CULL) is None
