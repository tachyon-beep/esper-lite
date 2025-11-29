"""Blueprint definitions and mappings.

This module defines the available seed blueprints and their mappings
to germinate actions. Centralized here to prevent duplication across
tamiyo, simic, and other modules.
"""

from __future__ import annotations

from esper.leyline.actions import Action


# Available blueprint IDs
BLUEPRINT_CONV_ENHANCE = "conv_enhance"
BLUEPRINT_ATTENTION = "attention"
BLUEPRINT_NORM = "norm"
BLUEPRINT_DEPTHWISE = "depthwise"

ALL_BLUEPRINTS = [
    BLUEPRINT_CONV_ENHANCE,
    BLUEPRINT_ATTENTION,
    BLUEPRINT_NORM,
    BLUEPRINT_DEPTHWISE,
]


# Blueprint ID → Germinate Action mapping
BLUEPRINT_TO_ACTION: dict[str, Action] = {
    BLUEPRINT_CONV_ENHANCE: Action.GERMINATE_CONV,
    BLUEPRINT_ATTENTION: Action.GERMINATE_ATTENTION,
    BLUEPRINT_NORM: Action.GERMINATE_NORM,
    BLUEPRINT_DEPTHWISE: Action.GERMINATE_DEPTHWISE,
}


# Germinate Action → Blueprint ID mapping (inverse)
ACTION_TO_BLUEPRINT: dict[Action, str] = {
    Action.GERMINATE_CONV: BLUEPRINT_CONV_ENHANCE,
    Action.GERMINATE_ATTENTION: BLUEPRINT_ATTENTION,
    Action.GERMINATE_NORM: BLUEPRINT_NORM,
    Action.GERMINATE_DEPTHWISE: BLUEPRINT_DEPTHWISE,
}


def blueprint_to_action(blueprint_id: str) -> Action:
    """Convert blueprint ID to corresponding germinate action.

    Args:
        blueprint_id: Blueprint identifier

    Returns:
        Corresponding GERMINATE_* action

    Raises:
        KeyError: If blueprint_id is unknown
    """
    return BLUEPRINT_TO_ACTION[blueprint_id]


def action_to_blueprint(action: Action) -> str | None:
    """Convert germinate action to blueprint ID.

    Args:
        action: Action enum value

    Returns:
        Blueprint ID if action is a germinate variant, None otherwise
    """
    return ACTION_TO_BLUEPRINT.get(action)


__all__ = [
    "BLUEPRINT_CONV_ENHANCE",
    "BLUEPRINT_ATTENTION",
    "BLUEPRINT_NORM",
    "BLUEPRINT_DEPTHWISE",
    "ALL_BLUEPRINTS",
    "BLUEPRINT_TO_ACTION",
    "ACTION_TO_BLUEPRINT",
    "blueprint_to_action",
    "action_to_blueprint",
]
