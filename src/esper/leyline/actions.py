"""Leyline Actions - Action name parsing utilities for Esper agents.

This module provides the single source of truth for parsing GERMINATE_<BLUEPRINT>
action names. All domains should import these utilities instead of duplicating
the parsing logic.

Note: For building dynamic action enums from blueprints, use
esper.tamiyo.action_enums.build_action_enum(). That function lives in Tamiyo
(not Leyline) because it depends on Kasmina's BlueprintRegistry, and Leyline
must remain free of domain package dependencies.

PPO training uses factored actions from esper.leyline.factored_actions.
"""

# =============================================================================
# GERMINATE Action Name Utilities
# =============================================================================
# These are the single source of truth for parsing GERMINATE_<BLUEPRINT> action names.
# All domains should import these instead of duplicating the parsing logic.

GERMINATE_PREFIX = "GERMINATE_"


def is_germinate_action_name(name: str) -> bool:
    """Check if action name is a germinate variant.

    Args:
        name: Action name to check (e.g., "GERMINATE_CONV_LIGHT").

    Returns:
        True if name starts with GERMINATE_ prefix.
    """
    return name.startswith(GERMINATE_PREFIX)


def get_blueprint_from_action_name(name: str) -> str | None:
    """Extract blueprint ID from germinate action name.

    Args:
        name: Action name (e.g., "GERMINATE_CONV_LIGHT").

    Returns:
        Lowercase blueprint name (e.g., "conv_light") or None if not a germinate action.
    """
    if name.startswith(GERMINATE_PREFIX):
        return name[len(GERMINATE_PREFIX) :].lower()
    return None


__all__ = [
    "GERMINATE_PREFIX",
    "get_blueprint_from_action_name",
    "is_germinate_action_name",
]
