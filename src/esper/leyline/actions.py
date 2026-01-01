"""Leyline Actions - Action space definitions for Esper agents.

Actions represent the discrete choices available to the strategic controller.
Per-topology action enums are built dynamically from registered blueprints.

Note: The flat action enum is used only by HeuristicTamiyo for baseline comparison.
PPO training uses factored actions from esper.leyline.factored_actions.
"""

from enum import IntEnum

# Cache for built enums
_action_enum_cache: dict[str, type[IntEnum]] = {}

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


def build_action_enum(topology: str) -> type[IntEnum]:
    """Build action enum from registered blueprints for a topology.

    Action layout:
        0: WAIT
        1-N: GERMINATE_<BLUEPRINT> (sorted by param estimate)
        N+1: FOSSILIZE
        N+2: PRUNE
        N+3: ADVANCE

    Note: This is used by HeuristicTamiyo for baseline comparison.
    PPO training uses factored actions instead.
    """
    if topology in _action_enum_cache:
        return _action_enum_cache[topology]

    from esper.kasmina.blueprints import BlueprintRegistry

    blueprints = BlueprintRegistry.list_for_topology(topology)

    members = {"WAIT": 0}
    for i, spec in enumerate(blueprints, start=1):
        members[f"GERMINATE_{spec.name.upper()}"] = i
    members["FOSSILIZE"] = len(blueprints) + 1
    members["PRUNE"] = len(blueprints) + 2
    members["ADVANCE"] = len(blueprints) + 3

    # Build list of tuples for IntEnum (avoids string literal requirement)
    member_list = list(members.items())
    # Type ignore needed: IntEnum expects string literal but we build dynamically
    action_enum = IntEnum(f"{topology.title()}Action", member_list)  # type: ignore[misc]
    _action_enum_cache[topology] = action_enum
    return action_enum


__all__ = [
    "GERMINATE_PREFIX",
    "build_action_enum",
    "get_blueprint_from_action_name",
    "is_germinate_action_name",
]
