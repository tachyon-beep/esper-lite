"""Leyline Actions - Action space definitions for Esper agents.

Actions represent the discrete choices available to the strategic controller.
Per-topology action enums are built dynamically from registered blueprints.

Note: The flat action enum is used only by HeuristicTamiyo for baseline comparison.
PPO training uses factored actions from esper.leyline.factored_actions.
"""

from enum import IntEnum

# Cache for built enums
_action_enum_cache: dict[str, type[IntEnum]] = {}


def build_action_enum(topology: str) -> type[IntEnum]:
    """Build action enum from registered blueprints for a topology.

    Action layout:
        0: WAIT
        1-N: GERMINATE_<BLUEPRINT> (sorted by param count)
        N+1: FOSSILIZE
        N+2: PRUNE

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

    enum_name = f"{topology.title()}Action"
    action_enum = IntEnum(enum_name, members)
    _action_enum_cache[topology] = action_enum
    return action_enum


__all__ = [
    "build_action_enum",
]
