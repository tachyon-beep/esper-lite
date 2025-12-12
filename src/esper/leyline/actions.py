"""Leyline Actions - Action space definitions for Esper agents.

Actions represent the discrete choices available to the strategic controller.
Per-topology action enums are built dynamically from registered blueprints.
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
        N+2: CULL
    """
    if topology in _action_enum_cache:
        return _action_enum_cache[topology]

    from esper.kasmina.blueprints import BlueprintRegistry

    blueprints = BlueprintRegistry.list_for_topology(topology)

    members = {"WAIT": 0}
    for i, spec in enumerate(blueprints, start=1):
        members[f"GERMINATE_{spec.name.upper()}"] = i
    members["FOSSILIZE"] = len(blueprints) + 1
    members["CULL"] = len(blueprints) + 2

    enum_name = f"{topology.title()}Action"
    action_enum = IntEnum(enum_name, members)
    _action_enum_cache[topology] = action_enum
    return action_enum


def get_blueprint_from_action(action: IntEnum) -> str | None:
    """Get blueprint name from a germinate action."""
    name = action.name
    if name.startswith("GERMINATE_"):
        return name[len("GERMINATE_"):].lower()
    return None


def is_germinate_action(action: IntEnum) -> bool:
    """Check if action is any germinate variant."""
    return action.name.startswith("GERMINATE_")


class Action(IntEnum):
    """Legacy action enum for Phase 1 CNN-only code.

    Action layout (sorted by blueprint param_estimate):
        0: WAIT
        1: GERMINATE_NORM (100 params)
        2: GERMINATE_ATTENTION (2K params)
        3: GERMINATE_DEPTHWISE (4.8K params)
        4: GERMINATE_CONV_LIGHT (37K params)
        5: GERMINATE_CONV_HEAVY (74K params)
        6: FOSSILIZE
        7: CULL
    """

    WAIT = 0
    GERMINATE_NORM = 1
    GERMINATE_ATTENTION = 2
    GERMINATE_DEPTHWISE = 3
    GERMINATE_CONV_LIGHT = 4
    GERMINATE_CONV_HEAVY = 5
    FOSSILIZE = 6
    CULL = 7

    @classmethod
    def is_germinate(cls, action: "Action") -> bool:
        """Check if action is any germinate variant."""
        return action in (
            cls.GERMINATE_NORM,
            cls.GERMINATE_ATTENTION,
            cls.GERMINATE_DEPTHWISE,
            cls.GERMINATE_CONV_LIGHT,
            cls.GERMINATE_CONV_HEAVY,
        )

    @classmethod
    def get_blueprint_id(cls, action: "Action") -> str | None:
        """Get blueprint ID for germinate actions, None for others."""
        legacy_map = {
            cls.GERMINATE_NORM: "norm",
            cls.GERMINATE_ATTENTION: "attention",
            cls.GERMINATE_DEPTHWISE: "depthwise",
            cls.GERMINATE_CONV_LIGHT: "conv_light",
            cls.GERMINATE_CONV_HEAVY: "conv_heavy",
        }
        return legacy_map.get(action)


__all__ = [
    "Action",
    "build_action_enum",
    "get_blueprint_from_action",
    "is_germinate_action",
]
