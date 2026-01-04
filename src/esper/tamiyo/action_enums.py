"""Tamiyo Action Enums - Dynamic action enum construction for heuristic policies.

This module builds topology-specific action enums from Leyline's BlueprintAction.
The enums are used by HeuristicTamiyo for baseline comparison and by TaskSpec
for flat action space representation.

Note: PPO training uses factored actions from esper.leyline.factored_actions.
This flat action enum is only for the heuristic baseline policy.

Architecture Note:
    This module uses Leyline's BlueprintAction and topology sets directly,
    avoiding the Kasmina dependency (and thus torch import overhead). This
    enables HeuristicTamiyo to run without torch for fast CI sweeps and
    hyperparameter exploration.
"""

from enum import IntEnum

from esper.leyline.factored_actions import (
    BlueprintAction,
    CNN_BLUEPRINTS,
    TRANSFORMER_BLUEPRINTS,
)

# Cache for built enums, keyed by topology
# Simple string key since BlueprintAction is static (no dynamic registry)
_action_enum_cache: dict[str, type[IntEnum]] = {}


def _get_topology_blueprints(topology: str) -> frozenset[BlueprintAction]:
    """Get valid blueprints for a topology.

    Args:
        topology: The topology type ("cnn" or "transformer")

    Returns:
        Frozenset of BlueprintAction values valid for this topology.

    Raises:
        ValueError: If topology is not recognized.
    """
    if topology == "cnn":
        return CNN_BLUEPRINTS
    elif topology == "transformer":
        return TRANSFORMER_BLUEPRINTS
    else:
        raise ValueError(f"Unknown topology: {topology}")


def build_action_enum(topology: str) -> type[IntEnum]:
    """Build action enum from Leyline's BlueprintAction for a topology.

    Action layout:
        0: WAIT
        1-N: GERMINATE_<BLUEPRINT> (sorted by enum value, NOOP excluded)
        N+1: FOSSILIZE
        N+2: PRUNE
        N+3: ADVANCE

    Note: This is used by HeuristicTamiyo for baseline comparison.
    PPO training uses factored actions instead.

    Args:
        topology: The topology type ("cnn" or "transformer")

    Returns:
        IntEnum class with action members for this topology.
    """
    if topology in _action_enum_cache:
        return _action_enum_cache[topology]

    blueprints = _get_topology_blueprints(topology)

    # Filter out NOOP (not a germination target) and sort by value for stable ordering
    germination_blueprints = sorted(
        (bp for bp in blueprints if bp != BlueprintAction.NOOP),
        key=lambda bp: bp.value,
    )

    members = {"WAIT": 0}
    for i, blueprint in enumerate(germination_blueprints, start=1):
        # Use the blueprint name (e.g., CONV_LIGHT -> GERMINATE_CONV_LIGHT)
        members[f"GERMINATE_{blueprint.name}"] = i

    members["FOSSILIZE"] = len(germination_blueprints) + 1
    members["PRUNE"] = len(germination_blueprints) + 2
    members["ADVANCE"] = len(germination_blueprints) + 3

    # Build list of tuples for IntEnum (avoids string literal requirement)
    member_list = list(members.items())
    # Type ignore needed: IntEnum expects string literal but we build dynamically
    action_enum = IntEnum(f"{topology.title()}Action", member_list)  # type: ignore[misc]
    _action_enum_cache[topology] = action_enum
    return action_enum


def clear_action_enum_cache() -> None:
    """Clear the action enum cache (for testing)."""
    _action_enum_cache.clear()


__all__ = ["build_action_enum", "clear_action_enum_cache"]
