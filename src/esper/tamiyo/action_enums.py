"""Tamiyo Action Enums - Dynamic action enum construction for heuristic policies.

This module builds topology-specific action enums from registered blueprints.
The enums are used by HeuristicTamiyo for baseline comparison and by TaskSpec
for flat action space representation.

Note: PPO training uses factored actions from esper.leyline.factored_actions.
This flat action enum is only for the heuristic baseline policy.

Architecture Note:
    This module lives in Tamiyo (not Leyline) because it depends on Kasmina's
    BlueprintRegistry. Leyline is "pure contracts" and should not import
    domain packages. Tamiyo owns action selection logic, so dynamic action
    enum construction belongs here.
"""

from enum import IntEnum

from esper.kasmina.blueprints import BlueprintRegistry

# Cache for built enums, keyed by (topology, registry_version)
# The version ensures we rebuild when blueprints change without needing callbacks
_action_enum_cache: dict[tuple[str, int], type[IntEnum]] = {}

# Simple version counter - incremented when we detect registry changes
_registry_version: int = 0


def _get_registry_version(topology: str) -> int:
    """Get a version number representing current registry state for topology.

    Uses blueprint count + names as a cheap proxy for "has registry changed".
    This avoids needing a callback from BlueprintRegistry -> Tamiyo.
    """
    blueprints = BlueprintRegistry.list_for_topology(topology)
    # Hash the blueprint names to detect changes
    return hash(tuple(spec.name for spec in blueprints))


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

    Args:
        topology: The topology type ("cnn" or "transformer")

    Returns:
        IntEnum class with action members for this topology.
    """
    version = _get_registry_version(topology)
    cache_key = (topology, version)

    if cache_key in _action_enum_cache:
        return _action_enum_cache[cache_key]

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
    _action_enum_cache[cache_key] = action_enum
    return action_enum


def clear_action_enum_cache() -> None:
    """Clear the action enum cache (for testing)."""
    _action_enum_cache.clear()


__all__ = ["build_action_enum", "clear_action_enum_cache"]
