"""Blueprint Registry - Plugin system for seed blueprints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch.nn as nn


@dataclass(frozen=True, slots=True)
class BlueprintSpec:
    """Specification for a registered blueprint."""

    name: str
    topology: str
    factory: Callable[[int], nn.Module]
    param_estimate: int
    description: str = ""

    def actual_param_count(self, dim: int) -> int:
        """Compute actual param count for given dimension."""
        module = self.factory(dim)
        return sum(p.numel() for p in module.parameters())


class BlueprintRegistry:
    """Registry of available seed blueprints."""

    _blueprints: dict[str, BlueprintSpec] = {}

    @classmethod
    def register(
        cls,
        name: str,
        topology: str,
        param_estimate: int,
        description: str = "",
    ):
        """Decorator to register a blueprint factory."""

        def decorator(factory: Callable[[int], nn.Module]):
            key = f"{topology}:{name}"
            cls._blueprints[key] = BlueprintSpec(
                name=name,
                topology=topology,
                factory=factory,
                param_estimate=param_estimate,
                description=description,
            )
            # Invalidate cached action enums for this topology so new blueprints appear
            try:
                from esper.leyline import actions as leyline_actions  # Local import to avoid cycle
                leyline_actions._action_enum_cache.pop(topology, None)
            except (ImportError, AttributeError, KeyError):
                # Cache invalidation best-effort; import cycle or missing cache is acceptable
                pass
            return factory

        return decorator

    @classmethod
    def list_for_topology(cls, topology: str) -> list[BlueprintSpec]:
        """All blueprints for a topology, sorted by param estimate."""
        return sorted(
            [s for s in cls._blueprints.values() if s.topology == topology],
            key=lambda s: s.param_estimate,
        )

    @classmethod
    def get(cls, topology: str, name: str) -> BlueprintSpec:
        """Get a specific blueprint spec."""
        key = f"{topology}:{name}"
        if key not in cls._blueprints:
            available = cls.list_for_topology(topology)
            names = [s.name for s in available]
            raise ValueError(
                f"Unknown blueprint {name!r} for {topology}. Available: {names}"
            )
        return cls._blueprints[key]

    @classmethod
    def create(cls, topology: str, name: str, dim: int) -> nn.Module:
        """Create a module from a registered blueprint."""
        spec = cls.get(topology, name)
        return spec.factory(dim)

    @classmethod
    def unregister(cls, topology: str, name: str) -> None:
        """Remove a blueprint from the registry (primarily for tests)."""
        key = f"{topology}:{name}"
        cls._blueprints.pop(key, None)
        try:
            from esper.leyline import actions as leyline_actions  # Local import to avoid cycle
            leyline_actions._action_enum_cache.pop(topology, None)
        except (ImportError, AttributeError, KeyError):
            pass

    @classmethod
    def reset(cls) -> None:
        """Reset registry to empty state (for test cleanup)."""
        cls._blueprints.clear()
        try:
            from esper.leyline import actions as leyline_actions
            leyline_actions._action_enum_cache.clear()
        except (ImportError, AttributeError, KeyError):
            pass


__all__ = ["BlueprintSpec", "BlueprintRegistry"]
