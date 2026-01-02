"""Blueprint Registry - Plugin system for seed blueprints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch.nn as nn

# Import the public cache invalidation API from Leyline.
# Note: This creates a Kasmina -> Leyline dependency, but Leyline.actions also
# imports from Kasmina (BlueprintRegistry). The import cycle is safe because:
#   1. This module only uses invalidate_action_enum_cache (a simple function)
#   2. The import happens at module level, not inside a function
#   3. Both modules complete their definitions before cross-referencing
from esper.leyline.actions import invalidate_action_enum_cache


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
    ) -> Callable[[Callable[[int], nn.Module]], Callable[[int], nn.Module]]:
        """Decorator to register a blueprint factory."""

        def decorator(factory: Callable[[int], nn.Module]) -> Callable[[int], nn.Module]:
            key = f"{topology}:{name}"
            cls._blueprints[key] = BlueprintSpec(
                name=name,
                topology=topology,
                factory=factory,
                param_estimate=param_estimate,
                description=description,
            )
            invalidate_action_enum_cache(topology)
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
    def create(cls, topology: str, name: str, dim: int, **kwargs: Any) -> nn.Module:
        """Create a module from a registered blueprint.

        Args:
            topology: The topology type ("cnn" or "transformer")
            name: The blueprint name (e.g., "norm", "lora", "attention")
            dim: Channel dimension (CNN) or embed dimension (transformer)
            **kwargs: Blueprint-specific options:
                - lora: rank (int, default 8)
                - attention: num_heads (int, auto-calculated if omitted)

        Returns:
            Instantiated nn.Module ready for seed injection.
        """
        spec = cls.get(topology, name)
        return spec.factory(dim, **kwargs)

    @classmethod
    def unregister(cls, topology: str, name: str) -> None:
        """Remove a blueprint from the registry (primarily for tests)."""
        key = f"{topology}:{name}"
        cls._blueprints.pop(key, None)
        invalidate_action_enum_cache(topology)


__all__ = ["BlueprintSpec", "BlueprintRegistry"]
