"""Blueprint Registry - Plugin system for seed blueprints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

import torch.nn as nn


class BlueprintFactory(Protocol):
    """Protocol for blueprint factory functions.

    Blueprint factories create nn.Module instances for seed injection.
    They take a dimension parameter and optional kwargs for blueprint-specific
    configuration (e.g., LoRA rank, attention heads).

    This protocol fixes the type mismatch where BlueprintSpec.factory was typed
    as Callable[[int], nn.Module] but called with **kwargs.
    """

    def __call__(self, dim: int, **kwargs: Any) -> nn.Module:
        """Create a module for the given dimension.

        Args:
            dim: Channel dimension (CNN) or embed dimension (transformer)
            **kwargs: Blueprint-specific options (e.g., rank for LoRA)

        Returns:
            Instantiated nn.Module ready for seed injection.
        """
        ...


@dataclass(frozen=True, slots=True)
class BlueprintSpec:
    """Specification for a registered blueprint."""

    name: str
    topology: str
    factory: BlueprintFactory
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
    ) -> Callable[[BlueprintFactory], BlueprintFactory]:
        """Decorator to register a blueprint factory.

        Note: Action enum caches (in tamiyo.action_enums) are automatically
        invalidated by version-keying on registry state. No callback needed.
        """

        def decorator(factory: BlueprintFactory) -> BlueprintFactory:
            key = f"{topology}:{name}"
            cls._blueprints[key] = BlueprintSpec(
                name=name,
                topology=topology,
                factory=factory,
                param_estimate=param_estimate,
                description=description,
            )
            return factory

        return decorator

    @classmethod
    def list_for_topology(cls, topology: str) -> list[BlueprintSpec]:
        """All blueprints for a topology, sorted by param estimate."""
        from .loader import ensure_loaded

        ensure_loaded(topology)
        return sorted(
            [s for s in cls._blueprints.values() if s.topology == topology],
            key=lambda s: s.param_estimate,
        )

    @classmethod
    def get(cls, topology: str, name: str) -> BlueprintSpec:
        """Get a specific blueprint spec."""
        from .loader import ensure_loaded

        ensure_loaded(topology)
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
        """Remove a blueprint from the registry (primarily for tests).

        Note: Action enum caches (in tamiyo.action_enums) are automatically
        invalidated by version-keying on registry state. No callback needed.
        """
        key = f"{topology}:{name}"
        cls._blueprints.pop(key, None)


__all__ = ["BlueprintFactory", "BlueprintSpec", "BlueprintRegistry"]
