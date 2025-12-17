"""Policy registry for hotswappable Tamiyo policies."""

from __future__ import annotations

from typing import Type, TypeVar

from esper.tamiyo.policy.protocol import PolicyBundle

T = TypeVar("T", bound=PolicyBundle)

_REGISTRY: dict[str, Type[PolicyBundle]] = {}


def register_policy(name: str):
    """Decorator to register a PolicyBundle implementation.

    Usage:
        @register_policy("lstm")
        class LSTMPolicyBundle:
            ...

    Args:
        name: Unique name for this policy (used in config files)

    Returns:
        Decorator that registers the class

    Raises:
        TypeError: If the class doesn't implement PolicyBundle protocol
        ValueError: If name is already registered
    """
    def decorator(cls: Type[T]) -> Type[T]:
        # Validate protocol compliance
        if not isinstance(cls, type):
            raise TypeError(f"{cls} is not a class")

        # Check for required methods
        required_methods = [
            'process_signals', 'get_action', 'forward', 'evaluate_actions',
            'get_q_values', 'sync_from', 'get_value', 'initial_hidden',
            'state_dict', 'load_state_dict', 'to', 'enable_gradient_checkpointing',
        ]
        required_properties = ['is_recurrent', 'supports_off_policy', 'device', 'dtype']

        # hasattr AUTHORIZED by John on 2025-12-17 06:30:00 UTC
        # Justification: Feature Detection - checking if class implements required
        # PolicyBundle protocol methods at registration time. Cannot instantiate
        # to use isinstance() since policies require constructor args.
        missing_methods = [m for m in required_methods if not hasattr(cls, m)]
        missing_props = [p for p in required_properties if not hasattr(cls, p)]

        if missing_methods or missing_props:
            missing = missing_methods + missing_props
            raise TypeError(
                f"{cls.__name__} does not implement PolicyBundle protocol. "
                f"Missing: {', '.join(missing)}"
            )

        if name in _REGISTRY:
            raise ValueError(f"Policy '{name}' is already registered")

        _REGISTRY[name] = cls
        return cls

    return decorator


def get_policy(name: str, config: dict) -> PolicyBundle:
    """Factory function to instantiate a policy by name.

    Args:
        name: Registered policy name (e.g., "lstm", "heuristic")
        config: Configuration dict passed to policy constructor

    Returns:
        Instantiated PolicyBundle

    Raises:
        ValueError: If name is not registered
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys())) or "(none registered)"
        raise ValueError(f"Unknown policy: '{name}'. Available: {available}")

    return _REGISTRY[name](**config)


def list_policies() -> list[str]:
    """List all registered policy names."""
    return list(_REGISTRY.keys())


def clear_registry() -> None:
    """Clear all registered policies. For testing only."""
    _REGISTRY.clear()


__all__ = [
    "register_policy",
    "get_policy",
    "list_policies",
    "clear_registry",
]
