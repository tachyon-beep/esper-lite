"""Policy registry for hotswappable Tamiyo policies."""

from __future__ import annotations

from typing import Any, Callable, Type, TypeVar

from esper.leyline import PolicyBundle

T = TypeVar("T", bound=PolicyBundle)

_REGISTRY: dict[str, Type[PolicyBundle]] = {}


def register_policy(name: str) -> Callable[[Type[PolicyBundle]], Type[PolicyBundle]]:
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

    Note:
        This performs a **best-effort structural check** only:

        - Verifies required method/property names exist via hasattr
        - Does NOT validate call signatures or return types
        - Does NOT verify that "properties" are actually @property decorated

        **Signature validation is delegated to static type checking:**

        For full Protocol compliance validation including method signatures,
        use mypy or pyright with the PolicyBundle Protocol definition.
        CI pipelines should run `mypy --strict` or `pyright` to catch
        signature drift before runtime errors occur. Runtime registration
        only checks structural presence, not correctness.

        The most failure-prone methods (get_action, evaluate_actions, forward)
        have complex signatures that hasattr cannot validate.
    """
    def decorator(cls: Type[T]) -> Type[T]:
        # Validate protocol compliance
        if not isinstance(cls, type):
            raise TypeError(f"{cls} is not a class")

        # Check for required methods
        # Note: process_signals is NOT in this list - feature extraction is
        # handled by Simic's signals_to_features() which requires training context.
        required_methods = [
            'get_action', 'forward', 'evaluate_actions',
            'get_q_values', 'sync_from', 'get_value', 'initial_hidden',
            'state_dict', 'load_state_dict', 'to', 'enable_gradient_checkpointing',
            'compile',  # Required for torch.compile integration in factory
        ]
        required_properties = [
            'is_recurrent', 'supports_off_policy', 'device', 'dtype',
            'slot_config', 'feature_dim', 'hidden_dim', 'network', 'is_compiled',
        ]

        # Structural check via hasattr: We verify that the class has required
        # method/property names defined. This is necessary because we can't
        # instantiate the class here (policies require constructor arguments)
        # and Protocol conformance can't be checked at runtime without an instance.
        # Static type checkers (mypy/pyright) provide full signature validation.
        # hasattr AUTHORIZED by Code Review 2025-12-17
        # Justification: Protocol structural verification - checking class attributes without instantiation
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


def get_policy(name: str, config: dict[str, Any]) -> PolicyBundle:
    """Factory function to instantiate a policy by name.

    Args:
        name: Registered policy name (e.g., "lstm"). Use list_policies()
            to see available policies.
        config: Configuration dict passed to policy constructor

    Returns:
        Instantiated PolicyBundle

    Raises:
        ValueError: If name is not registered

    Note:
        The heuristic adapter is NOT registered here because it doesn't
        implement the full PolicyBundle interface. Use create_heuristic_policy()
        from esper.tamiyo.policy instead.
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
