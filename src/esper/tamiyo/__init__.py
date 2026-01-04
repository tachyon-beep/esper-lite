"""Tamiyo - Strategic decision-making for Esper.

Tamiyo is the "brain" of the Esper system. She observes training signals
and makes strategic decisions about seed lifecycle management.

## Subpackages

- tamiyo.policy: PolicyBundle protocol and implementations (LSTM, Heuristic)

## Key Components

- SignalTracker: Aggregates training metrics into TrainingSignals
- PolicyBundle: Protocol for swappable policy implementations
- get_policy(): Factory function to instantiate policies by name

NOTE: This module uses PEP 562 lazy imports. Heavy modules (policy with torch,
tracker with nissa dependency) are only loaded when accessed.
"""

from typing import TYPE_CHECKING, Any

__all__ = [
    # Core
    "TamiyoDecision",
    "SignalTracker",
    # Heuristic policy (baseline for comparison - no torch dependency)
    "TamiyoPolicy",
    "HeuristicPolicyConfig",
    "HeuristicTamiyo",
    # Policy interface
    "PolicyBundle",
    "ActionResult",
    "EvalResult",
    "ForwardResult",
    "register_policy",
    "get_policy",
    "list_policies",
    "create_heuristic_policy",
]

# TYPE_CHECKING imports for static analysis (mypy, IDE autocomplete)
# These are never executed at runtime, preserving lazy import behavior.
if TYPE_CHECKING:
    from esper.tamiyo.decisions import TamiyoDecision as TamiyoDecision
    from esper.tamiyo.heuristic import HeuristicPolicyConfig as HeuristicPolicyConfig
    from esper.tamiyo.heuristic import HeuristicTamiyo as HeuristicTamiyo
    from esper.tamiyo.heuristic import TamiyoPolicy as TamiyoPolicy
    from esper.tamiyo.policy import ActionResult as ActionResult
    from esper.tamiyo.policy import EvalResult as EvalResult
    from esper.tamiyo.policy import ForwardResult as ForwardResult
    from esper.tamiyo.policy import PolicyBundle as PolicyBundle
    from esper.tamiyo.policy import create_heuristic_policy as create_heuristic_policy
    from esper.tamiyo.policy import get_policy as get_policy
    from esper.tamiyo.policy import list_policies as list_policies
    from esper.tamiyo.policy import register_policy as register_policy
    from esper.tamiyo.tracker import SignalTracker as SignalTracker


# Mapping from public name to (module, attr) for lazy imports
# Single source of truth for __getattr__ and __dir__
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # Core (lightweight)
    "TamiyoDecision": ("esper.tamiyo.decisions", "TamiyoDecision"),
    "SignalTracker": ("esper.tamiyo.tracker", "SignalTracker"),
    # Heuristic (no torch dependency after refactor)
    "TamiyoPolicy": ("esper.tamiyo.heuristic", "TamiyoPolicy"),
    "HeuristicPolicyConfig": ("esper.tamiyo.heuristic", "HeuristicPolicyConfig"),
    "HeuristicTamiyo": ("esper.tamiyo.heuristic", "HeuristicTamiyo"),
    # Policy subpackage (HEAVY - loads torch for LSTM registration)
    "PolicyBundle": ("esper.tamiyo.policy", "PolicyBundle"),
    "ActionResult": ("esper.tamiyo.policy", "ActionResult"),
    "EvalResult": ("esper.tamiyo.policy", "EvalResult"),
    "ForwardResult": ("esper.tamiyo.policy", "ForwardResult"),
    "register_policy": ("esper.tamiyo.policy", "register_policy"),
    "get_policy": ("esper.tamiyo.policy", "get_policy"),
    "list_policies": ("esper.tamiyo.policy", "list_policies"),
    "create_heuristic_policy": ("esper.tamiyo.policy", "create_heuristic_policy"),
}


def __getattr__(name: str) -> Any:
    """Lazy import using PEP 562.

    Defers heavy imports (torch, telemetry hub) until actual use.
    Caches the result in module globals to avoid repeated imports.
    """
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        import importlib

        module = importlib.import_module(module_path)
        value = getattr(module, attr_name)
        # Cache in module globals so subsequent accesses bypass __getattr__
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return public API for dir() and IDE discovery."""
    return sorted(__all__)
