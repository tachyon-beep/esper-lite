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

__all__ = [
    # Core
    "TamiyoDecision",
    "SignalTracker",
    # Legacy heuristic (kept for backwards compatibility)
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


from typing import Any


def __getattr__(name: str) -> Any:
    """Lazy import using PEP 562.

    Heavy modules (policy subpackage with torch, tracker with nissa) are only
    loaded when accessed, not at package import time.
    """
    # Decisions (lightweight)
    if name == "TamiyoDecision":
        from esper.tamiyo.decisions import TamiyoDecision
        return TamiyoDecision

    # Tracker (depends on nissa, but nissa is now lazy)
    if name == "SignalTracker":
        from esper.tamiyo.tracker import SignalTracker
        return SignalTracker

    # Heuristic (lightweight - no torch)
    if name in ("TamiyoPolicy", "HeuristicPolicyConfig", "HeuristicTamiyo"):
        from esper.tamiyo.heuristic import (
            TamiyoPolicy,
            HeuristicPolicyConfig,
            HeuristicTamiyo,
        )
        mapping: dict[str, Any] = {
            "TamiyoPolicy": TamiyoPolicy,
            "HeuristicPolicyConfig": HeuristicPolicyConfig,
            "HeuristicTamiyo": HeuristicTamiyo,
        }
        return mapping[name]

    # Policy subpackage (HEAVY - loads torch for LSTM registration)
    if name in ("PolicyBundle", "ActionResult", "EvalResult", "ForwardResult",
                "register_policy", "get_policy", "list_policies", "create_heuristic_policy"):
        from esper.tamiyo.policy import (
            PolicyBundle,
            ActionResult,
            EvalResult,
            ForwardResult,
            register_policy,
            get_policy,
            list_policies,
            create_heuristic_policy,
        )
        return {"PolicyBundle": PolicyBundle, "ActionResult": ActionResult,
                "EvalResult": EvalResult, "ForwardResult": ForwardResult,
                "register_policy": register_policy, "get_policy": get_policy,
                "list_policies": list_policies, "create_heuristic_policy": create_heuristic_policy}[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
