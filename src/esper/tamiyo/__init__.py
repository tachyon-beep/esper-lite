"""Tamiyo - Strategic decision-making for Esper.

Tamiyo is the "brain" of the Esper system. She observes training signals
and makes strategic decisions about seed lifecycle management.

## Subpackages

- tamiyo.policy: PolicyBundle protocol and implementations (LSTM, Heuristic)

## Key Components

- SignalTracker: Aggregates training metrics into TrainingSignals
- PolicyBundle: Protocol for swappable policy implementations
- get_policy(): Factory function to instantiate policies by name
"""

from esper.tamiyo.decisions import TamiyoDecision
from esper.tamiyo.tracker import SignalTracker
from esper.tamiyo.heuristic import (
    TamiyoPolicy,
    HeuristicPolicyConfig,
    HeuristicTamiyo,
)

# Policy subpackage exports
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
