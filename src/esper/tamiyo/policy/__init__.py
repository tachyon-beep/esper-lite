"""Tamiyo Policy - Hotswappable policy implementations.

This subpackage contains the PolicyBundle protocol and implementations:
- protocol.py: PolicyBundle interface definition
- types.py: ActionResult, EvalResult, ForwardResult dataclasses
- registry.py: Policy registration and factory
- features.py: Feature extraction for observations
- action_masks.py: Action masking for valid actions
- lstm_bundle.py: LSTM-based recurrent policy (Phase 3)
- heuristic_bundle.py: Rule-based heuristic adapter (NOT a full PolicyBundle)

Note on imports:
    Importing this package registers built-in neural policies (currently: "lstm").
    The heuristic adapter is NOT registered; use create_heuristic_policy() instead.

    This imports torch at module level (standard for a DRL package), but does
    NOT construct any models - construction is deferred to get_policy() calls.
    If import cost is a concern for non-training code paths, import specific
    submodules directly (e.g., `from esper.tamiyo.policy.protocol import PolicyBundle`).
"""

# Core protocol and types now from leyline
from esper.leyline import PolicyBundle, ActionResult, EvalResult, ForwardResult
from esper.tamiyo.policy.registry import (
    register_policy,
    get_policy,
    list_policies,
)
from esper.tamiyo.policy.factory import create_policy
from esper.tamiyo.policy.features import (
    get_feature_size,
    BASE_FEATURE_SIZE,
    SLOT_FEATURE_SIZE,
    TaskConfig,
)
from esper.tamiyo.policy.action_masks import (
    compute_action_masks,
    compute_batch_masks,
    MaskedCategorical,
)

# Import to trigger registration (must be after registry is defined)
from esper.tamiyo.policy import lstm_bundle as _lstm_bundle  # noqa: F401

# Heuristic imports are lazy - only loaded when accessed via __getattr__ or
# create_heuristic_policy(). This reduces import cost since heuristic is not
# registered and many code paths don't need it.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from esper.tamiyo.policy.heuristic_bundle import HeuristicPolicyBundle
    from esper.tamiyo.heuristic import HeuristicPolicyConfig


def create_heuristic_policy(
    config: "HeuristicPolicyConfig | None" = None,
    topology: str = "cnn",
) -> "HeuristicPolicyBundle":
    """Create a heuristic policy adapter.

    This is the recommended way to create heuristic policies. Unlike neural
    policies, the heuristic is not registered in the policy registry because
    it doesn't implement the full PolicyBundle interface.

    Args:
        config: Heuristic policy configuration
        topology: Model topology ("cnn" or "transformer")

    Returns:
        HeuristicPolicyBundle instance.
    """
    # Lazy import to avoid loading heuristic_bundle at package import time
    from esper.tamiyo.policy.heuristic_bundle import HeuristicPolicyBundle

    return HeuristicPolicyBundle(config=config, topology=topology)


def __getattr__(name: str) -> type:
    """Lazy import for HeuristicPolicyBundle (not needed for registration)."""
    if name == "HeuristicPolicyBundle":
        from esper.tamiyo.policy.heuristic_bundle import HeuristicPolicyBundle
        return HeuristicPolicyBundle
    if name == "HeuristicPolicyConfig":
        from esper.tamiyo.heuristic import HeuristicPolicyConfig
        return HeuristicPolicyConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Protocol
    "PolicyBundle",
    # Types
    "ActionResult",
    "EvalResult",
    "ForwardResult",
    # Registry
    "register_policy",
    "get_policy",
    "list_policies",
    # Factory
    "create_policy",
    # Neural policy bundles (via registry)
    # Note: LSTMPolicyBundle available via get_policy("lstm")
    # Heuristic adapter (NOT registered - use create_heuristic_policy())
    "HeuristicPolicyBundle",
    "HeuristicPolicyConfig",
    "create_heuristic_policy",
    # Features
    "get_feature_size",
    "BASE_FEATURE_SIZE",
    "SLOT_FEATURE_SIZE",
    "TaskConfig",
    # Action Masks
    "compute_action_masks",
    "compute_batch_masks",
    "MaskedCategorical",
]
