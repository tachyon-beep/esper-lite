"""Tamiyo Policy - Hotswappable policy implementations.

This subpackage contains the PolicyBundle protocol and implementations:
- protocol.py: PolicyBundle interface definition
- types.py: ActionResult, EvalResult, ForwardResult dataclasses
- registry.py: Policy registration and factory
- features.py: Feature extraction for observations
- action_masks.py: Action masking for valid actions
- lstm_bundle.py: LSTM-based recurrent policy (Phase 3)
- heuristic_bundle.py: Rule-based heuristic (Phase 4)
"""

from esper.tamiyo.policy.protocol import PolicyBundle
from esper.tamiyo.policy.types import ActionResult, EvalResult, ForwardResult
from esper.tamiyo.policy.registry import (
    register_policy,
    get_policy,
    list_policies,
)
from esper.tamiyo.policy.features import (
    obs_to_multislot_features,
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
    # Features
    "obs_to_multislot_features",
    "get_feature_size",
    "BASE_FEATURE_SIZE",
    "SLOT_FEATURE_SIZE",
    "TaskConfig",
    # Action Masks
    "compute_action_masks",
    "compute_batch_masks",
    "MaskedCategorical",
]
