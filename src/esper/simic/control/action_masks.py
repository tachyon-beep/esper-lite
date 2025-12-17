"""Action Masking - Re-exports from tamiyo.policy.action_masks.

DEPRECATED: Import from esper.tamiyo.policy.action_masks instead.
This module exists for backwards compatibility during migration.
"""

# Re-export everything from tamiyo
from esper.tamiyo.policy.action_masks import (
    MaskSeedInfo,
    build_slot_states,
    compute_action_masks,
    compute_batch_masks,
    slot_id_to_index,
    MaskedCategorical,
    InvalidStateMachineError,
    _validate_action_mask,
)

__all__ = [
    "MaskSeedInfo",
    "build_slot_states",
    "compute_action_masks",
    "compute_batch_masks",
    "slot_id_to_index",
    "MaskedCategorical",
    "InvalidStateMachineError",
    "_validate_action_mask",
]
