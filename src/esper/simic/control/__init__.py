"""Control interface between RL agent and environment.

This subpackage contains:
- normalization.py: Running statistics for observations/rewards

Re-exports from esper.tamiyo.policy:
- action_masks: compute_action_masks, MaskedCategorical, etc.
- features: obs_to_multislot_features, get_feature_size, etc.

The canonical location for these is esper.tamiyo.policy.
"""

from .normalization import RunningMeanStd, RewardNormalizer

# Re-export from tamiyo.policy (canonical location)
from esper.tamiyo.policy.features import (
    safe,
    TaskConfig,
    BASE_FEATURE_SIZE,
    SLOT_FEATURE_SIZE,
    MULTISLOT_FEATURE_SIZE,
    get_feature_size,
    obs_to_multislot_features,
)

from esper.tamiyo.policy.action_masks import (
    MaskSeedInfo,
    MaskedCategorical,
    InvalidStateMachineError,
    build_slot_states,
    compute_action_masks,
    compute_batch_masks,
    slot_id_to_index,
)

__all__ = [
    # Normalization
    "RunningMeanStd",
    "RewardNormalizer",
    # Features
    "safe",
    "TaskConfig",
    "BASE_FEATURE_SIZE",
    "SLOT_FEATURE_SIZE",
    "MULTISLOT_FEATURE_SIZE",
    "get_feature_size",
    "obs_to_multislot_features",
    # Action masks
    "MaskSeedInfo",
    "MaskedCategorical",
    "InvalidStateMachineError",
    "build_slot_states",
    "compute_action_masks",
    "compute_batch_masks",
    "slot_id_to_index",
]
