"""Control interface between RL agent and environment.

This subpackage contains stateless utilities for:
- action_masks.py: Masked action distributions, slot state building
- features.py: Observation feature extraction (hot path)
- normalization.py: Running statistics for observations/rewards
"""

from .normalization import RunningMeanStd, RewardNormalizer

from .features import (
    safe,
    TaskConfig,
    BASE_FEATURE_SIZE,
    SLOT_FEATURE_SIZE,
    MULTISLOT_FEATURE_SIZE,
    get_feature_size,
    obs_to_multislot_features,
)

from .action_masks import (
    MaskSeedInfo,
    MaskedCategorical,
    InvalidStateMachineError,
    build_slot_states,
    compute_action_masks,
    compute_batch_masks,
    slot_id_to_index,
    _validate_action_mask,
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
    "_validate_action_mask",
]
