"""Simic Features - Re-exports from tamiyo.policy.features.

DEPRECATED: Import from esper.tamiyo.policy.features instead.
This module exists for backwards compatibility during migration.
"""

# Re-export everything from tamiyo
from esper.tamiyo.policy.features import (
    safe,
    obs_to_multislot_features,
    MULTISLOT_FEATURE_SIZE,
    get_feature_size,
    BASE_FEATURE_SIZE,
    SLOT_FEATURE_SIZE,
    TaskConfig,
)

__all__ = [
    "safe",
    "obs_to_multislot_features",
    "MULTISLOT_FEATURE_SIZE",
    "get_feature_size",
    "BASE_FEATURE_SIZE",
    "SLOT_FEATURE_SIZE",
    "TaskConfig",
]
