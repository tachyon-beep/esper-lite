"""Tolaria - Model Alpha Training Infrastructure

This package owns the training loop for Model Alpha (the neural network being
enhanced with morphogenetic seeds). It provides:

- environment: Model factory
- trainer: Epoch training functions for different seed states
- governor: Fail-safe watchdog for catastrophic failure detection

Tolaria is a generic trainer - dataset loading is handled by esper.utils.
Tolaria is tightly coupled with Kasmina (seed mechanics) and Tamiyo (decisions).
Simic uses Tolaria to create the RL environment for training Tamiyo.

Public API:
    from esper.tolaria import create_model
    from esper.tolaria import train_epoch_normal, train_epoch_incubator_mode
    from esper.tolaria import validate_with_attribution, AttributionResult
    from esper.tolaria import TolariaGovernor, GovernorReport
"""

from esper.tolaria.environment import create_model
from esper.tolaria.governor import GovernorReport, TolariaGovernor
from esper.tolaria.trainer import (
    train_epoch_normal,
    train_epoch_incubator_mode,
    train_epoch_blended,
    validate_and_get_metrics,
    validate_with_attribution,
    AttributionResult,
)

__all__ = [
    # Environment
    "create_model",
    # Trainer
    "train_epoch_normal",
    "train_epoch_incubator_mode",
    "train_epoch_blended",
    "validate_and_get_metrics",
    "validate_with_attribution",
    "AttributionResult",
    # Governor
    "TolariaGovernor",
    "GovernorReport",
]
