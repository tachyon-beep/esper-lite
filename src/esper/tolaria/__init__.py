"""Tolaria - Model Alpha Training Infrastructure

This package owns the training loop for Model Alpha (the neural network being
enhanced with morphogenetic seeds). It provides:

- environment: Model factory
- trainer: Epoch training functions for different seed states

Tolaria is a generic trainer - dataset loading is handled by esper.utils.
Tolaria is tightly coupled with Kasmina (seed mechanics) and Tamiyo (decisions).
Simic uses Tolaria to create the RL environment for training Tamiyo.

Public API:
    from esper.tolaria import create_model
    from esper.tolaria import train_epoch_normal, train_epoch_seed_isolated
"""

from esper.tolaria.environment import create_model
from esper.tolaria.trainer import (
    train_epoch_normal,
    train_epoch_seed_isolated,
    train_epoch_blended,
    validate_and_get_metrics,
)

__all__ = [
    # Environment
    "create_model",
    # Trainer
    "train_epoch_normal",
    "train_epoch_seed_isolated",
    "train_epoch_blended",
    "validate_and_get_metrics",
]
