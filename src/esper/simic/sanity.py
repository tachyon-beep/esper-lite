"""Sanity Check Utilities for Simic Training."""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

SANITY_CHECKS_ENABLED = os.getenv("ESPER_SANITY_CHECKS", "0") == "1"


def check_reward_magnitude(reward: float, epoch: int, max_epochs: int, threshold: float = 10.0) -> None:
    """Warn if reward magnitude is unexpectedly large."""
    if abs(reward) > threshold:
        logger.warning(
            f"Large reward magnitude {reward:.2f} at epoch {epoch}/{max_epochs}. "
            "Consider adjusting loss_delta_weight or typical_loss_delta_std."
        )


def log_params_ratio(total_params: int, host_params: int, epoch: int) -> None:
    """Log params ratio for debugging rent calibration."""
    if host_params > 0:
        ratio = total_params / host_params
        logger.debug(f"Epoch {epoch}: params_ratio={ratio:.3f} ({total_params}/{host_params})")


def assert_slot_shape(x: torch.Tensor, expected_dim: int, topology: str) -> None:
    """Assert tensor has expected dimension for slot."""
    if topology == "cnn":
        if x.dim() != 4:
            raise AssertionError(f"CNN slot expects 4D tensor, got {x.dim()}D")
        actual = x.shape[1]
    elif topology == "transformer":
        if x.dim() != 3:
            raise AssertionError(f"Transformer slot expects 3D tensor, got {x.dim()}D")
        actual = x.shape[2]
    else:
        raise ValueError(f"Unknown topology: {topology}")

    if actual != expected_dim:
        raise AssertionError(f"Slot dimension mismatch: expected {expected_dim}, got {actual}")


__all__ = [
    "check_reward_magnitude",
    "log_params_ratio",
    "assert_slot_shape",
]
