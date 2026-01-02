"""Tests for per-head NaN/Inf detection in PPO training."""

import pytest
import torch
from esper.leyline import HEAD_NAMES


def test_detect_nan_per_head():
    """Verify NaN detection separates NaN from Inf per head."""
    # Simulate log_probs with NaN in one head, Inf in another
    log_probs = {head: torch.zeros(10) for head in HEAD_NAMES}
    log_probs["op"][0] = float("nan")
    log_probs["slot"][0] = float("inf")

    head_nan_detected = {head: False for head in HEAD_NAMES}
    head_inf_detected = {head: False for head in HEAD_NAMES}

    # Fast-path pattern: only drill down when isfinite fails
    for head in HEAD_NAMES:
        lp = log_probs[head]
        if not torch.isfinite(lp).all():
            if torch.isnan(lp).any():
                head_nan_detected[head] = True
            if torch.isinf(lp).any():
                head_inf_detected[head] = True

    assert head_nan_detected["op"] is True
    assert head_nan_detected["slot"] is False
    assert head_inf_detected["op"] is False
    assert head_inf_detected["slot"] is True


def test_detect_nan_and_inf_same_tensor():
    """A tensor can have both NaN and Inf - both flags should be set."""
    lp = torch.tensor([float("nan"), float("inf"), 1.0])

    has_nan = bool(torch.isnan(lp).any().item())
    has_inf = bool(torch.isinf(lp).any().item())

    assert has_nan is True
    assert has_inf is True


def test_detect_negative_inf():
    """Negative infinity is common for log_probs of impossible actions."""
    lp = torch.tensor([float("-inf"), 1.0, 2.0])

    # torch.isinf detects both +inf and -inf
    assert bool(torch.isinf(lp).any().item()) is True
    assert bool(torch.isnan(lp).any().item()) is False


def test_empty_tensor_returns_false():
    """Empty tensors should not trigger NaN/Inf detection."""
    lp = torch.zeros(0)

    assert bool(torch.isnan(lp).any().item()) is False
    assert bool(torch.isinf(lp).any().item()) is False


def test_clean_tensor_fast_path():
    """Clean tensors should pass isfinite check without drilling down."""
    lp = torch.tensor([0.1, 0.2, 0.3])

    # Fast path: isfinite.all() returns True, no need to check nan/inf
    assert torch.isfinite(lp).all().item() is True
