"""Tests for PPO ratio and KL metric edge cases."""

from __future__ import annotations

import pytest
import torch

from esper.simic.agent.ppo_update import compute_ratio_metrics


def test_zero_mask_head_does_not_dilute_approx_kl() -> None:
    """All-zero causal heads contribute no denominator weight to approx_kl."""
    old_log_probs = {
        "op": torch.zeros(4),
        "slot": torch.zeros(4),
    }
    log_probs = {
        "op": torch.full((4,), 0.5),
        "slot": torch.full((4,), 10.0),
    }
    head_masks = {
        "op": torch.ones(4),
        "slot": torch.zeros(4),
    }

    metrics = compute_ratio_metrics(
        log_probs=log_probs,
        old_log_probs=old_log_probs,
        head_masks=head_masks,
        clip_ratio=0.2,
        target_kl=0.09,
        head_names=("op", "slot"),
        total_timesteps=torch.tensor(4.0),
    )

    expected_op_kl = (torch.exp(torch.tensor(0.5)) - 1.0) - 0.5
    assert metrics.approx_kl.item() == pytest.approx(expected_op_kl.item())
    assert metrics.early_stop.item() is True


def test_all_zero_masks_produce_finite_zero_approx_kl() -> None:
    """A fully masked batch has no KL weight and should report explicit zero KL."""
    old_log_probs = {
        "op": torch.zeros(4),
        "slot": torch.zeros(4),
    }
    log_probs = {
        "op": torch.full((4,), 0.5),
        "slot": torch.full((4,), 0.5),
    }
    head_masks = {
        "op": torch.zeros(4),
        "slot": torch.zeros(4),
    }

    metrics = compute_ratio_metrics(
        log_probs=log_probs,
        old_log_probs=old_log_probs,
        head_masks=head_masks,
        clip_ratio=0.2,
        target_kl=None,
        head_names=("op", "slot"),
        total_timesteps=torch.tensor(4.0),
    )

    assert torch.isfinite(metrics.approx_kl)
    assert metrics.approx_kl.item() == pytest.approx(0.0)
