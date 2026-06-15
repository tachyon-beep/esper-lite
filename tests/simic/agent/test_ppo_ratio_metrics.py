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


def test_per_head_clip_fraction_is_per_head() -> None:
    """per_head_clip_fraction reports each head's own clipping, not the joint's.

    The 'op' head ratio is driven far outside [1-clip, 1+clip] for every sample
    (fraction 1.0); the 'slot' head ratio is exactly 1.0 (fraction 0.0). This is
    the head-level companion to the joint clip_fraction (esper-lite-deb6b11575).
    """
    old_log_probs = {
        "op": torch.zeros(4),
        "slot": torch.zeros(4),
    }
    log_probs = {
        "op": torch.full((4,), 10.0),  # ratio = exp(10) >> 1 + clip -> clipped
        "slot": torch.zeros(4),  # ratio = 1.0 -> never clipped
    }
    head_masks = {
        "op": torch.ones(4),
        "slot": torch.ones(4),
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

    assert set(metrics.per_head_clip_fraction.keys()) == {"op", "slot"}
    assert metrics.per_head_clip_fraction["op"].item() == pytest.approx(1.0)
    assert metrics.per_head_clip_fraction["slot"].item() == pytest.approx(0.0)
    for value in metrics.per_head_clip_fraction.values():
        assert 0.0 <= value.item() <= 1.0
    # The joint clip_fraction is its own (joint-ratio) quantity, computed
    # independently: joint ratio = exp(10) for every sample -> fully clipped.
    assert metrics.clip_fraction.item() == pytest.approx(1.0)


def test_per_head_clip_fraction_uses_causal_mask() -> None:
    """Masked timesteps do not count as clipped for sparse action heads."""
    old_log_probs = {
        "op": torch.zeros(4),
        "blueprint": torch.zeros(4),
    }
    log_probs = {
        "op": torch.zeros(4),
        "blueprint": torch.log(torch.tensor([1.0, 2.0, 2.0, 2.0])),
    }
    head_masks = {
        "op": torch.ones(4),
        "blueprint": torch.tensor([1.0, 0.0, 0.0, 0.0]),
    }

    metrics = compute_ratio_metrics(
        log_probs=log_probs,
        old_log_probs=old_log_probs,
        head_masks=head_masks,
        clip_ratio=0.2,
        target_kl=None,
        head_names=("op", "blueprint"),
        total_timesteps=torch.tensor(4.0),
    )

    assert metrics.per_head_clip_fraction["blueprint"].item() == pytest.approx(0.0)
