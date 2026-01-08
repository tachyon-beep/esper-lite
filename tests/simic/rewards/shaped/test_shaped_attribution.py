"""Executable spec: SHAPED attribution and anti-gaming.

These tests focus on the SHAPED branch of `compute_contribution_reward`:
- progress-based attribution (sqrt vs cap vs zero)
- proxy attribution when counterfactual is unavailable
- attribution discount for negative trajectories
- ratio penalty computation and its non-stacking guard
"""

from __future__ import annotations

import math

import pytest

from esper.simic.rewards import compute_contribution_reward

from tests.simic.rewards.shaped.harness import (
    LifecycleOp,
    SeedStage,
    seed_info,
    shaped_config,
    with_prune_good_seed_penalty,
)


def test_shaped_negative_counterfactual_is_immediate_penalty() -> None:
    config = shaped_config(contribution_weight=2.0)

    reward, components = compute_contribution_reward(
        action=LifecycleOp.WAIT,
        seed_contribution=-0.5,
        val_acc=70.0,
        seed_info=seed_info(stage=SeedStage.BLENDING, total_improvement=0.0),
        epoch=3,
        max_epochs=10,
        acc_at_germination=65.0,
        acc_delta=0.0,
        config=config,
        return_components=True,
    )

    assert components.bounded_attribution == pytest.approx(-1.0)
    assert reward == pytest.approx(-1.0)


def test_shaped_geometric_mean_when_contribution_exceeds_progress() -> None:
    config = shaped_config()

    reward, components = compute_contribution_reward(
        action=LifecycleOp.WAIT,
        seed_contribution=9.0,
        val_acc=70.0,
        seed_info=seed_info(stage=SeedStage.BLENDING, total_improvement=2.0),
        epoch=4,
        max_epochs=10,
        acc_at_germination=66.0,  # progress = 4.0
        config=config,
        return_components=True,
    )

    # sqrt(progress * contribution) = sqrt(4 * 9) = 6.0
    assert components.bounded_attribution == pytest.approx(6.0)
    assert reward == pytest.approx(6.0)


def test_shaped_caps_at_contribution_when_contribution_below_progress() -> None:
    config = shaped_config()

    reward, components = compute_contribution_reward(
        action=LifecycleOp.WAIT,
        seed_contribution=3.0,
        val_acc=70.0,
        seed_info=seed_info(stage=SeedStage.BLENDING, total_improvement=1.0),
        epoch=5,
        max_epochs=10,
        acc_at_germination=60.0,  # progress = 10.0
        config=config,
        return_components=True,
    )

    assert components.bounded_attribution == pytest.approx(3.0)
    assert reward == pytest.approx(3.0)


def test_shaped_zero_reward_without_progress() -> None:
    config = shaped_config()

    reward, components = compute_contribution_reward(
        action=LifecycleOp.WAIT,
        seed_contribution=5.0,
        val_acc=65.0,
        seed_info=seed_info(stage=SeedStage.BLENDING, total_improvement=1.0),
        epoch=6,
        max_epochs=10,
        acc_at_germination=65.0,  # progress = 0.0
        config=config,
        return_components=True,
    )

    assert components.bounded_attribution == pytest.approx(0.0)
    assert reward == pytest.approx(0.0)


def test_shaped_half_credit_when_progress_unknown() -> None:
    config = shaped_config()

    reward, components = compute_contribution_reward(
        action=LifecycleOp.WAIT,
        seed_contribution=4.0,
        val_acc=70.0,
        seed_info=seed_info(stage=SeedStage.BLENDING, total_improvement=1.0),
        epoch=7,
        max_epochs=10,
        acc_at_germination=None,  # progress unknown
        config=config,
        return_components=True,
    )

    assert components.bounded_attribution == pytest.approx(2.0)
    assert reward == pytest.approx(2.0)


def test_shaped_attribution_discount_scales_attribution_for_negative_total() -> None:
    config = shaped_config(contribution_weight=1.0)

    reward, components = compute_contribution_reward(
        action=LifecycleOp.WAIT,
        seed_contribution=2.0,
        val_acc=65.0,
        # Use TRAINING to avoid BLENDING warning interfering with the discount spec.
        seed_info=seed_info(stage=SeedStage.TRAINING, total_improvement=-0.5),
        epoch=8,
        max_epochs=10,
        acc_at_germination=60.0,  # progress = 5.0
        config=config,
        return_components=True,
    )

    # progress > 0 and contribution < progress => attributed = contribution
    attributed = 2.0
    exp_arg = min(-3.0 * -0.5, 700.0)
    expected_discount = 1.0 / (1.0 + math.exp(exp_arg))
    expected_attribution = attributed * expected_discount

    assert components.attribution_discount == pytest.approx(expected_discount)
    assert components.bounded_attribution == pytest.approx(expected_attribution)
    assert reward == pytest.approx(expected_attribution)


def test_shaped_ratio_penalty_danger_zone_subtracts_from_reward() -> None:
    base_config = shaped_config(disable_anti_gaming=False)
    config = with_prune_good_seed_penalty(base_config, prune_good_seed_penalty=-0.3)

    reward, components = compute_contribution_reward(
        action=LifecycleOp.WAIT,
        seed_contribution=2.0,  # >1.0 enables ratio penalty path
        val_acc=64.0,
        seed_info=seed_info(stage=SeedStage.BLENDING, total_improvement=0.05),
        epoch=9,
        max_epochs=10,
        acc_at_germination=None,  # attributed = 0.5 * contribution = 1.0
        config=config,
        return_components=True,
    )

    assert components.ratio_penalty == pytest.approx(-0.06)
    assert components.bounded_attribution == pytest.approx(0.94)
    assert reward == pytest.approx(0.94)


def test_shaped_ratio_penalty_high_ratio_uses_escalating_formula() -> None:
    base_config = shaped_config(contribution_weight=0.0, disable_anti_gaming=False)
    config = with_prune_good_seed_penalty(base_config, prune_good_seed_penalty=-0.3)

    reward, components = compute_contribution_reward(
        action=LifecycleOp.WAIT,
        seed_contribution=10.0,
        val_acc=0.0,
        seed_info=seed_info(stage=SeedStage.BLENDING, total_improvement=1.0),
        epoch=1,
        max_epochs=10,
        acc_at_germination=0.0,
        config=config,
        return_components=True,
    )

    assert components.ratio_penalty == pytest.approx(-0.1)
    assert components.bounded_attribution == pytest.approx(-0.1)
    assert reward == pytest.approx(-0.1)


def test_shaped_ratio_penalty_skipped_when_discount_is_small() -> None:
    base_config = shaped_config(disable_anti_gaming=False)
    config = with_prune_good_seed_penalty(base_config, prune_good_seed_penalty=-0.3)

    _, components = compute_contribution_reward(
        action=LifecycleOp.WAIT,
        seed_contribution=10.0,
        val_acc=65.0,
        seed_info=seed_info(stage=SeedStage.BLENDING, total_improvement=-2.0),
        epoch=2,
        max_epochs=10,
        acc_at_germination=60.0,
        config=config,
        return_components=True,
    )

    assert components.attribution_discount < 0.5
    assert components.ratio_penalty == 0.0


def test_shaped_prune_inverts_attribution_sign() -> None:
    config = shaped_config()

    reward, components = compute_contribution_reward(
        action=LifecycleOp.PRUNE,
        seed_contribution=3.0,
        val_acc=70.0,
        seed_info=seed_info(stage=SeedStage.BLENDING, total_improvement=1.0),
        epoch=4,
        max_epochs=10,
        acc_at_germination=65.0,  # progress=5, attributed=3
        config=config,
        return_components=True,
    )

    assert components.bounded_attribution == pytest.approx(-3.0)
    assert reward == pytest.approx(-3.0)


def test_timing_discount_config_defaults() -> None:
    """D3-Timing: Config should have timing discount parameters with sensible defaults."""
    from esper.simic.rewards import ContributionRewardConfig

    config = ContributionRewardConfig()

    # Default warmup period: 10 epochs before full credit
    assert config.germination_warmup_epochs == 10
    # Default floor: epoch-1 germination gets 40% credit
    assert config.germination_discount_floor == 0.4
    # Default: timing discount enabled
    assert config.disable_timing_discount is False


def test_timing_discount_epoch_1_interpolates() -> None:
    """D3-Timing: Epoch 1 germination gets interpolated discount."""
    from esper.simic.rewards.contribution import _compute_timing_discount

    # Epoch 1 out of 10 warmup, floor=0.4
    # discount = 0.4 + (1 - 0.4) * (1 / 10) = 0.4 + 0.06 = 0.46
    discount = _compute_timing_discount(
        germination_epoch=1,
        warmup_epochs=10,
        discount_floor=0.4,
    )
    assert discount == pytest.approx(0.46)


def test_timing_discount_at_warmup_gets_full_credit() -> None:
    """D3-Timing: Germination at warmup epoch gets full credit."""
    from esper.simic.rewards.contribution import _compute_timing_discount

    discount = _compute_timing_discount(
        germination_epoch=10,
        warmup_epochs=10,
        discount_floor=0.4,
    )
    assert discount == pytest.approx(1.0)


def test_timing_discount_after_warmup_gets_full_credit() -> None:
    """D3-Timing: Germination after warmup gets full credit."""
    from esper.simic.rewards.contribution import _compute_timing_discount

    discount = _compute_timing_discount(
        germination_epoch=50,
        warmup_epochs=10,
        discount_floor=0.4,
    )
    assert discount == pytest.approx(1.0)


def test_timing_discount_mid_warmup_interpolates() -> None:
    """D3-Timing: Mid-warmup germination gets linearly interpolated discount."""
    from esper.simic.rewards.contribution import _compute_timing_discount

    # Epoch 5 out of 10 warmup, floor=0.4
    # discount = 0.4 + (1 - 0.4) * (5 / 10) = 0.4 + 0.3 = 0.7
    discount = _compute_timing_discount(
        germination_epoch=5,
        warmup_epochs=10,
        discount_floor=0.4,
    )
    assert discount == pytest.approx(0.7)


def test_timing_discount_epoch_0_gets_floor() -> None:
    """D3-Timing: Edge case - epoch 0 gets discount_floor."""
    from esper.simic.rewards.contribution import _compute_timing_discount

    discount = _compute_timing_discount(
        germination_epoch=0,
        warmup_epochs=10,
        discount_floor=0.4,
    )
    assert discount == pytest.approx(0.4)
