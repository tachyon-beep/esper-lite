"""Tests to prevent PBRS lifecycle farming.

These tests exercise the real reward implementation (compute_contribution_reward)
to ensure a Germinate→(Wait...)→Prune loop cannot produce positive discounted
return purely from shaping.
"""

from __future__ import annotations

import math

from esper.leyline import LifecycleOp, SeedStage
from esper.simic.rewards import ContributionRewardConfig, SeedInfo, compute_contribution_reward


def _make_seed_info(
    *,
    stage: SeedStage,
    epochs_in_stage: int,
    previous_stage: SeedStage,
    previous_epochs_in_stage: int,
    seed_age_epochs: int,
) -> SeedInfo:
    return SeedInfo(
        stage=stage.value,
        improvement_since_stage_start=0.0,
        total_improvement=0.0,
        epochs_in_stage=epochs_in_stage,
        seed_params=0,
        previous_stage=previous_stage.value,
        previous_epochs_in_stage=previous_epochs_in_stage,
        seed_age_epochs=seed_age_epochs,
        interaction_sum=0.0,
        boost_received=0.0,
    )


def test_germinate_wait_prune_cycle_discounted_return_not_positive() -> None:
    """PBRS should not allow Germinate→Wait→Wait→Prune farming.

    This specifically guards against regressions where:
    - Germination receives a PBRS bonus, but pruning does not apply the matching PBRS penalty.
    - Epoch progress bonuses create a positive-return loop across a cull.
    """
    config = ContributionRewardConfig(
        pbrs_weight=1.0,
        germinate_cost=0.0,
        prune_cost=0.0,
        # D2: Disable first-germinate bonus - this test isolates PBRS, not capacity economics
        first_germinate_bonus=0.0,
    )

    # Freeze non-shaping signals so this test only measures PBRS/action shaping.
    val_acc = 0.0
    acc_at_germination = 0.0
    acc_delta = 0.0
    total_params = 0
    host_params = 1
    max_epochs = 100

    rewards: list[float] = []

    # Step 0: GERMINATE from empty slot (seed_info=None).
    r0, c0 = compute_contribution_reward(
        action=LifecycleOp.GERMINATE,
        seed_contribution=None,
        val_acc=val_acc,
        seed_info=None,
        epoch=1,
        max_epochs=max_epochs,
        total_params=total_params,
        host_params=host_params,
        acc_at_germination=acc_at_germination,
        acc_delta=acc_delta,
        config=config,
        return_components=True,
    )
    assert math.isfinite(r0)
    assert c0.pbrs_bonus == 0.0  # No seed_info => no stage PBRS bonus.
    assert c0.action_shaping > 0.0  # Germination PBRS lives in action_shaping.
    rewards.append(r0)

    # Step 1-2: WAIT twice while still GERMINATED (epoch progress bonuses accrue).
    for step_idx, epochs_in_stage in enumerate((1, 2), start=1):
        seed_info = _make_seed_info(
            stage=SeedStage.GERMINATED,
            epochs_in_stage=epochs_in_stage,
            previous_stage=SeedStage.DORMANT,
            previous_epochs_in_stage=0,
            seed_age_epochs=epochs_in_stage,
        )
        r, c = compute_contribution_reward(
            action=LifecycleOp.WAIT,
            seed_contribution=None,
            val_acc=val_acc,
            seed_info=seed_info,
            epoch=1 + step_idx,
            max_epochs=max_epochs,
            total_params=total_params,
            host_params=host_params,
            acc_at_germination=acc_at_germination,
            acc_delta=acc_delta,
            config=config,
            return_components=True,
        )
        assert math.isfinite(r)
        assert c.action_shaping == 0.0
        assert c.pbrs_bonus > 0.0
        rewards.append(r)

    # Step 3: PRUNE while still GERMINATED (pre-action semantics used in simic).
    # The cull PBRS penalty must be applied in action_shaping to close the loop.
    prune_info = _make_seed_info(
        stage=SeedStage.GERMINATED,
        epochs_in_stage=3,
        previous_stage=SeedStage.DORMANT,
        previous_epochs_in_stage=0,
        seed_age_epochs=3,
    )
    r3, c3 = compute_contribution_reward(
        action=LifecycleOp.PRUNE,
        seed_contribution=None,
        val_acc=val_acc,
        seed_info=prune_info,
        epoch=4,
        max_epochs=max_epochs,
        total_params=total_params,
        host_params=host_params,
        acc_at_germination=acc_at_germination,
        acc_delta=acc_delta,
        config=config,
        return_components=True,
    )
    assert math.isfinite(r3)
    assert c3.action_shaping < 0.0
    assert c3.pbrs_bonus > 0.0
    rewards.append(r3)

    # Discounted return under DEFAULT_GAMMA must be non-positive for this loop.
    discounted_return = 0.0
    discount = 1.0
    for r in rewards:
        discounted_return += discount * r
        discount *= config.gamma

    assert discounted_return <= 1e-6, (
        "PBRS farming detected: Germinate→Wait→Wait→Prune produced positive discounted return "
        f"({discounted_return})"
    )


def test_germinate_then_immediate_prune_not_profitable() -> None:
    """Germinate→(1 epoch)→Prune should not be NPV-positive under shaping alone."""
    config = ContributionRewardConfig(
        pbrs_weight=1.0,
        germinate_cost=0.0,
        prune_cost=0.0,
        # D2: Disable first-germinate bonus - this test isolates PBRS, not capacity economics
        first_germinate_bonus=0.0,
    )

    val_acc = 0.0
    acc_at_germination = 0.0
    acc_delta = 0.0
    total_params = 0
    host_params = 1
    max_epochs = 100

    r0, _ = compute_contribution_reward(
        action=LifecycleOp.GERMINATE,
        seed_contribution=None,
        val_acc=val_acc,
        seed_info=None,
        epoch=1,
        max_epochs=max_epochs,
        total_params=total_params,
        host_params=host_params,
        acc_at_germination=acc_at_germination,
        acc_delta=acc_delta,
        config=config,
        return_components=True,
    )

    seed_info = _make_seed_info(
        stage=SeedStage.GERMINATED,
        epochs_in_stage=1,
        previous_stage=SeedStage.DORMANT,
        previous_epochs_in_stage=0,
        seed_age_epochs=1,
    )
    r1, c1 = compute_contribution_reward(
        action=LifecycleOp.PRUNE,
        seed_contribution=None,
        val_acc=val_acc,
        seed_info=seed_info,
        epoch=2,
        max_epochs=max_epochs,
        total_params=total_params,
        host_params=host_params,
        acc_at_germination=acc_at_germination,
        acc_delta=acc_delta,
        config=config,
        return_components=True,
    )
    assert c1.action_shaping < 0.0

    discounted_return = r0 + config.gamma * r1
    assert discounted_return <= 1e-6, (
        "PBRS farming detected: Germinate→Prune produced positive discounted return "
        f"({discounted_return})"
    )
