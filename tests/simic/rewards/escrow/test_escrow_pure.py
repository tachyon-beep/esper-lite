"""Pure unit tests for the ESCROW reward branch.

These tests call `compute_contribution_reward` directly and assert the escrow
ledger semantics via RewardComponentsTelemetry:

- Escrow pays the *delta* in a per-slot credit target (not the balance).
- Negative contribution wipes escrow credit and pays immediate harm.
- Fossilized seeds freeze escrow (no further minting, no delta).
- WAIT can mint escrow delta if contribution/progress justify it.
"""

import pytest

from esper.simic.rewards import compute_contribution_reward

from tests.simic.rewards.escrow.harness import (
    LifecycleOp,
    SeedStage,
    escrow_config,
    seed_info,
    with_prune_good_seed_penalty,
)


def test_escrow_requires_stable_val_acc():
    config = escrow_config()
    with pytest.raises(ValueError, match="requires stable_val_acc"):
        compute_contribution_reward(
            action=LifecycleOp.WAIT,
            seed_contribution=None,
            val_acc=70.0,
            seed_info=seed_info(),
            epoch=1,
            max_epochs=10,
            total_params=100_000,
            host_params=100_000,
            config=config,
            return_components=True,
            stable_val_acc=None,
            escrow_credit_prev=0.0,
        )


def test_escrow_requires_return_components_for_ledger_update():
    config = escrow_config()
    with pytest.raises(ValueError, match="requires return_components=True"):
        compute_contribution_reward(
            action=LifecycleOp.WAIT,
            seed_contribution=None,
            val_acc=70.0,
            seed_info=seed_info(),
            epoch=1,
            max_epochs=10,
            total_params=100_000,
            host_params=100_000,
            config=config,
            return_components=False,
            stable_val_acc=70.0,
            escrow_credit_prev=0.0,
        )


def test_prune_refunds_existing_credit_delta_not_balance():
    config = escrow_config()

    reward, components = compute_contribution_reward(
        action=LifecycleOp.PRUNE,
        seed_contribution=None,
        val_acc=70.0,
        seed_info=seed_info(stage=SeedStage.TRAINING),
        epoch=3,
        max_epochs=10,
        total_params=100_000,
        host_params=100_000,
        config=config,
        return_components=True,
        stable_val_acc=70.0,
        escrow_credit_prev=1.25,
    )

    assert components.escrow_credit_prev == pytest.approx(1.25)
    assert components.escrow_credit_target == pytest.approx(0.0)
    assert components.escrow_delta == pytest.approx(-1.25)
    assert components.escrow_credit_next == pytest.approx(0.0)
    assert components.bounded_attribution == pytest.approx(-1.25)
    assert reward == pytest.approx(-1.25)


def test_fossilized_seed_freezes_credit_no_delta_even_if_contributing():
    config = escrow_config()

    reward, components = compute_contribution_reward(
        action=LifecycleOp.WAIT,
        seed_contribution=5.0,  # would mint in non-fossilized state
        val_acc=90.0,
        seed_info=seed_info(stage=SeedStage.FOSSILIZED),
        epoch=4,
        max_epochs=10,
        total_params=100_000,
        host_params=100_000,
        config=config,
        return_components=True,
        stable_val_acc=90.0,
        escrow_credit_prev=2.0,
        acc_at_germination=50.0,
    )

    assert components.escrow_credit_prev == pytest.approx(2.0)
    assert components.escrow_credit_target == pytest.approx(2.0)
    assert components.escrow_delta == pytest.approx(0.0)
    assert components.escrow_credit_next == pytest.approx(2.0)
    assert components.bounded_attribution == pytest.approx(0.0)
    assert reward == pytest.approx(0.0)


def test_negative_contribution_wipes_credit_and_pays_harm():
    config = escrow_config()

    reward, components = compute_contribution_reward(
        action=LifecycleOp.WAIT,
        seed_contribution=-0.25,
        val_acc=60.0,
        seed_info=seed_info(stage=SeedStage.TRAINING),
        epoch=5,
        max_epochs=10,
        total_params=100_000,
        host_params=100_000,
        config=config,
        return_components=True,
        stable_val_acc=60.0,
        escrow_credit_prev=1.0,
        acc_at_germination=55.0,
    )

    # Harm penalty + escrow clawback.
    assert components.escrow_credit_target == pytest.approx(0.0)
    assert components.escrow_delta == pytest.approx(-1.0)
    assert components.escrow_credit_next == pytest.approx(0.0)
    assert components.bounded_attribution == pytest.approx(-1.25)
    assert reward == pytest.approx(-1.25)


def test_wait_can_mint_credit_when_progress_and_contribution_are_positive():
    config = escrow_config()

    # progress = stable_val_acc - acc_at_germination = 1.0
    # seed_contribution = 4.0
    # attributed = sqrt(progress * contribution) = 2.0
    reward, components = compute_contribution_reward(
        action=LifecycleOp.WAIT,
        seed_contribution=4.0,
        val_acc=71.0,
        seed_info=seed_info(stage=SeedStage.TRAINING),
        epoch=6,
        max_epochs=10,
        total_params=100_000,
        host_params=100_000,
        config=config,
        return_components=True,
        stable_val_acc=70.0,
        escrow_credit_prev=0.0,
        acc_at_germination=69.0,
    )

    assert components.escrow_credit_target == pytest.approx(2.0)
    assert components.escrow_delta == pytest.approx(2.0)
    assert components.escrow_credit_next == pytest.approx(2.0)
    assert components.bounded_attribution == pytest.approx(2.0)
    assert reward == pytest.approx(2.0)


def test_wait_with_positive_contribution_but_no_progress_claws_back_to_zero():
    config = escrow_config()

    reward, components = compute_contribution_reward(
        action=LifecycleOp.WAIT,
        seed_contribution=2.0,
        val_acc=50.0,
        seed_info=seed_info(stage=SeedStage.TRAINING),
        epoch=7,
        max_epochs=10,
        total_params=100_000,
        host_params=100_000,
        config=config,
        return_components=True,
        stable_val_acc=50.0,
        escrow_credit_prev=1.0,
        acc_at_germination=50.0,  # progress == 0.0
    )

    assert components.escrow_credit_target == pytest.approx(0.0)
    assert components.escrow_delta == pytest.approx(-1.0)
    assert components.escrow_credit_next == pytest.approx(0.0)
    assert reward == pytest.approx(-1.0)


def test_no_progress_baseline_uses_half_contribution_when_acc_at_germination_missing():
    config = escrow_config()

    reward, components = compute_contribution_reward(
        action=LifecycleOp.WAIT,
        seed_contribution=1.0,
        val_acc=60.0,
        seed_info=seed_info(stage=SeedStage.TRAINING),
        epoch=8,
        max_epochs=10,
        total_params=100_000,
        host_params=100_000,
        config=config,
        return_components=True,
        stable_val_acc=60.0,
        escrow_credit_prev=0.0,
        acc_at_germination=None,
    )

    assert components.escrow_credit_target == pytest.approx(0.5)
    assert components.escrow_delta == pytest.approx(0.5)
    assert components.escrow_credit_next == pytest.approx(0.5)
    assert reward == pytest.approx(0.5)


def test_ratio_penalty_can_floor_credit_target_at_zero_never_negative():
    # Enable anti-gaming and make prune_good_seed_penalty strong enough to dominate.
    config = with_prune_good_seed_penalty(
        escrow_config(disable_anti_gaming=False),
        prune_good_seed_penalty=-1.0,
    )

    reward, components = compute_contribution_reward(
        action=LifecycleOp.WAIT,
        seed_contribution=10.0,  # triggers ratio penalty path (contribution > 1.0)
        val_acc=60.0,
        seed_info=seed_info(stage=SeedStage.TRAINING, total_improvement=0.0),
        epoch=9,
        max_epochs=10,
        total_params=100_000,
        host_params=100_000,
        config=config,
        return_components=True,
        stable_val_acc=60.01,
        escrow_credit_prev=0.5,
        acc_at_germination=60.0,  # progress = 0.01, attributed ~0.316
    )

    assert components.ratio_penalty < 0.0
    assert components.escrow_credit_target == pytest.approx(0.0)
    assert components.escrow_delta == pytest.approx(-0.5)
    assert components.escrow_credit_next == pytest.approx(0.0)
    assert reward == pytest.approx(-0.5)

