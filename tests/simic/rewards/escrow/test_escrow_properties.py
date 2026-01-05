"""Property tests for ESCROW reward semantics.

These complement the explicit "spec harness" tests by sweeping a wider input
space and asserting invariants that should *always* hold.
"""

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from esper.simic.rewards import compute_contribution_reward

from tests.simic.rewards.escrow.harness import (
    LifecycleOp,
    SeedStage,
    escrow_config,
    seed_info,
    with_prune_good_seed_penalty,
)

pytestmark = pytest.mark.property


@given(
    credit_prev=st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
    stable_val_acc=st.floats(min_value=0.0, max_value=100.0, allow_nan=False),
    seed_contribution=st.one_of(
        st.none(),
        st.floats(min_value=-10.0, max_value=10.0, allow_nan=False),
    ),
    stage=st.sampled_from([SeedStage.TRAINING, SeedStage.BLENDING, SeedStage.HOLDING, SeedStage.FOSSILIZED]),
    action=st.sampled_from([LifecycleOp.WAIT, LifecycleOp.PRUNE]),
)
def test_escrow_credit_next_is_prev_plus_delta(
    credit_prev: float,
    stable_val_acc: float,
    seed_contribution: float | None,
    stage: SeedStage,
    action: LifecycleOp,
) -> None:
    config = escrow_config()

    _, components = compute_contribution_reward(
        action=action,
        seed_contribution=seed_contribution,
        val_acc=stable_val_acc,
        seed_info=seed_info(stage=stage),
        epoch=1,
        max_epochs=10,
        total_params=100_000,
        host_params=100_000,
        config=config,
        return_components=True,
        stable_val_acc=stable_val_acc,
        escrow_credit_prev=credit_prev,
        acc_at_germination=None,
    )

    assert components.escrow_credit_next == pytest.approx(
        credit_prev + components.escrow_delta
    )


@given(credit_prev=st.floats(min_value=0.0, max_value=10.0, allow_nan=False))
def test_prune_always_refunds_credit_to_zero(credit_prev: float) -> None:
    config = escrow_config()

    reward, components = compute_contribution_reward(
        action=LifecycleOp.PRUNE,
        seed_contribution=None,
        val_acc=50.0,
        seed_info=seed_info(stage=SeedStage.TRAINING),
        epoch=1,
        max_epochs=10,
        total_params=100_000,
        host_params=100_000,
        config=config,
        return_components=True,
        stable_val_acc=50.0,
        escrow_credit_prev=credit_prev,
        acc_at_germination=None,
    )

    assert components.escrow_credit_target == pytest.approx(0.0)
    assert components.escrow_delta == pytest.approx(-credit_prev)
    assert components.escrow_credit_next == pytest.approx(0.0)
    assert components.bounded_attribution == pytest.approx(-credit_prev)
    assert reward == pytest.approx(-credit_prev)


@given(
    credit_prev=st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
    seed_contribution=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False),
)
def test_fossilized_freezes_credit_for_non_prune_actions(
    credit_prev: float,
    seed_contribution: float,
) -> None:
    config = escrow_config()

    reward, components = compute_contribution_reward(
        action=LifecycleOp.WAIT,
        seed_contribution=seed_contribution,
        val_acc=90.0,
        seed_info=seed_info(stage=SeedStage.FOSSILIZED),
        epoch=2,
        max_epochs=10,
        total_params=100_000,
        host_params=100_000,
        config=config,
        return_components=True,
        stable_val_acc=90.0,
        escrow_credit_prev=credit_prev,
        acc_at_germination=0.0,
    )

    assert components.escrow_credit_target == pytest.approx(credit_prev)
    assert components.escrow_delta == pytest.approx(0.0)
    assert components.escrow_credit_next == pytest.approx(credit_prev)
    assert components.bounded_attribution == pytest.approx(0.0)
    assert reward == pytest.approx(0.0)


@given(
    credit_prev=st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
    stage=st.sampled_from([SeedStage.TRAINING, SeedStage.BLENDING, SeedStage.HOLDING]),
)
def test_no_counterfactual_no_proxy_credit_no_ledger_change(
    credit_prev: float,
    stage: SeedStage,
) -> None:
    config = escrow_config()

    reward, components = compute_contribution_reward(
        action=LifecycleOp.WAIT,
        seed_contribution=None,
        val_acc=60.0,
        seed_info=seed_info(stage=stage),
        epoch=3,
        max_epochs=10,
        total_params=100_000,
        host_params=100_000,
        config=config,
        return_components=True,
        stable_val_acc=60.0,
        escrow_credit_prev=credit_prev,
        acc_at_germination=50.0,
    )

    assert components.escrow_delta == pytest.approx(0.0)
    assert components.escrow_credit_next == pytest.approx(credit_prev)
    assert components.bounded_attribution == pytest.approx(0.0)
    assert reward == pytest.approx(0.0)


@given(
    credit_prev=st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
    seed_contribution=st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
    clip=st.floats(min_value=0.01, max_value=2.0, allow_nan=False),
)
def test_delta_clipping_is_applied_to_positive_branch_delta(
    credit_prev: float,
    seed_contribution: float,
    clip: float,
) -> None:
    assume(clip > 0.0)
    config = escrow_config(escrow_delta_clip=clip)

    # progress None path is simplest: target = 0.5 * seed_contribution
    expected_target = 0.5 * seed_contribution
    unclipped = expected_target - credit_prev
    expected_delta = max(-clip, min(clip, unclipped))

    _, components = compute_contribution_reward(
        action=LifecycleOp.WAIT,
        seed_contribution=seed_contribution,
        val_acc=50.0,
        seed_info=seed_info(stage=SeedStage.TRAINING),
        epoch=4,
        max_epochs=10,
        total_params=100_000,
        host_params=100_000,
        config=config,
        return_components=True,
        stable_val_acc=50.0,
        escrow_credit_prev=credit_prev,
        acc_at_germination=None,
    )

    assert components.escrow_credit_target == pytest.approx(expected_target)
    assert components.escrow_delta == pytest.approx(expected_delta)
    assert abs(components.escrow_delta) <= clip + 1e-9


@given(
    credit_prev=st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
    seed_contribution=st.floats(min_value=1.01, max_value=20.0, allow_nan=False),
    penalty=st.floats(min_value=-5.0, max_value=-0.1, allow_nan=False),
)
def test_ratio_penalty_floor_never_mints_negative_credit(
    credit_prev: float,
    seed_contribution: float,
    penalty: float,
) -> None:
    # Force ratio_penalty to be negative and large enough to matter.
    config = with_prune_good_seed_penalty(
        escrow_config(disable_anti_gaming=False, escrow_delta_clip=0.0),
        prune_good_seed_penalty=penalty,
    )

    # progress == 0 => attributed remains 0 in escrow mode, so target = max(0, ratio_penalty).
    reward, components = compute_contribution_reward(
        action=LifecycleOp.WAIT,
        seed_contribution=seed_contribution,
        val_acc=60.0,
        seed_info=seed_info(stage=SeedStage.TRAINING, total_improvement=0.0),
        epoch=5,
        max_epochs=10,
        total_params=100_000,
        host_params=100_000,
        config=config,
        return_components=True,
        stable_val_acc=60.0,
        escrow_credit_prev=credit_prev,
        acc_at_germination=60.0,
    )

    assert components.ratio_penalty <= 0.0
    assert components.escrow_credit_target == pytest.approx(0.0)
    assert components.escrow_delta == pytest.approx(-credit_prev)
    assert components.escrow_credit_next == pytest.approx(0.0)
    assert reward == pytest.approx(-credit_prev)

