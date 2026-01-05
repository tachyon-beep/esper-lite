"""Property tests for SHAPED reward semantics.

These are small invariants that should hold regardless of exact numeric tuning.
They complement the explicit contract tests in the other files.
"""

from __future__ import annotations

import pytest
from hypothesis import given, strategies as st

from esper.simic.rewards import compute_contribution_reward

from tests.simic.rewards.shaped.harness import (
    LifecycleOp,
    SeedStage,
    seed_info,
    shaped_config,
)


@pytest.mark.property
@given(
    progress=st.floats(min_value=0.0, max_value=20.0, allow_nan=False, allow_infinity=False),
    seed_contribution=st.floats(
        min_value=0.0, max_value=20.0, allow_nan=False, allow_infinity=False
    ),
)
def test_shaped_attribution_is_nonnegative_and_capped(progress: float, seed_contribution: float) -> None:
    """With anti-gaming disabled, SHAPED attribution should be in [0, contribution]."""
    config = shaped_config(contribution_weight=1.0)
    acc_at_germination = 50.0
    val_acc = acc_at_germination + progress

    reward, components = compute_contribution_reward(
        action=LifecycleOp.WAIT,
        seed_contribution=seed_contribution,
        val_acc=val_acc,
        seed_info=seed_info(stage=SeedStage.TRAINING, total_improvement=1.0),
        epoch=1,
        max_epochs=10,
        acc_at_germination=acc_at_germination,
        config=config,
        return_components=True,
    )

    assert 0.0 <= components.bounded_attribution <= seed_contribution + 1e-9
    assert reward == pytest.approx(components.bounded_attribution)


@pytest.mark.property
@given(
    seed_contribution=st.floats(
        min_value=-20.0, max_value=-1e-6, allow_nan=False, allow_infinity=False
    )
)
def test_shaped_negative_counterfactual_scales_linearly(seed_contribution: float) -> None:
    config = shaped_config(contribution_weight=1.7)

    reward, components = compute_contribution_reward(
        action=LifecycleOp.WAIT,
        seed_contribution=seed_contribution,
        val_acc=0.0,
        seed_info=seed_info(stage=SeedStage.TRAINING, total_improvement=1.0),
        epoch=1,
        max_epochs=10,
        acc_at_germination=0.0,
        config=config,
        return_components=True,
    )

    assert reward == pytest.approx(1.7 * seed_contribution)
    assert components.bounded_attribution == pytest.approx(1.7 * seed_contribution)


@pytest.mark.property
@given(
    progress=st.floats(min_value=1e-6, max_value=20.0, allow_nan=False, allow_infinity=False),
    seed_contribution=st.floats(
        min_value=-20.0, max_value=20.0, allow_nan=False, allow_infinity=False
    ).filter(lambda x: x != 0.0),
)
def test_shaped_prune_inverts_attribution_sign(progress: float, seed_contribution: float) -> None:
    """PRUNE flips attribution sign in SHAPED mode (when no other shaping is enabled)."""
    config = shaped_config(contribution_weight=1.0)
    acc_at_germination = 50.0
    val_acc = acc_at_germination + progress

    reward, _ = compute_contribution_reward(
        action=LifecycleOp.PRUNE,
        seed_contribution=seed_contribution,
        val_acc=val_acc,
        seed_info=seed_info(stage=SeedStage.TRAINING, total_improvement=1.0),
        epoch=1,
        max_epochs=10,
        acc_at_germination=acc_at_germination,
        config=config,
        return_components=True,
    )

    if seed_contribution > 0:
        assert reward < 0
    else:
        assert reward > 0

