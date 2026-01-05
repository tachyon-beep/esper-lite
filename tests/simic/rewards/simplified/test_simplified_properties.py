"""Property tests for SIMPLIFIED reward semantics."""

from __future__ import annotations

import pytest
from hypothesis import given, strategies as st

from esper.simic.rewards import compute_simplified_reward

from tests.simic.rewards.simplified.harness import (
    LifecycleOp,
    simplified_config,
)


@pytest.mark.property
@given(
    action=st.sampled_from([LifecycleOp.WAIT, LifecycleOp.GERMINATE, LifecycleOp.PRUNE]),
)
def test_simplified_reward_is_zero_or_uniform_cost_when_pbrs_and_terminal_disabled(
    action: LifecycleOp,
) -> None:
    config = simplified_config(disable_pbrs=True, disable_terminal_reward=True)

    reward = compute_simplified_reward(
        action=action,
        seed_info=None,
        epoch=1,
        max_epochs=5,
        val_acc=0.0,
        num_contributing_fossilized=0,
        config=config,
    )

    expected = 0.0 if action == LifecycleOp.WAIT else -0.01
    assert reward == pytest.approx(expected)


@pytest.mark.property
@given(
    val_acc=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    fossils=st.integers(min_value=0, max_value=3),
    action=st.sampled_from([LifecycleOp.WAIT, LifecycleOp.PRUNE]),
)
def test_simplified_terminal_bonus_is_linear_and_adds_uniform_cost(
    val_acc: float, fossils: int, action: LifecycleOp
) -> None:
    config = simplified_config(disable_pbrs=True, disable_terminal_reward=False)

    reward = compute_simplified_reward(
        action=action,
        seed_info=None,
        epoch=5,
        max_epochs=5,
        val_acc=val_acc,
        num_contributing_fossilized=fossils,
        config=config,
    )

    expected = (val_acc / 100.0) * 3.0 + fossils * 2.0
    if action != LifecycleOp.WAIT:
        expected -= 0.01
    assert reward == pytest.approx(expected)

