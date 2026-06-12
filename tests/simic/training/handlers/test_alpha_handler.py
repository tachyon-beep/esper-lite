"""Unit tests for the SET_ALPHA_TARGET operation handler."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from esper.leyline import (
    ALPHA_TARGET_VALUES,
    AlphaCurveAction,
    AlphaMode,
    AlphaSpeedAction,
    SeedStage,
    STYLE_ALPHA_ALGORITHMS,
)
from esper.simic.training.handlers import (
    AlphaTargetParams,
    HandlerContext,
    execute_set_alpha_target,
)
from esper.simic.training.parallel_env_state import ParallelEnvState


@pytest.fixture
def alpha_target_context() -> HandlerContext:
    """Create a HandlerContext with a seed eligible for alpha retargeting."""
    env_state = MagicMock(spec=ParallelEnvState)

    seed_state = MagicMock()
    seed_state.stage = SeedStage.HOLDING
    seed_state.alpha_controller.alpha_mode = AlphaMode.HOLD

    slot = MagicMock()
    slot.alpha = 1.0
    slot.set_alpha_target.return_value = True

    return HandlerContext(
        env_idx=0,
        slot_id="r0c0",
        env_state=env_state,
        model=MagicMock(),
        slot=slot,
        seed_state=seed_state,
        epoch=5,
        max_epochs=150,
        episodes_completed=0,
    )


def test_execute_set_alpha_target_passes_sigmoid_steepness(
    alpha_target_context: HandlerContext,
) -> None:
    """Policy-selected sigmoid variants must reach the slot retarget call."""
    params = AlphaTargetParams(
        alpha_target_idx=0,
        alpha_speed_idx=AlphaSpeedAction.FAST.value,
        alpha_curve_idx=AlphaCurveAction.SIGMOID_SHARP.value,
        style_idx=0,
    )

    result = execute_set_alpha_target(alpha_target_context, params)

    assert result.success is True
    alpha_target_context.slot.set_alpha_target.assert_called_once_with(
        alpha_target=ALPHA_TARGET_VALUES[params.alpha_target_idx],
        steps=3,
        curve=AlphaCurveAction.SIGMOID_SHARP.to_curve(),
        steepness=AlphaCurveAction.SIGMOID_SHARP.to_steepness(),
        alpha_algorithm=STYLE_ALPHA_ALGORITHMS[params.style_idx],
        initiator="policy",
    )
    assert result.telemetry["steepness"] == pytest.approx(24.0)
