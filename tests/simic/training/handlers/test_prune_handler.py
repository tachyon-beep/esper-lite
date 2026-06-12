"""Unit tests for the PRUNE operation handler."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from esper.leyline import (
    MIN_PRUNE_AGE,
    AlphaCurveAction,
    AlphaMode,
    AlphaSpeedAction,
    SeedStage,
)
from esper.simic.rewards import SeedInfo
from esper.simic.training.handlers import HandlerContext, PruneParams, execute_prune
from esper.simic.training.parallel_env_state import ParallelEnvState


@pytest.fixture
def prunable_context() -> HandlerContext:
    """Create a HandlerContext with a training seed eligible for pruning."""
    env_state = MagicMock(spec=ParallelEnvState)
    env_state.prune_count = 0

    seed_state = MagicMock()
    seed_state.stage = SeedStage.TRAINING
    seed_state.alpha_controller.alpha_mode = AlphaMode.HOLD
    seed_state.can_transition_to.return_value = True

    slot = MagicMock()
    slot.schedule_prune.return_value = True

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


def test_execute_prune_passes_sigmoid_steepness(
    prunable_context: HandlerContext,
) -> None:
    """Policy-selected sigmoid variants must reach the slot schedule."""
    seed_info = SeedInfo(
        stage=SeedStage.TRAINING.value,
        improvement_since_stage_start=0.0,
        total_improvement=0.0,
        epochs_in_stage=1,
        seed_age_epochs=MIN_PRUNE_AGE,
    )
    params = PruneParams(
        alpha_speed_idx=AlphaSpeedAction.FAST.value,
        alpha_curve_idx=AlphaCurveAction.SIGMOID_GENTLE.value,
    )

    result = execute_prune(prunable_context, params, seed_info)

    assert result.success is True
    prunable_context.slot.schedule_prune.assert_called_once_with(
        steps=3,
        curve=AlphaCurveAction.SIGMOID_GENTLE.to_curve(),
        steepness=AlphaCurveAction.SIGMOID_GENTLE.to_steepness(),
        initiator="policy",
    )
    assert result.telemetry["steepness"] == pytest.approx(6.0)
    assert prunable_context.env_state.prune_count == 1
