"""Unit tests for the FOSSILIZE operation handler."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from esper.leyline import DEFAULT_MIN_FOSSILIZE_CONTRIBUTION, SeedStage
from esper.simic.rewards import SeedInfo
from esper.simic.training.handlers import HandlerContext, execute_fossilize
from esper.simic.training.parallel_env_state import ParallelEnvState


@pytest.fixture
def holding_context() -> HandlerContext:
    """Create a HandlerContext with a holding seed eligible for fossilization."""
    env_state = MagicMock(spec=ParallelEnvState)
    env_state.seeds_fossilized = 0
    env_state.fossilize_count = 0
    env_state.contributing_fossilized = 0
    env_state.acc_at_germination = {"r0c0": 0.5}
    env_state.scaffold_boost_ledger = {}
    env_state.pending_hindsight_credit = 0.0
    env_state.seed_optimizers = {"r0c0": object()}
    env_state.needs_governor_snapshot = False

    seed_state = MagicMock()
    seed_state.stage = SeedStage.HOLDING

    return HandlerContext(
        env_idx=0,
        slot_id="r0c0",
        env_state=env_state,
        model=MagicMock(),
        slot=MagicMock(),
        seed_state=seed_state,
        epoch=12,
        max_epochs=150,
        episodes_completed=0,
    )


def test_execute_fossilize_tracks_measured_total_improvement(
    holding_context: HandlerContext,
) -> None:
    seed_info = SeedInfo(
        stage=SeedStage.HOLDING.value,
        improvement_since_stage_start=0.2,
        total_improvement=DEFAULT_MIN_FOSSILIZE_CONTRIBUTION,
        epochs_in_stage=3,
    )

    result = execute_fossilize(
        holding_context,
        seed_info,
        lambda _model, _slot_id: True,
    )

    assert result.success is True
    assert result.telemetry["total_improvement"] == DEFAULT_MIN_FOSSILIZE_CONTRIBUTION
    assert result.telemetry["is_contributing"] is True
    assert holding_context.env_state.seeds_fossilized == 1
    assert holding_context.env_state.fossilize_count == 1
    assert holding_context.env_state.contributing_fossilized == 1
    assert "r0c0" not in holding_context.env_state.seed_optimizers
    assert holding_context.env_state.needs_governor_snapshot is True


def test_execute_fossilize_requires_total_improvement_measurement(
    holding_context: HandlerContext,
) -> None:
    seed_info = SeedInfo(
        stage=SeedStage.HOLDING.value,
        improvement_since_stage_start=None,
        total_improvement=None,
        epochs_in_stage=3,
    )

    with pytest.raises(
        ValueError,
        match="fossilize handler requires seed_info.total_improvement",
    ):
        execute_fossilize(
            holding_context,
            seed_info,
            lambda _model, _slot_id: True,
        )

    assert holding_context.env_state.seeds_fossilized == 0
    assert holding_context.env_state.fossilize_count == 0
    assert holding_context.env_state.contributing_fossilized == 0
    assert "r0c0" in holding_context.env_state.seed_optimizers
