"""Unit tests for the GERMINATE operation handler.

Tests the can_germinate() precondition check and execute_germinate()
operation for creating new seeds in slots.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call

import pytest

from esper.leyline import (
    BLUEPRINT_IDS,
    SeedStage,
    STYLE_ALPHA_ALGORITHMS,
    STYLE_BLEND_IDS,
    TEMPO_TO_EPOCHS,
    TempoAction,
)
from esper.simic.training.handlers import (
    HandlerContext,
    GerminateParams,
    can_germinate,
    execute_germinate,
)
from esper.simic.training.parallel_env_state import ParallelEnvState


@pytest.fixture
def mock_env_state() -> MagicMock:
    """Create a mock ParallelEnvState."""
    env_state = MagicMock(spec=ParallelEnvState)
    env_state.val_acc = 0.55
    env_state.seeds_created = 0
    env_state.germinate_count = 0
    env_state.acc_at_germination = {}
    env_state.escrow_credit = {}
    env_state.seed_optimizers = {}
    env_state.init_obs_v3_slot_tracking = MagicMock()
    return env_state


@pytest.fixture
def mock_slot() -> MagicMock:
    """Create a mock SeedSlot with no active seed."""
    slot = MagicMock()
    slot.state = None
    return slot


@pytest.fixture
def mock_slot_with_seed() -> MagicMock:
    """Create a mock SeedSlot with an active seed."""
    slot = MagicMock()
    seed_state = MagicMock()
    seed_state.stage = SeedStage.TRAINING
    slot.state = seed_state
    return slot


@pytest.fixture
def mock_model() -> MagicMock:
    """Create a mock SlottedHostProtocol."""
    model = MagicMock()
    model.germinate_seed = MagicMock()
    return model


@pytest.fixture
def empty_slot_context(
    mock_env_state: MagicMock,
    mock_slot: MagicMock,
    mock_model: MagicMock,
) -> HandlerContext:
    """Create a HandlerContext with an empty slot."""
    return HandlerContext(
        env_idx=0,
        slot_id="r0c0",
        env_state=mock_env_state,
        model=mock_model,
        slot=mock_slot,
        seed_state=None,
        epoch=5,
        max_epochs=150,
        episodes_completed=0,
    )


@pytest.fixture
def occupied_slot_context(
    mock_env_state: MagicMock,
    mock_slot_with_seed: MagicMock,
    mock_model: MagicMock,
) -> HandlerContext:
    """Create a HandlerContext with an occupied slot."""
    return HandlerContext(
        env_idx=0,
        slot_id="r0c0",
        env_state=mock_env_state,
        model=mock_model,
        slot=mock_slot_with_seed,
        seed_state=mock_slot_with_seed.state,
        epoch=5,
        max_epochs=150,
        episodes_completed=0,
    )


class TestCanGerminate:
    """Tests for can_germinate() precondition check."""

    def test_can_germinate_empty_slot(self, empty_slot_context: HandlerContext) -> None:
        """can_germinate returns True for empty slot."""
        assert can_germinate(empty_slot_context) is True

    def test_cannot_germinate_occupied_slot(
        self, occupied_slot_context: HandlerContext
    ) -> None:
        """can_germinate returns False for occupied slot."""
        assert can_germinate(occupied_slot_context) is False


class TestExecuteGerminate:
    """Tests for execute_germinate() operation."""

    def test_germinate_success(self, empty_slot_context: HandlerContext) -> None:
        """execute_germinate succeeds on empty slot."""
        params = GerminateParams(
            blueprint_idx=1,  # First non-null blueprint
            style_idx=0,
            tempo_idx=1,
            alpha_target=1.0,
        )

        result = execute_germinate(empty_slot_context, params)

        assert result.success is True
        assert result.error is None

    def test_germinate_calls_model_germinate_seed(
        self, empty_slot_context: HandlerContext
    ) -> None:
        """execute_germinate calls model.germinate_seed with correct args."""
        params = GerminateParams(
            blueprint_idx=1,
            style_idx=0,
            tempo_idx=1,
            alpha_target=1.0,
        )

        execute_germinate(empty_slot_context, params)

        # Verify germinate_seed was called
        empty_slot_context.model.germinate_seed.assert_called_once()
        call_kwargs = empty_slot_context.model.germinate_seed.call_args

        # Check key arguments
        assert call_kwargs[0][0] == BLUEPRINT_IDS[1]  # blueprint_id
        assert call_kwargs[1]["slot"] == "r0c0"
        assert call_kwargs[1]["blend_algorithm_id"] == STYLE_BLEND_IDS[0]
        assert call_kwargs[1]["alpha_target"] == 1.0

    def test_germinate_updates_env_state(
        self, empty_slot_context: HandlerContext
    ) -> None:
        """execute_germinate updates env_state counters."""
        params = GerminateParams(
            blueprint_idx=1,
            style_idx=0,
            tempo_idx=1,
            alpha_target=1.0,
        )
        ctx = empty_slot_context

        execute_germinate(ctx, params)

        assert ctx.env_state.seeds_created == 1
        assert ctx.env_state.germinate_count == 1

    def test_germinate_records_accuracy_at_germination(
        self, empty_slot_context: HandlerContext
    ) -> None:
        """execute_germinate records val_acc at germination."""
        params = GerminateParams(
            blueprint_idx=1,
            style_idx=0,
            tempo_idx=1,
            alpha_target=1.0,
        )
        ctx = empty_slot_context
        ctx.env_state.val_acc = 0.65

        execute_germinate(ctx, params)

        assert ctx.env_state.acc_at_germination["r0c0"] == 0.65

    def test_germinate_initializes_escrow_credit(
        self, empty_slot_context: HandlerContext
    ) -> None:
        """execute_germinate initializes escrow credit to zero."""
        params = GerminateParams(
            blueprint_idx=1,
            style_idx=0,
            tempo_idx=1,
            alpha_target=1.0,
        )
        ctx = empty_slot_context

        execute_germinate(ctx, params)

        assert ctx.env_state.escrow_credit["r0c0"] == 0.0

    def test_germinate_calls_init_obs_v3_slot_tracking(
        self, empty_slot_context: HandlerContext
    ) -> None:
        """execute_germinate initializes Obs V3 tracking for the slot."""
        params = GerminateParams(
            blueprint_idx=1,
            style_idx=0,
            tempo_idx=1,
            alpha_target=1.0,
        )
        ctx = empty_slot_context

        execute_germinate(ctx, params)

        ctx.env_state.init_obs_v3_slot_tracking.assert_called_once_with("r0c0")

    def test_germinate_clears_stale_optimizer(
        self, empty_slot_context: HandlerContext
    ) -> None:
        """execute_germinate removes any stale seed optimizer."""
        params = GerminateParams(
            blueprint_idx=1,
            style_idx=0,
            tempo_idx=1,
            alpha_target=1.0,
        )
        ctx = empty_slot_context
        ctx.env_state.seed_optimizers["r0c0"] = MagicMock()

        execute_germinate(ctx, params)

        assert "r0c0" not in ctx.env_state.seed_optimizers

    def test_germinate_fails_on_occupied_slot(
        self, occupied_slot_context: HandlerContext
    ) -> None:
        """execute_germinate fails when slot is occupied."""
        params = GerminateParams(
            blueprint_idx=1,
            style_idx=0,
            tempo_idx=1,
            alpha_target=1.0,
        )

        result = execute_germinate(occupied_slot_context, params)

        assert result.success is False
        assert "already has active seed" in result.error

    def test_germinate_succeeds_with_noop_blueprint(
        self, empty_slot_context: HandlerContext
    ) -> None:
        """execute_germinate succeeds with NOOP blueprint (index 0).

        Note: NOOP is a valid blueprint that creates a passthrough module.
        Action masking is responsible for preventing NOOP selection when
        appropriate, not the germinate handler itself.
        """
        params = GerminateParams(
            blueprint_idx=0,  # NOOP blueprint
            style_idx=0,
            tempo_idx=1,
            alpha_target=1.0,
        )

        result = execute_germinate(empty_slot_context, params)

        assert result.success is True
        assert result.telemetry["blueprint_id"] == "noop"

    def test_germinate_telemetry_includes_blueprint_id(
        self, empty_slot_context: HandlerContext
    ) -> None:
        """execute_germinate returns telemetry with blueprint_id."""
        params = GerminateParams(
            blueprint_idx=1,
            style_idx=0,
            tempo_idx=1,
            alpha_target=1.0,
        )

        result = execute_germinate(empty_slot_context, params)

        assert "blueprint_id" in result.telemetry
        assert result.telemetry["blueprint_id"] == BLUEPRINT_IDS[1]

    def test_germinate_generates_unique_seed_ids(
        self, empty_slot_context: HandlerContext
    ) -> None:
        """execute_germinate generates unique seed IDs per germination."""
        params = GerminateParams(
            blueprint_idx=1,
            style_idx=0,
            tempo_idx=1,
            alpha_target=1.0,
        )
        ctx = empty_slot_context

        # First germination
        result1 = execute_germinate(ctx, params)
        seed_id_1 = result1.telemetry["seed_id"]

        # Simulate slot becoming empty again for second germination
        ctx.slot.state = None

        # Second germination
        result2 = execute_germinate(ctx, params)
        seed_id_2 = result2.telemetry["seed_id"]

        assert seed_id_1 != seed_id_2

    def test_germinate_uses_style_parameters(
        self, empty_slot_context: HandlerContext
    ) -> None:
        """execute_germinate uses style_idx for blend and alpha algorithms."""
        params = GerminateParams(
            blueprint_idx=1,
            style_idx=1,  # Non-default style
            tempo_idx=1,
            alpha_target=0.5,
        )

        result = execute_germinate(empty_slot_context, params)

        assert result.telemetry["blend_algorithm_id"] == STYLE_BLEND_IDS[1]
        expected_alpha_algo = STYLE_ALPHA_ALGORITHMS[1]
        expected_name = expected_alpha_algo.name if expected_alpha_algo else None
        assert result.telemetry["alpha_algorithm"] == expected_name

    def test_germinate_uses_tempo_parameter(
        self, empty_slot_context: HandlerContext
    ) -> None:
        """execute_germinate uses tempo_idx for blend tempo epochs."""
        params = GerminateParams(
            blueprint_idx=1,
            style_idx=0,
            tempo_idx=2,  # Different tempo
            alpha_target=1.0,
        )

        result = execute_germinate(empty_slot_context, params)

        expected_tempo = TEMPO_TO_EPOCHS[TempoAction(2)]
        assert result.telemetry["blend_tempo_epochs"] == expected_tempo


class TestGerminateRegistry:
    """Tests for handler registry integration."""

    def test_germinate_in_registry(self) -> None:
        """execute_germinate is registered in HANDLER_REGISTRY."""
        from esper.simic.training.handlers import HANDLER_REGISTRY
        from esper.leyline import OP_GERMINATE

        assert OP_GERMINATE in HANDLER_REGISTRY
        assert HANDLER_REGISTRY[OP_GERMINATE] is execute_germinate

    def test_can_germinate_in_registry(self) -> None:
        """can_germinate is registered in CAN_EXECUTE_REGISTRY."""
        from esper.simic.training.handlers import CAN_EXECUTE_REGISTRY
        from esper.leyline import OP_GERMINATE

        assert OP_GERMINATE in CAN_EXECUTE_REGISTRY
        assert CAN_EXECUTE_REGISTRY[OP_GERMINATE] is can_germinate
