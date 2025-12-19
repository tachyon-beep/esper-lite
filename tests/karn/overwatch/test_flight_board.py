"""Tests for FlightBoard widget."""

from __future__ import annotations

import pytest

from esper.karn.overwatch.schema import (
    EnvSummary,
    SlotChipState,
    TuiSnapshot,
    ConnectionStatus,
    TamiyoState,
)


@pytest.fixture
def sample_snapshot() -> TuiSnapshot:
    """Create a sample snapshot with multiple envs."""
    return TuiSnapshot(
        schema_version=1,
        captured_at="2025-12-18T12:00:00Z",
        connection=ConnectionStatus(True, 1000.0, 0.5),
        tamiyo=TamiyoState(),
        flight_board=[
            EnvSummary(
                env_id=0,
                device_id=0,
                status="OK",
                throughput_fps=98.5,
                anomaly_score=0.1,
                slots={"r0c1": SlotChipState("r0c1", "TRAINING", "conv", 0.5)},
            ),
            EnvSummary(
                env_id=1,
                device_id=0,
                status="WARN",
                throughput_fps=45.0,
                anomaly_score=0.65,
                anomaly_reasons=["Low throughput"],
            ),
            EnvSummary(
                env_id=2,
                device_id=1,
                status="CRIT",
                throughput_fps=10.0,
                anomaly_score=0.85,
                anomaly_reasons=["Throughput critical", "Memory pressure"],
            ),
            EnvSummary(
                env_id=3,
                device_id=1,
                status="OK",
                throughput_fps=100.0,
                anomaly_score=0.05,
            ),
        ],
    )


class TestFlightBoard:
    """Tests for FlightBoard widget."""

    def test_flight_board_imports(self) -> None:
        """FlightBoard can be imported."""
        from esper.karn.overwatch.widgets.flight_board import FlightBoard

        assert FlightBoard is not None

    def test_flight_board_sorts_by_anomaly(self, sample_snapshot: TuiSnapshot) -> None:
        """FlightBoard sorts envs by anomaly score (highest first)."""
        from esper.karn.overwatch.widgets.flight_board import FlightBoard

        board = FlightBoard()
        board.update_snapshot(sample_snapshot)

        order = board.get_display_order()
        # Highest anomaly first: env 2 (0.85), env 1 (0.65), env 0 (0.1), env 3 (0.05)
        assert order[0] == 2
        assert order[1] == 1
        assert order[-1] == 3

    def test_flight_board_initial_selection(self, sample_snapshot: TuiSnapshot) -> None:
        """FlightBoard selects first env initially."""
        from esper.karn.overwatch.widgets.flight_board import FlightBoard

        board = FlightBoard()
        board.update_snapshot(sample_snapshot)

        # First in display order (highest anomaly)
        assert board.selected_env_id == 2

    def test_flight_board_navigate_down(self, sample_snapshot: TuiSnapshot) -> None:
        """FlightBoard navigates down with j/down."""
        from esper.karn.overwatch.widgets.flight_board import FlightBoard

        board = FlightBoard()
        board.update_snapshot(sample_snapshot)

        # Start at env 2 (first in order)
        assert board.selected_env_id == 2

        board.navigate_down()
        assert board.selected_env_id == 1  # Second in order

        board.navigate_down()
        assert board.selected_env_id == 0  # Third

    def test_flight_board_navigate_up(self, sample_snapshot: TuiSnapshot) -> None:
        """FlightBoard navigates up with k/up."""
        from esper.karn.overwatch.widgets.flight_board import FlightBoard

        board = FlightBoard()
        board.update_snapshot(sample_snapshot)

        # Go down first
        board.navigate_down()
        board.navigate_down()
        assert board.selected_env_id == 0

        # Navigate back up
        board.navigate_up()
        assert board.selected_env_id == 1

    def test_flight_board_navigate_wraps(self, sample_snapshot: TuiSnapshot) -> None:
        """FlightBoard navigation doesn't wrap by default."""
        from esper.karn.overwatch.widgets.flight_board import FlightBoard

        board = FlightBoard()
        board.update_snapshot(sample_snapshot)

        # At top, navigate up should stay at top
        board.navigate_up()
        assert board.selected_env_id == 2  # Still at first

    def test_flight_board_expand_collapse(self, sample_snapshot: TuiSnapshot) -> None:
        """FlightBoard expands and collapses envs."""
        from esper.karn.overwatch.widgets.flight_board import FlightBoard

        board = FlightBoard()
        board.update_snapshot(sample_snapshot)

        # Initially not expanded
        assert not board.is_expanded(2)

        # Expand
        board.toggle_expand()
        assert board.is_expanded(2)

        # Collapse
        board.toggle_expand()
        assert not board.is_expanded(2)

    def test_flight_board_empty_state(self) -> None:
        """FlightBoard handles empty snapshot."""
        from esper.karn.overwatch.widgets.flight_board import FlightBoard

        board = FlightBoard()

        # Should handle no data gracefully
        order = board.get_display_order()
        assert order == []
        assert board.selected_env_id is None
