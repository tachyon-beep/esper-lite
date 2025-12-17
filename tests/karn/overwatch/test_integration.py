"""Integration tests for Overwatch TUI."""

from __future__ import annotations

from pathlib import Path

import pytest


class TestOverwatchIntegration:
    """Integration tests for the full app."""

    @pytest.fixture
    def sample_replay_path(self, tmp_path: Path) -> Path:
        """Create a sample replay file for testing."""
        from esper.karn.overwatch import (
            SnapshotWriter,
            TuiSnapshot,
            ConnectionStatus,
            TamiyoState,
            EnvSummary,
            SlotChipState,
        )

        path = tmp_path / "test_replay.jsonl"
        with SnapshotWriter(path) as writer:
            snap = TuiSnapshot(
                schema_version=1,
                captured_at="2025-12-18T12:00:00Z",
                connection=ConnectionStatus(True, 1000.0, 0.5),
                tamiyo=TamiyoState(
                    kl_divergence=0.019,
                    entropy=1.24,
                    action_counts={"GERMINATE": 10, "BLEND": 20},
                ),
                run_id="test-run-001",
                task_name="cifar10",
                episode=47,
                flight_board=[
                    EnvSummary(
                        env_id=0,
                        device_id=0,
                        status="OK",
                        slots={"r0c1": SlotChipState("r0c1", "TRAINING", "conv", 0.5)},
                    ),
                    EnvSummary(
                        env_id=1,
                        device_id=1,
                        status="WARN",
                        anomaly_score=0.65,
                    ),
                ],
                envs_ok=1,
                envs_warn=1,
            )
            writer.write(snap)
        return path

    async def test_app_launches_with_replay(self, sample_replay_path: Path) -> None:
        """App launches and loads replay file."""
        from esper.karn.overwatch import OverwatchApp

        app = OverwatchApp(replay_path=sample_replay_path)

        async with app.run_test() as pilot:
            # App should have loaded the snapshot
            assert app._snapshot is not None
            assert app._snapshot.run_id == "test-run-001"

            # Header should show snapshot info
            header = app.query_one("#header")
            assert "test-run-001" in str(header.render())

    async def test_app_help_toggle(self, sample_replay_path: Path) -> None:
        """? key toggles help overlay."""
        from esper.karn.overwatch import OverwatchApp

        app = OverwatchApp(replay_path=sample_replay_path)

        async with app.run_test() as pilot:
            # Help should be hidden initially
            help_overlay = app.query_one("#help-overlay")
            assert "hidden" in help_overlay.classes

            # Press ? to show help
            await pilot.press("question_mark")
            assert "hidden" not in help_overlay.classes

            # Press Esc to hide help
            await pilot.press("escape")
            assert "hidden" in help_overlay.classes

    async def test_app_quit(self, sample_replay_path: Path) -> None:
        """q key quits the app."""
        from esper.karn.overwatch import OverwatchApp

        app = OverwatchApp(replay_path=sample_replay_path)

        async with app.run_test() as pilot:
            await pilot.press("q")
            # App should be exiting
            assert app._exit


class TestAppWithoutReplay:
    """Tests for app behavior without replay file."""

    async def test_app_launches_without_data(self) -> None:
        """App launches even without replay file."""
        from esper.karn.overwatch import OverwatchApp

        app = OverwatchApp()

        async with app.run_test() as pilot:
            # Should show placeholder content
            header = app.query_one("#header")
            assert "Waiting" in str(header.render())


class TestFlightBoardNavigation:
    """Integration tests for flight board navigation."""

    @pytest.fixture
    def multi_env_replay(self, tmp_path: Path) -> Path:
        """Create replay with multiple envs for navigation testing."""
        from esper.karn.overwatch import (
            SnapshotWriter,
            TuiSnapshot,
            ConnectionStatus,
            TamiyoState,
            EnvSummary,
            SlotChipState,
        )

        path = tmp_path / "multi_env.jsonl"
        with SnapshotWriter(path) as writer:
            snap = TuiSnapshot(
                schema_version=1,
                captured_at="2025-12-18T12:00:00Z",
                connection=ConnectionStatus(True, 1000.0, 0.5),
                tamiyo=TamiyoState(),
                flight_board=[
                    EnvSummary(
                        env_id=0, device_id=0, status="OK",
                        anomaly_score=0.1,
                        slots={"r0c1": SlotChipState("r0c1", "TRAINING", "conv", 0.5)},
                    ),
                    EnvSummary(
                        env_id=1, device_id=0, status="WARN",
                        anomaly_score=0.65,
                    ),
                    EnvSummary(
                        env_id=2, device_id=1, status="CRIT",
                        anomaly_score=0.85,
                        anomaly_reasons=["High gradient ratio"],
                    ),
                ],
            )
            writer.write(snap)
        return path

    @pytest.mark.asyncio
    async def test_navigation_j_k(self, multi_env_replay: Path) -> None:
        """j/k keys navigate between envs."""
        from esper.karn.overwatch import OverwatchApp
        from esper.karn.overwatch.widgets.flight_board import FlightBoard

        app = OverwatchApp(replay_path=multi_env_replay)

        async with app.run_test() as pilot:
            board = app.query_one(FlightBoard)

            # Should start at highest anomaly (env 2)
            assert board.selected_env_id == 2

            # Navigate down
            await pilot.press("j")
            assert board.selected_env_id == 1

            # Navigate down again
            await pilot.press("j")
            assert board.selected_env_id == 0

            # Navigate back up
            await pilot.press("k")
            assert board.selected_env_id == 1

    @pytest.mark.asyncio
    async def test_navigation_arrows(self, multi_env_replay: Path) -> None:
        """Arrow keys navigate between envs."""
        from esper.karn.overwatch import OverwatchApp
        from esper.karn.overwatch.widgets.flight_board import FlightBoard

        app = OverwatchApp(replay_path=multi_env_replay)

        async with app.run_test() as pilot:
            board = app.query_one(FlightBoard)

            # Navigate with arrows
            await pilot.press("down")
            assert board.selected_env_id == 1

            await pilot.press("up")
            assert board.selected_env_id == 2

    @pytest.mark.asyncio
    async def test_expand_collapse(self, multi_env_replay: Path) -> None:
        """Enter expands env, Esc collapses."""
        from esper.karn.overwatch import OverwatchApp
        from esper.karn.overwatch.widgets.flight_board import FlightBoard

        app = OverwatchApp(replay_path=multi_env_replay)

        async with app.run_test() as pilot:
            board = app.query_one(FlightBoard)

            # Initially not expanded
            assert not board.is_expanded(2)

            # Expand with Enter
            await pilot.press("enter")
            assert board.is_expanded(2)

            # Collapse with Enter again
            await pilot.press("enter")
            assert not board.is_expanded(2)

    @pytest.mark.asyncio
    async def test_detail_panel_updates_on_selection(self, multi_env_replay: Path) -> None:
        """Detail panel updates when env is selected via navigation."""
        from esper.karn.overwatch import OverwatchApp
        from esper.karn.overwatch.widgets.flight_board import FlightBoard
        from textual.widgets import Static

        app = OverwatchApp(replay_path=multi_env_replay)

        async with app.run_test() as pilot:
            # Verify flight board has initial selection
            board = app.query_one(FlightBoard)
            assert board.selected_env_id == 2

            # Navigate to env 1
            await pilot.press("j")
            await pilot.pause()

            # Panel should update after navigation
            detail = app.query_one("#detail-panel", Static)
            detail_text = str(detail.render())
            assert "Env 1" in detail_text, f"Expected 'Env 1' in detail panel, got: {detail_text}"
