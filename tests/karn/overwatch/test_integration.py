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
