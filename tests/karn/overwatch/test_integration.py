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
        from esper.karn.overwatch.widgets.run_header import RunHeader

        app = OverwatchApp(replay_path=sample_replay_path)

        async with app.run_test() as pilot:
            # App should have loaded the snapshot
            assert app._snapshot is not None
            assert app._snapshot.run_id == "test-run-001"

            # Header should show snapshot info
            header = app.query_one(RunHeader)
            assert "test-run-001" in header.render_line1()

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
        from esper.karn.overwatch.widgets.run_header import RunHeader

        app = OverwatchApp()

        async with app.run_test() as pilot:
            # Should show placeholder content
            header = app.query_one(RunHeader)
            assert "Waiting" in header.render_line1()


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
        from esper.karn.overwatch.widgets.context_panel import ContextPanel

        app = OverwatchApp(replay_path=multi_env_replay)

        async with app.run_test() as pilot:
            # Verify flight board has initial selection
            board = app.query_one(FlightBoard)
            assert board.selected_env_id == 2

            # Navigate to env 1
            await pilot.press("j")
            await pilot.pause()

            # Panel should update after navigation - query ContextPanel instead of Static
            context = app.query_one(ContextPanel)
            detail_text = context.render_content()
            assert "Env 1" in detail_text, f"Expected 'Env 1' in detail panel, got: {detail_text}"


class TestHeaderAndStripIntegration:
    """Integration tests for RunHeader and TamiyoStrip."""

    @pytest.fixture
    def tamiyo_replay(self, tmp_path: Path) -> Path:
        """Create replay with Tamiyo data."""
        from esper.karn.overwatch import (
            SnapshotWriter,
            TuiSnapshot,
            ConnectionStatus,
            TamiyoState,
            EnvSummary,
        )

        path = tmp_path / "tamiyo.jsonl"
        with SnapshotWriter(path) as writer:
            snap = TuiSnapshot(
                schema_version=1,
                captured_at="2025-12-18T14:00:00Z",
                connection=ConnectionStatus(True, 1000.0, 0.5),
                tamiyo=TamiyoState(
                    kl_divergence=0.015,
                    entropy=1.5,
                    explained_variance=0.75,
                    kl_trend=0.002,
                    entropy_trend=-0.05,
                    ev_trend=0.01,
                    action_counts={"GERMINATE": 10, "WAIT": 90},
                ),
                run_id="test-001",
                task_name="cifar10",
                episode=5,
                batch=100,
                runtime_s=600.0,
                envs_ok=2,
                envs_warn=1,
                envs_crit=0,
                flight_board=[
                    EnvSummary(env_id=0, device_id=0, status="OK", anomaly_score=0.1),
                    EnvSummary(env_id=1, device_id=0, status="WARN", anomaly_score=0.6),
                ],
            )
            writer.write(snap)
        return path

    @pytest.mark.asyncio
    async def test_run_header_displays_data(self, tamiyo_replay: Path) -> None:
        """RunHeader shows run identity when loaded."""
        from esper.karn.overwatch import OverwatchApp
        from esper.karn.overwatch.widgets.run_header import RunHeader

        app = OverwatchApp(replay_path=tamiyo_replay)

        async with app.run_test() as pilot:
            header = app.query_one(RunHeader)
            content = header.render_line1()

            assert "test-001" in content
            assert "cifar10" in content
            assert "5" in content  # episode

    @pytest.mark.asyncio
    async def test_tamiyo_strip_displays_vitals(self, tamiyo_replay: Path) -> None:
        """TamiyoStrip shows PPO vitals when loaded."""
        from esper.karn.overwatch import OverwatchApp
        from esper.karn.overwatch.widgets.tamiyo_strip import TamiyoStrip

        app = OverwatchApp(replay_path=tamiyo_replay)

        async with app.run_test() as pilot:
            strip = app.query_one(TamiyoStrip)
            content = strip.render_vitals()

            assert "KL" in content
            assert "Ent" in content
            assert "EV" in content

    @pytest.mark.asyncio
    async def test_tamiyo_strip_shows_trend_arrows(self, tamiyo_replay: Path) -> None:
        """TamiyoStrip shows trend arrows for metrics."""
        from esper.karn.overwatch import OverwatchApp
        from esper.karn.overwatch.widgets.tamiyo_strip import TamiyoStrip

        app = OverwatchApp(replay_path=tamiyo_replay)

        async with app.run_test() as pilot:
            strip = app.query_one(TamiyoStrip)
            content = strip.render_vitals()

            # entropy_trend=-0.05 should show ↓
            assert "↓" in content

    @pytest.mark.asyncio
    async def test_header_shows_env_counts(self, tamiyo_replay: Path) -> None:
        """RunHeader shows environment health counts."""
        from esper.karn.overwatch import OverwatchApp
        from esper.karn.overwatch.widgets.run_header import RunHeader

        app = OverwatchApp(replay_path=tamiyo_replay)

        async with app.run_test() as pilot:
            header = app.query_one(RunHeader)
            content = header.render_line2()

            # envs_ok=2, envs_warn=1
            assert "OK:2" in content or "2" in content
            assert "WARN:1" in content or "1" in content


class TestDetailPanelIntegration:
    """Integration tests for DetailPanel functionality."""

    @pytest.fixture
    def detail_replay(self, tmp_path: Path) -> Path:
        """Create replay with detailed env and tamiyo data."""
        from esper.karn.overwatch import (
            SnapshotWriter,
            TuiSnapshot,
            ConnectionStatus,
            TamiyoState,
            EnvSummary,
            SlotChipState,
        )

        path = tmp_path / "detail.jsonl"
        with SnapshotWriter(path) as writer:
            snap = TuiSnapshot(
                schema_version=1,
                captured_at="2025-12-18T14:00:00Z",
                connection=ConnectionStatus(True, 1000.0, 0.5),
                tamiyo=TamiyoState(
                    action_counts={"GERMINATE": 34, "BLEND": 28, "CULL": 12, "WAIT": 26},
                    recent_actions=["G", "B", "W", "W", "C"],
                    confidence_mean=0.73,
                    confidence_min=0.45,
                    confidence_max=0.92,
                    confidence_history=[0.5, 0.6, 0.7, 0.65, 0.8],
                    exploration_pct=0.65,
                    kl_divergence=0.015,
                    entropy=1.5,
                    explained_variance=0.75,
                ),
                flight_board=[
                    EnvSummary(
                        env_id=3,
                        device_id=1,
                        status="WARN",
                        anomaly_score=0.72,
                        anomaly_reasons=["High gradient ratio (3.2x)", "Memory pressure (94%)"],
                        throughput_fps=102.5,
                        slots={
                            "r0c1": SlotChipState("r0c1", "BLENDING", "conv_light", 0.78, gate_last="G2", gate_passed=True),
                        },
                    ),
                    EnvSummary(env_id=0, device_id=0, status="OK", anomaly_score=0.1),
                ],
            )
            writer.write(snap)
        return path

    @pytest.mark.asyncio
    async def test_detail_panel_shows_context_by_default(self, detail_replay: Path) -> None:
        """DetailPanel starts in context mode."""
        from esper.karn.overwatch import OverwatchApp
        from esper.karn.overwatch.widgets.detail_panel import DetailPanel

        app = OverwatchApp(replay_path=detail_replay)

        async with app.run_test() as pilot:
            panel = app.query_one(DetailPanel)
            assert panel.mode == "context"

    @pytest.mark.asyncio
    async def test_t_key_switches_to_tamiyo_mode(self, detail_replay: Path) -> None:
        """Pressing 't' switches to tamiyo mode."""
        from esper.karn.overwatch import OverwatchApp
        from esper.karn.overwatch.widgets.detail_panel import DetailPanel

        app = OverwatchApp(replay_path=detail_replay)

        async with app.run_test() as pilot:
            panel = app.query_one(DetailPanel)
            assert panel.mode == "context"

            await pilot.press("t")
            assert panel.mode == "tamiyo"

    @pytest.mark.asyncio
    async def test_c_key_switches_to_context_mode(self, detail_replay: Path) -> None:
        """Pressing 'c' switches to context mode."""
        from esper.karn.overwatch import OverwatchApp
        from esper.karn.overwatch.widgets.detail_panel import DetailPanel

        app = OverwatchApp(replay_path=detail_replay)

        async with app.run_test() as pilot:
            panel = app.query_one(DetailPanel)

            # Switch to tamiyo first
            await pilot.press("t")
            assert panel.mode == "tamiyo"

            # Switch back to context
            await pilot.press("c")
            assert panel.mode == "context"

    @pytest.mark.asyncio
    async def test_context_panel_shows_anomaly_reasons(self, detail_replay: Path) -> None:
        """Context panel displays anomaly reasons for selected env."""
        from esper.karn.overwatch import OverwatchApp
        from esper.karn.overwatch.widgets.context_panel import ContextPanel

        app = OverwatchApp(replay_path=detail_replay)

        async with app.run_test() as pilot:
            # Initial selection should be highest anomaly (env 3)
            context = app.query_one(ContextPanel)
            content = context.render_content()

            assert "High gradient ratio" in content
            assert "Memory pressure" in content

    @pytest.mark.asyncio
    async def test_tamiyo_detail_shows_action_distribution(self, detail_replay: Path) -> None:
        """Tamiyo detail panel shows action distribution."""
        from esper.karn.overwatch import OverwatchApp
        from esper.karn.overwatch.widgets.tamiyo_detail import TamiyoDetailPanel

        app = OverwatchApp(replay_path=detail_replay)

        async with app.run_test() as pilot:
            # Switch to tamiyo mode
            await pilot.press("t")

            tamiyo = app.query_one(TamiyoDetailPanel)
            content = tamiyo.render_content()

            assert "GERM" in content or "Germinate" in content
            assert "34" in content  # 34%
