"""Tests for TUI rendering components."""

import pytest
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from esper.karn.tui import TUIOutput


class TestEnvOverviewTable:
    """Tests for environment overview table rendering."""

    def test_render_env_overview_returns_panel(self):
        """_render_env_overview returns a Rich Panel containing a Table."""
        tui = TUIOutput()
        tui.state.n_envs = 2

        for i in range(2):
            env = tui.state.get_or_create_env(i)
            env.host_accuracy = 75.0 + i * 5
            env.reward_history.append(0.3 + i * 0.1)
            env.status = "healthy"

        panel = tui._render_env_overview()
        assert isinstance(panel, Panel)

    def test_env_overview_shows_all_envs(self):
        """Table has one row per environment plus separator and aggregate."""
        tui = TUIOutput()
        tui.state.n_envs = 4

        for i in range(4):
            env = tui.state.get_or_create_env(i)
            env.status = "healthy"

        panel = tui._render_env_overview()
        # Panel.renderable is the Table
        table = panel.renderable
        # 4 envs + separator row + aggregate row = 6 rows
        assert table.row_count == 6

    def test_env_overview_uses_sparklines(self):
        """Table uses sparkline properties from EnvState."""
        tui = TUIOutput()
        tui.state.n_envs = 2

        env0 = tui.state.get_or_create_env(0)
        env0.reward_history.extend([0.1, 0.2, 0.3, 0.4, 0.5])
        env0.accuracy_history.extend([60.0, 65.0, 70.0, 75.0, 80.0])
        env0.status = "healthy"

        env1 = tui.state.get_or_create_env(1)
        env1.status = "stalled"

        # Just verify it renders without error (sparklines are visual)
        panel = tui._render_env_overview()
        assert panel is not None

    def test_env_overview_status_colors(self):
        """Different statuses should render (color is visual, verify no crash)."""
        tui = TUIOutput()
        tui.state.n_envs = 4

        statuses = ["excellent", "healthy", "stalled", "degraded"]
        for i, status in enumerate(statuses):
            env = tui.state.get_or_create_env(i)
            env.status = status

        # Render to string to verify no errors
        console = Console(force_terminal=True, width=120)
        panel = tui._render_env_overview()
        with console.capture() as capture:
            console.print(panel)
        output = capture.get()
        assert "ENVIRONMENT OVERVIEW" in output
