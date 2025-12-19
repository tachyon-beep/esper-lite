"""Run Header Widget.

Displays run identity, connection status, and environment health summary
in two rows at the top of the Overwatch TUI.

Row 1: Run ID | Task | Episode | Batch | Runtime
Row 2: Connection | Best Metric | Env Counts (OK/WARN/CRIT)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Static

from esper.karn.overwatch.display_state import format_runtime

if TYPE_CHECKING:
    from esper.karn.overwatch.schema import TuiSnapshot


class RunHeader(Container):
    """Widget displaying run identity and connection status.

    Two-line format:
        exp-001 | cifar10 | Ep 5 | Batch 150 | 1h 2m
        ● Live | Best: 82.3% | OK:3 WARN:1 CRIT:0
    """

    DEFAULT_CSS = """
    RunHeader {
        width: 100%;
        height: 2;
        padding: 0 1;
        background: $surface;
    }

    RunHeader .header-line {
        width: 100%;
        height: 1;
    }

    RunHeader .connection-live {
        color: $success;
    }

    RunHeader .connection-stale {
        color: $warning;
    }

    RunHeader .connection-disconnected {
        color: $error;
    }

    RunHeader .env-ok {
        color: $success;
    }

    RunHeader .env-warn {
        color: $warning;
    }

    RunHeader .env-crit {
        color: $error;
    }
    """

    def __init__(self, **kwargs) -> None:
        """Initialize the run header."""
        super().__init__(**kwargs)
        self._snapshot: TuiSnapshot | None = None

    def render_line1(self) -> str:
        """Render first line: run identity."""
        if self._snapshot is None:
            return "-- | Waiting for data..."

        s = self._snapshot
        run_id = s.run_id or "--"
        task = s.task_name or "--"
        runtime = format_runtime(s.runtime_s)

        return f"{run_id} | {task} | Ep {s.episode} | Batch {s.batch} | {runtime}"

    def render_line2(self) -> str:
        """Render second line: connection and health summary."""
        if self._snapshot is None:
            return "○ Disconnected | -- | --"

        s = self._snapshot
        c = s.connection

        # Connection indicator
        if not c.connected:
            conn = "○ Disconnected"
        elif c.staleness_s < 2.0:
            conn = "● Live"
        elif c.staleness_s < 5.0:
            conn = f"● Live ({c.staleness_s:.0f}s)"
        else:
            conn = f"◐ Stale ({c.staleness_s:.0f}s)"

        # Best metric
        best = f"Best: {s.best_metric*100:.1f}%" if s.best_metric > 0 else "Best: --"

        # Env counts
        envs = f"OK:{s.envs_ok} WARN:{s.envs_warn} CRIT:{s.envs_crit}"

        return f"{conn} | {best} | {envs}"

    def compose(self) -> ComposeResult:
        """Compose the header layout."""
        yield Static(self.render_line1(), classes="header-line", id="header-line1")
        yield Static(self.render_line2(), classes="header-line", id="header-line2")

    def update_snapshot(self, snapshot: TuiSnapshot) -> None:
        """Update with new snapshot data."""
        self._snapshot = snapshot
        self._refresh_content()

    def _refresh_content(self) -> None:
        """Refresh the displayed content."""
        try:
            self.query_one("#header-line1", Static).update(self.render_line1())
            self.query_one("#header-line2", Static).update(self.render_line2())
        except Exception:
            # Widget not mounted yet
            pass
