"""Run Header Widget for Sanctum.

Displays run identity, connection status, and training progress
in a two-line header at the top of the Sanctum TUI.

Row 1: Episode | Epoch/Max | Batch | Runtime | Best Accuracy
Row 2: Connection Status | Env Health Summary | Seed Totals

Reference: Overwatch's run_header.py adapted for Sanctum schema.
"""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.widgets import Static

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import SanctumSnapshot


def _format_runtime(seconds: float) -> str:
    """Format runtime as Xh Ym Zs."""
    if seconds <= 0:
        return "--"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


class RunHeader(Static):
    """Run header widget showing training progress and connection status.

    Two-row format:
        Ep 5 | 150/500 epochs | Batch 3 | 1h 23m | Best: 82.3% (ep 4)
        ● Live | 4 healthy, 0 stalled | Train:8 Blend:2 Foss:12

    Provides at-a-glance training context that was in the old Rich TUI header.
    """

    def __init__(self, **kwargs) -> None:
        """Initialize RunHeader widget."""
        super().__init__(**kwargs)
        self._snapshot: SanctumSnapshot | None = None

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        """Update widget with new snapshot data."""
        self._snapshot = snapshot
        self.refresh()

    def _get_connection_status(self) -> tuple[str, str]:
        """Get connection indicator and style.

        Returns:
            Tuple of (indicator_text, style).
        """
        if self._snapshot is None:
            return ("○ Waiting", "dim")

        if not self._snapshot.connected:
            return ("○ Disconnected", "red")

        staleness = self._snapshot.staleness_seconds
        if staleness < 2.0:
            return ("● Live", "green")
        elif staleness < 5.0:
            return (f"● Live ({staleness:.0f}s)", "yellow")
        else:
            return (f"◐ Stale ({staleness:.0f}s)", "red")

    def _get_env_health_summary(self) -> str:
        """Get environment health summary."""
        if self._snapshot is None or not self._snapshot.envs:
            return "No envs"

        healthy = 0
        stalled = 0
        degraded = 0

        for env in self._snapshot.envs.values():
            status = env.status.lower()
            if status in ("healthy", "excellent"):
                healthy += 1
            elif status == "stalled":
                stalled += 1
            elif status == "degraded":
                degraded += 1
            else:
                healthy += 1  # initializing counts as healthy

        parts = []
        if healthy > 0:
            parts.append(f"[green]{healthy} OK[/]")
        if stalled > 0:
            parts.append(f"[yellow]{stalled} stall[/]")
        if degraded > 0:
            parts.append(f"[red]{degraded} deg[/]")

        return " ".join(parts) if parts else "No envs"

    def _get_seed_stage_counts(self) -> str:
        """Get seed stage counts across all envs."""
        if self._snapshot is None or not self._snapshot.envs:
            return ""

        training = 0
        blending = 0
        fossilized = 0

        for env in self._snapshot.envs.values():
            for seed in env.seeds.values():
                stage = seed.stage.upper()
                if stage == "TRAINING":
                    training += 1
                elif stage in ("BLENDING", "PROBATIONARY"):
                    blending += 1
                elif stage == "FOSSILIZED":
                    fossilized += 1

        parts = []
        if training > 0:
            parts.append(f"[yellow]T:{training}[/]")
        if blending > 0:
            parts.append(f"[cyan]B:{blending}[/]")
        if fossilized > 0:
            parts.append(f"[magenta]F:{fossilized}[/]")

        return " ".join(parts) if parts else "[dim]No seeds[/]"

    def render(self) -> Panel:
        """Render the run header panel."""
        if self._snapshot is None:
            return Panel(
                Text("Waiting for training data...", style="dim"),
                title="[bold]RUN STATUS[/bold]",
                border_style="blue",
            )

        s = self._snapshot

        # Build two-row table
        table = Table(show_header=False, box=None, padding=(0, 1), expand=True)
        table.add_column("Content", no_wrap=True)

        # === Row 1: Training Progress ===
        row1 = Text()
        row1.append("Ep ", style="dim")
        row1.append(f"{s.current_episode}", style="bold cyan")
        row1.append("  |  ", style="dim")

        # Epoch progress
        if s.max_epochs > 0:
            row1.append(f"{s.current_epoch}/{s.max_epochs}", style="cyan")
            row1.append(" epochs", style="dim")
        else:
            row1.append(f"{s.current_epoch} epochs", style="cyan")
        row1.append("  |  ", style="dim")

        # Batch
        row1.append("Batch ", style="dim")
        row1.append(f"{s.current_batch}", style="cyan")
        row1.append("  |  ", style="dim")

        # Runtime
        runtime = _format_runtime(s.runtime_seconds)
        row1.append(runtime, style="cyan")
        row1.append("  |  ", style="dim")

        # Best accuracy (global)
        best_accs = [e.best_accuracy for e in s.envs.values() if e.best_accuracy > 0]
        if best_accs:
            global_best = max(best_accs)
            # Find which episode achieved the best
            best_ep = 0
            for e in s.envs.values():
                if e.best_accuracy == global_best:
                    best_ep = e.best_accuracy_episode
                    break
            row1.append("Best: ", style="dim")
            row1.append(f"{global_best:.1f}%", style="bold green")
            if best_ep > 0:
                row1.append(f" (ep {best_ep})", style="dim")
        else:
            row1.append("Best: ", style="dim")
            row1.append("--", style="dim")

        table.add_row(row1)

        # === Row 2: Connection + Health ===
        row2 = Text()

        # Connection status
        conn_text, conn_style = self._get_connection_status()
        row2.append(conn_text, style=conn_style)
        row2.append("  |  ", style="dim")

        # Env health summary
        row2.append(self._get_env_health_summary())
        row2.append("  |  ", style="dim")

        # Seed stage counts
        row2.append(self._get_seed_stage_counts())

        # Task name if available
        if s.task_name:
            row2.append("  |  ", style="dim")
            row2.append(s.task_name, style="italic dim")

        table.add_row(row2)

        return Panel(
            table,
            title="[bold]RUN STATUS[/bold]",
            border_style="blue",
        )
