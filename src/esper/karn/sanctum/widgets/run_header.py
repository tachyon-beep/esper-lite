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

    def _get_system_alarm_indicator(self) -> str:
        """Get system alarm indicator for header.

        Returns:
            "OK" if no memory alarm, or alarm indicator like "cuda:0 95% │ RAM 92%"
        """
        if self._snapshot is None:
            return "OK"

        vitals = self._snapshot.vitals
        if not vitals.has_memory_alarm:
            return "OK"

        alarms = []
        for device in vitals.memory_alarm_devices:
            if device == "RAM":
                pct = int((vitals.ram_used_gb / vitals.ram_total_gb) * 100)
                alarms.append(f"RAM {pct}%")
            elif device.startswith("cuda:"):
                device_id = int(device.split(":")[1])
                stats = vitals.gpu_stats.get(device_id)
                if stats and stats.memory_total_gb > 0:
                    pct = int((stats.memory_used_gb / stats.memory_total_gb) * 100)
                    alarms.append(f"{device} {pct}%")
                elif device_id == 0 and vitals.gpu_memory_total_gb > 0:
                    # Fallback for single GPU
                    pct = int((vitals.gpu_memory_used_gb / vitals.gpu_memory_total_gb) * 100)
                    alarms.append(f"cuda:0 {pct}%")

        return " │ ".join(alarms) if alarms else "OK"

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

        # Rolling average with sparkline
        if s.mean_accuracy_history:
            from esper.karn.sanctum.schema import make_sparkline
            current_mean = s.mean_accuracy_history[-1] if s.mean_accuracy_history else 0
            sparkline = make_sparkline(s.mean_accuracy_history, width=10)
            row1.append("  |  ", style="dim")
            row1.append("Avg: ", style="dim")
            row1.append(f"{current_mean:.1f}%", style="cyan")
            row1.append(f" {sparkline}", style="cyan")

        table.add_row(row1)

        # === Row 2: Connection + Diagnostics + Health ===
        row2 = Text()

        # Connection status
        conn_text, conn_style = self._get_connection_status()
        row2.append(conn_text, style=conn_style)
        row2.append("  |  ", style="dim")

        # Thread status indicator (critical for debugging crashes)
        if s.training_thread_alive is True:
            row2.append("Thread ", style="dim")
            row2.append("✓", style="green")
        elif s.training_thread_alive is False:
            row2.append("Thread ", style="dim")
            row2.append("✗ DEAD", style="bold red")
        else:
            row2.append("Thread ", style="dim")
            row2.append("?", style="dim")
        row2.append("  |  ", style="dim")

        # Event stats - show rate if meaningful
        if s.total_events_received > 0 and s.runtime_seconds > 0:
            events_per_sec = s.total_events_received / s.runtime_seconds
            row2.append(f"{s.total_events_received:,}", style="cyan")
            row2.append(" events ", style="dim")
            row2.append(f"({events_per_sec:.1f}/s)", style="dim")
        else:
            row2.append(f"{s.total_events_received}", style="cyan")
            row2.append(" events", style="dim")
        row2.append("  |  ", style="dim")

        # Env health summary (uses Rich markup, need to parse it)
        row2.append_text(Text.from_markup(self._get_env_health_summary()))
        row2.append("  |  ", style="dim")

        # Seed stage counts (uses Rich markup, need to parse it)
        row2.append_text(Text.from_markup(self._get_seed_stage_counts()))

        # Task name if available
        if s.task_name:
            row2.append("  |  ", style="dim")
            row2.append(s.task_name, style="italic cyan")

        table.add_row(row2)

        # System alarm indicator
        alarm_indicator = self._get_system_alarm_indicator()
        alarm_style = "green" if alarm_indicator == "OK" else "bold red"

        return Panel(
            table,
            title="[bold]RUN STATUS[/bold]",
            subtitle=f"[{alarm_style}]{alarm_indicator}[/]",
            subtitle_align="right",
            border_style="blue",
        )
