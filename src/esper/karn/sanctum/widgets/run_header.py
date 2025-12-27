"""Run Header Widget for Sanctum.

Displays run identity, connection status, and training progress
in a two-line header at the top of the Sanctum TUI.

Row 1: Episode | Epoch/Max | Batch | Runtime | Best Accuracy
Row 2: Connection Status | Env Health Summary | Seed Totals

Reference: Overwatch's run_header.py adapted for Sanctum schema.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.widgets import Static

from esper.leyline import STAGE_COLORS

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

    In A/B mode, adds comparison info:
        ... | A/B: +7.0% acc | Leading: A

    Provides at-a-glance training context that was in the old Rich TUI header.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize RunHeader widget."""
        super().__init__(**kwargs)
        self._snapshot: SanctumSnapshot | None = None
        # A/B comparison state
        self._group_a_accuracy = 0.0
        self._group_b_accuracy = 0.0
        self._group_a_reward = 0.0
        self._group_b_reward = 0.0
        self._leader: str | None = None
        self._ab_mode: bool = False

    @property
    def leader(self) -> str | None:
        """Return group ID of current leader (A, B, or None if tied)."""
        return self._leader

    def update_comparison(
        self,
        group_a_accuracy: float,
        group_b_accuracy: float,
        group_a_reward: float,
        group_b_reward: float,
    ) -> None:
        """Update A/B comparison metrics.

        Args:
            group_a_accuracy: Mean accuracy for policy A.
            group_b_accuracy: Mean accuracy for policy B.
            group_a_reward: Mean reward for policy A.
            group_b_reward: Mean reward for policy B.
        """
        self._ab_mode = True
        self._group_a_accuracy = group_a_accuracy
        self._group_b_accuracy = group_b_accuracy
        self._group_a_reward = group_a_reward
        self._group_b_reward = group_b_reward

        # Determine leader: reward-first (primary RL objective), accuracy as tiebreaker
        reward_delta = group_a_reward - group_b_reward
        mean_reward = (abs(group_a_reward) + abs(group_b_reward)) / 2

        # Significant reward difference (>5% of mean) is decisive
        if mean_reward > 0 and abs(reward_delta) > 0.05 * mean_reward:
            self._leader = "A" if reward_delta > 0 else "B"
        # Fallback to accuracy for close reward races
        elif group_a_accuracy > group_b_accuracy:
            self._leader = "A"
        elif group_b_accuracy > group_a_accuracy:
            self._leader = "B"
        else:
            self._leader = None

        self.refresh()

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

        # Icons provide color-independent status (accessibility):
        # ● filled = good, ◐ half = warning, ○ empty = bad
        staleness = self._snapshot.staleness_seconds
        if staleness < 2.0:
            return ("● LIVE", "green")
        elif staleness < 5.0:
            return (f"◐ SLOW ({staleness:.0f}s)", "yellow")
        else:
            return (f"○ STALE ({staleness:.0f}s)", "red")

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
                elif stage in ("BLENDING", "HOLDING"):
                    blending += 1
                elif stage == "FOSSILIZED":
                    fossilized += 1

        parts = []
        if training > 0:
            parts.append(f"[{STAGE_COLORS['TRAINING']}]T:{training}[/]")
        if blending > 0:
            parts.append(f"[{STAGE_COLORS['BLENDING']}]B:{blending}[/]")
        if fossilized > 0:
            parts.append(f"[{STAGE_COLORS['FOSSILIZED']}]F:{fossilized}[/]")

        return " ".join(parts) if parts else "[dim]No seeds[/]"

    def _get_system_alarm_indicator(self) -> str:
        """Get system alarm indicator for header.

        Returns:
            "OK" if no alarms, or alarm indicator like "cuda:0 95% │ RAM 92% │ CPU 95%"
        """
        if self._snapshot is None:
            return "OK"

        vitals = self._snapshot.vitals
        alarms = []

        # Check CPU utilization (>90% is concerning)
        if vitals.cpu_percent is not None and vitals.cpu_percent > 90:
            alarms.append(f"CPU {int(vitals.cpu_percent)}%")

        # Check memory alarms
        for device in vitals.memory_alarm_devices:
            if device == "RAM":
                if vitals.ram_used_gb is not None and vitals.ram_total_gb is not None and vitals.ram_total_gb > 0:
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

    @property
    def has_system_alarm(self) -> bool:
        """Check if any system alarm is active (memory or CPU)."""
        if self._snapshot is None:
            return False
        vitals = self._snapshot.vitals
        # Memory alarm
        if vitals.has_memory_alarm:
            return True
        # CPU alarm (>90%)
        if vitals.cpu_percent is not None and vitals.cpu_percent > 90:
            return True
        return False

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

        # Throughput: epochs/sec and batches/min (always visible)
        eps = s.vitals.epochs_per_second
        bpm = s.vitals.batches_per_hour / 60  # Convert to per minute
        row1.append(f"{eps:.1f}", style="cyan")
        row1.append(" ep/s  ", style="dim")
        row1.append(f"{bpm:.1f}", style="cyan")
        row1.append(" batch/min", style="dim")
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

        # Rolling average with sparkline (always visible)
        row1.append("  |  ", style="dim")
        row1.append("Avg: ", style="dim")
        if s.mean_accuracy_history:
            from esper.karn.sanctum.schema import make_sparkline
            current_mean = s.mean_accuracy_history[-1]
            sparkline = make_sparkline(s.mean_accuracy_history, width=10)
            row1.append(f"{current_mean:.1f}%", style="cyan")
            row1.append(f" {sparkline}", style="cyan")
        else:
            row1.append("--", style="dim")

        # A/B Comparison (only when in A/B testing mode)
        if self._ab_mode:
            row1.append("  |  ", style="dim")
            delta_acc = self._group_a_accuracy - self._group_b_accuracy
            sign = "+" if delta_acc >= 0 else ""
            # Bold colors for significant differences (>5%)
            if abs(delta_acc) > 5:
                acc_style = "green bold" if delta_acc > 0 else "red bold"
            else:
                acc_style = "dim"
            row1.append("A/B: ", style="dim")
            row1.append(f"{sign}{delta_acc:.1f}%", style=acc_style)
            row1.append(" acc", style="dim")
            row1.append("  ", style="dim")
            # Leader indicator
            if self._leader:
                color = "green" if self._leader == "A" else "cyan"
                row1.append(f"Leading: {self._leader}", style=f"{color} bold")
            else:
                row1.append("Tied", style="dim italic")

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

        # Dynamic border: red when any system alarm active (memory or CPU)
        border_style = "bold red" if self.has_system_alarm else "blue"

        return Panel(
            table,
            title="[bold]RUN STATUS[/bold]",
            subtitle=f"[{alarm_style}]{alarm_indicator}[/]",
            subtitle_align="right",
            border_style=border_style,
        )
