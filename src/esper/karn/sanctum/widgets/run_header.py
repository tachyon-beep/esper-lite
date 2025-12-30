"""Run Header Widget for Sanctum.

Single-line status bar showing essential training state at a glance.
Designed with fixed-width segments to prevent text jumping.

Layout:
│ ● LIVE ✓ │ run_name │ Ep 47 ████████░░░░ 150/500 │ 1h 23m │ 0.8e/s 2.1b/m │

Segments (all fixed-width):
- Connection: ● LIVE / ◐ SLOW / ○ STALE (6 chars)
- Thread: ✓ / ✗ (1 char, bold red if dead)
- Run name: Task/experiment name (14 chars max, truncated)
- Episode: Current episode number
- Progress: Visual bar + epoch fraction
- Runtime: Elapsed time
- Throughput: epochs/sec and batches/min

System alarms shown in subtitle (right-aligned).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

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
    """Single-line status bar for training state.

    Fixed-width layout (no text jumping):
        ● LIVE ✓ │ my_experiment │ Ep 47 ████████░░░░ 150/500 │ 1h 23m │ 0.8e/s 2.1b/m

    Design principles:
    - Fixed-width segments prevent jumping as values change
    - Only essential metrics (duplicates removed, other panels show details)
    - Visual progress bar for epoch completion
    - System alarms in subtitle, not main content
    """

    def __init__(self, **kwargs: Any) -> None:
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

        # Icons provide color-independent status (accessibility):
        # ● filled = good, ◐ half = warning, ○ empty = bad
        staleness = self._snapshot.staleness_seconds
        if staleness < 2.0:
            return ("● LIVE", "green")
        elif staleness < 5.0:
            return (f"◐ SLOW ({staleness:.0f}s)", "yellow")
        else:
            return (f"○ STALE ({staleness:.0f}s)", "red")

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

    def _render_progress_bar(self, current: int, max_epochs: int, width: int = 16) -> str:
        """Render epoch progress bar with fixed width.

        Args:
            current: Current epoch number.
            max_epochs: Maximum epochs (0 = unbounded).
            width: Bar width in characters.

        Returns:
            Progress string like "████████░░░░░░░░ 150/500" or "Epoch 150 (unbounded)".
        """
        if max_epochs <= 0:
            return f"Epoch {current} (unbounded)"

        progress = min(current / max_epochs, 1.0)
        filled = int(progress * width)
        bar = "█" * filled + "░" * (width - filled)
        return f"{bar} {current}/{max_epochs}"

    def _render_batch_progress(self, current: int, max_batches: int, width: int = 8) -> str:
        """Render batch progress meter (Tamiyo's training epochs).

        Args:
            current: Current batch number.
            max_batches: Maximum batches per episode.
            width: Bar width in characters.

        Returns:
            Compact progress string like "B:██░░ 25/100".
        """
        if max_batches <= 0:
            return f"B:{current}"

        progress = min(current / max_batches, 1.0)
        filled = int(progress * width)
        bar = "█" * filled + "░" * (width - filled)
        return f"B:{bar} {current}/{max_batches}"

    def _format_run_name(self, name: str | None, max_width: int = 14) -> str:
        """Format run name with truncation to fixed width.

        Args:
            name: Run/task name (may be None or empty).
            max_width: Maximum width including truncation ellipsis.

        Returns:
            Left-aligned name padded to max_width, or spaces if empty.
        """
        if not name:
            return " " * max_width
        if len(name) > max_width:
            return name[: max_width - 1] + "…"
        return f"{name:<{max_width}}"

    def _format_throughput(self, eps: float, bpm: float) -> str:
        """Format throughput metrics to fixed width.

        Args:
            eps: Epochs per second.
            bpm: Batches per minute.

        Returns:
            Fixed-width string like "0.8e/s 2.1b/m" (13 chars).
        """
        return f"{eps:>3.1f}e/s {bpm:>4.1f}b/m"

    def render(self) -> Text:
        """Render the run header as a single-line status bar (no border).

        Layout (1 line, fixed-width segments):
        ● LIVE ✓ │ run_name │ Ep 47 ████████░░░░ 150/500 │ 1h 23m │ 0.8e/s 2.1b/m │ ✓ System
        """
        if self._snapshot is None:
            return Text("○ WAIT   │ Waiting for training data...", style="dim")

        s = self._snapshot
        row = Text()

        # === Segment 1: Connection status (6 chars) ===
        conn_text, conn_style = self._get_connection_status()
        # Extract icon and status word
        parts = conn_text.split()
        icon = parts[0] if parts else "○"
        # Normalize status to 4 chars: LIVE, SLOW, STAL, DISC
        if "LIVE" in conn_text:
            status_word = "LIVE"
        elif "SLOW" in conn_text:
            status_word = "SLOW"
        elif "STALE" in conn_text:
            status_word = "STAL"
        elif "Disconnected" in conn_text:
            status_word = "DISC"
        else:
            status_word = "WAIT"
        row.append(f"{icon} {status_word}", style=conn_style)
        row.append(" ", style="dim")

        # === Segment 2: Thread status (1 char) ===
        if s.training_thread_alive is True:
            row.append("✓", style="green")
        elif s.training_thread_alive is False:
            # Bold red is sufficient urgency - blink is distracting and can cause issues
            row.append("✗", style="bold red")
        else:
            row.append("?", style="dim")

        row.append(" │ ", style="dim")

        # === Segment 3: Run name (14 chars fixed) ===
        run_name = self._format_run_name(s.task_name)
        row.append(run_name, style="italic cyan")

        row.append(" │ ", style="dim")

        # === Segment 4: Episode (right-aligned in 6 chars) ===
        row.append("Ep ", style="dim")
        row.append(f"{s.current_episode:>3}", style="bold cyan")
        row.append(" ", style="dim")

        # === Segment 5: Epoch progress bar + fraction ===
        progress = self._render_progress_bar(s.current_epoch, s.max_epochs)
        row.append(progress, style="cyan")

        row.append(" ", style="dim")

        # === Segment 5b: Batch progress (Tamiyo's epochs = system batches) ===
        batch_progress = self._render_batch_progress(s.current_batch, s.max_batches)
        row.append(batch_progress, style="magenta")

        row.append(" │ ", style="dim")

        # === Segment 6: Runtime (right-aligned) ===
        runtime = _format_runtime(s.runtime_seconds)
        row.append(f"{runtime:>7}", style="cyan")

        row.append(" │ ", style="dim")

        # === Segment 7: Throughput (fixed 15 chars) ===
        eps = s.vitals.epochs_per_second
        bpm = s.vitals.batches_per_hour / 60  # Convert to per minute
        throughput = self._format_throughput(eps, bpm)
        row.append(throughput, style="dim")

        # === Segment 8: System alarm indicator (at end) ===
        row.append(" │ ", style="dim")
        if self.has_system_alarm:
            alarm_indicator = self._get_system_alarm_indicator()
            row.append(f"⚠ {alarm_indicator}", style="bold red")
        else:
            row.append("✓ System", style="green dim")

        return row
