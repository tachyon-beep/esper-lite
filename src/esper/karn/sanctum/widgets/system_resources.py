"""SystemResources widget - CPU/RAM/GPU monitoring with visual bars.

Replaces EsperStatus with a more compact, visually-oriented display.
Shows resource utilization with progress bars and color-coded thresholds.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Group
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text
from textual.widgets import Static

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import SanctumSnapshot


def _usage_style(percent: float) -> str:
    """Get color style based on usage percentage.

    <75%: green (healthy)
    75-90%: yellow (warning)
    >90%: red (critical)
    """
    if percent > 90:
        return "red"
    elif percent > 75:
        return "yellow"
    return "green"


def _make_bar(used: float, total: float, width: int = 15) -> Text:
    """Create a compact usage bar with percentage.

    Format: [=====     ] 45%
    Color based on usage threshold.
    """
    if total <= 0:
        return Text("─" * width, style="dim")

    percent = (used / total) * 100
    filled = int((percent / 100) * width)
    empty = width - filled
    style = _usage_style(percent)

    bar = Text()
    bar.append("[", style="dim")
    bar.append("=" * filled, style=style)
    bar.append(" " * empty, style="dim")
    bar.append("]", style="dim")
    bar.append(f" {percent:3.0f}%", style=style)
    return bar


class SystemResources(Static):
    """System resource monitor with visual bars.

    Shows:
    - CPU usage bar
    - RAM usage bar with GB values
    - Per-GPU memory bars with GB values
    - GPU utilization bars (if available)
    - Throughput metrics (epochs/sec, batches/hr)

    All metrics use color-coded thresholds:
    - Green: <75% (healthy)
    - Yellow: 75-90% (warning)
    - Red: >90% (critical)
    """

    def __init__(self, **kwargs) -> None:
        """Initialize SystemResources widget."""
        super().__init__(**kwargs)
        self._snapshot: SanctumSnapshot | None = None

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        """Update widget with new snapshot data."""
        self._snapshot = snapshot
        self.refresh()

    def render(self) -> Group:
        """Render system resources with visual bars."""
        if self._snapshot is None:
            return Group(Text("Waiting for data...", style="dim"))

        vitals = self._snapshot.vitals
        lines = []

        # CPU
        cpu_pct = vitals.cpu_percent
        if cpu_pct > 0:
            cpu_bar = _make_bar(cpu_pct, 100, width=12)
            cpu_line = Text()
            cpu_line.append("CPU  ", style="dim")
            cpu_line.append_text(cpu_bar)
            lines.append(cpu_line)

        # RAM
        if vitals.ram_total_gb > 0:
            ram_bar = _make_bar(vitals.ram_used_gb, vitals.ram_total_gb, width=12)
            ram_line = Text()
            ram_line.append("RAM  ", style="dim")
            ram_line.append_text(ram_bar)
            ram_line.append(f" {vitals.ram_used_gb:.0f}/{vitals.ram_total_gb:.0f}G", style="dim")
            lines.append(ram_line)

        # GPU(s)
        if vitals.gpu_stats:
            for dev_id, stats in sorted(vitals.gpu_stats.items()):
                if stats.memory_total_gb > 0:
                    # Memory bar
                    gpu_bar = _make_bar(stats.memory_used_gb, stats.memory_total_gb, width=12)
                    # Label (GPU0, GPU1, etc. or just GPU if single)
                    label = dev_id.replace("cuda:", "GPU") if len(vitals.gpu_stats) > 1 else "GPU "

                    gpu_line = Text()
                    gpu_line.append(f"{label} ", style="dim")
                    gpu_line.append_text(gpu_bar)
                    gpu_line.append(f" {stats.memory_used_gb:.1f}/{stats.memory_total_gb:.0f}G", style="dim")
                    lines.append(gpu_line)

                    # Utilization if available (requires pynvml)
                    if stats.utilization > 0:
                        util_bar = _make_bar(stats.utilization, 100, width=12)
                        util_line = Text()
                        util_line.append("  util", style="dim")
                        util_line.append_text(util_bar)
                        lines.append(util_line)
        elif vitals.gpu_memory_total_gb > 0:
            # Fallback single GPU
            gpu_bar = _make_bar(vitals.gpu_memory_used_gb, vitals.gpu_memory_total_gb, width=12)
            gpu_line = Text()
            gpu_line.append("GPU  ", style="dim")
            gpu_line.append_text(gpu_bar)
            gpu_line.append(f" {vitals.gpu_memory_used_gb:.1f}/{vitals.gpu_memory_total_gb:.0f}G", style="dim")
            lines.append(gpu_line)
        else:
            lines.append(Text("GPU  [no CUDA]", style="dim"))

        # Separator
        lines.append(Text(""))

        # Throughput
        throughput = Text()
        throughput.append("Throughput: ", style="dim")
        if vitals.epochs_per_second > 0:
            throughput.append(f"{vitals.epochs_per_second:.1f}", style="cyan")
            throughput.append(" ep/s", style="dim")
        else:
            throughput.append("--", style="dim")
        lines.append(throughput)

        return Group(*lines)
