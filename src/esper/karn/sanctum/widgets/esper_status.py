"""EsperStatus widget - System vitals and performance metrics.

Port of tui.py _render_esper_status() (lines 1648-1748).
Shows Esper system status including seed stages, performance, and resources.

Reference: src/esper/karn/tui.py lines 1648-1748 (_render_esper_status method)
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.widgets import Static

from esper.leyline import STAGE_ABBREVIATIONS, STAGE_COLORS

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import SanctumSnapshot


class EsperStatus(Static):
    """EsperStatus widget - System vitals and performance.

    Shows:
    1. Seed stage counts (aggregate across all envs) - Train, Blend, Prob, Foss, etc.
    2. Host params (format: M/K/raw)
    3. Throughput (epochs/sec, batches/hr)
    4. Runtime (Xh Ym Zs)
    5. GPU stats (memory used/total, utilization) - multi-GPU support
    6. RAM usage
    7. CPU percentage (FIX: was collected but never displayed)

    Color thresholds:
    - GPU/RAM memory: green <75%, yellow 75-90%, red >90%
    - GPU utilization: green <80%, yellow 80-95%, red >95%
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize EsperStatus widget."""
        super().__init__(**kwargs)
        self._snapshot: SanctumSnapshot | None = None

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        """Update widget with new snapshot data."""
        self._snapshot = snapshot
        self.refresh()

    def render(self) -> Panel:
        """Render the Esper status panel."""
        if self._snapshot is None:
            return Panel("No data", title="ESPER STATUS", border_style="cyan")

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Metric", style="dim")
        table.add_column("Value", justify="right")

        # Seed stage counts (aggregate across all envs)
        stage_counts: dict[str, int] = {}
        for env in self._snapshot.envs.values():
            for seed in env.seeds.values():
                if seed.stage != "DORMANT":
                    stage_counts[seed.stage] = stage_counts.get(seed.stage, 0) + 1

        if stage_counts:
            for stage, count in sorted(stage_counts.items()):
                short = STAGE_ABBREVIATIONS.get(stage, stage[:4])
                style = STAGE_COLORS.get(stage, "dim")
                table.add_row(f"{short}:", Text(str(count), style=style))
            table.add_row("", "")

        # Host network params (always visible)
        vitals = self._snapshot.vitals
        if vitals.host_params > 0:
            if vitals.host_params >= 1_000_000:
                params_str = f"{vitals.host_params / 1_000_000:.1f}M"
            elif vitals.host_params >= 1_000:
                params_str = f"{vitals.host_params / 1_000:.0f}K"
            else:
                params_str = str(vitals.host_params)
            table.add_row("Host Params:", params_str)
        else:
            table.add_row("Host Params:", Text("--", style="dim"))
        table.add_row("", "")

        # Throughput
        table.add_row("Epochs/sec:", f"{vitals.epochs_per_second:.2f}")
        table.add_row("Batches/hr:", f"{vitals.batches_per_hour:.0f}")

        # Runtime
        if self._snapshot.runtime_seconds > 0:
            elapsed_seconds = int(self._snapshot.runtime_seconds)
            hours, remainder = divmod(elapsed_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            table.add_row("Runtime:", f"{hours}h {minutes}m {seconds}s")
        else:
            table.add_row("Runtime:", "-")

        table.add_row("", "")

        # GPU stats - show all configured GPUs (memory + utilization)
        if vitals.gpu_stats:
            for dev_id, stats in sorted(vitals.gpu_stats.items()):
                if stats.memory_total_gb > 0:
                    gpu_pct = (stats.memory_used_gb / stats.memory_total_gb) * 100
                    mem_style = "red" if gpu_pct > 90 else "yellow" if gpu_pct > 75 else "green"
                    label = f"GPU{dev_id}:" if len(vitals.gpu_stats) > 1 else "GPU:"
                    # Show memory usage
                    table.add_row(
                        label,
                        Text(f"{stats.memory_used_gb:.1f}/{stats.memory_total_gb:.1f}GB", style=mem_style)
                    )
                    # Show utilization if available (from pynvml)
                    if stats.utilization > 0:
                        util_style = "red" if stats.utilization > 95 else "yellow" if stats.utilization > 80 else "green"
                        util_label = "  util:" if len(vitals.gpu_stats) > 1 else "GPU util:"
                        table.add_row(
                            util_label,
                            Text(f"{stats.utilization:.0f}%", style=util_style)
                        )
        elif vitals.gpu_memory_total_gb > 0:
            # Fallback to legacy single-GPU fields
            gpu_pct = (vitals.gpu_memory_used_gb / vitals.gpu_memory_total_gb) * 100
            mem_style = "red" if gpu_pct > 90 else "yellow" if gpu_pct > 75 else "green"
            table.add_row(
                "GPU:",
                Text(f"{vitals.gpu_memory_used_gb:.1f}/{vitals.gpu_memory_total_gb:.1f}GB", style=mem_style)
            )
            if vitals.gpu_utilization > 0:
                util_style = "red" if vitals.gpu_utilization > 95 else "yellow" if vitals.gpu_utilization > 80 else "green"
                table.add_row(
                    "GPU util:",
                    Text(f"{vitals.gpu_utilization:.0f}%", style=util_style)
                )
        else:
            table.add_row("GPU:", "-")

        # RAM (always visible)
        if vitals.ram_total_gb is not None and vitals.ram_used_gb is not None and vitals.ram_total_gb > 0:
            ram_pct = (vitals.ram_used_gb / vitals.ram_total_gb) * 100
            ram_style = "red" if ram_pct > 90 else "yellow" if ram_pct > 75 else "dim"
            table.add_row(
                "RAM:",
                Text(f"{vitals.ram_used_gb:.1f}/{vitals.ram_total_gb:.0f}GB", style=ram_style)
            )
        else:
            table.add_row("RAM:", Text("--", style="dim"))

        # CPU percentage (always visible)
        if vitals.cpu_percent is not None and vitals.cpu_percent > 0:
            table.add_row("CPU:", f"{vitals.cpu_percent:.1f}%")
        else:
            table.add_row("CPU:", Text("--", style="dim"))

        return Panel(table, title="[bold]ESPER STATUS[/bold]", border_style="cyan")
