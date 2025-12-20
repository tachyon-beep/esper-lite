"""Scoreboard widget - Best Runs leaderboard.

Port of tui.py _render_scoreboard() (lines 1083-1190).
Shows stats header and top 10 environments by best accuracy.

Reference: src/esper/karn/tui.py lines 1083-1190 (_render_scoreboard method)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Group
from rich.table import Table
from rich.text import Text
from textual.widgets import Static

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import SanctumSnapshot, SeedState


class Scoreboard(Static):
    """Best Runs scoreboard widget.

    Shows:
    1. Stats header: global best, mean best, fossilized/culled counts
    2. Leaderboard: top 10 completed env runs with peak accuracy + seed composition
    """

    def __init__(self, **kwargs) -> None:
        """Initialize Scoreboard widget."""
        super().__init__(**kwargs)
        self._snapshot: SanctumSnapshot | None = None
        self.border_title = "BEST RUNS"  # Top-left title (CSS provides border)

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        """Update widget with new snapshot data."""
        self._snapshot = snapshot
        self.refresh()

    def render(self):
        """Render the scoreboard content (border provided by CSS)."""
        if self._snapshot is None:
            return Text("No data", style="dim")

        best_runs = list(self._snapshot.best_runs)

        # === AGGREGATE STATS ===
        stats_table = Table(show_header=False, box=None, padding=(0, 1), expand=True)
        stats_table.add_column("Label", style="dim", width=12)
        stats_table.add_column("Value", justify="right", width=8)
        stats_table.add_column("Label2", style="dim", width=12)
        stats_table.add_column("Value2", justify="right", width=8)

        # Compute aggregates (prefer best_runs so values don't disappear after batch reset)
        all_envs = list(self._snapshot.envs.values())
        total_fossilized = sum(e.fossilized_count for e in all_envs)
        total_culled = sum(e.culled_count for e in all_envs)

        if best_runs:
            global_best = max(r.peak_accuracy for r in best_runs)
            mean_best = sum(r.peak_accuracy for r in best_runs) / len(best_runs)
        else:
            best_accs = [e.best_accuracy for e in all_envs if e.best_accuracy > 0]
            mean_best = sum(best_accs) / len(best_accs) if best_accs else 0.0
            global_best = max(best_accs) if best_accs else 0.0

        stats_table.add_row(
            "Global Best:",
            f"[bold green]{global_best:.1f}%[/]",
            "Mean Best:",
            f"{mean_best:.1f}%",
        )
        stats_table.add_row(
            "Fossilized:",
            f"[green]{total_fossilized}[/]",
            "Culled:",
            f"[red]{total_culled}[/]",
        )

        # === LEADERBOARD (TOP 10) ===
        lb_table = Table(show_header=True, box=None, padding=(0, 1), expand=True)
        lb_table.add_column("#", style="dim", width=2)
        lb_table.add_column("Ep", justify="right", width=4)
        lb_table.add_column("Acc", justify="right", width=6)
        lb_table.add_column("Growth", justify="right", width=6)
        lb_table.add_column("Seeds", justify="left", width=20)

        if not best_runs:
            lb_table.add_row("[dim]No completed runs yet[/]", "", "", "", "")
        else:
            best_runs = sorted(best_runs, key=lambda r: r.peak_accuracy, reverse=True)[:10]
            for i, record in enumerate(best_runs, start=1):
                lb_table.add_row(
                    str(i),
                    str(record.episode + 1),
                    f"[bold green]{record.peak_accuracy:.1f}[/]",
                    self._format_growth_ratio(record.growth_ratio),
                    self._format_seeds(record.seeds),
                )

        # Combine stats and leaderboard (no Panel wrapper - CSS provides border)
        return Group(
            stats_table,
            Text("─" * 40, style="dim"),
            lb_table,
        )

    def _format_seeds(self, seeds: dict[str, "SeedState"]) -> str:
        """Format seed composition at peak accuracy (simple, top-10 friendly)."""
        contributing = [
            seed
            for seed in seeds.values()
            if seed.blueprint_id and seed.stage in {"FOSSILIZED", "BLENDING", "PROBATIONARY"}
        ]
        if not contributing:
            return "─"

        stage_order = {"FOSSILIZED": 0, "BLENDING": 1, "PROBATIONARY": 2}
        stage_colors = {"FOSSILIZED": "green", "BLENDING": "magenta", "PROBATIONARY": "yellow"}
        contributing.sort(key=lambda s: (stage_order.get(s.stage, 9), s.blueprint_id or ""))

        parts = []
        for seed in contributing[:3]:
            bp = (seed.blueprint_id or "?")[:6]
            color = stage_colors.get(seed.stage, "dim")
            parts.append(f"[{color}]{bp}[/{color}]")

        if len(contributing) > 3:
            parts.append(f"[dim]+{len(contributing) - 3}[/]")

        return " ".join(parts)

    def _format_growth_ratio(self, growth_ratio: float) -> str:
        """Format growth ratio: (base params + seed params) / base params."""
        ratio = float(growth_ratio) if growth_ratio else 1.0
        if ratio <= 1.0:
            return "[dim]1.00x[/]"
        if ratio < 1.1:
            return f"[cyan]{ratio:.2f}x[/]"
        return f"[bold cyan]{ratio:.2f}x[/]"
