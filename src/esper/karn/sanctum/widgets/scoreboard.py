"""Scoreboard widget - Best Runs leaderboard.

Port of tui.py _render_scoreboard() (lines 1083-1190).
Shows stats header and top 10 environments by best accuracy.

Reference: src/esper/karn/tui.py lines 1083-1190 (_render_scoreboard method)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.widgets import Static

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import SanctumSnapshot, BestRunRecord


# A/B test cohort styling - colored pips for reward modes
_AB_STYLES: dict[str, tuple[str, str]] = {
    "shaped": ("●", "bright_blue"),      # Blue pip for shaped
    "simplified": ("●", "bright_yellow"), # Yellow pip for simplified
    "sparse": ("●", "bright_cyan"),       # Cyan pip for sparse
}


class Scoreboard(Static):
    """Best Runs scoreboard widget.

    Shows:
    1. Stats header: global best, mean best, fossilized/culled counts
    2. Leaderboard table: top 10 envs by best accuracy
       - Rank with medals (🥇🥈🥉) for top 3
       - Episode number
       - High (best accuracy)
       - Cur (current accuracy with delta-based styling)
       - Seeds at best (blueprints or count)
       - A/B cohort pip for visual continuity
    """

    def __init__(self, **kwargs) -> None:
        """Initialize Scoreboard widget."""
        super().__init__(**kwargs)
        self._snapshot: SanctumSnapshot | None = None

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        """Update widget with new snapshot data."""
        self._snapshot = snapshot
        self.refresh()

    def render(self) -> Panel:
        """Render the scoreboard panel."""
        if self._snapshot is None:
            return Panel("No data", title="BEST RUNS", border_style="cyan")

        # === AGGREGATE STATS ===
        stats_table = Table(show_header=False, box=None, padding=(0, 1), expand=True)
        stats_table.add_column("Label", style="dim", width=12)
        stats_table.add_column("Value", justify="right", width=8)
        stats_table.add_column("Label2", style="dim", width=12)
        stats_table.add_column("Value2", justify="right", width=8)

        # Compute aggregates
        all_envs = list(self._snapshot.envs.values())
        total_fossilized = sum(e.fossilized_count for e in all_envs)
        total_culled = sum(e.culled_count for e in all_envs)
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

        # === LEADERBOARD ===
        lb_table = Table(show_header=True, box=None, padding=(0, 1), expand=True)
        lb_table.add_column("#", style="dim", width=3)
        lb_table.add_column("@Ep", justify="right", width=4)
        lb_table.add_column("Peak", justify="right", width=6)
        lb_table.add_column("End", justify="right", width=6)
        lb_table.add_column("Seeds", justify="left", width=20)

        medals = ["🥇", "🥈", "🥉"]

        # Get best runs (if available in snapshot)
        best_runs = getattr(self._snapshot, 'best_runs', [])

        if not best_runs:
            lb_table.add_row("-", "-", "-", "-", "-")
        else:
            for i, record in enumerate(best_runs):
                # Rank with medal or number
                rank = medals[i] if i < 3 else f"{i + 1}"

                # A/B cohort pip (from env's reward_mode)
                env = self._snapshot.envs.get(record.env_id)
                if env and env.reward_mode and env.reward_mode in _AB_STYLES:
                    pip, color = _AB_STYLES[env.reward_mode]
                    rank = f"[{color}]{pip}[/{color}]{rank}"

                # Current accuracy styling based on delta from best
                delta = record.final_accuracy - record.peak_accuracy
                if delta >= -0.5:
                    cur_style = "green"
                elif delta >= -2.0:
                    cur_style = "yellow"
                else:
                    cur_style = "dim"

                # Format seeds at best
                seeds_str = self._format_seeds(record.seeds)

                lb_table.add_row(
                    rank,
                    str(record.absolute_episode),
                    f"[bold green]{record.peak_accuracy:.1f}[/]",
                    f"[{cur_style}]{record.final_accuracy:.1f}[/]",
                    seeds_str,
                )

        # Combine stats and leaderboard
        content = Group(
            stats_table,
            Text("─" * 40, style="dim"),
            lb_table,
        )

        return Panel(
            content,
            title="[bold]BEST RUNS[/bold]",
            border_style="cyan",
        )

    def _format_seeds(self, seeds: dict[str, any]) -> str:
        """Format seeds at best accuracy.

        ≤3 seeds: Show blueprint names (first 6 chars each) with stage-based colors
        >3 seeds: Show permanent+provisional count format

        Colors:
        - FOSSILIZED → green
        - PROBATIONARY → yellow
        - BLENDING → magenta
        - Other → dim
        """
        if not seeds:
            return "─"

        n_seeds = len(seeds)

        if n_seeds <= 3:
            # Show individual blueprints with stage-based colors
            seed_parts = []
            for seed in seeds.values():
                bp = seed.blueprint_id[:6] if seed.blueprint_id else "?"
                if seed.stage == "FOSSILIZED":
                    seed_parts.append(f"[green]{bp}[/]")
                elif seed.stage == "PROBATIONARY":
                    seed_parts.append(f"[yellow]{bp}[/]")
                elif seed.stage == "BLENDING":
                    seed_parts.append(f"[magenta]{bp}[/]")
                else:
                    seed_parts.append(f"[dim]{bp}[/]")
            return " ".join(seed_parts)
        else:
            # Count by stage for many seeds
            permanent = sum(1 for s in seeds.values() if s.stage == "FOSSILIZED")
            provisional = len(seeds) - permanent
            if permanent and provisional:
                return f"[green]{permanent}[/]+[yellow]{provisional}[/]"
            elif permanent:
                return f"[green]{permanent} seeds[/]"
            else:
                return f"[yellow]{provisional} prov[/]"
