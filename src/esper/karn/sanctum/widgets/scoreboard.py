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
        self.border_title = "BEST RUNS"  # Top-left title (CSS provides border)

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        """Update widget with new snapshot data."""
        self._snapshot = snapshot
        self.refresh()

    def render(self):
        """Render the scoreboard content (border provided by CSS)."""
        if self._snapshot is None:
            return Text("No data", style="dim")

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
        lb_table.add_column("@Step", justify="right", width=5)  # Epoch within episode
        lb_table.add_column("Peak", justify="right", width=6)
        lb_table.add_column("End", justify="right", width=5)
        lb_table.add_column("Size", justify="right", width=5)  # Growth ratio
        lb_table.add_column("Seeds", justify="left", width=18)

        medals = ["🥇", "🥈", "🥉"]

        # Get best runs (if available in snapshot)
        best_runs = getattr(self._snapshot, 'best_runs', [])

        if not best_runs:
            # Debug: Show why best_runs is empty with diagnostic info
            all_envs = list(self._snapshot.envs.values())
            envs_with_best = [e for e in all_envs if e.best_accuracy > 0]
            batch_count = self._snapshot.current_batch
            ep_count = self._snapshot.current_episode
            total_events = self._snapshot.total_events_received

            if total_events == 0:
                lb_table.add_row("[dim]no events received yet[/]", "", "", "", "", "")
            elif batch_count == 0 and ep_count == 0:
                lb_table.add_row("[dim]waiting for first batch...[/]", "", "", "", "", "")
            elif not envs_with_best:
                lb_table.add_row(f"[dim]0/{len(all_envs)} envs have best[/]", "", "", "", "", "")
            else:
                # best_accuracy exists but best_runs empty - detailed diagnostic
                # Show which episode envs think they improved in vs current
                sample_env = envs_with_best[0] if envs_with_best else None
                if sample_env:
                    lb_table.add_row(
                        f"[yellow]BUG: {len(envs_with_best)} envs[/]",
                        f"[dim]cur_ep={ep_count}[/]",
                        f"[dim]env0.best_ep={sample_env.best_accuracy_episode}[/]",
                        "",
                        "",
                        ""
                    )
                    # Show if there's an episode mismatch
                    if sample_env.best_accuracy_episode != ep_count:
                        lb_table.add_row(
                            "[red]MISMATCH[/]",
                            "",
                            f"[dim]{sample_env.best_accuracy_episode}!={ep_count}[/]",
                            "",
                            "",
                            ""
                        )
                else:
                    lb_table.add_row("[dim]diagnostic failed[/]", "", "", "", "", "")
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

                # Format growth ratio (1.0 = no growth, >1.0 = model grew)
                growth = record.growth_ratio
                if growth <= 1.0:
                    growth_str = "[dim]1.0x[/]"
                elif growth < 1.1:
                    growth_str = f"[cyan]{growth:.2f}x[/]"
                else:
                    growth_str = f"[bold cyan]{growth:.2f}x[/]"

                lb_table.add_row(
                    rank,
                    str(record.epoch),
                    f"[bold green]{record.peak_accuracy:.1f}[/]",
                    f"[{cur_style}]{record.final_accuracy:.1f}[/]",
                    growth_str,
                    seeds_str,
                )

        # Combine stats and leaderboard (no Panel wrapper - CSS provides border)
        return Group(
            stats_table,
            Text("─" * 40, style="dim"),
            lb_table,
        )

    def _format_seeds(self, seeds: dict[str, any]) -> str:
        """Format seeds at best accuracy with blend tempo.

        ≤3 seeds: Show blueprint names (first 5 chars) + tempo indicator
        >3 seeds: Show permanent+provisional count format

        Tempo indicators: F=fast(≤3), M=medium(≤5), S=slow(>5)
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
            # Show individual blueprints with stage-based colors + tempo
            seed_parts = []
            for seed in seeds.values():
                bp = seed.blueprint_id[:6] if seed.blueprint_id else "?"
                tempo = seed.blend_tempo_epochs or 5  # Default to medium if None
                tempo_char = "F" if tempo <= 3 else ("M" if tempo <= 5 else "S")
                if seed.stage == "FOSSILIZED":
                    seed_parts.append(f"[green]{bp}:{tempo_char}[/]")
                elif seed.stage == "PROBATIONARY":
                    seed_parts.append(f"[yellow]{bp}:{tempo_char}[/]")
                elif seed.stage == "BLENDING":
                    seed_parts.append(f"[magenta]{bp}:{tempo_char}[/]")
                else:
                    seed_parts.append(f"[dim]{bp}:{tempo_char}[/]")
            return " ".join(seed_parts)
        else:
            # Count by stage for many seeds, show tempo distribution
            permanent = sum(1 for s in seeds.values() if s.stage == "FOSSILIZED")
            provisional = len(seeds) - permanent
            # Count tempos
            fast = sum(1 for s in seeds.values() if s.blend_tempo_epochs <= 3)
            slow = sum(1 for s in seeds.values() if s.blend_tempo_epochs > 5)
            tempo_str = ""
            if fast or slow:
                tempo_str = f" [dim]({fast}F/{slow}S)[/]"
            if permanent and provisional:
                return f"[green]{permanent}[/]+[yellow]{provisional}[/]{tempo_str}"
            elif permanent:
                return f"[green]{permanent} seeds[/]{tempo_str}"
            else:
                return f"[yellow]{provisional} prov[/]{tempo_str}"
