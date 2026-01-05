"""Scoreboard widget - Best Runs and Worst Trajectory panels.

Two-panel layout (equal size):
1. Best Runs (1fr): Stats header + top 5 environments by peak accuracy
2. Worst Trajectory (1fr): Bottom 5 runs with most regression from peak

Columns:
- # (rank): Plain numeric rank (1-5)
- Ep: Episode number
- @: Epoch when peak was achieved (early peak = more potential)
- Peak: Best accuracy achieved
- Traj: Trajectory arrow showing peak→final (↗ climbing, ─→ held, ↘ regressed)
- Growth: Parameter growth ratio
- Seeds: Seed status counts as "blending/holding/fossilized" (e.g., "1/0/2")
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator

from textual.containers import Vertical
from textual.css.query import NoMatches
from textual.message import Message
from textual.widgets import DataTable, Static

from esper.leyline import STAGE_COLORS

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import BestRunRecord, SanctumSnapshot, SeedState


class Scoreboard(Static):
    """Best Runs scoreboard widget (display only).

    Shows:
    1. Best Runs panel: stats header + top 5 runs by peak accuracy
    2. Worst Trajectory panel: bottom 5 runs with most regression
    """

    class BestRunSelected(Message):
        """Posted when a best run row is selected for detail view."""

        def __init__(self, record: "BestRunRecord") -> None:
            super().__init__()
            self.record = record

    DEFAULT_CSS = """
    Scoreboard {
        layout: vertical;
    }

    #best-runs-panel {
        height: 6fr;  /* Larger - has stats header + 4 data rows */
        border: round $surface-lighten-2;
        border-title-color: cyan;
        padding: 0 1;
    }

    #worst-runs-panel {
        height: 5fr;  /* Smaller - just 4 data rows, gives 1 row to bottom */
        border: round $surface-lighten-2;
        border-title-color: red;
        padding: 0 1;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize Scoreboard widget."""
        super().__init__(**kwargs)
        self._snapshot: SanctumSnapshot | None = None
        self._displayed_records: list["BestRunRecord"] = []
        self._bottom_records: list["BestRunRecord"] = []
        self.table: DataTable[Any] = DataTable[Any](zebra_stripes=True, cursor_type="row")
        self.bottom_table: DataTable[Any] = DataTable[Any](zebra_stripes=True, cursor_type="row")

    def compose(self) -> Iterator[Vertical | Static | DataTable[Any]]:
        """Compose the widget.

        Layout (equal heights):
        - Best Runs panel (1fr): stats + top 5 runs
        - Worst Trajectory panel (1fr): 5 worst regression runs
        """
        with Vertical(id="best-runs-panel") as panel:
            panel.border_title = "BEST RUNS"
            yield Static(id="scoreboard-stats")
            yield self.table

        with Vertical(id="worst-runs-panel") as panel:
            panel.border_title = "WORST TRAJECTORY"
            yield self.bottom_table

    def on_mount(self) -> None:
        """Setup table columns on mount."""
        self._setup_columns()
        self._setup_bottom_columns()

    def _setup_columns(self) -> None:
        """Setup leaderboard table columns.

        Layout:
        │ #  │ Ep │ @ │ Peak │  Traj  │Growth│ Seeds  │
        │ 1  │ 47 │12 │85.5% │ ─→85.2 │1.03x │ 1/0/2  │
        """
        self.table.clear(columns=True)
        self.table.add_column("#", key="rank", width=4)      # Rank number (1-5)
        self.table.add_column("Ep", key="episode", width=3)  # Episode number
        self.table.add_column("@", key="epoch", width=2)     # Epoch of peak
        self.table.add_column("Peak", key="peak", width=6)   # Peak accuracy
        self.table.add_column("Traj", key="traj", width=7)   # Trajectory arrow + final
        self.table.add_column("Grw", key="growth", width=5)  # Growth ratio (shortened)
        self.table.add_column("Seeds", key="seeds", width=7)  # Seed status counts (B/H/F)

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        """Update widget with new snapshot data."""
        self._snapshot = snapshot
        self._refresh_stats()
        self._refresh_table()
        self._refresh_bottom_table()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection from either scoreboard table.

        Emits BestRunSelected for both the BEST RUNS and WORST TRAJECTORY panels.
        Stops propagation so SanctumApp's global DataTable handler doesn't also fire.
        """
        if event.data_table is not self.table and event.data_table is not self.bottom_table:
            return

        event.stop()

        record = self._get_record_for_cursor(event.data_table, event.cursor_row)
        if record is None:
            return
        self.post_message(self.BestRunSelected(record))

    def _get_record_for_cursor(
        self, table: DataTable[Any], cursor_row: int
    ) -> "BestRunRecord | None":
        """Return the BestRunRecord for the given table row, if available."""
        if cursor_row < 0:
            return None
        records = self._displayed_records if table is self.table else self._bottom_records
        if cursor_row >= len(records):
            return None
        return records[cursor_row]

    def _refresh_stats(self) -> None:
        """Refresh the stats header."""
        if self._snapshot is None:
            return

        try:
            stats_widget = self.query_one("#scoreboard-stats", Static)
        except NoMatches:
            # UI-01 fix: Narrow to NoMatches - only expected exception from query_one
            return

        best_runs = list(self._snapshot.best_runs)
        total_fossilized = self._snapshot.cumulative_fossilized
        total_pruned = self._snapshot.cumulative_pruned

        if best_runs:
            global_best = max(r.peak_accuracy for r in best_runs)
            mean_best = sum(r.peak_accuracy for r in best_runs) / len(best_runs)
        else:
            all_envs = list(self._snapshot.envs.values())
            best_accs = [e.best_accuracy for e in all_envs if e.best_accuracy > 0]
            mean_best = sum(best_accs) / len(best_accs) if best_accs else 0.0
            global_best = max(best_accs) if best_accs else 0.0

        # Compact two-line stats
        stats_text = (
            f"[dim]Best:[/dim] [bold green]{global_best:.1f}%[/bold green]  "
            f"[dim]Mean:[/dim] {mean_best:.1f}%  "
            f"[dim]Foss:[/dim] [green]{total_fossilized}[/green]  "
            f"[dim]Prune:[/dim] [red]{total_pruned}[/red]"
        )

        tail = list(self._snapshot.mean_accuracy_history)[-10:]
        if tail:
            tail_mean = sum(tail) / len(tail)
            stats_text += f"\n[dim]Tail10:[/dim] {tail_mean:.1f}%"
        else:
            stats_text += "\n[dim]Tail10:[/dim] --"

        stats_widget.update(stats_text)

        # Update panel title
        try:
            panel = self.query_one("#best-runs-panel", Vertical)
            panel.border_title = "BEST RUNS"
        except NoMatches:
            # UI-02 fix: Narrow to NoMatches - only expected exception from query_one
            pass

    def _refresh_table(self) -> None:
        """Refresh the leaderboard table.

        Preserves cursor and scroll position across refresh cycles.
        """
        if self._snapshot is None:
            return

        # Save cursor and scroll position
        saved_cursor = self.table.cursor_row
        saved_scroll_x = self.table.scroll_target_x
        saved_scroll_y = self.table.scroll_target_y

        self.table.clear()

        best_runs = list(self._snapshot.best_runs)
        if not best_runs:
            self._displayed_records = []
            self.table.add_row("[dim]No runs yet[/dim]", "", "", "", "", "", "")
            self.table.refresh()
            return

        # Sort by peak accuracy, display top 5 best runs
        best_runs = sorted(best_runs, key=lambda r: r.peak_accuracy, reverse=True)[:5]
        self._displayed_records = best_runs

        for i, record in enumerate(best_runs, start=1):
            # Use record_id if available, otherwise fallback to index-based key
            row_key = record.record_id if record.record_id else f"row_{i}"
            self.table.add_row(
                str(i),
                str(record.episode + 1),
                self._format_epoch(record),
                f"[bold green]{record.peak_accuracy:.1f}[/bold green]",
                self._format_trajectory(record),
                self._format_growth_ratio(record.growth_ratio),
                self._format_seeds(record.seeds),
                key=row_key,
            )

        # Restore cursor position
        if self.table.row_count > 0 and saved_cursor is not None:
            target = min(saved_cursor, self.table.row_count - 1)
            if target != self.table.cursor_row:
                self.table.move_cursor(row=target)

        # Restore scroll position
        if saved_scroll_x or saved_scroll_y:
            self.table.scroll_to(x=saved_scroll_x, y=saved_scroll_y, animate=False)

        self.table.refresh()

    def _setup_bottom_columns(self) -> None:
        """Setup bottom 5 table columns (same structure as best runs panel).

        Same columns as best runs panel but for runs with worst trajectory.
        """
        self.bottom_table.clear(columns=True)
        self.bottom_table.add_column("#", key="rank", width=4)
        self.bottom_table.add_column("Ep", key="episode", width=3)
        self.bottom_table.add_column("@", key="epoch", width=2)
        self.bottom_table.add_column("Peak", key="peak", width=6)
        self.bottom_table.add_column("Traj", key="traj", width=7)
        self.bottom_table.add_column("Grw", key="growth", width=5)
        self.bottom_table.add_column("Seeds", key="seeds", width=7)  # Seed status counts (B/H/F)

    def _refresh_bottom_table(self) -> None:
        """Refresh the bottom 5 table showing worst trajectory runs.

        Sorts by trajectory (final - peak) ascending to find runs
        that regressed the most from their peak.
        """
        if self._snapshot is None:
            return

        saved_cursor = self.bottom_table.cursor_row
        saved_scroll_x = self.bottom_table.scroll_target_x
        saved_scroll_y = self.bottom_table.scroll_target_y

        self.bottom_table.clear()

        best_runs = list(self._snapshot.best_runs)
        if not best_runs:
            self._bottom_records = []
            self.bottom_table.add_row("[dim]No data[/dim]", "", "", "", "", "", "")
            self.bottom_table.refresh()
            return

        # Sort by trajectory delta (final - peak), ascending = worst regression first
        # Exclude runs with positive trajectory (still climbing)
        runs_with_regression = [
            r for r in best_runs
            if (r.final_accuracy - r.peak_accuracy) < -0.5  # At least 0.5% regression
        ]

        if not runs_with_regression:
            self._bottom_records = []
            self.bottom_table.add_row("[dim]No regressions[/dim]", "", "", "", "", "", "")
            self.bottom_table.refresh()
            return

        # Sort by worst trajectory (most negative delta first)
        runs_with_regression.sort(key=lambda r: r.final_accuracy - r.peak_accuracy)
        bottom_5 = runs_with_regression[:5]
        self._bottom_records = bottom_5

        for i, record in enumerate(bottom_5, start=1):
            row_key = f"bottom_{record.record_id}" if record.record_id else f"bottom_row_{i}"
            self.bottom_table.add_row(
                str(i),
                str(record.episode + 1),
                self._format_epoch(record),
                f"[yellow]{record.peak_accuracy:.1f}[/yellow]",
                self._format_trajectory(record),
                self._format_growth_ratio(record.growth_ratio),
                self._format_seeds(record.seeds),
                key=row_key,
            )

        if self.bottom_table.row_count > 0 and saved_cursor is not None:
            target = min(saved_cursor, self.bottom_table.row_count - 1)
            if target != self.bottom_table.cursor_row:
                self.bottom_table.move_cursor(row=target)

        if saved_scroll_x or saved_scroll_y:
            self.bottom_table.scroll_to(x=saved_scroll_x, y=saved_scroll_y, animate=False)

        self.bottom_table.refresh()

    def _format_epoch(self, record: "BestRunRecord") -> str:
        """Format epoch when peak was achieved.

        Color coding:
        - Green: Early peak (epoch < 25) - lots of room to grow
        - White: Mid peak (25-50)
        - Yellow: Late peak (50-65)
        - Red: Very late peak (65+) - near max epochs
        """
        epoch = record.epoch
        if epoch < 25:
            return f"[green]{epoch}[/green]"
        elif epoch < 50:
            return str(epoch)
        elif epoch < 65:
            return f"[yellow]{epoch}[/yellow]"
        else:
            return f"[red]{epoch}[/red]"

    def _format_trajectory(self, record: "BestRunRecord") -> str:
        """Format trajectory showing peak→final relationship.

        Arrows:
        - ↗ (green): Still climbing - final > peak (rare but possible with smoothing)
        - ─→ (dim): Held steady - within 1% of peak
        - ↘ (yellow/red): Regressed - final < peak

        Color intensity based on regression severity:
        - <2% drop: dim arrow
        - 2-5% drop: yellow
        - >5% drop: red
        """
        peak = record.peak_accuracy
        final = record.final_accuracy
        delta = final - peak

        if delta > 0.5:
            # Still climbing (final > peak)
            return f"[green]↗{final:.1f}[/green]"
        elif delta >= -1.0:
            # Held steady (within 1%)
            return f"[dim]─→{final:.1f}[/dim]"
        elif delta >= -2.0:
            # Small regression
            return f"[dim]↘{final:.1f}[/dim]"
        elif delta >= -5.0:
            # Moderate regression
            return f"[yellow]↘{final:.1f}[/yellow]"
        else:
            # Severe regression (>5% drop)
            return f"[red]↘{final:.1f}[/red]"

    def _format_seeds(self, seeds: dict[str, "SeedState"]) -> str:
        """Format seed composition as status counts: blending/holding/fossilized.

        Shows counts in format "1/0/2" with each number colored by stage.
        Example: [cyan]1[/cyan]/[yellow]0[/yellow]/[green]2[/green]
        """
        # Count seeds by status
        blending_count = sum(1 for s in seeds.values() if s.stage == "BLENDING")
        holding_count = sum(1 for s in seeds.values() if s.stage == "HOLDING")
        fossilized_count = sum(1 for s in seeds.values() if s.stage == "FOSSILIZED")

        # If no contributing seeds, show dash
        if blending_count + holding_count + fossilized_count == 0:
            return "─"

        # Get colors from STAGE_COLORS
        blending_color = STAGE_COLORS.get("BLENDING", "cyan")
        holding_color = STAGE_COLORS.get("HOLDING", "yellow")
        fossilized_color = STAGE_COLORS.get("FOSSILIZED", "green")

        # Format as colored counts
        return (
            f"[{blending_color}]{blending_count}[/{blending_color}]/"
            f"[{holding_color}]{holding_count}[/{holding_color}]/"
            f"[{fossilized_color}]{fossilized_count}[/{fossilized_color}]"
        )

    def _format_growth_ratio(self, growth_ratio: float) -> str:
        """Format growth ratio."""
        ratio = float(growth_ratio) if growth_ratio else 1.0
        if ratio <= 1.0:
            return "[dim]1.00x[/]"
        if ratio < 1.1:
            return f"[cyan]{ratio:.2f}x[/]"
        return f"[bold cyan]{ratio:.2f}x[/]"
