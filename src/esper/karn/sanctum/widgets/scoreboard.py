"""Scoreboard widget - Best Runs leaderboard.

Port of tui.py _render_scoreboard() (lines 1083-1190).
Shows stats header and top 10 environments by best accuracy.

Reference: src/esper/karn/tui.py lines 1083-1190 (_render_scoreboard method)

Interactive features:
- j/k or ↑/↓: Navigate rows
- Enter: Open HistoricalEnvDetail modal with snapshot at peak
- p: Toggle pin status (pinned rows never get removed)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from textual.binding import Binding
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import DataTable, Static

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import BestRunRecord, SanctumSnapshot, SeedState


class Scoreboard(Static):
    """Best Runs scoreboard widget with keyboard navigation.

    Shows:
    1. Stats header: global best, mean best, fossilized/pruned counts
    2. Leaderboard: top 10 completed env runs with peak accuracy + seed composition

    Keyboard:
    - j/k or ↑/↓: Navigate rows
    - Enter: Open historical detail view
    - p: Toggle pin on selected row
    """

    BINDINGS = [
        Binding("p", "toggle_pin", "Toggle Pin", show=False),
    ]

    class BestRunSelected(Message):
        """Posted when user selects a best run to view details."""

        def __init__(self, record: "BestRunRecord") -> None:
            super().__init__()
            self.record = record

    class BestRunPinToggled(Message):
        """Posted when user toggles pin on a best run."""

        def __init__(self, record_id: str) -> None:
            super().__init__()
            self.record_id = record_id

    def __init__(self, **kwargs) -> None:
        """Initialize Scoreboard widget."""
        super().__init__(**kwargs)
        self._snapshot: SanctumSnapshot | None = None
        self._displayed_records: list["BestRunRecord"] = []
        self.border_title = "BEST RUNS"
        self.table = DataTable(zebra_stripes=True, cursor_type="row")

    def compose(self):
        """Compose the widget."""
        yield Static(id="scoreboard-stats")
        yield self.table

    def on_mount(self) -> None:
        """Setup table columns on mount."""
        self._setup_columns()

    def _setup_columns(self) -> None:
        """Setup leaderboard table columns."""
        self.table.clear(columns=True)
        self.table.add_column("#", key="rank", width=3)
        self.table.add_column("Ep", key="episode", width=4)
        self.table.add_column("Acc", key="accuracy", width=6)
        self.table.add_column("Growth", key="growth", width=7)
        self.table.add_column("Seeds", key="seeds", width=18)

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        """Update widget with new snapshot data."""
        self._snapshot = snapshot
        self._refresh_stats()
        self._refresh_table()

    def _refresh_stats(self) -> None:
        """Refresh the stats header."""
        if self._snapshot is None:
            return

        try:
            stats_widget = self.query_one("#scoreboard-stats", Static)
        except Exception:
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
        stats_widget.update(stats_text)

        # Update title
        pinned_count = sum(1 for r in best_runs if r.pinned)
        if pinned_count > 0:
            self.border_title = f"BEST RUNS ({pinned_count} 📌) [dim]Enter=view p=pin[/dim]"
        else:
            self.border_title = "BEST RUNS [dim]Enter=view p=pin[/dim]"

    def _refresh_table(self) -> None:
        """Refresh the leaderboard table.

        Preserves cursor and scroll position across refresh cycles.
        """
        if self._snapshot is None:
            return

        # Save cursor and scroll position
        saved_cursor = self.table.cursor_row
        saved_scroll_y = self.table.scroll_y

        self.table.clear()

        best_runs = list(self._snapshot.best_runs)
        if not best_runs:
            self._displayed_records = []
            self.table.add_row("[dim]No runs yet[/dim]", "", "", "", "")
            return

        # Sort by peak accuracy, take top 10
        best_runs = sorted(best_runs, key=lambda r: r.peak_accuracy, reverse=True)[:10]
        self._displayed_records = best_runs

        for i, record in enumerate(best_runs, start=1):
            rank_display = f"📌{i}" if record.pinned else str(i)
            # Use record_id if available, otherwise fallback to index-based key
            row_key = record.record_id if record.record_id else f"row_{i}"
            self.table.add_row(
                rank_display,
                str(record.episode + 1),
                f"[bold green]{record.peak_accuracy:.1f}[/bold green]",
                self._format_growth_ratio(record.growth_ratio),
                self._format_seeds(record.seeds),
                key=row_key,
            )

        # Restore cursor position
        if self.table.row_count > 0 and saved_cursor is not None:
            target = min(saved_cursor, self.table.row_count - 1)
            self.table.move_cursor(row=target)

        # Restore scroll position
        if saved_scroll_y > 0:
            self.table.scroll_y = saved_scroll_y

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle Enter key on table row."""
        if not self._displayed_records:
            return

        row_key = event.row_key
        if row_key is None:
            return

        key_value = row_key.value

        # Find record by key - check both record_id and fallback row_N format
        for i, record in enumerate(self._displayed_records, start=1):
            if record.record_id and record.record_id == key_value:
                self.post_message(self.BestRunSelected(record))
                break
            elif key_value == f"row_{i}":
                self.post_message(self.BestRunSelected(record))
                break

    def action_toggle_pin(self) -> None:
        """Toggle pin on currently selected row."""
        if not self._displayed_records or self.table.row_count == 0:
            return

        cursor_row = self.table.cursor_row
        if cursor_row is None or cursor_row >= len(self._displayed_records):
            return

        record = self._displayed_records[cursor_row]
        if record.record_id:
            self.post_message(self.BestRunPinToggled(record.record_id))

    def _format_seeds(self, seeds: dict[str, "SeedState"]) -> str:
        """Format seed composition at peak accuracy."""
        contributing = [
            seed
            for seed in seeds.values()
            if seed.blueprint_id and seed.stage in {"FOSSILIZED", "BLENDING", "HOLDING"}
        ]
        if not contributing:
            return "─"

        stage_order = {"FOSSILIZED": 0, "BLENDING": 1, "HOLDING": 2}
        stage_colors = {"FOSSILIZED": "green", "BLENDING": "magenta", "HOLDING": "yellow"}
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
        """Format growth ratio."""
        ratio = float(growth_ratio) if growth_ratio else 1.0
        if ratio <= 1.0:
            return "[dim]1.00x[/]"
        if ratio < 1.1:
            return f"[cyan]{ratio:.2f}x[/]"
        return f"[bold cyan]{ratio:.2f}x[/]"
