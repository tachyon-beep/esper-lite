"""GovernorLedgerPanel - the durable rollback trace.

A governor panic rolls back an env and self-erases from the env row in ~5s;
this ledger is the durable record so "why did env N roll back at epoch M?" is
answerable long after. Rows are newest-first and chronological (NOT sorted by
severity — rollback_severity is a run-constant penalty, not an event magnitude).
loss_threshold is shown only for a divergence panic, where it was the trip line;
for nan/lobotomy panics it was computed but not the bar that fired.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.table import Table
from rich.text import Text
from textual.widgets import Static

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import GovernorRollbackRecord, SanctumSnapshot

_REASON_STYLE: dict[str, str] = {
    "governor_nan": "red bold",
    "governor_lobotomy": "red bold",
    "governor_divergence": "red",
    "governor_rollback": "red",
}


class GovernorLedgerPanel(Static):
    """Durable, newest-first governor rollback ledger."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._snapshot: SanctumSnapshot | None = None
        self.classes = "panel"
        self.border_title = "ROLLBACK LEDGER"

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        self._snapshot = snapshot
        self.refresh()

    def render(self) -> Table | Text:
        if self._snapshot is None:
            return Text("[no data]", style="dim")

        ledger = self._snapshot.governor.rollback_ledger
        if not ledger:
            return Text(
                "No governor rollbacks — the gate has not fired.", style="green"
            )

        table = Table(expand=False, pad_edge=False, border_style="dim")
        table.add_column("env/epoch", style="cyan", min_width=9)
        table.add_column("reason", min_width=13)
        table.add_column("loss@panic", justify="right", min_width=10)
        table.add_column("vs thresh", justify="right", min_width=9)
        table.add_column("panics", justify="right", min_width=6)
        table.add_column("action", min_width=10)

        for rec in reversed(ledger):  # newest-first
            # The "governor_" prefix is redundant on the governor tab; strip it so
            # the reason fits the column without truncation.
            reason = rec.panic_reason.removeprefix("governor_")
            table.add_row(
                f"{rec.env_id} / e{rec.epoch}",
                Text(reason, style=_REASON_STYLE.get(rec.panic_reason, "red")),
                self._fmt(rec.loss_at_panic),
                self._threshold_cell(rec),
                "—" if rec.consecutive_panics is None else str(rec.consecutive_panics),
                self._action_cell(rec),
            )
        return table

    @staticmethod
    def _fmt(value: float | None) -> str:
        return "—" if value is None else f"{value:.2f}"

    @classmethod
    def _threshold_cell(cls, rec: "GovernorRollbackRecord") -> str:
        # loss_threshold was the trip line only for a divergence panic; for
        # nan/lobotomy it was computed but not what fired, so don't imply it.
        if rec.panic_reason != "governor_divergence":
            return "—"
        return cls._fmt(rec.loss_threshold)

    @staticmethod
    def _action_cell(rec: "GovernorRollbackRecord") -> Text:
        if rec.attributed and rec.triggering_action_id:
            return Text(rec.triggering_action_id, style="dim")
        return Text("(unattr)", style="yellow")
