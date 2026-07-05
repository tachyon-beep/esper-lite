"""MorphologyCausalPanel - the growth causal chain, finally rendered.

The morphology causal log (proposal → verdict → mutation → dispatch → commit →
rollback/fossilization → cooldown/audit) is captured in-snapshot but was
rendered by ZERO Sanctum widgets — it had been ceded to the Overwatch web UI.
This panel surfaces the recent lifecycle trail so an operator can answer "what
did the system try to grow, did the governor approve it, and how did it end?"
without leaving the terminal.

Each row is one causal-log entry: the phase, the env/slot it acted on, the
operation, the governor's verdict on it, and the watch-window evidence. Rows
are newest-first; identity/RNG-provenance fields (proposal/verdict/mutation IDs,
observation hash, rng stream/seed) back the join but are not all shown inline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.table import Table
from rich.text import Text
from textual.widgets import Static

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import MorphologyCausalLogEntry, SanctumSnapshot

# Phase → display color. The lifecycle runs proposal → verdict → mutation →
# dispatch → commit → (rollback | fossilization) → cooldown → audit.
_PHASE_STYLE: dict[str, str] = {
    "proposal": "dim",
    "verdict": "cyan",
    "mutation": "blue",
    "dispatch": "blue",
    "commit": "green",
    "rollback": "red bold",
    "fossilization": "green bold",
    "cooldown": "yellow",
    "audit": "dim",
}

_MAX_ROWS = 12


class MorphologyCausalPanel(Static):
    """Recent morphology causal-log trail (growth lifecycle)."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._snapshot: SanctumSnapshot | None = None
        self.classes = "panel"
        self.border_title = "MORPHOLOGY CAUSAL LOG"

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        self._snapshot = snapshot
        self.refresh()

    def render(self) -> Table | Text:
        if self._snapshot is None:
            return Text("[no data]", style="dim")

        log = self._snapshot.morphology_causal_log
        if not log:
            return Text(
                "No morphology events yet — proposals, verdicts, commits and "
                "rollbacks appear here as slots differentiate.",
                style="dim",
            )

        table = Table(expand=False, pad_edge=False, border_style="dim")
        table.add_column("phase", min_width=13)
        table.add_column("env/slot", style="cyan", min_width=8)
        table.add_column("operation", min_width=10)
        table.add_column("governor", min_width=12)
        table.add_column("evidence", justify="right", min_width=8)

        for entry in list(log)[-_MAX_ROWS:][::-1]:  # newest first
            phase_style = _PHASE_STYLE.get(entry.phase, "white")
            gov_text, gov_style = self._governor_cell(entry)
            evidence = (
                f"{entry.watch_window_evidence:.2f}"
                if entry.watch_window_evidence is not None
                else "—"
            )
            table.add_row(
                Text(entry.phase, style=phase_style),
                f"{entry.env_id}/{entry.slot_id}",
                entry.operation or "—",
                Text(gov_text, style=gov_style),
                evidence,
            )

        return table

    @staticmethod
    def _governor_cell(entry: "MorphologyCausalLogEntry") -> tuple[str, str]:
        """The governor's verdict on this entry (approved / blocked / n-a).

        governor_approved is None when the phase carries no governor decision;
        that is rendered as a neutral em dash, never as an implied approval.
        """
        if entry.governor_approved is None:
            return "—", "dim"
        if entry.governor_approved:
            return "✓ approved", "green"
        blocked = entry.governor_blocked_factor or entry.governor_reason or "blocked"
        return f"✗ {blocked}", "red"
