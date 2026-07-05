"""GovernorScreen - the Governor & Growth tab.

Gives the safety gate its own surface (independence is legible when the gate is
not nested inside a policy panel): the always-on governor status on top, with
the durable rollback ledger and the morphology causal chain beneath.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from textual.app import ComposeResult
from textual.containers import Container, Horizontal

from esper.karn.sanctum.widgets.governor_ledger_panel import GovernorLedgerPanel
from esper.karn.sanctum.widgets.governor_status_panel import GovernorStatusPanel
from esper.karn.sanctum.widgets.morphology_causal_panel import MorphologyCausalPanel
from esper.karn.sanctum.widgets.prune_attribution_panel import PruneAttributionPanel

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import SanctumSnapshot


class GovernorScreen(Container):
    """Full-width governor + growth diagnostics tab."""

    DEFAULT_CSS = """
    GovernorScreen {
        layout: vertical;
        height: 1fr;
        padding: 0 1;
        overflow-y: auto;
    }

    GovernorScreen #governor-status {
        width: 100%;
        height: auto;
        min-height: 5;
        border: round $surface-lighten-2;
        border-title-color: $text-muted;
        padding: 0 1;
    }

    GovernorScreen #governor-lower {
        width: 100%;
        height: 1fr;
        min-height: 12;
    }

    GovernorScreen #governor-prunes {
        width: 100%;
        height: auto;
        min-height: 4;
        border: round $surface-lighten-2;
        border-title-color: $text-muted;
        padding: 0 1;
    }

    /* Child selector, NOT a class: the panels overwrite self.classes with
       "panel" in __init__, so classes passed at construction are lost. */
    GovernorScreen #governor-lower > * {
        width: 1fr;
        min-width: 44;
        height: 100%;
        border: round $surface-lighten-2;
        border-title-color: $text-muted;
        padding: 0 1;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._snapshot: SanctumSnapshot | None = None

    def compose(self) -> ComposeResult:
        yield GovernorStatusPanel(id="governor-status")
        with Horizontal(id="governor-lower"):
            yield GovernorLedgerPanel(id="governor-ledger")
            yield MorphologyCausalPanel(id="governor-causal")
        yield PruneAttributionPanel(id="governor-prunes")

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        """Fan the snapshot out to all governor/growth panels."""
        self._snapshot = snapshot
        self.query_one("#governor-status", GovernorStatusPanel).update_snapshot(snapshot)
        self.query_one("#governor-ledger", GovernorLedgerPanel).update_snapshot(snapshot)
        self.query_one("#governor-causal", MorphologyCausalPanel).update_snapshot(snapshot)
        self.query_one("#governor-prunes", PruneAttributionPanel).update_snapshot(snapshot)

    @property
    def snapshot(self) -> "SanctumSnapshot | None":
        """Latest snapshot routed to this tab (test seam)."""
        return self._snapshot
