"""PruneAttributionPanel - who pruned each slot, and why.

``seed.prune_reason`` + ``seed.auto_pruned`` carry prune attribution (a
system/governor auto-prune vs. a deliberate Tamiyo policy decision) but were
only visible in the per-seed detail modal. This panel promotes that attribution
to the Governor & Growth tab so "which slots were pruned, by whom, and why" is
legible without opening a modal.

Rows are the currently-pruned slots across all envs (the same seed state the
modal reads); a slot leaves the list when it is re-germinated. Attribution is
ephemeral by nature — this surfaces the live "just pruned" picture, not a
durable graveyard (blueprint prune counts live in the env graveyard).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.table import Table
from rich.text import Text
from textual.widgets import Static

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import SanctumSnapshot, SeedState

_MAX_ROWS = 16


class PruneAttributionPanel(Static):
    """Currently-pruned slots with initiator (system/governor vs policy) + reason."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._snapshot: SanctumSnapshot | None = None
        self.classes = "panel"
        self.border_title = "PRUNE ATTRIBUTION"

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        self._snapshot = snapshot
        self.refresh()

    def render(self) -> Table | Text:
        if self._snapshot is None:
            return Text("[no data]", style="dim")

        # Enumerate currently-pruned slots across all envs, stable-sorted by
        # (env, slot) since seed state carries no prune timestamp.
        pruned: list[tuple[int, SeedState]] = [
            (env_id, seed)
            for env_id, env in self._snapshot.envs.items()
            for seed in env.seeds.values()
            if seed.stage == "PRUNED"
        ]
        pruned.sort(key=lambda es: (es[0], es[1].slot_id))

        if not pruned:
            return Text(
                "No pruned slots right now — when the governor or the policy "
                "prunes a slot, who did it and why appears here.",
                style="dim",
            )

        table = Table(expand=False, pad_edge=False, border_style="dim")
        table.add_column("env/slot", style="cyan", min_width=8)
        table.add_column("blueprint", min_width=10)
        table.add_column("initiator", min_width=9)
        table.add_column("reason", min_width=14)

        for env_id, seed in pruned[:_MAX_ROWS]:
            initiator_text, initiator_style = self._initiator_cell(seed)
            table.add_row(
                f"{env_id}/{seed.slot_id}",
                seed.blueprint_id or "—",
                Text(initiator_text, style=initiator_style),
                seed.prune_reason or "—",
            )
        return table

    @staticmethod
    def _initiator_cell(seed: "SeedState") -> tuple[str, str]:
        """Who initiated the prune.

        ``auto_pruned`` True = the system/governor pruned it automatically
        (a safety/health intervention); False = the Tamiyo policy chose to
        prune it. Distinct colours so an operator can tell a forced prune from
        a deliberate one at a glance.
        """
        if seed.auto_pruned:
            return "system", "yellow"
        return "policy", "cyan"
