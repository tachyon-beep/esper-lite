"""GovernorStatusPanel - the safety gate's independence, made legible.

The governor is not the policy: it is not trained, and the controller cannot
disable it or tune its thresholds (the controller-cannot-disable-gate
invariant). This panel renders its status in its OWN region — never nested in a
policy panel — so that separation reads at a glance. It shows an always-on
ARMED/warming indicator (the gate has no disable path in the code) plus the
rollback-starvation counters, and it deliberately does not derive anything from
policy health.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.text import Text
from textual.widgets import Static

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import GovernorState, SanctumSnapshot


class GovernorStatusPanel(Static):
    """Governor arming + rollback counters (independent of the policy)."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._snapshot: SanctumSnapshot | None = None
        self.classes = "panel"
        self.border_title = "GOVERNOR (safety gate)"

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        self._snapshot = snapshot
        self.refresh()

    def render(self) -> Text:
        if self._snapshot is None:
            return Text("[no data]", style="dim")

        gov = self._snapshot.governor
        result = Text()

        # Arming line — two-state so we never overclaim "all rules" during the
        # NaN-only warmup window.
        result.append("Status  ", style="dim")
        self._render_arming(result, gov)
        result.append("   independent of policy — cannot be disabled", style="dim")
        result.append("\n")

        # Rollback-starvation counters (a property of the gate firing).
        result.append("Rollout ", style="dim")
        attempts = gov.rollback_attempt_count
        result.append("attempts:", style="dim")
        result.append(f"{attempts}", style="yellow" if attempts > 0 else "green")
        unattr = gov.rollback_unattributed_count
        result.append("  unattributed:", style="dim")
        result.append(f"{unattr}", style="yellow" if unattr > 0 else "green")
        result.append("\n")

        # Cumulative rollbacks (survive ledger ring eviction).
        result.append("Total   ", style="dim")
        total = gov.total_rollbacks
        result.append("rollbacks:", style="dim")
        result.append(f"{total}", style="red bold" if total > 0 else "green")
        if gov.rollbacks_by_reason:
            by_reason = ", ".join(
                f"{reason}:{count}"
                for reason, count in sorted(gov.rollbacks_by_reason.items())
            )
            result.append(f"  ({by_reason})", style="dim")

        return result

    @staticmethod
    def _render_arming(result: Text, gov: "GovernorState") -> None:
        if not gov.present:
            result.append("● NOT WIRED", style="red bold")
            return
        if gov.total_env_count == 0:
            result.append("● present — no envs yet", style="dim")
            return
        if gov.warming_env_count > 0:
            result.append(
                f"◐ warming {gov.warming_env_count}/{gov.total_env_count} (NaN-only)",
                style="yellow",
            )
            if gov.armed_env_count > 0:
                result.append(
                    f" · armed {gov.armed_env_count}", style="green"
                )
            return
        result.append(f"● ARMED — watching {gov.armed_env_count} envs", style="green bold")
