"""LifecyclePanel - Displays seed lifecycle event history.

Shows Tamiyo's decisions and automatic transitions for seeds.
Used in both live and historical seed modals.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.text import Text
from textual.widgets import Static

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import SeedLifecycleEvent


# Stage transition colors
STAGE_COLORS = {
    "DORMANT": "dim",
    "GERMINATED": "cyan",
    "TRAINING": "yellow",
    "BLENDING": "blue",
    "HOLDING": "magenta",
    "FOSSILIZED": "green",
    "PRUNED": "red",
}


class LifecyclePanel(Static):
    """Widget displaying seed lifecycle event history.

    Args:
        events: List of lifecycle events to display.
        slot_filter: If set, only show events for this slot. None shows all
            (and displays slot column).
    """

    def __init__(
        self,
        events: list["SeedLifecycleEvent"],
        slot_filter: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._events = events
        self._slot_filter = slot_filter
        self._show_slot_column = slot_filter is None

    def update_events(
        self,
        events: list["SeedLifecycleEvent"],
        slot_filter: str | None = None,
    ) -> None:
        """Update the events and refresh display."""
        self._events = events
        self._slot_filter = slot_filter
        self._show_slot_column = slot_filter is None
        self.refresh()

    def _get_filtered_events(self) -> list["SeedLifecycleEvent"]:
        """Get events filtered by slot."""
        if self._slot_filter is None:
            return self._events
        return [e for e in self._events if e.slot_id == self._slot_filter]

    def render(self) -> Panel:
        """Render the lifecycle panel."""
        events = self._get_filtered_events()
        filter_label = self._slot_filter or "All"

        if not events:
            content: RenderableType = Text("No lifecycle events", style="dim italic")
            return Panel(
                content,
                title=f"Lifecycle [f] Filter: {filter_label}",
                border_style="dim",
            )

        lines: list[Text] = []
        for event in events:
            line = Text()

            # Epoch
            line.append(f"e{event.epoch:<3} ", style="dim")

            # Slot ID (if showing all)
            if self._show_slot_column:
                line.append(f"{event.slot_id:<5} ", style="cyan")

            # Action
            action_style = "bold yellow" if event.action == "[auto]" else "bold white"
            line.append(f"{event.action:<22} ", style=action_style)

            # Transition
            from_color = STAGE_COLORS.get(event.from_stage, "white")
            to_color = STAGE_COLORS.get(event.to_stage, "white")
            line.append(f"{event.from_stage}", style=from_color)
            line.append(" -> ", style="dim")
            line.append(f"{event.to_stage}", style=to_color)

            # Alpha (if present)
            if event.alpha is not None:
                line.append(f"  a={event.alpha:.2f}", style="blue")

            # Accuracy delta (if present)
            if event.accuracy_delta is not None:
                delta_style = "green" if event.accuracy_delta >= 0 else "red"
                line.append(f"  {event.accuracy_delta:+.1f}%", style=delta_style)

            lines.append(line)

        return Panel(
            Group(*lines),
            title=f"Lifecycle [f] Filter: {filter_label}",
            border_style="cyan",
        )
