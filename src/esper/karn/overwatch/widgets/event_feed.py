"""Event Feed Widget.

Displays timestamped lifecycle events with type badges:
- GATE: Gate evaluations (cyan)
- STAGE: Stage transitions (blue)
- PPO: Policy updates (magenta)
- GERM: Seed germinations (green)
- PRUNE: Seed prunes (red)

Supports filtering by event type and compact/expanded modes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import ScrollableContainer
from textual.widgets import Static

if TYPE_CHECKING:
    from esper.karn.overwatch.schema import FeedEvent


# Event type to color mapping
EVENT_COLORS: dict[str, str] = {
    "GATE": "cyan",
    "STAGE": "blue",
    "PPO": "magenta",
    "GERM": "green",
    "PRUNE": "red",
    "WARN": "yellow",
    "ERROR": "red",
}


def event_badge(event_type: str) -> str:
    """Render event type as colored badge.

    Args:
        event_type: Event type string (GATE, PPO, etc.)

    Returns:
        Rich markup string for badge like "[cyan][GATE][/cyan]"
    """
    color = EVENT_COLORS.get(event_type, "white")
    return f"[{color}][{event_type}][/{color}]"


class EventFeed(ScrollableContainer):
    """Scrollable feed of timestamped lifecycle events.

    Features:
    - Type badges with color coding
    - Compact (4 lines) / expanded (8 lines) modes
    - Filter by event type
    - Newest events at bottom (log style)

    Usage:
        feed = EventFeed()
        feed.update_events(snapshot.event_feed)
        feed.set_filter("GATE")  # Only show gate events
        feed.toggle_expanded()   # Toggle size
    """

    DEFAULT_CSS = """
    EventFeed {
        height: 6;
        border: solid $primary;
        padding: 0 1;
    }

    EventFeed.expanded {
        height: 12;
    }

    EventFeed .event-line {
        height: 1;
    }

    EventFeed .timestamp {
        color: $text-muted;
    }

    EventFeed .env-id {
        color: $secondary;
    }
    """

    def __init__(self, **kwargs) -> None:
        """Initialize the event feed."""
        super().__init__(**kwargs)
        self._events: list[FeedEvent] = []
        self._filter_type: str | None = None
        self._expanded = False

    @property
    def expanded(self) -> bool:
        """Whether feed is in expanded mode."""
        return self._expanded

    def toggle_expanded(self) -> None:
        """Toggle between compact and expanded mode."""
        self._expanded = not self._expanded
        if self._expanded:
            self.add_class("expanded")
        else:
            self.remove_class("expanded")

    def set_filter(self, event_type: str) -> None:
        """Filter events to show only specified type.

        Args:
            event_type: Event type to filter (e.g., "GATE")
        """
        self._filter_type = event_type
        self._refresh_content()

    def clear_filter(self) -> None:
        """Clear filter to show all events."""
        self._filter_type = None
        self._refresh_content()

    def update_events(self, events: list[FeedEvent]) -> None:
        """Update the event list.

        Args:
            events: List of FeedEvent objects
        """
        self._events = events
        self._refresh_content()

    def _get_filtered_events(self) -> list[FeedEvent]:
        """Get events after applying filter."""
        if self._filter_type is None:
            return self._events
        return [e for e in self._events if e.event_type == self._filter_type]

    def render_events(self) -> str:
        """Render all events as text.

        Returns:
            Multi-line string with all event lines
        """
        events = self._get_filtered_events()

        if not events:
            filter_text = f" (filter: {self._filter_type})" if self._filter_type else ""
            return f"[dim]No events{filter_text}[/dim]"

        lines = []
        for event in events:
            line = self._render_event_line(event)
            lines.append(line)

        return "\n".join(lines)

    def _render_event_line(self, event: FeedEvent) -> str:
        """Render a single event line.

        Args:
            event: FeedEvent to render

        Returns:
            Formatted line like "12:00:01 [GATE] [E3] Gate G1 passed"
        """
        parts = []

        # Timestamp (dimmed)
        parts.append(f"[dim]{event.timestamp}[/dim]")

        # Event badge
        parts.append(event_badge(event.event_type))

        # Env ID if present
        if event.env_id is not None:
            parts.append(f"[cyan][E{event.env_id}][/cyan]")

        # Message
        parts.append(event.message)

        return " ".join(parts)

    def compose(self) -> ComposeResult:
        """Compose the feed layout."""
        yield Static(self.render_events(), id="event-content")

    def _refresh_content(self) -> None:
        """Refresh the displayed content."""
        try:
            self.query_one("#event-content", Static).update(self.render_events())
        except Exception:
            pass  # Widget not mounted yet
