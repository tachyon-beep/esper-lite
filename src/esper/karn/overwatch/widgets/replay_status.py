"""Replay Status Bar Widget.

Displays replay playback status:
- Play/pause indicator
- Speed multiplier
- Progress bar
- Frame counter
- Timestamp from current snapshot
"""

from __future__ import annotations

from textual.widgets import Static


def progress_bar(progress: float, width: int = 20) -> str:
    """Generate a progress bar.

    Args:
        progress: Progress 0.0 to 1.0
        width: Bar width in characters

    Returns:
        Progress bar like "████████░░░░░░░░░░░░"
    """
    filled = int(progress * width)
    filled = max(0, min(width, filled))
    return "█" * filled + "░" * (width - filled)


class ReplayStatusBar(Static):
    """Status bar showing replay playback state.

    Displays:
    - Mode icon (▶ playing, ⏸ paused)
    - Speed multiplier
    - Visual progress bar
    - Frame counter (current/total)
    - Timestamp from snapshot

    Usage:
        bar = ReplayStatusBar()
        bar.update_status(
            playing=True,
            speed=2.0,
            current=5,
            total=100,
            timestamp="12:00:05"
        )
    """

    DEFAULT_CSS = """
    ReplayStatusBar {
        height: 1;
        background: $primary-darken-2;
        padding: 0 1;
    }

    ReplayStatusBar.hidden {
        display: none;
    }
    """

    def __init__(self, **kwargs) -> None:
        """Initialize the replay status bar."""
        super().__init__("", **kwargs)
        self._playing = False
        self._speed = 1.0
        self._current = 0
        self._total = 0
        self._timestamp = ""
        self._visible = True

    @property
    def is_visible(self) -> bool:
        """Whether the bar is visible."""
        return self._visible

    def set_visible(self, visible: bool) -> None:
        """Set visibility.

        Args:
            visible: True to show, False to hide
        """
        self._visible = visible
        if visible:
            self.remove_class("hidden")
        else:
            self.add_class("hidden")

    def update_status(
        self,
        playing: bool,
        speed: float,
        current: int,
        total: int,
        timestamp: str = "",
    ) -> None:
        """Update replay status.

        Args:
            playing: Whether playback is active
            speed: Playback speed multiplier
            current: Current frame index (0-based)
            total: Total frame count
            timestamp: Timestamp from current snapshot
        """
        self._playing = playing
        self._speed = speed
        self._current = current
        self._total = total
        self._timestamp = timestamp
        self.update(self.render_bar())

    def render_bar(self) -> str:
        """Render the status bar content.

        Returns:
            Formatted status string
        """
        # Mode icon
        icon = "[green]▶[/green]" if self._playing else "[yellow]⏸[/yellow]"

        # Speed
        speed_str = f"{self._speed}x" if self._speed != 1.0 else "1x"

        # Progress
        progress = self._current / max(1, self._total - 1) if self._total > 1 else 0.0
        bar = progress_bar(progress, width=15)
        pct = int(progress * 100)

        # Frame counter (1-indexed for display)
        frame_str = f"{self._current + 1}/{self._total}"

        # Timestamp
        time_str = f" {self._timestamp}" if self._timestamp else ""

        return f"{icon} [bold]REPLAY[/bold] {speed_str} [{bar}] {frame_str} {pct}%{time_str}"
