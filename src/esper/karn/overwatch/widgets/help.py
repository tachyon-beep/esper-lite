"""Help Overlay Widget.

Displays keyboard shortcuts and navigation help.
Press ? to show, Esc to dismiss.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Static


HELP_TEXT = """\
[bold cyan]Overwatch Keyboard Shortcuts[/bold cyan]

[bold]Navigation[/bold]
  [cyan]j / ↓[/cyan]    Move down in flight board
  [cyan]k / ↑[/cyan]    Move up in flight board
  [cyan]Enter[/cyan]    Expand selected environment
  [cyan]Esc[/cyan]      Collapse / Close overlay

[bold]Panels[/bold]
  [cyan]c[/cyan]        Show context panel (why flagged)
  [cyan]t[/cyan]        Show Tamiyo detail panel
  [cyan]f[/cyan]        Toggle event feed size

[bold]Replay[/bold]
  [cyan]Space[/cyan]    Play / Pause replay
  [cyan].[/cyan]        Step forward one snapshot
  [cyan],[/cyan]        Step backward one snapshot
  [cyan]< / >[/cyan]    Decrease / Increase playback speed

[bold]General[/bold]
  [cyan]?[/cyan]        Toggle this help overlay
  [cyan]q[/cyan]        Quit Overwatch

[dim]Press Esc to close this help[/dim]
"""


class HelpOverlay(Container):
    """Modal overlay showing keyboard shortcuts.

    Usage:
        # In app compose():
        yield HelpOverlay(id="help-overlay", classes="hidden")

        # Toggle visibility:
        self.query_one("#help-overlay").toggle_class("hidden")
    """

    DEFAULT_CSS = """
    HelpOverlay {
        align: center middle;
        width: 60;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
    }

    HelpOverlay.hidden {
        display: none;
    }
    """

    def compose(self) -> ComposeResult:
        """Compose the help content."""
        yield Static(HELP_TEXT, markup=True)
