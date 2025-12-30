"""ThreadDeathModal - Prominent notification when training thread dies.

Shows a large, unmissable modal when the training thread stops.
This is a critical failure mode that requires operator attention.
"""
from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import Static


class ThreadDeathModal(ModalScreen[None]):
    """Modal shown when training thread dies.

    This is a critical failure notification. The modal is large and
    prominent to ensure the operator notices the crash.
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Close", show=True),
        Binding("q", "dismiss", "Close", show=False),
        Binding("enter", "dismiss", "Close", show=False),
    ]

    DEFAULT_CSS = """
    ThreadDeathModal {
        align: center middle;
        background: $error-darken-3 90%;
    }

    ThreadDeathModal > #death-container {
        width: 60;
        height: auto;
        max-height: 20;
        background: $error-darken-2;
        border: thick $error;
        padding: 2 4;
    }

    ThreadDeathModal .death-title {
        text-align: center;
        text-style: bold;
        color: $error;
        margin-bottom: 1;
    }

    ThreadDeathModal .death-message {
        text-align: center;
        margin-bottom: 1;
    }

    ThreadDeathModal .death-hint {
        text-align: center;
        color: $text-muted;
    }
    """

    def compose(self) -> ComposeResult:
        """Compose the death modal."""
        with Container(id="death-container"):
            yield Static(
                Text("⚠ TRAINING THREAD DIED ⚠", style="bold red"),
                classes="death-title",
            )
            yield Static(
                Text(
                    "The training thread has stopped unexpectedly.\n"
                    "Check the terminal for stack trace.",
                    style="white",
                ),
                classes="death-message",
            )
            yield Static(
                Text("Press ESC, Q, Enter or click to close", style="dim"),
                classes="death-hint",
            )

    def on_click(self) -> None:
        """Dismiss modal on click."""
        self.dismiss()
