"""Detail Panel Container Widget.

Switchable container that shows either:
- ContextPanel: Environment details and "why flagged"
- TamiyoDetailPanel: Full Tamiyo agent diagnostics

Toggle with 'c' for context, 't' for tamiyo.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from textual.app import ComposeResult
from textual.containers import Container
from textual.reactive import reactive

from esper.karn.overwatch.widgets.context_panel import ContextPanel
from esper.karn.overwatch.widgets.tamiyo_detail import TamiyoDetailPanel

if TYPE_CHECKING:
    from esper.karn.overwatch.schema import EnvSummary, TamiyoState


PanelMode = Literal["context", "tamiyo"]


class DetailPanel(Container):
    """Switchable detail panel container.

    Displays either:
    - Context mode: Selected env details, anomaly reasons, slot info
    - Tamiyo mode: Full Tamiyo diagnostics, action distribution, confidence

    Toggle modes with set_mode() or toggle_mode().
    """

    DEFAULT_CSS = """
    DetailPanel {
        width: 100%;
        height: 100%;
    }

    DetailPanel > ContextPanel {
        display: block;
    }

    DetailPanel > TamiyoDetailPanel {
        display: none;
    }

    DetailPanel.tamiyo-mode > ContextPanel {
        display: none;
    }

    DetailPanel.tamiyo-mode > TamiyoDetailPanel {
        display: block;
    }
    """

    mode: reactive[PanelMode] = reactive("context")

    def __init__(self, **kwargs) -> None:
        """Initialize the detail panel."""
        super().__init__(**kwargs)
        self._env: EnvSummary | None = None
        self._tamiyo: TamiyoState | None = None

    def compose(self) -> ComposeResult:
        """Compose both panels (visibility controlled by CSS)."""
        yield ContextPanel(id="context-panel")
        yield TamiyoDetailPanel(id="tamiyo-panel")

    def set_mode(self, mode: PanelMode) -> None:
        """Set the panel mode.

        Args:
            mode: "context" or "tamiyo"
        """
        self.mode = mode
        self._update_mode_class()

    def toggle_mode(self, mode: PanelMode) -> None:
        """Toggle to a mode, or back to context if already in that mode.

        Args:
            mode: "context" or "tamiyo"
        """
        if self.mode == mode:
            self.mode = "context"
        else:
            self.mode = mode
        self._update_mode_class()

    def _update_mode_class(self) -> None:
        """Update CSS class based on current mode."""
        if self.mode == "tamiyo":
            self.add_class("tamiyo-mode")
        else:
            self.remove_class("tamiyo-mode")

    def update_env(self, env: EnvSummary | None) -> None:
        """Update the selected environment.

        Args:
            env: Environment summary or None
        """
        self._env = env
        try:
            self.query_one(ContextPanel).update_env(env)
        except Exception:
            pass  # Not mounted yet

    def update_tamiyo(self, tamiyo: TamiyoState | None) -> None:
        """Update the Tamiyo state.

        Args:
            tamiyo: Tamiyo state or None
        """
        self._tamiyo = tamiyo
        try:
            self.query_one(TamiyoDetailPanel).update_tamiyo(tamiyo)
        except Exception:
            pass  # Not mounted yet
