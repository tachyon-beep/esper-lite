"""Sanctum Textual Application.

Developer diagnostic TUI for debugging PPO training runs.
Layout matches existing Rich TUI (tui.py _render() method).

LAYOUT FIX: TamiyoBrain spans full width as dedicated row (size=11),
NOT embedded in right column. Event Log included at bottom-left.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Footer, Header, Static

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import SanctumSnapshot


class SanctumApp(App):
    """Sanctum diagnostic TUI for Esper training.

    Provides deep inspection of PPO training for debugging.
    Layout mirrors existing Rich TUI for 1:1 port.

    TERMINAL SIZE CONSTRAINTS:
    - Minimum: 120x40 (width x height) for readable display
    - Recommended: 140x50 or larger
    - TamiyoBrain requires width ≥ 80 for 4-column layout
    """

    TITLE = "Esper Sanctum"
    SUB_TITLE = "Diagnostic Console"

    CSS_PATH = "styles.tcss"

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("question_mark", "toggle_help", "Help", show=True),
        Binding("tab", "focus_next", "Next Panel", show=False),
        Binding("shift+tab", "focus_previous", "Prev Panel", show=False),
        Binding("1-9", "focus_env", "Focus Env", show=False),
        Binding("r", "refresh", "Refresh", show=True),
    ]

    def __init__(self, backend=None, num_envs: int = 16, **kwargs) -> None:
        """Initialize Sanctum app.

        Args:
            backend: Optional backend for live updates (can be None for testing)
            num_envs: Number of training environments (default 16)
        """
        super().__init__(**kwargs)
        self._backend = backend
        self.num_envs = num_envs
        self._snapshot: SanctumSnapshot | None = None

    def compose(self) -> ComposeResult:
        """Compose the application layout.

        CORRECTED LAYOUT (FIX from review):
        - Header: Run info
        - Top section (horizontal):
          - Left (65%): Environment Overview table
          - Right (35%): Scoreboard (Best Runs)
        - TamiyoBrain: Full-width dedicated row (size=11)
        - Bottom section (horizontal):
          - Left (65%): Event Log
          - Right (35%): Reward Components + Esper Status (vertical stack)
        - Footer: Keybindings
        """
        yield Header()

        with Container(id="sanctum-main"):
            # Top section: Env Overview + Scoreboard
            with Horizontal(id="top-section"):
                # Left: Environment Overview (65% width)
                env_overview = Static("[Environment Overview]", id="env-overview", classes="panel focusable")
                env_overview.can_focus = True
                yield env_overview

                # Right: Scoreboard (35% width)
                scoreboard = Static("[Best Runs]", id="scoreboard", classes="panel focusable")
                scoreboard.can_focus = True
                yield scoreboard

            # TamiyoBrain: Full-width dedicated row
            tamiyo = Static("[Tamiyo Brain]", id="tamiyo-brain", classes="panel focusable brain-panel")
            tamiyo.can_focus = True
            yield tamiyo

            # Bottom section: Event Log + (Rewards + Status)
            with Horizontal(id="bottom-section"):
                # Left: Event Log (65% width)
                event_log = Static("[Event Log]", id="event-log", classes="panel focusable")
                event_log.can_focus = True
                yield event_log

                # Right: Reward Components + Esper Status (vertical stack, 35% width)
                with Vertical(id="right-bottom"):
                    rewards = Static("[Reward Components]", id="reward-components", classes="panel focusable")
                    rewards.can_focus = True
                    yield rewards

                    status = Static("[Esper Status]", id="esper-status", classes="panel focusable")
                    status.can_focus = True
                    yield status

        yield Footer()

    def action_toggle_help(self) -> None:
        """Toggle help overlay."""
        self.notify("Help: q=quit, Tab=navigate, 1-9=focus env, r=refresh, ?=help")

    def action_focus_env(self, env_id: str) -> None:
        """Focus a specific environment by number."""
        try:
            env_num = int(env_id) - 1  # 1-indexed for user
            if 0 <= env_num < self.num_envs:
                if self._snapshot:
                    self._snapshot.focused_env_id = env_num
                    self._refresh_focused_panels()
        except ValueError:
            pass

    def action_refresh(self) -> None:
        """Refresh display from backend."""
        if self._backend:
            self._snapshot = self._backend.get_snapshot()
            self._refresh_all_panels()
        else:
            self.notify("No backend connected")

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        """Update all widgets with new snapshot."""
        self._snapshot = snapshot
        self._refresh_all_panels()

    def _refresh_all_panels(self) -> None:
        """Refresh all panels with current snapshot."""
        # TODO: [Task 7] Implement when widgets are wired up
        pass

    def _refresh_focused_panels(self) -> None:
        """Refresh panels that depend on focused env."""
        # TODO: [Task 7] Implement when focused-env-aware widgets are added
        pass
