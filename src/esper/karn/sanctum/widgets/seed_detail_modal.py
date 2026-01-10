"""SeedDetailModal - Modal for viewing detailed seed state with lifecycle.

Shows the current seed state and its lifecycle history.
Triggered by clicking a SeedCard in the env detail screen.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.console import Group
from rich.panel import Panel
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Static

from esper.karn.sanctum.formatting import format_params
from esper.karn.sanctum.widgets.lifecycle_panel import LifecyclePanel
from esper.leyline import ALPHA_CURVE_GLYPHS, STAGE_COLORS

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import SeedLifecycleEvent, SeedState


class SeedDetailRequested(Message):
    """Request to open a drill-down view for a seed.

    Posted by SeedCard when clicked. Handled by EnvDetailScreen.
    """

    bubble = True

    def __init__(self, *, slot_id: str, seed: "SeedState | None") -> None:
        super().__init__()
        self.slot_id = slot_id
        self.seed = seed


class SeedDetailModal(ModalScreen[None]):
    """Modal for viewing seed details with lifecycle history."""

    BINDINGS = [
        Binding("escape", "dismiss", "Close", show=True),
        Binding("q", "dismiss", "Close", show=False),
        Binding("f", "toggle_filter", "Filter", show=True),
    ]

    DEFAULT_CSS = """
    SeedDetailModal {
        align: center middle;
        background: $surface-darken-1 90%;
    }

    SeedDetailModal > #modal-container {
        width: 80%;
        height: 80%;
        background: $surface;
        border: thick $secondary;
        padding: 1 2;
    }

    SeedDetailModal #seed-header {
        height: auto;
        padding: 0 1;
        margin-bottom: 1;
    }

    SeedDetailModal #seed-details {
        height: auto;
        padding: 0 1;
        margin-bottom: 1;
    }

    SeedDetailModal #lifecycle-container {
        height: 1fr;
        min-height: 10;
    }

    SeedDetailModal #footer {
        height: 1;
        text-align: center;
        color: $text-muted;
    }
    """

    def __init__(
        self,
        seed: "SeedState | None",
        slot_id: str,
        lifecycle_events: list["SeedLifecycleEvent"],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._seed = seed
        self._slot_id = slot_id
        self._lifecycle_events = lifecycle_events
        self._filter_slot: str | None = slot_id  # Start filtered to this slot

    def compose(self) -> ComposeResult:
        with Container(id="modal-container"):
            yield Static(self._render_header(), id="seed-header")
            yield Static(self._render_seed_details(), id="seed-details")
            with Vertical(id="lifecycle-container"):
                yield LifecyclePanel(
                    events=self._lifecycle_events,
                    slot_filter=self._filter_slot,
                    id="lifecycle-panel",
                )
            yield Static("[dim]Press ESC to close, F to toggle filter[/dim]", id="footer")

    def action_toggle_filter(self) -> None:
        """Toggle between filtered and all slots."""
        if self._filter_slot is None:
            self._filter_slot = self._slot_id
        else:
            self._filter_slot = None

        lifecycle = self.query_one("#lifecycle-panel", LifecyclePanel)
        lifecycle.update_events(self._lifecycle_events, self._filter_slot)

    def _render_header(self) -> Text:
        header = Text()
        header.append(f"Seed: {self._slot_id}", style="bold")
        if self._seed and self._seed.blueprint_id:
            header.append(f" ({self._seed.blueprint_id})", style="cyan")
        return header

    def _render_seed_details(self) -> Panel:
        if self._seed is None or self._seed.stage == "DORMANT":
            return Panel(Text("DORMANT", style="dim"), title="State", border_style="dim")

        seed = self._seed
        lines = []

        # Stage with color (STAGE_COLORS contains all valid stages from leyline)
        stage_color = STAGE_COLORS[seed.stage]
        lines.append(Text(f"Stage: {seed.stage}", style=f"bold {stage_color}"))

        # Blueprint
        if seed.blueprint_id:
            lines.append(Text(f"Blueprint: {seed.blueprint_id}", style="white"))

        # Parameters
        if seed.seed_params and seed.seed_params > 0:
            lines.append(Text(f"Params: {format_params(seed.seed_params)}", style="dim"))

        # Alpha with progress bar (alpha is always a float per SeedState schema)
        # Show for BLENDING/HOLDING always, show for others if non-zero
        if seed.alpha > 0 or seed.stage in ("BLENDING", "HOLDING"):
            alpha_bar = self._make_alpha_bar(seed.alpha)
            lines.append(Text(f"Alpha: {seed.alpha:.2f} {alpha_bar}"))

        # Blend tempo and curve
        if seed.stage in ("BLENDING", "HOLDING") and seed.blend_tempo_epochs is not None:
            tempo = seed.blend_tempo_epochs
            tempo_name = "FAST" if tempo <= 3 else ("STANDARD" if tempo <= 5 else "SLOW")
            tempo_arrows = ">>>" if tempo <= 3 else (">>" if tempo <= 5 else ">")
            # alpha_curve is always valid (default "LINEAR", all values in ALPHA_CURVE_GLYPHS)
            curve_glyph = ALPHA_CURVE_GLYPHS[seed.alpha_curve]
            lines.append(Text(f"Tempo: {tempo_arrows} {tempo_name} ({tempo} epochs) {curve_glyph}"))

        # Accuracy delta
        if seed.accuracy_delta is not None and seed.accuracy_delta != 0:
            delta_style = "green" if seed.accuracy_delta > 0 else "red"
            lines.append(Text(f"Accuracy Delta: {seed.accuracy_delta:+.2f}%", style=delta_style))

        # Gradient health
        grad_text = Text("Gradient: ")
        if seed.has_exploding:
            grad_text.append("EXPLODING", style="bold red")
        elif seed.has_vanishing:
            grad_text.append("VANISHING", style="bold yellow")
        elif seed.grad_ratio is not None and seed.grad_ratio > 0:
            grad_text.append(f"ratio={seed.grad_ratio:.2f}", style="green")
        else:
            grad_text.append("OK", style="green")
        lines.append(grad_text)

        # Epochs in stage
        lines.append(Text(f"Epochs in Stage: {seed.epochs_in_stage}", style="dim"))

        # Inter-slot interaction metrics
        if seed.interaction_sum != 0:
            interaction_style = "green" if seed.interaction_sum > 0 else "red"
            lines.append(Text(f"Interaction Sum: {seed.interaction_sum:+.2f}", style=interaction_style))
        if seed.boost_received > 0:
            lines.append(Text(f"Boost Received: {seed.boost_received:.2f}", style="cyan"))

        # Contribution velocity
        if seed.contribution_velocity != 0:
            velocity_style = "green" if seed.contribution_velocity > 0 else "yellow"
            velocity_label = "improving" if seed.contribution_velocity > 0 else "declining"
            lines.append(Text(f"Contribution Trend: {velocity_label} ({seed.contribution_velocity:+.3f})", style=velocity_style))

        return Panel(Group(*lines), title="State", border_style=stage_color)

    def _make_alpha_bar(self, alpha: float, width: int = 10) -> str:
        """Create a text-based progress bar for alpha."""
        filled = int(alpha * width)
        empty = width - filled
        return f"[{'#' * filled}{'-' * empty}]"
