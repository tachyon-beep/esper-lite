"""Slot Chip Widget.

Renders a single slot's state as a compact chip showing:
- Slot ID: [r0c1]
- Stage name: TRAINING, BLENDING, etc.
- Alpha progress bar: ████░░ 0.7α
- Gate status: G2✓ or G1✗
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.widgets import Static

from esper.karn.overwatch.schema import SlotChipState


# Stage colors (CSS class names)
STAGE_COLORS = {
    "DORMANT": "stage-dormant",
    "GERMINATED": "stage-germinated",
    "TRAINING": "stage-training",
    "BLENDING": "stage-blending",
    "PROBATIONARY": "stage-probationary",
    "FOSSILIZED": "stage-fossilized",
    "CULLED": "stage-culled",
    "EMBARGOED": "stage-embargoed",
    "RESETTING": "stage-resetting",
}

# Compact stage names for display
STAGE_SHORT = {
    "DORMANT": "DORM",
    "GERMINATED": "GERM",
    "TRAINING": "TRAIN",
    "BLENDING": "BLEND",
    "PROBATIONARY": "PROB",
    "FOSSILIZED": "FOSSIL",
    "CULLED": "CULL",
    "EMBARGOED": "EMBAR",
    "RESETTING": "RESET",
}


class SlotChip(Static):
    """Widget displaying a single slot's state.

    Compact format:
        [r0c1] BLEND ████░░ 0.7α G2✓

    Expanded format (when env is expanded):
        [r0c1] BLENDING
        Blueprint: conv_light
        Alpha: ████████░░ 0.78
        Gate: G2 ✓ (epoch 15/20)
    """

    DEFAULT_CSS = """
    SlotChip {
        width: auto;
        height: 1;
        padding: 0 1;
    }

    SlotChip.stage-dormant {
        color: #666666;
    }

    SlotChip.stage-germinated {
        color: #98c379;
    }

    SlotChip.stage-training {
        color: #61afef;
    }

    SlotChip.stage-blending {
        color: #c678dd;
    }

    SlotChip.stage-probationary {
        color: #e5c07b;
    }

    SlotChip.stage-fossilized {
        color: #56b6c2;
    }

    SlotChip.stage-culled {
        color: #e06c75;
    }

    SlotChip.stage-embargoed {
        color: #be5046;
    }

    SlotChip.stage-resetting {
        color: #d19a66;
    }
    """

    def __init__(
        self,
        state: SlotChipState,
        expanded: bool = False,
        **kwargs,
    ) -> None:
        """Initialize the slot chip.

        Args:
            state: SlotChipState data
            expanded: Show expanded view with more details
            **kwargs: Additional args for Static
        """
        super().__init__(**kwargs)
        self._state = state
        self._expanded = expanded

        # Apply stage color class
        stage_class = STAGE_COLORS.get(state.stage, "")
        if stage_class:
            self.add_class(stage_class)

    def render_chip(self) -> str:
        """Render the chip content."""
        if self._expanded:
            return self._render_expanded()
        return self._render_compact()

    def _render_compact(self) -> str:
        """Render compact single-line format."""
        s = self._state

        # Slot ID
        slot_id = f"[{s.slot_id}]"

        # Stage (shortened)
        stage = STAGE_SHORT.get(s.stage, s.stage[:5])

        # Alpha bar (6 chars wide)
        alpha_bar = self._render_alpha_bar(s.alpha, width=6)
        alpha_val = f"{s.alpha:.1f}α" if s.alpha < 1.0 else "1.0α"

        # Gate status
        gate = ""
        if s.gate_last:
            gate_icon = "✓" if s.gate_passed else "✗"
            gate = f" {s.gate_last}{gate_icon}"

        return f"{slot_id} {stage} {alpha_bar} {alpha_val}{gate}"

    def _render_expanded(self) -> str:
        """Render expanded multi-line format."""
        s = self._state
        lines = [
            f"[{s.slot_id}] {s.stage}",
            f"  Blueprint: {s.blueprint_id}",
            f"  Alpha: {self._render_alpha_bar(s.alpha, width=10)} {s.alpha:.2f}",
        ]

        if s.gate_last:
            gate_icon = "✓" if s.gate_passed else "✗"
            lines.append(f"  Gate: {s.gate_last} {gate_icon} (epoch {s.epochs_in_stage}/{s.epochs_total})")

        return "\n".join(lines)

    def _render_alpha_bar(self, alpha: float, width: int = 6) -> str:
        """Render alpha as a progress bar.

        Args:
            alpha: Value 0.0-1.0
            width: Total bar width in characters

        Returns:
            Progress bar string like "████░░"
        """
        filled = int(alpha * width)
        empty = width - filled
        return "█" * filled + "░" * empty

    def compose(self) -> ComposeResult:
        """Compose is not used - we render directly."""
        yield from []

    def render(self) -> str:
        """Render the widget content."""
        return self.render_chip()

    def update_state(self, state: SlotChipState) -> None:
        """Update with new state."""
        self._state = state
        self.refresh()
