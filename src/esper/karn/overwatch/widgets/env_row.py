"""Environment Row Widget.

Renders a single training environment in the Flight Board:
- Status indicator and env ID
- Device (GPU) assignment
- Throughput metrics
- Slot chips (inline or expanded)
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Static

from esper.karn.overwatch.schema import EnvSummary, SlotChipState
from esper.karn.overwatch.widgets.slot_chip import SlotChip, STAGE_SHORT


# Status styling
STATUS_ICONS = {
    "OK": "[ ]",
    "INFO": "[i]",
    "WARN": "[!]",
    "CRIT": "[‼]",
}

STATUS_COLORS = {
    "OK": "green",
    "INFO": "blue",
    "WARN": "yellow",
    "CRIT": "red",
}


class EnvRow(Container):
    """Widget displaying a single environment row.

    Compact format (single line):
        [ ] Env 0  gpu:0  OK     98 fps  [r0c1] TRAIN ████░░ 0.5α

    Expanded format (multiple lines):
        [▶] Env 0  gpu:0  OK     98 fps
            [r0c1] TRAINING
            Blueprint: conv_light
            Alpha: ████████░░ 0.78
            Gate: G2 ✓
    """

    DEFAULT_CSS = """
    EnvRow {
        width: 100%;
        height: auto;
        padding: 0;
    }

    EnvRow.selected {
        background: $primary-darken-1;
    }

    EnvRow.status-ok .env-status {
        color: $success;
    }

    EnvRow.status-warn .env-status {
        color: $warning;
    }

    EnvRow.status-crit .env-status {
        color: $error;
    }

    EnvRow .env-header {
        width: 100%;
        height: 1;
    }

    EnvRow .env-slots-expanded {
        padding-left: 4;
    }
    """

    def __init__(
        self,
        env: EnvSummary,
        selected: bool = False,
        expanded: bool = False,
        **kwargs,
    ) -> None:
        """Initialize the environment row.

        Args:
            env: EnvSummary data
            selected: Is this row currently selected
            expanded: Show expanded slot details
            **kwargs: Additional args for Container
        """
        super().__init__(**kwargs)
        self._env = env
        self._selected = selected
        self._expanded = expanded

        # Apply status class
        status_lower = env.status.lower()
        self.add_class(f"status-{status_lower}")

        if selected:
            self.add_class("selected")

    def render_header(self) -> str:
        """Render the header line."""
        e = self._env

        # Selection/status indicator
        if self._selected:
            indicator = "[▶]"
        else:
            indicator = STATUS_ICONS.get(e.status, "[ ]")

        # Format: [▶] Env 0  gpu:0  OK     98 fps
        status_styled = f"[{STATUS_COLORS.get(e.status, 'white')}]{e.status}[/]"

        throughput = f"{e.throughput_fps:.0f} fps" if e.throughput_fps > 0 else ""

        header = f"{indicator} Env {e.env_id}  gpu:{e.device_id}  {e.status:<4}  {throughput}"

        # Add inline slots if not expanded
        if not self._expanded and e.slots:
            slots_inline = self.render_slots_inline()
            header = f"{header}  {slots_inline}"

        return header

    def render_slots_inline(self) -> str:
        """Render slots inline (compact, single line)."""
        if not self._env.slots:
            return ""

        parts = []
        # Sort slots by slot_id for consistent display
        for slot_id in sorted(self._env.slots.keys()):
            slot = self._env.slots[slot_id]
            chip = SlotChip(slot)
            parts.append(chip.render_chip())

        return "  ".join(parts)

    def render_slots_expanded(self) -> list[str]:
        """Render slots expanded (multi-line)."""
        if not self._env.slots:
            return ["    (no slots)"]

        lines = []
        for slot_id in sorted(self._env.slots.keys()):
            slot = self._env.slots[slot_id]
            chip = SlotChip(slot, expanded=True)
            # Indent expanded slot content
            for line in chip.render_chip().split("\n"):
                lines.append(f"    {line}")

        return lines

    def compose(self) -> ComposeResult:
        """Compose the row layout."""
        yield Static(self.render_header(), classes="env-header env-status")

        if self._expanded:
            for line in self.render_slots_expanded():
                yield Static(line, classes="env-slots-expanded")

    def update_env(self, env: EnvSummary) -> None:
        """Update with new environment data."""
        self._env = env
        self.refresh()

    def set_selected(self, selected: bool) -> None:
        """Update selection state."""
        self._selected = selected
        if selected:
            self.add_class("selected")
        else:
            self.remove_class("selected")
        self.refresh()

    def set_expanded(self, expanded: bool) -> None:
        """Update expansion state."""
        self._expanded = expanded
        # Need to recompose to add/remove slot lines
        self.refresh(recompose=True)
