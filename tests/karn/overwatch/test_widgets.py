"""Tests for Overwatch TUI widgets."""

from __future__ import annotations

import pytest


class TestHelpOverlay:
    """Tests for HelpOverlay widget."""

    def test_help_overlay_imports(self) -> None:
        """HelpOverlay can be imported."""
        from esper.karn.overwatch.widgets.help import HelpOverlay

        assert HelpOverlay is not None

    def test_help_overlay_has_content(self) -> None:
        """HelpOverlay contains help content."""
        from esper.karn.overwatch.widgets.help import HelpOverlay

        widget = HelpOverlay()
        # Widget should have compose method for rendering
        assert callable(getattr(widget, "compose", None))


class TestSlotChip:
    """Tests for SlotChip widget."""

    def test_slot_chip_imports(self) -> None:
        """SlotChip can be imported."""
        from esper.karn.overwatch.widgets.slot_chip import SlotChip

        assert SlotChip is not None

    def test_slot_chip_renders_slot_id(self) -> None:
        """SlotChip displays slot ID in [r0c1] format."""
        from esper.karn.overwatch.widgets.slot_chip import SlotChip
        from esper.karn.overwatch.schema import SlotChipState

        state = SlotChipState(
            slot_id="r0c1",
            stage="TRAINING",
            blueprint_id="conv_light",
            alpha=0.5,
        )
        chip = SlotChip(state)

        rendered = chip.render_chip()
        assert "[r0c1]" in rendered

    def test_slot_chip_renders_stage(self) -> None:
        """SlotChip displays stage name."""
        from esper.karn.overwatch.widgets.slot_chip import SlotChip
        from esper.karn.overwatch.schema import SlotChipState

        state = SlotChipState(
            slot_id="r0c1",
            stage="BLENDING",
            blueprint_id="mlp",
            alpha=0.7,
        )
        chip = SlotChip(state)

        rendered = chip.render_chip()
        # Compact view shows shortened stage name
        assert "BLEND" in rendered

    def test_slot_chip_renders_alpha_bar(self) -> None:
        """SlotChip displays alpha progress bar."""
        from esper.karn.overwatch.widgets.slot_chip import SlotChip
        from esper.karn.overwatch.schema import SlotChipState

        state = SlotChipState(
            slot_id="r0c1",
            stage="BLENDING",
            blueprint_id="mlp",
            alpha=0.5,  # 50%
        )
        chip = SlotChip(state)

        rendered = chip.render_chip()
        # Should have some filled and some empty
        assert "█" in rendered or "▓" in rendered
        assert "░" in rendered or "▒" in rendered

    def test_slot_chip_renders_gate_status(self) -> None:
        """SlotChip displays gate status when present."""
        from esper.karn.overwatch.widgets.slot_chip import SlotChip
        from esper.karn.overwatch.schema import SlotChipState

        state = SlotChipState(
            slot_id="r0c1",
            stage="BLENDING",
            blueprint_id="mlp",
            alpha=0.7,
            gate_last="G2",
            gate_passed=True,
        )
        chip = SlotChip(state)

        rendered = chip.render_chip()
        assert "G2" in rendered
        assert "✓" in rendered or "✔" in rendered

    def test_slot_chip_gate_failed_indicator(self) -> None:
        """SlotChip shows failure indicator for failed gate."""
        from esper.karn.overwatch.widgets.slot_chip import SlotChip
        from esper.karn.overwatch.schema import SlotChipState

        state = SlotChipState(
            slot_id="r0c0",
            stage="PROBATIONARY",
            blueprint_id="bad_seed",
            alpha=0.3,
            gate_last="G1",
            gate_passed=False,
        )
        chip = SlotChip(state)

        rendered = chip.render_chip()
        assert "G1" in rendered
        assert "✗" in rendered or "✘" in rendered or "×" in rendered
