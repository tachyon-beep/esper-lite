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


class TestEnvRow:
    """Tests for EnvRow widget."""

    def test_env_row_imports(self) -> None:
        """EnvRow can be imported."""
        from esper.karn.overwatch.widgets.env_row import EnvRow

        assert EnvRow is not None

    def test_env_row_renders_env_id(self) -> None:
        """EnvRow displays environment ID."""
        from esper.karn.overwatch.widgets.env_row import EnvRow
        from esper.karn.overwatch.schema import EnvSummary

        env = EnvSummary(
            env_id=3,
            device_id=1,
            status="OK",
        )
        row = EnvRow(env)

        rendered = row.render_header()
        assert "Env 3" in rendered or "env:3" in rendered.lower()

    def test_env_row_renders_device_id(self) -> None:
        """EnvRow displays device/GPU ID."""
        from esper.karn.overwatch.widgets.env_row import EnvRow
        from esper.karn.overwatch.schema import EnvSummary

        env = EnvSummary(
            env_id=0,
            device_id=2,
            status="OK",
        )
        row = EnvRow(env)

        rendered = row.render_header()
        assert "gpu:2" in rendered.lower() or "GPU 2" in rendered

    def test_env_row_renders_status_ok(self) -> None:
        """EnvRow displays OK status with appropriate styling."""
        from esper.karn.overwatch.widgets.env_row import EnvRow
        from esper.karn.overwatch.schema import EnvSummary

        env = EnvSummary(
            env_id=0,
            device_id=0,
            status="OK",
        )
        row = EnvRow(env)

        rendered = row.render_header()
        assert "OK" in rendered

    def test_env_row_renders_status_warn(self) -> None:
        """EnvRow displays WARN status."""
        from esper.karn.overwatch.widgets.env_row import EnvRow
        from esper.karn.overwatch.schema import EnvSummary

        env = EnvSummary(
            env_id=1,
            device_id=0,
            status="WARN",
            anomaly_score=0.65,
        )
        row = EnvRow(env)

        rendered = row.render_header()
        assert "WARN" in rendered

    def test_env_row_renders_status_crit(self) -> None:
        """EnvRow displays CRIT status."""
        from esper.karn.overwatch.widgets.env_row import EnvRow
        from esper.karn.overwatch.schema import EnvSummary

        env = EnvSummary(
            env_id=2,
            device_id=1,
            status="CRIT",
            anomaly_score=0.85,
        )
        row = EnvRow(env)

        rendered = row.render_header()
        assert "CRIT" in rendered

    def test_env_row_renders_throughput(self) -> None:
        """EnvRow displays throughput."""
        from esper.karn.overwatch.widgets.env_row import EnvRow
        from esper.karn.overwatch.schema import EnvSummary

        env = EnvSummary(
            env_id=0,
            device_id=0,
            status="OK",
            throughput_fps=98.5,
        )
        row = EnvRow(env)

        rendered = row.render_header()
        assert "98" in rendered or "fps" in rendered.lower()

    def test_env_row_renders_slots_inline(self) -> None:
        """EnvRow renders slot chips inline."""
        from esper.karn.overwatch.widgets.env_row import EnvRow
        from esper.karn.overwatch.schema import EnvSummary, SlotChipState

        env = EnvSummary(
            env_id=0,
            device_id=0,
            status="OK",
            slots={
                "r0c1": SlotChipState("r0c1", "TRAINING", "conv", 0.5),
            },
        )
        row = EnvRow(env)

        content = row.render_slots_inline()
        assert "[r0c1]" in content
        assert "TRAIN" in content

    def test_env_row_focus_indicator(self) -> None:
        """EnvRow shows focus indicator when selected."""
        from esper.karn.overwatch.widgets.env_row import EnvRow
        from esper.karn.overwatch.schema import EnvSummary

        env = EnvSummary(env_id=0, device_id=0, status="OK")
        row = EnvRow(env, selected=True)

        # Should have some visual indicator
        header = row.render_header()
        # Either [!] prefix or special character
        assert "[" in header or "▶" in header or "●" in header


class TestWidgetExports:
    """Tests for widget package exports."""

    def test_all_widgets_importable(self) -> None:
        """All widgets are importable from package."""
        from esper.karn.overwatch.widgets import (
            HelpOverlay,
            SlotChip,
            EnvRow,
            FlightBoard,
        )

        assert HelpOverlay is not None
        assert SlotChip is not None
        assert EnvRow is not None
        assert FlightBoard is not None
