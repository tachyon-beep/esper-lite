"""Tests for Overwatch TUI snapshot schema."""

from __future__ import annotations

import json
import pytest


class TestSlotChipState:
    """Tests for SlotChipState dataclass."""

    def test_slot_chip_state_creation(self) -> None:
        """SlotChipState can be created with required fields."""
        from esper.karn.overwatch.schema import SlotChipState

        chip = SlotChipState(
            slot_id="r0c1",
            stage="TRAINING",
            blueprint_id="conv_light",
            alpha=0.45,
        )

        assert chip.slot_id == "r0c1"
        assert chip.stage == "TRAINING"
        assert chip.blueprint_id == "conv_light"
        assert chip.alpha == 0.45
        # Defaults
        assert chip.epochs_in_stage == 0
        assert chip.epochs_total == 0
        assert chip.gate_last is None
        assert chip.gate_passed is None

    def test_slot_chip_state_to_dict(self) -> None:
        """SlotChipState serializes to dict."""
        from esper.karn.overwatch.schema import SlotChipState

        chip = SlotChipState(
            slot_id="r0c1",
            stage="BLENDING",
            blueprint_id="mlp_narrow",
            alpha=0.78,
            epochs_in_stage=5,
            epochs_total=10,
            gate_last="G2",
            gate_passed=True,
        )

        d = chip.to_dict()

        assert d["slot_id"] == "r0c1"
        assert d["stage"] == "BLENDING"
        assert d["alpha"] == 0.78
        assert d["gate_last"] == "G2"
        assert d["gate_passed"] is True

    def test_slot_chip_state_from_dict(self) -> None:
        """SlotChipState deserializes from dict."""
        from esper.karn.overwatch.schema import SlotChipState

        d = {
            "slot_id": "r1c0",
            "stage": "FOSSILIZED",
            "blueprint_id": "conv_light",
            "alpha": 1.0,
            "epochs_in_stage": 0,
            "epochs_total": 42,
            "gate_last": "G3",
            "gate_passed": True,
        }

        chip = SlotChipState.from_dict(d)

        assert chip.slot_id == "r1c0"
        assert chip.stage == "FOSSILIZED"
        assert chip.alpha == 1.0

    def test_slot_chip_state_json_roundtrip(self) -> None:
        """SlotChipState survives JSON serialization."""
        from esper.karn.overwatch.schema import SlotChipState

        chip = SlotChipState(
            slot_id="r0c0",
            stage="GERMINATED",
            blueprint_id="test",
            alpha=0.1,
        )

        json_str = json.dumps(chip.to_dict())
        restored = SlotChipState.from_dict(json.loads(json_str))

        assert restored.slot_id == chip.slot_id
        assert restored.stage == chip.stage
        assert restored.alpha == chip.alpha
