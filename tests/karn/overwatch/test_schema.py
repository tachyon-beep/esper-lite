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


class TestEnvSummary:
    """Tests for EnvSummary dataclass."""

    def test_env_summary_creation(self) -> None:
        """EnvSummary can be created with required fields."""
        from esper.karn.overwatch.schema import EnvSummary, SlotChipState

        env = EnvSummary(
            env_id=0,
            device_id=0,
            status="OK",
        )

        assert env.env_id == 0
        assert env.device_id == 0
        assert env.status == "OK"
        assert env.slots == {}
        assert env.anomaly_score == 0.0
        assert env.anomaly_reasons == []

    def test_env_summary_with_slots(self) -> None:
        """EnvSummary contains slot states."""
        from esper.karn.overwatch.schema import EnvSummary, SlotChipState

        chip = SlotChipState(
            slot_id="r0c1",
            stage="TRAINING",
            blueprint_id="conv_light",
            alpha=0.3,
        )

        env = EnvSummary(
            env_id=3,
            device_id=1,
            status="WARN",
            slots={"r0c1": chip},
            anomaly_score=0.65,
            anomaly_reasons=["High gradient ratio (3.2x)"],
        )

        assert "r0c1" in env.slots
        assert env.slots["r0c1"].stage == "TRAINING"
        assert env.anomaly_score == 0.65

    def test_env_summary_to_dict(self) -> None:
        """EnvSummary serializes to dict with nested slots."""
        from esper.karn.overwatch.schema import EnvSummary, SlotChipState

        chip = SlotChipState(
            slot_id="r0c1",
            stage="BLENDING",
            blueprint_id="mlp",
            alpha=0.7,
        )

        env = EnvSummary(
            env_id=2,
            device_id=0,
            status="OK",
            throughput_fps=98.5,
            slots={"r0c1": chip},
        )

        d = env.to_dict()

        assert d["env_id"] == 2
        assert d["throughput_fps"] == 98.5
        assert "r0c1" in d["slots"]
        assert d["slots"]["r0c1"]["stage"] == "BLENDING"

    def test_env_summary_from_dict(self) -> None:
        """EnvSummary deserializes from dict."""
        from esper.karn.overwatch.schema import EnvSummary

        d = {
            "env_id": 1,
            "device_id": 0,
            "status": "CRIT",
            "throughput_fps": 45.0,
            "reward_last": -0.5,
            "slots": {
                "r0c0": {
                    "slot_id": "r0c0",
                    "stage": "CULLED",
                    "blueprint_id": "bad_seed",
                    "alpha": 0.0,
                }
            },
            "anomaly_score": 0.85,
            "anomaly_reasons": ["Throughput 55% below baseline", "Negative reward"],
        }

        env = EnvSummary.from_dict(d)

        assert env.env_id == 1
        assert env.status == "CRIT"
        assert env.anomaly_score == 0.85
        assert len(env.anomaly_reasons) == 2
        assert env.slots["r0c0"].stage == "CULLED"

    def test_env_summary_json_roundtrip(self) -> None:
        """EnvSummary survives JSON serialization."""
        from esper.karn.overwatch.schema import EnvSummary, SlotChipState

        env = EnvSummary(
            env_id=0,
            device_id=0,
            status="OK",
            slots={
                "r0c1": SlotChipState("r0c1", "TRAINING", "conv", 0.5)
            },
        )

        json_str = json.dumps(env.to_dict())
        restored = EnvSummary.from_dict(json.loads(json_str))

        assert restored.env_id == env.env_id
        assert restored.slots["r0c1"].alpha == 0.5
