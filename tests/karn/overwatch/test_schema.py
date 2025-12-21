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
        from esper.karn.overwatch.schema import EnvSummary

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
                    "stage": "PRUNED",
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
        assert env.slots["r0c0"].stage == "PRUNED"

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


class TestTamiyoState:
    """Tests for TamiyoState dataclass."""

    def test_tamiyo_state_creation(self) -> None:
        """TamiyoState can be created with defaults."""
        from esper.karn.overwatch.schema import TamiyoState

        state = TamiyoState()

        assert state.action_counts == {}
        assert state.recent_actions == []
        assert state.confidence_mean == 0.0
        assert state.exploration_pct == 0.0

    def test_tamiyo_state_with_data(self) -> None:
        """TamiyoState stores action distribution and PPO vitals."""
        from esper.karn.overwatch.schema import TamiyoState

        state = TamiyoState(
            action_counts={"GERMINATE": 34, "BLEND": 28, "PRUNE": 12, "WAIT": 26},
            recent_actions=["G", "B", "B", "W", "G"],
            confidence_mean=0.73,
            confidence_min=0.42,
            confidence_max=0.94,
            exploration_pct=31.0,
            kl_divergence=0.019,
            entropy=1.24,
            clip_fraction=0.048,
            explained_variance=0.42,
            learning_rate=3e-4,
        )

        assert state.action_counts["GERMINATE"] == 34
        assert len(state.recent_actions) == 5
        assert state.kl_divergence == 0.019

    def test_tamiyo_state_to_dict(self) -> None:
        """TamiyoState serializes to dict."""
        from esper.karn.overwatch.schema import TamiyoState

        state = TamiyoState(
            action_counts={"GERMINATE": 10},
            recent_actions=["G", "G"],
            confidence_mean=0.8,
            kl_divergence=0.02,
        )

        d = state.to_dict()

        assert d["action_counts"] == {"GERMINATE": 10}
        assert d["recent_actions"] == ["G", "G"]
        assert d["kl_divergence"] == 0.02

    def test_tamiyo_state_from_dict(self) -> None:
        """TamiyoState deserializes from dict."""
        from esper.karn.overwatch.schema import TamiyoState

        d = {
            "action_counts": {"BLEND": 5, "PRUNE": 3},
            "recent_actions": ["B", "C"],
            "confidence_mean": 0.65,
            "confidence_min": 0.4,
            "confidence_max": 0.9,
            "exploration_pct": 25.0,
            "kl_divergence": 0.015,
            "entropy": 1.5,
            "clip_fraction": 0.03,
            "explained_variance": 0.5,
            "learning_rate": 0.0003,
        }

        state = TamiyoState.from_dict(d)

        assert state.action_counts["BLEND"] == 5
        assert state.entropy == 1.5

    def test_tamiyo_state_json_roundtrip(self) -> None:
        """TamiyoState survives JSON serialization."""
        from esper.karn.overwatch.schema import TamiyoState

        state = TamiyoState(
            action_counts={"WAIT": 100},
            kl_divergence=0.01,
        )

        json_str = json.dumps(state.to_dict())
        restored = TamiyoState.from_dict(json.loads(json_str))

        assert restored.action_counts == state.action_counts


class TestConnectionStatus:
    """Tests for ConnectionStatus dataclass."""

    def test_connection_status_live(self) -> None:
        """ConnectionStatus reports Live when fresh."""
        from esper.karn.overwatch.schema import ConnectionStatus

        status = ConnectionStatus(
            connected=True,
            last_event_ts=1000.0,
            staleness_s=0.5,
        )

        assert status.connected is True
        assert "Live" in status.display_text

    def test_connection_status_stale(self) -> None:
        """ConnectionStatus reports Stale when old."""
        from esper.karn.overwatch.schema import ConnectionStatus

        status = ConnectionStatus(
            connected=True,
            last_event_ts=1000.0,
            staleness_s=8.0,
        )

        assert "Stale" in status.display_text
        assert "8" in status.display_text

    def test_connection_status_disconnected(self) -> None:
        """ConnectionStatus reports Disconnected when not connected."""
        from esper.karn.overwatch.schema import ConnectionStatus

        status = ConnectionStatus(
            connected=False,
            last_event_ts=0.0,
            staleness_s=30.0,
        )

        assert "Disconnected" in status.display_text

    def test_connection_status_json_roundtrip(self) -> None:
        """ConnectionStatus survives JSON serialization."""
        from esper.karn.overwatch.schema import ConnectionStatus

        status = ConnectionStatus(True, 1234.5, 2.0)

        json_str = json.dumps(status.to_dict())
        restored = ConnectionStatus.from_dict(json.loads(json_str))

        assert restored.connected == status.connected
        assert restored.staleness_s == status.staleness_s


class TestDeviceVitals:
    """Tests for DeviceVitals dataclass."""

    def test_device_vitals_creation(self) -> None:
        """DeviceVitals stores GPU metrics."""
        from esper.karn.overwatch.schema import DeviceVitals

        vitals = DeviceVitals(
            device_id=0,
            name="GPU 0",
            utilization_pct=94.0,
            memory_used_gb=11.2,
            memory_total_gb=12.0,
            temperature_c=72,
        )

        assert vitals.device_id == 0
        assert vitals.utilization_pct == 94.0
        assert vitals.memory_pct == pytest.approx(93.33, rel=0.01)

    def test_device_vitals_json_roundtrip(self) -> None:
        """DeviceVitals survives JSON serialization."""
        from esper.karn.overwatch.schema import DeviceVitals

        vitals = DeviceVitals(0, "GPU 0", 90.0, 10.0, 12.0, 70)

        json_str = json.dumps(vitals.to_dict())
        restored = DeviceVitals.from_dict(json.loads(json_str))

        assert restored.device_id == vitals.device_id
        assert restored.memory_used_gb == vitals.memory_used_gb


class TestTuiSnapshot:
    """Tests for TuiSnapshot dataclass (root schema)."""

    def test_tui_snapshot_creation(self) -> None:
        """TuiSnapshot can be created with minimal fields."""
        from esper.karn.overwatch.schema import (
            TuiSnapshot,
            ConnectionStatus,
            TamiyoState,
        )

        snap = TuiSnapshot(
            schema_version=1,
            captured_at="2025-12-18T12:00:00Z",
            connection=ConnectionStatus(True, 1000.0, 0.5),
            tamiyo=TamiyoState(),
        )

        assert snap.schema_version == 1
        assert snap.connection.connected is True
        assert snap.flight_board == []

    def test_tui_snapshot_with_envs(self) -> None:
        """TuiSnapshot contains flight board envs."""
        from esper.karn.overwatch.schema import (
            TuiSnapshot,
            ConnectionStatus,
            TamiyoState,
            EnvSummary,
            SlotChipState,
        )

        envs = [
            EnvSummary(
                env_id=0,
                device_id=0,
                status="OK",
                slots={"r0c1": SlotChipState("r0c1", "TRAINING", "conv", 0.5)},
            ),
            EnvSummary(
                env_id=1,
                device_id=1,
                status="WARN",
                anomaly_score=0.6,
            ),
        ]

        snap = TuiSnapshot(
            schema_version=1,
            captured_at="2025-12-18T12:00:00Z",
            connection=ConnectionStatus(True, 1000.0, 0.5),
            tamiyo=TamiyoState(),
            flight_board=envs,
        )

        assert len(snap.flight_board) == 2
        assert snap.flight_board[0].slots["r0c1"].stage == "TRAINING"

    def test_tui_snapshot_to_dict(self) -> None:
        """TuiSnapshot serializes completely."""
        from esper.karn.overwatch.schema import (
            TuiSnapshot,
            ConnectionStatus,
            TamiyoState,
            EnvSummary,
            DeviceVitals,
        )

        snap = TuiSnapshot(
            schema_version=1,
            captured_at="2025-12-18T12:00:00Z",
            connection=ConnectionStatus(True, 1000.0, 0.5),
            tamiyo=TamiyoState(kl_divergence=0.02),
            run_id="run-123",
            task_name="cifar10",
            episode=47,
            batch=1203,
            best_metric=82.1,
            runtime_s=8040.0,
            devices=[DeviceVitals(0, "GPU 0", 94.0, 11.2, 12.0, 72)],
            flight_board=[EnvSummary(0, 0, "OK")],
            envs_ok=4,
            envs_warn=0,
            envs_crit=0,
        )

        d = snap.to_dict()

        assert d["schema_version"] == 1
        assert d["tamiyo"]["kl_divergence"] == 0.02
        assert len(d["devices"]) == 1
        assert d["devices"][0]["utilization_pct"] == 94.0

    def test_tui_snapshot_from_dict(self) -> None:
        """TuiSnapshot deserializes completely."""
        from esper.karn.overwatch.schema import TuiSnapshot

        d = {
            "schema_version": 1,
            "captured_at": "2025-12-18T12:00:00Z",
            "connection": {"connected": True, "last_event_ts": 1000.0, "staleness_s": 1.0},
            "tamiyo": {"kl_divergence": 0.015, "action_counts": {"BLEND": 5}},
            "run_id": "test-run",
            "task_name": "mnist",
            "episode": 10,
            "batch": 500,
            "best_metric": 95.0,
            "runtime_s": 600.0,
            "devices": [
                {"device_id": 0, "name": "GPU 0", "utilization_pct": 80.0,
                 "memory_used_gb": 8.0, "memory_total_gb": 12.0}
            ],
            "flight_board": [
                {"env_id": 0, "device_id": 0, "status": "OK", "slots": {}}
            ],
            "event_feed": [],
            "envs_ok": 1,
            "envs_warn": 0,
            "envs_crit": 0,
        }

        snap = TuiSnapshot.from_dict(d)

        assert snap.schema_version == 1
        assert snap.tamiyo.kl_divergence == 0.015
        assert snap.tamiyo.action_counts["BLEND"] == 5
        assert len(snap.devices) == 1
        assert snap.devices[0].name == "GPU 0"

    def test_tui_snapshot_json_roundtrip(self) -> None:
        """TuiSnapshot survives full JSON serialization cycle."""
        from esper.karn.overwatch.schema import (
            TuiSnapshot,
            ConnectionStatus,
            TamiyoState,
            EnvSummary,
            SlotChipState,
            DeviceVitals,
        )

        original = TuiSnapshot(
            schema_version=1,
            captured_at="2025-12-18T12:00:00Z",
            connection=ConnectionStatus(True, 1000.0, 0.5),
            tamiyo=TamiyoState(
                action_counts={"GERMINATE": 10, "BLEND": 20},
                kl_divergence=0.019,
            ),
            devices=[DeviceVitals(0, "GPU 0", 90.0, 10.0, 12.0, 68)],
            flight_board=[
                EnvSummary(
                    env_id=0,
                    device_id=0,
                    status="OK",
                    slots={"r0c1": SlotChipState("r0c1", "BLENDING", "conv", 0.7)},
                )
            ],
        )

        json_str = json.dumps(original.to_dict())
        restored = TuiSnapshot.from_dict(json.loads(json_str))

        assert restored.schema_version == original.schema_version
        assert restored.tamiyo.kl_divergence == original.tamiyo.kl_divergence
        assert restored.flight_board[0].slots["r0c1"].alpha == 0.7


class TestPackageExports:
    """Tests that public API is exported correctly."""

    def test_all_schemas_importable_from_package(self) -> None:
        """All schema classes are importable from overwatch package."""
        from esper.karn.overwatch import (
            TuiSnapshot,
            EnvSummary,
            SlotChipState,
            TamiyoState,
            ConnectionStatus,
            DeviceVitals,
            FeedEvent,
        )

        # Just verify imports work
        assert TuiSnapshot is not None
        assert EnvSummary is not None
        assert SlotChipState is not None
        assert TamiyoState is not None
        assert ConnectionStatus is not None
        assert DeviceVitals is not None
        assert FeedEvent is not None
