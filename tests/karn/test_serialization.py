"""Tests for Karn event serialization."""

import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import pytest
from esper.karn.serialization import serialize_event, _payload_to_dict
from esper.karn.contracts import TelemetryEventLike
from esper.leyline.telemetry import (
    EpochCompletedPayload,
    TrainingStartedPayload,
    PPOUpdatePayload,
)


class MockEventType(Enum):
    TEST_EVENT = "test"


class MockEvent:
    """Mock event implementing TelemetryEventLike protocol."""

    def __init__(
        self,
        event_type: MockEventType | str = MockEventType.TEST_EVENT,
        timestamp: datetime | None = None,
    ):
        self._event_type = event_type
        self._timestamp = timestamp or datetime(2025, 1, 1, 12, 0, 0)

    @property
    def event_type(self) -> MockEventType | str:
        return self._event_type

    @property
    def timestamp(self) -> datetime:
        return self._timestamp

    @property
    def data(self) -> dict | None:
        return {"test": "value"}

    @property
    def epoch(self) -> int | None:
        return 5

    @property
    def seed_id(self) -> str | None:
        return "seed_123"

    @property
    def slot_id(self) -> str | None:
        return "r0c0"

    @property
    def severity(self) -> str | None:
        return "info"

    @property
    def message(self) -> str | None:
        return "test message"


class TestSerializeEvent:
    """Test event serialization."""

    def test_serializes_enum_event_type(self) -> None:
        """Enum event_type should be converted to string name."""
        event = MockEvent(event_type=MockEventType.TEST_EVENT)
        result = serialize_event(event)
        data = json.loads(result)
        assert data["event_type"] == "TEST_EVENT"

    def test_serializes_string_event_type(self) -> None:
        """String event_type should pass through."""
        event = MockEvent(event_type="STRING_TYPE")
        result = serialize_event(event)
        data = json.loads(result)
        assert data["event_type"] == "STRING_TYPE"

    def test_serializes_datetime(self) -> None:
        """Datetime should be ISO formatted."""
        ts = datetime(2025, 6, 15, 10, 30, 0)
        event = MockEvent(timestamp=ts)
        result = serialize_event(event)
        data = json.loads(result)
        assert data["timestamp"] == "2025-06-15T10:30:00"

    def test_includes_all_fields(self) -> None:
        """All TelemetryEventLike fields should be present."""
        event = MockEvent()
        result = serialize_event(event)
        data = json.loads(result)

        assert "event_type" in data
        assert "timestamp" in data
        assert "data" in data
        assert "epoch" in data
        assert "seed_id" in data
        assert "slot_id" in data
        assert "severity" in data
        assert "message" in data

    def test_returns_valid_json(self) -> None:
        """Result should be valid JSON string."""
        event = MockEvent()
        result = serialize_event(event)
        # Should not raise
        parsed = json.loads(result)
        assert isinstance(parsed, dict)


class MockEventWithTypedPayload:
    """Mock event with a typed dataclass payload."""

    def __init__(self, payload: object):
        self._payload = payload

    @property
    def event_type(self) -> MockEventType:
        return MockEventType.TEST_EVENT

    @property
    def timestamp(self) -> datetime:
        return datetime(2025, 1, 1, 12, 0, 0)

    @property
    def data(self) -> object:
        return self._payload

    @property
    def epoch(self) -> int | None:
        return 5

    @property
    def seed_id(self) -> str | None:
        return None

    @property
    def slot_id(self) -> str | None:
        return None

    @property
    def severity(self) -> str | None:
        return "info"

    @property
    def message(self) -> str | None:
        return None


class TestTypedPayloadSerialization:
    """Test serialization of typed dataclass payloads."""

    def test_epoch_completed_payload_serializes_to_dict(self) -> None:
        """EpochCompletedPayload should serialize to a JSON object, not a string."""
        payload = EpochCompletedPayload(
            env_id=0,
            val_accuracy=0.85,
            val_loss=0.15,
            inner_epoch=10,
            train_loss=0.12,
            train_accuracy=0.90,
        )
        event = MockEventWithTypedPayload(payload)
        result = serialize_event(event)
        data = json.loads(result)

        # The payload should be a dict, not a string
        assert isinstance(data["data"], dict), (
            f"Expected data to be dict, got {type(data['data']).__name__}: {data['data']!r}"
        )
        assert data["data"]["env_id"] == 0
        assert data["data"]["val_accuracy"] == 0.85
        assert data["data"]["val_loss"] == 0.15
        assert data["data"]["inner_epoch"] == 10
        assert data["data"]["train_loss"] == 0.12
        assert data["data"]["train_accuracy"] == 0.90

    def test_training_started_payload_serializes_to_dict(self) -> None:
        """TrainingStartedPayload should serialize to a JSON object."""
        payload = TrainingStartedPayload(
            n_envs=4,
            max_epochs=100,
            task="test_task",
            host_params=1000000,
            slot_ids=("r0c0", "r0c1"),
            seed=42,
            n_episodes=50,
            lr=0.0003,
            clip_ratio=0.2,
            entropy_coef=0.01,
            param_budget=100000,
            policy_device="cuda:0",
            env_devices=("cuda:0", "cuda:1"),
        )
        event = MockEventWithTypedPayload(payload)
        result = serialize_event(event)
        data = json.loads(result)

        assert isinstance(data["data"], dict)
        assert data["data"]["n_envs"] == 4
        assert data["data"]["task"] == "test_task"
        # Tuples become lists in JSON
        assert data["data"]["slot_ids"] == ["r0c0", "r0c1"]
        assert data["data"]["env_devices"] == ["cuda:0", "cuda:1"]

    def test_ppo_update_payload_serializes_to_dict(self) -> None:
        """PPOUpdatePayload should serialize to a JSON object."""
        payload = PPOUpdatePayload(
            policy_loss=0.5,
            value_loss=0.3,
            entropy=0.1,
            grad_norm=1.5,
            kl_divergence=0.02,
            clip_fraction=0.05,
            nan_grad_count=0,
            explained_variance=0.8,
            lr=0.0003,
        )
        event = MockEventWithTypedPayload(payload)
        result = serialize_event(event)
        data = json.loads(result)

        assert isinstance(data["data"], dict)
        assert data["data"]["policy_loss"] == 0.5
        assert data["data"]["grad_norm"] == 1.5
        assert data["data"]["explained_variance"] == 0.8

    def test_none_payload_serializes_to_null(self) -> None:
        """None payload should serialize to null."""
        event = MockEventWithTypedPayload(None)
        result = serialize_event(event)
        data = json.loads(result)
        assert data["data"] is None

    def test_dict_payload_passes_through(self) -> None:
        """Dict payloads should pass through unchanged."""
        event = MockEventWithTypedPayload({"legacy": "data", "count": 42})
        result = serialize_event(event)
        data = json.loads(result)
        assert data["data"] == {"legacy": "data", "count": 42}


class TestPayloadToDict:
    """Test the _payload_to_dict helper function."""

    def test_dataclass_converts_to_dict(self) -> None:
        """Dataclass instances should be converted to dicts."""
        payload = EpochCompletedPayload(
            env_id=0,
            val_accuracy=0.5,
            val_loss=0.5,
            inner_epoch=1,
        )
        result = _payload_to_dict(payload)
        assert isinstance(result, dict)
        assert result["env_id"] == 0

    def test_none_returns_none(self) -> None:
        """None should return None."""
        assert _payload_to_dict(None) is None

    def test_dict_passes_through(self) -> None:
        """Dicts should pass through unchanged."""
        original = {"key": "value"}
        assert _payload_to_dict(original) is original

    def test_nested_enum_converted_to_name(self) -> None:
        """Enums nested in dataclasses should be converted to names."""

        class TestEnum(Enum):
            VALUE_A = "a"
            VALUE_B = "b"

        @dataclass(frozen=True)
        class PayloadWithEnum:
            status: TestEnum
            count: int

        payload = PayloadWithEnum(status=TestEnum.VALUE_A, count=5)
        result = _payload_to_dict(payload)
        assert result["status"] == "VALUE_A"
        assert result["count"] == 5
