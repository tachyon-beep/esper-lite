"""Tests for Karn event serialization."""

import json
from datetime import datetime
from enum import Enum

import pytest
from esper.karn.serialization import serialize_event
from esper.karn.contracts import TelemetryEventLike


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
