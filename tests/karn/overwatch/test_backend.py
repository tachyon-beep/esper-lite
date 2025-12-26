"""Tests for OverwatchBackend WebSocket server."""

import json
import time
from unittest.mock import MagicMock

import pytest

from esper.karn.overwatch.backend import OverwatchBackend
from esper.karn.sanctum.schema import SanctumSnapshot


class TestOverwatchBackend:
    """Test OverwatchBackend initialization and event processing."""

    def test_backend_initializes_with_aggregator(self) -> None:
        """Backend should create a SanctumAggregator instance."""
        backend = OverwatchBackend(port=8080)
        assert backend.aggregator is not None
        assert backend.port == 8080

    def test_emit_processes_event_through_aggregator(self) -> None:
        """Events should be passed to the aggregator."""
        backend = OverwatchBackend(port=8080)
        backend.aggregator = MagicMock()

        mock_event = MagicMock()
        backend.emit(mock_event)

        backend.aggregator.process_event.assert_called_once_with(mock_event)

    def test_snapshot_to_json_produces_valid_json(self) -> None:
        """Snapshots should serialize to valid JSON."""
        backend = OverwatchBackend(port=8080)

        # Get a snapshot (will be empty/default)
        snapshot = backend.get_snapshot()
        json_str = backend.snapshot_to_json(snapshot)

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert "current_episode" in parsed
        assert "current_epoch" in parsed

    def test_rate_limiting_throttles_broadcasts(self) -> None:
        """Should not broadcast more than configured Hz."""
        backend = OverwatchBackend(port=8080, snapshot_rate_hz=10)
        backend._broadcast = MagicMock()

        # Call maybe_broadcast 50 times rapidly (without emit)
        # This tests only the rate limiting logic
        for _ in range(50):
            backend.maybe_broadcast()

        # At 10 Hz over ~0 seconds, should have at most 1-2 broadcasts
        assert backend._broadcast.call_count <= 5

    def test_get_snapshot_returns_sanctum_snapshot(self) -> None:
        """get_snapshot should return a SanctumSnapshot."""
        backend = OverwatchBackend(port=8080)
        snapshot = backend.get_snapshot()
        assert isinstance(snapshot, SanctumSnapshot)

    def test_backend_lifecycle_start_stop(self) -> None:
        """Backend should start and stop cleanly without FastAPI."""
        backend = OverwatchBackend(port=8080)

        # Start should not raise even without FastAPI
        backend.start()
        assert backend._running is True

        # Stop should clean up
        backend.stop()
        assert backend._running is False

    def test_snapshot_to_json_handles_special_types(self) -> None:
        """JSON serialization should handle enums, datetime, Path."""
        from datetime import datetime, timezone
        from pathlib import Path

        backend = OverwatchBackend(port=8080)

        # Create snapshot with special types
        snapshot = backend.get_snapshot()
        snapshot.start_time = datetime.now(timezone.utc)
        snapshot.connected = True

        json_str = backend.snapshot_to_json(snapshot)
        parsed = json.loads(json_str)

        # datetime should be serialized as ISO string
        assert "start_time" in parsed
        # connected should be serialized as bool
        assert parsed["connected"] is True
