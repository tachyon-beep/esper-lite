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

    def test_emit_triggers_broadcast(self) -> None:
        """emit() should call maybe_broadcast() to notify WebSocket clients.

        Regression test: Prior to fix, emit() processed events but never
        triggered broadcasts, leaving WebSocket clients stuck on initial state.
        """
        backend = OverwatchBackend(port=8080)
        backend.aggregator = MagicMock()
        backend.maybe_broadcast = MagicMock()

        mock_event = MagicMock()
        backend.emit(mock_event)

        backend.maybe_broadcast.assert_called_once()

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
        """Backend should start and stop cleanly, returning success status."""
        backend = OverwatchBackend(port=8080)

        # Start should return bool indicating success
        # Returns True if FastAPI available, False if missing dependencies
        result = backend.start()
        assert isinstance(result, bool)

        # Stop should clean up regardless of start state
        backend.stop()
        assert backend._running is False

    def test_close_delegates_to_stop(self) -> None:
        """close() should delegate to stop() for hub shutdown compatibility.

        Regression test: Nissa hub calls backend.close() during shutdown,
        but OverwatchBackend originally only had stop().
        """
        backend = OverwatchBackend(port=8080)
        backend.stop = MagicMock()

        backend.close()

        backend.stop.assert_called_once()

    def test_headless_mode_skips_broadcasts(self) -> None:
        """When FastAPI not installed, emit() should not queue broadcasts.

        Regression test: Prior to fix, _running was set True before import,
        causing emit() to queue broadcasts with no consumer (memory leak).
        """
        backend = OverwatchBackend(port=8080)

        # Start without FastAPI - _running should stay False
        backend.start()

        # Queue should start empty
        assert backend._broadcast_queue.empty()

        # Emit events - should NOT queue broadcasts since _running is False
        backend.aggregator = MagicMock()
        for _ in range(10):
            backend.emit(MagicMock())

        # Queue should still be empty (no leak)
        assert backend._broadcast_queue.empty()

    def test_close_clears_broadcast_queue(self) -> None:
        """close() should clear any remaining queued messages."""
        backend = OverwatchBackend(port=8080)

        # Manually queue some messages
        backend._broadcast_queue.put("message1")
        backend._broadcast_queue.put("message2")
        assert not backend._broadcast_queue.empty()

        backend.close()

        # Queue should be cleared
        assert backend._broadcast_queue.empty()

    def test_broadcast_skipped_when_no_clients(self) -> None:
        """_broadcast() should skip queueing when no clients connected.

        Regression test: Prior to fix, broadcasts were queued even with no
        clients, causing unbounded memory growth and flooding first client.
        """
        backend = OverwatchBackend(port=8080)
        backend._running = True  # Simulate running server

        # No clients connected
        assert len(backend._clients) == 0
        assert backend._broadcast_queue.empty()

        # Call _broadcast multiple times
        for _ in range(100):
            backend._broadcast()

        # Queue should still be empty - no point queueing with no clients
        assert backend._broadcast_queue.empty()

    def test_broadcast_queue_is_bounded(self) -> None:
        """Queue should have bounded size to prevent memory leaks."""
        backend = OverwatchBackend(port=8080)

        # Queue should have maxsize
        assert backend._broadcast_queue.maxsize == 10

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
