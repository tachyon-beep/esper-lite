"""Tests for OverwatchBackend."""

import pytest
from esper.leyline import TelemetryEvent, TelemetryEventType
from esper.karn.overwatch.backend import OverwatchBackend


class TestOverwatchBackend:
    """Test OverwatchBackend OutputBackend implementation."""

    def test_backend_protocol_compliance(self):
        """Backend implements OutputBackend protocol."""
        backend = OverwatchBackend()

        # Should have required methods
        assert hasattr(backend, "start")
        assert hasattr(backend, "emit")
        assert hasattr(backend, "close")

    def test_emit_updates_aggregator(self):
        """emit() passes events to aggregator."""
        backend = OverwatchBackend()
        backend.start()

        event = TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data={"run_id": "test-run", "task": "cifar10"},
        )
        backend.emit(event)

        snapshot = backend.get_snapshot()
        assert snapshot.run_id == "test-run"
        assert snapshot.connection.connected is True

    def test_get_snapshot_thread_safe(self):
        """get_snapshot() returns copy safe for cross-thread access."""
        backend = OverwatchBackend()
        backend.start()

        backend.emit(TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data={"run_id": "test"},
        ))

        snapshot1 = backend.get_snapshot()
        snapshot2 = backend.get_snapshot()

        # Should be separate objects
        assert snapshot1 is not snapshot2

    def test_close_is_idempotent(self):
        """close() can be called multiple times safely."""
        backend = OverwatchBackend()
        backend.start()
        backend.close()
        backend.close()  # Should not raise
