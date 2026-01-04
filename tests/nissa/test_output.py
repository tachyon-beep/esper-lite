"""Tests for Nissa output backends."""

import json
import logging
import time
from datetime import datetime
from pathlib import Path

import pytest

from esper.leyline import TelemetryEvent, TelemetryEventType
from esper.leyline.telemetry import (
    EpochCompletedPayload,
    SeedGerminatedPayload,
)
from esper.nissa.output import DirectoryOutput, NissaHub, OutputBackend, _join_with_timeout


class TestDirectoryOutput:
    """Tests for DirectoryOutput backend."""

    def test_creates_timestamped_subdirectory(self, tmp_path: Path):
        """DirectoryOutput creates a timestamped subdirectory."""
        backend = DirectoryOutput(tmp_path)

        # Should have created a subdirectory matching pattern
        subdirs = list(tmp_path.iterdir())
        assert len(subdirs) == 1
        assert subdirs[0].is_dir()
        assert subdirs[0].name.startswith("telemetry_")

        backend.close()

    def test_writes_events_to_jsonl_file(self, tmp_path: Path):
        """DirectoryOutput writes events to events.jsonl in the timestamped folder."""
        backend = DirectoryOutput(tmp_path)

        event = TelemetryEvent(
            event_type=TelemetryEventType.EPOCH_COMPLETED,
            seed_id="seed_0",
            epoch=5,
            message="Test event",
            data=EpochCompletedPayload(
                env_id=0,
                inner_epoch=5,
                val_loss=0.5,
                val_accuracy=85.5,
                seeds=None,
            ),
        )
        backend.emit(event)
        backend.close()

        # Find the events.jsonl file
        subdirs = list(tmp_path.iterdir())
        events_file = subdirs[0] / "events.jsonl"
        assert events_file.exists()

        # Verify content
        with open(events_file) as f:
            line = f.readline()
            data = json.loads(line)
            assert data["event_type"] == "EPOCH_COMPLETED"
            assert data["seed_id"] == "seed_0"
            assert data["data"]["val_accuracy"] == 85.5

    def test_output_dir_property_returns_timestamped_path(self, tmp_path: Path):
        """DirectoryOutput.output_dir returns the full path to timestamped directory."""
        backend = DirectoryOutput(tmp_path)

        assert backend.output_dir.parent == tmp_path
        assert backend.output_dir.name.startswith("telemetry_")
        assert backend.output_dir.is_dir()

        backend.close()

    def test_timestamp_format_is_sortable(self, tmp_path: Path):
        """Timestamp uses YYYY-MM-DD_HHMMSS format for sortability."""
        backend = DirectoryOutput(tmp_path)

        # Pattern: telemetry_YYYY-MM-DD_HHMMSS
        subdir_name = backend.output_dir.name
        assert subdir_name.startswith("telemetry_")

        # Extract timestamp part
        ts_part = subdir_name.replace("telemetry_", "")
        # Should be parseable
        datetime.strptime(ts_part, "%Y-%m-%d_%H%M%S")

        backend.close()


class TestNissaHubWithDirectoryOutput:
    """Integration tests for NissaHub with DirectoryOutput."""

    def test_hub_routes_to_directory_output(self, tmp_path: Path):
        """NissaHub correctly routes events to DirectoryOutput backend."""
        from esper.nissa.output import NissaHub

        hub = NissaHub()
        dir_backend = DirectoryOutput(tmp_path)
        hub.add_backend(dir_backend)

        event = TelemetryEvent(
            event_type=TelemetryEventType.SEED_GERMINATED,
            seed_id="seed_1",
            epoch=0,
            message="Germinated",
            data=SeedGerminatedPayload(
                slot_id="slot_0",
                env_id=0,
                blueprint_id="test_bp",
                params=1000,
                alpha=0.0,
            ),
        )
        hub.emit(event)
        hub.close()

        # Verify event was written
        events_file = dir_backend.output_dir / "events.jsonl"
        assert events_file.exists()

        with open(events_file) as f:
            data = json.loads(f.readline())
            assert data["event_type"] == "SEED_GERMINATED"
            assert data["data"]["blueprint_id"] == "test_bp"


class TestNissaHubEmitAfterClose:
    """Tests for emit() after close() behavior."""

    def test_emit_after_close_drops_events(self, tmp_path: Path):
        """emit() after close() should silently drop events."""
        hub = NissaHub()
        dir_backend = DirectoryOutput(tmp_path)
        hub.add_backend(dir_backend)

        event1 = TelemetryEvent(
            event_type=TelemetryEventType.EPOCH_COMPLETED,
            epoch=1,
            data=EpochCompletedPayload(
                env_id=0,
                inner_epoch=1,
                val_loss=0.5,
                val_accuracy=80.0,
                seeds=None,
            ),
        )
        hub.emit(event1)
        hub.close()

        # Emit after close should be dropped
        event2 = TelemetryEvent(
            event_type=TelemetryEventType.EPOCH_COMPLETED,
            epoch=2,
            data=EpochCompletedPayload(
                env_id=0,
                inner_epoch=2,
                val_loss=0.3,
                val_accuracy=90.0,
                seeds=None,
            ),
        )
        hub.emit(event2)

        # Only the first event should be in the file
        events_file = dir_backend.output_dir / "events.jsonl"
        with open(events_file) as f:
            lines = f.readlines()
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert data["epoch"] == 1

    def test_emit_after_close_logs_warning_once(self, caplog: pytest.LogCaptureFixture):
        """emit() after close() should log warning only once."""
        hub = NissaHub()
        hub.close()

        event = TelemetryEvent(
            event_type=TelemetryEventType.EPOCH_COMPLETED,
            epoch=1,
            data=EpochCompletedPayload(
                env_id=0,
                inner_epoch=1,
                val_loss=0.5,
                val_accuracy=80.0,
                seeds=None,
            ),
        )

        with caplog.at_level(logging.WARNING, logger="esper.nissa.output"):
            hub.emit(event)
            hub.emit(event)
            hub.emit(event)

        # Should only log warning once
        warning_count = sum(
            1 for record in caplog.records
            if "emit() called on closed NissaHub" in record.message
        )
        assert warning_count == 1

    def test_reset_clears_emit_after_close_warning_flag(self, caplog: pytest.LogCaptureFixture):
        """reset() should allow warning to be logged again."""
        hub = NissaHub()
        hub.close()

        event = TelemetryEvent(
            event_type=TelemetryEventType.EPOCH_COMPLETED,
            epoch=1,
            data=EpochCompletedPayload(
                env_id=0,
                inner_epoch=1,
                val_loss=0.5,
                val_accuracy=80.0,
                seeds=None,
            ),
        )

        with caplog.at_level(logging.WARNING, logger="esper.nissa.output"):
            hub.emit(event)  # First warning
            hub.reset()  # Reset clears the flag
            hub.close()
            hub.emit(event)  # Second warning after reset

        warning_count = sum(
            1 for record in caplog.records
            if "emit() called on closed NissaHub" in record.message
        )
        assert warning_count == 2


class TestNissaHubAddBackendFailure:
    """Tests for add_backend() failure handling."""

    def test_add_backend_raises_on_start_failure(self):
        """add_backend() should re-raise exception if backend.start() fails."""

        class FailingBackend(OutputBackend):
            def start(self) -> None:
                raise RuntimeError("Backend initialization failed")

            def emit(self, event: TelemetryEvent) -> None:
                pass

            def close(self) -> None:
                pass

        hub = NissaHub()

        with pytest.raises(RuntimeError, match="Backend initialization failed"):
            hub.add_backend(FailingBackend())

        # Backend should not have been added
        assert len(hub._backends) == 0

    def test_add_backend_logs_error_on_failure(self, caplog: pytest.LogCaptureFixture):
        """add_backend() should log error before re-raising."""

        class FailingBackend(OutputBackend):
            def start(self) -> None:
                raise RuntimeError("Backend initialization failed")

            def emit(self, event: TelemetryEvent) -> None:
                pass

            def close(self) -> None:
                pass

        hub = NissaHub()

        with caplog.at_level(logging.ERROR, logger="esper.nissa.output"):
            with pytest.raises(RuntimeError):
                hub.add_backend(FailingBackend())

        assert any(
            "Failed to start backend" in record.message
            for record in caplog.records
        )

    def test_add_backend_raises_on_closed_hub(self):
        """add_backend() should raise if hub is closed."""

        class MockBackend(OutputBackend):
            def start(self) -> None:
                pass

            def emit(self, event: TelemetryEvent) -> None:
                pass

            def close(self) -> None:
                pass

        hub = NissaHub()
        hub.close()

        with pytest.raises(RuntimeError, match="Cannot add backend to closed NissaHub"):
            hub.add_backend(MockBackend())

    def test_add_backend_works_after_reset(self):
        """add_backend() should work after reset() reopens the hub."""

        class MockBackend(OutputBackend):
            def start(self) -> None:
                pass

            def emit(self, event: TelemetryEvent) -> None:
                pass

            def close(self) -> None:
                pass

        hub = NissaHub()
        hub.close()
        hub.reset()  # Reopen the hub

        # Should work now
        hub.add_backend(MockBackend())
        assert len(hub._backends) == 1


class TestNissaHubShutdownRace:
    """Tests for the shutdown race condition fix.

    Regression tests for the race where emit() could add events after the
    sentinel was placed in the queue, causing those events to be silently
    dropped.
    """

    def test_close_processes_all_pending_events(self):
        """close() should process all events enqueued before close() was called.

        Regression test: Prior to fix, setting _closed after join() meant events
        could be enqueued after the sentinel and never processed.
        """
        import threading
        import time

        received_events: list[TelemetryEvent] = []
        lock = threading.Lock()

        class CountingBackend(OutputBackend):
            def start(self) -> None:
                pass

            def emit(self, event: TelemetryEvent) -> None:
                # Simulate slow processing to increase race window
                time.sleep(0.001)
                with lock:
                    received_events.append(event)

            def close(self) -> None:
                pass

        hub = NissaHub()
        hub.add_backend(CountingBackend())

        # Emit a batch of events
        num_events = 50
        for i in range(num_events):
            event = TelemetryEvent(
                event_type=TelemetryEventType.EPOCH_COMPLETED,
                epoch=i,
                data=EpochCompletedPayload(
                    env_id=0,
                    val_accuracy=0.5,
                    val_loss=0.1,
                    inner_epoch=i,
                ),
            )
            hub.emit(event)

        # Close should wait for all pending events
        hub.close()

        # All events emitted before close() should have been processed
        with lock:
            assert len(received_events) == num_events, (
                f"Expected {num_events} events, got {len(received_events)}. "
                "Events were lost during shutdown."
            )

    def test_emit_after_close_is_rejected(self):
        """Events emitted after close() starts should be rejected, not silently lost.

        The fix sets _closed early to prevent enqueueing after the sentinel.
        """

        class MockBackend(OutputBackend):
            def start(self) -> None:
                pass

            def emit(self, event: TelemetryEvent) -> None:
                pass

            def close(self) -> None:
                pass

        hub = NissaHub()
        hub.add_backend(MockBackend())

        # Emit one event to ensure worker is processing
        event = TelemetryEvent(
            event_type=TelemetryEventType.EPOCH_COMPLETED,
            epoch=0,
            data=EpochCompletedPayload(
                env_id=0,
                val_accuracy=0.5,
                val_loss=0.1,
                inner_epoch=0,
            ),
        )
        hub.emit(event)

        # Close the hub
        hub.close()

        # Verify _closed is True (events after this are dropped, not silently lost)
        assert hub._closed is True

        # Emitting now should not raise, but event should be dropped
        hub.emit(event)  # Should not raise


class TestJoinWithTimeout:
    """Tests for _join_with_timeout helper function."""

    def test_join_with_none_timeout_waits_indefinitely(self):
        """None timeout should wait until queue drains."""
        import queue
        import threading

        q: queue.Queue[int] = queue.Queue()
        q.put(1)

        # Process item in background after small delay
        def processor():
            time.sleep(0.05)
            q.get()
            q.task_done()

        t = threading.Thread(target=processor)
        t.start()

        result = _join_with_timeout(q, timeout=None)
        t.join()

        assert result is True

    def test_join_with_zero_timeout_returns_immediately(self):
        """Zero timeout should return False immediately if queue not empty."""
        import queue

        q: queue.Queue[int] = queue.Queue()
        q.put(1)  # Put item but don't process

        start = time.monotonic()
        result = _join_with_timeout(q, timeout=0)
        elapsed = time.monotonic() - start

        assert result is False
        assert elapsed < 0.1  # Should be nearly instant

    def test_join_timeout_returns_false_when_exceeded(self):
        """Timeout should return False when queue doesn't drain in time."""
        import queue

        q: queue.Queue[int] = queue.Queue()
        q.put(1)  # Put item but don't process

        start = time.monotonic()
        result = _join_with_timeout(q, timeout=0.1)
        elapsed = time.monotonic() - start

        assert result is False
        assert 0.09 < elapsed < 0.2  # Should wait approximately timeout duration

    def test_join_returns_true_when_queue_drains(self):
        """Should return True when queue drains within timeout."""
        import queue
        import threading

        q: queue.Queue[int] = queue.Queue()
        q.put(1)

        def processor():
            time.sleep(0.02)
            q.get()
            q.task_done()

        t = threading.Thread(target=processor)
        t.start()

        result = _join_with_timeout(q, timeout=1.0)
        t.join()

        assert result is True


class TestNissaHubTimeoutBehavior:
    """Tests for timeout behavior in NissaHub operations.

    These tests verify that flush() and close() respect their timeout parameters
    and don't hang indefinitely when backends are slow or stuck.
    """

    def test_flush_returns_within_timeout_on_slow_backend(self):
        """flush() should return within timeout even if backend is slow."""

        class BlockingBackend(OutputBackend):
            """Backend that blocks forever on emit()."""

            def start(self) -> None:
                pass

            def emit(self, event: TelemetryEvent) -> None:
                # Block forever - simulates stuck backend
                import threading
                threading.Event().wait()

            def close(self) -> None:
                pass

        hub = NissaHub()
        hub.add_backend(BlockingBackend())

        # Emit an event that will get stuck
        event = TelemetryEvent(
            event_type=TelemetryEventType.EPOCH_COMPLETED,
            epoch=0,
            data=EpochCompletedPayload(
                env_id=0,
                val_accuracy=0.5,
                val_loss=0.1,
                inner_epoch=0,
            ),
        )
        hub.emit(event)

        # flush() should return within timeout, not hang forever
        start = time.monotonic()
        result = hub.flush(timeout=0.2)
        elapsed = time.monotonic() - start

        assert result is False  # Should indicate timeout
        assert elapsed < 0.5  # Should return within reasonable time of timeout

        # Cleanup - close with short timeout
        hub.close(timeout=0.1)

    def test_close_returns_within_timeout_on_stuck_worker(self):
        """close() should return within timeout even if worker is stuck."""

        class BlockingBackend(OutputBackend):
            """Backend that blocks forever on emit()."""

            def start(self) -> None:
                pass

            def emit(self, event: TelemetryEvent) -> None:
                import threading
                threading.Event().wait()

            def close(self) -> None:
                pass

        hub = NissaHub()
        hub.add_backend(BlockingBackend())

        event = TelemetryEvent(
            event_type=TelemetryEventType.EPOCH_COMPLETED,
            epoch=0,
            data=EpochCompletedPayload(
                env_id=0,
                val_accuracy=0.5,
                val_loss=0.1,
                inner_epoch=0,
            ),
        )
        hub.emit(event)

        # close() should return within timeout, not hang forever
        start = time.monotonic()
        hub.close(timeout=0.3)
        elapsed = time.monotonic() - start

        assert hub._closed is True
        assert elapsed < 1.0  # Should return within reasonable time of timeout

    def test_flush_returns_true_on_healthy_backend(self):
        """flush() should return True when all events processed."""

        class FastBackend(OutputBackend):
            def __init__(self):
                self.events: list[TelemetryEvent] = []

            def start(self) -> None:
                pass

            def emit(self, event: TelemetryEvent) -> None:
                self.events.append(event)

            def close(self) -> None:
                pass

        backend = FastBackend()
        hub = NissaHub()
        hub.add_backend(backend)

        for i in range(10):
            event = TelemetryEvent(
                event_type=TelemetryEventType.EPOCH_COMPLETED,
                epoch=i,
                data=EpochCompletedPayload(
                    env_id=0,
                    val_accuracy=0.5,
                    val_loss=0.1,
                    inner_epoch=i,
                ),
            )
            hub.emit(event)

        result = hub.flush(timeout=5.0)
        hub.close()

        assert result is True
        assert len(backend.events) == 10

    def test_flush_logs_warning_on_timeout(self, caplog: pytest.LogCaptureFixture):
        """flush() should log warning when timeout expires."""
        class SlowBackend(OutputBackend):
            def start(self) -> None:
                pass

            def emit(self, event: TelemetryEvent) -> None:
                time.sleep(10)  # Very slow

            def close(self) -> None:
                pass

        hub = NissaHub()
        hub.add_backend(SlowBackend())

        event = TelemetryEvent(
            event_type=TelemetryEventType.EPOCH_COMPLETED,
            epoch=0,
            data=EpochCompletedPayload(
                env_id=0,
                val_accuracy=0.5,
                val_loss=0.1,
                inner_epoch=0,
            ),
        )
        hub.emit(event)

        with caplog.at_level(logging.WARNING, logger="esper.nissa.output"):
            hub.flush(timeout=0.1)

        # Should have logged timeout warning
        timeout_warnings = [
            r for r in caplog.records
            if "timed out" in r.message.lower()
        ]
        assert len(timeout_warnings) >= 1

        hub.close(timeout=0.1)


class TestBackendWorkerSentinelRace:
    """Tests for BackendWorker sentinel race condition fix.

    Regression test for the race where events could be enqueued after the
    sentinel was placed but before _stopped=True was set, leaving events
    unprocessed forever.
    """

    def test_worker_processes_all_events_before_stop(self):
        """BackendWorker.stop() should process all enqueued events.

        Regression test: Prior to fix, enqueue() could race with stop(),
        placing events after the sentinel that were never processed.
        """
        import threading

        received_count = 0
        lock = threading.Lock()

        class CountingBackend(OutputBackend):
            def start(self) -> None:
                pass

            def emit(self, event: TelemetryEvent) -> None:
                nonlocal received_count
                with lock:
                    received_count += 1

            def close(self) -> None:
                pass

        from esper.nissa.output import BackendWorker

        backend = CountingBackend()
        backend.start()
        worker = BackendWorker(backend, max_queue_size=1000)

        # Enqueue many events rapidly
        num_events = 100
        for i in range(num_events):
            event = TelemetryEvent(
                event_type=TelemetryEventType.EPOCH_COMPLETED,
                epoch=i,
                data=EpochCompletedPayload(
                    env_id=0,
                    val_accuracy=0.5,
                    val_loss=0.1,
                    inner_epoch=i,
                ),
            )
            worker.enqueue(event)

        # Stop should wait for all events
        worker.stop(timeout=5.0)

        with lock:
            assert received_count == num_events, (
                f"Expected {num_events} events, got {received_count}. "
                "Events were lost due to sentinel race."
            )

    def test_enqueue_after_stop_is_rejected(self):
        """Events enqueued after stop() should be silently rejected.

        This verifies the fix: _stopped is set before sentinel, so enqueue()
        returns early instead of placing events after sentinel.
        """
        received_events: list[TelemetryEvent] = []

        class RecordingBackend(OutputBackend):
            def start(self) -> None:
                pass

            def emit(self, event: TelemetryEvent) -> None:
                received_events.append(event)

            def close(self) -> None:
                pass

        from esper.nissa.output import BackendWorker

        backend = RecordingBackend()
        backend.start()
        worker = BackendWorker(backend, max_queue_size=100)

        # Enqueue one event
        event1 = TelemetryEvent(
            event_type=TelemetryEventType.EPOCH_COMPLETED,
            epoch=1,
            data=EpochCompletedPayload(
                env_id=0, val_accuracy=0.5, val_loss=0.1, inner_epoch=1
            ),
        )
        worker.enqueue(event1)

        # Stop the worker
        worker.stop(timeout=5.0)

        # Verify stopped
        assert worker._stopped is True

        # Enqueue after stop should be rejected
        event2 = TelemetryEvent(
            event_type=TelemetryEventType.EPOCH_COMPLETED,
            epoch=2,
            data=EpochCompletedPayload(
                env_id=0, val_accuracy=0.5, val_loss=0.1, inner_epoch=2
            ),
        )
        worker.enqueue(event2)  # Should not raise, but should be dropped

        # Only the first event should have been processed
        assert len(received_events) == 1
        assert received_events[0].epoch == 1


class TestNissaHubConcurrentAddRemove:
    """Tests for thread-safe add_backend/remove_backend operations.

    Regression tests for the race where worker loop iteration could
    crash or miss backends due to unsynchronized list mutations.
    """

    def test_remove_backend_during_emission(self):
        """remove_backend() should be safe during active emission.

        Regression test: Prior to fix, remove_backend() mutated lists
        while worker loop was iterating, causing crashes or missed events.
        """
        import threading

        events_received: list[tuple[str, int | None]] = []
        events_lock = threading.Lock()

        class SlowBackend(OutputBackend):
            def __init__(self, name: str):
                self.name = name

            def start(self) -> None:
                pass

            def emit(self, event: TelemetryEvent) -> None:
                time.sleep(0.01)  # Slow enough to allow race
                with events_lock:
                    events_received.append((self.name, event.epoch))

            def close(self) -> None:
                pass

        hub = NissaHub()
        backend1 = SlowBackend("backend1")
        backend2 = SlowBackend("backend2")
        hub.add_backend(backend1)
        hub.add_backend(backend2)

        # Emit some events
        for i in range(10):
            event = TelemetryEvent(
                event_type=TelemetryEventType.EPOCH_COMPLETED,
                epoch=i,
                data=EpochCompletedPayload(
                    env_id=0, val_accuracy=0.5, val_loss=0.1, inner_epoch=i
                ),
            )
            hub.emit(event)

        # Remove backend1 while events are still being processed
        hub.remove_backend(backend1)

        # Emit more events (should only go to backend2)
        for i in range(10, 15):
            event = TelemetryEvent(
                event_type=TelemetryEventType.EPOCH_COMPLETED,
                epoch=i,
                data=EpochCompletedPayload(
                    env_id=0, val_accuracy=0.5, val_loss=0.1, inner_epoch=i
                ),
            )
            hub.emit(event)

        # Flush and close
        hub.flush(timeout=5.0)
        hub.close()

        # Backend2 should have received all events (0-14)
        with events_lock:
            backend2_events = [e for name, e in events_received if name == "backend2"]

        # Should have at least 10 events from backend2 (the post-removal ones)
        assert len(backend2_events) >= 5, (
            f"Backend2 should have received events after backend1 removal. "
            f"Got {len(backend2_events)} events."
        )
