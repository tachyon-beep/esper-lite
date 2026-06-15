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
from esper.nissa.output import (
    DirectoryOutput,
    FileOutput,
    NissaHub,
    OutputBackend,
    _join_with_timeout,
)


class TestFileOutputEpisodeIdx:
    """LN-001: FileOutput JSONL rows must carry episode_idx."""

    def test_jsonl_row_includes_episode_idx(self, tmp_path: Path):
        backend = FileOutput(tmp_path / "events.jsonl", buffer_size=1)

        event = TelemetryEvent(
            event_type=TelemetryEventType.EPOCH_COMPLETED,
            seed_id="seed_0",
            epoch=3,
            data=EpochCompletedPayload(
                env_id=1,
                inner_epoch=3,
                val_loss=0.4,
                val_accuracy=80.0,
                episode_idx=7,
                seeds=None,
            ),
        )
        backend.emit(event)
        backend.close()

        with open(tmp_path / "events.jsonl") as f:
            data = json.loads(f.readline())
        assert data["data"]["episode_idx"] == 7


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

    def test_jsonl_row_includes_episode_idx(self, tmp_path: Path):
        """LN-001: DirectoryOutput JSONL rows must carry episode_idx (no silent drop)."""
        backend = DirectoryOutput(tmp_path)

        event = TelemetryEvent(
            event_type=TelemetryEventType.EPOCH_COMPLETED,
            seed_id="seed_0",
            epoch=5,
            data=EpochCompletedPayload(
                env_id=0,
                inner_epoch=5,
                val_loss=0.5,
                val_accuracy=85.5,
                episode_idx=9,
                seeds=None,
            ),
        )
        backend.emit(event)
        backend.close()

        events_file = list(tmp_path.iterdir())[0] / "events.jsonl"
        with open(events_file) as f:
            data = json.loads(f.readline())
        assert "episode_idx" in data["data"]
        assert data["data"]["episode_idx"] == 9

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

    def test_same_second_outputs_get_unique_run_directories(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """DirectoryOutput must not share events.jsonl across rapid proof cohorts."""
        import esper.nissa.output as output

        class FixedDateTime:
            @classmethod
            def now(cls) -> datetime:
                return datetime(2026, 6, 15, 12, 30, 45)

        monkeypatch.setattr(output, "datetime", FixedDateTime)

        first = DirectoryOutput(tmp_path)
        second = DirectoryOutput(tmp_path)

        assert first.output_dir != second.output_dir
        assert first.output_dir.name == "telemetry_2026-06-15_123045"
        assert second.output_dir.name == "telemetry_2026-06-15_123045_001"

        first.close()
        second.close()


class TestNissaHubHealth:
    """Tests for queue-pressure health reporting."""

    def test_health_snapshot_surfaces_hub_and_backend_drops(self):
        class NoopBackend(OutputBackend):
            def start(self) -> None:
                pass

            def emit(self, event: TelemetryEvent) -> None:
                pass

            def close(self) -> None:
                pass

        hub = NissaHub()
        backend = NoopBackend()
        hub.add_backend(backend)
        hub._dropped_events = 2
        hub._backend_workers[0]._dropped_events = 3

        health = hub.get_health_snapshot()

        assert health["status"] == "degraded"
        assert health["dropped_events"] == 5
        assert health["hub_dropped_events"] == 2
        assert health["backend_dropped_events"] == 3

        hub.close()


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
        import threading

        release_emit = threading.Event()

        class SlowBackend(OutputBackend):
            def start(self) -> None:
                pass

            def emit(self, event: TelemetryEvent) -> None:
                release_emit.wait()

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

        release_emit.set()
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
        removal_started = threading.Event()
        release_removed_backend = threading.Event()

        class SlowBackend(OutputBackend):
            def __init__(self, name: str):
                self.name = name

            def start(self) -> None:
                pass

            def emit(self, event: TelemetryEvent) -> None:
                if self.name == "backend1":
                    removal_started.set()
                    release_removed_backend.wait()
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

        assert removal_started.wait(timeout=1.0)
        # Remove backend1 while events are still being processed
        remove_thread = threading.Thread(target=hub.remove_backend, args=(backend1,))
        remove_thread.start()
        assert remove_thread.is_alive()
        release_removed_backend.set()
        remove_thread.join(timeout=1.0)
        assert not remove_thread.is_alive()

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


class TestTGV001JsonlPayloadFields:
    """TGV-001: Nissa JSONL output must preserve non-default payload fields.

    The FileOutput/DirectoryOutput JSONL path serializes a payload via
    ``_payload_to_dict`` (``payload.to_dict()`` if present, else
    ``dataclasses.asdict``). These tests assert the key proof/diagnostic
    payloads survive that path with their distinctive non-default fields
    intact, AND that the emitted JSONL re-parses back through ``from_dict``
    without losing those fields. This is the Nissa-path complement to
    ``tests/leyline/test_payload_round_trip.py``.
    """

    @staticmethod
    def _emit_and_read(tmp_path: Path, event: TelemetryEvent) -> dict:
        backend = FileOutput(tmp_path / "events.jsonl", buffer_size=1)
        backend.emit(event)
        backend.close()
        with open(tmp_path / "events.jsonl") as f:
            return json.loads(f.readline())

    def test_seed_germinated_proof_fields_survive_jsonl(self, tmp_path: Path):
        """Morphology proof / RNG provenance fields survive the JSONL path."""
        from esper.leyline.telemetry import SeedGerminatedPayload

        payload = SeedGerminatedPayload(
            slot_id="slot_3",
            env_id=2,
            blueprint_id="bp_xyz",
            params=4096,
            episode_idx=11,
            alpha=0.33,
            grad_ratio=1.7,
            has_vanishing=True,
            has_exploding=True,
            epochs_in_stage=9,
            blend_tempo_epochs=8,
            alpha_curve="COSINE",
            morphology_proposal_id="prop-1",
            morphology_verdict_id="ver-2",
            morphology_mutation_id="mut-3",
            rng_stream="germinate",
            rng_seed=123456,
        )
        event = TelemetryEvent(
            event_type=TelemetryEventType.SEED_GERMINATED,
            data=payload,
        )
        row = self._emit_and_read(tmp_path, event)
        data = row["data"]
        # Distinctive non-default fields must be present in JSONL.
        assert data["episode_idx"] == 11
        assert data["morphology_proposal_id"] == "prop-1"
        assert data["morphology_verdict_id"] == "ver-2"
        assert data["morphology_mutation_id"] == "mut-3"
        assert data["rng_stream"] == "germinate"
        assert data["rng_seed"] == 123456
        assert data["alpha_curve"] == "COSINE"
        # And the JSONL re-parses back to an equal payload.
        assert SeedGerminatedPayload.from_dict(data) == payload

    def test_seed_stage_changed_proof_fields_survive_jsonl(self, tmp_path: Path):
        from esper.leyline.telemetry import SeedStageChangedPayload

        payload = SeedStageChangedPayload(
            slot_id="slot_1",
            env_id=4,
            from_stage="TRAINING",
            to_stage="BLENDING",
            episode_idx=22,
            alpha=0.6,
            accuracy_delta=2.5,
            epochs_in_stage=7,
            grad_ratio=1.25,
            has_vanishing=True,
            has_exploding=True,
            alpha_curve="EXPONENTIAL",
            morphology_proposal_id="prop-9",
            morphology_verdict_id="ver-8",
            morphology_mutation_id="mut-7",
            rng_stream="stage",
            rng_seed=99,
        )
        event = TelemetryEvent(
            event_type=TelemetryEventType.SEED_STAGE_CHANGED,
            data=payload,
        )
        row = self._emit_and_read(tmp_path, event)
        data = row["data"]
        assert data["episode_idx"] == 22
        assert data["morphology_proposal_id"] == "prop-9"
        assert data["rng_seed"] == 99
        assert data["alpha_curve"] == "EXPONENTIAL"
        assert SeedStageChangedPayload.from_dict(data) == payload

    def test_episode_outcome_diagnostics_survive_jsonl(self, tmp_path: Path):
        from esper.leyline.telemetry import EpisodeOutcomePayload

        payload = EpisodeOutcomePayload(
            env_id=3,
            episode_idx=14,
            final_accuracy=91.2,
            param_ratio=1.18,
            num_fossilized=2,
            num_contributing_fossilized=1,
            episode_reward=12.5,
            stability_score=0.87,
            reward_mode="shaped",
            episode_length=25,
            outcome_type="success",
            germinate_count=3,
            prune_count=1,
            fossilize_count=2,
        )
        event = TelemetryEvent(
            event_type=TelemetryEventType.EPISODE_OUTCOME,
            data=payload,
        )
        row = self._emit_and_read(tmp_path, event)
        data = row["data"]
        assert data["episode_length"] == 25
        assert data["outcome_type"] == "success"
        assert data["germinate_count"] == 3
        assert data["prune_count"] == 1
        assert data["fossilize_count"] == 2
        assert EpisodeOutcomePayload.from_dict(data) == payload

    def test_governor_rollback_panic_fields_survive_jsonl(self, tmp_path: Path):
        from esper.leyline.telemetry import GovernorRollbackPayload

        payload = GovernorRollbackPayload(
            env_id=0,
            device="cuda:0",
            reason="loss diverged",
            episode_idx=5,
            loss_at_panic=42.0,
            loss_threshold=10.0,
            consecutive_panics=3,
            panic_reason="governor_divergence",
            triggering_action_id="morph-b0-e0-env0-r0c0-op0",
            raw_penalty=-10.0,
            normalized_penalty=-10.0,
            rollback_severity=10.0,
            watch_window_evidence=10.0,
        )
        event = TelemetryEvent(
            event_type=TelemetryEventType.GOVERNOR_ROLLBACK,
            data=payload,
        )
        row = self._emit_and_read(tmp_path, event)
        data = row["data"]
        assert data["loss_at_panic"] == 42.0
        assert data["consecutive_panics"] == 3
        assert data["panic_reason"] == "governor_divergence"
        assert data["episode_idx"] == 5
        assert data["triggering_action_id"] == "morph-b0-e0-env0-r0c0-op0"
        assert data["raw_penalty"] == -10.0
        assert data["normalized_penalty"] == -10.0
        assert data["rollback_severity"] == 10.0
        assert data["watch_window_evidence"] == 10.0
        assert GovernorRollbackPayload.from_dict(data) == payload

    def test_morphology_causal_log_identity_fields_survive_jsonl(self, tmp_path: Path):
        from esper.leyline.telemetry import MorphologyCausalLogPayload

        payload = MorphologyCausalLogPayload(
            phase="verdict",
            env_id=1,
            slot_id="slot_0",
            operation="GERMINATE",
            action_id="act-1",
            proposal_id="prop-1",
            verdict_id="ver-1",
            mutation_id="mut-1",
            observation_hash="abc123",
            rng_stream="morph",
            rng_seed=7,
            topology="resnet",
            blueprint_id="bp_1",
            governor_approved=True,
            governor_reason="ok",
            governor_blocked_factor="none",
            watch_window_evidence=0.42,
            linked_event_id="evt-1",
        )
        event = TelemetryEvent(
            event_type=TelemetryEventType.MORPHOLOGY_CAUSAL_LOG,
            data=payload,
        )
        row = self._emit_and_read(tmp_path, event)
        data = row["data"]
        assert data["proposal_id"] == "prop-1"
        assert data["verdict_id"] == "ver-1"
        assert data["mutation_id"] == "mut-1"
        assert data["observation_hash"] == "abc123"
        assert data["watch_window_evidence"] == 0.42
        assert MorphologyCausalLogPayload.from_dict(data) == payload

    def test_epoch_completed_observation_stats_survive_jsonl(self, tmp_path: Path):
        """EpochCompletedPayload carries nested ObservationStatsTelemetry through JSONL."""
        from esper.leyline.telemetry import EpochCompletedPayload
        from esper.simic.telemetry.observation_stats import ObservationStatsTelemetry

        obs = ObservationStatsTelemetry(
            slot_features_mean=1.0,
            slot_features_std=2.0,
            outlier_pct=0.3,
            nan_count=1,
            inf_count=2,
            batch_size=16,
        )
        payload = EpochCompletedPayload(
            env_id=1,
            val_accuracy=88.0,
            val_loss=0.31,
            inner_epoch=4,
            episode_idx=8,
            train_loss=0.41,
            train_accuracy=86.0,
            host_grad_norm=1.9,
            seeds=None,
            observation_stats=obs,
        )
        event = TelemetryEvent(
            event_type=TelemetryEventType.EPOCH_COMPLETED,
            data=payload,
        )
        row = self._emit_and_read(tmp_path, event)
        data = row["data"]
        assert data["episode_idx"] == 8
        assert data["train_loss"] == 0.41
        assert data["host_grad_norm"] == 1.9
        assert data["observation_stats"]["batch_size"] == 16
        assert data["observation_stats"]["nan_count"] == 1
        assert EpochCompletedPayload.from_dict(data) == payload
