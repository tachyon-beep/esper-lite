"""TDD tests for P1-1: eliminate silent except Exception swallowing in Nissa.

Tests are written FIRST (red) before the fixes are applied.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import patch

from esper.leyline import TelemetryEvent, TelemetryEventType
from esper.nissa.output import BackendWorker, NissaHub, OutputBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event() -> TelemetryEvent:
    return TelemetryEvent(
        event_type=TelemetryEventType.EPOCH_COMPLETED,
        seed_id="seed_0",
        epoch=1,
    )


class _AlwaysFailBackend(OutputBackend):
    """Backend whose emit() always raises RuntimeError."""

    def start(self) -> None:
        pass

    def emit(self, event: TelemetryEvent) -> None:
        raise RuntimeError("intentional emit failure")

    def close(self) -> None:
        pass


class _NoopBackend(OutputBackend):
    def start(self) -> None:
        pass

    def emit(self, event: TelemetryEvent) -> None:
        pass

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# BUG-1: Emit failure must increment dropped_events counter
# ---------------------------------------------------------------------------

class TestEmitFailureIncrementsDroppedCounter:
    """BackendWorker: emit() failure must increment _dropped_events."""

    def test_backend_emit_failure_increments_dropped_counter(self):
        """When backend.emit() raises, dropped_events must be incremented."""
        backend = _AlwaysFailBackend()
        worker = BackendWorker(backend, max_queue_size=10, name="test-fail")

        # Enqueue one event and wait for the worker loop to process it
        worker.enqueue(_make_event())

        # Give the worker thread time to process (it runs in daemon thread)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            stats = worker.get_stats()
            if stats["dropped_events"] > 0 or stats["processed_events"] > 0:
                break
            time.sleep(0.02)

        worker.stop(timeout=2.0)
        stats = worker.get_stats()

        # The event should NOT count as processed (emit raised), and it must
        # be counted as dropped.
        assert stats["dropped_events"] == 1, (
            f"Expected dropped_events=1 after emit() raised, got stats={stats}"
        )
        assert stats["processed_events"] == 0, (
            f"Expected processed_events=0 after emit() raised, got stats={stats}"
        )


# ---------------------------------------------------------------------------
# BUG-1b: Dropped events from backend failures surface in hub health snapshot
# ---------------------------------------------------------------------------

class TestEmitFailureVisibleInHubHealthSnapshot:
    """NissaHub: emit failures must be visible in get_health_snapshot()."""

    def test_backend_emit_failure_visible_in_hub_health_snapshot(self):
        """Hub health snapshot must reflect events dropped by a failing backend."""
        hub = NissaHub()
        hub.add_backend(_AlwaysFailBackend())

        # Emit one event
        hub.emit(_make_event())

        # Flush with a timeout — the hub routes to worker, worker processes it
        hub.flush(timeout=2.0)

        # Give a little extra time for the worker thread to update counters
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            snapshot = hub.get_health_snapshot()
            if snapshot["dropped_events"] > 0:
                break
            time.sleep(0.02)

        hub.close()
        snapshot = hub.get_health_snapshot()

        # After close, backend_dropped_events should still be available from the
        # last snapshot before close cleared workers.  Re-query at close-time:
        # We need to capture BEFORE close drains everything.
        # Re-run the test more carefully:
        hub2 = NissaHub()
        hub2.add_backend(_AlwaysFailBackend())
        hub2.emit(_make_event())

        deadline = time.monotonic() + 2.0
        captured_snapshot: dict | None = None
        while time.monotonic() < deadline:
            snap = hub2.get_health_snapshot()
            if snap["dropped_events"] > 0:
                captured_snapshot = snap
                break
            time.sleep(0.02)

        hub2.close()

        assert captured_snapshot is not None, (
            "Health snapshot never showed dropped_events > 0 after emit() failure"
        )
        assert captured_snapshot["status"] == "degraded"
        assert captured_snapshot["dropped_events"] >= 1


# ---------------------------------------------------------------------------
# BUG-2: flush() must return False when _join_with_timeout raises
# ---------------------------------------------------------------------------

class TestFlushReturnsFalseOnBrokenQueue:
    """NissaHub.flush() must return False when backend queue join raises RuntimeError."""

    def test_flush_returns_false_when_backend_worker_queue_broken(self):
        """flush() returns False if _join_with_timeout raises RuntimeError for a worker."""
        hub = NissaHub()
        hub.add_backend(_NoopBackend())

        # Emit something so there's a worker active
        hub.emit(_make_event())
        # Wait for main queue to drain normally
        hub.flush(timeout=1.0)

        # Now mock _join_with_timeout to raise on the second call (backend worker drain)
        call_count = [0]

        def _fake_join(q, timeout):  # noqa: ANN001
            call_count[0] += 1
            if call_count[0] >= 2:
                raise RuntimeError("simulated broken queue")
            return True

        import esper.nissa.output as output_module
        with patch.object(output_module, "_join_with_timeout", side_effect=_fake_join):
            result = hub.flush(timeout=2.0)

        hub.close()

        assert result is False, (
            "flush() must return False when _join_with_timeout raises RuntimeError "
            f"for a backend worker queue; got {result!r}"
        )


# ---------------------------------------------------------------------------
# BUG-3: Worker loop must break (not spin) when queue.get raises RuntimeError
# ---------------------------------------------------------------------------

class TestHubWorkerLoopBreaksOnBrokenQueue:
    """BackendWorker._worker_loop must exit when queue.get raises RuntimeError."""

    def test_hub_worker_loop_breaks_on_broken_queue(self):
        """Worker thread exits within ~2s when queue.get raises RuntimeError."""
        backend = _NoopBackend()
        worker = BackendWorker(backend, max_queue_size=10, name="test-broken-q")

        # The worker thread is already running. Patch queue.get to raise RuntimeError.
        original_get = worker._queue.get

        call_count = [0]

        def _failing_get(*args, **kwargs):  # noqa: ANN002, ANN003
            call_count[0] += 1
            if call_count[0] >= 2:
                raise RuntimeError("simulated queue corruption")
            return original_get(*args, **kwargs)

        worker._queue.get = _failing_get  # type: ignore[method-assign]

        # Enqueue a sentinel-like event to kick the first get() call through
        worker.enqueue(_make_event())

        # Worker thread should exit within 2 seconds
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if not worker._thread.is_alive():
                break
            time.sleep(0.05)

        assert not worker._thread.is_alive(), (
            "Worker thread must exit when queue.get raises RuntimeError, "
            "not spin forever"
        )


# ---------------------------------------------------------------------------
# BUG-4 (train.py): training_wrapper must set shutdown_event on crash
# ---------------------------------------------------------------------------

class TestTrainingWrapperSetsShutdownEventOnException:
    """The REAL train.py crash-handling helper must signal shutdown on crash.

    These exercise esper.scripts.train._run_training_capturing_errors directly
    (the body extracted from the sanctum training_wrapper closure), so they fail
    if the shutdown-signalling fix is reverted.
    """

    def test_training_wrapper_sets_shutdown_event_on_exception(self):
        from esper.scripts.train import _run_training_capturing_errors

        shutdown_event = threading.Event()
        training_error: list[str | None] = [None]

        def _crashing_run_training() -> None:
            raise RuntimeError("training crashed")

        # Run via a thread, mirroring the production daemon-thread usage.
        t = threading.Thread(
            target=_run_training_capturing_errors,
            args=(_crashing_run_training, training_error, shutdown_event),
        )
        t.start()
        t.join(timeout=2.0)

        assert shutdown_event.is_set(), (
            "_run_training_capturing_errors must set shutdown_event when "
            "run_training() raises, so the process exits non-zero on crash"
        )
        assert training_error[0] is not None, "training_error should record the traceback"
        assert "RuntimeError" in training_error[0]

    def test_training_wrapper_clean_run_does_not_signal_shutdown(self):
        from esper.scripts.train import _run_training_capturing_errors

        shutdown_event = threading.Event()
        training_error: list[str | None] = [None]
        ran = []

        def _clean_run_training() -> None:
            ran.append(True)

        _run_training_capturing_errors(_clean_run_training, training_error, shutdown_event)

        assert ran == [True], "run_training must be invoked"
        assert not shutdown_event.is_set(), "a clean run must NOT signal shutdown"
        assert training_error[0] is None, "a clean run must not record an error"


class TestSanctumTrainingCrashExitsNonzero:
    """A crash must populate training_error (the non-zero exit signal)."""

    def test_sanctum_training_crash_exits_nonzero(self):
        from esper.scripts.train import _run_training_capturing_errors

        training_error: list[str | None] = [None]
        shutdown_event = threading.Event()

        def _crashing_run_training() -> None:
            raise RuntimeError("training exploded")

        _run_training_capturing_errors(_crashing_run_training, training_error, shutdown_event)

        assert training_error[0] is not None, "training_error must be populated on crash"
        assert shutdown_event.is_set(), "shutdown_event must be set to signal non-zero exit"
        assert "RuntimeError" in training_error[0]
