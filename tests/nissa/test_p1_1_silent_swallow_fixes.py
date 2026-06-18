"""TDD tests for P1-1: eliminate silent except Exception swallowing in Nissa.

Tests are written FIRST (red) before the fixes are applied.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import patch

import pytest

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

    def test_flush_returns_false_when_main_queue_broken(self):
        """flush() returns False if _join_with_timeout raises on the MAIN queue (first call).

        Regression for P2-c: the main-queue join at output.py:819 was unguarded, so a
        RuntimeError there propagated out of flush() instead of being surfaced as a
        False return value (delivery failed). Raising on the FIRST call exercises the
        main-queue path specifically (the backend-worker path is the second+ call).
        """
        hub = NissaHub()
        hub.add_backend(_NoopBackend())

        # Emit something so there's a worker active and a non-empty main queue path.
        hub.emit(_make_event())
        # Wait for main queue to drain normally
        hub.flush(timeout=1.0)

        # Mock _join_with_timeout to raise on the FIRST call (the main queue drain).
        def _fake_join(q, timeout):  # noqa: ANN001
            raise RuntimeError("simulated broken main queue")

        import esper.nissa.output as output_module
        with patch.object(output_module, "_join_with_timeout", side_effect=_fake_join):
            result = hub.flush(timeout=2.0)

        hub.close()

        assert result is False, (
            "flush() must return False (not raise) when _join_with_timeout raises "
            f"RuntimeError for the MAIN queue; got {result!r}"
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
    """A captured sanctum-thread crash must drive an actual non-zero process exit.

    This exercises the REAL exit decision (esper.scripts.train._exit_nonzero_if_training_failed),
    which main() calls after the TUI/finally. A background-thread crash cannot propagate to the
    main thread, so without this the process would exit 0 and report success to the shell/CI.
    """

    def test_captured_crash_drives_nonzero_exit(self):
        from esper.scripts.train import (
            _exit_nonzero_if_training_failed,
            _run_training_capturing_errors,
        )

        training_error: list[str | None] = [None]
        shutdown_event = threading.Event()

        def _crashing_run_training() -> None:
            raise RuntimeError("training exploded")

        # The full chain: the background wrapper captures the crash...
        _run_training_capturing_errors(_crashing_run_training, training_error, shutdown_event)
        assert training_error[0] is not None and "RuntimeError" in training_error[0]

        # ...and the post-TUI exit decision turns it into a non-zero exit (SystemExit(1)).
        with pytest.raises(SystemExit) as exc_info:
            _exit_nonzero_if_training_failed(training_error)
        assert exc_info.value.code == 1

    def test_clean_run_does_not_exit(self):
        from esper.scripts.train import _exit_nonzero_if_training_failed

        # No recorded error -> no SystemExit (clean runs exit 0).
        assert _exit_nonzero_if_training_failed([None]) is None


# ---------------------------------------------------------------------------
# BUG-5 (train.py, P1-c): a degraded REQUESTED telemetry backend must fail loud
# ---------------------------------------------------------------------------

class TestRequestedTelemetryBackendHealthGate:
    """The REAL train.py shutdown gate must fail when a REQUESTED backend degraded.

    Exercises esper.scripts.train._degraded_requested_backends directly (the body
    of the shutdown health check), so it fails if the gate is reverted. The health
    snapshot is the same shape NissaHub.get_health_snapshot() returns.
    """

    def _snapshot_with(self, backend_drops: dict[str, int]) -> dict:
        """Build a health snapshot whose backend_stats carry the given drop counts."""
        hub = NissaHub()
        for name in backend_drops:
            hub.add_backend(_NoopBackend())
            # Rename the just-added worker so backend_stats keys are deterministic.
            hub._backend_workers[-1]._name = name
        snapshot = hub.get_health_snapshot()
        for name, drops in backend_drops.items():
            snapshot["backend_stats"][name]["dropped_events"] = drops
        snapshot["backend_dropped_events"] = sum(backend_drops.values())
        snapshot["dropped_events"] = snapshot["hub_dropped_events"] + sum(backend_drops.values())
        snapshot["status"] = "degraded" if snapshot["dropped_events"] > 0 else "healthy"
        hub.close()
        return snapshot

    def test_requested_backend_dropped_events_is_flagged(self):
        from esper.scripts.train import _degraded_requested_backends

        snapshot = self._snapshot_with({"FileOutput": 3})
        degraded = _degraded_requested_backends(snapshot, requested_backend_names={"FileOutput"})
        assert degraded == {"FileOutput"}, (
            f"a requested backend that dropped events must be flagged; got {degraded!r}"
        )

    def test_non_requested_backend_drops_are_not_flagged(self):
        from esper.scripts.train import _degraded_requested_backends

        # The always-on research collector dropped events, but it was not a
        # requested streaming sink -> must NOT trip the gate.
        snapshot = self._snapshot_with({"KarnCollector": 5})
        degraded = _degraded_requested_backends(snapshot, requested_backend_names=set())
        assert degraded == set(), (
            f"non-requested backend drops must not fail the run; got {degraded!r}"
        )

    def test_clean_requested_backend_is_not_flagged(self):
        from esper.scripts.train import _degraded_requested_backends

        snapshot = self._snapshot_with({"FileOutput": 0})
        degraded = _degraded_requested_backends(snapshot, requested_backend_names={"FileOutput"})
        assert degraded == set(), (
            f"a healthy requested backend must not be flagged; got {degraded!r}"
        )

    def test_missing_requested_backend_is_loud_not_silent(self):
        from esper.scripts.train import _degraded_requested_backends

        # A requested backend that is absent from backend_stats is a real bug
        # (CLAUDE.md no-bug-hiding) -> must raise, NOT be silently treated as healthy.
        snapshot = self._snapshot_with({"FileOutput": 0})
        with pytest.raises(KeyError):
            _degraded_requested_backends(
                snapshot, requested_backend_names={"FileOutput", "WandbBackend"}
            )
