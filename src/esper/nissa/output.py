"""Nissa Output - Telemetry output backends.

Output backends receive carbon copies of telemetry events from all domains
and route them to configured destinations (console, files, metrics systems).

Usage:
    from esper.nissa.output import NissaHub, ConsoleOutput, FileOutput

    hub = NissaHub()
    hub.add_backend(ConsoleOutput())
    hub.add_backend(FileOutput("telemetry.jsonl"))

    # Emit events
    hub.emit(event)
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from esper.leyline import OutputBackend, TelemetryEvent
from esper.leyline.telemetry import (
    CheckpointLoadedPayload,
    EpochCompletedPayload,
    BatchEpochCompletedPayload,
    AnomalyDetectedPayload,
    SeedGerminatedPayload,
    SeedStageChangedPayload,
    SeedFossilizedPayload,
    SeedPrunedPayload,
    PPOUpdatePayload,
    GovernorRollbackPayload,
    TamiyoInitiatedPayload,
)

_logger = logging.getLogger(__name__)


def _join_with_timeout(q: queue.Queue[Any], timeout: float | None) -> bool:
    """Join queue with optional timeout.

    Waits for all items in the queue to be processed (task_done() called).
    Python's queue.Queue.join() has no timeout parameter, so we implement
    timed waiting using the internal condition variable.

    Args:
        q: The queue to wait on.
        timeout: Maximum seconds to wait. None means wait forever.

    Returns:
        True if queue drained successfully, False if timeout expired.
    """
    if timeout is None:
        q.join()
        return True

    deadline = time.monotonic() + timeout
    with q.all_tasks_done:
        while q.unfinished_tasks:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            q.all_tasks_done.wait(timeout=remaining)
    return True


def _payload_to_dict(payload: Any) -> Any:
    """Convert a TelemetryEvent.data payload into a JSON-friendly structure.

    PERF: Prefer payload.to_dict() when present to avoid dataclasses.asdict() deep-copy overhead
    on already-JSONable containers (slot_states, per-seed dicts, etc.).
    """
    if payload is None:
        return None
    if is_dataclass(payload) and not isinstance(payload, type):
        # hasattr AUTHORIZED by operator on 2026-01-01 00:00:00 UTC
        # Justification: Serialization - prefer to_dict() for typed payloads that provide it
        if hasattr(payload, "to_dict") and callable(payload.to_dict):
            return payload.to_dict()
        return asdict(payload)
    return payload


def _telemetry_event_to_dict(event: TelemetryEvent) -> dict[str, Any]:
    """Convert TelemetryEvent to a dict for JSON serialization (fast path)."""
    # TelemetryEvent.event_type is TelemetryEventType (Enum) - always has .name
    # TelemetryEvent.timestamp is datetime - always has .isoformat()
    event_type = event.event_type.name
    timestamp = event.timestamp.isoformat()

    return {
        "event_id": event.event_id,
        "event_type": event_type,
        "timestamp": timestamp,
        "seed_id": event.seed_id,
        "slot_id": event.slot_id,
        "epoch": event.epoch,
        "group_id": event.group_id,
        "message": event.message,
        "data": _payload_to_dict(event.data),
        "severity": event.severity,
    }


class BackendWorker:
    """Worker thread that processes events for a single backend.

    Isolates slow backends from fast ones by giving each backend its own
    queue and processing thread. Prevents head-of-line blocking where one
    slow backend (e.g. disk I/O) stalls all others (e.g. in-memory aggregation).

    Thread-safe: enqueue() can be called from main worker thread while
    _worker_loop() runs in background thread.
    """

    def __init__(
        self,
        backend: "OutputBackend",
        max_queue_size: int = 100,
        name: str | None = None,
    ):
        """Initialize backend worker.

        Args:
            backend: The output backend to wrap.
            max_queue_size: Maximum events to buffer before dropping.
            name: Thread name (defaults to backend class name).
        """
        self._backend = backend
        self._queue: queue.Queue[TelemetryEvent | None] = queue.Queue(maxsize=max_queue_size)
        self._name = name or backend.__class__.__name__
        self._thread = threading.Thread(
            target=self._worker_loop,
            name=f"BackendWorker-{self._name}",
            daemon=True,
        )
        self._stopped = False
        self._stop_lock = threading.Lock()  # Guards _stopped to prevent sentinel race
        self._dropped_events = 0
        self._processed_events = 0
        self._total_processing_time = 0.0
        self._thread.start()

    def enqueue(self, event: TelemetryEvent) -> None:
        """Enqueue event for processing (non-blocking).

        Args:
            event: The telemetry event to process.
        """
        # Use lock to prevent race with stop() - ensures no events enqueued after sentinel
        with self._stop_lock:
            if self._stopped:
                return

            try:
                # Non-blocking put - if backend is slow and queue is full, drop event
                self._queue.put_nowait(event)
            except queue.Full:
                self._dropped_events += 1
                if self._dropped_events % 100 == 1:  # Log every 100th drop
                    _logger.warning(
                        f"Backend {self._name} queue full, dropped {self._dropped_events} events"
                    )

    def get_stats(self) -> dict[str, int | float]:
        """Get performance statistics for this backend worker.

        Returns:
            Dictionary with processed count, dropped count, avg processing time.
        """
        avg_time = (
            self._total_processing_time / self._processed_events
            if self._processed_events > 0
            else 0.0
        )
        return {
            "processed_events": self._processed_events,
            "dropped_events": self._dropped_events,
            "avg_processing_time_ms": avg_time * 1000,
            "total_processing_time_s": self._total_processing_time,
        }

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the worker thread gracefully.

        Waits for all pending events to be processed before stopping.

        Thread-safety: Sets _stopped=True under lock BEFORE putting sentinel,
        ensuring no events can be enqueued after the sentinel. This fixes the
        race where events could land after sentinel but before _stopped=True.

        Args:
            timeout: Maximum time to wait for worker to finish. The timeout
                is split between queue drain and thread join phases.
        """
        # Acquire lock and set _stopped atomically - prevents enqueue() race
        with self._stop_lock:
            if self._stopped:
                return
            self._stopped = True

        deadline = time.monotonic() + timeout

        # Wait for queue to drain (all pending events processed)
        # _stopped is already True so no new events can be enqueued
        try:
            drained = _join_with_timeout(self._queue, timeout)
            if not drained:
                _logger.warning(
                    f"Backend {self._name} queue drain timed out after {timeout}s, "
                    f"forcing shutdown (some events may be lost)"
                )
        except Exception:
            _logger.warning(
                "Queue join failed during %s shutdown (worker may have died)",
                self._name,
                exc_info=True,
            )

        # Send shutdown signal - no race possible since _stopped=True prevents new enqueues
        try:
            self._queue.put(None, timeout=1.0)
        except queue.Full:
            _logger.warning(f"Queue full during {self._name} shutdown, forcing stop")

        # Wait for worker to finish with remaining time budget
        remaining = max(0.1, deadline - time.monotonic())
        if self._thread.is_alive():
            self._thread.join(timeout=remaining)
            if self._thread.is_alive():
                _logger.warning(f"Backend worker {self._name} did not stop within {timeout}s")

        # Log performance stats on shutdown
        stats = self.get_stats()
        if stats["processed_events"] > 0:
            _logger.debug(
                f"Backend {self._name}: processed={stats['processed_events']}, "
                f"dropped={stats['dropped_events']}, "
                f"avg_time={stats['avg_processing_time_ms']:.2f}ms"
            )

    def _worker_loop(self) -> None:
        """Background worker loop that processes events from the queue."""
        while True:
            try:
                event = self._queue.get(timeout=1.0)
                if event is None:  # Shutdown signal
                    self._queue.task_done()
                    break

                # Track processing time
                start_time = time.time()
                try:
                    self._backend.emit(event)
                    self._processed_events += 1
                except Exception:
                    _logger.exception("Error in backend %s", self._name)
                finally:
                    elapsed = time.time() - start_time
                    self._total_processing_time += elapsed

                self._queue.task_done()
            except queue.Empty:
                continue
            except Exception:
                _logger.exception("Unexpected error in backend worker %s", self._name)


# OutputBackend Protocol now lives in leyline (re-exported above)


class ConsoleOutput(OutputBackend):
    """Output telemetry events to console/stdout.

    Args:
        verbose: If True, show full event details. If False, show summary.
        use_color: If True, use ANSI color codes for formatting.
        min_severity: Minimum severity level to display (debug, info, warning, error, critical).
    """

    # Severity levels ordered by increasing importance
    _SEVERITY_ORDER = {"debug": 0, "info": 1, "warning": 2, "error": 3, "critical": 4}

    def __init__(self, verbose: bool = False, use_color: bool = True, min_severity: str = "info"):
        self.verbose = verbose
        self.use_color = use_color
        self.min_severity = min_severity

    def start(self) -> None:
        """No-op for console output."""
        pass

    def close(self) -> None:
        """No-op for console output."""
        pass

    def emit(self, event: TelemetryEvent) -> None:
        """Emit event to console if it meets minimum severity threshold."""
        # Filter by severity - debug events are suppressed by default
        if self._SEVERITY_ORDER.get(event.severity, 1) < self._SEVERITY_ORDER.get(self.min_severity, 1):
            return

        if self.verbose:
            self._emit_verbose(event)
        else:
            self._emit_summary(event)

    def _emit_summary(self, event: TelemetryEvent) -> None:
        """Emit compact summary to console."""
        timestamp = event.timestamp.strftime("%H:%M:%S") if event.timestamp else datetime.now().strftime("%H:%M:%S")
        # hasattr AUTHORIZED by operator on 2025-11-29 15:05:24 UTC
        # Justification: Serialization - handle both enum and string event_type values
        event_type = event.event_type.name if hasattr(event.event_type, 'name') else str(event.event_type)
        seed_id = event.seed_id or "system"

        # Format message based on event type
        if event_type == "EPOCH_COMPLETED":
            if event.data is None:
                _logger.warning("EPOCH_COMPLETED event has no data payload")
                return
            # Handle typed payload
            if isinstance(event.data, EpochCompletedPayload):
                loss = event.data.val_loss
                acc = event.data.val_accuracy
                epoch = event.epoch if event.epoch is not None else "?"
                print(f"[{timestamp}] {seed_id} | Epoch {epoch}: loss={loss} acc={acc}")
        elif event_type.startswith("SEED_"):
            if event.data is None:
                _logger.warning("SEED_%s event has no data payload", event_type)
                return

            if event_type == "SEED_GERMINATED":
                if isinstance(event.data, SeedGerminatedPayload):
                    blueprint_id = event.data.blueprint_id
                    params = event.data.params
                    msg = f"Germinated ({blueprint_id}, {params/1000:.1f}K params)"
                else:
                    msg = event.message or event_type
            elif event_type == "SEED_STAGE_CHANGED":
                if isinstance(event.data, SeedStageChangedPayload):
                    from_stage = event.data.from_stage
                    to_stage = event.data.to_stage
                    msg = f"Stage transition: {from_stage} \u2192 {to_stage}"
                else:
                    msg = event.message or event_type
            elif event_type == "SEED_FOSSILIZED":
                if isinstance(event.data, SeedFossilizedPayload):
                    blueprint_id = event.data.blueprint_id
                    improvement = event.data.improvement
                    msg = f"Fossilized ({blueprint_id}, \u0394acc {improvement:+.2f}%)"
                else:
                    msg = event.message or event_type
            elif event_type == "SEED_PRUNED":
                if isinstance(event.data, SeedPrunedPayload):
                    blueprint_id = event.data.blueprint_id or "?"
                    improvement = event.data.improvement
                    reason = event.data.reason
                    reason_str = f" ({reason})" if reason else ""
                    msg = f"Pruned ({blueprint_id}, \u0394acc {improvement:+.2f}%){reason_str}"
                else:
                    msg = event.message or event_type
            else:
                msg = event.message or event_type
            print(f"[{timestamp}] {seed_id} | {msg}")
        elif event_type == "GOVERNOR_ROLLBACK":
            payload = event.data
            if not isinstance(payload, GovernorRollbackPayload):
                _logger.error("GOVERNOR_ROLLBACK event missing typed payload")
                return
            loss_str = f"{payload.loss_at_panic:.4f}" if payload.loss_at_panic is not None else "?"
            threshold_str = f"{payload.loss_threshold:.4f}" if payload.loss_threshold is not None else "?"
            panics_str = str(payload.consecutive_panics) if payload.consecutive_panics is not None else "?"
            print(f"[{timestamp}] GOVERNOR | 🚨 ROLLBACK: {payload.reason} (loss={loss_str}, threshold={threshold_str}, panics={panics_str})")
        elif event_type == "BATCH_EPOCH_COMPLETED":
            if event.data is None:
                _logger.warning("BATCH_EPOCH_COMPLETED event has no data payload")
                return
            # Handle typed payload
            if isinstance(event.data, BatchEpochCompletedPayload):
                batch_idx = event.data.batch_idx
                episodes = event.data.episodes_completed
                total = event.data.total_episodes
                avg_acc = event.data.avg_accuracy
                rolling_acc = event.data.rolling_accuracy
                avg_reward = event.data.avg_reward
                env_accs = event.data.env_accuracies or ()
                env_acc_str = ", ".join(f"{a:.1f}%" for a in env_accs) if env_accs else ""
                print(f"[{timestamp}] BATCH {batch_idx} | Episodes {episodes}/{total}")
                if env_acc_str:
                    print(f"[{timestamp}]   Env accs: [{env_acc_str}]")
                print(f"[{timestamp}]   Avg: {avg_acc:.1f}% (rolling: {rolling_acc:.1f}%), reward: {avg_reward:.1f}")
        elif event_type == "CHECKPOINT_LOADED":
            if not isinstance(event.data, CheckpointLoadedPayload):
                raise TypeError(f"CHECKPOINT_LOADED event has invalid payload type: {type(event.data)}")
            payload = event.data
            if payload.source:
                acc_str = f" (acc={payload.avg_accuracy:.1f}%)" if payload.avg_accuracy is not None else ""
                print(f"[{timestamp}] CHECKPOINT | Loaded {payload.source}{acc_str}")
            else:
                print(f"[{timestamp}] CHECKPOINT | Loaded from {payload.path} (resuming at episode {payload.start_episode})")
        elif event_type == "TAMIYO_INITIATED":
            if event.data is None:
                _logger.warning("TAMIYO_INITIATED event has no data payload")
                return
            # TAMIYO_INITIATED always uses typed payload
            if not isinstance(event.data, TamiyoInitiatedPayload):
                raise TypeError(f"TAMIYO_INITIATED event has invalid payload type: {type(event.data)}")
            payload = event.data
            env_str = f"env{payload.env_id}"
            if payload.stabilization_epochs == 0:
                print(f"[{timestamp}] {env_str} | Host stabilized at epoch {payload.epoch} - germination now allowed")
            else:
                print(f"[{timestamp}] {env_str} | Host stabilized at epoch {payload.epoch} ({payload.stable_count}/{payload.stabilization_epochs} stable) - germination now allowed")
        elif event_type == "PPO_UPDATE_COMPLETED":
            if event.data is None:
                _logger.warning("PPO_UPDATE_COMPLETED event has no data payload")
                return
            if isinstance(event.data, PPOUpdatePayload):
                if event.data.skipped:
                    print(f"[{timestamp}] PPO | Update skipped")
                else:
                    policy_loss = event.data.policy_loss
                    value_loss = event.data.value_loss
                    entropy = event.data.entropy
                    entropy_coef = event.data.entropy_coef or 0.0
                    print(f"[{timestamp}] PPO | policy={policy_loss:.4f}, value={value_loss:.4f}, entropy={entropy:.3f} (coef={entropy_coef:.4f})")
        elif event_type in ("RATIO_EXPLOSION_DETECTED", "RATIO_COLLAPSE_DETECTED",
                           "VALUE_COLLAPSE_DETECTED", "NUMERICAL_INSTABILITY_DETECTED",
                           "GRADIENT_ANOMALY", "GRADIENT_PATHOLOGY_DETECTED"):
            if isinstance(event.data, AnomalyDetectedPayload):
                anomaly_name = event_type.replace("_DETECTED", "").replace("_", " ").title()
                print(f"[{timestamp}] ⚠️  ANOMALY | {anomaly_name} at episode {event.data.episode}: {event.data.detail}")
        else:
            msg = event.message or event_type
            print(f"[{timestamp}] {seed_id} | {msg}")

    def _emit_verbose(self, event: TelemetryEvent) -> None:
        """Emit full event details to console."""
        # Convert event to dict for display
        event_dict = self._event_to_dict(event)
        print(json.dumps(event_dict, indent=2, default=str))

    def _event_to_dict(self, event: TelemetryEvent) -> dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return _telemetry_event_to_dict(event)


class FileOutput(OutputBackend):
    """Output telemetry events to a file in JSONL format.

    Args:
        path: Path to output file. Events are appended in JSONL format.
        buffer_size: Number of events to buffer before flushing to disk.
    """

    def __init__(self, path: str | Path, buffer_size: int = 10):
        self.path = Path(path)
        self.buffer_size = buffer_size
        self._buffer: list[dict[str, Any]] = []

        # Ensure parent directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Open file in append mode
        self._file = open(self.path, 'a', encoding='utf-8')

    def start(self) -> None:
        """No-op for file output (file opened in __init__)."""
        pass

    def emit(self, event: TelemetryEvent) -> None:
        """Emit event to file."""
        event_dict = self._event_to_dict(event)

        self._buffer.append(event_dict)

        if len(self._buffer) >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        """Flush buffered events to disk."""
        for event_dict in self._buffer:
            json_line = json.dumps(event_dict, default=str)
            self._file.write(json_line + '\n')
        self._file.flush()
        self._buffer.clear()

    def close(self) -> None:
        """Close the file backend."""
        # hasattr AUTHORIZED by operator on 2025-11-29 15:05:24 UTC
        # Justification: Cleanup guard - defensive check in close() to avoid errors
        if hasattr(self, '_file') and not self._file.closed:
            self.flush()
            self._file.close()

    def _event_to_dict(self, event: TelemetryEvent) -> dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return _telemetry_event_to_dict(event)

    def __del__(self) -> None:
        """Ensure file is closed on deletion."""
        # hasattr AUTHORIZED by operator on 2025-11-29 15:05:24 UTC
        # Justification: Cleanup guard - defensive check in __del__ to avoid errors during teardown
        if hasattr(self, '_file') and not self._file.closed:
            self.close()


class DirectoryOutput(OutputBackend):
    """Output telemetry events to a timestamped directory.

    Creates a subdirectory with format `telemetry_YYYY-MM-DD_HHMMSS/` and
    writes events to `events.jsonl` inside it.

    Args:
        base_path: Base directory where timestamped subdirectory will be created.
        buffer_size: Number of events to buffer before flushing to disk.
    """

    def __init__(self, base_path: str | Path, buffer_size: int = 10):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Generate timestamped subdirectory
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self._output_dir = self.base_path / f"telemetry_{timestamp}"
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Create internal FileOutput for actual writing
        self._file_output = FileOutput(self._output_dir / "events.jsonl", buffer_size)

    @property
    def output_dir(self) -> Path:
        """Return the full path to the timestamped output directory."""
        return self._output_dir

    def start(self) -> None:
        """No-op for directory output (directory created in __init__)."""
        pass

    def emit(self, event: TelemetryEvent) -> None:
        """Emit event to the directory's events.jsonl file."""
        self._file_output.emit(event)

    def flush(self) -> None:
        """Flush buffered events to disk."""
        self._file_output.flush()

    def close(self) -> None:
        """Close the directory backend."""
        self._file_output.close()


class NissaHub:
    """Central telemetry hub that routes events to multiple backends.

    The hub receives carbon copies of events from all domains and
    distributes them to all registered output backends.

    Performance:
        Emission is asynchronous with parallel backend processing. Events
        are placed in a main queue, then fanned out to per-backend worker
        threads. This ensures slow backends (e.g. disk I/O) don't block
        fast ones (e.g. in-memory aggregation), preventing head-of-line
        blocking under burst loads.

        Architecture:
            1. Main worker: Dequeues events from central queue
            2. Fan-out: Distributes events to all backend workers
            3. Backend workers: Process events independently in parallel
            4. Back-pressure: Each backend has bounded queue (drops on overflow)

    Lifecycle Contract:
        - add_backend(): Starts backend immediately; raises RuntimeError if hub is closed
        - emit(): Places event in queue; silently drops (with warning) if hub is closed
        - close(): Idempotent; flushes queue, closes all backends, and stops worker
        - reset(): Closes all backends, clears list, and reopens hub for reuse

        This contract is shared with KarnCollector for consistency across telemetry systems.

    Usage:
        hub = NissaHub()
        hub.add_backend(ConsoleOutput())
        hub.add_backend(FileOutput("telemetry.jsonl"))

        # Emit events
        hub.emit(event)
    """

    def __init__(self, max_queue_size: int = 1000, backend_queue_size: int = 100):
        self._backends: list[OutputBackend] = []
        self._backend_workers: list[BackendWorker] = []
        self._backend_queue_size = backend_queue_size
        self._closed = False  # Idempotency flag for close()
        self._emit_after_close_warned = False
        self._queue: queue.Queue[TelemetryEvent | None] = queue.Queue(maxsize=max_queue_size)
        self._worker_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._dropped_events = 0

    def _start_worker(self) -> None:
        """Start the background worker thread if not already running."""
        with self._lock:
            if self._worker_thread is None or not self._worker_thread.is_alive():
                self._worker_thread = threading.Thread(
                    target=self._worker_loop, name="NissaHubWorker", daemon=True
                )
                self._worker_thread.start()

    def _worker_loop(self) -> None:
        """Background worker loop that fans out events to all backend workers.

        This is the main multiplexer: it dequeues events from the central queue
        and distributes them to all per-backend worker queues in parallel.
        Backends process independently, so slow backends don't block fast ones.

        Thread-safety: Snapshots _backend_workers under lock to prevent races
        with add_backend()/remove_backend() modifying the list during iteration.
        """
        while True:
            try:
                event = self._queue.get(timeout=1.0)
                if event is None:  # Shutdown signal
                    self._queue.task_done()
                    break

                # Snapshot workers under lock to prevent race with add/remove
                with self._lock:
                    workers = list(self._backend_workers)

                # Fan out to all backend workers (outside lock to avoid blocking)
                for worker in workers:
                    worker.enqueue(event)

                self._queue.task_done()
            except queue.Empty:
                continue
            except Exception:
                _logger.exception("Unexpected error in NissaHub worker loop")

    def add_backend(self, backend: OutputBackend) -> None:
        """Add an output backend to the hub.

        The backend is started immediately and wrapped in a BackendWorker
        with its own processing thread. This ensures slow backends don't
        block fast ones.

        Thread-safety: Acquires lock when modifying backend lists.

        Args:
            backend: The output backend to add.

        Raises:
            RuntimeError: If the hub has been closed. Use reset() to reopen.
            Exception: If backend.start() fails, the original exception is
                logged and re-raised so callers can detect misconfiguration.
        """
        if self._closed:
            raise RuntimeError(
                "Cannot add backend to closed NissaHub. "
                "Call reset() to reopen the hub before adding backends."
            )
        try:
            backend.start()

            # Create worker thread for this backend
            worker = BackendWorker(
                backend=backend,
                max_queue_size=self._backend_queue_size,
            )

            # Add to lists under lock
            with self._lock:
                self._backends.append(backend)
                self._backend_workers.append(worker)

            self._start_worker()
        except Exception:
            _logger.exception(f"Failed to start backend {backend.__class__.__name__}, not adding")
            raise

    def remove_backend(self, backend: OutputBackend) -> None:
        """Remove an output backend from the hub.

        Stops the backend's worker thread and closes the backend.

        Thread-safety: Removes from lists under lock BEFORE stopping worker,
        ensuring the worker loop won't target this worker after removal.

        Args:
            backend: The output backend to remove.
        """
        worker = None

        # Remove from lists under lock - must happen BEFORE stop()
        # so worker loop snapshot won't include this worker
        with self._lock:
            if backend not in self._backends:
                return
            idx = self._backends.index(backend)
            self._backends.remove(backend)
            if idx < len(self._backend_workers):
                worker = self._backend_workers.pop(idx)

        # Stop worker outside lock (can take time, don't block other operations)
        if worker is not None:
            worker.stop()

        # Close the backend
        try:
            backend.close()
        except Exception:
            _logger.exception("Error closing backend %s", backend.__class__.__name__)

    def emit(self, event: TelemetryEvent) -> None:
        """Emit a telemetry event to all backends (asynchronously).

        Args:
            event: The telemetry event to emit.

        Note:
            Does nothing if the hub has been closed. This prevents sending
            events to backends that have already been shut down.
        """
        if self._closed:
            if not self._emit_after_close_warned:
                _logger.warning("emit() called on closed NissaHub (event dropped)")
                self._emit_after_close_warned = True
            return

        try:
            # Non-blocking put to avoid stalling training if queue is full.
            # If queue is full, we drop the event and warn.
            self._queue.put_nowait(event)
        except queue.Full:
            self._dropped_events += 1
            # Log first drop immediately, then throttle to avoid hitching under overload.
            if self._dropped_events == 1 or self._dropped_events % 1000 == 0:
                _logger.warning(
                    "NissaHub queue full, dropped %d telemetry events",
                    self._dropped_events,
                )

    def flush(self, timeout: float | None = None) -> bool:
        """Block until all queued events have been processed.

        Waits for both the main queue AND all backend worker queues to drain.
        This ensures the documented contract: "all events have been delivered
        to backends before proceeding".

        Useful for tests or shutdown sequences where you need to ensure
        all events have been delivered to backends before proceeding.

        Args:
            timeout: Maximum time to wait in seconds. None means wait forever.

        Returns:
            True if all queues drained successfully, False if timeout expired.
        """
        if self._closed:
            return True
        # Only wait if there's actually a worker running to process events
        if self._worker_thread is None or not self._worker_thread.is_alive():
            return True

        deadline = time.monotonic() + timeout if timeout is not None else None

        # Stage 1: Wait for main queue to drain (fan-out complete)
        # join() blocks until all items in the queue have been processed
        # (i.e., task_done() has been called for each get())
        if not _join_with_timeout(self._queue, timeout):
            _logger.warning(f"NissaHub main queue flush timed out after {timeout}s")
            return False

        # Stage 2: Wait for all backend worker queues to drain (processing complete)
        # Without this, flush() would return while backend workers are still emitting
        for worker in self._backend_workers:
            remaining = deadline - time.monotonic() if deadline is not None else None
            if remaining is not None and remaining <= 0:
                _logger.warning(f"NissaHub flush timed out waiting for backend {worker._name}")
                return False
            try:
                if not _join_with_timeout(worker._queue, remaining):
                    _logger.warning(f"Backend {worker._name} flush timed out")
                    return False
            except Exception:
                _logger.warning(
                    "Queue join failed for %s during flush (worker may have died)",
                    worker._name,
                    exc_info=True,
                )

        return True

    def get_backend_stats(self) -> dict[str, dict[str, int | float]]:
        """Get performance statistics for all backend workers.

        Returns:
            Dictionary mapping backend name to its stats.
        """
        stats = {}
        for worker in self._backend_workers:
            stats[worker._name] = worker.get_stats()
        return stats

    def reset(self) -> None:
        """Reset the hub: close all backends and clear the list.

        Use this to ensure clean state between test runs or episodes.

        Warning:
            NOT THREAD-SAFE. Must be called when no other threads are emitting
            events. Intended for test cleanup or between training runs, not
            during active training. Calling reset() while emit() is running
            concurrently may cause dropped events or backend errors.
        """
        self.close()
        self._backends.clear()
        self._backend_workers.clear()
        self._closed = False  # Allow hub to be reused after reset
        self._emit_after_close_warned = False
        # Create fresh queue to ensure no stale events from previous runs
        self._queue = queue.Queue(maxsize=self._queue.maxsize)
        self._worker_thread = None  # Will be recreated on next add_backend()
        self._dropped_events = 0

    def close(self, timeout: float = 10.0) -> None:
        """Close all backends (idempotent).

        Safe to call multiple times - only closes backends once.
        Flushes pending events before shutting down workers.

        Shutdown order:
            1. Stop accepting new events (set _closed flag)
            2. Drain main queue and stop main worker
            3. Stop all backend workers (drains their queues)
            4. Close all backends

        Args:
            timeout: Maximum time to wait for graceful shutdown. The timeout
                is split across queue drain, worker stop, and backend close phases.
        """
        if self._closed:
            return

        deadline = time.monotonic() + timeout

        # CRITICAL: Set _closed FIRST to prevent new events being enqueued
        # after the sentinel. This fixes the race where emit() could add events
        # after the sentinel, causing those events to be silently dropped.
        self._closed = True

        # Signal main worker to stop and wait for it to finish
        if self._worker_thread is not None and self._worker_thread.is_alive():
            try:
                # Drain any events already in the queue before sending sentinel.
                # This ensures the "flushes pending events" guarantee in the docstring.
                remaining = max(0.1, deadline - time.monotonic())
                if not _join_with_timeout(self._queue, remaining):
                    _logger.warning(
                        "Hub main queue drain timed out during close, some events may be lost"
                    )

                # Add None to queue as sentinel for worker shutdown.
                # Use blocking put with timeout to ensure signal is received.
                self._queue.put(None, timeout=2.0)
                remaining = max(0.1, deadline - time.monotonic())
                self._worker_thread.join(timeout=remaining)
            except (queue.Full, RuntimeError):
                _logger.warning(
                    "Hub shutdown interrupted (worker may be dead or queue unreachable)",
                    exc_info=True,
                )

        # Stop all backend workers (this drains their queues)
        # Split remaining time among workers
        n_workers = len(self._backend_workers)
        for i, worker in enumerate(self._backend_workers):
            remaining = max(0.1, deadline - time.monotonic())
            # Give each worker a fair share of remaining time
            per_worker = remaining / max(1, n_workers - i)
            try:
                worker.stop(timeout=per_worker)
            except Exception:
                _logger.exception("Error stopping backend worker %s", worker._name)

        # Close all backends
        for backend in self._backends:
            try:
                backend.close()
            except Exception:
                _logger.exception("Error closing backend %s", backend.__class__.__name__)

    def __del__(self) -> None:
        """Ensure all backends are closed on deletion."""
        self.close()


# Global hub instance
_global_hub: NissaHub | None = None


def get_hub() -> NissaHub:
    """Get or create the global NissaHub instance.

    Returns:
        The global NissaHub singleton.
    """
    global _global_hub
    if _global_hub is None:
        _global_hub = NissaHub()
    return _global_hub


def reset_hub() -> None:
    """Reset the global NissaHub instance.

    Closes all existing backends and clears the backend list, but keeps the
    singleton instance alive (allows reuse without creating a new hub).
    Useful for test cleanup.
    """
    global _global_hub
    if _global_hub is not None:
        _global_hub.reset()


def emit(event: TelemetryEvent) -> None:
    """Emit a telemetry event to the global hub.

    Convenience function for emitting to the global hub.

    Args:
        event: The telemetry event to emit.
    """
    get_hub().emit(event)


__all__ = [
    "OutputBackend",
    "ConsoleOutput",
    "FileOutput",
    "DirectoryOutput",
    "NissaHub",
    "get_hub",
    "reset_hub",
    "emit",
]
