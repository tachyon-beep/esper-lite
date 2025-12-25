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
from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path

from esper.leyline import TelemetryEvent
from esper.leyline.telemetry import (
    EpochCompletedPayload,
    BatchEpochCompletedPayload,
    AnomalyDetectedPayload,
)

_logger = logging.getLogger(__name__)


class OutputBackend(ABC):
    """Base class for telemetry output backends."""

    def start(self) -> None:
        """Start the backend (e.g., open files, start threads)."""
        pass

    @abstractmethod
    def emit(self, event: TelemetryEvent) -> None:
        """Emit a telemetry event to this backend.

        Args:
            event: The telemetry event to emit.
        """
        pass

    def close(self) -> None:
        """Close the backend and release resources."""
        pass


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
            # Handle typed payload or dict fallback
            if isinstance(event.data, EpochCompletedPayload):
                loss = event.data.val_loss
                acc = event.data.val_accuracy
            else:
                if event.data is None:
                    _logger.warning("EPOCH_COMPLETED event has no data payload")
                    return
                loss = event.data.get("val_loss", "?")
                acc = event.data.get("val_accuracy", "?")
            epoch = event.epoch if event.epoch is not None else "?"
            print(f"[{timestamp}] {seed_id} | Epoch {epoch}: loss={loss} acc={acc}")
        elif "COMMAND" in event_type:
            if event.data is None:
                _logger.warning("COMMAND event has no data payload")
                return
            action = event.data.get("action", "unknown")
            print(f"[{timestamp}] {seed_id} | Command: {action}")
        elif event_type.startswith("SEED_"):
            if event.data is None:
                _logger.warning("SEED_%s event has no data payload", event_type)
                return
            data = event.data
            if event_type == "SEED_GERMINATED":
                blueprint_id = data.get("blueprint_id", "?")
                params = data.get("params")
                if isinstance(params, (int, float)):
                    msg = f"Germinated ({blueprint_id}, {params/1000:.1f}K params)"
                else:
                    msg = f"Germinated ({blueprint_id})"
            elif event_type == "SEED_STAGE_CHANGED":
                from_stage = data.get("from", "?")
                to_stage = data.get("to", "?")
                msg = f"Stage transition: {from_stage} \u2192 {to_stage}"
            elif event_type == "SEED_FOSSILIZED":
                blueprint_id = data.get("blueprint_id", "?")
                improvement = data.get("improvement")
                if isinstance(improvement, (int, float)):
                    msg = f"Fossilized ({blueprint_id}, \u0394acc {improvement:+.2f}%)"
                else:
                    msg = f"Fossilized ({blueprint_id})"
            elif event_type == "SEED_PRUNED":
                blueprint_id = data.get("blueprint_id", "?")
                improvement = data.get("improvement")
                reason = data.get("reason")
                reason_str = f" ({reason})" if reason else ""
                if isinstance(improvement, (int, float)):
                    msg = f"Pruned ({blueprint_id}, \u0394acc {improvement:+.2f}%){reason_str}"
                else:
                    msg = f"Pruned ({blueprint_id}){reason_str}"
            else:
                msg = event.message or event_type
            print(f"[{timestamp}] {seed_id} | {msg}")
        elif event_type == "GOVERNOR_ROLLBACK":
            if event.data is None:
                _logger.warning("GOVERNOR_ROLLBACK event has no data payload")
                return
            data = event.data
            reason = data.get("reason", "unknown")
            loss = data.get("loss_at_panic", "?")
            threshold = data.get("loss_threshold", "?")
            panics = data.get("consecutive_panics", "?")
            if isinstance(loss, float):
                loss = f"{loss:.4f}"
            if isinstance(threshold, float):
                threshold = f"{threshold:.4f}"
            print(f"[{timestamp}] GOVERNOR | 🚨 ROLLBACK: {reason} (loss={loss}, threshold={threshold}, panics={panics})")
        elif event_type == "GOVERNOR_PANIC":
            # TODO: [DEAD CODE] - This formatting code for GOVERNOR_PANIC is unreachable
            # because these events are never emitted. See: leyline/telemetry.py dead event TODOs.
            if event.data is None:
                _logger.warning("GOVERNOR_PANIC event has no data payload")
                return
            data = event.data
            loss = data.get("current_loss", "?")
            panics = data.get("consecutive_panics", 0)
            if isinstance(loss, float):
                loss = f"{loss:.4f}"
            print(f"[{timestamp}] GOVERNOR | ⚠️  PANIC #{panics}: loss={loss}")
        elif event_type == "BATCH_EPOCH_COMPLETED":
            # Handle typed payload or dict fallback
            if isinstance(event.data, BatchEpochCompletedPayload):
                batch_idx = event.data.batch_idx
                episodes = event.data.episodes_completed
                total = event.data.total_episodes
                avg_acc = event.data.avg_accuracy
                rolling_acc = event.data.rolling_accuracy
                avg_reward = event.data.avg_reward
                env_accs = event.data.env_accuracies or []
            else:
                if event.data is None:
                    _logger.warning("BATCH_EPOCH_COMPLETED event has no data payload")
                    return
                data = event.data
                batch_idx = data.get("batch_idx", "?")
                episodes = data.get("episodes_completed", "?")
                total = data.get("total_episodes", "?")
                avg_acc = data.get("avg_accuracy", 0.0)
                rolling_acc = data.get("rolling_accuracy", 0.0)
                avg_reward = data.get("avg_reward", 0.0)
                env_accs = data.get("env_accuracies", [])
            env_acc_str = ", ".join(f"{a:.1f}%" for a in env_accs) if env_accs else ""
            print(f"[{timestamp}] BATCH {batch_idx} | Episodes {episodes}/{total}")
            if env_acc_str:
                print(f"[{timestamp}]   Env accs: [{env_acc_str}]")
            print(f"[{timestamp}]   Avg: {avg_acc:.1f}% (rolling: {rolling_acc:.1f}%), reward: {avg_reward:.1f}")
        elif event_type == "COUNTERFACTUAL_COMPUTED":
            if event.data is None:
                _logger.warning("COUNTERFACTUAL_COMPUTED event has no data payload")
                return
            data = event.data
            env_id = data.get("env_id", "?")
            slot_id = data.get("slot_id", "?")
            available = data.get("available", True)
            if available is False:
                reason = data.get("reason", "unknown")
                print(f"[{timestamp}] env{env_id} | Counterfactual {slot_id}: unavailable ({reason})")
            else:
                real_acc = data.get("real_accuracy", 0.0)
                baseline_acc = data.get("baseline_accuracy", 0.0)
                contribution = data.get("contribution", 0.0)
                print(f"[{timestamp}] env{env_id} | Counterfactual {slot_id}: {real_acc:.1f}% real, {baseline_acc:.1f}% baseline, Δ={contribution:+.1f}%")
        elif event_type == "CHECKPOINT_SAVED":
            # TODO: [DEAD CODE] - This formatting code for CHECKPOINT_SAVED is unreachable
            # because these events are never emitted. See: leyline/telemetry.py dead event TODOs.
            if event.data is None:
                _logger.warning("CHECKPOINT_SAVED event has no data payload")
                return
            data = event.data
            path = data.get("path", "?")
            avg_acc = data.get("avg_accuracy", 0.0)
            print(f"[{timestamp}] CHECKPOINT | Saved to {path} (acc={avg_acc:.1f}%)")
        elif event_type == "CHECKPOINT_LOADED":
            if event.data is None:
                _logger.warning("CHECKPOINT_LOADED event has no data payload")
                return
            data = event.data
            path = data.get("path", "?")
            episode = data.get("start_episode", 0)
            source = data.get("source", "")
            if source:
                print(f"[{timestamp}] CHECKPOINT | Loaded {source} (acc={data.get('avg_accuracy', 0.0):.1f}%)")
            else:
                print(f"[{timestamp}] CHECKPOINT | Loaded from {path} (resuming at episode {episode})")
        elif event_type == "TAMIYO_INITIATED":
            if event.data is None:
                _logger.warning("TAMIYO_INITIATED event has no data payload")
                return
            data = event.data
            env_id = data.get("env_id")
            epoch = data.get("epoch", "?")
            stable_count = data.get("stable_count", 0)
            stabilization_epochs = data.get("stabilization_epochs", 0)
            env_str = f"env{env_id}" if env_id is not None else "Tamiyo"
            if stabilization_epochs == 0:
                print(f"[{timestamp}] {env_str} | Host stabilized at epoch {epoch} - germination now allowed")
            else:
                print(f"[{timestamp}] {env_str} | Host stabilized at epoch {epoch} ({stable_count}/{stabilization_epochs} stable) - germination now allowed")
        elif event_type == "PPO_UPDATE_COMPLETED":
            if event.data is None:
                _logger.warning("PPO_UPDATE_COMPLETED event has no data payload")
                return
            data = event.data
            if data.get("skipped"):
                reason = data.get("reason", "unknown")
                print(f"[{timestamp}] PPO | Update skipped ({reason})")
            else:
                policy_loss = data.get("policy_loss", 0.0)
                value_loss = data.get("value_loss", 0.0)
                entropy = data.get("entropy", 0.0)
                entropy_coef = data.get("entropy_coef", 0.0)
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

    def _event_to_dict(self, event: TelemetryEvent) -> dict:
        """Convert event to dictionary for serialization."""
        if is_dataclass(event):
            data = asdict(event)
        else:
            data = event.__dict__

        # Convert datetime objects to ISO format strings
        # hasattr AUTHORIZED by operator on 2025-11-29 15:05:24 UTC
        # Justification: Serialization - safely handle datetime objects from various sources
        if 'timestamp' in data and hasattr(data['timestamp'], 'isoformat'):
            data['timestamp'] = data['timestamp'].isoformat()

        # Convert enum to string
        # hasattr AUTHORIZED by operator on 2025-11-29 15:05:24 UTC
        # Justification: Serialization - safely handle enum objects with .name attribute
        if 'event_type' in data and hasattr(data['event_type'], 'name'):
            data['event_type'] = data['event_type'].name

        return data


class FileOutput(OutputBackend):
    """Output telemetry events to a file in JSONL format.

    Args:
        path: Path to output file. Events are appended in JSONL format.
        buffer_size: Number of events to buffer before flushing to disk.
    """

    def __init__(self, path: str | Path, buffer_size: int = 10):
        self.path = Path(path)
        self.buffer_size = buffer_size
        self._buffer: list[dict] = []

        # Ensure parent directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Open file in append mode
        self._file = open(self.path, 'a', encoding='utf-8')

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

    def _event_to_dict(self, event: TelemetryEvent) -> dict:
        """Convert event to dictionary for serialization."""
        if is_dataclass(event):
            data = asdict(event)
        else:
            data = event.__dict__

        # Convert datetime objects to ISO format strings
        # hasattr AUTHORIZED by operator on 2025-11-29 15:05:24 UTC
        # Justification: Serialization - safely handle datetime objects from various sources
        if 'timestamp' in data and hasattr(data['timestamp'], 'isoformat'):
            data['timestamp'] = data['timestamp'].isoformat()

        # Convert enum to string
        # hasattr AUTHORIZED by operator on 2025-11-29 15:05:24 UTC
        # Justification: Serialization - safely handle enum objects with .name attribute
        if 'event_type' in data and hasattr(data['event_type'], 'name'):
            data['event_type'] = data['event_type'].name

        return data

    def __del__(self):
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
        Emission is asynchronous. Events are placed in a queue and processed
        by a background worker thread. This prevents slow I/O backends
        (e.g. console, network) from blocking the training loop.

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

    def __init__(self, max_queue_size: int = 1000):
        self._backends: list[OutputBackend] = []
        self._closed = False  # Idempotency flag for close()
        self._emit_after_close_warned = False
        self._queue: queue.Queue[TelemetryEvent | None] = queue.Queue(maxsize=max_queue_size)
        self._worker_thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def _start_worker(self) -> None:
        """Start the background worker thread if not already running."""
        with self._lock:
            if self._worker_thread is None or not self._worker_thread.is_alive():
                self._worker_thread = threading.Thread(
                    target=self._worker_loop, name="NissaHubWorker", daemon=True
                )
                self._worker_thread.start()

    def _worker_loop(self) -> None:
        """Background worker loop that processes events from the queue."""
        while True:
            try:
                event = self._queue.get(timeout=1.0)
                if event is None:  # Shutdown signal
                    self._queue.task_done()
                    break

                for backend in self._backends:
                    try:
                        backend.emit(event)
                    except Exception as e:
                        # Log error but don't let one backend failure break others
                        _logger.error(f"Error in backend {backend.__class__.__name__}: {e}")
                
                self._queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                _logger.error(f"Unexpected error in NissaHub worker loop: {e}")

    def add_backend(self, backend: OutputBackend) -> None:
        """Add an output backend to the hub.

        The backend is started immediately on add. If start() fails,
        the backend is NOT added and the exception is re-raised
        (fail-fast to prevent half-initialized state and silent misconfiguration).

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
            self._backends.append(backend)
            self._start_worker()
        except Exception:
            _logger.exception(f"Failed to start backend {backend.__class__.__name__}, not adding")
            raise

    def remove_backend(self, backend: OutputBackend) -> None:
        """Remove an output backend from the hub.

        Args:
            backend: The output backend to remove.
        """
        if backend in self._backends:
            try:
                backend.close()
            except Exception as e:
                _logger.error(f"Error closing backend {backend.__class__.__name__}: {e}")
            self._backends.remove(backend)

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
            _logger.warning("NissaHub queue full, dropping telemetry event")

    def flush(self, timeout: float | None = None) -> None:
        """Block until all queued events have been processed.

        Useful for tests or shutdown sequences where you need to ensure
        all events have been delivered to backends before proceeding.

        Args:
            timeout: Maximum time to wait in seconds. None means wait forever.
        """
        if self._closed:
            return
        # Only wait if there's actually a worker running to process events
        if self._worker_thread is None or not self._worker_thread.is_alive():
            return
        # join() blocks until all items in the queue have been processed
        # (i.e., task_done() has been called for each get())
        self._queue.join()

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
        self._closed = False  # Allow hub to be reused after reset
        self._emit_after_close_warned = False
        # Create fresh queue to ensure no stale events from previous runs
        self._queue = queue.Queue(maxsize=self._queue.maxsize)
        self._worker_thread = None  # Will be recreated on next add_backend()

    def close(self) -> None:
        """Close all backends (idempotent).

        Safe to call multiple times - only closes backends once.
        Flushes pending events before shutting down the worker.
        """
        if self._closed:
            return
        
        # Signal worker to stop and wait for it to finish
        if self._worker_thread is not None and self._worker_thread.is_alive():
            try:
                # Add None to queue as sentinel for worker shutdown.
                # Use blocking put with timeout to ensure signal is received.
                self._queue.put(None, timeout=2.0)
                self._worker_thread.join(timeout=5.0)
            except (queue.Full, RuntimeError):
                pass  # Worker might already be dead or queue unreachable

        self._closed = True
        for backend in self._backends:
            try:
                backend.close()
            except Exception as e:
                _logger.error(f"Error closing backend {backend.__class__.__name__}: {e}")

    def __del__(self):
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
