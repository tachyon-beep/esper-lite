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
from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path

from esper.leyline import TelemetryEvent

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
        if "EPOCH" in event_type:
            loss = event.data.get("val_loss", "?")
            acc = event.data.get("val_accuracy", "?")
            epoch = event.epoch or "?"
            print(f"[{timestamp}] {seed_id} | Epoch {epoch}: loss={loss} acc={acc}")
        elif "COMMAND" in event_type:
            action = event.data.get("action", "unknown")
            print(f"[{timestamp}] {seed_id} | Command: {action}")
        elif event_type.startswith("SEED_"):
            data = event.data or {}
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
            elif event_type == "SEED_CULLED":
                blueprint_id = data.get("blueprint_id", "?")
                improvement = data.get("improvement")
                reason = data.get("reason")
                reason_str = f" ({reason})" if reason else ""
                if isinstance(improvement, (int, float)):
                    msg = f"Culled ({blueprint_id}, \u0394acc {improvement:+.2f}%){reason_str}"
                else:
                    msg = f"Culled ({blueprint_id}){reason_str}"
            else:
                msg = event.message or event_type
            print(f"[{timestamp}] {seed_id} | {msg}")
        elif event_type == "REWARD_COMPUTED":
            data = event.data or {}
            env_id = data.get("env_id", "?")
            action = data.get("action_name", "?")
            total = data.get("total_reward", 0.0)
            # Show key components
            bounded = data.get("bounded_attribution")
            base = data.get("base_acc_delta", 0.0)
            rent = data.get("compute_rent", 0.0)
            val_acc = data.get("val_acc", 0.0)
            # Warning signals telemetry
            blending_warning = data.get("blending_warning", 0.0)
            probation_warning = data.get("probation_warning", 0.0)
            attribution_discount = data.get("attribution_discount", 1.0)
            ratio_penalty = data.get("ratio_penalty", 0.0)
            # Terminal bonus telemetry
            fossilize_terminal_bonus = data.get("fossilize_terminal_bonus", 0.0)
            num_fossilized_seeds = data.get("num_fossilized_seeds", 0)
            # Format: env, action, total reward, and key breakdown
            # Note: rent is stored as negative (penalty), display as positive cost
            rent_display = abs(rent)
            # Build extra info for warning signals
            extra = ""
            if blending_warning < 0:
                extra += f" BLEND={blending_warning:.2f}"
            if probation_warning < 0:
                extra += f" PROB={probation_warning:.2f}"
            if attribution_discount < 1.0:
                extra += f" disc={attribution_discount:.2f}"
            if ratio_penalty < 0:
                extra += f" ratio={ratio_penalty:.2f}"
            if fossilize_terminal_bonus > 0:
                extra += f" fossil_term={fossilize_terminal_bonus:+.1f}({num_fossilized_seeds})"
            if bounded is not None:
                # Contribution-primary reward
                print(f"[{timestamp}] env{env_id} | {action}: r={total:+.2f} (attr={bounded:+.2f}, rent={rent_display:.2f}{extra}) acc={val_acc:.1f}%")
            else:
                # Legacy shaped reward
                print(f"[{timestamp}] env{env_id} | {action}: r={total:+.2f} (Δacc={base:+.2f}, rent={rent_display:.2f}{extra}) acc={val_acc:.1f}%")
        elif event_type == "GOVERNOR_ROLLBACK":
            data = event.data or {}
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
            data = event.data or {}
            loss = data.get("current_loss", "?")
            panics = data.get("consecutive_panics", 0)
            if isinstance(loss, float):
                loss = f"{loss:.4f}"
            print(f"[{timestamp}] GOVERNOR | ⚠️  PANIC #{panics}: loss={loss}")
        elif event_type == "BATCH_COMPLETED":
            data = event.data or {}
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
            data = event.data or {}
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
            data = event.data or {}
            path = data.get("path", "?")
            avg_acc = data.get("avg_accuracy", 0.0)
            print(f"[{timestamp}] CHECKPOINT | Saved to {path} (acc={avg_acc:.1f}%)")
        elif event_type == "CHECKPOINT_LOADED":
            data = event.data or {}
            path = data.get("path", "?")
            episode = data.get("start_episode", 0)
            source = data.get("source", "")
            if source:
                print(f"[{timestamp}] CHECKPOINT | Loaded {source} (acc={data.get('avg_accuracy', 0.0):.1f}%)")
            else:
                print(f"[{timestamp}] CHECKPOINT | Loaded from {path} (resuming at episode {episode})")
        elif event_type == "TAMIYO_INITIATED":
            data = event.data or {}
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
            data = event.data or {}
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
                           "GRADIENT_ANOMALY"):
            data = event.data or {}
            episode = data.get("episode", "?")
            detail = data.get("detail", "")
            anomaly_name = event_type.replace("_DETECTED", "").replace("_", " ").title()
            print(f"[{timestamp}] ⚠️  ANOMALY | {anomaly_name} at episode {episode}: {detail}")
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

    Usage:
        hub = NissaHub()
        hub.add_backend(ConsoleOutput())
        hub.add_backend(FileOutput("telemetry.jsonl"))

        # Emit events
        hub.emit(event)
    """

    def __init__(self):
        self._backends: list[OutputBackend] = []
        self._closed = False  # Idempotency flag for close()
        self._emit_after_close_warned = False

    def add_backend(self, backend: OutputBackend) -> None:
        """Add an output backend to the hub.

        The backend is started immediately on add. If start() fails,
        the backend is NOT added and the exception is re-raised
        (fail-fast to prevent half-initialized state and silent misconfiguration).

        Args:
            backend: The output backend to add.

        Raises:
            Exception: If backend.start() fails, the original exception is
                logged and re-raised so callers can detect misconfiguration.
        """
        try:
            backend.start()
            self._backends.append(backend)
        except Exception as e:
            _logger.error(f"Failed to start backend {backend.__class__.__name__}, not adding: {e}")
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
        """Emit a telemetry event to all backends.

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
        for backend in self._backends:
            try:
                backend.emit(event)
            except Exception as e:
                # Log error but don't let one backend failure break others
                _logger.error(f"Error in backend {backend.__class__.__name__}: {e}")

    def reset(self) -> None:
        """Reset the hub: close all backends and clear the list.

        Use this to ensure clean state between test runs or episodes.
        """
        self.close()
        self._backends.clear()
        self._closed = False  # Allow hub to be reused after reset
        self._emit_after_close_warned = False

    def close(self) -> None:
        """Close all backends (idempotent).

        Safe to call multiple times - only closes backends once.
        """
        if self._closed:
            return
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
