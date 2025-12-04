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
import sys
from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from esper.leyline import TelemetryEvent


class OutputBackend(ABC):
    """Base class for telemetry output backends."""

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
    """

    def __init__(self, verbose: bool = False, use_color: bool = True):
        self.verbose = verbose
        self.use_color = use_color

    def emit(self, event: TelemetryEvent) -> None:
        """Emit event to console."""
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

    def add_backend(self, backend: OutputBackend) -> None:
        """Add an output backend to the hub.

        Args:
            backend: The output backend to add.
        """
        self._backends.append(backend)

    def remove_backend(self, backend: OutputBackend) -> None:
        """Remove an output backend from the hub.

        Args:
            backend: The output backend to remove.
        """
        if backend in self._backends:
            self._backends.remove(backend)

    def emit(self, event: TelemetryEvent) -> None:
        """Emit a telemetry event to all backends.

        Args:
            event: The telemetry event to emit.
        """
        for backend in self._backends:
            try:
                backend.emit(event)
            except Exception as e:
                # Log error but don't let one backend failure break others
                print(f"Error in backend {backend.__class__.__name__}: {e}", file=sys.stderr)

    def close(self) -> None:
        """Close all backends."""
        for backend in self._backends:
            try:
                backend.close()
            except Exception as e:
                print(f"Error closing backend {backend.__class__.__name__}: {e}", file=sys.stderr)

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
    "emit",
]
