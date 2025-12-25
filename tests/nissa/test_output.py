"""Tests for Nissa output backends."""

import json
import logging
from datetime import datetime
from pathlib import Path

import pytest

from esper.leyline import TelemetryEvent, TelemetryEventType
from esper.leyline.telemetry import (
    EpochCompletedPayload,
    SeedGerminatedPayload,
)
from esper.nissa.output import DirectoryOutput, NissaHub, OutputBackend


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
