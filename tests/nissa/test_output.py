"""Tests for Nissa output backends."""

import json
from datetime import datetime
from pathlib import Path

import pytest

from esper.leyline import TelemetryEvent, TelemetryEventType
from esper.nissa.output import DirectoryOutput


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
            data={"val_accuracy": 85.5},
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
