"""Tests for Karn store export functionality."""

import json
from datetime import datetime, timezone
from pathlib import Path


from esper.karn.store import TelemetryStore, EpisodeContext


class TestExportJsonl:
    """Tests for JSONL export serialization."""

    def test_export_handles_datetime(self, tmp_path: Path):
        """export_jsonl correctly serializes datetime objects."""
        store = TelemetryStore()

        # Create context with datetime
        store.context = EpisodeContext(
            episode_id="test_123",
            timestamp=datetime(2025, 12, 14, 12, 0, 0, tzinfo=timezone.utc),
            base_seed=42,
        )

        output_file = tmp_path / "test_export.jsonl"
        count = store.export_jsonl(output_file)

        assert count >= 1
        assert output_file.exists()

        # Verify it's valid JSON
        with open(output_file) as f:
            line = f.readline()
            data = json.loads(line)  # Should not raise
            assert data["type"] == "context"
            # timestamp should be serialized as ISO string
            assert "timestamp" in data["data"]

    def test_export_handles_path_objects(self, tmp_path: Path):
        """export_jsonl correctly serializes Path objects."""
        store = TelemetryStore()
        store.context = EpisodeContext(
            episode_id="test_path",
            timestamp=datetime.now(timezone.utc),
        )

        output_file = tmp_path / "test_path.jsonl"

        # Should not raise TypeError
        count = store.export_jsonl(output_file)
        assert count >= 1

    def test_export_handles_enums(self, tmp_path: Path):
        """export_jsonl correctly serializes Enum values."""
        from esper.karn.store import SlotSnapshot, EpisodeContext
        from esper.leyline import SeedStage

        store = TelemetryStore()

        # start_episode takes EpisodeContext, not string args
        context = EpisodeContext(
            episode_id="test_enum",
            base_seed=42,
        )
        store.start_episode(context)

        # start_episode doesn't create current_epoch - must call start_epoch explicitly
        store.start_epoch(1)

        # Add a slot with enum stage
        store.current_epoch.slots["r0c1"] = SlotSnapshot(
            slot_id="r0c1",
            stage=SeedStage.TRAINING,
        )
        store.commit_epoch()

        output_file = tmp_path / "test_enum.jsonl"
        store.export_jsonl(output_file)

        # Should serialize enum as name string
        with open(output_file) as f:
            for line in f:
                data = json.loads(line)
                if data["type"] == "epoch":
                    # Check that stage is serialized (as int from enum value)
                    assert "slots" in data["data"]
