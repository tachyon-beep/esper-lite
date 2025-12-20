"""Tests for Overwatch replay infrastructure."""

from __future__ import annotations

import json
from pathlib import Path


class TestSnapshotWriter:
    """Tests for SnapshotWriter class."""

    def test_writer_creates_file(self, tmp_path: Path) -> None:
        """SnapshotWriter creates JSONL file."""
        from esper.karn.overwatch.replay import SnapshotWriter
        from esper.karn.overwatch.schema import (
            TuiSnapshot,
            ConnectionStatus,
            TamiyoState,
        )

        path = tmp_path / "test.jsonl"
        writer = SnapshotWriter(path)

        snap = TuiSnapshot(
            schema_version=1,
            captured_at="2025-12-18T12:00:00Z",
            connection=ConnectionStatus(True, 1000.0, 0.5),
            tamiyo=TamiyoState(),
        )

        writer.write(snap)
        writer.close()

        assert path.exists()
        assert path.stat().st_size > 0

    def test_writer_writes_one_line_per_snapshot(self, tmp_path: Path) -> None:
        """SnapshotWriter writes JSONL format (one JSON per line)."""
        from esper.karn.overwatch.replay import SnapshotWriter
        from esper.karn.overwatch.schema import (
            TuiSnapshot,
            ConnectionStatus,
            TamiyoState,
        )

        path = tmp_path / "test.jsonl"
        writer = SnapshotWriter(path)

        for i in range(3):
            snap = TuiSnapshot(
                schema_version=1,
                captured_at=f"2025-12-18T12:00:0{i}Z",
                connection=ConnectionStatus(True, 1000.0 + i, 0.5),
                tamiyo=TamiyoState(),
            )
            writer.write(snap)

        writer.close()

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 3

        # Each line is valid JSON
        for line in lines:
            data = json.loads(line)
            assert "schema_version" in data

    def test_writer_context_manager(self, tmp_path: Path) -> None:
        """SnapshotWriter works as context manager."""
        from esper.karn.overwatch.replay import SnapshotWriter
        from esper.karn.overwatch.schema import (
            TuiSnapshot,
            ConnectionStatus,
            TamiyoState,
        )

        path = tmp_path / "test.jsonl"

        with SnapshotWriter(path) as writer:
            snap = TuiSnapshot(
                schema_version=1,
                captured_at="2025-12-18T12:00:00Z",
                connection=ConnectionStatus(True, 1000.0, 0.5),
                tamiyo=TamiyoState(),
            )
            writer.write(snap)

        # File should be closed and flushed
        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1

    def test_writer_flushes_on_each_write(self, tmp_path: Path) -> None:
        """SnapshotWriter flushes after each write for crash safety."""
        from esper.karn.overwatch.replay import SnapshotWriter
        from esper.karn.overwatch.schema import (
            TuiSnapshot,
            ConnectionStatus,
            TamiyoState,
        )

        path = tmp_path / "test.jsonl"
        writer = SnapshotWriter(path)

        snap = TuiSnapshot(
            schema_version=1,
            captured_at="2025-12-18T12:00:00Z",
            connection=ConnectionStatus(True, 1000.0, 0.5),
            tamiyo=TamiyoState(),
        )
        writer.write(snap)

        # Without closing, file should still have content (due to flush)
        assert path.stat().st_size > 0

        writer.close()


class TestSnapshotReader:
    """Tests for SnapshotReader class."""

    def test_reader_yields_snapshots(self, tmp_path: Path) -> None:
        """SnapshotReader yields TuiSnapshot objects."""
        from esper.karn.overwatch.replay import SnapshotWriter, SnapshotReader
        from esper.karn.overwatch.schema import (
            TuiSnapshot,
            ConnectionStatus,
            TamiyoState,
        )

        # Write test data
        path = tmp_path / "test.jsonl"
        with SnapshotWriter(path) as writer:
            for i in range(5):
                snap = TuiSnapshot(
                    schema_version=1,
                    captured_at=f"2025-12-18T12:00:0{i}Z",
                    connection=ConnectionStatus(True, 1000.0 + i, 0.5),
                    tamiyo=TamiyoState(),
                    episode=i,
                )
                writer.write(snap)

        # Read back
        reader = SnapshotReader(path)
        snapshots = list(reader)

        assert len(snapshots) == 5
        assert snapshots[0].episode == 0
        assert snapshots[4].episode == 4
        assert all(isinstance(s, TuiSnapshot) for s in snapshots)

    def test_reader_with_filter(self, tmp_path: Path) -> None:
        """SnapshotReader filters snapshots."""
        from esper.karn.overwatch.replay import SnapshotWriter, SnapshotReader
        from esper.karn.overwatch.schema import (
            TuiSnapshot,
            ConnectionStatus,
            TamiyoState,
        )

        # Write test data
        path = tmp_path / "test.jsonl"
        with SnapshotWriter(path) as writer:
            for i in range(10):
                snap = TuiSnapshot(
                    schema_version=1,
                    captured_at=f"2025-12-18T12:00:{i:02d}Z",
                    connection=ConnectionStatus(True, 1000.0 + i, 0.5),
                    tamiyo=TamiyoState(),
                    episode=i,
                )
                writer.write(snap)

        # Read with filter
        reader = SnapshotReader(path, filter_fn=lambda s: s.episode >= 5)
        snapshots = list(reader)

        assert len(snapshots) == 5
        assert snapshots[0].episode == 5
        assert snapshots[4].episode == 9

    def test_reader_handles_empty_file(self, tmp_path: Path) -> None:
        """SnapshotReader handles empty file gracefully."""
        from esper.karn.overwatch.replay import SnapshotReader

        path = tmp_path / "empty.jsonl"
        path.touch()

        reader = SnapshotReader(path)
        snapshots = list(reader)

        assert snapshots == []

    def test_reader_preserves_nested_data(self, tmp_path: Path) -> None:
        """SnapshotReader preserves nested slot and env data."""
        from esper.karn.overwatch.replay import SnapshotWriter, SnapshotReader
        from esper.karn.overwatch.schema import (
            TuiSnapshot,
            ConnectionStatus,
            TamiyoState,
            EnvSummary,
            SlotChipState,
        )

        path = tmp_path / "test.jsonl"
        with SnapshotWriter(path) as writer:
            snap = TuiSnapshot(
                schema_version=1,
                captured_at="2025-12-18T12:00:00Z",
                connection=ConnectionStatus(True, 1000.0, 0.5),
                tamiyo=TamiyoState(
                    action_counts={"BLEND": 10, "CULL": 5},
                    kl_divergence=0.019,
                ),
                flight_board=[
                    EnvSummary(
                        env_id=0,
                        device_id=0,
                        status="OK",
                        slots={
                            "r0c1": SlotChipState(
                                slot_id="r0c1",
                                stage="BLENDING",
                                blueprint_id="conv_light",
                                alpha=0.7,
                                gate_last="G2",
                                gate_passed=True,
                            )
                        },
                    )
                ],
            )
            writer.write(snap)

        reader = SnapshotReader(path)
        restored = list(reader)[0]

        assert restored.tamiyo.action_counts["BLEND"] == 10
        assert restored.tamiyo.kl_divergence == 0.019
        assert restored.flight_board[0].slots["r0c1"].alpha == 0.7
        assert restored.flight_board[0].slots["r0c1"].gate_passed is True

    def test_reader_is_iterable(self, tmp_path: Path) -> None:
        """SnapshotReader can be iterated multiple times."""
        from esper.karn.overwatch.replay import SnapshotWriter, SnapshotReader
        from esper.karn.overwatch.schema import (
            TuiSnapshot,
            ConnectionStatus,
            TamiyoState,
        )

        path = tmp_path / "test.jsonl"
        with SnapshotWriter(path) as writer:
            for i in range(3):
                snap = TuiSnapshot(
                    schema_version=1,
                    captured_at=f"2025-12-18T12:00:0{i}Z",
                    connection=ConnectionStatus(True, 1000.0, 0.5),
                    tamiyo=TamiyoState(),
                )
                writer.write(snap)

        reader = SnapshotReader(path)

        # First iteration
        first = list(reader)
        assert len(first) == 3

        # Second iteration
        second = list(reader)
        assert len(second) == 3


class TestReplayExports:
    """Tests that replay classes are exported correctly."""

    def test_replay_classes_importable_from_package(self) -> None:
        """Replay classes are importable from overwatch package."""
        from esper.karn.overwatch import SnapshotWriter, SnapshotReader

        assert SnapshotWriter is not None
        assert SnapshotReader is not None
