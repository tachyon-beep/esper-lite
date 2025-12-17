"""Tests for Overwatch replay infrastructure."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


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
