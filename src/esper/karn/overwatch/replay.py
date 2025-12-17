"""Overwatch Replay Infrastructure.

Provides snapshot persistence for replay functionality:
- SnapshotWriter: Writes TuiSnapshot to JSONL files
- SnapshotReader: Reads TuiSnapshot from JSONL files with filtering
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterator

if TYPE_CHECKING:
    from esper.karn.overwatch.schema import TuiSnapshot


class SnapshotWriter:
    """Writes TuiSnapshot objects to JSONL file.

    Each snapshot is written as a single JSON line, enabling:
    - Streaming writes during training
    - Crash-safe persistence (flush after each write)
    - Easy append for long-running sessions

    Usage:
        with SnapshotWriter(path) as writer:
            writer.write(snapshot)
    """

    def __init__(self, path: Path | str) -> None:
        """Initialize writer.

        Args:
            path: Path to JSONL file (will be created/overwritten)
        """
        self._path = Path(path)
        self._file = open(self._path, "w", encoding="utf-8")

    def write(self, snapshot: TuiSnapshot) -> None:
        """Write a snapshot as a single JSON line.

        Args:
            snapshot: TuiSnapshot to serialize
        """
        json_str = json.dumps(snapshot.to_dict(), separators=(",", ":"))
        self._file.write(json_str + "\n")
        self._file.flush()  # Crash safety

    def close(self) -> None:
        """Close the file."""
        self._file.close()

    def __enter__(self) -> SnapshotWriter:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
