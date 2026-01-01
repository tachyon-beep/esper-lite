# Telemetry Directory CLI Flag Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `--telemetry-dir` CLI flag that saves all telemetry to a timestamped folder.

**Architecture:** Create a `DirectoryOutput` backend class that wraps `FileOutput`, automatically generating a timestamped subdirectory (e.g., `telemetry_2025-12-05_101523/events.jsonl`). Add the CLI flag and wire it through the training script.

**Tech Stack:** Python pathlib, datetime, existing OutputBackend ABC

---

## Task 1: Create DirectoryOutput Backend

**Files:**
- Modify: `src/esper/nissa/output.py:147-218`
- Test: `tests/nissa/test_output.py` (create)

### Step 1: Write the failing test

Create `tests/nissa/test_output.py`:

```python
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
```

### Step 2: Run test to verify it fails

Run: `uv run pytest tests/nissa/test_output.py -v`

Expected: FAIL with `ImportError: cannot import name 'DirectoryOutput'`

### Step 3: Implement DirectoryOutput class

Add to `src/esper/nissa/output.py` after the `FileOutput` class (around line 218):

```python
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
```

### Step 4: Update __all__ exports

In `src/esper/nissa/output.py`, update the `__all__` list (around line 308):

```python
__all__ = [
    "OutputBackend",
    "ConsoleOutput",
    "FileOutput",
    "DirectoryOutput",
    "NissaHub",
    "get_hub",
    "emit",
]
```

### Step 5: Run test to verify it passes

Run: `uv run pytest tests/nissa/test_output.py -v`

Expected: PASS (4 tests)

### Step 6: Commit

```bash
git add src/esper/nissa/output.py tests/nissa/test_output.py
git commit -m "feat(nissa): add DirectoryOutput backend for timestamped telemetry"
```

---

## Task 2: Add CLI Flag to Training Script

**Files:**
- Modify: `src/esper/scripts/train.py:24-26`
- Modify: `src/esper/scripts/train.py:79-84`

### Step 1: Add --telemetry-dir argument

In `src/esper/scripts/train.py`, after line 25 (the `--telemetry-file` argument), add:

```python
    parser.add_argument("--telemetry-dir", type=str, default=None,
                        help="Save Nissa telemetry to timestamped folder in this directory")
```

### Step 2: Update imports

At line 17, update the import:

```python
from esper.nissa import get_hub, ConsoleOutput, FileOutput, DirectoryOutput
```

### Step 3: Wire up the DirectoryOutput backend

After the `file_backend` block (around line 84), add:

```python
    # Add directory output if requested
    dir_backend = None
    if args.telemetry_dir:
        dir_backend = DirectoryOutput(args.telemetry_dir)
        hub.add_backend(dir_backend)
        print(f"Telemetry will be saved to: {dir_backend.output_dir}")
```

### Step 4: Run training script with --help to verify

Run: `PYTHONPATH=src python -m esper.scripts.train ppo --help`

Expected: Shows `--telemetry-dir` in help output

### Step 5: Commit

```bash
git add src/esper/scripts/train.py
git commit -m "feat(cli): add --telemetry-dir flag for timestamped telemetry output"
```

---

## Task 3: Update Nissa __init__.py Export

**Files:**
- Modify: `src/esper/nissa/__init__.py`

### Step 1: Check current exports

Read the file to see current state.

### Step 2: Add DirectoryOutput to exports

Update the import line to include `DirectoryOutput`:

```python
from esper.nissa.output import (
    ConsoleOutput,
    DirectoryOutput,
    FileOutput,
    NissaHub,
    OutputBackend,
    emit,
    get_hub,
)
```

And update `__all__` if present.

### Step 3: Verify import works

Run: `PYTHONPATH=src python -c "from esper.nissa import DirectoryOutput; print('OK')"`

Expected: Prints `OK`

### Step 4: Commit

```bash
git add src/esper/nissa/__init__.py
git commit -m "feat(nissa): export DirectoryOutput from package"
```

---

## Task 4: Integration Test

**Files:**
- Test: `tests/nissa/test_output.py` (extend)

### Step 1: Add integration test

Add to `tests/nissa/test_output.py`:

```python
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
            data={"blueprint_id": "test_bp"},
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
```

### Step 2: Run all tests

Run: `uv run pytest tests/nissa/test_output.py -v`

Expected: PASS (5 tests)

### Step 3: Commit

```bash
git add tests/nissa/test_output.py
git commit -m "test(nissa): add integration test for DirectoryOutput with hub"
```

---

## Summary

After completing all tasks:

1. `DirectoryOutput` class in `src/esper/nissa/output.py` creates timestamped folders
2. `--telemetry-dir` CLI flag in `src/esper/scripts/train.py` enables the feature
3. Running `train.py ppo --telemetry-dir ./runs` creates `./runs/telemetry_2025-12-05_101523/events.jsonl`

**Usage:**
```bash
PYTHONPATH=src python -m esper.scripts.train ppo \
    --vectorized \
    --n-envs 4 \
    --episodes 100 \
    --telemetry-dir ./telemetry
```

Creates: `./telemetry/telemetry_2025-12-05_101523/events.jsonl`
