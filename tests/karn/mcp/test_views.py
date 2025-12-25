"""Tests for DuckDB view definitions."""
import tempfile
import json
from pathlib import Path

import duckdb

from esper.karn.mcp.views import create_views, VIEW_DEFINITIONS


def test_view_definitions_exist():
    """All expected views are defined."""
    expected = {"raw_events", "runs", "epochs", "ppo_updates", "seed_lifecycle", "rewards", "anomalies"}
    assert set(VIEW_DEFINITIONS.keys()) == expected


def test_create_views_on_empty_dir():
    """Views can be created even with no telemetry files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        conn = duckdb.connect(":memory:")
        create_views(conn, tmpdir)
        # Should not raise - views exist but return empty
        result = conn.execute("SELECT * FROM runs LIMIT 1").fetchall()
        assert result == []


def test_epochs_view_parses_jsonl():
    """Epochs view correctly extracts fields from JSONL."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create nested structure matching real telemetry layout
        run_dir = Path(tmpdir) / "telemetry_2025-01-01_000000"
        run_dir.mkdir()
        events_file = run_dir / "events.jsonl"

        event = {
            "event_id": "test-event-id",
            "event_type": "EPOCH_COMPLETED",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "seed_id": None,
            "slot_id": None,
            "epoch": 5,
            "message": "",
            "data": {
                "env_id": 0,
                "inner_epoch": 10,
                "val_accuracy": 75.5,
                "val_loss": 0.25,
                "train_accuracy": 80.0,
                "train_loss": 0.20
            },
            "severity": "info"
        }
        events_file.write_text(json.dumps(event) + "\n")

        conn = duckdb.connect(":memory:")
        create_views(conn, tmpdir)

        result = conn.execute("SELECT env_id, val_accuracy FROM epochs").fetchone()
        assert result == (0, 75.5)
