"""Tests for MCP server tools."""
import tempfile
import json
from pathlib import Path

import pytest

from esper.karn.mcp.server import KarnMCPServer


@pytest.fixture
def server_with_data():
    """Create server with sample telemetry data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "telemetry_2025-01-01_000000"
        run_dir.mkdir()
        events_file = run_dir / "events.jsonl"

        events = [
            {
                "event_type": "TRAINING_STARTED",
                "timestamp": "2025-01-01T00:00:00+00:00",
                "seed_id": None,
                "slot_id": None,
                "epoch": None,
                "message": "",
                "data": {"episode_id": "test_run", "task": "cifar10", "n_envs": 4},
                "severity": "info"
            },
            {
                "event_type": "EPOCH_COMPLETED",
                "timestamp": "2025-01-01T00:01:00+00:00",
                "seed_id": None,
                "slot_id": None,
                "epoch": 1,
                "message": "",
                "data": {"env_id": 0, "val_accuracy": 50.0, "val_loss": 1.0},
                "severity": "info"
            },
        ]
        events_file.write_text("\n".join(json.dumps(e) for e in events) + "\n")

        server = KarnMCPServer(tmpdir)
        yield server


def test_query_sql_returns_markdown(server_with_data):
    """query_sql returns markdown table."""
    result = server_with_data.query_sql_sync("SELECT * FROM runs")
    assert "| run_id |" in result
    assert "| test_run |" in result


def test_query_sql_handles_error(server_with_data):
    """query_sql returns error message for invalid SQL."""
    result = server_with_data.query_sql_sync("SELECT * FROM nonexistent")
    assert "SQL Error:" in result


def test_list_views_returns_documentation(server_with_data):
    """list_views returns view documentation."""
    result = server_with_data.list_views_sync()
    assert "runs" in result
    assert "epochs" in result
    assert "ppo_updates" in result
