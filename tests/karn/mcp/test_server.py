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
                "event_id": "start-1",
                "event_type": "TRAINING_STARTED",
                "timestamp": "2025-01-01T00:00:00+00:00",
                "seed_id": None,
                "slot_id": None,
                "epoch": None,
                "group_id": "default",
                "message": "",
                "data": {"episode_id": "test_run", "task": "cifar_baseline", "n_envs": 4},
                "severity": "info"
            },
            {
                "event_id": "epoch-1",
                "event_type": "EPOCH_COMPLETED",
                "timestamp": "2025-01-01T00:01:00+00:00",
                "seed_id": None,
                "slot_id": None,
                "epoch": 1,
                "group_id": "default",
                "message": "",
                "data": {"env_id": 0, "val_accuracy": 50.0, "val_loss": 1.0},
                "severity": "info"
            },
        ]
        events_file.write_text("\n".join(json.dumps(e) for e in events) + "\n")

        server = KarnMCPServer(tmpdir)
        yield server


def test_query_sql_returns_markdown(server_with_data):
    """query_sql returns structured result."""
    result = server_with_data.query_sql_sync("SELECT episode_id FROM runs")
    assert result["ok"] is True
    assert result["row_count"] == 1
    assert result["rows"][0]["episode_id"] == "test_run"


def test_query_sql_handles_error(server_with_data):
    """query_sql returns error result for invalid SQL."""
    result = server_with_data.query_sql_sync("SELECT * FROM nonexistent")
    assert result["ok"] is False
    assert "SQL Error:" in result["error"]


def test_list_views_returns_documentation(server_with_data):
    """list_views returns view documentation."""
    result = server_with_data.list_views_sync()
    assert result["ok"] is True
    view_names = {view["name"] for view in result["views"]}
    assert "runs" in view_names
    assert "epochs" in view_names
    assert "ppo_updates" in view_names


def test_run_overview_returns_report(server_with_data):
    report = server_with_data.run_overview_sync()
    assert report["ok"] is True
    assert report["run"]["episode_id"] == "test_run"


def test_describe_view_returns_columns(server_with_data):
    result = server_with_data.describe_view_sync("runs")
    assert result["ok"] is True
    column_names = {col["name"] for col in result["columns"]}
    assert "episode_id" in column_names


def test_list_runs_returns_metadata(server_with_data):
    result = server_with_data.list_runs_sync()
    assert result["ok"] is True
    assert result["row_count"] == 1
    assert result["runs"][0]["episode_id"] == "test_run"
