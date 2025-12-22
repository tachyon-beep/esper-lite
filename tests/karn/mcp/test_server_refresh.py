"""Tests for Karn MCP server view refresh behavior."""
import json
import tempfile
from pathlib import Path

from esper.karn.mcp.server import KarnMCPServer


def test_server_refreshes_stubbed_views_when_files_appear() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        server = KarnMCPServer(telemetry_dir=tmpdir)

        before = server.query_sql_sync("SELECT COUNT(*) AS n FROM raw_events")
        assert "| 0 |" in before

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
            "data": {"env_id": 0},
            "severity": "info",
        }
        events_file.write_text(json.dumps(event) + "\n")

        after = server.query_sql_sync("SELECT COUNT(*) AS n FROM raw_events")
        assert "| 1 |" in after

