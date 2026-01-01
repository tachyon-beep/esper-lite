"""Integration tests using real telemetry data."""
from pathlib import Path

import pytest

from esper.karn.mcp.server import KarnMCPServer
from esper.karn.mcp.views import telemetry_has_event_files


# Skip if no telemetry directory exists or contains no event files.
TELEMETRY_DIR = Path("telemetry")
pytestmark = pytest.mark.skipif(
    not TELEMETRY_DIR.exists() or not telemetry_has_event_files(str(TELEMETRY_DIR)),
    reason="No telemetry event files available",
)


@pytest.fixture
def real_server():
    """Create server pointing to real telemetry."""
    return KarnMCPServer(str(TELEMETRY_DIR))


def test_runs_view_has_data(real_server):
    """runs view returns real training runs."""
    result = real_server.query_sql_sync("SELECT episode_id, task FROM runs LIMIT 5")
    assert result["ok"] is True
    assert "episode_id" in result["columns"]


def test_epochs_aggregation_works(real_server):
    """Can aggregate epochs data."""
    result = real_server.query_sql_sync(
        "SELECT env_id, MAX(val_accuracy) as peak FROM epochs GROUP BY env_id LIMIT 5"
    )
    assert result["ok"] is True
    assert "peak" in result["columns"]


def test_seed_lifecycle_query_works(real_server):
    """Can query seed lifecycle events."""
    result = real_server.query_sql_sync(
        "SELECT blueprint_id, COUNT(*) as cnt FROM seed_lifecycle "
        "WHERE event_type = 'SEED_FOSSILIZED' GROUP BY blueprint_id"
    )
    assert result["ok"] is True
    assert "cnt" in result["columns"]


def test_complex_join_works(real_server):
    """Can join across views."""
    result = real_server.query_sql_sync("""
        SELECT
            r.episode_id,
            COUNT(DISTINCT e.env_id) as envs_seen
        FROM runs r
        LEFT JOIN epochs e ON e.timestamp > r.started_at
        GROUP BY r.episode_id
        LIMIT 3
    """)
    assert result["ok"] is True
    assert "envs_seen" in result["columns"]
