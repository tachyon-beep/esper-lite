"""Karn MCP Server - SQL interface to telemetry data.

This is a standalone entry point for the MCP server.
Run with: uv run python -m esper.karn.mcp.server
"""
from __future__ import annotations

import asyncio
import threading
from typing import Any

import duckdb

from esper.karn.mcp.views import create_views, telemetry_has_event_files
from esper.karn.mcp.query import execute_query, format_as_markdown

# View documentation for list_views tool
VIEW_DOCS = """Available telemetry views:

- `runs` - Run metadata (task, hyperparameters, devices)
- `epochs` - Per-environment training progress (accuracy, loss per epoch)
- `ppo_updates` - PPO health metrics (entropy, KL divergence, clip fraction)
- `seed_lifecycle` - Seed state machine (germinate, stage changes, fossilize, cull)
- `rewards` - Per-step reward breakdown (all reward components)
- `anomalies` - Training pathologies (collapses, rollbacks, plateaus)
- `raw_events` - All events (use for custom queries)

Example queries:
  SELECT * FROM runs;
  SELECT env_id, AVG(val_accuracy) FROM epochs GROUP BY env_id;
  SELECT blueprint_id, COUNT(*) FROM seed_lifecycle WHERE event_type = 'SEED_FOSSILIZED' GROUP BY blueprint_id;
"""


class KarnMCPServer:
    """MCP server exposing telemetry SQL queries."""

    def __init__(self, telemetry_dir: str = "telemetry") -> None:
        """Initialize server with DuckDB connection."""
        self._conn = duckdb.connect(":memory:")
        self._conn_lock = threading.Lock()
        self._telemetry_dir = telemetry_dir
        with self._conn_lock:
            create_views(self._conn, telemetry_dir)
            self._raw_events_stubbed = self._is_raw_events_stubbed()

    def _is_raw_events_stubbed(self) -> bool:
        try:
            row = self._conn.execute(
                "SELECT sql FROM duckdb_views() WHERE view_name = 'raw_events'"
            ).fetchone()
        except duckdb.Error:
            return False

        if row is None:
            return False
        view_sql = row[0]
        if view_sql is None:
            return False
        return "read_json_auto" not in view_sql.lower()

    def _refresh_views_if_needed(self) -> None:
        if not self._raw_events_stubbed:
            return
        if not telemetry_has_event_files(self._telemetry_dir):
            return

        create_views(self._conn, self._telemetry_dir)
        self._raw_events_stubbed = self._is_raw_events_stubbed()

    def _execute_query_with_refresh(
        self, query: str, limit: int
    ) -> tuple[list[str], list[tuple[Any, ...]]]:
        with self._conn_lock:
            self._refresh_views_if_needed()
            return execute_query(self._conn, query, limit)

    def query_sql_sync(self, query: str, limit: int = 100) -> str:
        """Execute SQL query and return markdown result (sync version for testing)."""
        try:
            columns, rows = self._execute_query_with_refresh(query, limit)
            return format_as_markdown(columns, rows)
        except duckdb.Error as e:
            return f"SQL Error: {e}"

    def list_views_sync(self) -> str:
        """Return view documentation (sync version for testing)."""
        return VIEW_DOCS

    async def query_sql(self, query: str, limit: int = 100) -> str:
        """Execute SQL query and return markdown result."""
        try:
            columns, rows = await asyncio.wait_for(
                asyncio.to_thread(self._execute_query_with_refresh, query, limit),
                timeout=30.0,
            )
            return format_as_markdown(columns, rows)
        except asyncio.TimeoutError:
            return "Query exceeded 30s timeout. Try adding filters or reducing scope."
        except duckdb.Error as e:
            return f"SQL Error: {e}"

    async def list_views(self) -> str:
        """Return view documentation."""
        return VIEW_DOCS


# MCP Protocol Wiring
try:
    from mcp.server import FastMCP

    _mcp_app = FastMCP("esper-karn")
    _server_instance: KarnMCPServer | None = None

    @_mcp_app.tool()
    async def query_sql(query: str, limit: int = 100) -> str:
        """Execute SQL against telemetry data. Returns Markdown table.

        Args:
            query: SQL query (views: runs, epochs, ppo_updates, seed_lifecycle, rewards, anomalies)
            limit: Maximum rows to return (default 100)
        """
        assert _server_instance is not None
        return await _server_instance.query_sql(query, limit)

    @_mcp_app.tool()
    async def list_views() -> str:
        """List available telemetry views and example queries."""
        assert _server_instance is not None
        return await _server_instance.list_views()

    def run_mcp_server(telemetry_dir: str = "telemetry") -> None:
        """Run the MCP server (stdio transport)."""
        global _server_instance
        _server_instance = KarnMCPServer(telemetry_dir)
        _mcp_app.run()

except ImportError:
    # MCP not installed - server class still usable for testing
    def run_mcp_server(telemetry_dir: str = "telemetry") -> None:
        raise ImportError("MCP package not installed. Run: uv add mcp")
