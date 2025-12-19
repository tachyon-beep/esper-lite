"""Karn MCP Server - SQL interface to telemetry data.

This is a standalone entry point for the MCP server.
Run with: uv run python -m esper.karn.mcp.server
"""
from __future__ import annotations

import asyncio

import duckdb

from esper.karn.mcp.views import create_views
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
        self._telemetry_dir = telemetry_dir
        create_views(self._conn, telemetry_dir)

    def query_sql_sync(self, query: str, limit: int = 100) -> str:
        """Execute SQL query and return markdown result (sync version for testing)."""
        try:
            columns, rows = execute_query(self._conn, query, limit)
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
                asyncio.to_thread(execute_query, self._conn, query, limit),
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
