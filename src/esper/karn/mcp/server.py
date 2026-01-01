"""Karn MCP Server - SQL interface to telemetry data.

This is a standalone entry point for the MCP server.
Run with: uv run python -m esper.karn.mcp.server
"""
from __future__ import annotations

import asyncio
import threading
from typing import Any

import duckdb

from esper.karn.mcp.query import execute_query, format_as_json, format_as_markdown, rows_to_records
from esper.karn.mcp.reports import build_run_overview
from esper.karn.mcp.views import VIEW_DEFINITIONS, create_views, telemetry_has_event_files

VIEW_CATALOG: list[dict[str, str]] = [
    {"name": "runs", "description": "Run metadata (task, hyperparameters, devices)."},
    {"name": "epochs", "description": "Per-environment epoch metrics (accuracy/loss)."},
    {"name": "ppo_updates", "description": "PPO health metrics (entropy, KL, clip frac, grad norms)."},
    {"name": "batch_epochs", "description": "Batch/episode progress events (throughput + rolling accuracy)."},
    {"name": "batch_stats", "description": "Batch-level PPO/accuracy summary snapshots."},
    {"name": "seed_lifecycle", "description": "Seed lifecycle events (germinate, stage change, fossilize, prune)."},
    {"name": "decisions", "description": "Decision snapshots (last_action context + head telemetry)."},
    {"name": "rewards", "description": "Decision snapshots (reward components breakdown)."},
    {"name": "trends", "description": "Detected trends (plateau/degradation/improvement)."},
    {"name": "anomalies", "description": "Training pathologies (collapses, rollbacks, numerical issues)."},
    {"name": "episode_outcomes", "description": "Per-episode outcome summary events."},
    {"name": "raw_events", "description": "Raw event envelope + payload JSON (for custom queries)."},
]

VIEW_EXAMPLES: list[str] = [
    "SELECT * FROM runs ORDER BY started_at DESC LIMIT 5;",
    "SELECT run_dir, env_id, MAX(val_accuracy) AS peak FROM epochs GROUP BY run_dir, env_id;",
    "SELECT blueprint_id, COUNT(*) FROM seed_lifecycle WHERE event_type = 'SEED_FOSSILIZED' GROUP BY blueprint_id;",
    "SELECT * FROM anomalies WHERE run_dir = '<run_dir>' ORDER BY timestamp DESC LIMIT 20;",
]


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

    def query_sql_sync(self, query: str, limit: int = 100) -> dict[str, Any]:
        """Execute SQL query and return JSON-friendly result (sync version for testing)."""
        try:
            columns, rows = self._execute_query_with_refresh(query, limit)
            return {"ok": True, **format_as_json(columns, rows)}
        except duckdb.Error as e:
            return {"ok": False, "error": f"SQL Error: {e}"}

    def query_sql_markdown_sync(self, query: str, limit: int = 100) -> str:
        """Execute SQL query and return markdown table (sync version for testing)."""
        try:
            columns, rows = self._execute_query_with_refresh(query, limit)
            return format_as_markdown(columns, rows)
        except duckdb.Error as e:
            return f"SQL Error: {e}"

    def list_views_sync(self) -> dict[str, Any]:
        """Return view catalog and usage notes (sync version for testing)."""
        return {
            "ok": True,
            "views": VIEW_CATALOG,
            "notes": [
                "All views are best filtered by run_dir (multiple runs can coexist under telemetry/).",
                "Most derived views include event_id + run_dir for joins back to raw_events.",
            ],
            "examples": VIEW_EXAMPLES,
        }

    def describe_view_sync(self, view_name: str) -> dict[str, Any]:
        if view_name not in VIEW_DEFINITIONS:
            return {
                "ok": False,
                "error": f"Unknown view: {view_name}",
                "known_views": sorted(VIEW_DEFINITIONS.keys()),
            }

        with self._conn_lock:
            self._refresh_views_if_needed()
            result = self._conn.execute(f"PRAGMA table_info('{view_name}')")
            columns = [desc[0] for desc in result.description]
            rows = result.fetchall()
        return {"ok": True, "view": view_name, "columns": rows_to_records(columns, rows)}

    def list_runs_sync(self, limit: int = 50) -> dict[str, Any]:
        with self._conn_lock:
            self._refresh_views_if_needed()
            result = self._conn.execute(
                f"""
                SELECT
                    run_dir,
                    group_id,
                    episode_id,
                    started_at,
                    task,
                    reward_mode,
                    n_envs,
                    n_episodes,
                    max_epochs,
                    param_budget
                FROM runs
                ORDER BY started_at DESC
                LIMIT {limit}
                """
            )
            columns = [desc[0] for desc in result.description]
            rows = result.fetchall()
        return {"ok": True, "runs": rows_to_records(columns, rows), "row_count": len(rows)}

    def run_overview_sync(
        self, run_dir: str | None = None, group_id: str | None = None, recent_limit: int = 20
    ) -> dict[str, Any]:
        with self._conn_lock:
            self._refresh_views_if_needed()
            return build_run_overview(
                self._conn, run_dir=run_dir, group_id=group_id, recent_limit=recent_limit
            )

    async def query_sql(self, query: str, limit: int = 100) -> dict[str, Any]:
        """Execute SQL query and return JSON-friendly result."""
        try:
            columns, rows = await asyncio.wait_for(
                asyncio.to_thread(self._execute_query_with_refresh, query, limit),
                timeout=30.0,
            )
            return {"ok": True, **format_as_json(columns, rows)}
        except asyncio.TimeoutError:
            return {
                "ok": False,
                "error": "Query exceeded 30s timeout. Try adding filters or reducing scope.",
            }
        except duckdb.Error as e:
            return {"ok": False, "error": f"SQL Error: {e}"}

    async def query_sql_markdown(self, query: str, limit: int = 100) -> str:
        """Execute SQL query and return markdown table."""
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

    async def list_views(self) -> dict[str, Any]:
        """Return view catalog and usage notes."""
        return self.list_views_sync()

    async def describe_view(self, view_name: str) -> dict[str, Any]:
        return await asyncio.wait_for(
            asyncio.to_thread(self.describe_view_sync, view_name), timeout=30.0
        )

    async def list_runs(self, limit: int = 50) -> dict[str, Any]:
        return await asyncio.wait_for(
            asyncio.to_thread(self.list_runs_sync, limit), timeout=30.0
        )

    async def run_overview(
        self, run_dir: str | None = None, group_id: str | None = None, recent_limit: int = 20
    ) -> dict[str, Any]:
        return await asyncio.wait_for(
            asyncio.to_thread(self.run_overview_sync, run_dir, group_id, recent_limit),
            timeout=30.0,
        )


# MCP Protocol Wiring
try:
    from mcp.server import FastMCP

    _mcp_app = FastMCP("esper-karn")
    _server_instance: KarnMCPServer | None = None

    @_mcp_app.tool()
    async def query_sql(query: str, limit: int = 100) -> dict[str, Any]:
        """Execute SQL against telemetry data. Returns structured JSON.

        Args:
            query: SQL query (see list_views for available views)
            limit: Maximum rows to return (default 100)
        """
        assert _server_instance is not None
        return await _server_instance.query_sql(query, limit)

    @_mcp_app.tool()
    async def query_sql_markdown(query: str, limit: int = 100) -> str:
        """Execute SQL against telemetry data. Returns Markdown table."""
        assert _server_instance is not None
        return await _server_instance.query_sql_markdown(query, limit)

    @_mcp_app.tool()
    async def list_views() -> dict[str, Any]:
        """List available telemetry views and usage notes."""
        assert _server_instance is not None
        return await _server_instance.list_views()

    @_mcp_app.tool()
    async def describe_view(view_name: str) -> dict[str, Any]:
        """Describe a view's columns and types."""
        assert _server_instance is not None
        return await _server_instance.describe_view(view_name)

    @_mcp_app.tool()
    async def list_runs(limit: int = 50) -> dict[str, Any]:
        """List available runs (run_dir + basic metadata)."""
        assert _server_instance is not None
        return await _server_instance.list_runs(limit)

    @_mcp_app.tool()
    async def run_overview(
        run_dir: str | None = None, group_id: str | None = None, recent_limit: int = 20
    ) -> dict[str, Any]:
        """Return a TUI-style overview report for a run."""
        assert _server_instance is not None
        return await _server_instance.run_overview(run_dir, group_id, recent_limit)

    def run_mcp_server(telemetry_dir: str = "telemetry") -> None:
        """Run the MCP server (stdio transport)."""
        global _server_instance
        _server_instance = KarnMCPServer(telemetry_dir)
        _mcp_app.run()

except ImportError:
    # MCP not installed - server class still usable for testing
    def run_mcp_server(telemetry_dir: str = "telemetry") -> None:
        raise ImportError("MCP package not installed. Run: uv add mcp")
