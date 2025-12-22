"""Karn MCP Server - SQL interface to telemetry data.

Usage:
    # As CLI:
    uv run python -m esper.karn.mcp [telemetry_dir]

    # As library:
    from esper.karn.mcp import KarnMCPServer
    server = KarnMCPServer("telemetry")
    result = server.query_sql_sync("SELECT * FROM runs")
"""
from esper.karn.mcp.server import KarnMCPServer, run_mcp_server
from esper.karn.mcp.views import VIEW_DEFINITIONS, create_views
from esper.karn.mcp.query import execute_query, format_as_markdown

__all__ = [
    "KarnMCPServer",
    "run_mcp_server",
    "VIEW_DEFINITIONS",
    "create_views",
    "execute_query",
    "format_as_markdown",
]
