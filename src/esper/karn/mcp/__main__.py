"""Entry point for Karn MCP server.

Run with: uv run python -m esper.karn.mcp
"""
import sys

from esper.karn.mcp.server import run_mcp_server


def main() -> None:
    """Run the MCP server."""
    telemetry_dir = sys.argv[1] if len(sys.argv) > 1 else "telemetry"
    run_mcp_server(telemetry_dir)


if __name__ == "__main__":
    main()
