"""Query execution and result formatting for MCP server."""
from __future__ import annotations

import re
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import duckdb


def add_limit_if_missing(query: str, limit: int) -> str:
    """Add LIMIT clause if not already present."""
    if re.search(r"\bLIMIT\s+\d+", query, re.IGNORECASE):
        return query
    return f"{query.rstrip().rstrip(';')} LIMIT {limit}"


def format_as_markdown(columns: list[str], rows: list[tuple[Any, ...]]) -> str:
    """Format query result as Markdown table."""
    if not rows:
        return "Query returned 0 rows."

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = "\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in rows)

    return f"{header}\n{separator}\n{body}"


def execute_query(
    conn: "duckdb.DuckDBPyConnection",
    query: str,
    limit: int = 100,
) -> tuple[list[str], list[tuple[Any, ...]]]:
    """Execute SQL query and return columns and rows.

    Args:
        conn: DuckDB connection with views initialized
        query: SQL query string
        limit: Maximum rows to return (applied if no LIMIT in query)

    Returns:
        Tuple of (column_names, row_tuples)
    """
    query = add_limit_if_missing(query, limit)
    result = conn.execute(query)
    columns = [desc[0] for desc in result.description]
    rows = result.fetchall()
    return columns, rows
