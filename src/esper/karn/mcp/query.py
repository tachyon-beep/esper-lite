"""Query execution and result formatting for MCP server."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal
import re
from typing import Any, TYPE_CHECKING
from uuid import UUID

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


def _jsonify_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, timedelta):
        return value.total_seconds()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, bytes):
        return value.hex()
    return str(value)


def rows_to_records(columns: list[str], rows: list[tuple[Any, ...]]) -> list[dict[str, Any]]:
    return [
        {column: _jsonify_value(value) for column, value in zip(columns, row)}
        for row in rows
    ]


def format_as_json(columns: list[str], rows: list[tuple[Any, ...]]) -> dict[str, Any]:
    """Format query result as JSON-friendly dict."""
    return {"columns": columns, "rows": rows_to_records(columns, rows), "row_count": len(rows)}


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
