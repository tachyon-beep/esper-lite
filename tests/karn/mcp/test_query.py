"""Tests for query execution and formatting."""
import pytest

from esper.karn.mcp.query import execute_query, format_as_markdown, add_limit_if_missing


class TestAddLimitIfMissing:
    def test_adds_limit_when_absent(self):
        result = add_limit_if_missing("SELECT * FROM foo", 50)
        assert result == "SELECT * FROM foo LIMIT 50"

    def test_preserves_existing_limit(self):
        result = add_limit_if_missing("SELECT * FROM foo LIMIT 10", 50)
        assert result == "SELECT * FROM foo LIMIT 10"

    def test_case_insensitive(self):
        result = add_limit_if_missing("SELECT * FROM foo limit 10", 50)
        assert result == "SELECT * FROM foo limit 10"

    def test_strips_trailing_semicolon(self):
        result = add_limit_if_missing("SELECT * FROM foo;", 50)
        assert result == "SELECT * FROM foo LIMIT 50"


class TestFormatAsMarkdown:
    def test_empty_result(self):
        result = format_as_markdown([], [])
        assert result == "Query returned 0 rows."

    def test_single_row(self):
        columns = ["id", "name"]
        rows = [(1, "Alice")]
        result = format_as_markdown(columns, rows)
        assert "| id | name |" in result
        assert "| 1 | Alice |" in result

    def test_multiple_rows(self):
        columns = ["x"]
        rows = [(1,), (2,), (3,)]
        result = format_as_markdown(columns, rows)
        assert result.count("|") == 10  # 2 per row * 5 rows (header + sep + 3 data)
