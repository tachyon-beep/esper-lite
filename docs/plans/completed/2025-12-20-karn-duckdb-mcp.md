# Karn DuckDB MCP Server Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expose telemetry logs via a SQL interface (MCP) for surgical analysis of training runs.

**Architecture:** DuckDB queries JSONL files in-place via virtual views. MCP server exposes `query_sql` and `list_views` tools. Async wrapper around blocking DuckDB calls.

**Tech Stack:** DuckDB (analytical SQL), MCP Python SDK (stdio transport), asyncio (async wrapper)

---

## Task 1: Add Dependencies

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add duckdb and mcp packages**

Run:
```bash
uv add duckdb mcp
```

**Step 2: Verify installation**

Run:
```bash
uv run python -c "import duckdb; import mcp; print('OK')"
```
Expected: `OK`

**Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add duckdb and mcp dependencies for Karn Data Link"
```

---

## Task 2: Create View Definitions Module

**Files:**
- Create: `src/esper/karn/mcp/views.py`
- Test: `tests/karn/mcp/test_views.py`

**Step 1: Write the failing test**

Create `tests/karn/mcp/__init__.py` (empty) and `tests/karn/mcp/test_views.py`:

```python
"""Tests for DuckDB view definitions."""
import tempfile
import json
from pathlib import Path

import duckdb
import pytest

from esper.karn.mcp.views import create_views, VIEW_DEFINITIONS


def test_view_definitions_exist():
    """All expected views are defined."""
    expected = {"raw_events", "runs", "epochs", "ppo_updates", "seed_lifecycle", "rewards", "anomalies"}
    assert set(VIEW_DEFINITIONS.keys()) == expected


def test_create_views_on_empty_dir():
    """Views can be created even with no telemetry files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        conn = duckdb.connect(":memory:")
        create_views(conn, tmpdir)
        # Should not raise - views exist but return empty
        result = conn.execute("SELECT * FROM runs LIMIT 1").fetchall()
        assert result == []


def test_epochs_view_parses_jsonl():
    """Epochs view correctly extracts fields from JSONL."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create nested structure matching real telemetry layout
        run_dir = Path(tmpdir) / "telemetry_2025-01-01_000000"
        run_dir.mkdir()
        events_file = run_dir / "events.jsonl"

        event = {
            "event_type": "EPOCH_COMPLETED",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "epoch": 5,
            "data": {
                "env_id": 0,
                "inner_epoch": 10,
                "val_accuracy": 75.5,
                "val_loss": 0.25,
                "train_accuracy": 80.0,
                "train_loss": 0.20
            }
        }
        events_file.write_text(json.dumps(event) + "\n")

        conn = duckdb.connect(":memory:")
        create_views(conn, tmpdir)

        result = conn.execute("SELECT env_id, val_accuracy FROM epochs").fetchone()
        assert result == (0, 75.5)
```

**Step 2: Create directory structure**

```bash
mkdir -p src/esper/karn/mcp tests/karn/mcp
touch src/esper/karn/mcp/__init__.py tests/karn/mcp/__init__.py
```

**Step 3: Run test to verify it fails**

Run:
```bash
PYTHONPATH=src uv run pytest tests/karn/mcp/test_views.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'esper.karn.mcp.views'`

**Step 4: Write minimal implementation**

Create `src/esper/karn/mcp/views.py`:

```python
"""DuckDB view definitions for telemetry data.

NOTE: Views depend on esper.leyline.telemetry.TelemetryEvent schema.
Breaking changes to TelemetryEvent.data fields may require view updates.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import duckdb

VIEW_DEFINITIONS: dict[str, str] = {
    "raw_events": """
        CREATE OR REPLACE VIEW raw_events AS
        SELECT * FROM read_json_auto(
            '{telemetry_dir}/*/events.jsonl',
            ignore_errors=true,
            maximum_object_size=16777216
        )
    """,
    "runs": """
        CREATE OR REPLACE VIEW runs AS
        SELECT
            json_extract_string(data, '$.episode_id') as run_id,
            timestamp as started_at,
            json_extract_string(data, '$.task') as task,
            json_extract_string(data, '$.topology') as topology,
            json_extract_string(data, '$.reward_mode') as reward_mode,
            json_extract(data, '$.n_envs')::INTEGER as n_envs,
            json_extract(data, '$.n_episodes')::INTEGER as n_episodes,
            json_extract(data, '$.max_epochs')::INTEGER as max_epochs,
            json_extract(data, '$.lr')::DOUBLE as learning_rate,
            json_extract(data, '$.entropy_coef')::DOUBLE as entropy_coef,
            json_extract(data, '$.clip_ratio')::DOUBLE as clip_ratio,
            json_extract(data, '$.param_budget')::INTEGER as param_budget,
            json_extract_string(data, '$.policy_device') as policy_device,
            json_extract(data, '$.host_params')::INTEGER as host_params
        FROM raw_events
        WHERE event_type = 'TRAINING_STARTED'
    """,
    "epochs": """
        CREATE OR REPLACE VIEW epochs AS
        SELECT
            timestamp,
            epoch as global_epoch,
            json_extract(data, '$.env_id')::INTEGER as env_id,
            json_extract(data, '$.inner_epoch')::INTEGER as inner_epoch,
            json_extract(data, '$.val_accuracy')::DOUBLE as val_accuracy,
            json_extract(data, '$.val_loss')::DOUBLE as val_loss,
            json_extract(data, '$.train_accuracy')::DOUBLE as train_accuracy,
            json_extract(data, '$.train_loss')::DOUBLE as train_loss
        FROM raw_events
        WHERE event_type = 'EPOCH_COMPLETED'
    """,
    "ppo_updates": """
        CREATE OR REPLACE VIEW ppo_updates AS
        SELECT
            timestamp,
            epoch as episodes_completed,
            json_extract(data, '$.inner_epoch')::INTEGER as inner_epoch,
            json_extract(data, '$.policy_loss')::DOUBLE as policy_loss,
            json_extract(data, '$.value_loss')::DOUBLE as value_loss,
            json_extract(data, '$.entropy')::DOUBLE as entropy,
            json_extract(data, '$.kl_divergence')::DOUBLE as kl_divergence,
            json_extract(data, '$.clip_fraction')::DOUBLE as clip_fraction,
            json_extract(data, '$.explained_variance')::DOUBLE as explained_variance,
            json_extract(data, '$.avg_accuracy')::DOUBLE as avg_accuracy,
            json_extract(data, '$.avg_reward')::DOUBLE as avg_reward,
            json_extract(data, '$.grad_norm')::DOUBLE as grad_norm,
            json_extract(data, '$.entropy_collapsed')::BOOLEAN as entropy_collapsed,
            json_extract(data, '$.slot_entropy')::DOUBLE as slot_entropy,
            json_extract(data, '$.blueprint_entropy')::DOUBLE as blueprint_entropy,
            json_extract(data, '$.blend_entropy')::DOUBLE as blend_entropy,
            json_extract(data, '$.op_entropy')::DOUBLE as op_entropy
        FROM raw_events
        WHERE event_type = 'PPO_UPDATE_COMPLETED'
    """,
    "seed_lifecycle": """
        CREATE OR REPLACE VIEW seed_lifecycle AS
        SELECT
            timestamp,
            event_type,
            seed_id,
            slot_id,
            json_extract(data, '$.env_id')::INTEGER as env_id,
            json_extract_string(data, '$.blueprint_id') as blueprint_id,
            json_extract(data, '$.params')::INTEGER as params,
            json_extract(data, '$.alpha')::DOUBLE as alpha,
            json_extract(data, '$.inner_epoch')::INTEGER as inner_epoch,
            json_extract(data, '$.global_epoch')::INTEGER as global_epoch,
            json_extract_string(data, '$.from') as from_stage,
            json_extract_string(data, '$.to') as to_stage,
            json_extract(data, '$.improvement')::DOUBLE as improvement,
            json_extract(data, '$.counterfactual')::DOUBLE as counterfactual,
            json_extract(data, '$.epochs_total')::INTEGER as epochs_total,
            json_extract(data, '$.gradient_health')::DOUBLE as gradient_health,
            json_extract_string(data, '$.reason') as prune_reason,
            json_extract(data, '$.auto_pruned')::BOOLEAN as auto_pruned
        FROM raw_events
        WHERE event_type IN (
            'SEED_GERMINATED',
            'SEED_STAGE_CHANGED',
            'SEED_FOSSILIZED',
            'SEED_PRUNED'
        )
    """,
    "rewards": """
        CREATE OR REPLACE VIEW rewards AS
        SELECT
            timestamp,
            epoch,
            json_extract(data, '$.env_id')::INTEGER as env_id,
            json_extract(data, '$.episode')::INTEGER as episode,
            json_extract_string(data, '$.ab_group') as ab_group,
            json_extract_string(data, '$.action_name') as action_name,
            json_extract(data, '$.action_success')::BOOLEAN as action_success,
            json_extract_string(data, '$.seed_stage') as seed_stage,
            json_extract(data, '$.total_reward')::DOUBLE as total_reward,
            json_extract(data, '$.base_acc_delta')::DOUBLE as base_acc_delta,
            json_extract(data, '$.bounded_attribution')::DOUBLE as bounded_attribution,
            json_extract(data, '$.ratio_penalty')::DOUBLE as ratio_penalty,
            json_extract(data, '$.compute_rent')::DOUBLE as compute_rent,
            json_extract(data, '$.stage_bonus')::DOUBLE as stage_bonus,
            json_extract(data, '$.action_shaping')::DOUBLE as action_shaping,
            json_extract(data, '$.terminal_bonus')::DOUBLE as terminal_bonus,
            json_extract(data, '$.val_acc')::DOUBLE as val_acc,
            json_extract(data, '$.num_fossilized_seeds')::INTEGER as num_fossilized_seeds
        FROM raw_events
        WHERE event_type = 'REWARD_COMPUTED'
    """,
    "anomalies": """
        CREATE OR REPLACE VIEW anomalies AS
        SELECT
            timestamp,
            event_type,
            message,
            data
        FROM raw_events
        WHERE event_type IN (
            'VALUE_COLLAPSE_DETECTED',
            'RATIO_EXPLOSION_DETECTED',
            'RATIO_COLLAPSE_DETECTED',
            'GRADIENT_PATHOLOGY_DETECTED',
            'NUMERICAL_INSTABILITY_DETECTED',
            'GOVERNOR_PANIC',
            'GOVERNOR_ROLLBACK',
            'PLATEAU_DETECTED'
        )
    """,
}


def create_views(conn: "duckdb.DuckDBPyConnection", telemetry_dir: str) -> None:
    """Create all telemetry views on the given connection.

    Args:
        conn: DuckDB connection (in-memory or file-based)
        telemetry_dir: Path to telemetry directory containing run subdirectories
    """
    conn.execute("PRAGMA threads=4")

    for view_name, view_sql in VIEW_DEFINITIONS.items():
        sql = view_sql.format(telemetry_dir=telemetry_dir)
        conn.execute(sql)
```

**Step 5: Run test to verify it passes**

Run:
```bash
PYTHONPATH=src uv run pytest tests/karn/mcp/test_views.py -v
```
Expected: All 3 tests PASS

**Step 6: Commit**

```bash
git add src/esper/karn/mcp/ tests/karn/mcp/
git commit -m "feat(karn): add DuckDB view definitions for telemetry"
```

---

## Task 3: Create Query Execution Module

**Files:**
- Create: `src/esper/karn/mcp/query.py`
- Test: `tests/karn/mcp/test_query.py`

**Step 1: Write the failing test**

Create `tests/karn/mcp/test_query.py`:

```python
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
        assert result.count("|") == 12  # 3 per row * 4 rows (header + sep + 3 data)
```

**Step 2: Run test to verify it fails**

Run:
```bash
PYTHONPATH=src uv run pytest tests/karn/mcp/test_query.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'esper.karn.mcp.query'`

**Step 3: Write minimal implementation**

Create `src/esper/karn/mcp/query.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run:
```bash
PYTHONPATH=src uv run pytest tests/karn/mcp/test_query.py -v
```
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add src/esper/karn/mcp/query.py tests/karn/mcp/test_query.py
git commit -m "feat(karn): add query execution and markdown formatting"
```

---

## Task 4: Create MCP Server

**Files:**
- Create: `src/esper/karn/mcp/server.py`
- Test: `tests/karn/mcp/test_server.py`

**Step 1: Write the failing test**

Create `tests/karn/mcp/test_server.py`:

```python
"""Tests for MCP server tools."""
import tempfile
import json
from pathlib import Path

import pytest

from esper.karn.mcp.server import KarnMCPServer


@pytest.fixture
def server_with_data():
    """Create server with sample telemetry data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "telemetry_2025-01-01_000000"
        run_dir.mkdir()
        events_file = run_dir / "events.jsonl"

        events = [
            {
                "event_type": "TRAINING_STARTED",
                "timestamp": "2025-01-01T00:00:00+00:00",
                "data": {"episode_id": "test_run", "task": "cifar10", "n_envs": 4}
            },
            {
                "event_type": "EPOCH_COMPLETED",
                "timestamp": "2025-01-01T00:01:00+00:00",
                "epoch": 1,
                "data": {"env_id": 0, "val_accuracy": 50.0, "val_loss": 1.0}
            },
        ]
        events_file.write_text("\n".join(json.dumps(e) for e in events) + "\n")

        server = KarnMCPServer(tmpdir)
        yield server


def test_query_sql_returns_markdown(server_with_data):
    """query_sql returns markdown table."""
    result = server_with_data.query_sql_sync("SELECT * FROM runs")
    assert "| run_id |" in result
    assert "| test_run |" in result


def test_query_sql_handles_error(server_with_data):
    """query_sql returns error message for invalid SQL."""
    result = server_with_data.query_sql_sync("SELECT * FROM nonexistent")
    assert "SQL Error:" in result


def test_list_views_returns_documentation(server_with_data):
    """list_views returns view documentation."""
    result = server_with_data.list_views_sync()
    assert "runs" in result
    assert "epochs" in result
    assert "ppo_updates" in result
```

**Step 2: Run test to verify it fails**

Run:
```bash
PYTHONPATH=src uv run pytest tests/karn/mcp/test_server.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'esper.karn.mcp.server'`

**Step 3: Write minimal implementation**

Create `src/esper/karn/mcp/server.py`:

```python
"""Karn MCP Server - SQL interface to telemetry data.

This is a standalone entry point for the MCP server.
Run with: uv run python -m esper.karn.mcp.server
"""
from __future__ import annotations

import asyncio
from typing import Any

import duckdb

from esper.karn.mcp.views import create_views, VIEW_DEFINITIONS
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
```

**Step 4: Run test to verify it passes**

Run:
```bash
PYTHONPATH=src uv run pytest tests/karn/mcp/test_server.py -v
```
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add src/esper/karn/mcp/server.py tests/karn/mcp/test_server.py
git commit -m "feat(karn): add MCP server with query_sql and list_views tools"
```

---

## Task 5: Add MCP Protocol Wiring

**Files:**
- Modify: `src/esper/karn/mcp/server.py`
- Create: `src/esper/karn/mcp/__main__.py`

**Step 1: Update server.py with MCP protocol wiring**

Add to the end of `src/esper/karn/mcp/server.py`:

```python
# MCP Protocol Wiring
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server

    _mcp_app = Server("esper-karn")
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

    async def run_mcp_server(telemetry_dir: str = "telemetry") -> None:
        """Run the MCP server (stdio transport)."""
        global _server_instance
        _server_instance = KarnMCPServer(telemetry_dir)
        async with stdio_server(_mcp_app):
            await asyncio.Event().wait()

except ImportError:
    # MCP not installed - server class still usable for testing
    async def run_mcp_server(telemetry_dir: str = "telemetry") -> None:
        raise ImportError("MCP package not installed. Run: uv add mcp")
```

**Step 2: Create __main__.py entry point**

Create `src/esper/karn/mcp/__main__.py`:

```python
"""Entry point for Karn MCP server.

Run with: uv run python -m esper.karn.mcp
"""
import asyncio
import sys

from esper.karn.mcp.server import run_mcp_server


def main() -> None:
    """Run the MCP server."""
    telemetry_dir = sys.argv[1] if len(sys.argv) > 1 else "telemetry"
    asyncio.run(run_mcp_server(telemetry_dir))


if __name__ == "__main__":
    main()
```

**Step 3: Test the entry point runs**

Run:
```bash
timeout 2 uv run python -m esper.karn.mcp --help 2>&1 || true
```
Expected: Should start (and timeout after 2s since it waits for stdio), no import errors

**Step 4: Commit**

```bash
git add src/esper/karn/mcp/server.py src/esper/karn/mcp/__main__.py
git commit -m "feat(karn): add MCP protocol wiring and CLI entry point"
```

---

## Task 6: Integration Test with Real Telemetry

**Files:**
- Test: `tests/karn/mcp/test_integration.py`

**Step 1: Write integration test**

Create `tests/karn/mcp/test_integration.py`:

```python
"""Integration tests using real telemetry data."""
from pathlib import Path

import pytest

from esper.karn.mcp.server import KarnMCPServer


# Skip if no telemetry directory exists
TELEMETRY_DIR = Path("telemetry")
pytestmark = pytest.mark.skipif(
    not TELEMETRY_DIR.exists() or not any(TELEMETRY_DIR.iterdir()),
    reason="No telemetry data available"
)


@pytest.fixture
def real_server():
    """Create server pointing to real telemetry."""
    return KarnMCPServer(str(TELEMETRY_DIR))


def test_runs_view_has_data(real_server):
    """runs view returns real training runs."""
    result = real_server.query_sql_sync("SELECT run_id, task FROM runs LIMIT 5")
    # Should have header row at minimum
    assert "|" in result


def test_epochs_aggregation_works(real_server):
    """Can aggregate epochs data."""
    result = real_server.query_sql_sync(
        "SELECT env_id, MAX(val_accuracy) as peak FROM epochs GROUP BY env_id LIMIT 5"
    )
    assert "peak" in result or "0 rows" in result


def test_seed_lifecycle_query_works(real_server):
    """Can query seed lifecycle events."""
    result = real_server.query_sql_sync(
        "SELECT blueprint_id, COUNT(*) as cnt FROM seed_lifecycle "
        "WHERE event_type = 'SEED_FOSSILIZED' GROUP BY blueprint_id"
    )
    # Either has data or returns 0 rows message
    assert "|" in result or "0 rows" in result


def test_complex_join_works(real_server):
    """Can join across views."""
    result = real_server.query_sql_sync("""
        SELECT
            r.run_id,
            COUNT(DISTINCT e.env_id) as envs_seen
        FROM runs r
        LEFT JOIN epochs e ON e.timestamp > r.started_at
        GROUP BY r.run_id
        LIMIT 3
    """)
    assert "|" in result or "0 rows" in result
```

**Step 2: Run integration tests**

Run:
```bash
PYTHONPATH=src uv run pytest tests/karn/mcp/test_integration.py -v
```
Expected: All tests PASS (or skip if no telemetry data)

**Step 3: Commit**

```bash
git add tests/karn/mcp/test_integration.py
git commit -m "test(karn): add integration tests for MCP server with real telemetry"
```

---

## Task 7: Update Module Exports

**Files:**
- Modify: `src/esper/karn/mcp/__init__.py`

**Step 1: Add public exports**

Update `src/esper/karn/mcp/__init__.py`:

```python
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
```

**Step 2: Run all tests**

Run:
```bash
PYTHONPATH=src uv run pytest tests/karn/mcp/ -v
```
Expected: All tests PASS

**Step 3: Commit**

```bash
git add src/esper/karn/mcp/__init__.py
git commit -m "feat(karn): export MCP server public API"
```

---

## Task 8: Add MCP Client Configuration Documentation

**Files:**
- Create: `docs/manuals/mcp-setup.md`

**Step 1: Write setup documentation**

Create `docs/manuals/mcp-setup.md`:

```markdown
# Karn MCP Server Setup

The Karn MCP server exposes telemetry data via SQL queries.

## Claude Code Configuration

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "esper-karn": {
      "command": "uv",
      "args": ["run", "python", "-m", "esper.karn.mcp"],
      "cwd": "/path/to/esper-lite"
    }
  }
}
```

## Available Tools

### `query_sql`
Execute SQL against telemetry views.

**Parameters:**
- `query` (string): SQL query
- `limit` (int, default 100): Max rows to return

**Example:**
```sql
SELECT blueprint_id, COUNT(*) as fossilized
FROM seed_lifecycle
WHERE event_type = 'SEED_FOSSILIZED'
GROUP BY blueprint_id
ORDER BY fossilized DESC
```

### `list_views`
List available views and example queries.

## Views

| View | Description |
|------|-------------|
| `runs` | Run metadata (task, hyperparameters) |
| `epochs` | Per-env training progress |
| `ppo_updates` | PPO health metrics |
| `seed_lifecycle` | Seed state machine events |
| `rewards` | Per-step reward breakdown |
| `anomalies` | Training pathologies |
| `raw_events` | All events (for custom queries) |
```

**Step 2: Commit**

```bash
git add docs/manuals/mcp-setup.md
git commit -m "docs: add MCP server setup guide"
```

---

## Task 9: Final Verification

**Step 1: Run full test suite**

Run:
```bash
PYTHONPATH=src uv run pytest tests/karn/mcp/ -v --tb=short
```
Expected: All tests PASS

**Step 2: Verify CLI works**

Run:
```bash
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | timeout 5 uv run python -m esper.karn.mcp 2>/dev/null | head -1
```
Expected: JSON response listing `query_sql` and `list_views` tools

**Step 3: Clean up old plan file**

```bash
rm docs/plans/2025-12-19-karn-duckdb-mcp.md
git add -A
git commit -m "feat(karn): complete MCP server implementation

Adds DuckDB-based MCP server for querying telemetry data:
- 7 SQL views over JSONL telemetry files
- query_sql tool with timeout and limit enforcement
- list_views tool for discoverability
- Async wrapper for blocking DuckDB calls
- Integration tests with real telemetry

Usage: uv run python -m esper.karn.mcp"
```

---

## Summary

| Task | Description | Tests |
|------|-------------|-------|
| 1 | Add dependencies | - |
| 2 | View definitions | 3 |
| 3 | Query execution | 7 |
| 4 | MCP server class | 3 |
| 5 | MCP protocol wiring | - |
| 6 | Integration tests | 4 |
| 7 | Module exports | - |
| 8 | Documentation | - |
| 9 | Final verification | - |

**Total: 9 tasks, 17 tests**

---

Plan complete and saved to `docs/plans/2025-12-20-karn-duckdb-mcp.md`. Two execution options:

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach?
