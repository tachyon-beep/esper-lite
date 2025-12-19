# Plan: Karn Data Link (DuckDB MCP Integration)

**Status:** READY FOR IMPLEMENTATION
**Date:** 2025-12-19 (Updated 2025-12-20)
**Goal:** Expose telemetry logs via a SQL interface (MCP) for surgical analysis of training runs.

---

## 1. Strategic Context

Currently, analyzing training runs requires reading raw JSONL files. This is token-expensive and imprecise.
By integrating **DuckDB** via the **Model Context Protocol (MCP)**, we convert the `telemetry/` directory into a queryable database.

**Why DuckDB?**
- **Zero-Copy Ingest:** Can query `*.jsonl` files directly without importing them.
- **Analytical Speed:** Optimized for aggregates (AVG, SUM, P95) needed for RL analysis.
- **SQL Interface:** Allows the agent to ask specific questions ("Show me the entropy trend for env 5") rather than reading the entire history.

**Why MCP?**
- Purpose-built for LLM tool use with structured responses
- Native integration with Claude Code and similar tools
- SQL *is* the interface - no API design needed

---

## 2. Architecture

We will build a custom MCP server within the `esper.karn` domain.

### 2.1 Component Stack
- **Library:** `duckdb` (Python bindings)
- **Protocol:** `mcp` (Official Python SDK from PyPI)
- **Location:** `src/esper/karn/mcp_server.py`

### 2.2 Directory Structure
Telemetry files are organized as:
```
telemetry/
├── telemetry_2025-12-19_115120/
│   └── events.jsonl
├── telemetry_2025-12-20_022855/
│   └── events.jsonl
└── ...
```

---

## 3. Data Views

The server will auto-initialize with views over the raw JSONL data. These are based on the **actual telemetry schema** (audited 2025-12-20).

### 3.1 Base View: `raw_events`
```sql
CREATE VIEW raw_events AS
SELECT * FROM read_json_auto(
    'telemetry/*/events.jsonl',
    ignore_errors=true,
    maximum_object_size=16777216
);
```

### 3.2 View: `runs` (Run Metadata)
Extracted from `TRAINING_STARTED` events.
```sql
CREATE VIEW runs AS
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
    json_extract(data, '$.host_params')::INTEGER as host_params,
    data->'$.slot_ids' as slot_ids
FROM raw_events
WHERE event_type = 'TRAINING_STARTED';
```

### 3.3 View: `epochs` (Per-Environment Training Progress)
Extracted from `EPOCH_COMPLETED` events (~25k events in sample).
```sql
CREATE VIEW epochs AS
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
WHERE event_type = 'EPOCH_COMPLETED';
```

### 3.4 View: `ppo_updates` (PPO Training Health)
Extracted from `PPO_UPDATE_COMPLETED` events. Critical for diagnosing training issues.
```sql
CREATE VIEW ppo_updates AS
SELECT
    timestamp,
    epoch as episodes_completed,
    json_extract(data, '$.inner_epoch')::INTEGER as inner_epoch,
    json_extract(data, '$.policy_loss')::DOUBLE as policy_loss,
    json_extract(data, '$.value_loss')::DOUBLE as value_loss,
    json_extract(data, '$.entropy')::DOUBLE as entropy,
    json_extract(data, '$.entropy_coef')::DOUBLE as entropy_coef,
    json_extract(data, '$.kl_divergence')::DOUBLE as kl_divergence,
    json_extract(data, '$.clip_fraction')::DOUBLE as clip_fraction,
    json_extract(data, '$.explained_variance')::DOUBLE as explained_variance,
    json_extract(data, '$.ratio_max')::DOUBLE as ratio_max,
    json_extract(data, '$.ratio_min')::DOUBLE as ratio_min,
    json_extract(data, '$.avg_accuracy')::DOUBLE as avg_accuracy,
    json_extract(data, '$.avg_reward')::DOUBLE as avg_reward,
    json_extract(data, '$.rolling_avg_accuracy')::DOUBLE as rolling_avg_accuracy,
    json_extract(data, '$.grad_norm')::DOUBLE as grad_norm,
    json_extract(data, '$.update_time_ms')::DOUBLE as update_time_ms,
    json_extract(data, '$.dead_layers')::INTEGER as dead_layers,
    json_extract(data, '$.exploding_layers')::INTEGER as exploding_layers,
    json_extract(data, '$.entropy_collapsed')::BOOLEAN as entropy_collapsed,
    -- Per-head entropy breakdown
    json_extract(data, '$.slot_entropy')::DOUBLE as slot_entropy,
    json_extract(data, '$.blueprint_entropy')::DOUBLE as blueprint_entropy,
    json_extract(data, '$.blend_entropy')::DOUBLE as blend_entropy,
    json_extract(data, '$.op_entropy')::DOUBLE as op_entropy
FROM raw_events
WHERE event_type = 'PPO_UPDATE_COMPLETED';
```

### 3.5 View: `seed_lifecycle` (Seed State Machine)
Tracks seed germination, stage transitions, fossilization, and culling.
```sql
CREATE VIEW seed_lifecycle AS
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
    -- Stage change fields
    json_extract_string(data, '$.from') as from_stage,
    json_extract_string(data, '$.to') as to_stage,
    -- Fossilized/Culled outcome fields
    json_extract(data, '$.improvement')::DOUBLE as improvement,
    json_extract(data, '$.blending_delta')::DOUBLE as blending_delta,
    json_extract(data, '$.counterfactual')::DOUBLE as counterfactual,
    json_extract(data, '$.epochs_total')::INTEGER as epochs_total,
    json_extract(data, '$.gradient_health')::DOUBLE as gradient_health,
    json_extract_string(data, '$.reason') as cull_reason,
    json_extract(data, '$.auto_culled')::BOOLEAN as auto_culled
FROM raw_events
WHERE event_type IN (
    'SEED_GERMINATED',
    'SEED_STAGE_CHANGED',
    'SEED_FOSSILIZED',
    'SEED_CULLED'
);
```

### 3.6 View: `rewards` (Per-Step Reward Breakdown)
Extracted from `REWARD_COMPUTED` events (~45k in sample). Essential for reward debugging.
```sql
CREATE VIEW rewards AS
SELECT
    timestamp,
    epoch,
    json_extract(data, '$.env_id')::INTEGER as env_id,
    json_extract(data, '$.episode')::INTEGER as episode,
    json_extract_string(data, '$.ab_group') as ab_group,
    json_extract_string(data, '$.action_name') as action_name,
    json_extract(data, '$.action_success')::BOOLEAN as action_success,
    json_extract_string(data, '$.seed_stage') as seed_stage,
    -- Reward components
    json_extract(data, '$.total_reward')::DOUBLE as total_reward,
    json_extract(data, '$.base_acc_delta')::DOUBLE as base_acc_delta,
    json_extract(data, '$.bounded_attribution')::DOUBLE as bounded_attribution,
    json_extract(data, '$.ratio_penalty')::DOUBLE as ratio_penalty,
    json_extract(data, '$.compute_rent')::DOUBLE as compute_rent,
    json_extract(data, '$.stage_bonus')::DOUBLE as stage_bonus,
    json_extract(data, '$.pbrs_bonus')::DOUBLE as pbrs_bonus,
    json_extract(data, '$.action_shaping')::DOUBLE as action_shaping,
    json_extract(data, '$.terminal_bonus')::DOUBLE as terminal_bonus,
    json_extract(data, '$.fossilize_terminal_bonus')::DOUBLE as fossilize_terminal_bonus,
    -- Context
    json_extract(data, '$.val_acc')::DOUBLE as val_acc,
    json_extract(data, '$.growth_ratio')::DOUBLE as growth_ratio,
    json_extract(data, '$.num_fossilized_seeds')::INTEGER as num_fossilized_seeds
FROM raw_events
WHERE event_type = 'REWARD_COMPUTED';
```

### 3.7 View: `anomalies` (Training Pathologies)
Aggregates all anomaly detection events for quick diagnosis.
```sql
CREATE VIEW anomalies AS
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
);
```

---

## 4. Implementation Tasks

### Task 1: Dependencies
Add required packages to `pyproject.toml`:
```bash
uv add duckdb mcp
```

### Task 2: MCP Server (`src/esper/karn/mcp_server.py`)

Implement an MCP server exposing a `query_sql` tool. This is a **standalone entry point** (like `overwatch/app.py`), not imported by other modules.

**Key Features:**
- **Startup:** Initialize DuckDB in-memory connection with `PRAGMA threads=4` for parallel scans
- **Auto-Discovery:** Scan `telemetry/` for JSONL files
- **View Creation:** Execute all `CREATE VIEW` statements on startup
- **Tool:** `query_sql(query: str, limit: int = 100) -> str`
    - Executes the query with `LIMIT` enforcement (detect existing LIMIT to avoid double-limiting)
    - Returns results as Markdown table
    - Handles errors gracefully with informative messages (see error handling below)
- **Query Timeout:** 30 second limit to prevent runaway scans

**Error Handling:**

| Error Type | Response |
|------------|----------|
| SQL syntax error | `"SQL Error: {duckdb_error_message}"` |
| Query timeout (>30s) | `"Query exceeded 30s timeout. Try adding filters or reducing scope."` |
| Empty result set | `"Query returned 0 rows."` (not an error, but clarify) |
| No telemetry files | `"No telemetry data found in telemetry/ directory."` |

**Async Pattern:**

DuckDB Python bindings are synchronous (blocking). Wrap in `asyncio.to_thread()` to keep the event loop responsive:

```python
"""Karn MCP Server - SQL interface to telemetry data.

NOTE: Views depend on esper.leyline.telemetry.TelemetryEvent schema.
Breaking changes to TelemetryEvent.data fields may require view updates.
"""
from __future__ import annotations

import asyncio
import re
from typing import Any

import duckdb
from mcp.server import Server
from mcp.server.stdio import stdio_server

app = Server("esper-karn")

# Module-level connection (initialized on startup)
_conn: duckdb.DuckDBPyConnection | None = None


def _init_connection(telemetry_dir: str = "telemetry") -> duckdb.DuckDBPyConnection:
    """Initialize DuckDB connection with views."""
    conn = duckdb.connect(":memory:")
    conn.execute("PRAGMA threads=4")
    # Create views here...
    return conn


def _execute_query_sync(query: str, limit: int) -> tuple[list[str], list[tuple[Any, ...]]]:
    """Execute query synchronously (called via to_thread)."""
    assert _conn is not None
    # Add LIMIT if not present
    if not re.search(r"\bLIMIT\s+\d+", query, re.IGNORECASE):
        query = f"{query.rstrip().rstrip(';')} LIMIT {limit}"
    result = _conn.execute(query)
    columns = [desc[0] for desc in result.description]
    rows = result.fetchall()
    return columns, rows


def _format_as_markdown(columns: list[str], rows: list[tuple[Any, ...]]) -> str:
    """Format query result as Markdown table."""
    if not rows:
        return "Query returned 0 rows."
    # Header
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    # Rows
    body = "\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in rows)
    return f"{header}\n{separator}\n{body}"


@app.tool()
async def query_sql(query: str, limit: int = 100) -> str:
    """Execute SQL against telemetry data. Returns Markdown table."""
    try:
        columns, rows = await asyncio.wait_for(
            asyncio.to_thread(_execute_query_sync, query, limit),
            timeout=30.0
        )
        return _format_as_markdown(columns, rows)
    except asyncio.TimeoutError:
        return "Query exceeded 30s timeout. Try adding filters or reducing scope."
    except duckdb.Error as e:
        return f"SQL Error: {e}"


async def main() -> None:
    """Run the MCP server."""
    global _conn
    _conn = _init_connection()
    async with stdio_server(app):
        await asyncio.Event().wait()  # Run forever


if __name__ == "__main__":
    asyncio.run(main())
```

### Task 3: CLI Entry Point

The skeleton above includes `if __name__ == "__main__"` block for direct execution:
```bash
uv run python -m esper.karn.mcp_server
```

**Note:** No changes to `src/esper/karn/__init__.py` required - this is a standalone entry point not imported by other Karn modules.

### Task 4: MCP Client Configuration

Add to user's Claude Code MCP config (e.g., `~/.claude/claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "esper-karn": {
      "command": "uv",
      "args": ["run", "python", "-m", "esper.karn.mcp_server"],
      "cwd": "/path/to/esper-lite"
    }
  }
}
```

---

## 5. Usage Examples

### Example 1: Compare Cohort Entropy Stability
```sql
SELECT
    r.run_id,
    AVG(p.entropy) as avg_entropy,
    STDDEV(p.entropy) as entropy_stddev
FROM runs r
JOIN ppo_updates p ON p.timestamp > r.started_at
GROUP BY r.run_id
ORDER BY entropy_stddev ASC;
```

### Example 2: Seed Success Rate by Blueprint
```sql
SELECT
    blueprint_id,
    COUNT(*) FILTER (WHERE event_type = 'SEED_FOSSILIZED') as fossilized,
    COUNT(*) FILTER (WHERE event_type = 'SEED_CULLED') as culled,
    ROUND(100.0 * COUNT(*) FILTER (WHERE event_type = 'SEED_FOSSILIZED') / COUNT(*), 1) as success_rate
FROM seed_lifecycle
WHERE event_type IN ('SEED_FOSSILIZED', 'SEED_CULLED')
GROUP BY blueprint_id
ORDER BY success_rate DESC;
```

### Example 3: Reward Component Analysis
```sql
SELECT
    action_name,
    COUNT(*) as count,
    AVG(total_reward) as avg_reward,
    AVG(action_shaping) as avg_action_shaping,
    AVG(compute_rent) as avg_rent,
    AVG(bounded_attribution) as avg_attribution
FROM rewards
GROUP BY action_name
ORDER BY count DESC;
```

### Example 4: Training Anomalies Timeline
```sql
SELECT
    timestamp,
    event_type,
    message
FROM anomalies
ORDER BY timestamp DESC
LIMIT 20;
```

### Example 5: Per-Environment Accuracy Progression
```sql
SELECT
    env_id,
    MAX(val_accuracy) as peak_accuracy,
    MAX(val_accuracy) - MIN(val_accuracy) as accuracy_range
FROM epochs
GROUP BY env_id
ORDER BY peak_accuracy DESC;
```

---

## 6. Success Criteria

1. **Latency:** Queries return in <1s for typical aggregates
2. **Stability:** Server handles malformed JSONL lines gracefully (`ignore_errors=true`)
3. **Coverage:** All 7 views queryable and joinable
4. **Token Efficiency:** 10x reduction in tokens needed for telemetry analysis vs raw JSONL reads
5. **Integration:** Works with Claude Code MCP configuration

---

## 7. Event Type Reference

Telemetry event frequency from recent runs (for reference):

| Event Type | Count | Description |
|------------|-------|-------------|
| `ANALYTICS_SNAPSHOT` | 45,964 | Full state snapshots for dashboards |
| `REWARD_COMPUTED` | 45,329 | Per-step reward breakdown |
| `EPOCH_COMPLETED` | 24,776 | Per-env epoch metrics |
| `SEED_STAGE_CHANGED` | 16,276 | Lifecycle transitions |
| `SEED_GERMINATED` | 6,397 | New seed creation |
| `SEED_CULLED` | 5,801 | Seed removal |
| `COUNTERFACTUAL_COMPUTED` | 381 | Ablation measurements |
| `TAMIYO_INITIATED` | 372 | Host stabilization events |
| `SEED_FOSSILIZED` | 359 | Successful integration |
| `PPO_UPDATE_COMPLETED` | 61 | PPO batch updates |
| `BATCH_COMPLETED` | 56 | Training batch completion |
| `PLATEAU_DETECTED` | 51 | Training plateau alerts |
| `VALUE_COLLAPSE_DETECTED` | 26 | Value function anomalies |
| `GOVERNOR_ROLLBACK` | 21 | Emergency rollbacks |
| `TRAINING_STARTED` | 4 | Run initialization |

---

## 8. Implementation Notes

### 8.1 DuckDB Considerations

- **Concurrent Access:** DuckDB in-memory mode is single-process. Each MCP client spawns a separate process with its own DB instance. This is fine for read-only queries.
- **View Materialization:** Views are virtual (not materialized). Every query re-scans JSONL files. `PRAGMA threads=4` helps parallelize scans for large telemetry directories.
- **JSON Null Handling:** `json_extract()` returns NULL for missing fields - safe for schema evolution.

### 8.2 Additional Tool: `list_views`

Consider adding a helper tool for discoverability:

```python
@app.tool()
async def list_views() -> str:
    """List available views and their descriptions."""
    return """Available views:
- `runs` - Run metadata (task, hyperparameters, devices)
- `epochs` - Per-environment training progress (accuracy, loss)
- `ppo_updates` - PPO health metrics (entropy, KL, clip fraction)
- `seed_lifecycle` - Seed state machine (germinate, stage change, fossilize, cull)
- `rewards` - Per-step reward breakdown (all components)
- `anomalies` - Training pathologies (collapses, rollbacks, plateaus)
- `raw_events` - All events (use for custom queries)

Example: SELECT * FROM ppo_updates ORDER BY timestamp DESC LIMIT 5;"""
```

### 8.3 Testing Strategy

1. **Unit tests:** Test view SQL against fixture JSONL files
2. **Integration test:** Start server, call `query_sql` via MCP client
3. **Error handling:** Test malformed queries, timeout simulation, empty results

### 8.4 Future Enhancements (Out of Scope)

- **Materialized views:** For very large telemetry dirs (1000+ runs), consider persisting to DuckDB file
- **Time-range filtering:** Add `--since` flag to limit scan scope
- **Schema introspection:** Auto-generate views from TelemetryEventType enum
