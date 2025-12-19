# Plan: Karn Data Link (DuckDB MCP Integration)

**Status:** DRAFT
**Date:** 2025-12-19
**Goal:** Expose telemetry logs via a SQL interface (MCP) for surgical analysis of the "Final Exam" (Phase 1) and ongoing Phase 3 operations.

---

## 1. Strategic Context
Currently, analyzing training runs requires reading raw JSONL files. This is token-expensive and imprecise.
By integrating **DuckDB** via the **Model Context Protocol (MCP)**, we convert the `telemetry/` directory into a queryable database.

**Why DuckDB?**
- **Zero-Copy Ingest:** Can query `*.jsonl` files directly without importing them.
- **Analytical Speed:** Optimized for aggregates (AVG, SUM, P95) needed for RL analysis.
- **SQL Interface:** Allows the agent to ask specific questions ("Show me the entropy trend for Cohort B") rather than reading the entire history.

---

## 2. Architecture

We will build a custom MCP server within the `esper.karn` domain.

### 2.1 Component Stack
- **Library:** `duckdb` (Python bindings)
- **Protocol:** `mcp` (Python SDK)
- **Location:** `src/esper/karn/mcp_server.py`

### 2.2 Data Views
The server will auto-initialize with "Virtual Views" over the raw JSONL data to simplify querying:

1.  **`raw_events`**: The base view over all JSONL files.
    ```sql
    CREATE VIEW raw_events AS SELECT * FROM read_json_auto('telemetry/*.jsonl');
    ```
2.  **`runs`**: High-level run metadata (extracted from `TRAINING_STARTED` events).
3.  **`metrics`**: Time-series data (epoch, loss, entropy) extracted from `PPO_UPDATE` events.
4.  **`decisions`**: Tamiyo's choices (germinate/cull/wait) extracted from `SEED_*` and `REWARD_COMPUTED` events.

---

## 3. Implementation Steps

### Task 1: Dependencies
Add required packages to `pyproject.toml`:
```bash
uv add duckdb mcp
```

### Task 2: The Server (`src/esper/karn/mcp_server.py`)
Implement an MCP server that exposes a single resource or tool: `query_sql`.

**Key Features:**
- **Startup:** Initialize DuckDB connection.
- **Auto-Discovery:** Scan `telemetry/` for `.jsonl` files.
- **View Creation:** Execute `CREATE OR REPLACE VIEW` statements to structure the JSON data.
- **Tool:** `query_sql(query: str) -> str`
    - Executes the query.
    - Returns results as Markdown table or JSON.
    - Limits rows (default 100) to prevent token floods.

### Task 3: The Schema Definition
Define the standard views to abstract away JSON parsing complexity.

*Example View (`decisions`):*
```sql
CREATE VIEW decisions AS
SELECT
    timestamp,
    json_extract(data, '$.env_id') as env_id,
    json_extract(data, '$.action_name') as action,
    json_extract(data, '$.blueprint_id') as blueprint,
    json_extract(data, '$.stage') as stage
FROM raw_events
WHERE event_type IN ('SEED_GERMINATED', 'SEED_CULLED', 'REWARD_COMPUTED')
```

### Task 4: Integration
Add the server config to the user's MCP client configuration (e.g., `claude_desktop_config.json` or `cursor settings`).

```json
{
  "mcpServers": {
    "esper-karn": {
      "command": "uv",
      "args": ["run", "python", "-m", "esper.karn.mcp_server"]
    }
  }
}
```

---

## 4. Usage Example (The "Final Exam")

**User:** "Analyze the A/B test results. Which cohort had better entropy stability?"

**Agent (via MCP):**
1.  Calls `query_sql("SELECT * FROM runs")` to identify Cohort A vs B IDs.
2.  Calls `query_sql("SELECT run_id, AVG(entropy), STDDEV(entropy) FROM metrics GROUP BY run_id")`.
3.  **Result:** "Cohort B (Simplified) had 15% lower entropy variance (0.12 vs 0.14) indicating more decisive policy convergence."

---

## 5. Success Criteria
1.  **Latency:** Queries return in <1s.
2.  **Stability:** Server handles malformed JSONL lines gracefully (DuckDB `ignore_errors=true`).
3.  **Utility:** Can successfully join `runs` (metadata) with `metrics` (time-series) in a single SQL query.
