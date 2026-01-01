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

**Returns:**
- JSON object: `{ok, columns, rows, row_count}` (or `{ok: false, error}` on failure)

**Example:**
```sql
SELECT blueprint_id, COUNT(*) as fossilized
FROM seed_lifecycle
WHERE event_type = 'SEED_FOSSILIZED'
GROUP BY blueprint_id
ORDER BY fossilized DESC
```

### `query_sql_markdown`
Execute SQL against telemetry views, returning a Markdown table (for copy/paste).

### `list_views`
List available views and example queries.

### `describe_view`
Describe a view’s columns and DuckDB types.

### `list_runs`
List available runs with basic metadata (`run_dir`, `episode_id`, etc).

### `run_overview`
Return a TUI-style overview report for a run (recommended starting point).

## Views

| View | Description |
|------|-------------|
| `runs` | Run metadata (task, hyperparameters) |
| `epochs` | Per-env training progress |
| `ppo_updates` | PPO health metrics |
| `batch_epochs` | Batch/episode progress events |
| `batch_stats` | Batch-level analytics snapshots |
| `seed_lifecycle` | Seed state machine events |
| `decisions` | Decision snapshots (last_action) |
| `rewards` | Per-step reward breakdown |
| `trends` | Plateau/degradation/improvement events |
| `anomalies` | Training pathologies |
| `episode_outcomes` | Per-episode outcome summaries |
| `raw_events` | All events (for custom queries) |
