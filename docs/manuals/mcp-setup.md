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
