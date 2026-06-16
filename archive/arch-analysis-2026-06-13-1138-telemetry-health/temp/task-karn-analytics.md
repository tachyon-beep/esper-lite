## Task: Karn Analytics and Proof Audit

Write output to: `docs/arch-analysis-2026-06-13-1138-telemetry-health/temp/karn-analytics-findings.md`

Read-only scope:

- `src/esper/karn/collector.py`
- `src/esper/karn/store.py`
- `src/esper/karn/ingest.py`
- `src/esper/karn/serialization.py`
- `src/esper/karn/health.py`
- `src/esper/karn/pareto.py`
- `src/esper/karn/mcp/`
- `scripts/proof_packet.py`
- tests under `tests/karn/`, `tests/karn/mcp/`, and `tests/telemetry/`

Goal:

- Audit event ingestion, stateful aggregation, store import/export, DuckDB/MCP views, proof packet queries, health/pareto derived metrics, field preservation, identity preservation, and misleading derived metrics.
- Identify proof-invalidating gaps for reward-efficiency or signal-of-life claims.

Required output:

- Karn/proof feed inventory.
- Findings with file/line evidence.
- Tracker-ready issue rows and acceptance tests.

Constraints:

- Do not edit source.
- Treat proof packet failure to fail-closed as high severity.

