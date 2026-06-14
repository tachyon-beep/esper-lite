## Task: UI Consumer Audit

Write output to: `docs/arch-analysis-2026-06-13-1138-telemetry-health/temp/ui-consumer-findings.md`

Read-only scope:

- `src/esper/karn/sanctum/`
- `src/esper/karn/overwatch/`
- `src/esper/karn/websocket_output.py`
- `scripts/generate_overwatch_types.py`
- `src/esper/karn/overwatch/web/src/`
- tests under `tests/karn/sanctum/`, `tests/karn/overwatch/`, and `src/esper/karn/overwatch/web/src/**/*.spec.ts`

Goal:

- Audit Sanctum schema/aggregator/widgets, Overwatch backend/web types/components, generated TypeScript drift, websocket output, sentinel/fallback display paths, and stale/misleading UI state.
- Identify where UI presents missing/zero/default telemetry as real signal.

Required output:

- UI consumer feed table.
- Findings with file/line evidence.
- Proposed tests or type-generation checks.

Constraints:

- Do not edit source.
- Do not run npm commands unless necessary; static source/test review is sufficient for this lane.

