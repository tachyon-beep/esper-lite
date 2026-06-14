## Task: Kasmina and Tolaria Signal Audit

Write output to: `docs/arch-analysis-2026-06-13-1138-telemetry-health/temp/kasmina-tolaria-findings.md`

Read-only scope:

- `src/esper/kasmina/`
- `src/esper/tolaria/`
- lifecycle/governor telemetry consumers in Simic/Karn as needed
- tests under `tests/kasmina/`, `tests/tolaria/`, and `tests/integration/test_governor_rollback.py`

Goal:

- Audit lifecycle, slot/seed/gate, gradient health, alpha, fossilize/prune, rollback, and governor telemetry.
- Determine whether safety/lifecycle events carry real causal evidence or placeholders/defaults.
- Look for events emitted out of order, stale state, missing env/slot/event identity, or rollback/fossilization facts that cannot support proof claims.

Required output:

- Lifecycle/governor feed inventory.
- Findings with source line evidence.
- Tracker-ready issue rows and acceptance tests.

Constraints:

- Do not edit source.
- Keep body/organism metaphor for architecture and botanical terms for seed lifecycle only.

