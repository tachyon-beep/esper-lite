## Task: Leyline and Nissa Contract Audit

Write output to: `docs/arch-analysis-2026-06-13-1138-telemetry-health/temp/leyline-nissa-findings.md`

Read-only scope:

- `src/esper/leyline/telemetry.py`
- `src/esper/nissa/config.py`
- `src/esper/nissa/output.py`
- `src/esper/nissa/tracker.py`
- `src/esper/nissa/analytics.py`
- `src/esper/nissa/wandb_backend.py`
- `tests/leyline/test_telemetry*.py`
- `tests/nissa/`

Goal:

- Audit shared schemas, event envelope requirements, type/field preservation, telemetry level semantics, file/directory/console/W&B output behavior, and global hub reset behavior.
- Look for missing required fields, optional fields rendered as real facts, dropped fields, duplicate backend state, stale hub state, and serialization mutation.

Required output:

- Producer/contract/backend matrix.
- Findings with current line evidence.
- Test coverage notes and proposed acceptance tests.
- Tracker-ready issue rows.

Constraints:

- Do not edit source.
- Treat absent data as suspicious when downstream consumers use it as evidence.

