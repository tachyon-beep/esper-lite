## Task: Test and Guardrail Validation Audit

Write output to: `docs/arch-analysis-2026-06-13-1138-telemetry-health/temp/test-guardrail-findings.md`

Read-only scope:

- `pytest.ini`
- `scripts/lint_defensive_patterns.py`
- `scripts/lint_gpu_sync.py`
- telemetry-focused tests listed in the campaign plan
- test fixtures under `tests/fixtures/telemetry/`

Goal:

- Audit whether current tests cover telemetry producer/contract/backend/consumer paths.
- Identify missing tests for each telemetry failure mode.
- Run the planned guardrails and focused telemetry tests if feasible, capturing exact commands and outcomes.
- Validate the final report once coordinator writes it, if asked later.

Initial verification commands:

```bash
uv run python scripts/lint_defensive_patterns.py
uv run python scripts/lint_gpu_sync.py
uv run pytest tests/leyline/test_telemetry.py tests/leyline/test_telemetry_events.py tests/nissa/test_output.py tests/nissa/test_tracker.py tests/nissa/test_wandb_backend.py tests/simic/telemetry/test_emitters.py tests/simic/test_telemetry_fields.py tests/simic/test_telemetry_integration.py tests/integration/test_reward_telemetry_flow.py tests/integration/test_q_values_telemetry.py tests/karn/test_store_export.py tests/karn/test_ingest.py tests/karn/mcp/test_views.py tests/karn/sanctum/test_schema.py tests/telemetry
```

Required output:

- Command outcomes with exit codes and key failure text.
- Coverage gaps grouped by telemetry failure mode.
- Recommendations for acceptance tests in tracker-ready format.

Constraints:

- Do not edit source.
- Running tests may write caches/artifacts; do not modify tracked source files.

