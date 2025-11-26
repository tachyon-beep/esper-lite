# Work Package — Tamiyo Execution Remediation (WP-A)

## Context
Key issues from `TAMIYO_REVIEW_FINDINGS.md`, delta docs, and `lint_static_analysis.md`:
- `TamiyoService._apply_risk_engine` and `_evaluate` extremely complex; masking failures with pause commands.
- Blend annotations accept missing IDs/defaults; confidence mode uses activations, not logits.
- Telemetry gaps: coverage metrics, command verifier failures, emergency routing.
- Persistence: WAL rewrite lacks directory fsync; retention handles missing timestamps via silent defaults.
- Performance harness & rollback SLA missing.

## Work Streams
| ID | Goal |
|----|------|
| WP-A1 | Strict decision path & timeout behaviour |
| WP-A2 | Policy/blending validation & strict dependencies |
| WP-A3 | Telemetry completeness & routing |
| WP-A4 | Persistence & registry hardening |

## Status Summary (2025-10-03)
- **WP-A1** — ✅ Complete. Shared worker adoption and evaluator refactor merged (`src/esper/tamiyo/service.py:1380`), tests green (`tests/tamiyo/test_service.py:520`).
- **WP-A2** — ✅ Complete. Tamiyo policy now requires real seed/blueprint IDs (strict mode default), blend/ID dependency guard coverage added in unit/integration suites, and confidence telemetry validated. (`src/esper/tamiyo/policy.py:902`, `src/esper/tamiyo/service.py:2408`, `tests/tamiyo/test_service.py:394`, `tests/integration/test_control_loop.py:150`).
- **WP-A3** — ✅ Complete. Telemetry priority wiring and Weatherlight routing validated (`src/esper/tamiyo/service.py:2205`, `tests/weatherlight/test_service_priority.py:1`).
- **WP-A4** — ✅ Complete. WAL strict validation, backup/soak tooling, and telemetry metrics landed (`src/esper/tamiyo/persistence.py:67`, `scripts/tamiyo_wal_backup.py:1`, `scripts/tamiyo_wal_soak.py:1`).

### WP-A1 — Strict Decision Path *(Status: Complete 2025-09-27)*
Tasks:
1. Use shared async worker for policy inference and metadata; remove per-call executors.
2. Fail fast on inference/metadata timeout (no synthetic pause); surface CRITICAL telemetry.
3. Refactor `_apply_risk_engine` into per-signal evaluators; ensure conservative mode logic explicit.
4. Complexity reduction: target `_apply_risk_engine`, `_evaluate` ≤ C.
Acceptance:
- Integration test demonstrates timeout returns failure, not pause.
- Complexity report updated.
Risks:
- Policy pipeline depends on Tolaria updates; coordinate worker rollout.
- Risk engine refactor may change thresholds; add regression tests.

**Status Notes**
- Shared async worker adopted; timeout paths now emit `timeout_inference` telemetry with no synthetic pause fallback.
- `_apply_risk_engine` and `_evaluate` refactored into evaluator/orchestrator helpers and now grade at radon **A (3)** and **A (1)** respectively.
- Regression coverage: `tests/tamiyo/test_risk_engine.py` fixtures and `tests/tamiyo/test_service.py` confirm parity; docs/lint updated (see `CHANGELOG_RC1.md`, `lint_static_analysis.md`).

### WP-A2 — Policy & Blending Validation
Tasks:
1. Reject commands missing seed/blueprint IDs; remove fallback `seed-1`/`bp-demo`.
2. Validate blend annotations (channel/alpha vector limits, source tagging).
3. Ensure confidence mode gating uses logits from TamiyoPolicy; pass through via graph builder or step state.
4. Update tests (command parser, blend config) and telemetry.
Acceptance:
- Unit/integration tests covering invalid annotations/IDs.
- Confidence gating test verifying logits feed.
Risks:
- Tamiyo/Tolaria integration may need new annotations; document contract.
- Failure to supply logits may require Tamiyo policy update first.

**Status (2025-09-30)**
- Strict command IDs default on (`TamiyoPolicyConfig.strict_command_ids`), dependency guard now enforces non-empty `target_seed_id` and `seed_operation.blueprint_id` (`src/esper/tamiyo/service.py:2408`).
- Tests cover missing/invalid IDs and timeout blends (`tests/tamiyo/test_service.py:394`, `tests/tamiyo/test_service.py:409`, `tests/integration/test_control_loop.py:150`).
- Logit/blend annotations validated via existing policy tests; Kasmina integration confirms telemetry parity.

### WP-A3 — Telemetry & Routing
Tasks:
1. Emit coverage/annotation metrics per decision (feature coverage, risk reasons).
2. Hook command verifier to telemetry on failure (missing signature, stale command).
3. Ensure telemetry packets include priority and new fields; integrate with Weatherlight emergency stream.
4. Reduce complexity in telemetry builders (`_emit_field_report`, `_build_health_indicators`).
Acceptance:
- Telemetry inspection shows coverage map/types, verifier failures, emergency routing.
- Complexity metrics at ≤ C.
Risks:
- Weatherlight changes required; coordinate timeline.
- Additional telemetry may affect bandwidth; monitor queue lengths.

**Status (2025-09-29)**
- `_ensure_priority_indicator` stamps telemetry packets before publication (`src/esper/tamiyo/service.py:1380`), and Weatherlight honours the indicator (`src/esper/weatherlight/service_runner.py:1148`).
- Coverage and verifier telemetry asserted in `tests/tamiyo/test_service.py:520`, `tests/integration/test_weatherlight_tamiyo_emergency.py:1`.
- Observability runbook documents new metrics and emergency routing drill (`docs/project/observability_runbook.md:121`).

### WP-A4 — Persistence & Registry
Tasks:
1. WAL rewrite ensures parent directory fsync; resilience to truncated entries.
2. Enforce issued_at presence; fail retention if missing rather than default to “now”.
3. Policy registry: digest validation updates with clearer errors; doc differences.
4. Add CLI/regression tests for WAL retention & registry mismatches.
Acceptance:
- WAL corruption tests pass; retention errors surfaced.
- Registry mismatch raises actionable error with telemetry.
Risks:
- fsync changes may slow writes; measure and adjust.
- Registry digest rebuild may need heavier caching; plan accordingly.

**Status (2025-09-29)**
- WAL append now temp+rename+fsync (`src/esper/tamiyo/persistence.py:67`), strict reload surfaces truncation.
- Backup and soak scripts shipped with unit coverage (`scripts/tamiyo_wal_backup.py:1`, `tests/scripts/test_tamiyo_wal_backup.py:1`, `scripts/tamiyo_wal_soak.py:1`, `tests/scripts/test_tamiyo_wal_soak.py:1`).
- Summary telemetry exposes retry/drop/load-error metrics (`src/esper/tamiyo/service.py:2627`).

## Testing
- Unit: command verifier, policy config parsing, WAL retention.
- Integration: policy inference path with worker, coverage telemetry, Oona routing.
- Performance: existing Tamiyo perf tests (ensure still <45 ms p95).

## Rollback Plan
- Maintain ability to fall back to pause command via feature flag during testing.
- Keep WAL rewrite behind configuration until validated.

## Telemetry Verification
- `tamiyo.timeout_*`, `tamiyo.command_rejected`, coverage map, emergency routing events verified in logs.
- Blend telemetry aligns with Kasmina expectations.

## Sign-off Status
- WP-A1, WP-A3, and WP-A4 complete with tests & telemetry; WP-A2 remains open pending real ID enforcement and updated fixtures.
- Complexity reductions captured in `lint_static_analysis.md`.
- `CHANGELOG_RC1.md` updated with Tamiyo entries; final sign-off will occur once WP-A2 closes.

## Next Steps
- Tamiyo WPs are complete. Monitor Kasmina/Tolaria harness results under WP-CS1; no further Tamiyo-specific remediation planned unless harness uncovers regressions.
