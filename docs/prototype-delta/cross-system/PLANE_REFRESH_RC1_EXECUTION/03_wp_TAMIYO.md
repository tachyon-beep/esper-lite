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

## Sign-off
- WP-A1..A4 tasks completed with tests&telemetry.
- Complexity reductions captured in `lint_static_analysis.md`.
- `CHANGELOG_RC1.md` updated with Tamiyo entries.
