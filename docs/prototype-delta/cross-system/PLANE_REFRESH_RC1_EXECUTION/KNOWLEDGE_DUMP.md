# Plane Refresh RC1 – Execution Knowledge Pack

## Scope & Goals
- Align Tolaria, Tamiyo, Kasmina with prototype-delta (remove fallbacks, adopt shared async worker, enforce strict telemetry/command contracts).
- Eliminate stringly-typed maps: `AdaptationCommand.annotations`, `FieldReport.metrics`, `SystemStatePacket.training_metrics`.
- Preserve telemetry routing, command security, and integration with Weatherlight, Oona, Nissa, Simic.

## Key References
- Work package folder: `docs/prototype-delta/cross-system/PLANE_REFRESH_RC1_EXECUTION/`
  - Subsystem work packages: `02_wp_TOLARIA.md`, `03_wp_TAMIYO.md`, `04_wp_KASMINA.md`
  - Shared foundations: `05_shared_foundations.md` (state assessments, draft schema, helper APIs, consumer mapping, integration touchpoints)
  - Testing plan: `06_testing_validation_plan.md`
  - Risk register: `07_risk_register.md`
  - Status tracker: `08_status_tracker.md`
- Architecture summary: `docs/architecture_summary.md`
- Observability docs: `docs/project/observability_runbook.md`, `docs/prototype-delta/nissa/README.md`

## Current Architecture Notes
- Tamiyo enriches commands via annotations (policy, risk, blend, coverage, pause, signature).
- Field reports store metrics in a map; Simic replay uses `loss_delta` and positional value ordering.
- Tolaria populates `SystemStatePacket.training_metrics` map; Tamiyo uses it for global features and field-report latencies.
- Telemetry priority currently stored in `system_health.indicators["priority"]`; Weatherlight re-derives priority when publishing.

## Planned Proto Changes
```
message BusEnvelope { ... MessagePriority priority = 4; }
message TelemetryPacket { ... MessagePriority priority = 50; }
message AdaptationCommand { DecisionMetadata metadata = 16; ... }
message CommandSeedOperation { ... optional fields; map deprecated }
message FieldReport { FieldReportDelta delta = 14; repeated extra_metrics }
message SystemStatePacket { TrainingMetrics metrics_struct = 50; }
message CommandSecurity { string signature = 1; string nonce = 2; google.protobuf.Timestamp issued_at = 3; google.protobuf.Duration freshness_window = 4; }
```
(See `05_shared_foundations.md` for full draft definitions.)

## Helper Modules
- `esper.core.telemetry` – build/publish packets with structured priority.
- `esper.core.decisions` – attach/parse `DecisionMetadata` (temporary mirroring to annotations during migration if needed).
- `esper.core.training_metrics` – pack/unpack structured Tolaria metrics.
- `esper.core.dependency_guard` – shared strict dependency checks (IDs, fallbacks, training-run IDs); now used by Tamiyo/Tolaria/Kasmina after 2025-09-26 guard rollout.

## Subsystem Update Summary
- **Tolaria**: adopt shared async worker, populate structured `TrainingMetrics`, set `TelemetryPacket.priority`, include `DecisionMetadata`. Simplify aggregator per WP.
- **Tamiyo**: write structured `AdaptationCommand`, `FieldReportDelta`, telemetry; enforce dependency guard; update WAL tests.
- **Kasmina**: consume structured command fields (blend, resume, security), remove annotation parsing, fail fast on missing IDs, emit telemetry priority.
- **Weatherlight/Oona**: forward `TelemetryPacket.priority`/`BusEnvelope.priority` without recomputing; retain emergency routing behaviour.
- **Simic**: ingest `FieldReportDelta`, remove map-order reliance, update trainer/replay tests.

## External Consumers & Contracts
- **Nissa**: metrics `tamiyo.gnn.feature_coverage*`, `tamiyo.blueprint.risk`, `tamiyo.bsds.*`; expect unchanged names. (src/esper/nissa/observability.py, alerts.py)
- **Grafana dashboards**: rely on Prometheus counter `esper_field_reports_total` (infra/grafana/...)
- **Docs/Runbooks**: reference tamiyo metrics & field-report locations; update alongside changes.

## Testing Expectations
- Update unit/integration suites (Tamiyo, Kasmina, Tolaria, Simic, Weatherlight) to exercise new structs & strict behaviours.
- Add coverage for async worker/cancellation, telemetry priority routing, `FieldReportDelta` ingestion.
- Ensure `FieldReport` WAL serialization remains valid (proto binary, no external parser).

## Integration Touchpoints
- Tolaria → Tamiyo: synchronous command calls, telemetry (trainer.py ↔ service.py).
- Tamiyo → Kasmina: commands consumed in training loop (seed_manager.py).
- Tamiyo ↔ Urza: metadata fetch for risk/blend.
- Tamiyo → Oona/Nissa: telemetry & field reports published via Oona; Weatherlight flush & priority handling.
- Kasmina ↔ Urza/Oona: kernel fetch, telemetry forwarding.
- Weatherlight: flush telemetry/history; adopt priority fields.
- Simic replay/trainer: Wal + Oona ingestion uses field reports.
- Observability stack: metrics must retain names for dashboards/alerts.

## Prototype Principles Alignment
- Strict deps: enforce via `dependency_guard`; no new pseudo-optionals.
- No backwards compatibility: migrate directly to structured fields; remove maps/annotations in same PR.
- No masking: command security, dependency guard, telemetry priority all fail fast.
- No partial degradation: continue to treat missing subsystems as hard failures; maintain Weatherlight mode signalling.
- Code hygiene: remove legacy map handling; single implementation paths documented in shared foundations.
- Testing posture: update tests to cover success/guardrails with explicit fakes; integration through real contracts.
- Documentation: update architecture summary, runbooks, Grafana references during implementation.

## Suggested Execution Order
1. Finalize proto/schema updates + regenerate leyline bindings.
2. Implement shared helpers and dependency guard.
3. Update Tolaria/Tamiyo/Kasmina to produce/consume structured messages + priority. ✅
4. Adapt Weatherlight/Oona for priority field. ✅ (priority guard in Weatherlight preflight pending telemetry load tests).
5. Refactor Simic replay/trainer to `FieldReportDelta`.
6. Update tests/docs/dashboards.
7. Run full test suite + targeted integration checks.

## Monitoring Risks
- Async worker cancellation regressions (existing risk register items).
- Exposing previously masked dependency failures once fallbacks removed.
- Metric name drift causing observability regressions.
- Coordinated proto adoption across subsystems/tests (no adapters).

## Kasmina R4c Baseline (2025-09-27)
- **Unit suites**: `tests/kasmina/test_blend_annotations.py`, `test_blending.py`, `test_seed_manager.py`, `test_lifecycle.py`, `test_safety.py`, `test_isolation_scope.py` (15 failures in `test_seed_manager.py` due to strict dependency guard rejecting commands without `training_run_id`; illustrates current fallback reliance).
- **Integration**: `tests/integration/test_control_loop.py::test_kasmina_emits_one_packet_per_seed_per_step` fails (no per-seed telemetry packets emitted when fallback identity kernels are used).
- **Complexity snapshot**:
  - `KasminaSeedManager.handle_command` — F (58)
  - `_graft_seed` — D (22)
  - `_apply_blend_annotations` — C (20)
  - `_resume_seed` — C (19)
  - `KasminaGates.evaluate` — B (7)
- **Telemetry observations**: Seed-manager tests expect fallback kernels to keep commands alive; telemetry lacks CRITICAL gate failure events when fallbacks engaged. Command annotations omit `training_run_id`, triggering dependency violations.
- **Action items**: R4c must supply seed commands with real IDs (or enforce preflight), ensure gate failures emit telemetry, and remove fallback identity usage.

## Kasmina R4c Completion (2025-09-28)
- **Dispatcher cut-over**: `_DISPATCHER_EXPERIMENTAL` removed; `handle_command` always routes through the dispatcher with `_finalize_command_outcome` managing telemetry queues.
- **Strict failure**: Seed graft/resume paths raise `DependencyViolationError` on runtime failures; fallback kernels eliminated. `_GateEvaluator` now triggers CRITICAL `gate_failure` telemetry for fallback/stage mismatch/latency violations.
- **Blend enforcement**: `_BlendManager` enforces bounded `alpha_vec` (≤64) and deterministic signing; missing or malformed channel configs fail fast. Confidence mode requires Tamiyo logits and emits telemetry when absent.
- **Async worker resilience**: `AsyncWorker.shutdown` now applies bounded joins and final `loop.stop`, preventing Tolaria integration tests from hanging. Integration fixtures disable the worker (`tamiyo_timeout_s=0.0`) for deterministic execution.
- **Tests executed**: `pytest tests/kasmina -q`, `pytest tests/integration/test_control_loop.py -q`, `radon cc -s src/esper/kasmina/seed_manager.py` (reports `handle_command` at grade A).
- **Open items**: Kasmina command verifier telemetry/nonce cleanup, prefetch/cache locking, and performance benchmarks remain tied to WP-K3/WP-K4.

## Kasmina WP-K3 Baseline (2025-09-29)
- **Command verifier telemetry**: `_verify_command` appends a single `TelemetryEvent(description="command_rejected", level=ERROR)` when verification fails. Priority resolves to `MESSAGE_PRIORITY_HIGH`, so CRITICAL routing for signature/nonce failures is presently absent. No metrics or counters accompany verifier outcomes.
- **Coverage gaps**: Unit tests only cover missing-signature rejection (`tests/kasmina/test_seed_manager.py::test_manager_rejects_unsigned_command`). There is no fixture exercising invalid signatures, nonce replay, or stale timestamp handling, and the integration control loop never asserts on verifier telemetry.
- **Nonce ledger lifecycle**: `NonceLedger` purges stale entries only during `register` calls. Long idle windows while the service is not ingesting commands allow the in-memory ledger to grow without bounds, and there is no telemetry capturing ledger size, evictions, or TTL breaches.

## Kasmina WP-K3 Progress (2025-09-29)
- **Telemetry updates**: `_verify_command` now emits CRITICAL `command_rejected` events (priority escalates via seed/global packets) for signature/nonce/timestamp failures, tracks accept/reject counters, and records validation latency. `tests/kasmina/test_seed_manager.py::test_nonce_replay_emits_critical_and_metrics` asserts routing + metrics.
- **Nonce ledger instrumentation**: `NonceLedger` gained max-entry enforcement, eviction accounting, maintenance hooks, and telemetry snapshot APIs. Kasmina publishes `kasmina.nonce_ledger.{size,evictions_total,ttl_seconds}` and warns via `nonce_ledger_truncated` when the cap trims history (`tests/kasmina/test_seed_manager.py::test_nonce_ledger_truncation_emits_warning`).
- **Administrative resets**: `KasminaSeedManager.reset_registry()` and `reset_teacher_model()` clear the registry + nonce ledger and emit telemetry, covering R4c follow-up item RST-K3. Both paths are exercised in new unit tests.

## Kasmina WP-K4 Wrap-up (2025-09-28)
- **Async coordinator**: `KasminaPrefetchCoordinator` runs on the shared `AsyncWorker`, spawning per-task Oona clients while forcing worker clones to disable stale-claim scans (fixes the cross-loop future crash seen in Weatherlight). Shutdown now cancels publisher handles, closes spawned clients, and clears asyncio tasks deterministically.
- **Prefetch metrics & timeouts**: `KasminaSeedManager` maintains per-status counters (`kasmina.prefetch.requests_total{status}`), inflight depth, and latency (avg/last). Stale entries expire at ~2× latency budget with CRITICAL `prefetch_timeout` telemetry, and unit coverage exercises ready/error/timeout/cancel flows.
- **Cache locking**: Kernel attachment continues to use per-blueprint locks; `kasmina.cache.lock_wait_ms` plus `cache_lock_contention` warnings surface contention. Prefetch-ready + resume paths share the locking helper.
- **Reset hygiene**: Registry/teacher resets and seed retirement cancel outstanding requests, decrement inflight counters, and emit `prefetch_canceled` so operators can distinguish administrative cleanup from timeouts.
- **Integration/tests**: `tests/integration/test_kasmina_prefetch_async.py` covers ready flow and shutdown cancellations; unit suites assert metrics/telemetry. Weatherlight no longer crashes when prefetch is active.
- **Benchmark**: `scripts/bench_kasmina_prefetch.py --requests 300 --ready-latency-ms 40 --jitter-ms 8 --concurrency 6` produced 40.1 ms mean / 39.9 ms p50 / 53.4 ms p95 (0 errors). Results and alert thresholds are recorded in the observability runbook.
- **Live Oona check**: Redis-backed coordinator (Weatherlight) runs cleanly with the async worker fix; no cache contention observed and telemetry reports expected latency envelope.
- **Docs**: Observability runbook and changelog now cover async worker enablement, benchmark results, and alert thresholds.
