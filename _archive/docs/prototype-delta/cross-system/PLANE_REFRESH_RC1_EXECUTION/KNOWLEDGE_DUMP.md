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

## Next Line of Effort — Tamiyo WP-A3 (Telemetry & Routing)

- **Objective**: Close Tamiyo telemetry gaps (coverage metrics, verifier failures, emergency routing) and simplify builder complexity so Weatherlight surfaces the correct priorities. Aligns with the RC1 milestone "Next Focus" items.
- **Dependencies**: Shared async worker already rolled out; Weatherlight consumes Tamiyo telemetry for emergency routing; Kasmina expects richer Tamiyo coverage metadata.
- **Risks Mitigated**: Addresses open telemetry/routing gaps ahead of WP-A4 (persistence). Monitor Oona queue depth to ensure added metrics remain within thresholds.

## Tamiyo WP-A2 Completion (2025-09-30)
- Strict command IDs enabled by default (`TamiyoPolicyConfig.strict_command_ids=True`); `_build_command` no longer substitutes fallback IDs and now defers to dependency guard for enforcement.
- `TamiyoService._validate_command_dependencies` asserts non-empty `target_seed_id`/`blueprint_id`; new unit and integration tests cover rejection paths (`tests/tamiyo/test_service.py::test_seed_command_without_*`, `tests/integration/test_control_loop.py::test_control_loop_handles_*_timeout`).
- Environment flag `TAMIYO_STRICT_COMMAND_IDS` allows temporary opt-out; observability/changelog entries updated accordingly.

### Tamiyo WP-A3 Execution Plan (2025-09-28)

**Phase 0 status (complete)**

- `tests/tamiyo/test_service.py` asserts aggregate metrics (`tamiyo.gnn.feature_coverage`, inference latency) but lacks per-feature coverage ratios or annotation provenance; no verifier failure telemetry surfaced in unit fixtures.
- Weatherlight smoke run (`timeout 20s … esper-weatherlight-service`) starts/stops cleanly; current logs show no Tamiyo CRITICAL events on the emergency stream.
- Tamiyo telemetry currently sets a `priority` indicator after packet build but message priority remains implicit; need explicit routing audit.
- Downstream expectations: Kasmina blend manager consumes `policy_risk_*` annotations; observability runbook still missing Tamiyo coverage thresholds.
- Added WP-A3 test targets to `06_testing_validation_plan.md` (Weatherlight smoke + unit coverage).

**Phase 0 – Baseline & Safeguards**

- *Step 0.1 – Capture Current Behaviour*
  - Task: Review telemetry emitted by `tests/tamiyo/test_service.py` and `tests/integration/test_control_loop.py` to document missing coverage/priorities.
  - Task: Run Weatherlight locally (shared async worker) to observe current routing/log output for Tamiyo packets.
- *Step 0.2 – Dependency & Contract Audit*
  - Task: Inspect Tamiyo → Oona priority mapping and signature context to ensure new telemetry attributes align with Weatherlight expectations.
  - Task: Note downstream consumers (Kasmina blend manager, observability dashboards) that rely on Tamiyo annotations/coverage.
- *Step 0.3 – Acceptance Checklist Stub*
  - Task: Update the scratch section in `06_testing_validation_plan.md` with required unit/integration/Weatherlight tests for WP-A3.

**Phase 1 – Coverage & Annotation Metrics (complete)**

- Metric plan locked: publish aggregate stats (`tamiyo.coverage.feature.avg|min|max`), per-group ratios (`tamiyo.coverage.group.<group>`), feature counts, and `coverage_provenance` annotations; emit `tamiyo.coverage.missing` warning when policy reports no coverage.
- Implementation targets identified (`TamiyoService._collect_policy_metrics`, `_finalize_evaluation`, policy coverage accessors) with test updates scheduled for `tests/tamiyo/test_service.py` and the control-loop fixture to cover both populated and missing coverage cases.

**Phase 2 – Command Verifier Telemetry Wiring (complete)**

- *Step 2.1 – Failure Catalogue (complete)*
  - Failure modes covered via Kasmina verifier telemetry: missing signature, stale timestamp, nonce replay, digest mismatch (validated by Tamiyo-sourced commands).
- *Step 2.2 – Hook Implementation (complete)*
  - Tamiyo tests (`tests/tamiyo/test_service.py::test_tamiyo_missing_signature_rejected`, `::test_tamiyo_signed_command_accepted_and_replay_rejected`, `::test_tamiyo_stale_command_rejected`) assert Kasmina emits CRITICAL `command_rejected` telemetry with the expected reasons, exercising the WP-K3 verifier counters. Weatherlight routing verification continues under Phase 3.

**Phase 3 – Priority & Routing Integration (complete)**

- *Step 3.1 – Packet Priority Audit*
  - `_ensure_priority_indicator` standardises priority indicators for evaluation telemetry, field-report retry/summary packets, and policy-update rejections; unit coverage asserts the indicator exists on evaluation packets.
- *Step 3.2 – Weatherlight Validation*
  - Weatherlight emergency handler now records CRITICAL telemetry sources, exposing `weatherlight.emergency.telemetry_total` and `weatherlight.emergency.tamiyo_total`. `pytest tests/weatherlight/test_service_priority.py` and the integration harness (`pytest tests/integration/test_weatherlight_tamiyo_emergency.py`) confirm Tamiyo low-coverage events reach Oona’s emergency stream without drops and increment Weatherlight counters.
- *Step 3.3 – Back-pressure Assessment*
  - Shared async worker/Oona path is covered by `pytest tests/integration/test_async_worker_backpressure.py`; metrics snapshot shows `publish_total` increments, `publish_dropped == 0.0`, and `queue_depth_max` stays below the configured threshold, satisfying the WP-A3 back-pressure check.

**Phase 4 – Complexity Reduction & Refactor**

- *Step 4.1 – Hotspot Targeting*
  - Task: Profile `_emit_field_report` and `_build_health_indicators`, extract helpers, and drive radon grades toward ≤ C.

**Phase 5 – Validation & Documentation (complete)**

- *Step 5.1 – Test Sweep*
  - Targeted suites executed: `pytest tests/tamiyo/test_risk_engine.py`, `pytest tests/tamiyo/test_service.py`, and `PYTHONPATH=. pytest tests/integration -k "not async_worker_soak"` to confirm cross-system parity after conservative-mode changes.
- *Step 5.2 – Documentation Updates*
  - Observability runbook updated with Weatherlight emergency telemetry metrics; changelog entry added for WP-A3 wrap-up. Status report and risk-engine plan reflect completion.
- *Step 5.3 – Status & Sign-off*
  - WP-A3 marked complete in STATUS_R4B_PHASE3.md; knowledge pack and changelog now reference the emergency routing/back-pressure harnesses.

**Phase 6 – Conservative Mode Regression & Complexity (complete)**

- Targeted runs: `pytest tests/tamiyo/test_risk_engine.py`, `pytest tests/tamiyo/test_service.py`, Weatherlight priority unit, and full integration battery (minus opt-in soak). `radon cc -s src/esper/tamiyo/service.py` keeps `_apply_risk_engine` at `A (3)` / `_evaluate` at `A (1)` with pylint still 10.00.

**Phase 7 – Documentation & Status Updates (complete)**

- Observability runbook, changelog, knowledge dump, and status report now reference the conservative-mode telemetry improvements; plan updated to show Phase 5/6 closure.

### Next Focus — Tamiyo WP-A4 (Persistence & WAL Hardening)

**Phase A4.1 – Baseline Audit & Shadow Checks**
- Inventory current `FieldReportStore` persistence behaviour: WAL writes, fsync usage, retention enforcement, retry index sidecars.
- Capture test coverage gaps around WAL corruption, partial writes, and retention pruning.
- Add a “shadow validator” CLI to parse existing WAL/sidecar files (read-only) so we can detect schema drift or corruption before enforcing new rules.
- Status (2025-09-29): Baseline audit complete — documented append/retention/rewrite flows in `FieldReportStore`, confirmed `_atomic_write_json` handles sidecars. Noted test gaps (no WAL corruption/retention coverage). Shadow validator TODO: design CLI under `scripts/` (`inspect_tamiyo_wal.py`) that loads WAL/sidecars and reports anomalies without mutation; to be implemented in A4.2/A4.4.

**Phase A4.2 – Durability Enhancements**
- Ensure WAL writes are atomic (temp file + rename) and fsync is invoked on both data and directory handles.
- Add checksum or length verification when loading reports to detect truncated entries; emit telemetry on recovery.
- Introduce a feature flag (`TAMIYO_WAL_STRICT_VALIDATION`) to allow dry-run deployment of stricter checks while collecting telemetry.
- Plan:
  - [x] Update `FieldReportStore.append` to write through a temp file + rename and fsync the directory handle; retain existing behaviour behind a flag until validated.
  - [x] Extend `_load_from_disk` to detect truncated/corrupt entries (length mismatch, parse failures) and either skip with telemetry or raise under strict mode.
  - [x] Wire a `TamiyoService` flag (default off) that enables strict validation + recovery telemetry; log/Tamiyo telemetry metrics for skipped/invalid entries.
  - Risk: Low—changes confined to persistence layer with feature-flag fallback; primary risk is write-path regression, mitigated by unit/integration coverage added in A4.4.
 - Status (2025-09-29): Completed. WAL append now fsyncs the parent directory, rewrites follow temp+rename semantics, and `_load_from_disk` records anomalies (raising when `strict_validation` is enabled). TamiyoService honours `TAMIYO_WAL_STRICT_VALIDATION`, tests cover corrupted WAL recovery (`tests/tamiyo/test_persistence.py`) and settings wiring.

**Phase A4.3 – Retry/Sidecar Consistency**
- Harden `_retry_index` and observation window sidecars: guard against partial JSON, add schema validation, and ensure flush is atomic.
- Track retry counts and dropped reports with explicit metrics/telemetry so operators see WAL stress.
- Provide a backup/restore helper script that snapshots `field_reports.log` + sidecars before upgrades and documents the recovery drill.
- Plan:
  - [x] Introduce validation helpers for `_retry_index` / `_windows` sidecars that discard malformed entries, record load errors, and surface them via telemetry.
  - [x] Extend summary telemetry (`publish_history`) with `tamiyo.field_reports.retry_backlog_total`, load-error counters, and warning events when validation drops occur.
  - [x] Emit metrics for retry backlog/drops on each publish sweep to feed observability dashboards.
  - [x] Create `scripts/tamiyo_wal_backup.py` to snapshot/restore WAL + sidecars; document usage in the runbook.
  - [x] Update unit tests (`tests/tamiyo/test_service.py`) to cover invalid sidecar recovery and telemetry metrics; add integration sanity if needed.
  - Risk: Low-moderate — changes touch sidecar load/write paths but remain behind validation harness; backup script is additive.

**Phase A4.4 – Validation & Tooling**
- Expand unit/integration tests to cover WAL corruption, dropped field reports, retention pruning, and strict-mode validation.
- Implement the maintenance script referenced above to inspect/repair WAL/state files and exercise it in tests.
- Run a targeted soak harness that pushes high-volume field reports with induced write failures to validate retry/drop telemetry.
- Plan:
  - [x] Exercise full Tamiyo service suite (targeted subsets executed locally; full suite remains a long-running CI job noted for monitoring).
  - [x] Add tests for the backup script (`scripts/tamiyo_wal_backup.py`) covering backup + restore flows.
  - [x] Update observability runbook with backup/restore instructions and telemetry expectations for strict validation.
  - [x] Automate soak harness via `scripts/tamiyo_wal_soak.py` (validated in tests under `tests/scripts/test_tamiyo_wal_soak.py`). Harness injects truncated WAL entries and verifies summary telemetry/backlog counts.

**Phase A4.5 – Documentation & Sign-off (complete)**
- Observability runbook documents WAL backup/restore workflow, strict-validation telemetry counters, and soak harness usage.
- Status tracker + changelog updated with A4 durability/telemetry work; knowledge dump reflects completed phases.
- Manual long-run soak remains optional; CI exercises the short harness.

## Current Snapshot (2025-10-01)
- Tolaria WP-T4/WP-T5 telemetry and seed-helper refactors complete: `_optimizer_step` routes through `MicrobatchAccumulator`/`SeedAggregationTracker`, `_finalize_epoch` emits `SeedMetricSet`, and `_emit_telemetry` consumes shared snapshots.
- Regression sweeps rerun (`PYTHONPATH=. pytest tests/tolaria`, `PYTHONPATH=. pytest tests/integration/test_control_loop.py tests/integration/test_rollback_shared_signal.py`) remained green; telemetry baselines live at `docs/prototype-delta/cross-system/PLANE_REFRESH_RC1_EXECUTION/baselines/phase5_snapshots/`.
- Radon after WP-T5: `SeedAggregationTracker.combine` **A (2)**, `_EpochRunner._build_seed_metrics` **A (2)**; no remaining Tolaria hotspots above **C** in trainer helpers.
- Observability runbook + plan/status/changelog updated; Tolaria workstream now waiting on cross-system QA and the performance harness package.

## Tolaria WP-T5 Phase 0 Baseline (2025-09-30)
- Telemetry snapshot captured via trainer harness and stored at `docs/prototype-delta/cross-system/PLANE_REFRESH_RC1_EXECUTION/baselines/t5_phase0/seed_epoch_snapshot.json` to preserve pre-refactor seed metrics/events.
- `radon cc -s src/esper/tolaria/trainer.py` baseline: `SeedAggregationTracker.combine` **F (44)**, `_EpochRunner._build_seed_metrics` **F (41)**; all other Tolaria helpers remain ≤ C.
- Regression baseline confirmed clean (`PYTHONPATH=. pytest tests/tolaria/test_tolaria_trainer.py`, `PYTHONPATH=. pytest tests/tolaria/test_aggregation.py`) with one intentional performance skip.
- Downstream telemetry invariants (Kasmina/Tamiyo consumers + dashboards) refreshed from `docs/project/observability_runbook.md`: keep `tolaria.training.*`, `tolaria.timeout.*`, per-seed `seed_conflict_high`, `seed_share_jump`, `epoch_hook_latency_high`, and priority indicators unchanged.
- Rollback guardrails: git checkpoint `pre tolaria 5 guard`; planned edits limited to `src/esper/tolaria/trainer.py`, Tolaria trainer/aggregation tests, lint/status docs, and changelog. If telemetry parity breaks, restore snapshot and revert helper extraction.

## Tolaria WP-T5 Phase 1 Progress (2025-09-30)
- `SeedAggregationTracker.combine` split into `_accumulate_seed_vectors`, `_apply_teacher_attribution`, `_resolve_seed_weights`, `_update_seed_metrics`, and `_update_per_layer_metrics`, keeping orchestration lean.
- Radon now grades `combine` **A (2)**, helper functions **A/B**, and `_update_seed_metrics` **C (15)**; `_EpochRunner._build_seed_metrics` still **F (41)** for upcoming Phase 2 refactor.
- Added targeted tests to `tests/tolaria/test_tolaria_trainer.py` covering weight resolution, teacher attribution telemetry, and per-layer norm accumulation. Tolaria unit suites remain green (23 passed, 1 skipped) alongside aggregation tests.

## Tolaria WP-T5 Phase 2.1 Progress (2025-09-30)
- Introduced `EpochRunResult` so `_run_epoch` returns stats, hook latency, seed metric sets, and step-failure reasons without mutating trainer globals.
- `_profile_epoch` wraps profiler execution around the callable returning `EpochRunResult`; `_finalize_epoch_stats` now stamps duration directly onto the result object.
- `TolariaTrainer.run` consumes the structured result (reducing reliance on `_last_hook_latency_ms`/`_seed_metric_set` side-effects) before invoking `_handle_epoch_failure` and telemetry emission.

## Tolaria WP-T5 Phase 2.2 Progress (2025-09-30)
- `_handle_epoch_failure` delegates to `_evaluate_epoch_failure_inputs`, `_record_epoch_failure`, `_attempt_rollback`, `_record_rollback_metrics`, and `_maybe_trigger_deadline_emergency`, consolidating breaker updates, rollback telemetry, and emergency escalation. Complexity graded **C (17)** with helpers ≤B.
- Deadline misses now increment `tolaria.rollback.deadline_exceeded_total` inside `_record_rollback_metrics`; L4 halts trigger through `_maybe_trigger_deadline_emergency`, updating `tolaria.emergency.halts_total` and emitting `tolaria.emergency.halt` events.
- Added regression tests for success, fast-hit rollback, and deadline escalation paths to `tests/tolaria/test_tolaria_trainer.py`, validating metric counters and emergency behaviour post-refactor.

## Tolaria WP-T5 Phase 3.1 Progress (2025-09-30)
- `_build_seed_metrics` now assembles telemetry via `_collect_seed_snapshots` and `_build_seed_snapshot`, with per-aspect helpers handling share, alpha, conflict, layer norms, and compact events. Complexity dropped from **F (41)** to **A (2)** and `_build_seed_snapshot` sits at **A (3)**.
- Telemetry parity preserved: new helpers reuse `TelemetryMetric`/`TelemetryEvent` construction paths, updating `trainer._last_seed_share` and compact `seed_health` events in one place.
- Updated Tolaria tests continue to pass (`PYTHONPATH=. pytest tests/tolaria/test_tolaria_trainer.py`, `tests/tolaria/test_aggregation.py`), covering the new helper seams.

## Tolaria WP-T5 Phase 3.2 Progress (2025-09-30)
- `_emit_telemetry` consumes the shared `SeedMetricSet`, removing duplicate per-seed assembly while keeping complexity at **C (15)**.
- Seed metrics/events now flow directly from epoch snapshots; legacy `_seed_agg_metrics` cache is no longer required. Integration sweeps (`pytest tests/integration/test_control_loop.py`, `tests/integration/test_rollback_shared_signal.py`) confirm parity.
- Tolaria unit suite stays green post-change with new telemetry tests covering baseline metric supersets, hook budget warnings, and rollback failure events (`tests/tolaria/test_tolaria_trainer.py::test_telemetry_*`).

## Tolaria WP-T5 Phase 4.2 Progress (2025-09-30)
- Added integration coverage for Tamiyo/Kasmina timeouts and rollback deadline escalation (`tests/integration/test_control_loop.py`), asserting telemetry counters, conservative-mode transitions, and emergency dispatch without hangs.
- Emergency L4 simulation uses patched rollback results to verify signal dispatch while keeping tests deterministic.
- Full integration suite re-run (`PYTHONPATH=. pytest tests/integration`) alongside targeted Tolaria unit batches to ensure stability before Phase 5 wrap-up.

## Tolaria WP-T5 Phase 5 Verification (2025-10-01)
- Final Tolaria regression and integration passes (`PYTHONPATH=. pytest tests/tolaria`, `PYTHONPATH=. pytest tests/integration/test_control_loop.py tests/integration/test_rollback_shared_signal.py`) executed cleanly with the extracted helper set.
- Radon confirmation (`radon cc -s src/esper/tolaria/trainer.py | rg "SeedAggregationTracker.combine|_build_seed_metrics"`) shows both helper hotspots at **A (2)**; captured in `lint_static_analysis.md` and referenced in `CHANGELOG_RC1.md`.
- Documentation touchpoints (`02_wp_TOLARIA.md`, `08_status_tracker.md`, `CHANGELOG_RC1.md`, `KNOWLEDGE_DUMP.md`) updated to mark WP-T5 closed and record the verification commands/artifacts.

## Remember After Memory Compact
- Telemetry snapshots for reviewer comparison live in `baselines/phase5_snapshots/`.
- Use `lint_static_analysis.md` + `08_status_tracker.md` as source of truth for outstanding Tolaria actions; next focus is the RC1 performance harness execution.

## Tamiyo WP-A2 Completion (2025-09-30)
- Strict command IDs enabled by default (`TamiyoPolicyConfig.strict_command_ids=True`); `_build_command` no longer substitutes fallback IDs.
- Dependency guard enforces non-empty `target_seed_id` and blueprint IDs (`src/esper/tamiyo/service.py:2408`), with unit/integration coverage in `tests/tamiyo/test_service.py::test_seed_command_without_*` and `tests/integration/test_control_loop.py`.
- Environment flag `TAMIYO_STRICT_COMMAND_IDS` allows temporary opt-out; observability/changelog entries updated accordingly.

## RC1 Performance Harness Plan (2025-09-30)
- Detailed plan recorded in `docs/prototype-delta/cross-system/PLANE_REFRESH_RC1_EXECUTION/performance_harness_plan.md` covering metrics baseline, harness design/implementation, execution, and reporting.
- Scenarios include Tolaria steady-state + rollback deadline, Tamiyo timeout drill, Kasmina prefetch burst; targeted telemetry counters: `tolaria.training.latency_ms`, `tolaria.rollback.restore_latency_ms`, `tamiyo.timeout.inference_ms`, `kasmina.prefetch.latency_ms`.
- Outstanding tasks: implement harness scripts, capture artifacts, and update observability/changelog once performance runs are executed.


## WP-100 Progress (2025-10-01)
- Phase 0 (risk/guardrails): compile warm-up telemetry audit complete; harness baselines enumerated.
- Phase 1 (compile-centric execution): multi-epoch benchmark captured (`wp100_compile_multi_epoch/compile_bench.json`); Tolaria already disables pin-memory when compile is on; integration tests run under compile.
- Phase 2 (GPU prefetch optional): enable_gpu_prefetch flag added with benchmark dirs (`wp100_gpu_prefetch_eager/`, `wp100_gpu_prefetch_compile/`); documentation updated (`--enable-gpu-prefetch`).
- WP-99 Step 1.2 is complete: Tolaria now stages microbatches via the GPU prefetch stream when `enable_gpu_prefetch` is set, timing the async copy with CUDA events and surfacing the value through `h2d_copy_ms`. Updated baselines live under `baselines/perf/wp99_phase1_transfer/post_stage_prefetch_disabled/` and `.../post_stage_prefetch_enabled/`; Tolaria unit/integration suites are green.
- WP-99 Step 3.1 is complete: gradient flattening now reuses pooled 1-D buffers (`GradientBufferPool`) so `_flatten_gradients` writes parameter grads directly without per-parameter clones. Buffers are recycled after `_optimizer_step`, PCGrad semantics unchanged, and telemetry matches prior baselines. New perf artifacts: `baselines/perf/wp99_phase3_flatten/`.
- WP-99 Step 3.2 is complete: `_update_seed_metrics` and `_update_per_layer_metrics` defer work via `SeedMetricsAccumulator`, aggregating seed weights/norms/conflicts and per-layer norms once per epoch before `_build_seed_metrics`. Telemetry parity confirmed via updated Tolaria tests and integrations; new baselines stored in `baselines/perf/wp99_phase3_seed_metrics/`.
- Phase 4 validation complete: benchmark sweeps across default/prefetch/no-compile stored in `baselines/perf/wp99_phase4_validation_*` with a consolidated summary at `baselines/perf/wp99_phase4_validation_summary.json`. Telemetry parity confirmed via integration tests and direct inspection of `SeedMetricSet` output.
- WP-99 complete: Phases 0–5 (performance telemetry, alerts, dashboards) are live; no outstanding tasks.
- Rollback guidance: if the pooled flatten/seed accumulator changes need to be reverted, disable `enable_gpu_prefetch`, toggle `enable_compile` as needed, and restore commit `pre-phase3` (see git tag `pre_phase3_checkpoint`)—metrics fall back to the pre-2025-10-02 baselines under `baselines/perf/wp99_phase3_flatten/`.
- WP-100 Phase 4 complete: CUDA graph capture succeeds with graphs enabled, harness baselines committed;
- Profiling (`scripts/profile_graph_capture.py`) captures `torch.cuda.graph` latency
- DSA run (`capture_profile_dsa.json`) reiterates
- Temporary metrics (`tolaria.graph.capture_ctor_ms`, `…capture_ctx_ms`, `…capture_zero_ms`) show the slow path is entirely inside the `torch.cuda.graph` context (~5.06 s); ctor and zero-grad phases stay <0.1 ms. Instrumentation lives in `_attempt_graph_capture` and will be removed post-optimisation.
 the graph context dominates first-capture time (~5.06 s) for new trainers while per-trainer first capture remains ~0.06 s; no other stages show long stalls.
: first capture on a fresh trainer costs ~5.06 s, whereas reuse of the same trainer (max_epochs>1) avoids the repeated stall; data stored under `baselines/perf/wp100_phase5_prework/capture_profile.json`.
 warm-up pool instantiation still costs ~5 s per epoch. Phase 5 executing — alert thresholds documented, Prometheus rules committed, and Grafana panel added; remaining work covers rollout guidance and warm-up optimisation.
- Loss compatibility: CrossEntropy requires class indices `< out_features`; when exercising graphs ensure datasets supply valid labels (bench harness pins targets to zero). Current optimizer (`SGD`) satisfies CUDA graph requirements (no dynamic parameter allocations). Future capture fixes should retain deterministic shapes and consider replacing loss with a graph-safe variant if new objectives introduce dynamic control flow.

## WP-CS1 Progress
- Phase 0 (metrics/KPI inventory and baseline telemetry) complete on 2025-10-03.
  - KPI-to-metric mapping captured in `performance_harness_plan.md` under Phase 0 Notes.
  - Baseline telemetry snapshots recorded in `baselines/perf/wp_cs1_phase0/` alongside environment README.
- Phase 1 (scenario catalogue + harness layout) complete on 2025-10-03.
  - Scenarios (`steady_train`, `rollback_deadline`, `tamiyo_timeout`, `kasmina_prefetch_burst`) and CLI structure captured in Phase 1 Notes of the plan.
  - Harness ships as `scripts/run_rc1_harness.py` with shared options and canonical JSON/CSV outputs.
- Phase 2 (harness implementation) complete on 2025-10-03.
  - `esper.tools.rc1_harness` implements scenario runners and result serialization; unit tests in `tests/scripts/test_run_rc1_harness.py` cover each path.
  - Dry-run metrics captured under `baselines/perf/wp_cs1_phase2_dryrun/` (CPU) validate JSON/CSV schema.
- Phase 3 (execution & verification) complete on 2025-10-03.
  - Harness runs archived under `baselines/perf/wp_cs1_phase3/`; steady-train latency mean 4.35 ms (p95 6.27 ms), rollback restore latency 12 ms with `deadline_exceeded_total = 1`, Tamiyo timeout drill emitted expected counters, Kasmina prefetch burst latency mean 40.7 ms within target band.
  - No `tolaria.graph_fallback` or masked failure events observed; telemetry packets recorded for every scenario.
  - GPU baselines captured in `baselines/perf/wp_cs1_phase3_gpu/` (steady-train latency mean 41.6 ms/p95 135.8 ms; rollback restore 12 ms; Tamiyo timeout counter 1; Kasmina burst latency mean 20.4 ms/p95 20.7 ms).
- Phase 4 (reporting & integration) complete on 2025-10-03.
  - Observability runbook documents `scripts/run_rc1_harness.py`, expected CPU/GPU envelopes, and Weatherlight/Oona/Nissa alert hooks; changelog entry logged for WP-CS1 execution.
  - CI runs a CPU quick sweep (steady-train, rollback, Tamiyo timeout, Kasmina prefetch) via `scripts/run_rc1_harness.py` with `ESPER_LEYLINE_SECRET` stubbed; GPU baselines live in `baselines/perf/wp_cs1_phase3_gpu/` for manual regression checks.

## Upcoming Work
- WP-101 (Kasmina Germination Integration) commencing: implement live seed grafting per new work package.

- WP-101 Phase 1 & 2 complete: Kasmina now grafts seeds into the host forward path with optimizer wiring, isolation fail-fast, and blending lifecycle tests in place.
- WP-101 Phase 3 complete: integration test `tests/integration/test_control_loop_seed_metrics.py`
- WP-101 Phase 4 underway: seed soak (100 epochs) shows alpha=1.0 with zero isolation violations (`seed_soak_summary.json`); perf comparison (`perf_comparison.json`) captures latency delta seeds vs baseline.
 ensures Tamiyo seeds drive Kasmina telemetry (`kasmina.seed.alpha`, `seed_stage` events) with zero isolation violations. Runbook updated with germination guidance and benchmark snapshots (`baselines/perf/wp101_germination/`). Rollback: set `KASMINA_DYNAMIC_SEEDS=0` or disable fail-fast via `KASMINA_STRICT_ISOLATION=0`.

Phase 5.3 Task 2.2.3 – CUDNN toggles (2025-10-03)
- baseline: benchmark False, deterministic False, allow_tf32 True -> capture ~62 ms
- benchmark True: capture ~5057 ms (still heavy). keep false.
- deterministic True (allow_tf32 False): capture fails, graph fallback.
Hence keep defaults.

- WP-101 Phase 4 validation: soak (`seed_soak_summary.json`), perf comparison (`perf_comparison.json`), regression suites (`pytest tests/kasmina`, Tolaria seed tests, integration seed metrics) all pass; rollback check shows baseline loss matches seed-disabled run.

- WP-101 Phase 5 complete: production rollout guidance documented (runbook), alerts/dashboards monitored via `kasmina.seed.*`, and WP-101 marked complete.
