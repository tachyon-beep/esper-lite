# Work Package — Kasmina Execution Remediation (WP-K)

## Context

Key findings from `KASMINA_REVIEW_FINDINGS.md`, delta docs, and `lint_static_analysis.md`:

- Fallback kernels mask failures; gate inputs ignore fallback/stage expectations.
- Confidence/channels blending uses activations; lacks telemetry, size limits.
- Prefetch uses placeholder training IDs, blocks on event loop assumptions.
- Command verifier lacks telemetry; nonce ledger growth unmanaged.
- Complexity hotspots: `KasminaSeedManager.handle_command` (F/51), `_graft_seed` (D/22), blend annotations/resume (C), etc.

## Work Streams

| ID | Goal |
|----|------|
| WP-K1 | Gate & fallback enforcement |
| WP-K2 | Blending & isolation upgrades |
| WP-K3 | Command/security/telemetry hardening |
| WP-K4 | Prefetch/cache reliability |

### WP-K1 — Gate & Fallback Enforcement

Tasks:

1. Make fallback kernel assignment a gate failure; emit CRITICAL telemetry per prototype policy.
2. Enforce expected stage in gate inputs; fail when telemetry stage mismatch.
3. Update G2/G4 to react to fallback_used/performance_status.
4. Add tests for gate failures and fallback handling.
5. Land the command dispatcher scaffolding (see `kasmina_command_dispatcher_refactor_plan.md`) so seed handling routes through explicit helpers with strict-failure semantics.
Acceptance:

- Gate tests confirm fallback seeds are culled/embargoed.
- Telemetry shows gate_failure events with reason.
Risks:
- Behaviour change affects Tamiyo/Tolaria; coordinate messaging.

Status:

- ✅ Completed 2025-09-28 — Dispatcher is authoritative, fallbacks now raise `DependencyViolationError`, and CRITICAL `gate_failure` telemetry is emitted. Coverage: `pytest tests/kasmina/test_seed_manager.py`, `pytest tests/kasmina/test_command_dispatcher.py`, `pytest tests/integration/test_control_loop.py::test_kasmina_emits_one_packet_per_seed_per_step`.

### WP-K2 — Blending & Isolation Upgrades

Tasks:

1. Ensure confidence gating receives Tamiyo logits; remove activation-based fallback.
2. Add blend telemetry (mode, alpha mean/p95, sparsity, gate stats).
3. Limit alpha_vec length; fail invalid annotations.
4. Normalise isolation verify: compute cosine similarity, reduce projection memory.
5. Reduce complexity (`handle_command`, `_graft_seed`, `_resume_seed`) to ≤ C via helper classes.
Acceptance:

- Unit tests for blend annotations/logit gating pass.
- Isolation telemetry reflects cosine values; memory footprint stable.
Risks:
- Requires Tamiyo to emit logits; ensure dependency tracked in WP-A2.
- Refactor touches large code paths; incremental PRs recommended.

Status:

- ✅ Completed 2025-09-28 — `_BlendManager` enforces logits/alpha_vec bounds, telemetry includes blend metadata, and isolation monitoring remains intact. Coverage: `pytest tests/kasmina/test_blend_annotations.py`, `pytest tests/kasmina/test_seed_manager.py`.

### WP-K3 — Command/Security/Telemetry

Tasks:

1. Hook command verifier failures to telemetry; emit `command_rejected` at `TelemetryLevel.CRITICAL` (priority CRITICAL) for `{missing_signature, invalid_signature, nonce_replayed, missing_timestamp, stale_command}` and attach reason/command_id attributes.
2. Add periodic nonce cleanup with configurable TTL (default 300 s), hard cap (10 k entries), and expose gauges/counters (`kasmina.nonce_ledger.size`, `kasmina.nonce_ledger.evictions_total`, `kasmina.nonce_ledger.ttl_seconds`).
3. Provide teacher deregistration/reset entry points that clear the nonce ledger and seed registry consistently before accepting new commands.
4. Update telemetry helpers to publish verifier acceptance/rejection counters, nonce ledger truncation warnings, and blend mode/clamping metadata.
Acceptance:

- Telemetry logs `command_rejected` with critical priority for security failures; metrics counters/gauges exported and asserted in tests.
- Registry tests cover teacher swaps and ensure nonce ledger reset hooks run.
Risks:
- Telemetry volume increase; monitor Oona queue.
- Registry reset may affect existing seeds; plan migration.

Status:

- ✅ Completed 2025-09-28 — Command verifier telemetry emits CRITICAL `command_rejected` events with metrics, nonce ledger cleanup/counters are exposed (TTL/evictions), administrative resets clear registry + ledger state, and Kasmina telemetry/tests cover failure/success paths. Prefetch/cache reliability continues under WP-K4.

### WP-K4 — Prefetch & Cache Reliability

Tasks:

1. Require real `training_run_id`; remove `prototype` fallback (already enforced via dependency guard) and thread the ID through telemetry/metrics.
2. Run prefetch publish/consume loops on the shared `AsyncWorker`, track outstanding requests, and cancel on shutdown/deadline expiry. Surface CRITICAL `prefetch_timeout` telemetry with request metadata.
3. Add per-blueprint locking (or equivalent single-writer enforcement) around GPU/cache attachments to eliminate concurrent mutation; emit contention metrics/events when waits exceed thresholds.
4. Emit `kasmina.prefetch.requests_total`, `kasmina.prefetch.inflight`, `kasmina.prefetch.latency_ms`, and extend cache telemetry with lock-wait metrics so operators can observe prefetch health.
Acceptance:

- Prefetch integration tests demonstrate cancellation/timeouts without hanging the worker; coordinator shutdown returns cleanly.
- Cache telemetry reports lock-wait metrics and hit/miss counters without races; unit tests cover contention scenarios.
Risks:
- Async changes must align with Tolaria/Tamiyo worker.
- Additional locks may impact throughput; benchmark.

Status:

- ✅ Completed 2025-09-28 — Async worker spawn path disables stale-claim scans to avoid cross-loop futures, prefetch integration test (`tests/integration/test_kasmina_prefetch_async.py`) verifies ready flow and shutdown cancellation, and the new `scripts/bench_kasmina_prefetch.py` harness records prefetch latency (300 requests → 40.1 ms mean / 53.4 ms p95). Observability runbook + changelog updated with benchmark results.

## Testing

- Unit: `pytest tests/kasmina` (seed manager, blending, dispatcher, prewarm suites), `radon cc -s src/esper/kasmina/seed_manager.py` (complexity snapshot).
- Integration: `pytest tests/integration/test_control_loop.py` (round trip, Kasmina telemetry uniqueness).
- Performance: `scripts/bench_kasmina_prefetch.py --requests 300 --ready-latency-ms 40 --jitter-ms 8 --concurrency 6` → 40.1 ms mean / 53.4 ms p95 (0 errors).

## Rollback Plan

- Guard new fallback behaviour behind config until validated.
- Keep existing prefetch path available during staged rollout.

## Telemetry Verification

- `kasmina.gate_failure`, `kasmina.command_rejected`, blend telemetry, cache metrics verify.
- Emergency telemetry routes via Weatherlight per shared foundations.

## Sign-off

- WP-K1–WP-K4 complete; Kasmina command security and prefetch/cache reliability now meet RC1 acceptance criteria.
- `CHANGELOG_RC1.md` includes WP-K4 async worker/prefetch benchmark summary and observability updates.


### WP-101 — Germination Integration

Tasks:

1. Graft germinated seeds into the host forward path using KasminaLayer hooks.
2. Register seed parameters with optimizers and enforce fail-fast gradient isolation.
3. Validate blending/lifecycle telemetry (`kasmina.seed.*`) and integrate with Tolaria tests.
4. Update observability docs and capture reviewer snapshots.

Status:

- ✅ Phase 1 (Host forward integration) — `_insert_seed_module` now wires kernels into the host model in deterministic order, culls tear down modules/optimizer state, and integration tests cover multi-seed ordering (`tests/kasmina/test_seed_manager.py::test_seed_modules_execute_in_order`).
- ✅ Phase 2 (Blending & lifecycle validation) — Live seeds contribute to outputs with alpha scheduling and detached host grads; fail-fast isolation and optimizer state lifecycle tests are in place.
- ⚙️ Phase 3 (Integration & observability) — Tolaria integration asserts seed metrics, runbook/knowledge dump updates underway, and telemetry snapshots captured (`baselines/perf/wp101_germination/`).
- ☐ Phase 4/5 pending: long-run validation + rollout documentation once performance harness completes.

References: `WP101_kasmina_germination.md`, integration tests `tests/integration/test_control_loop.py`, and observer artifacts under `baselines/perf/wp101_germination/`.

## Next Steps
- Kasmina WP-101 closed. Await WP-CS1 harness results to validate cross-system performance; no additional Kasmina remediation scheduled.
