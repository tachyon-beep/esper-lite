# Testing & Validation Plan (RC1)

## Goals
- Ensure each work package lands with adequate unit/integration/performance coverage.
- Validate telemetry routing and strict dependency enforcement under real workloads.
- Track acceptance gates (tests to run, artifacts to capture) before sign-off.

## Test Matrix (executed vs pending)
| Module | Work Packages | Key Tests | Status |
|--------|---------------|-----------|--------|
| Tolaria | WP-T1..T3 | `pytest tests/tolaria/test_aggregation.py`, `pytest tests/tolaria/test_tolaria_trainer.py::test_tolaria_timeout_metrics_incremented`, `pytest tests/tolaria/test_rollback_cache.py`, `pytest tests/tolaria/test_durability.py`, `pytest tests/tolaria/test_profiler.py`, `pytest tests/integration/test_control_loop.py`, `pytest tests/integration/test_rollback_shared_signal.py` | ✅ executed (see `CHANGELOG_RC1.md:60`, `:108`, `:126`)
| Tolaria | WP-T4 | `pytest tests/tolaria/test_rollback.py`, perf benchmark script | ✅ executed (telemetry helpers validated during WP-T4/WP-100 Phase 5)
| Tamiyo | WP-A1..A4 | `pytest tests/tamiyo/test_service.py`, `pytest tests/tamiyo/test_risk_engine.py`, `pytest tests/tamiyo/test_persistence.py`, `pytest tests/integration/test_weatherlight_tamiyo_emergency.py` | ✅ executed (`CHANGELOG_RC1.md:108`, `Tamiyo` entries)
| Kasmina | WP-K1..K4 | `pytest tests/kasmina`, `pytest tests/integration/test_kasmina_prefetch_async.py`, `pytest tests/integration/test_control_loop.py`, `scripts/bench_kasmina_prefetch.py --requests 300 --ready-latency-ms 40 --jitter-ms 8 --concurrency 6` | ✅ executed (`CHANGELOG_RC1.md:82`)
| Shared | Foundations | `pytest tests/integration/test_async_worker_backpressure.py`, `pytest tests/weatherlight/test_service_priority.py`, dependency guard unit coverage | ✅ executed (telemetry + async harnesses)

## New/Updated Tests Required
- Async worker cancellation & timeout propagation. ✅ Covered by `tests/integration/test_async_worker_backpressure.py:1` and control-loop slice.
- Telemetry routing CRITICAL paths. ✅ `tests/weatherlight/test_service_priority.py:1`, `tests/integration/test_weatherlight_tamiyo_emergency.py:1`.
- Blend telemetry metrics (alpha, sparsity) in Kasmina/Tamiyo. ✅ `tests/kasmina/test_blend_annotations.py:1`, `tests/tamiyo/test_service.py:150`.
- Command verification telemetry (all subsystems). ✅ Kasmina verifier suites and Tamiyo command rejection tests (`tests/kasmina/test_seed_manager.py:1`, `tests/tamiyo/test_service.py:441`).
- Cross-system performance harness (rollback SLA, latency comparisons) — ✅ Executed via `scripts/run_rc1_harness.py` (artifacts under `baselines/perf/wp_cs1_phase3/` and `_gpu/`), now wired into CI (`performance-harness` matrix job running CPU quick checks).
- Prefetch cancellation/regression tests with Oona fakes. ✅ `tests/integration/test_kasmina_prefetch_async.py:1`.
- Tamiyo WP-A3 telemetry routing smoke tests. ✅ Weatherlight emergency harness executed (`tests/integration/test_weatherlight_tamiyo_emergency.py:1`).

## Performance Validation
- Tolaria: graph capture latency benchmarks recorded (`baselines/perf/wp100_phase5_prework/`, pool reuse ~63 ms).
- Tamiyo: inference latency tracked in service tests; p95 <45 ms documented (`tests/tamiyo/test_service.py:150`).
- Kasmina: prefetch throughput benchmark executed (`scripts/bench_kasmina_prefetch.py` run recorded in `CHANGELOG_RC1.md:108`).
- Kasmina germination integration: `PYTHONPATH=. pytest tests/kasmina tests/integration/test_control_loop.py -k seed_states` (seed telemetry & isolation). ✅ 2025-10-02 snapshot `baselines/perf/wp101_germination/`.
- Outstanding: execute WP-CS1 cross-system performance harness before RC1 close.

## Telemetry Verification Checklist
- `tolaria.timeout.*`, `tamiyo.timeout_*`, `kasmina.gate_failure`, `kasmina.command_rejected` events.
- Blend metrics (mode, alpha mean/p95) present.
- Coverage map/types in Tamiyo telemetry.
- Cache metrics (hits/evictions) updated.
- Emergency routing: confirm CRITICAL events appear on emergency stream.

## Automation
- Update CI to run new unit suites and integration tests where feasible.
- Provide manual verification steps for telemetry (e.g., CLI command to fetch recent packets).
- CI runs the harness quick sweep (steady-train, rollback, Tamiyo timeout, Kasmina prefetch) via `scripts/run_rc1_harness.py` on CPU to guard telemetry regressions; extend to GPU pool manually when hardware is allocated.

## Acceptance Gate
A work package can be marked complete only when:
1. Associated tests (unit/integration/perf) pass and are recorded in PR.
2. Telemetry checklist items verified (attach logs/screenshots).
3. `lint_static_analysis.md` updated if complexity changes.
4. Change log entry created with tests run & results.
