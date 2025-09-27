# Testing & Validation Plan (RC1)

## Goals
- Ensure each work package lands with adequate unit/integration/performance coverage.
- Validate telemetry routing and strict dependency enforcement under real workloads.
- Track acceptance gates (tests to run, artifacts to capture) before sign-off.

## Test Matrix
| Module | Work Packages | Key Tests |
|--------|---------------|-----------|
| Tolaria | WP-T1..T4 | `tests/tolaria/test_aggregation.py`, `tests/tolaria/test_timeout.py`, `tests/tolaria/test_rollback.py`, `tests/integration/test_control_loop.py`, perf benchmark script |
| Tamiyo | WP-A1..A4 | `tests/tamiyo/test_policy.py`, `tests/tamiyo/test_service.py`, `tests/tamiyo/test_risk_engine.py`, `tests/tamiyo/test_wal.py`, integration control loop, perf policy benchmark |
| Kasmina | WP-K1..K4 | `pytest tests/kasmina` (dispatcher/blend suites), `pytest tests/integration/test_control_loop.py` (round trip, per-seed telemetry) |
| Shared | Foundations | New async worker tests, telemetry routing tests, dependency guard unit tests |

## New/Updated Tests Required
- Async worker cancellation & timeout propagation. ✅ (shared foundations soak harness + integration control loop)
- Telemetry routing: ensure CRITICAL events reach emergency stream; verify via Weatherlight harness.
- Blend telemetry metrics (alpha, sparsity) in Kasmina/Tamiyo. ✅ (Kasmina/Tamiyo suites)
- Command verification telemetry (all subsystems). ◻️ pending WP-K3 / WP-A3 follow-up.
- Rollback SLA measurement tests (500 ms / 12 s) for Tolaria/Kasmina. ◻️ deferred to post-R4c.
- Prefetch cancellation/regression tests with Oona fakes. ✅ covered via Kasmina prefetch worker tests (WP-K4).

## Performance Validation
- Tolaria: step latency benchmark before/after aggregator refactor.
- Tamiyo: inference benchmark (<45 ms p95) with new async worker.
- Kasmina: kernel fetch/prefetch throughput; memory footprint under projection changes.
- Document results in `CHANGELOG_RC1.md`.

## Telemetry Verification Checklist
- `tolaria.timeout.*`, `tamiyo.timeout_*`, `kasmina.gate_failure`, `kasmina.command_rejected` events.
- Blend metrics (mode, alpha mean/p95) present.
- Coverage map/types in Tamiyo telemetry.
- Cache metrics (hits/evictions) updated.
- Emergency routing: confirm CRITICAL events appear on emergency stream.

## Automation
- Update CI to run new unit suites and integration tests where feasible.
- Provide manual verification steps for telemetry (e.g., CLI command to fetch recent packets).

## Acceptance Gate
A work package can be marked complete only when:
1. Associated tests (unit/integration/perf) pass and are recorded in PR.
2. Telemetry checklist items verified (attach logs/screenshots).
3. `lint_static_analysis.md` updated if complexity changes.
4. Change log entry created with tests run & results.
