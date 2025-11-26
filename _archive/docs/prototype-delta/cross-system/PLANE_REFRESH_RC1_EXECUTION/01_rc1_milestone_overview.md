# RC1 Milestone Overview

## Objective
Bring Tolaria, Tamiyo, and Kasmina to the prototype-delta standard by removing masked failures, aligning on PyTorch 2.8 behaviour, tightening lifecycle/telemetry guarantees, and delivering clean complexity metrics across the execution stack.

## Success Criteria
- No subsystem masks missing dependencies (fallback kernels, synthetic pause commands, placeholder IDs).
- Shared async worker & telemetry routing adopted by Tolaria, Tamiyo, and Kasmina.
- All work packages (WP-T1..4, WP-A1..4, WP-K1..4) completed with passing tests and telemetry verification.
- Complexity hot-spots (F/E/D) reduced to ≤ C; `lint_static_analysis.md` updated.
- Performance harness executed; rollback SLA metrics instrumented.

## In-Scope
- Execution-layer code under `src/esper/tolaria`, `src/esper/tamiyo`, `src/esper/kasmina`.
- Shared RC1 foundations: async worker, telemetry routing, strict dependency policy.
- Test suites and telemetry for execution stack.

## Out-of-Scope
- Upstream subsystem changes outside execution plane (Urza, Weatherlight except telemetry routing hooks).
- New feature work beyond prototype-delta scope (e.g. KD training loops, distributed modes).

## Dependencies & Sequencing
1. Shared foundations (strict deps, async worker, telemetry routing, Leyline alignment).
2. Tolaria WP-T1/T2 → Tamiyo WP-A1 → Kasmina WP-K1.
3. Mid-phase: Tolaria WP-T3, Tamiyo WP-A2, Kasmina WP-K2.
4. Telemetry pass: Tolaria WP-T4, Tamiyo WP-A3, Kasmina WP-K3.
5. Persistence/cache: Tamiyo WP-A4, Kasmina WP-K4.
6. Cross-system QA (performance, rollback SLA, lint cleanup).

## Ownership Matrix
| Work Package | Lead | Reviewers |
|--------------|------|-----------|
| Shared Foundations | (tbd) | Leads from Tolaria, Tamiyo, Kasmina |
| Tolaria WP-T1/T2 | Codex | Tolaria & Tamiyo reps |
| Tolaria WP-T3/T4 | (tbd) | Shared foundations lead |
| Tamiyo WP-A1/A2 | (tbd) | Tolaria & Kasmina reps |
| Tamiyo WP-A3/A4 | (tbd) | Shared foundations lead |
| Kasmina WP-K1/K2 | (tbd) | Tolaria & Tamiyo reps |
| Kasmina WP-K3/K4 | Codex | Shared foundations lead |

## Checkpoints
- Kickoff: confirm owners, dependencies, test plan.
- Mid-iteration syncs: weekly status review using `08_status_tracker.md`.
- Telemetry validation checkpoint post WP-A3/WP-K3.
- Final sign-off: all success criteria met; change log updated.

## Dependency Execution Status (2025-10-03)
1. **Shared foundations** — ✅ Complete. Shared async worker, dependency guard, and telemetry routing landed (`src/esper/core/async_runner.py:1`, `src/esper/core/dependency_guard.py:1`, `src/esper/weatherlight/service_runner.py:1148`). Covered by `tests/integration/test_async_worker_backpressure.py:1` and `tests/weatherlight/test_service_priority.py:1`.
2. **Tolaria WP-T1/T2 → Tamiyo WP-A1 → Kasmina WP-K1** — ✅ Complete. Tolaria gradient/timeout fixes live (`src/esper/tolaria/aggregation.py:77`, `src/esper/tolaria/trainer.py:2174`); Tamiyo strict decision path refactor shipped (`src/esper/tamiyo/service.py:1380`, `tests/tamiyo/test_service.py:520`); Kasmina dispatcher/fallback enforcement active (`src/esper/kasmina/seed_manager.py:1`).
3. **Mid-phase (Tolaria WP-T3, Tamiyo WP-A2, Kasmina WP-K2)** — ✅ Complete. Tolaria WP-T3, Tamiyo WP-A2, and Kasmina WP-K2 are all merged with corresponding tests/telemetry captured in status tracker and plan docs.
4. **Telemetry pass (Tolaria WP-T4, Tamiyo WP-A3, Kasmina WP-K3)** — ✅ Complete. All telemetry pass workstreams signed off; Tolaria WP-T4 refactors, Tamiyo WP-A3, and Kasmina WP-K3 updates are recorded in changelog/status tracker.
5. **Persistence/cache (Tamiyo WP-A4, Kasmina WP-K4)** — ✅ Complete. WAL strict validation, backup tooling, and cache/prefetch reliability landed (`src/esper/tamiyo/persistence.py:67`, `scripts/tamiyo_wal_backup.py:1`, `scripts/tamiyo_wal_soak.py:1`, `docs/.../04_wp_KASMINA.md:67`).
6. **Cross-system QA (performance, rollback SLA, lint cleanup)** — ⚠️ In progress. Tolaria/Tamiyo/Kasmina work packages are complete; outstanding item is the RC1 performance harness + cross-system QA validation tracked under the upcoming WP-CS1.

## Progress Snapshot (2025-10-03)
- Shared foundations verified end-to-end (async worker, dependency guard, telemetry routing) with integration harnesses and observability updates.
- Tolaria WP-T1–T5, Tamiyo WP-A1–A4, Kasmina WP-K1–K4 and WP-101 are complete; telemetry, WAL, and germination flows are fully instrumented.
- Tolaria WP-100 Phase 5 delivered graph pool reuse/alert updates; Kasmina WP-101 Phase 4/5 validated soak + rollout.
- Remaining scope centres on the cross-system QA/performance harness (RC1 performance validation).

## Next Focus (RC1 Close-out)
- Launch **WP-CS1 — Cross-System QA & Performance Harness** to execute the RC1 harness (per `performance_harness_plan.md`), capture rollback/SLA metrics, and refresh lint baselines.
- Consolidate harness results into changelog/runbook and prepare the final RC1 close-out report.
- Confirm no outstanding telemetry or dependency gaps across Tolaria/Tamiyo/Kasmina before sign-off.
