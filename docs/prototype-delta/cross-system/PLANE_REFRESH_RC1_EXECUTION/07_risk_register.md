# RC1 Risk Register

| ID | Risk Description | Probability | Impact | Owner | Mitigation | Contingency | Status |
|----|------------------|-------------|--------|-------|------------|-------------|--------|
| R1 | Async worker cancellation bugs cause hangs or crashes | Medium | High | Shared foundations lead | Stress test cancellation, staged rollout behind flag (soak harness in repo, shared worker shipping) | Revert to threaded executors temporarily | Completed (2025-09-26, soak + integration coverage signed off 2025-09-27) |
| R2 | Removing fallbacks (pause, identity kernel) exposes hidden dependency failures | High | High | Module leads | Execute strict dependency guard plan (`09_strict_dependency_plan.md`), ship telemetry + preflight before removing fallbacks | Re-enable fallback flags temporarily with logging (flagged, time-boxed) | Completed (2025-09-26, guard telemetry verified 2025-09-27) |
| R3 | Telemetry routing changes overload emergency stream | Medium | Medium | Codex (Weatherlight owner) | Load-test routing, throttle fallback, monitor queue | Rollback routing change | Completed (2025-09-27, load harness + counter instrumentation) |
| R4 | Complexity refactors introduce regressions | Medium | Medium | Module leads | Incremental PRs, extensive unit/integration tests (fixture parity guard landed for Tolaria) | Partial rollback of refactor | Completed (Tolaria R4a + Tamiyo R4b + Kasmina R4c signed off 2025-09-28) |
| R5 | Confidence gating needs logits not yet emitted by Tamiyo | Medium | Medium | Tamiyo lead | Implement logits export early (WP-A2 dependency) | Temporarily fallback to conservative gate with telemetry warning | Completed (2025-09-27, Tamiyo annotations + Kasmina enforcement) |
| R6 | WAL fsync changes degrade persistence performance | Low | Medium | Tamiyo lead | Benchmark writes, batch operations | Provide configuration toggle to disable fsync (dev only) | Open |
| R7 | Cache locking reduces throughput | Medium | Low | Kasmina lead | Benchmark before/after, tune lock granularity | Make cache locking optional per env | Open |
| R8 | Schedule slip due to cross-module dependencies | Medium | Medium | Project lead | Weekly sync, status tracker updates, unblock quickly | Reprioritize low-impact work packages | Open |
| R9 | Telemetry additions increase network/storage costs | Low | Low | Telemetry owner | Monitor telemetry volume, aggregate metrics | Disable noisy metrics | Open |
| R10 | Lint/complexity goals not met within timeframe | Medium | Medium | Module leads | Schedule dedicated refactor time, track in status board | Carryover to next milestone with explicit debt | Open |
