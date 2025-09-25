# RC1 Risk Register

| ID | Risk Description | Probability | Impact | Owner | Mitigation | Contingency | Status |
|----|------------------|-------------|--------|-------|------------|-------------|--------|
| R1 | Async worker cancellation bugs cause hangs or crashes | Medium | High | Shared foundations lead | Stress test cancellation, staged rollout behind flag | Revert to threaded executors temporarily | Open |
| R2 | Removing fallbacks (pause, identity kernel) exposes hidden dependency failures | High | High | Module leads | Coordinate rollout, improve diagnostics, update docs | Re-enable fallback flags temporarily with logging | Open |
| R3 | Telemetry routing changes overload emergency stream | Medium | Medium | Weatherlight owner | Load-test routing, throttle fallback, monitor queue | Rollback routing change | Open |
| R4 | Complexity refactors introduce regressions | Medium | Medium | Module leads | Incremental PRs, extensive unit/integration tests | Partial rollback of refactor | Open |
| R5 | Confidence gating needs logits not yet emitted by Tamiyo | Medium | Medium | Tamiyo lead | Implement logits export early (WP-A2 dependency) | Temporarily fallback to conservative gate with telemetry warning | Open |
| R6 | WAL fsync changes degrade persistence performance | Low | Medium | Tamiyo lead | Benchmark writes, batch operations | Provide configuration toggle to disable fsync (dev only) | Open |
| R7 | Cache locking reduces throughput | Medium | Low | Kasmina lead | Benchmark before/after, tune lock granularity | Make cache locking optional per env | Open |
| R8 | Schedule slip due to cross-module dependencies | Medium | Medium | Project lead | Weekly sync, status tracker updates, unblock quickly | Reprioritize low-impact work packages | Open |
| R9 | Telemetry additions increase network/storage costs | Low | Low | Telemetry owner | Monitor telemetry volume, aggregate metrics | Disable noisy metrics | Open |
| R10 | Lint/complexity goals not met within timeframe | Medium | Medium | Module leads | Schedule dedicated refactor time, track in status board | Carryover to next milestone with explicit debt | Open |
