# Telemetry Health Report Coordination

Date: 2026-06-13

## Analysis Plan

- Scope: Esper telemetry from producer code through Leyline contracts, Nissa outputs, Simic/Tamiyo/Kasmina/Tolaria signal generation, Karn stores/MCP/views, Sanctum/Overwatch consumers, proof scripts, and focused tests.
- Deliverable: detailed telemetry health report plus tracker-ready issue map. Do not create Filigree issues automatically.
- Strategy: parallel subagent exploration with disjoint write targets, coordinator synthesis, then independent validation.
- Source policy: source code is read-only for this campaign. Existing modified source files are user-owned current reality; do not revert or edit them.
- Verification: static guardrails, focused telemetry pytest subset, and one tiny telemetry-producing smoke capture.
- Complexity estimate: High. The telemetry network spans training control, policy diagnostics, reward accounting, lifecycle/governor signals, analytics stores, operator UI, and proof tooling.

## Required Findings Shape

Every confirmed issue must include:

- Failure mode: unwired, fake/defaulted, missing field, miswired, mutated/lost, duplicate/stale, or proof-invalidating.
- Severity: P0/P1/P2/P3.
- Evidence: current file and line references or command/test output.
- Real-vs-placeholder assessment.
- Producer -> payload -> backend/store -> consumer path.
- Tracker-ready title and acceptance test.
- Whether it blocks solid signals of life.

## Execution Log

- 2026-06-13 11:38: Created analysis workspace.
- 2026-06-13 11:38: Confirmed current branch is `confounder-drain`; source tree has pre-existing modified Simic files and must be treated as current user-owned state.
- 2026-06-13 11:38: Prepared parallel subagent task briefs.

