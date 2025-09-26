# Strict Dependency Guard Execution Plan (Risk R2)

## Objective
Eliminate synthetic fallbacks (pause commands, identity kernels, placeholder IDs) by rolling out a shared dependency guard and hard failure surfacing across Tolaria, Tamiyo, and Kasmina before the work packages remove the fallbacks outright.

## Scope & Deliverables
1. **Shared Guard Module** (`esper.core.dependency_guard`)
   - Guard APIs for IDs (seed/blueprint/training-run), Urza artifacts, kernel availability, and command completeness.
   - Typed exceptions (`DependencyViolationError`) carrying subsystem + artifact metadata.
   - Optional helper to attach structured telemetry payloads for violations.

2. **Subsystem Integration**
   - **Tolaria**: invoke guard before invoking Tamiyo/Kasmina; emit CRITICAL telemetry on violations; short-circuit epoch loop with conservative mode signal.
   - **Tamiyo**: enforce guard when parsing annotations/metadata; fail fast on missing IDs and Urza lookups; drop synthetic pause command path.
   - **Kasmina**: guard `handle_command`, prefetch registrations, and registry lookups; treat guard failures as gate failures with explicit telemetry.

3. **Weatherlight Preflight & Telemetry**
   - Startup preflight ensuring Urza, Oona, Redis connectivity; fail the supervisor if guard raises.
   - Central telemetry mapping (`dependency_guard.violation`) with reason codes for Nissa dashboards.

4. **Testing & Tooling**
   - Unit tests for guard module (positive/negative cases, telemetry metadata).
   - Subsystem tests covering failure scenarios: missing blueprint ID, removed kernel, invalid training_run_id.
   - Integration tests (`tests/integration/test_control_loop.py`) updated to assert guard raises instead of silent fallbacks.
   - CLI drill (`scripts/run_dependency_guard.py`) to simulate missing artifacts in dev environments.

## Sequencing
1. ✅ Implement guard module + tests (owners: shared foundations lead, week 1).
2. ✅ Add preflight checks in Weatherlight & demo script; update telemetry mappings (week 1).
3. ✅ Integrate guard into Tamiyo strict decision path (WP-A1 dependency) and update tests (week 2).
4. ✅ Integrate guard into Tolaria before WP-T1/T2 removal work (week 2).
5. ✅ Integrate guard into Kasmina command handling (prefetch training-run guard pending) (week 2).
6. Remove synthetic fallbacks & update docs once guard telemetry verified (week 3).

## Telemetry & Observability
- New metric/event: `dependency_guard.violation` (attributes: subsystem, artifact_type, artifact_id, reason).
- Weatherlight emergency routing: CRITICAL priority when guard trips in “live” loop.
- Update Nissa alerts to page on >0 violations per 5 minutes.

## Risks
- Guard false positives blocking live runs → mitigate via staged rollout flag + soak tests.
- Telemetry volume increase → monitor queue depth during rollout; apply rate limit if required.
- Tests relying on synthetic fallbacks → audit suites, update fixtures to supply valid IDs.

## Exit Criteria
- Guard module merged with passing tests and documentation.
- Tolaria/Tamiyo/Kasmina produce actionable telemetry when guard triggers.
- Weatherlight demo fails fast with explicit dependency errors instead of proceeding.
- Synthetic fallbacks removed (tracked under WP-T1/WP-A1/WP-K1) with guard active.
