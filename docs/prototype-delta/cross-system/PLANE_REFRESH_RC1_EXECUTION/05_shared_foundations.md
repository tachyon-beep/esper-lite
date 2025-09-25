# Shared Foundations — Async, Telemetry, Strict Dependencies

## Objectives
- Provide common infrastructure underpinning Tolaria, Tamiyo, and Kasmina during RC1 execution.
- Ensure strict dependency policy (no synthetic fallbacks/IDs) is enforced consistently.
- Standardise async execution and telemetry routing to reduce duplicated logic and failure cases.

## Components
| Component | Description |
|-----------|-------------|
| Async Worker | Shared cancellable task runner used for Tamiyo inference, Kasmina metadata/prefetch, Tolaria service calls. |
| Telemetry Router | Defines priority mapping (INFO/NORMAL vs WARNING/HIGH vs CRITICAL/EMERGENCY) and Weatherlight routing rules. |
| Strict Dependency Guard | Shared helper raising errors for missing IDs, fallback requests, invalid configurations. |
| Shared Config Schema | Defines PT2.8 assumptions, worker settings, fallback disabling. |
| Telemetry Schema | Common fields for timeout/gate/blend/command verifications across subsystems. |

## Async Worker Plan
- Implement `esper.core.async_runner.AsyncWorker` with:
  - Worker pool (size configurable) using `asyncio` + cancellation tokens.
  - Submit API returning cancellable futures with timeout handling.
  - Metrics hooks (tasks scheduled, cancelled, completed, latency).
- Adoption steps:
  1. Integrate worker into Tolaria WP-T2 (Tamiyo/Kasmina calls).
  2. Replace Tamiyo per-call executors (WP-A1) and Kasmina prefetch loops (WP-K4).
  3. Provide synchronous adapter for contexts without running loop.
- Risks: ensure cancellation stops underlying coroutine; guard exceptions.

## Telemetry Router Plan
- Define priority mapping:
  - CRITICAL events → emergency stream.
  - WARNING/ERROR events → high-priority normal stream.
  - INFO → normal stream.
- Weatherlight integration: route based on `TelemetryPacket.system_health.indicators["priority"]`.
- Provide helper for subsystems to set priority consistent with risk engine.
- Verification: telemetry tests after WP-A3/WP-K3.

## Strict Dependency Guard
- Create `esper.core.dependency_guard` with checks:
  - Validate IDs non-empty.
  - Prevent fallback kernel usage unless explicitly flagged.
  - Ensure training run IDs provided for prefetch.
- Usage: Tamiyo command parsing, Kasmina seed ops, Tolaria rollback/profiler.

## Shared Config Schema
- Define dataclass `ExecutionConfig` with sections:
  - Async worker settings.
  - PT2.8 toggles (matmul precision, inference mode enforcement).
  - Dependency policies (fallback allowed bools).
  - Telemetry options.
- Each subsystem consumes `ExecutionConfig` to remove scattered `EsperSettings` lookups.

## Telemetry Schema
- Document common metrics/events: timeout, gate failure, blend telemetry, command rejection.
- Provide helper to attach coverage map/types, command verification results, fallback flags.

## Timeline & Integration
1. Finalise Async Worker & Config schema (before WP-T2).
2. Weatherlight telemetry routing update.
3. Subsystem adoption as per work packages (Tolaria → Tamiyo → Kasmina).
4. Update documentation & tests.

## Risks & Mitigations
- Async worker needs robust cancellation; include stress tests.
- Telemetry routing changes require Weatherlight deploy; coordinate with ops.
- Config schema adoption may require staged rollout; maintain backward compatibility until all modules switched.

## Deliverables
- `esper/core/async_runner.py` + tests.
- `esper/core/dependency_guard.py` + tests.
- Updated Weatherlight supervisor for priority routing.
- Documentation in `README` + subsystem config updates.
