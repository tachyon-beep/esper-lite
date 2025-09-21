# ADR-003: Enum Canonicalization via Leyline

Status: Accepted
Date: 2025-09-20

## Context

Cross-system enums were diverging across subsystems (e.g., Kasmina lifecycle vs. external state). Leyline is the canonical contract layer and must define all public enums used on the wire (telemetry, events, state, commands). Internal enums are allowed but must be mapped to Leyline at boundaries.

## Decision

- All enums used by any subsystem for lifecycle, control, messaging, observability, or outcomes MUST be the Leyline enums defined in `contracts/leyline/leyline.proto`.
- No parallel/internal alternative enums for lifecycle are allowed. Subsystems must use Leyline enums internally and externally.
- Operational conditions (e.g., degraded/isolated/rollback) are expressed via `TelemetryPacket.system_health` and events, not by introducing new lifecycle stages.
- Enum changes follow additive-only, budget-aware governance (Option B size/latency constraints maintained).

## Pre‑Implementation Validation (Mandatory)

Before rollout or refactors:

1. Validate contract docs (`docs/design/detailed_design/00-leyline.md` and legacy `00-leyline-*.md`) against `leyline.proto` to ensure parity.
2. Confirm required lifecycle semantics are covered by existing `SeedLifecycleStage` members. If not:
   - Propose adding members with justification and budget impact.
3. Verify all subsystems use Leyline enums directly (no internal duplicates). Remove any internal lifecycle enums.
4. Run serialization size/latency checks on affected messages.

## Lifecycle Enforcement Example (Kasmina)

- Use `SeedLifecycleStage` internally for all stage transitions.
- Allowed-next transitions are codified using Leyline values.

## Consequences

- Uniform enum semantics across systems; reduced integration friction.
- Strict separation of lifecycle vs. operational conditions.
- Slight adapter boilerplate in subsystems; offset by clarity and stability.

## Follow‑Up Tasks

- Add CI contract tests to ensure only Leyline enums appear in public messages and code.
- Document the policy in `docs/design/detailed_design/00-leyline.md` and runbooks.
