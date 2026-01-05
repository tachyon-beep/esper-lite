# Phase 2: Typed Boundaries (Reduce Dict Surfaces)

**Intent:** Make refactors safer by replacing ad-hoc dict contracts with typed dataclasses at key seams.

## Targets

### Reward plumbing

- Introduce `RewardInputs` dataclass (Simic-internal) to replace the long parameter list when calling reward functions.
- Keep reward function behavior identical; this is a call-site simplification and contract hardening step.

### Action execution

- Introduce `ActionSpec` and `ActionOutcome` dataclasses:
  - `ActionSpec` holds decoded head indices and derived values (target slot, op, alpha targets, etc).
  - `ActionOutcome` holds what happened (success/failure, mutations applied, penalty injected, telemetry-relevant fields).

### Batch summaries

- Replace ad-hoc `dict[str, Any]` merges with a `BatchSummary` dataclass, then serialize to dict only at the boundary where we append to `history`.

## Constraints

- Do not introduce “optional key” patterns to make refactors easier.
- Prefer “required fields” + construction-time correctness over runtime `.get()` fallbacks.

## Decision point: external API surface

Pick one:

1) **Keep current entrypoint signature** and use typed objects internally (recommended).
2) **Switch to config-object-only entrypoint** and update all call sites in one PR (no dual path).

Write the decision into the promoted ready plan for this phase.

## Done means

- Reward calls go through a single typed container (no 20+ argument call sites).
- Action decode/execute uses typed objects and has unit tests for the rules.
- The training loop’s “data types” are obvious at the call sites.

