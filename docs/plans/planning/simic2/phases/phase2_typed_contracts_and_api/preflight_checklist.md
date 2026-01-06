# Phase 2 Preflight Checklist (Typed Boundaries)

## Objective
Prepare Phase 2 execution by locking scope, risks, guardrails, and validation so the
typed-contract refactor can ship without behavior, telemetry, or performance drift.

## Pre-Phase Activities (must complete before Phase 2 starts)

### Entry gates (readiness)
- [ ] Phase 1 extraction is stable and merged (vectorized_trainer, action_execution, batch_ops).
- [ ] Full test suite passes on main branch (no local-only fixes).
- [ ] Telemetry event counts baseline is captured for a short PPO run (Phase 0 method).
- [ ] Performance baseline captured for a short PPO run (throughput, per-epoch time).

### Contract inventory and planning
- [ ] Enumerate dict surfaces to be replaced and their call sites:
  - action decode/execute inputs and outputs
  - reward plumbing input arguments
  - batch summary accumulation and history serialization
- [ ] Map each dict surface to a proposed typed container (field list + ownership).
- [ ] Decide where new dataclasses live (proposed: `src/esper/simic/training/vectorized_types.py`).
- [ ] Decide external API stance:
  - **Option A (preferred):** keep `train_ppo_vectorized(...)` signature stable.
  - **Option B:** config-only entrypoint; enumerate and update all call sites in one PR.
- [ ] Record the API decision in the phase’s promoted ready plan (no dual paths).

### Risk reduction: correctness
- [ ] Action decode/validation rules are documented and covered by unit tests
      (ADVANCE vs FOSSILIZE, slot validity, mask invariants).
- [ ] Reward inputs are defined once, with a single construction site and tests.
- [ ] BatchSummary fields are explicit, serialized only at history boundary.
- [ ] All new dataclasses use `slots=True` to avoid accidental attribute drift.

### Risk reduction: telemetry
- [ ] Event payload shapes remain unchanged (field names and types).
- [ ] Telemetry emission order preserved (no reordering in hot path).
- [ ] Add or update tests for:
  - event payload keys
  - op/slot head telemetry consistency
  - action success/failure accounting

### Risk reduction: performance and GPU semantics
- [ ] Confirm no new `.cpu()` or `.item()` calls are introduced in the hot path.
- [ ] Ensure typed objects do not capture device tensors across streams incorrectly.
- [ ] Validate that batched D2H transfers remain batched (no per-head sync).
- [ ] Capture a before/after timing budget target (no regression beyond noise).

### Implementation planning artifacts
- [ ] Per-file change list (modules touched, new files added).
- [ ] Call-site update list (train script, configs, tests, helper modules).
- [ ] Test plan (unit tests + small PPO run + lint/mypy).
- [ ] Rollback strategy (if regression found, revert whole PR; no partial shims).

### Acceptance criteria (Phase 2 done means)
- [ ] No dict-based optional-key contracts remain at the Phase 2 seams.
- [ ] Reward plumbing uses a single typed container.
- [ ] Action execution uses typed ActionSpec/ActionOutcome with unit tests.
- [ ] Batch summaries are typed and serialized at the history boundary only.
- [ ] Telemetry baselines match (event counts and payload keys).
- [ ] Throughput within baseline tolerance (no measurable regression).

## Notes
- No compatibility shims. If a contract changes, update all call sites in the same PR.
- Avoid defensive programming: missing fields must fail fast.
