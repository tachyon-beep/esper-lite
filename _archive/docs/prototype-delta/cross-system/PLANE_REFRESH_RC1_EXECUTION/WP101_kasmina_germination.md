# Work Package — Kasmina Germination Integration (WP-101)

## Objective
Enable Kasmina to fully graft germinated seeds into the host model so newly issued Tamiyo `SEED`
commands result in live kernels participating in forward/backward passes without breaking gradient-isolation
guarantees.

## Acceptance Criteria
- `KasminaSeedManager._attach_kernel` instantiates the blueprint module and splices it into the trainer model’s forward path via the configured `KasminaLayer` wrappers.
- Seed parameters register in `SeedParameterRegistry` and participate in optimizer updates without leaking gradients to host parameters (monitored via the isolation hooks).
- `SEED` → `BLEND` → `CULL` lifecycle tests demonstrate that the grafted module contributes to loss/gradients, blends to configured alpha targets, and is removed cleanly.
- Integration test covering Tamiyo→Kasmina command flow verifies seeds execute during Tolaria training (e.g., seeded module’s output changes loss and seed metrics) with no regression to existing telemetry.
- Observability docs updated with germination telemetry expectations and rollback guidance.

## Phase 0 – Baseline & Risk Assessment
- **Step 0.1 – Source audit**
  - Task 0.1.1: Trace seed lifecycle entry points (`seed_manager.py`, `blending.py`, `registry.py`) and note where the host trainer/model references are currently stored (e.g., `_model_ref`, `_layer_factory`).
  - Task 0.1.2: Capture existing gradient isolation/registry invariants (how `SeedParameterRegistry` maps IDs, how `_isolation_monitor` records hooks) so new attachments can assert the same pre/post conditions.
  - Task 0.1.3: Summarise the expected KasminaLayer contract from `docs/design/detailed_design/02-kasmina-unified-design.md` (insertion slots, blend interfaces).
- **Step 0.2 – Test harness review**
  - Task 0.2.1: Catalogue seed-related unit tests (`tests/kasmina/test_seed_manager.py`, `test_blending.py`, Tolaria integration suites) and mark which ones need seed-execution coverage versus telemetry-only assertions.
  - Task 0.2.2: Define quick rollback toggles (environment flag or settings knob) to disable dynamic seeds during development, and document expected baseline telemetry when seeds are disabled.
  - Task 0.2.3: Identify benchmark scenarios (e.g., seed improves loss on synthetic dataset) for validation later in Phase 3.

## Phase 1 – Host Forward Integration *(Completed)*
- **Step 1.1 – Model insertion strategy**
  - Task 1.1.1: Implement helper (`_insert_seed_module`) that wires a newly instantiated seed module into the host model graph according to the unified design (e.g., update module list or inject into composite layer).
  - Task 1.1.2: Update `KasminaSeedManager._attach_kernel` to call the helper, handling both synchronous and prefetch paths, and persist references for later removal.
  - Task 1.1.3: Add deterministic ordering tests ensuring multiple seeds insert in predictable positions and are addressable for culling.
- **Step 1.2 – Gradient & optimizer wiring**
  - Task 1.2.1: Register seed parameters with `SeedParameterRegistry` and extend optimizer parameter groups, mirroring `SeedParameterContext` semantics (add unit tests validating membership).
  - Task 1.2.2: Expand isolation hooks/tests to assert no host parameter receives gradient contributions; surface violations via telemetry and fail fast in tests (e.g., raise on isolation breach).
  - Task 1.2.3: Ensure optimizer state initialisation handles new parameters without leaking state (verify with a phased training test).

## Phase 2 – Blending & Lifecycle Validation *(Completed)*
- **Step 2.1 – Blending execution**
  - Task 2.1.1: Verify `blending.py` paths operate on live seed outputs (introduce targeted unit tests for convex blend, warmup schedules).
  - Task 2.1.2: Add tests ensuring alpha schedules adjust seed contribution while host gradients remain detached.
- **Step 2.2 – Seed lifecycle tests**
  - Task 2.2.1: Extend `tests/kasmina/test_seed_manager.py` to cover SEED→BLEND→CULL on a real model (enable/disable seed and assert loss impact).
  - Task 2.2.2: Add regression test ensuring `CULL` removes optimizer entries and host graph hooks.

## Phase 3 – Integration & Observability *(Completed)*
- **Step 3.1 – Tolaria integration test**
  - Delivered via `tests/integration/test_control_loop_seed_metrics.py`, asserting Kasmina emits `kasmina.seed.alpha`, `seed_stage` events, and zero isolation violations during Tolaria runs.
- **Step 3.2 – Telemetry & docs**
  - Observability runbook/knowledge dump updated with germination telemetry and rollback guidance; perf snapshots stored in `baselines/perf/wp101_germination/`.

## Risks
- Host model architecture must expose explicit insertion points; plan assumes unified design spec slots are available.
- Dynamic parameter registration may interact with optimizer state (need to ensure no stale state remains after culling).
- Run-time graph edits could conflict with PyTorch graph capture/compile modes; gate with existing feature flags and document limitations.

## Rollback Plan
- Feature flag to disable dynamic seeds (revert to placeholder `KasminaLayer` no-op).
- Revert commits touching `_attach_kernel` and blending helpers if isolation violations occur; rely on existing seed lifecycle code (v0) until fixes are ready.

## Phase 4 — Validation *(Completed)*
- Regression suites pass (`pytest tests/kasmina`, Tolaria seed tests, integration seed metrics).
- Rollback validation: baseline vs seed runs show identical loss when seeds disabled, confirming immediate revert.
- Soak run (100 epochs) confirms alpha stabilises at 1.0 with zero isolation violations (`seed_soak_summary.json`).
- Seed vs baseline benchmark captured (`perf_comparison.json`) showing latency overhead due to grafted kernel.
- Next: extend benchmarks to staging hardware and document rollback in Phase 5.

## Phase 5 — Rollout & Monitoring *(Completed)*
- Production checklist in observability runbook (enable/monitor/rollback).
- Status tracker/changelog updated; WP-101 closed.
