# Kasmina Command Dispatcher Refactor (R4c)

## Goal
Reduce `KasminaSeedManager.handle_command` from F (51) complexity by introducing a structured command dispatcher that enforces strict dependency policy, centralises blend/gate handling, and surfaces telemetry without relying on hidden fallbacks.

## Phase Overview

| Phase | Objective |
|-------|-----------|
| 0 | Capture current behaviour, fixtures, and complexity baselines |
| 1 | Map command flows and design the dispatcher contract |
| 2 | Introduce dispatcher scaffolding (context/result objects, routing stub) |
| 3 | Extract blend/gate helpers with strict-failure semantics |
| 4 | Move command handling into dispatcher routes (SEED/PAUSE/OPTIMIZER/etc.) |
| 5 | Finalise orchestration, update telemetry/tests, and clean up legacy paths |
| 6 | Documentation, lint, and status updates |

Each phase is broken into detailed steps below.

## Phase 0 — Baseline & Safeguards
- **Step 0.1: Execute Baseline Test Suite**
  - Task 0.1.1: Run Kasmina unit suites (`tests/kasmina/test_blend_annotations.py`, `tests/kasmina/test_blending.py`, `tests/kasmina/test_seed_manager.py`, `tests/kasmina/test_lifecycle.py`, `tests/kasmina/test_safety.py`, `tests/kasmina/test_isolation_scope.py`).
  - Task 0.1.2: Run control-loop integration focusing on Kasmina commands (`tests/integration/test_control_loop.py::test_kasmina_emits_one_packet_per_seed_per_step`).
  - Task 0.1.3: Capture command/telemetry expectations from existing fixtures (note any reliance on fallback kernels or placeholder IDs).
- **Step 0.2: Snapshot Complexity Metrics**
  - Task 0.2.1: Collect radon complexity for `KasminaSeedManager.handle_command` and related helpers (`radon cc -s src/esper/kasmina/seed_manager.py`).
  - Task 0.2.2: Record supporting file hotspots (`blending.py`, `gates.py`) for later tracking.
- **Step 0.3: Telemetry & Behaviour Baseline**
  - Task 0.3.1: Review current telemetry emitted during seed graft/blend (using test logs) for fallback usage, gate events, and blend annotations.
  - Task 0.3.2: Summarise findings in `KNOWLEDGE_DUMP.md` to reference during R4c execution.
- **Deliverables:** Baseline test logs, complexity figures, and telemetry notes stored alongside the plan.

### Phase 0 Results (2025-09-27)
- Kasmina unit suites executed. `tests/kasmina/test_seed_manager.py` currently reports 15 failures because `KasminaSeedManager.handle_command` enforces `training_run_id` via the shared dependency guard; other unit modules remain green. This confirms the existing fallback behaviour we plan to remove.
- Control-loop integration (`tests/integration/test_control_loop.py::test_kasmina_emits_one_packet_per_seed_per_step`) fails: Kasmina emits no per-seed telemetry packets when identity fallback kernels are used.
- Complexity snapshot: `handle_command` **F (58)**, `_graft_seed` **D (22)**, `_apply_blend_annotations` **C (20)**, `_resume_seed` **C (19)**. These are the primary refactor targets.
- Telemetry baseline and dependency violations are documented in `KNOWLEDGE_DUMP.md` (“Kasmina R4c Baseline (2025-09-27)”).

## Phase 1 — Flow Mapping & Dispatcher Design
- **Step 1.1:** Trace command paths (`SEED`, `OPTIMIZER`, `PAUSE`, `CIRCUIT_BREAKER`) through `handle_command` → `_ensure_gate` → `_graft_seed` → blend helpers; note side effects and telemetry.
- **Step 1.2:** Decide dispatcher API (e.g. `SeedCommandContext`, `SeedCommandResult`) mirroring the Tamiyo evaluation context/outcome approach.
- **Step 1.3:** Identify strict-failure points: fallback kernels, missing blueprint IDs, stage mismatches, blend vector limits.
- **Step 1.4:** Update `04_wp_KASMINA.md` plan section with design notes and acceptance criteria.
- **Deliverables:** Flow diagram (in notes), dispatcher contract definition, updated WP plan.

### Phase 1 Findings (2025-09-27)
- Command flow recap:
  - `COMMAND_SEED` with `seed_operation` branches into GERMINATE → `_graft_seed`, CULL/CANCEL → `_retire_seed`; blend annotations applied before gate evaluation; identity/fallback kernels injected inside `_graft_seed` on Urza failure.
  - `COMMAND_OPTIMIZER` emits telemetry only; no seed state touches but still uses annotations/adjustments.
  - `COMMAND_PAUSE` toggles `_pause_seed`/`_resume_seed`, relying on telemetry annotations (e.g. `resume=true`).
  - `COMMAND_CIRCUIT_BREAKER` forwards to `_apply_breaker_command` with limited telemetry.
  - Rejected commands currently enqueue degraded-input telemetry but otherwise bail silently.
- Proposed dispatcher contract:
  - `KasminaCommandContext`: captures raw command, inferred seed/blueprint/training_run IDs, Tamiyo annotations, gate expectations, runtime/cache handles, and flags for fallback usage.
  - `KasminaCommandOutcome`: aggregates telemetry events, state mutations (e.g., seeds added/removed), cache actions, and final status (`handled`, `failed`, `noop`).
  - `_dispatch_command(context) -> outcome` routes to concrete handlers (`_handle_seed`, `_handle_optimizer`, `_handle_pause`, `_handle_breaker`, `_handle_emergency`).
- Strict-failure points to enforce in dispatcher/helpers:
  - Missing `training_run_id` (current dependency guard triggers test failures) — future dispatcher should surface actionable telemetry before raising.
  - Fallback kernel injection (`_graft_seed`/`_load_fallback`) — must become gate failure with CRITICAL telemetry.
  - Stage mismatches (`GateInputs.expected_stage` vs `telemetry_stage`) currently ignored — dispatcher should fail fast.
  - Blend vector size/logit availability — switch to logits-only gating with bounded `alpha_vec` length.
  - Resume/pause flow relies on annotation flags; dispatcher should validate presence/values.
- `04_wp_KASMINA.md` updated to reference dispatcher approach and status for WP-K1.

## Phase 2 — Scaffolding
- **Step 2.1:** Introduce dataclasses (`KasminaCommandContext`, `KasminaCommandOutcome`) carrying command metadata, seed state, telemetry events, and blend metrics.
- **Step 2.2:** Create `_dispatch_command` stub that accepts a context and routes to placeholder handlers while retaining legacy code path (feature flag guarded).
- **Step 2.3:** Add focused unit scaffold (`tests/kasmina/test_command_dispatcher.py`) verifying context/outcome defaults and stub routing order.
- **Deliverables:** New scaffolding code with tests green under legacy flag.

### Phase 2 Results (2025-09-27)
- Added `KasminaCommandContext`/`KasminaCommandOutcome` dataclasses and exposed them via `KasminaSeedManager.CommandContext/CommandOutcome` for downstream use.
- Introduced the initial dispatcher scaffolding (`_build_command_context`, `_dispatch_command`) behind an experimental toggle, with unit coverage establishing parity while the legacy path stayed default. The flag was removed during Phase 5 once the dispatcher became authoritative.
- Added `tests/kasmina/test_command_dispatcher.py` to verify context/outcome defaults and stub behaviour.

### Phase 2 Tasks
- **Step 2.1: Introduce Context/Outcome Dataclasses**
  - Task 2.1.1: Add `KasminaCommandContext` dataclass capturing command, Tamiyo annotations, resolved IDs (seed, blueprint, training run), gate expectations, runtime/cache handles, and flags (fallback_used, resume requested).
  - Task 2.1.2: Add `KasminaCommandOutcome` dataclass accumulating telemetry events, state mutations, kernel/cache operations, and a dispatcher status enum.
- **Step 2.2: Dispatcher Stub with Feature Flag**
  - Task 2.2.1: Define `_dispatch_command(context, legacy=True)` that, when `legacy=True`, simply forwards to existing logic; otherwise returns a placeholder outcome. Guard with a module-level flag (e.g., `_DISPATCHER_EXPERIMENTAL`).
  - Task 2.2.2: Add helper to build context from incoming `AdaptationCommand` without changing behaviour.
- **Step 2.3: Unit Scaffolding**
  - Task 2.3.1: Create `tests/kasmina/test_command_dispatcher.py` verifying context defaults (IDs empty, flags false) and that the stub route returns a `KasminaCommandOutcome` without side effects.
  - Task 2.3.2: Ensure the feature flag defaults to legacy behaviour and that toggling it in tests does not break existing flows.
- **Deliverables:** Scaffolding code merged with tests green and legacy path untouched (flag off).

## Phase 3 — Helper Extraction
- **Step 3.1:** Extract blend validation/annotation logic into `_BlendManager` helper enforcing vector bounds, logits requirement, and telemetry emission.
- **Step 3.2:** Extract gate enforcement into `_GateEvaluator` that fails on fallback/stage mismatches and surfaces CRITICAL telemetry per R4c goals.
- **Step 3.3:** Refactor isolation/projection hooks (as needed) to accept the new context without pulling in global state.
- **Step 3.4:** Update unit tests for blend/gate helpers (`tests/kasmina/test_blending.py`, `tests/kasmina/test_gates.py`) to assert strict-failure behaviours.
- **Deliverables:** Standalone helpers with coverage and legacy path still active.

### Phase 3 Results (2025-09-27)
- Implemented `_BlendManager` and `_GateEvaluator` helpers; `KasminaSeedManager` now delegates `_apply_blend_annotations` and `_ensure_gate` through these abstractions. Behaviour remains unchanged (legacy semantics preserved) to keep the system stable ahead of strict-failure enforcement.
- Instantiated helpers in `KasminaSeedManager.__init__` and wired dispatcher scaffolding to reuse emerging context structures.
- Existing Kasmina tests (including the known failing `test_seed_manager` scenarios) continue to run without additional regressions. Strict-failure checks for blend/gate remain TODO for later phases when dispatcher routing flips over.

## Phase 4 — Dispatcher Routing
- **Step 4.1:** Implement concrete route handlers (`_handle_seed`, `_handle_optimizer`, `_handle_pause`, `_handle_breaker`) consuming the helpers from Phase 3.
- **Step 4.2:** Wire `_dispatch_command` to call the new handlers while returning `KasminaCommandOutcome` (legacy code still invoked for parity until Phase 5 cut-over).
- **Step 4.3:** Run targeted tests (`tests/kasmina/test_lifecycle.py`, control-loop integration) under flag-on/off to ensure parity.
- **Deliverables:** Dispatcher routes operational with behaviour parity proven via flag toggles.

### Phase 4 Task Breakdown
- **Step 4.1: Context Population Enhancements**
  - Task 4.1.1: Extend `_build_command_context` to resolve seed/blueprint/training-run IDs, Tamiyo annotations (blend metadata), and stage expectations.
  - Task 4.1.2: Capture legacy state references (existing `SeedContext`, registry, caches) inside the context for handlers to consume.
- **Step 4.2: Implement Route Handlers**
  - Task 4.2.1: `_handle_seed(context)` — orchestrate blend manager, gate evaluator, `_graft_seed`/`_retire_seed`, telemetry aggregation, and fallback detection (still deferring strict failure to Phase 5).
  - Task 4.2.2: `_handle_optimizer(context)`, `_handle_pause(context)`, `_handle_breaker(context)`, `_handle_emergency(context)` — mirror current behaviour with structured outcome events.
  - Task 4.2.3: Allow handlers to return `KasminaCommandOutcome` indicating whether to queue seed/global events or remove seeds after flush.
- **Step 4.3: Dispatcher Integration**
  - Task 4.3.1: Update `_dispatch_command` to route based on `command_type`, invoking handlers and returning their outcomes.
  - Task 4.3.2: Keep legacy code path reachable (flag off) while enabling dispatcher when `_DISPATCHER_EXPERIMENTAL` is true; ensure double execution is avoided.
- **Step 4.4: Telemetry & Queue Wiring**
  - Task 4.4.1: Update dispatcher outcomes to feed `KasminaSeedManager._queue_seed_events/_queue_global_events` without duplicating logic.
  - Task 4.4.2: Preserve existing event attributes/priorities to maintain parity (compare against recorded baseline).
- **Step 4.5: Verification**
  - Task 4.5.1: Run seeded unit tests (`tests/kasmina/test_seed_manager.py`) and integration control-loop with dispatcher flag on/off; confirm outcomes match legacy behaviour (still expecting dependency-guard failures flag-off).
  - Task 4.5.2: Capture radon complexity improvement (goal: `handle_command` drops below F once routing moved) even if legacy remains for now.
- **Deliverables:** Dispatcher routes in place, parity verified under experimental flag, and plan ready for Phase 5 legacy removal/strict enforcement.

### Phase 4 Progress (2025-09-27)
- `_dispatch_command` now routes to dedicated handlers: `_handle_seed_command`, `_handle_optimizer_command`, `_handle_pause_command`, `_handle_breaker_command`, `_handle_emergency_command`, and `_handle_unknown_command`.
- Handlers reuse the new `_BlendManager`, `_GateEvaluator`, and `_append_tamiyo_annotations` helpers to preserve legacy telemetry/metadata behaviour. Outcomes bubble `seed_id`, telemetry events, and `remove_after_flush` signals back to the caller.
- Legacy routing parity was validated under the experimental flag; the flag was subsequently removed in Phase 5 when the dispatcher became the default path.
- Complexity reduction is tracked for Phase 5 once the duplicate logic is eliminated.

## Phase 5 — Orchestrator Finalisation
- **Step 5.1:** Remove the legacy `handle_command` branches in favour of dispatcher orchestration; ensure strict dependency failures throw rather than fallback.
- **Step 5.2:** Integrate telemetry flush (gate failures, blend metrics, degraded inputs) into `KasminaCommandOutcome` finaliser akin to Tamiyo’s `_finalize_evaluation`.
- **Step 5.3:** Update blend/seed fixtures (`tests/fixtures/kasmina_seed_fixture.json`) if command/telemetry output changed.
- **Step 5.4:** Re-run full Kasmina suite + relevant integration tests; capture radon improvement (target `_handle_command` ≤ C).
- **Deliverables:** Clean dispatcher path, tests green, risk R4c mitigated.

### Phase 5 Task Breakdown
- **Step 5.1: Remove Legacy Path / Enable Dispatcher**
  - Task 5.1.1: Drop `_DISPATCHER_EXPERIMENTAL` flag; make dispatcher the default path.
  - Task 5.1.2: Delete redundant legacy logic in `handle_command` once dispatcher is authoritative.
- **Step 5.2: Enforce Strict Failure**
  - Task 5.2.1: Modify `_graft_seed` / `_load_fallback` to fail instead of injecting fallback kernels; ensure `KasminaCommandOutcome` captures gate failure telemetry.
  - Task 5.2.2: Update gate evaluator or handling to treat fallback_used and stage mismatch as CRITICAL events that stop progression.
  - Task 5.2.3: Enforce blend vector length/logits requirements in `_BlendManager` with actionable errors.
  - **Step 5.3: Telemetry Finalisation**
  - Task 5.3.1: Ensure dispatcher outcomes flow through `_queue_seed_events/_queue_global_events` with proper priority and metadata (no missing per-seed packets).
  - Task 5.3.2: Verify degraded-input, seed annotations, and emergency telemetry still recorded via `_append_tamiyo_annotations`.
- **Step 5.4: Update Tests & fixtures**
  - Task 5.4.1: Update `tests/kasmina/test_seed_manager.py` expectations (fallback tests now expect exceptions or CRITICAL events); add coverage for stage/fallback failures.
  - Task 5.4.2: Adjust integration control-loop test to expect per-seed telemetry without fallback path.
  - Task 5.4.3: Regenerate fixtures if telemetry changed (seed export, prewarm, etc.).
- **Step 5.5: Regression & Complexity Validation**
  - Task 5.5.1: Run full Kasmina + relevant integration suites; ensure green.
  - Task 5.5.2: Capture radon complexity improvement (`handle_command` ≤ C) and note results in `lint_static_analysis.md`.
  - Task 5.5.3: Update docs (`04_wp_KASMINA.md`, changelog, status tracker, knowledge dump) with final outcomes.

## Phase 6 — Documentation & Status Updates
- **Step 6.1:** Update `04_wp_KASMINA.md`, `lint_static_analysis.md`, `CHANGELOG_RC1.md`, and `STATUS_R4B_PHASE3.md` (or new Kasmina status doc) with outcomes.
- **Step 6.2:** Note lessons learned and telemetry adjustments in `KNOWLEDGE_DUMP.md`.
- **Step 6.3:** Ensure risk register reflects R4c progress.
- **Deliverables:** Documentation synced with implementation.

### Phase 5 Progress (2025-09-28)
- **Step 5.1 complete:** Dispatcher is the sole orchestration path; the `_DISPATCHER_EXPERIMENTAL` flag and legacy branches were removed, and unit coverage now exercises the default dispatcher.
- **Step 5.2 complete:** Fallback kernels no longer attach—runtime fetch failures now raise `DependencyViolationError`s and trigger CRITICAL gate telemetry. Blend annotations enforce bounded `alpha_vec` values (≤64) with deterministic signing updates, and pause/resume strictness propagates dependency violations when kernels are missing.
- **Step 5.3 complete:** `KasminaCommandOutcome` now flows through a dedicated finaliser that pushes seed/global telemetry via `_queue_*` helpers even when `remove_after_flush` is requested with no new events. Degraded-input and gate-failure events surface with correct priorities, and regression coverage exercises pause/resume, blend validation, and dispatcher outcomes.
- **Step 5.4 complete:** Kasmina unit suites (`pytest tests/kasmina`) and the control-loop integration test (`pytest tests/integration/test_control_loop.py::test_kasmina_emits_one_packet_per_seed_per_step`) now pass after hardening `AsyncWorker.shutdown` to avoid hanging joins. Radon reports `handle_command` at grade A.
- **Step 5.5 complete:** Full Kasmina and integration suites (`pytest tests/kasmina`, `pytest tests/integration/test_control_loop.py`) are green; radon snapshot captured (`radon cc -s src/esper/kasmina/seed_manager.py`) with `handle_command` at grade A. Documentation/status updates queued for Phase 6.

### Phase 6 Progress (2025-09-28)
- Documentation tidied across `04_wp_KASMINA.md`, `CHANGELOG_RC1.md`, `KNOWLEDGE_DUMP.md`, `lint_static_analysis.md`, `08_status_tracker.md`, and `kasmina_command_dispatcher_refactor_plan.md` to reflect completed R4c work. Outstanding items (WP-K3/K4, telemetry/registry follow-ups) explicitly deferred to future milestones.

## Acceptance Criteria
- `KasminaSeedManager.handle_command` refactored to orchestrate dispatcher helpers with complexity ≤ C.
- Fallback kernels trigger gate failures with CRITICAL telemetry; no synthetic identities remain.
- Blend annotations enforce Tamiyo logits/size limits with accompanying telemetry.
- Tests (`tests/kasmina/*`, integration control loop) green; radon snapshot updated.
- Documentation/status trackers reflect R4c completion.

## Risks & Mitigations
- **Telemetry drift:** Use existing fixtures/tests to diff command and telemetry outputs at each phase.
- **Behaviour regressions:** Maintain feature flag until Phase 5, allowing staged verification.
- **Integration dependencies:** Coordinate with Tamiyo WP-A2 for logits availability; ensure dispatcher gracefully errors if logits still absent (with actionable telemetry).
