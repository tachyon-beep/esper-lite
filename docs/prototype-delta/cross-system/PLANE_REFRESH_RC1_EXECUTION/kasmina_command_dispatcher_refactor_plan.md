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
- Introduced `_DISPATCHER_EXPERIMENTAL` feature flag, `_build_command_context`, and a no-op `_dispatch_command` stub. When the flag is enabled, the dispatcher short-circuits `handle_command` without executing legacy logic—tests confirm no exceptions.
- Added `tests/kasmina/test_command_dispatcher.py` to verify context/outcome defaults and stub behaviour behind the feature flag.
- Legacy path remains the default (`_DISPATCHER_EXPERIMENTAL = False`), so Kasmina behaviour and existing test failures are unchanged.

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

## Phase 5 — Orchestrator Finalisation
- **Step 5.1:** Remove the legacy `handle_command` branches in favour of dispatcher orchestration; ensure strict dependency failures throw rather than fallback.
- **Step 5.2:** Integrate telemetry flush (gate failures, blend metrics, degraded inputs) into `KasminaCommandOutcome` finaliser akin to Tamiyo’s `_finalize_evaluation`.
- **Step 5.3:** Update blend/seed fixtures (`tests/fixtures/kasmina_seed_fixture.json`) if command/telemetry output changed.
- **Step 5.4:** Re-run full Kasmina suite + relevant integration tests; capture radon improvement (target `_handle_command` ≤ C).
- **Deliverables:** Clean dispatcher path, tests green, risk R4c mitigated.

## Phase 6 — Documentation & Status Updates
- **Step 6.1:** Update `04_wp_KASMINA.md`, `lint_static_analysis.md`, `CHANGELOG_RC1.md`, and `STATUS_R4B_PHASE3.md` (or new Kasmina status doc) with outcomes.
- **Step 6.2:** Note lessons learned and telemetry adjustments in `KNOWLEDGE_DUMP.md`.
- **Step 6.3:** Ensure risk register reflects R4c progress.
- **Deliverables:** Documentation synced with implementation.

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
