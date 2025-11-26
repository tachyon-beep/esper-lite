# WP0 Execution Plan – Safety & Strict Failure Enforcement

Objectives:
- Eliminate silent fallbacks across Kasmina, Tamiyo, and Tolaria.
- Enforce strict dependency failure semantics per prototype policy.
- Close known data-loss paths (checkpointing, emergency bypass) before RC2 validation.

## Phase Overview

| Phase | Goal | Key Outcomes |
|-------|------|--------------|
| **Phase 0 – Risk Reduction & Safeguards** | Stabilise verification signals before code changes | Enhanced telemetry, targeted tests, rollback rehearsals |
| **Phase 1 – Kasmina Safety Fixes** | Enforce gating failures and remove fallback masking | Confidence gating on logits, mandatory channel vectors, G3/G4 hard fails |
| **Phase 2 – Tamiyo Strict Failure Enforcement** | Remove synthetic commands and ensure strict dependency propagation | Hard-fail command builder, no silent pause fallbacks, persistence fixes |
| **Phase 3 – Tolaria Checkpoint & Emergency Hardening** | Secure emergency delivery and safe checkpointing | Emergency queue preservation, checkpoint deserialisation with weights-only |
| **Phase 4 – Integration Validation & Sign-off** | Verify cross-subsystem behaviour under failure drills | Combined tests, rollback/emergency drills, metrics review |

## Phase 0 – Risk Reduction & Safeguards (P0 Risk Mitigation)

**Status:** Complete (2024-02-01). Telemetry scaffolding landed, baseline tests captured, and rollback playbooks documented during Phase 0 execution.

**Step 0.1 – Telemetry Instrumentation Prep**
- Task 0.1.1: Enable high-severity logging for existing fallback paths (feature flags or temporary counters).
- Task 0.1.2: Add temporary SLO alerts for fallback usage in Kasmina/Tamiyo (if telemetry pipeline supports it).

**Step 0.2 – Targeted Tests & Drills**
- Task 0.2.1: Capture baseline unit/integration tests covering current fallback behaviours (Kasmina blending, Tamiyo pause command, Tolaria emergency bypass).
- Task 0.2.2: Dry-run fast rollback vs WAL restore using existing artefacts to confirm recovery works before changes.

**Step 0.3 – Rollback & Release Safety**
- Task 0.3.1: Snapshot current branch checkpoints and config for recovery.
- Task 0.3.2: Document a rollback plan (per subsystem) recording toggles/flags to disable new behaviour if needed.

## Phase 1 – Kasmina Safety Fixes

**Status:** In progress. Step 1.1 completed 2024-02-01; subsequent steps track below.

**Step 1.1 – Blend Mode Enforcement**
**Status:** Complete (2024-02-01). Channel vectors now mandatory and Tamiyo logits drive confidence gating.
- Task 1.1.1: Update `blend_with_config` to require channel vectors and feed Tamiyo logits into confidence gate (Kasmina #2/#3).
- Task 1.1.2: Extend unit tests simulating missing `alpha_vec`/insufficient logits to assert hard failures and telemetry signalling.

**Step 1.2 – Gate Hardening**
**Status:** Complete (2024-02-01). G3 now validates Tamiyo-provided mesh requirements against registered host layers and surfaces explicit telemetry, and G4 treats fallback status as a terminal failure with CRITICAL alerts.

- Mesh coverage enforcement: `KasminaSeedManager` ingests `mesh_host_layers` annotations, persists the requirements per seed, and `KasminaGates` raises `mesh_incomplete` with a `mesh_coverage_missing` telemetry event when coverage gaps exist.
- Fallback rejection: `KasminaGates` fails G4 on `performance_status="fallback"` and the gate evaluator emits `fallback_status_rejected` telemetry, ensuring degraded kernels cannot progress.
- Testing: Added `tests/kasmina/test_gates.py` and mesh regression cases in `tests/kasmina/test_seed_manager.py` to cover both failing and successful paths.

**Step 1.3 – Emergency & Command Validation**
**Status:** Complete (2024-02-02). `_GateEvaluator` now terminates seeds immediately when Tamiyo marks a fallback, emitting `tamiyo_fallback_requested` + `gate_failure` telemetry, and the new `CommandAnnotationValidator` enforces training IDs, mesh coverage, and confidence-logit requirements ahead of execution (tests updated in `tests/kasmina/test_seed_manager.py`).
- Task 1.3.1: Treat fallback usage in `_GateEvaluator` as terminal failure (Kasmina #19).
- Task 1.3.2: Reject commands lacking critical annotations via the shared validator (Kasmina #20, Arch #10).

**Step 1.4 – Integration Validation**
**Status:** Complete (2024-02-02). Full Kasmina unit suite is green under the stricter validator, targeted regression tests cover fallback rejection (`tests/kasmina/test_blend_annotations.py`, `tests/kasmina/test_isolation_failfast.py`, `tests/kasmina/test_prewarm.py`, `tests/kasmina/test_lifecycle.py`), and Tolaria+Tamiyo integration now asserts that missing confidence/channel annotations surface as `DependencyViolationError` (`tests/integration/test_tamiyo_kasmina_annotations.py`).
- Task 1.4.1: Run Kasmina unit/integration suites; add targeted tests for fallback rejection.
- Task 1.4.2: Verify Tamiyo → Kasmina flow handles rejected annotations (confidence/channel).

## Phase 2 – Tamiyo Strict Failure Enforcement

**Status:** In progress. Step 2.1 completed 2024-02-01; downstream tasks continue to track.

**Step 2.1 – Command Builder Hard Failures**
- **Status:** Complete (2024-02-01). Command builder now emits dependency violations when IDs are missing, and unit coverage updated accordingly.
- Task 2.1.1: Remove placeholder IDs in `_build_command`; enforce strict command IDs (#13).
- Task 2.1.2: Update tests covering missing IDs to expect raised exceptions.

**Step 2.2 – Synthetic Pause Removal**
**Status:** Complete (2024-02-02). `TamiyoPolicy.select_action` now raises a `DependencyViolationError` when no seed candidates are returned, and policy timeouts bubble up as exceptions with dedicated telemetry (`tests/tamiyo/test_policy_gnn.py::test_policy_raises_when_no_seed_candidates`, `tests/integration/test_tamiyo_kasmina_annotations.py`).
- Task 2.2.1: Replace synthetic pause command when no seed candidates (#14) with explicit failure (raises / telemetry).
- Task 2.2.2: Propagate runtime/compile failures instead of degrading to pause (#15); ensure telemetry records reason.

**Step 2.3 – Persistence & Normalisation Safety**
- Task 2.3.1: Update `_load_from_disk` to skip truncated entries without aborting (#11); maintain load_errors telemetry.
- Task 2.3.2: Require `issued_at` in field reports (#12); enforce validation and unit test coverage.
- Task 2.3.3: Emit telemetry when normaliser flush fails (#5) to pair with Phase 0 instrumentation.

**Step 2.4 – Emergency & Timeout Propagation**
- Task 2.4.1: Replace synthetic timeout commands (#19) with explicit failure propagation through service.
- Task 2.4.2: Ensure timeout exceptions are exposed to Tolaria via client (ties into Phase 3).

**Step 2.5 – Validation**
- Task 2.5.1: Execute Tamiyo unit/integration suite; add tests for missing IDs, no candidates, timeout propagation.
- Task 2.5.2: Manual drill: simulate Tamiyo runtime failure and confirm Tolaria receives explicit failure.

## Phase 3 – Tolaria Checkpoint & Emergency Hardening

**Step 3.1 – Emergency Signal Handling**
- Task 3.1.1: Preserve unsent emergency signals after cap (#24); decide on retry/backoff strategy.
- Task 3.1.2: Introduce telemetry for dropped signals and error conditions (#24).

**Step 3.2 – Checkpoint Safety**
- Task 3.2.1: Replace `torch.save` pickle usage with `weights_only` guard (Kasmina already uses, replicate here) (#25).
- Task 3.2.2: Update rollback loader to enforce `weights_only` and validate CRC before loading (#25).

**Step 3.3 – Emergency Telemetry & Device Handling**
- Task 3.3.1: Ensure profiler only activates with valid directory (#21) – prevents IO exceptions during emergency states.
- Task 3.3.2: Adjust `_compute_loss` to infer device from outputs (#22) to avoid mismatched CPU fallbacks.

**Step 3.4 – Cross-System Dependencies**
- Task 3.4.1: Harden dependency validation (existing `_validate_command_dependencies`) to consume Tamiyo’s strict failures (links to Phase 2).
- Task 3.4.2: Run fast rollback and WAL restore after modifications.

## Phase 4 – Integration Validation & Sign-off

**Step 4.1 – Automated Testing**
- Task 4.1.1: Run full test matrix (unit + integration) across Kasmina, Tamiyo, Tolaria.
- Task 4.1.2: Add regression targets to CI for fallback rejection and checkpoint loading.

**Step 4.2 – Manual Drills**
- Task 4.2.1: Conduct cross-subsystem drills: Tamiyo runtime failure, Kasmina fallback rejection, Tolaria emergency dispatch with capped queue.
- Task 4.2.2: Verify telemetry dashboards reflect new hard failures.

**Step 4.3 – Residual Risk Assessment**
- Task 4.3.1: Review outstanding WP1+ items to ensure no new safety gaps opened.
- Task 4.3.2: Document residual risks and mitigation plan (see section below).

## Residual Risk Assessment

After WP0 completion, residual risk should drop from P0 to P1 for safety categories but remains non-zero:

- **Kasmina**: With fallbacks eliminated, remaining risk stems from async safety (addressed in WP1). Risk level ≈ P1 until kernel cache and prefetch concurrency are hardened.
- **Tamiyo**: Strict failure enforcement means policy errors now surface immediately. Residual risk comes from async cancellation (WP1) and configuration (WP2). Risk level ≈ P1.
- **Tolaria**: Checkpoint safety improves, but emergency handling still depends on async worker (WP1) and telemetry instrumentation (WP3). Risk level ≈ P1.

**Overall residual risk**: P1 (High) – acceptable for continued prototype iteration provided WP1 follows immediately and telemetry monitors remain in place. The system should no longer silently degrade under safety-critical scenarios but still requires concurrency hardening and observability investments.
