# Plane Refresh RC1 Execution — Lint & Static Analysis Notes

This document captures the lint/static-analysis signals that must be addressed during the RC1 remediation pass. Pylint is clean across the board (score 10/10 using `.codacy/tools-configs/pylint.rc`), so the actionable items come primarily from complexity analysis (`radon cc`).

## Tolaria
- `SeedAggregationTracker.combine` — **A (2)** complexity after WP‑T5 Phase 1 helper extraction; aggregation orchestration now delegates to dedicated helper methods.
- `_update_seed_metrics` — **C (15)** complexity; acceptable, monitor if telemetry bookkeeping expands.
- `_EpochRunner._build_seed_metrics` — **A (2)** complexity after WP‑T5 Phase 3. Helper decomposition retired the prior hotspot while preserving telemetry ordering.
- `TolariaTrainer.__init__` — **E (31)** complexity. Post Phase 1 helper extraction; remaining footprint comes from device setup + optional controllers. Track clean-up under lint backlog.
- `TolariaTrainer._handle_epoch_failure` — **C (17)** complexity. Centralised failure handling already meets ≤C target; monitor if telemetry expands.
- `_EpochRunner._invoke_control_loop`, `_EpochRunner.run`, `SeedAggregationTracker` class, `_EpochRunner._accumulate_microbatch`, `_EpochRunner._update_step_metrics` — **C/D (22–26)** complexity; acceptable for now, but keep monitoring after Phase 4 completion.
- `TolariaTrainer._emit_telemetry` — **C (17)** complexity after helper extraction; no further action required unless telemetry expands.
- All other Tolaria helpers now sit at ≤C following WP‑T1–T4 refactors.
- `LRController.build_controller` — **B (8)**; ensure dead code paths removed once strict dependency plan lands.
- **2025-09-30 WP-T5 Baseline** — Pre-refactor snapshot captured before Phase 1: `SeedAggregationTracker.combine` **F (44)** and `_EpochRunner._build_seed_metrics` **F (41)** while other helpers were ≤ C.
- **2025-09-30 WP-T5 Phase 1** — Post-helper extraction snapshot: `_apply_teacher_attribution`, `_accumulate_seed_vectors`, `_resolve_seed_weights`, and `_update_per_layer_metrics` sit at **A/B** grades; `_update_seed_metrics` lands at **C (15)**. `_EpochRunner._build_seed_metrics` remained **F (41)** for Phase 2.
- **2025-09-30 WP-T5 Phase 2.2** — `_handle_epoch_failure` now grades **C (17)** after factoring telemetry helpers (`_evaluate_epoch_failure_inputs` **B (9)**, `_record_epoch_failure` **A (3)**, `_record_rollback_metrics` **A (5)**, `_maybe_trigger_deadline_emergency` **A (3)**); Phase 3 subsequently retired the `_build_seed_metrics` hotspot.
- **2025-09-30 WP-T5 Phase 3.1** — `_build_seed_metrics` decomposed into `_collect_seed_snapshots` **B (6)**, `_build_seed_snapshot` **A (3)**, and per-seed helper methods (all **A/B**), retiring the prior **F** hotspot.

- `TamiyoService._apply_risk_engine` — **A (3)** complexity after Phase 3E. Legacy logic now lives in dedicated evaluators; keep monitoring helper drift as Phase 5 lands (2025-09-27).
- `TamiyoService._evaluate` — **A (1)** complexity after Phase 4.5. Orchestrator now delegates to dedicated helpers for policy prep, blueprint resolution, risk enforcement, and telemetry finalisation (2025-09-27).
- `TamiyoService.__init__`, `_emit_field_report`, `_update_observation_windows`, `_serialize_blueprint_record`, `_synthesise_due_windows`, `publish_history` — **D (21–29)** complexity each; create subcomponents (field-report manager, metadata cache, telemetry router).
- `TamiyoPolicy._build_command` — **C (18)** after strict-ID refactor (2025-09-30); remaining higher-complexity helpers (`select_action`, `_maybe_emit_blend_mode_annotations`) tracked separately.
- `TamiyoGraphBuilder` functions:
  - `_populate_edges` — **F (122)** complexity; rewrite using smaller helpers per relation type.
  - `_build_layer_features`, `_build_seed_features`, `_build_parameter_features` — **E (31–38)**; extract feature builders per entity.
  - `_build_activation_features`, `_build_blueprint_features`, `_build_global_features` — **D (22–27)**; share normalization helpers.
- `TamiyoPolicy.__init__` — **E (31)**; carve out registry setup/config ingestion.
- `TamiyoPolicy.select_action`, `_maybe_emit_blend_mode_annotations`, `validate_state_dict`, `_warmup_compiled_model` — **D (22–28)**; split action selection pipeline (graph build, inference, command assembly) and telemetry.

## Kasmina
- `KasminaSeedManager.handle_command` — **A (4)** complexity after dispatcher refactor (2025-09-28). Legacy branching removed; dispatcher outcome finaliser handles telemetry routing.
- `_BlendManager.apply` — **D (26)** complexity; consider splitting confidence/channel handlers if further changes land.
- `_graft_seed` — **D (22)** complexity; future work could extract blueprint fetch vs gate orchestration but no longer blocks R4c.
- `_resume_seed`, `_build_seed_packet`, `_build_global_packet`, `_attach_kernel`, `_flush_seed_packets` — **C (11–19)**; acceptable given current scope, revisit if additional features land.
- `CommandVerifier.verify` — **B (8)**; telemetry hook work pending WP-K3.
- Prefetch/cache modules remain at **B** complexity; locking improvements tracked under WP-K4.

## General Notes
- Pylint continues to report 10/10 for Tolaria, Tamiyo, and Kasmina with the project configuration; no direct lint errors surfaced.
- Tolaria trainer helpers now sit at ≤C complexity after WP‑T5; continue monitoring `_update_seed_metrics` if telemetry expands.

## Pytype Findings
- Tolaria: `rollback.attempt_two_tier_rollback` references undefined `fut` in exception path (pytype name-error).
- Tamiyo: `graph_builder` and `security` reference `leyline_pb2.SystemStatePacket` / `AdaptationCommand`; pytype reports missing attributes (regen stubs or adjust imports).
- Kasmina: `security.CommandVerifier` references `AdaptationCommand` missing in stubs; align with generated Leyline bindings.
