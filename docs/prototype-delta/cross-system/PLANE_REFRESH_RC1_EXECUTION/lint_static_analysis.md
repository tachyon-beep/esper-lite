# Plane Refresh RC1 Execution — Lint & Static Analysis Notes

This document captures the lint/static-analysis signals that must be addressed during the RC1 remediation pass. Pylint is clean across the board (score 10/10 using `.codacy/tools-configs/pylint.rc`), so the actionable items come primarily from complexity analysis (`radon cc`).

## Tolaria
- `TolariaTrainer._train_single_epoch` — **F (210)** complexity. Needs refactoring (extract helpers for data prep, aggregation, telemetry, Tamiyo/Kasmina orchestration).
- `TolariaTrainer.__init__` — **F (49)** complexity. Split configuration, device setup, and optional controllers into dedicated builders.
- `TolariaTrainer.run` — **E (33)** complexity; separate epoch loop, failure handling, and telemetry flush into utilities.
- `TolariaTrainer._emit_telemetry` — **D (24)** complexity; isolate metric construction into smaller helpers to reduce branching.
- `TolariaTrainer._emit_state`, `_checkpoint`, and rollback helpers — **C (14)** complexity each; consider introducing a checkpoint/telemetry service class.
- `LRController.build_controller` — **B (8)**; ensure dead code paths removed once strict dependency plan lands.

## Tamiyo
- `TamiyoService._apply_risk_engine` — **A (3)** complexity after Phase 3E. Legacy logic now lives in dedicated evaluators; keep monitoring helper drift as Phase 5 lands (2025-09-27).
- `TamiyoService._evaluate` — **A (1)** complexity after Phase 4.5. Orchestrator now delegates to dedicated helpers for policy prep, blueprint resolution, risk enforcement, and telemetry finalisation (2025-09-27).
- `TamiyoService.__init__`, `_emit_field_report`, `_update_observation_windows`, `_serialize_blueprint_record`, `_synthesise_due_windows`, `publish_history` — **D (21–29)** complexity each; create subcomponents (field-report manager, metadata cache, telemetry router).
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
- Pylint reports 10/10 for Tolaria, Tamiyo, and Kasmina with the project configuration; no direct lint errors surfaced.
- Focus on complexity refactors above while addressing the design issues already enumerated in the subsystem review findings. Complexity grades F/E/D should drop to C or better as part of RC1 execution.

## Pytype Findings
- Tolaria: `rollback.attempt_two_tier_rollback` references undefined `fut` in exception path (pytype name-error).
- Tamiyo: `graph_builder` and `security` reference `leyline_pb2.SystemStatePacket` / `AdaptationCommand`; pytype reports missing attributes (regen stubs or adjust imports).
- Kasmina: `security.CommandVerifier` references `AdaptationCommand` missing in stubs; align with generated Leyline bindings.
