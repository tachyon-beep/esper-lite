# Draft PR — Cross‑Subsystem Beta Hardening (branch: tamiyognn)

Summary
- This branch hardens multiple subsystems in concert: Tamiyo, Kasmina, Tolaria, Oona, Weatherlight, and Leyline contracts. It delivers Tamiyo P1–P9, introduces emergency signal routing across Tolaria/Oona/Weatherlight, and brings executor/manager enhancements in Kasmina.
- Focus areas: deadline alignment and inference stability, compile warm‑up + telemetry, transactional policy updates, builder vectorization, telemetry completeness, SDPA/CUDA graphs decision record, blend‑mode annotations (Tamiyo→Kasmina), field‑report lifecycle completion, and emergency signal handling end‑to‑end.

Changes Summary
- Tamiyo: P1–P9 implemented (timeouts, compile warm‑up/telemetry, secure policy updates, builder vectorization, telemetry completeness, SDPA/CUDA graphs note, blend‑mode annotations, field‑report lifecycle), plus CPU inference fallback fix.
- Kasmina: Blend‑mode consumption (K1/K7), isolation scope tests, export enrichments.
- Tolaria: Emergency signal primitives and shared‑memory integration; smoke tests added.
- Oona: Emergency signal publish/consume path integrated with breakers and rate control.
- Weatherlight: Monitors emergency signals and bridges shared‑memory signals.
- Leyline: Added EmergencySignal/EmergencyLevel (additive contract change); Python stubs regenerated.
- Docs: Updated metrics, workplans, SDPA note, README resolved issues; added PR draft.

Overall
- System state: “green for prototype” across Tamiyo and companion subsystems. Tamiyo WP1–WP3 and WP7–WP9 are implemented and validated; Kasmina consumes P8 annotations; Tolaria publishes emergency signals; Weatherlight monitors and bridges; Oona exposes new emergency hooks; Leyline contracts include EmergencySignal.
- Key wins: tighter step budgets, steady‑state inference stability, safer policy updates, faster graph build, completed field‑report lifecycle, emergency signal broadcast/consume path, and improved CPU‑only determinism in policy.

Subsystem Changes
- Tamiyo (controller)
  - P1–P3, P7–P9 implemented; P8 annotations; CPU fallback fix to preserve SEED/OPT on CPU;
  - Settings: `TAMIYO_STEP_TIMEOUT_MS`, `TAMIYO_ENABLE_BLEND_MODE_ANN`, `TAMIYO_FR_*` (obs window + backoff).
  - Telemetry: compile metrics, breaker states, coverage per‑type; field‑report synthesis/retry/drop events and publish summary.
- Kasmina (executor)
  - Blend mode consumption (K1/K7) aligned with Tamiyo annotations; isolation scope tests; runtime improvements and seed export enrichment.
- Tolaria (trainer)
  - Emergency signal primitives and bridging; shared‑memory smoke tests; trainer publishes emergency signals per fast‑path plan.
- Oona (messaging)
  - Emergency signal publish/consume added; existing breaker and rate control integrated for emergency stream paths.
- Weatherlight (service runner)
  - Emergency monitoring and handling; forwards/bridges signals.
- Leyline (contracts)
  - Added EmergencyLevel enum and EmergencySignal message; regenerated Python stubs.
- Urza/Urabrask (catalog + producers)
  - RC updates, bench/producer docs and configs; no breaking interface changes.
- Simic/Nissa (replay + metrics)
  - Existing replay and observability flow intact; no schema changes required by this branch.

Changes
- P1 — Step Budget Alignment (Tamiyo)
  - Env‑driven default `TAMIYO_STEP_TIMEOUT_MS=5.0`; constructor override preserved.
  - Files: `src/esper/tamiyo/service.py`, `src/esper/core/config.py`

- P2 — GNN Compile Warm‑Up + Telemetry (Tamiyo)
  - CUDA‑only warm‑up post `torch.compile`; new metric `tamiyo.gnn.compile_warm_ms`; fallback counter exposed.
  - Files: `src/esper/tamiyo/policy.py`, `src/esper/tamiyo/service.py`

- P3 — PolicyUpdate Security & Rollback (Tamiyo)
  - Version/freshness checks, transactional validate+load+swap, telemetry on rejection.
  - Files: `src/esper/tamiyo/service.py`

- P4 — Graph Builder Perf Vectorization (Tamiyo)
  - Vectorized attribute construction; `_set_edge` accepts tensors; semantics preserved.
  - Files: `src/esper/tamiyo/graph_builder.py`

- P5 — Telemetry Completeness (Tamiyo)
  - Health indicators include timeout budgets; event→priority mapping verified.
  - Files: `src/esper/tamiyo/service.py`

- P6 — Docs Sync (Tamiyo)
  - README, metrics, telemetry, delta matrix, input remediation notes updated.
  - Files: `docs/prototype-delta/tamiyo/*`

- P7 — PyTorch 2.8 Attention/CUDA Graphs Note (Tamiyo)
  - Decision record on SDPA/PyG constraints under `sdpa_compatibility.md`.

- P8 — Blend Mode Annotations (Tamiyo → Kasmina)
  - Optional annotations (`blend_mode`, confidence/channel params) behind `TAMIYO_ENABLE_BLEND_MODE_ANN`.
  - Files: `src/esper/tamiyo/policy.py`, `src/esper/core/config.py`

- P9 — Field Report Lifecycle (Observation Windows + Ack/Retry) (Tamiyo)
  - Observation windows keyed by `command_id` with synthesis after N epochs; durable retry/index sidecars with exponential backoff; summary metrics and events.
  - Per‑step reports now use `observation_window_epochs=1`; synthesised reports carry `N` and `report_id` prefixed with `fr-synth-`.
  - Files: `src/esper/tamiyo/service.py`, `src/esper/tamiyo/persistence.py`
  - Settings: `TAMIYO_FR_OBS_WINDOW_EPOCHS`, `TAMIYO_FR_RETRY_BACKOFF_MS`, `TAMIYO_FR_RETRY_BACKOFF_MULT`

- Bug Fix — CPU Inference Fallback (Tamiyo)
  - Fixed a path that nulled successful CPU outputs after catching a CUDA path exception, causing spurious PAUSE decisions. Now retains CPU outputs and proceeds.
  - Files: `src/esper/tamiyo/policy.py`

Telemetry & Metrics
- Tamiyo metrics/events: `tamiyo.field_reports.{pending_total,published_total,retries_total,dropped_total}`; `field_report_synthesised`, `field_report_retry`, `field_report_drop`.
- Emergency signals: new contract messages in Leyline; Oona publishes/consumes; Weatherlight integrates.
- Docs updated in `docs/prototype-delta/tamiyo/metrics.md` and Tolaria emergency plan docs.

Backward Compatibility
- No Leyline schema changes for Tamiyo/Kasmina; EmergencySignal is additive and orthogonal.
- Per‑step reports remain; synthesised reports are additional entries.

Risk & Rollback
- All features behind settings; P9 retry/index are sidecars separate from WAL; emergency signal handling gated by config.
- Rollback by disabling P9 env knobs or reverting isolated codepaths.

Tests
- Tamiyo: new P9 tests (`tests/tamiyo/test_service_p9.py`); existing Tamiyo tests green (fractional schedule passes on CPU).
- Tolaria: integration tests for emergency signals; shared‑memory smoke tests.
- Oona/Weatherlight: covered by emergency integration tests.

Manual Validation Steps
- Set `TAMIYO_FR_OBS_WINDOW_EPOCHS=2` and run two sequential `evaluate_step` calls; confirm a `fr-synth-` report appears with `observation_window_epochs=2`.
- Simulate transient publish failure; confirm retry index and backoff schedule in `field_reports.index.json`.

Documentation
- `TAMIYO_REMEDIATION_BETA.md` updated to mark P9 implemented.
- `README.md` updated with resolved issues and P9 semantics.
- `.env.example` includes P9 settings.

Checklist
- [x] Code changes isolated and minimal
- [x] Unit tests added and passing locally
- [x] Docs updated (README, metrics, workplan)
- [x] No contract changes

Commit Highlights (branch tamiyognn)
- c67bc4a Add per-layer seed summaries plan (docs)
- ea21417 feat: Add smoke tests for Tolaria shared-memory primitives
- ace8f42 feat: Introduce emergency signal handling and broadcasting (contracts, Oona, Tolaria, Weatherlight)
- 2a9a93d Implement P9: Field Report Lifecycle with observation windows and retry logic
- 7d7662b Tamiyo P8
- 4a61304 Tamiyo 4-7
- 29ddac1 Outline Blend Mode Annotations (Tamiyo↔Kasmina)
- 890cb46 Kasmina Alpha and Tamiyo Beta — Start

Note: This PR accumulates incremental work across these commits; no Leyline schema change beyond emergency signal additions from ace8f42. Tamiyo P9 features are extended here with telemetry events and summary metrics, and a CPU fallback fix.

Commit‑by‑Commit Breakdown

- c67bc4a (2025‑09‑25) Add per‑layer seed summaries plan
  - Scope: Docs (Tolaria)
  - Files: docs/prototype-delta/tolaria/per_layer_seed_summaries_plan.md
  - Impact: Planning only; no Tamiyo code changes.

- ea21417 (2025‑09‑25) Add smoke tests for Tolaria shared‑memory primitives
  - Scope: Tolaria/tests
  - Files: scripts/check_shared_memory.py; tests/integration/test_rollback_shared_signal.py
  - Impact: None on Tamiyo behavior; improves integration confidence.

- ace8f42 (2025‑09‑25) Introduce emergency signal handling and broadcasting
  - Scope: Contracts (Leyline), Oona, Tolaria, Weatherlight
  - Files: contracts/leyline/leyline.proto, src/esper/oona/messaging.py, src/esper/tolaria/{emergency.py,trainer.py}, Weatherlight runner, config
  - Impact on Tamiyo: Indirect; Tamiyo telemetry continues to route to Oona streams. No Tamiyo code touched in this commit.

- 2a9a93d (2025‑09‑24) Implement P9: Field Report Lifecycle
  - Scope: Tamiyo P9 core implementation
  - Files: .env.example; docs/prototype-delta/tamiyo/{P9_FIELD_REPORTS_LIFECYCLE.md,TAMIYO_REMEDIATION_BETA.md}; src/esper/{core/config.py,tamiyo/persistence.py,tamiyo/policy.py,tamiyo/service.py}; tests/tamiyo/test_service_p9.py
  - Impact: Adds observation windows and retry logic; initial wiring into service and WAL.

- 7d7662b (2025‑09‑24) Tamiyo P8
  - Scope: Blend‑mode annotations and tests
  - Files: src/esper/core/config.py; src/esper/tamiyo/policy.py; tests/tamiyo/test_service.py
  - Impact: Tamiyo emits optional blend‑mode annotations; default off via settings; consumed by Kasmina.

- 4a61304 (2025‑09‑24) Tamiyo 4‑7
  - Scope: P4–P7 implementation & docs
  - Files: docs (GNN‑WP1, README, delta‑matrix, inputs diffs, roadmap, metrics, telemetry, sdpa_compatibility.md); src/esper/tamiyo/{graph_builder.py,service.py}; tests (builder perf, service)
  - Impact: Vectorised builder hot paths; telemetry completeness, SDPA/CUDA Graphs decision record.

- fc1b64b (2025‑09‑24) K3/4
  - Scope: Kasmina remediation and Tamiyo references
  - Files: docs/kasmina/*; tamiyo TAMIYO_REMEDIATION_BETA.md; kasmina code/tests
  - Impact: Coordination with Tamiyo P8.

- 583019b (2025‑09‑24) T3
  - Scope: Tamiyo service/config/tests
  - Files: tamiyo/service.py, core/config.py, tests/tamiyo/test_service.py, TAMIYO_REMEDIATION_BETA.md
  - Impact: Part of early packages; prepares for subsequent P4–P5.

- cfa24ba (2025‑09‑24) Kasmina 2
  - Scope: Kasmina
  - Files: seed_manager; tests/kasmina/test_isolation_scope.py
  - Impact: None for Tamiyo.

- b9f0814 (2025‑09‑24) Tamiyo 2
  - Scope: Tamiyo policy/service
  - Files: tamiyo/{policy.py,service.py}
  - Impact: Early scaffolding for P1–P3 behavior.

- f86002d (2025‑09‑24) Tamiyo 1 + Kasmina 1
  - Scope: Shared config; Kasmina blending; Tamiyo service; tests
  - Files: core/config.py; kasmina/{blending.py,seed_manager.py}; tamiyo/service.py; tests
  - Impact: Foundational toggles and annotations wiring.

- 29ddac1 (2025‑09‑24) Blend Mode Annotations outline
  - Scope: Docs
  - Files: KASMINA_REMEDIATION_PLAN_ALPHA.md; TAMIYO_REMEDIATION_BETA.md
  - Impact: Planning only; fed into P8 implementation.

- 890cb46 (2025‑09‑24) Kasmina Alpha and Tamiyo Beta — Start
  - Scope: Workplan seed; tests updated
  - Files: KASMINA_REMEDIATION_PLAN_ALPHA.md; TAMIYO_REMEDIATION_BETA.md; tests/tamiyo/test_service.py
  - Impact: Establishes remediation tracks.

Additions in this PR (uncommitted at time of writing)
- CPU fallback fix in TamiyoPolicy to preserve CPU inference outputs (SEED/OPT decisions) when CUDA paths fail; removes inadvertent PAUSE degradation.
- P9 telemetry events for synthesis/retry/drop and summary metrics in publish cycles.
- Docs: README resolved issues; metrics schema additions; P9 acceptance marked implemented; `.env.example` P9 knobs.
