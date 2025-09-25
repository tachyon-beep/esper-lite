# Draft PR — Tamiyo Beta Hardening (P1–P9) + CPU Fallback Fix

Summary
- Implements Tamiyo remediation packages P1–P3 and P7–P9 as per TAMIYO_REMEDIATION_BETA.md, plus a targeted bug fix to CPU inference fallback in the policy.
- Focus areas: step timeout alignment, compile warm‑up and fallback telemetry, transactional policy updates, builder vectorization, telemetry completeness, SDPA/CUDA graphs decision record, blend‑mode annotations, and completing the field‑report lifecycle with observation windows and durable retries.

Changes
- P1 — Step Budget Alignment
  - Env‑driven default `TAMIYO_STEP_TIMEOUT_MS=5.0`; constructor override preserved.
  - Files: `src/esper/tamiyo/service.py`, `src/esper/core/config.py`

- P2 — GNN Compile Warm‑Up + Telemetry
  - CUDA‑only warm‑up post `torch.compile`; new metric `tamiyo.gnn.compile_warm_ms`; fallback counter exposed.
  - Files: `src/esper/tamiyo/policy.py`, `src/esper/tamiyo/service.py`

- P3 — PolicyUpdate Security & Rollback
  - Version/freshness checks, transactional validate+load+swap, telemetry on rejection.
  - Files: `src/esper/tamiyo/service.py`

- P4 — Graph Builder Perf Vectorization (targeted)
  - Vectorized attribute construction; `_set_edge` accepts tensors; semantics preserved.
  - Files: `src/esper/tamiyo/graph_builder.py`

- P5 — Telemetry Completeness
  - Health indicators include timeout budgets; event→priority mapping verified.
  - Files: `src/esper/tamiyo/service.py`

- P6 — Docs Sync
  - README, metrics, telemetry, delta matrix, input remediation notes updated.
  - Files: `docs/prototype-delta/tamiyo/*`

- P7 — PyTorch 2.8 Attention/CUDA Graphs Note
  - Decision record on SDPA/PyG constraints under `sdpa_compatibility.md`.

- P8 — Blend Mode Annotations
  - Optional annotations (`blend_mode`, confidence/channel params) behind `TAMIYO_ENABLE_BLEND_MODE_ANN`.
  - Files: `src/esper/tamiyo/policy.py`, `src/esper/core/config.py`

- P9 — Field Report Lifecycle (Observation Windows + Ack/Retry)
  - Observation windows keyed by `command_id` with synthesis after N epochs; durable retry/index sidecars with exponential backoff; summary metrics and events.
  - Per‑step reports now use `observation_window_epochs=1`; synthesised reports carry `N` and `report_id` prefixed with `fr-synth-`.
  - Files: `src/esper/tamiyo/service.py`, `src/esper/tamiyo/persistence.py`
  - Settings: `TAMIYO_FR_OBS_WINDOW_EPOCHS`, `TAMIYO_FR_RETRY_BACKOFF_MS`, `TAMIYO_FR_RETRY_BACKOFF_MULT`

- Bug Fix — CPU Inference Fallback
  - Fixed a path that nulled successful CPU outputs after catching a CUDA path exception, causing spurious PAUSE decisions. Now retains CPU outputs and proceeds.
  - Files: `src/esper/tamiyo/policy.py`

Telemetry & Metrics
- New metrics: `tamiyo.field_reports.{pending_total,published_total,retries_total,dropped_total}`
- New events: `field_report_synthesised` (INFO), `field_report_retry` (WARNING), `field_report_drop` (WARNING)
- Docs updated in `docs/prototype-delta/tamiyo/metrics.md`.

Backward Compatibility
- No Leyline schema changes. All additions are optional telemetry or settings.
- Per‑step reports remain; synthesised reports are additional entries.

Risk & Rollback
- All features behind settings; retry/index are sidecars separate from WAL.
- Rollback by disabling P9 env knobs or reverting isolated codepaths.

Tests
- Unit: new P9 tests added under `tests/tamiyo/test_service_p9.py` (synthesis; retry; drop; restart restore).
- Existing Tamiyo tests verified selectively; the fractional schedule test now passes on CPU due to the fallback fix.

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

Note: This PR accumulates incremental work across these commits; no Leyline schema change beyond emergency signal additions from ace8f42 (orthogonal to Tamiyo). Tamiyo P9 features are extended here with telemetry events and summary metrics, and a CPU fallback fix.
