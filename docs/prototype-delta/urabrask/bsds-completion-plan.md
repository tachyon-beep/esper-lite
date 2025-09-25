# Urabrask BSDS Completion Plan (Prototype Delta)

## Current State Snapshot
- Leyline enums/messages for BSDS and benchmark payloads exist; Python bindings are generated and exercised by Urabrask helpers.
- `compute_bsds` produces canonical protobufs plus BSDS-Lite JSON mirrors, and Weatherlight can run the crucible/bench workers when explicitly enabled.
- Urza persists BSDS mirrors in `extras["bsds"]`, Tamiyo consumes them for gating, and Nissa surfaces Tamiyo-originated hazard telemetry.
- Gaps: blueprint compilation does not attach BSDS automatically, canonical protobufs are not persisted alongside JSON mirrors, and fleet coverage/enablement is opt-in with limited visibility.

## Work Packages

### WP-URA-BSDS-001 — Inline Heuristic Attachment on Compile
- **Objective:** Ensure every new blueprint published by the Tezzeret->Urza pipeline carries at least a heuristic BSDS immediately so Tamiyo decisions honour risk bands without waiting for background workers.
- **Scope:**
  - Extend `BlueprintPipeline` to call `produce_and_attach_bsds` (heuristics) after a successful `UrzaLibrary.save`, guarded by `URABRASK_AUTO_ATTACH_ON_COMPILE`.
  - Update `EsperSettings` so `urabrask_mode` drives defaults: `single` and `continuous` both enable inline attach unless explicitly overridden.
  - Preserve existing extras while merging BSDS; log/telemetry on failures and treat them as defects requiring immediate fix.
  - Update Weatherlight bootstrap so enabling Tamiyo BSDS gating implies auto-attach when mode is `continuous` or `single` with gating enabled.
- **Dependencies:** Urza library available locally; EsperSettings wiring for new `urabrask_mode` enum.
- **Acceptance Criteria:**
  - New blueprint compilations persist `extras["bsds"]` with provenance `HEURISTIC` in Urza immediately.
  - Failures to attach BSDS do not abort the pipeline but emit telemetry `urabrask.inline_attach_failed` with blueprint id context.
  - Configuration docs in `docs/prototype-delta/urabrask/bsds-lite-integration.md` mention `urabrask_mode`, default behaviours, and override options.
- **Validation:**
  - Unit: new tests under `tests/urza/test_blueprint_pipeline_bsds.py` verifying extras merge/flag handling and mode-driven defaults.
  - Integration: extend `tests/integration/test_urza_pipeline.py` (or new) to exercise pipeline with `urabrask_mode` set to `single` and `continuous`.
- **Estimate:** 1.5 engineer-days; Complexity **Medium** (touches pipeline + settings wiring).
- **Rollback Strategy:** revert the change set if inline attachment causes regressions; no alternative code path retained.
- **Risks & Mitigations:**
  - Added latency in compile path (low, heuristics are lightweight); monitor compile duration metric; include timeout guard.
  - Possible recursion with concurrent extras updates; mitigate via get/update/save pattern already in helper.

### WP-URA-BSDS-002 — Canonical Proto Persistence & Tamiyo Preference
- **Objective:** Eliminate JSON-only transport by persisting canonical `leyline_pb2.BSDS` payloads in Urza and teaching Tamiyo to consume them exclusively.
- **Scope:**
  - Extend `UrzaLibrary._build_extras` to store the canonical protobuf payload (`extras["bsds_proto_b64"]`) and drop legacy JSON mirrors.
  - Update `UrzaRecord` to expose decoded protobuf on load; add helper `record.get_bsds_proto()` for consumers.
  - Modify Tamiyo service to consume only canonical proto (with signature verification when `URABRASK_SIGNING_ENABLED`) and remove JSON parsing code.
  - Document the proto-only contract and explicit removal of BSDS-Lite JSON support.
- **Dependencies:** Existing signing/WAL utilities; base64 helpers (Python stdlib).
- **Acceptance Criteria:**
  - Urza round-trips canonical proto without loss; Tamiyo & downstream consumers operate without BSDS-Lite JSON data.
  - Legacy JSON parsing paths are removed; targeted tests confirm absence of JSON keys does not affect decisions.
- **Validation:**
  - Unit: new tests for Urza save/load verifying proto persistence; Tamiyo service tests covering proto ingestion and signature failure handling.
  - Integration: extend Weatherlight/Tamiyo integration test to assert proto ingestion path and signature verification behaviour.
- **Estimate:** 2 engineer-days; Complexity **Medium-High** (touches storage model + critical path consumer).
- **Rollback Strategy:** revert to previous commit set (git rollback); no legacy JSON path retained.
- **Risks & Mitigations:**
  - Proto schema bugs would immediately surface; mitigate via exhaustive tests and smoke validation before release.
  - Larger extras payload; monitor Urza record size, cap via compression if needed (not expected for small proto).

### WP-URA-BSDS-003 — Coverage Telemetry & Operational Enablement
- **Objective:** Provide operators and Tamiyo owners confidence that BSDS coverage is complete and Urabrask workers stay healthy.
- **Scope:**
  - Add Urza metric `urza.bsds.attached_ratio` (URABRASK provenance vs total blueprints) exposed via Weatherlight telemetry.
  - Update Weatherlight bootstrap to enable the Urabrask producer automatically when coverage ratio drops below threshold, with behaviour influenced by `urabrask_mode`.
  - Extend Nissa ingest/dashboard docs with alerts on low coverage and long producer backlogs; surface metrics for inline failures, crucible failures, signing failures.
  - Add runbook section in `docs/prototype-delta/urabrask/README.md` covering how to interpret coverage telemetry and intervene.
- **Dependencies:** Weatherlight telemetry aggregator, Nissa dashboards.
- **Acceptance Criteria:**
  - Prometheus metrics exist for coverage, inline failure counts, producer lag.
  - Alert rules fire when coverage < configured threshold for 10 minutes and respect maintenance windows.
  - Documentation updated with operational guidance and sample Grafana queries.
- **Validation:**
  - Unit: tests for new metric calculations and coverage thresholds.
  - Integration: extend Weatherlight telemetry integration test to include coverage metric; add Nissa alert unit tests.
- **Estimate:** 1 engineer-day; Complexity **Medium**.
- **Rollback Strategy:** metrics/alerts are additive; disable by removing new configuration if noisy.
- **Risks & Mitigations:**
  - Metric accuracy relies on Urza listing; ensure cache coherency by calling fresh list in metric calculation.
  - Alert fatigue; tune threshold/duration during rollout with dry-run mode.

### WP-URA-BSDS-004 — Mode-Aware Continuous T&E Orchestrator
- **Objective:** Provide an always-on Urabrask mode that continuously re-evaluates blueprints with extended hazard batteries, complementing the single-shot path.
- **Scope:**
  - Introduce `EsperSettings.urabrask_mode` (enum: `single`, `continuous`; default `single`). Wire into Weatherlight to select between existing periodic producer and new continuous runner.
  - Implement `UrabraskContinuousTester` (or extend producer) with scheduling policies (round-robin with prioritisation for stale/changed blueprints), concurrency limit, time budget per blueprint, and adaptive backoff.
  - Persist per-run metadata (issued_at, hazards exercised, duration, outcome) in crucible artifacts and optional SQLite table; emit telemetry (`urabrask.continuous.runs_total`, `fail_total`, `queue_depth`).
  - Honour `URABRASK_AUTO_ATTACH_ON_COMPILE` for first-shot BSDS, then reissue BSDS updates (provenance `URABRASK`) when continuous runs detect changes; ensure WAL/signature chain updates accordingly.
  - Provide CLI/admin controls to pause/resume continuous mode and to trigger immediate evaluation for a given blueprint.
- **Dependencies:** Crucible harness, WAL/signing, telemetry infrastructure.
- **Acceptance Criteria:**
  - Setting `urabrask_mode=continuous` results in continuous evaluation without manual intervention; metrics reflect activity; blueprints receive updated BSDS when hazards change.
  - Mode switching at runtime (single->continuous and back) is safe and does not orphan workers; documented procedures exist.
  - Continuous runs respect resource caps (configurable concurrency/timeouts) and include regression tests for scheduling and throttling.
- **Validation:**
  - Unit: scheduler policy tests (staleness ordering, backoff), WAL/signature append tests for multiple runs, configuration tests covering all enum values.
  - Integration: Weatherlight end-to-end test toggling modes and verifying telemetry/BSDS updates; soak test script to simulate long-running mode with synthetic blueprints.
- **Estimate:** 3 engineer-days; Complexity **High** (new worker + scheduling + telemetry).
- **Rollback Strategy:** set `urabrask_mode=single` to fall back to default behaviour; continuous worker stops cleanly.
- **Risks & Mitigations:**
  - Resource exhaustion from continuous tests; mitigate with strict concurrency/timeouts and runtime guard rails.
  - Noise from aggressive hazard probes; provide tiered hazard profiles and default to conservative suite.
  - Operator error when switching modes; document procedures and provide CLI status command.

## Subsystem Impact Assessment

### Tamiyo (Acting Owner Review)
- `src/esper/tamiyo/service.py` already ingests BSDS mirrors and enforces hazard bands (lines ~1365-1445). Mode changes will increase BSDS issuance frequency; logic is idempotent and will simply receive updated mirrors/protos. No command branching depends on run cadence, so continuous mode is safe.
- Planned proto-only persistence (WP-URA-BSDS-002) aligns with existing optional signature verification (`EsperSettings.urabrask_signing_enabled`); we will extend settings to surface `urabrask_mode` without altering Tamiyo’s synchronous decision loop.
- Requirement: remove BSDS JSON parsing paths and ensure settings schema recognises `urabrask_mode` so inline attachment defaults remain consistent when BSDS gating is enabled.

### Nissa (Acting Owner Review)
- `src/esper/nissa/observability.py` derives BSDS hazard counters from telemetry events (lines ~150-230). Continuous mode will increase event volume; counters and gauges are cumulative and thread-safe under Prometheus client usage.
- New coverage metrics/alerts (WP-URA-BSDS-003) integrate naturally with existing gauge/counter patterns. Ensure alert dry-run (RRP-URA-02) to tune thresholds and avoid false positives.
- Requirement: document new metrics in `docs/prototype-delta/urabrask/metrics.md` and extend Nissa config to consume optional coverage thresholds; no schema-breaking changes anticipated.

## Go/No-Go Recommendation
- **Decision:** GO, provided risk reduction packages RRP-URA-01..04 are executed before enabling `urabrask_mode=continuous` in shared environments.
- **Rationale:** Tamiyo and Nissa paths are tolerant of higher BSDS update frequency and proto-first storage. Inline attachment plus continuous T&E improves safety posture, and mitigation steps address resource and telemetry risks.

## Implementation Risk & Risk Reduction
- **Overall Risk Rating:** Medium-High. Inline attach and proto persistence are moderate changes, but continuous mode adds scheduling complexity and sustained resource usage.
- **Primary Risk Drivers:**
  1. Continuous mode could overload limited environments (CI, staging) if throttling is misconfigured.
  2. Proto persistence touches critical storage/consumer paths; regression could break Tamiyo gating.
  3. Increased telemetry/alerts may overwhelm operators unless tuned carefully.

## Risk Reduction Packages
- **RRP-URA-01 — Staging Load Test Harness:** Before enabling continuous mode in production, run the continuous tester against a staging Urza store with representative blueprints, capturing CPU/memory usage and crucible duration distributions.
- **RRP-URA-02 — Telemetry Dry-Run & Alert Tuning:** Deploy coverage and continuous-mode metrics/alerts in dry-run mode (alert manager silence) to calibrate thresholds and ensure signal quality.
- **RRP-URA-03 — Proto Persistence Smoke Validation:** Enable `URABRASK_STORE_PROTO_ENABLED` during pre-production smoke runs, validate Tamiyo/Tolaria integration against a representative blueprint set, and roll back the change immediately on failure (no legacy JSON path available).
- **RRP-URA-04 — Operator Runbook & Training:** Deliver updated runbook with mode-switch procedures, CLI usage, and troubleshooting steps; host knowledge transfer session for Tamiyo/Nissa owners.

## Validation Strategy
- Environment: pre-production; no canary window. Ship changes behind configuration, run smoke suites, and revert to `urabrask_mode=single` on failure.
- Expand automated tests (unit + integration) per work package.
- Run focused `pytest tests/urabrask`, `pytest tests/tamiyo/test_service_bsds.py`, and Weatherlight integration suites before flag flips.
- Stage rollout: enable inline attachment/proto persistence in staging, soak continuous mode with throttled settings, then progressively increase coverage thresholds.

## Residual Risk
- **Rating:** Medium after WP-URA-BSDS-001..004 and RRP-URA-01..04 land; remaining exposure centers on continuous runner resource pressure, proto schema regressions, and telemetry tuning.
- **Follow-up:** After continuous mode stabilises, revisit hazard expansion (GPU timing, randomized fault injection) and evaluate direct Tamiyo consumption of `BSDSIssued` events.

## Execution Confidence
- **Assessment:** High confidence. Surfaces are well understood, automated coverage is strong, and staged smoke validation plus load/alert rehearsals reduce the chance of latent defects.

