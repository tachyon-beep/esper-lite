# Esper-Lite Prototype Backlog

Backlog items are grouped by delivery pillar. Each ticket includes a short description and acceptance criteria for agile planning.

## Sprint 0 – Environment & Contracts Enablement

### TKT-001: Repository & CI Scaffold ✅
- **Description:** Initialize Esper-Lite codebase structure, coding standards (formatter, lint, type checks), and CI pipeline skeleton.
- **Acceptance Criteria:**
  - Repo contains root README, src/tests layout, and tooling configs (format/lint/type).
  - CI workflow executes lint + type check on push and reports status.
  - Developer setup instructions documented.

### TKT-002: Leyline Contract Module ✅
- **Description:** Create shared `.proto` repository/module, generate Python bindings, and add serialization benchmarks to CI.
- **Acceptance Criteria:**
  - Leyline proto files live in dedicated module with build script.
  - Generated bindings consumable by Tolaria/Kasmina/Tamiyo/Simic.
  - Benchmark test verifies `<80µs/<280B/≤4 allocations` for `SystemStatePacket` and `AdaptationCommand`.

### TKT-003: Local Infrastructure Bootstrap ✅
- **Description:** Provision docker-compose stack for Redis, Prometheus, Elasticsearch; document ports, volumes, credentials.
- **Acceptance Criteria:**
  - `docker-compose.yml` starts services with healthy status.
  - Documentation lists connection details and troubleshooting steps.
  - Basic health check proves Redis/Prometheus/Elasticsearch reachable.

### TKT-004: Oona Skeleton Service ✅
- **Description:** Implement minimal Oona service connecting to Redis Streams, exposing `/health` and metrics endpoint.
- **Acceptance Criteria:**
  - Service publishes/consumes test messages via NORMAL stream.
  - `/health` returns 200 when Redis reachable.
  - Metrics endpoint exposes publish/consume counters.

### TKT-005: Nissa Skeleton Ingest ✅
- **Description:** Build initial Nissa ingestion worker that reads telemetry envelopes from Oona and writes to Prometheus/Elasticsearch.
- **Acceptance Criteria:**
  - Sample telemetry event arrives in Prometheus/ES.
  - Service exposes `/health`.
  - Placeholder dashboard accessible (even if minimal).

### TKT-006: Developer Bootstrap Guide ✅
- **Description:** Document local setup (dependencies, environment variables, service launch order).
- **Acceptance Criteria:**
  - GUIDE.md (or equivalent) describes step-by-step setup.
  - Fresh environment can go from clone → running stack using the guide.

## Slice 1 – Control Loop Core (Sprints 1–2)

### TKT-101: Tolaria Minimal Trainer ✅
- **Description:** Implement single-node trainer with configurable host model (decide ResNet-lite vs Transformer-lite), broadcasting `SystemStatePacket`s.
- **Acceptance Criteria:**
  - Trainer runs epochs with configurable dataset stub.
  - Emits serialized `SystemStatePacket` passing Leyline tests.
  - Hooks ready for Tamiyo call-out and Kasmina grafting.
  - Control loop behaviour consistent with `docs/design/detailed_design/old/01-tolaria.md` (end-of-epoch handshake, WAL checkpoints, rollback stack).

### TKT-102: Kasmina Seed Manager Skeleton ✅
- **Description:** Implement seed registration, placeholder kernel grafting, and gradient isolation checks.
- **Acceptance Criteria:**
  - Seeds can register/deregister with Tolaria.
  - Placeholder kernel graft modifies model without breaking training.
  - Gradient isolation check logs violation when host/seed grads overlap.
  - Eleven-stage lifecycle (Dormant → … → Terminated) implemented as defined in `docs/design/detailed_design/old/02-kasmina.md`.

### TKT-103: Tamiyo Stub Policy Inference ✅
- **Description:** Implement Tamiyo service with initial GNN (or MLP stub) returning deterministic adaptation commands, risk gating, and conservative mode toggles.
- **Acceptance Criteria:**
  - Service accepts `SystemStatePacket`, responds with `AdaptationCommand`.
  - Risk thresholds configurable; conservative mode toggled on demand.
  - Latency ≤45 ms on target hardware.
  - Risk engine, telemetry aggregation, and field-report generation follow `docs/design/detailed_design/old/03-tamiyo.md`.

### TKT-104: Oona Control Loop Integration ✅
- **Description:** Wire Tolaria→Tamiyo telemetry and Tamiyo→Kasmina commands over Redis streams; configure stream names & consumer groups.
- **Acceptance Criteria:**
  - Telemetry publishes to defined topic; Tamiyo consumes.
  - Commands delivered to Kasmina within expected latency.
  - Retry/backoff handles transient Redis errors.
  - Message semantics honour at-least-once guarantees from `docs/design/detailed_design/old/09-oona.md`.

### TKT-105: Tamiyo Field Report Stub ✅
- **Description:** Implement Tamiyo field report creation and publication to Oona; stub schema aligned with Leyline.
- **Acceptance Criteria:**
  - Field report appears on `tamiyo.field_reports` stream.
  - Payload validates against Leyline schema.
  - Tamiyo persists report metadata for at least one epoch.
  - Field-report content matches lifecycle documented in `docs/design/detailed_design/old/03-tamiyo.md`.

### TKT-106: Control Loop Integration Test Harness ✅
- **Description:** Create automated test verifying epoch hook timing, message delivery, and telemetry round-trip.
- **Acceptance Criteria:**
  - Test spins up Tolaria/Kasmina/Tamiyo with mocked blueprint fetch.
  - Validates 18 ms epoch boundary budget is not exceeded.
  - Verifies telemetry and commands exchange successfully.

## Slice 2 – Blueprint Pipeline (Sprints 3–4)

### TKT-201: Karn Template Catalog ✅
- **Description:** Implement static blueprint catalog service with metadata files and simple logging (no breakers).
- **Acceptance Criteria:**
  - 50 templates defined with ids, tiers, and allowed params.
  - Service responds to blueprint query with deterministic template.
  - Basic request/response logging in place.
  - Safety tiers, approval flags, and conservative pool behaviour match `docs/design/detailed_design/old/05-karn.md`.

### TKT-202: Tezzeret Standard Compiler ✅
- **Description:** Implement Tezzeret startup compiler running single Standard `torch.compile` pipeline with basic retry logic.
- **Acceptance Criteria:**
  - On startup, all templates compiled to artifacts stored on disk.
  - Job retry logic handles at least one simulated failure.
  - WAL ensures restart resumes incomplete compile in line with `docs/design/detailed_design/old/06-tezzeret.md`.

### TKT-203: Urza Catalog Service ✅
- **Description:** Build Urza service using SQLite (or Postgres) metadata + local filesystem artifacts; include in-process cache.
- **Acceptance Criteria:**
  - API stores/retrieves blueprint metadata and artifact path.
  - Cache hits after warmup achieve <10 ms p50 fetch.
  - WAL/transaction logs allow recovery after crash; metadata governance matches `docs/design/detailed_design/old/08-urza.md`.

### TKT-204: Kasmina Kernel Fetch Integration ✅
- **Description:** Connect Kasmina to Urza for kernel retrieval during seed activation; enforce latency budget.
- **Acceptance Criteria:**
  - Kasmina requests blueprint id → receives compiled module.
  - Activation path maintains training continuity.
  - Observed fetch latency meets <10 ms p50 target.

### TKT-205: Tamiyo Blueprint Metadata Consumption ✅
- **Description:** Extend Tamiyo to pull blueprint metadata for decision context.
- **Acceptance Criteria:**
  - Tamiyo can query Urza/Karn for blueprint info synchronously/asynchronously as needed.
  - Metadata influences adaptation command selection (configurable).
  - Integration honours metadata contracts defined in `docs/design/detailed_design/old/03-tamiyo.md` and `old/08-urza.md`.

### TKT-206: Blueprint Pipeline Tests ✅
- **Description:** Add tests covering blueprint query flow, compile retry, and WAL recovery.
- **Acceptance Criteria:**
  - Test suite simulates compile failure and verifies retry path.
  - Crash/restart resumes compile using WAL.
  - End-to-end test: Tamiyo request → Karn → Tezzeret artifacts → Urza fetch.

## Slice 3 – Telemetry, Observability, Safety (Sprints 5–6)

### TKT-301: Telemetry Schema Finalisation ✅
- **Description:** Finalise telemetry payloads for Tolaria, Kasmina, Tamiyo; add serialization property tests.
- **Acceptance Criteria:**
  - Telemetry schema documented with example payloads.
  - Serialization tests ensure Option B budgets satisfied.
  - Telemetry fields cover metrics required by Nissa and legacy subsystem diagrams (`docs/design/detailed_design/old/01-tolaria.md`, `old/02-kasmina.md`, `old/03-tamiyo.md`).

### TKT-302: Nissa Alert/SLO Implementation ✅
- **Description:** Implement alert rules (`training_latency_high`, `kasmina_isolation_violation`, `oona_queue_depth`, `tezzeret_compile_retry_high`) and routing stubs.
- **Acceptance Criteria:**
  - Alerts configured and fire during simulated faults.
  - Routing stubs deliver notifications (Slack/email placeholders).
  - SLO dashboard displays error budget burn consistent with `docs/design/detailed_design/old/10-nissa.md`.

### TKT-303: Oona Backpressure & Testing ✅
- **Description:** Implement priority handling/backpressure without breakers; create load-test to exercise queue thresholds.
- **Acceptance Criteria:**
  - Emergency path bypasses normal queue under load.
  - Load-test demonstrates dropping/delaying messages within defined thresholds.
  - Metrics expose queue depth & retry counts.
  - Behaviour validated against routing/backpressure rules in `docs/design/detailed_design/old/09-oona.md`.

### TKT-304: Breaker & Rollback Drills ✅
- **Description:** Simulate gradient isolation failure, Tamiyo timeout, Tezzeret compile retry escalation; ensure telemetry/alerts respond.
- **Acceptance Criteria:**
  - Each drill triggers appropriate alert and logs.
  - Runbook documents recovery steps.
  - System returns to nominal state post-drill.

### TKT-305: Operator Runbook ✅
- **Description:** Document standard operations, alert handling, rollback procedures for Tolaria/Kasmina/Tamiyo/Oona/Nissa.
- **Acceptance Criteria:**
  - Runbook covers start/stop, health checks, alert response, rollback.
  - Includes references to dashboards and metrics.

## Slice 4 – Offline Learning Loop (Sprints 7–8)

### TKT-401: Field Report Schema & Persistence ✅
- **Description:** Finalise field report schema, implement WAL-backed persistence, enforce retention policy.
- **Acceptance Criteria:**
  - Schema documented; payloads pass Leyline validation.
  - Reports stored with WAL guaranteeing recovery after crash.
  - Retention policy (e.g., 24 h) enforced via cleanup in alignment with `docs/design/detailed_design/old/03-tamiyo.md` and `old/04-simic.md` field-report workflow.

### TKT-402: Simic Ingestion & PPO+LoRA Training ✅
- **Description:** Implement field report ingestion, replay buffer (PyG `HeteroData`), and single-node PPO training with LoRA fine-tuning.
- **Acceptance Criteria:**
  - Reports ingested into replay buffer with TTL/size bounds.
  - PPO training loop runs on sample data producing updated policy weights.
  - LoRA adapter support toggled via config.
  - Replay buffer management and validation gating follow `docs/design/detailed_design/old/04-simic.md`.

### TKT-403: Policy Validation Harness ✅
- **Description:** Implement chaos/property tests and gating logic for new Tamiyo policies.
- **Acceptance Criteria:**
  - Validation suite runs automatically after training.
  - Policy promoted only if validation passes.
  - Failures emit telemetry/alert.

### TKT-404: Policy Update Publication & Hot Reload ✅
- **Description:** Publish policy updates via Oona, implement Tamiyo hot-reload with safety checks.
- **Acceptance Criteria:**
  - Policy update message delivered to Tamiyo.
  - Tamiyo performs staged load (verify → activate) with rollback on failure.
  - Hot reload is logged and observable.
  - Deployment workflow matches `docs/design/detailed_design/old/04-simic.md` (publish) and `old/03-tamiyo.md` (hot reload).

### TKT-405: End-to-End Learning Demo ✅
- **Description:** Demonstrate full loop: initial policy deploy → adaptations → Simic training → policy redeploy → new adaptations.
- **Acceptance Criteria:**
  - Demo script executes full flow without manual intervention (except start/stop).
  - Metrics captured for policy impact before/after update.
  - Results summarized for prototype report.

## Cross-Cutting & Supporting Tickets

### TKT-501: Security & Secrets Management ◔
- **Description:** Implement shared secret/HMAC handling across Leyline messages; manage secrets via environment config.
- **Acceptance Criteria:**
  - Messages signed/verified in Kasmina/Tamiyo/Tolaria as per design.
  - Secrets stored securely (env vars, not committed).
  - Documentation covers rotation procedure.

### TKT-502: CI Test Matrix ✅
- **Description:** Expand CI to run unit, integration, contract, and serialization tests per slice.
- **Acceptance Criteria:**
  - CI executes relevant tests on PR.
  - Failures block merge with clear reporting.

### TKT-503: Performance Profiling Harness ◔
- **Description:** Add profiling tools (PyTorch profiler, tracing) to verify latency budgets.
- **Acceptance Criteria:**
  - Profiling scripts produce latency reports for key operations.
  - Dashboards/metrics show epoch timing, graft latency, publish latency.

### TKT-504: Prototype Report Compilation ◑
- **Description:** Maintain living document capturing metrics, lessons learned, and recommendations for Phase 2.
- **Acceptance Criteria:**
  - Report updated at end of each slice.
  - Final version summarises success criteria and open risks.

## Recent Progress Highlights
- Tolaria/Tamiyo stream telemetry and field reports through Oona; Nissa ingests those packets and exposes Prometheus metrics via the new ASGI helper.
- Simic consumes Tamiyo field reports, trains against buffered telemetry, and publishes serialized policy updates that Tamiyo can hot-reload safely.
- Urza/Karn/Tezzeret blueprint pipeline handles synchronous requests, compiles artifacts, persists them in a SQLite catalog, and provides a runtime loader for Kasmina.
- Oona client now enforces priority routing and backpressure thresholds while exporting reroute/drop counters for Nissa alerts, bringing TKT-303 to completion.
- GitHub Actions CI matrix executes unit, integration, contract, and serialization suites (with optional perf benchmark) to guard new changes end-to-end.

---

This backlog should be refined during sprint planning; tickets can be split/merged as capacity dictates.
