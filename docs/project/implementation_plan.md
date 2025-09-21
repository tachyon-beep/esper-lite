# Esper-Lite Prototype Implementation Plan

This plan translates the charter and capability matrix into concrete work packages. It is organised as four vertical slices that deliver incremental end-to-end functionality, plus cross-cutting enablement tasks.

> The full legacy specifications live under `docs/design/detailed_design/old/`. Treat those documents as the authoritative reference for lifecycle/state-machine behaviour while implementing the lightweight slices below.

## Delivery Pillars

1. **Environment & Contracts Enablement (Sprint 0)**
   - *Goal:* Stand up shared infrastructure and guarantee schema alignment before feature work begins.
   - **Tasks**
     - Stand up Git repo structure, CI scaffold, and coding standards (formatter, lint, type checks).
     - Create Leyline contract repository/module; generate language bindings; wire serialization benchmark tests into CI.
     - Provision prototype infrastructure containers (Redis, Prometheus, Elasticsearch) via docker-compose; document ports, credentials, persistent volumes.
     - Build Oona skeleton service running against local Redis; expose health endpoint and metrics.
     - Build Nissa skeleton ingest pipeline with Prometheus/Elasticsearch wiring and placeholder dashboards.
     - Produce developer bootstrapping guide (setup, env vars, service launch order).

2. **Slice 1 – Control Loop Core (Sprints 1–2)**
   - *Goal:* Demonstrate real-time morphogenesis with stubbed blueprint retrieval.
   - **Tasks**
     - Implement minimal Tolaria trainer with configurable host model (decide ResNet-lite vs Transformer-lite); emit `SystemStatePacket`s via Leyline contracts, adhering to the control loop defined in `docs/design/detailed_design/old/01-tolaria.md`.
     - Implement Kasmina seed manager supporting seed registration, graft placeholder kernels, gradient isolation checks (initially verifying host tensor IDs) while preserving the full 11-state lifecycle from `old/02-kasmina.md`.
     - Implement Tamiyo inference service with stub GNN (pretrained or simple MLP) returning deterministic adaptation command; integrate risk thresholds and conservative mode toggles per `old/03-tamiyo.md`.
     - Implement Oona integration for Tolaria→Tamiyo telemetry and Tamiyo→Kasmina commands; configure stream names, consumer groups.
     - Add Kasmina telemetry publishing + Tamiyo aggregation path; deliver field report stub.
     - Create integration test harness to assert epoch hook budget, command delivery, and telemetry round-trip.

3. **Slice 2 – Blueprint Pipeline (Sprints 3–4)**
   - *Goal:* Replace stubs with real static blueprint lifecycle.
   - **Tasks**
     - Populate Karn template catalog (metadata files + service interface); implement basic request routing and logging (no breakers) while respecting tiered safeguards from `old/05-karn.md`.
     - Build Tezzeret startup compiler: load blueprint definitions, run single Standard torch.compile pipeline, persist artifacts, handle basic retries; maintain WAL/telemetry requirements from `old/06-tezzeret.md`.
     - Implement Urza catalog service: metadata storage (SQLite or Postgres) with local FS object store and in-process cache; add WAL persistence (no Redis tier) in line with `old/08-urza.md`.
     - Wire Kasmina seed activation path to request kernels from Urza; validate fetch latency budget (<10 ms p50).
     - Extend Tamiyo to request blueprint metadata for policy context.
     - Add tests covering blueprint query flow, compilation recovery, WAL replay scenario.

4. **Slice 3 – Telemetry, Observability, and Safety (Sprints 5–6)**
   - *Goal:* Operational visibility and breaker behaviours.
   - **Tasks**
     - Flesh out Kasmina/Tolaria/Tamiyo telemetry payloads; map metrics to Nissa dashboards; implement serialization property tests (see telemetry responsibilities in `old/01-tolaria.md`, `old/02-kasmina.md`, `old/03-tamiyo.md`).
     - Complete Nissa alert/SLO rules (`training_latency_high`, `kasmina_isolation_violation`, `oona_queue_depth`, `tezzeret_compile_retry_high`); wire routing stubs (Slack/email) as defined in `old/10-nissa.md`.
     - Implement Oona conservative-mode triggers; create load-test script to exercise queue depth thresholds.
     - Add breaker and rollback drills: simulate Kasmina gradient isolation failure, Tamiyo timeout, Tezzeret compile error; ensure telemetry/alerts fire.
     - Document operator runbook (normal operations, breaker handling, rollback, alert response).

5. **Slice 4 – Offline Learning Loop (Sprints 7–8)**
   - *Goal:* Close the learning loop by updating Tamiyo policy from field reports.
   - **Tasks**
     - Finalise field report schema (metrics, outcome enums) and persistence (WAL + retention).
     - Implement Simic field report ingestion, replay buffer (PyG `HeteroData`), and single-node PPO training with LoRA fine-tuning support (IMPALA deferred), following the process in `old/04-simic.md`.
     - Add policy validation harness (chaos/property tests scaled to prototype) and gating logic.
     - Implement Oona-based policy update publication; integrate Tamiyo hot-reload with safety checks.
     - Demonstrate end-to-end run: initial policy deploy → adaptation events → Simic training → policy redeploy → new adaptations.

6. **Cross-Cutting Engineering**
   - **Security & Access Control:** Implement shared secret/HMAC for Leyline messages (following Kasmina/Tamiyo detailed designs); configure secrets management (.env or vault stub).
   - **Testing & QA:** Build CI suite covering unit tests, contract tests (protobuf compatibility), integration scenarios per slice, serialization benchmarks.
   - **Performance Profiling:** Establish profiling harness (PyTorch profiler, tracing) to validate latency targets; add dashboards for key metrics.
   - **Documentation:** Maintain subsystem READMEs, API docs, and sequence diagrams; update prototype report throughout milestones.

## Dependencies & Sequencing
- Environment enablement must complete before slice 1 development starts.
- Blueprint pipeline requires Karn templates prepared during slice 1 backlog (parallelizable by separate engineer).
- Observability slice depends on telemetry definitions stabilised in slices 1–2.
- Offline learning slice requires field reports emitted in slice 2 and telemetry plumbing from slice 3.

## Roles & Ownership (suggested)
- **Control Loop Lead:** Own Tolaria/Kasmina/Tamiyo integration.
- **Blueprint Lead:** Own Karn/Tezzeret/Urza stack and blueprint catalog.
- **Infrastructure Lead:** Own Leyline, Oona, Nissa, and deployment tooling.
- **Learning Lead:** Own Simic implementation and policy lifecycle.
- **QA/Tooling:** Cross-functional support for CI, tests, and runbooks.

## Open Decisions Affecting Implementation
- Host model architecture selection (impacts Tolaria/Kasmina work).
- Deployment footprint (docker-compose vs Kubernetes) which affects automation scripts.
- Telemetry retention requirements (affects Nissa storage sizing).
- Policy training hardware availability (GPU vs CPU) for Simic.

## Next Actions
1. Pre-Implementation Enum Validation (Leyline)
   - Audit enum usage in code and docs vs `contracts/leyline/leyline.proto`.
   - Produce subsystem mapping tables (internal → Leyline) and attach to PR.
   - Confirm no functional coverage is lost by mapping (use `system_health`/events for operational states).
   - Run serialization size/latency checks to ensure Option B budgets hold.
2. Review and confirm slice sequencing and resource assignments.
3. Convert slice tasks into backlog tickets with acceptance criteria.
4. Finalise open decisions (host model, deployment tooling) before starting Slice 1.
5. Kick off Sprint 0 focusing on environment and contract enablement.
