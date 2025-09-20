# Esper-Lite Prototype Charter

## Purpose
Deliver an end-to-end Esper-Lite implementation that proves morphogenetic adaptation on a single-node, sub-10M parameter model. The prototype must exercise the seven functional subsystems (Tolaria, Kasmina, Tamiyo, Simic, Urza, Tezzeret, Karn) and their shared services (Leyline, Oona, Nissa) exactly as defined in the current HLD and detailed-design docs.

## Objectives
- Demonstrate online architectural adaptation: Tamiyo issues `AdaptationCommand`s that Kasmina executes inside Tolaria’s training loop without breaching the 18 ms epoch boundary budget.
- Validate blueprint lifecycle: Karn serves pre-approved templates, Tezzeret precompiles kernels, and Urza fulfills Kasmina fetches within published latency targets (<10 ms p50).
- Prove telemetry and safety loop: Kasmina isolation alerts, Tolaria timing metrics, and Tamiyo field reports travel over Oona using Leyline contracts and surface in Nissa dashboards/alerts.
- Show offline learning readiness: Simic consumes Tamiyo field reports and produces a policy artifact ready for redeployment (no distributed scale expected).

## Scope
### In-Scope Functional Subsystems
- **Tolaria:** Epoch orchestration, checkpoint handoff, rollback hooks, and Tamiyo call-outs.
- **Kasmina:** Seed lifecycle management, kernel grafting, gradient isolation enforcement, KD-off by default.
- **Tamiyo:** Policy inference, risk gating, telemetry aggregation, field report publication.
- **Simic:** Field report ingestion, replay buffer, single-node IMPALA/PPO training loop, policy checkpoint publish.
- **Urza:** Static blueprint catalog, kernel artifact storage, WAL-backed durability, cache tiering.
- **Tezzeret:** Startup compilation of the fixed blueprint set, WAL recovery, conservative-mode throttling.
- **Karn:** Runtime blueprint query service exposing the 50-template, tiered catalog with circuit breakers.

### Supporting Infrastructure
- **Leyline:** Option B protobuf contracts, schema governance, serialization benchmarks (<80 µs / <280 B / ≤4 allocations).
- **Oona:** Redis Streams message bus with NORMAL/EMERGENCY routing, at-least-once delivery, breaker-protected publish/consume paths.
- **Nissa:** Prometheus + Elasticsearch ingest, alerting rules (`training_latency_high`, `kasmina_isolation_violation`, etc.), mission-control API/WS endpoints.

## Success Criteria
- **Latency:** Tolaria end-of-epoch hook ≤18 ms (3.5 ms state assembly + 12 ms Tamiyo inference + 1.5 ms adaptation + 1 ms guard); Kasmina kernel graft <0.5 ms hot; Oona publish p95 <25 ms.
- **Correctness:** Kasmina enforces `∇L_host ⋂ ∇L_seed = ∅`; Leyline schema version locked at 1; blueprint fetch hit rate ≥95 % after warmup.
- **Observability:** Nissa dashboards show live timing, breaker, and queue-depth metrics; defined alerts fire in simulated fault drills.
- **Learning Loop:** Simic produces at least one updated Tamiyo policy checkpoint from replayed field reports; deployment-ready artifact passes validation gates.

## Deliverables
1. **Codebase:** Implementations for all subsystems and infrastructure services aligned with detailed-design contracts.
2. **Configuration & Deployment Assets:** Helm/docker-compose or equivalent to bootstrap Redis, Prometheus, Elasticsearch, and subsystem services.
3. **Test & Demo Scripts:** Automated integration tests for control loop, blueprint fetch, and telemetry; reproducible demo scenario showcasing blueprint injection.
4. **Operational Runbook:** SOP for start/stop, monitoring, rollback, and alert response covering Tolaria/Kasmina/Tamiyo/Oona/Nissa.
5. **Prototype Report:** Summary of metrics collected, gaps discovered, and recommendations for Phase 2.

## Approach & Milestones
- **M0 – Foundations (Week 0-1):** Finalize Leyline schema repo, provision Oona/Nissa infrastructure, stub services with contract tests.
- **M1 – Control Loop Slice (Week 2-4):** Implement Tolaria ↔ Tamiyo ↔ Kasmina path with mock blueprint artifacts; demonstrate zero-disruption graft.
- **M2 – Blueprint Pipeline (Week 5-6):** Bring Karn/Tezzeret/Urza online with static library; validate fetch/compile latencies.
- **M3 – Telemetry & Observability (Week 7-8):** Complete telemetry emission, Nissa dashboards, alert drills, breaker exercises.
- **M4 – Offline Learning (Week 9-10):** Enable Simic replay/IMPALA loop, deliver policy update roundtrip, finalize prototype report.

## Assumptions & Constraints
- Single GPU node (H100-class) with sufficient CPU for supporting services; no multi-node coordination.
- PyTorch 2.8 baseline; CUDA graphs optional but nice-to-have.
- Redis Streams, Prometheus, Elasticsearch available as managed or containerized dependencies.
- Knowledge Distillation features disabled unless memory budget validated (≤7 GB headroom).
- Governance: schema changes require Leyline approval workflow; no ad-hoc contract edits during sprint.

## Out of Scope / Non-Goals
- Dynamic blueprint generation, Urabrask evaluation, or innovation-plane expansion.
- Multi-tenant or cross-datacenter Oona deployments.
- Distributed Simic training or actor farm build-out.
- Automated policy rollout pipelines beyond manual validation/deploy.
- Production-grade security hardening (TLS, IAM); prototype may rely on shared secrets.

## Dependencies
- Tooling: PyTorch 2.8, PyG, Redis 7+, Prometheus 2.x, Elasticsearch 8.x, FastAPI/Flask, CUDA 12.x.
- Contracts: `leyline.schema_version = 1`; message envelopes must adhere to Option B specs.
- External services: Email/Slack/PagerDuty stubs for Nissa alert routing.

## Risks & Mitigations
- **Latency overruns:** Enforce profiling budget reviews each sprint; fallback to conservative mode configs.
- **Contract drift:** Centralized CI checks for Leyline schema hash; reject mismatches on build.
- **Telemetry overload:** Configure Oona/Nissa conservative modes; rehearse drop/degenerate scenarios.
- **Blueprint catalog gaps:** Seed library upfront; if coverage insufficient, expand Karn templates before sprint end.
- **Team bandwidth:** Prioritize vertical slices; postpone optional features (KD, IMPALA fan-out) if timeline slips.

## Open Questions
- Exact host model architecture for the demo (resnet-lite vs transformer-lite) – decision needed by M1 kickoff.
- Source of replay data for Simic (synthetic vs captured from prototype runs).
- Preferred deployment mechanism (docker-compose vs Kubernetes) for the prototype showcase.

## Governance & Reporting
- Weekly checkpoint review with architecture + engineering leads.
- CI gates on serialization benchmarks, latency regression tests, and basic integration checks before merges.
- Prototype sign-off requires demonstration of success criteria plus review of outstanding risks/non-goals.
