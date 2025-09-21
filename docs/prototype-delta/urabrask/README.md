# Urabrask — Prototype Delta (Evaluation & Benchmarking; Planning Baseline)

Lead note: Urabrask is out of scope to implement in the Esper‑Lite prototype, but it is in scope to define what we will look for. This delta captures the intended behaviours, outputs, and integration points so the team can assess readiness and plan a bring‑up later.

Intent
- Define the evaluation crucible and benchmarking suite that generate Blueprint Safety Data Sheets (BSDS) and performance profiles for blueprints.
- Make clear, testable outcomes so downstream controllers can consume risk/perf signals consistently.

Design references
- `docs/design/detailed_design/07-urabrask-unified-design.md`
- `docs/design/detailed_design/07.1-urabrask-safety-validation.md`
- `docs/design/detailed_design/07.2-urabrask-performance-benchmarks.md`

What we’ll be looking for (deliverables)
- Evaluation Harness (Crucible)
  - Deterministic test battery for gradient stability, memory profile, numerical stability, and failure behaviour.
  - Reproducible inputs; bounded duration; seed control; report artefacts.
- BSDS Generator (authoritative)
  - Structured risk artefact (BSDS) with hazards, recommended handling, perf references, provenance, and versioning.
  - Signatures and immutability; attached to blueprint records in Urza.
- Performance Benchmarks (reference suite)
  - Latency/throughput measurements over standard batch profiles; resource curves.
  - Regression detection with tolerances; telemetry export for dashboards.
- Integration
  - Urza: store BSDS + perf artefacts; expose query filters by hazard class/bands.
  - Oona: publish evaluation events; topic taxonomy for BSDS issuance/failures.
  - Nissa: dashboards for hazard distributions, benchmark trends; alerts on CRITICAL hazards.
  - Tamiyo/Karn: consume BSDS fields for gating and selection; annotate decisions with provenance.

Planned Leyline contracts (schema RFC)
- `message BSDS` (Blueprint Safety Data Sheet) with fields aligned to BSDS‑Lite and extended metrics.
- `message BlueprintBenchmark` with latency/throughput and environment metadata.
- `enum HazardLevel { LOW, MODERATE, HIGH, CRITICAL }` and `enum Provenance { CURATED, HEURISTIC, URABRASK }`.
- Evaluation events: `BSDSIssued`, `BSDSFailed`, `BenchmarkReport`.

Telemetry & SLOs
- Metrics: `urabrask.crucible.duration_ms`, `urabrask.benchmark.latency_ms{profile}`, `urabrask.bsds.issued_total`, `urabrask.bsds.failed_total`.
- SLOs: crucible job p95 ≤ configured target; WAL recovery ≤ 12s; BSDS issuance success ≥ 99% for healthy inputs.

Acceptance (for bring‑up later)
- Harness produces stable, repeatable BSDS/benchmark artefacts for top N templates; stored in Urza; visible in Nissa; Tamiyo/Karn consume fields for policy hints.
- Signed BSDS with provenance URABRASK available; CURATED/HEURISTIC entries deprecated.

Files in this folder
- `delta-matrix.md` — intended capabilities and current status (prototype baseline is “not present”).
- `traceability-map.md` — design anchors and intended integration points.
- `implementation-roadmap.md` — phased plan to bring up Urabrask from BSDS‑Lite to full crucible.
