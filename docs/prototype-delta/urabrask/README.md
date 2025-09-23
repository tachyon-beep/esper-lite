# Urabrask — Prototype Delta (Evaluation & Benchmarking; Planning Baseline)

Lead note: Urabrask has been rescoped to support BSDS for Tamiyo decision‑making in the prototype. We will adopt Leyline contracts from day 1 and land a BSDS‑Lite path via Urza extras as transport. See the completion packages plan for the sequence and acceptance criteria.

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

Planned Leyline contracts (Day 1)
- See `leyline-schema-draft.md` for enums/messages to add in `contracts/leyline/leyline.proto`:
  - Enums: HazardBand, HandlingClass, ResourceProfile, Provenance
  - Messages: BSDS, BlueprintBenchmark(+Profile), BSDSIssued/Failed, BenchmarkReport

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
- `URABRASK_COMPLETION_PACKAGES.md` — authoritative work packages with acceptance criteria and estimates (start here).
- `leyline-schema-draft.md` — Day‑1 Leyline additions for BSDS/benchmarks.
Also see:
- `bsds-lite-integration.md` — transport via Urza extras and consumer mapping.
- `../speculative/bsds-lite/` — JSON schema + examples (mirror of Leyline BSDS for prototype transport).
- `metrics.md` — prototype metrics/alerts; `timeout-matrix.md` — budgets.
