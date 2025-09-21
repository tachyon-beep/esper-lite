# Urabrask — Implementation Roadmap (From BSDS‑Lite to Crucible)

Goal: bring up Urabrask in phases, starting from the BSDS‑Lite groundwork to a minimal, repeatable evaluation harness and benchmark suite.

Phases
- Phase 0 (now): BSDS‑Lite on descriptors (see speculative/bsds-lite). Consumers wired; dashboards live.
- Phase 0.5: Heuristic micro‑benchmarks during Tezzeret compile in CI; populate perf fields with CURATED/HEURISTIC provenance.
- Phase 1: Minimal crucible service (single‑host) that runs the hazard battery and emits BSDS + Benchmark artefacts (provenance=URABRASK) for top N templates.
- Phase 2: Expand hazard coverage, add signatures/immutability/WAL, integrate Oona topics and Nissa dashboards; deprecate CURATED entries.

Work Items
1) Leyline schema
- Add `BSDS`, `BlueprintBenchmark`, `HazardLevel`, `Provenance`, `BSDSIssued/Failed`, `BenchmarkReport`.

2) Urabrask service (minimal)
- CLI/daemon that accepts a BlueprintDescriptor/artifact, runs tests with fixed seeds/time caps, and writes BSDS+Benchmark protobufs.
- WAL for in‑flight runs; resume on restart; artefacts stored beside Urza entries or a configured object store.

3) Hazard test battery (v1)
- Gradient stability (explode/vanish cases), memory profile under standard shapes, numerical precision sensitivity, failure handling (NaN, OOM).
- Deterministic inputs, fixed seeds; configurable duration budget.

4) Performance suite (v1)
- Reference latency/throughput profiles; p50/p95; environment capture (device, driver, torch version); regression tolerances.

5) Integration – Urza
- Attach BSDS/Benchmark artefacts to blueprint records; index by hazard/score bands; add filters for queries.

6) Integration – Tamiyo/Karn
- Consume BSDS bands/recommendations for gating/selection; annotate AdaptationCommands with risk & mitigation.

7) Observability & Ops
- Telemetry: crucible durations, issued/failed counts, regressions found.
- Dashboards: hazard histograms, benchmark trend lines; alerts for CRITICAL hazards.
- Runbook: failure triage, waivers, and re‑evaluation process.

Acceptance Criteria
- Phase 1: BSDS+Benchmark artefacts produced for top N templates; visible in Urza; consumed by Tamiyo; dashboards active; provenance URABRASK.
- Phase 2: Signatures/immutability/WAL in place; CURATED/HEURISTIC retired; Oona topics integrated; SLOs tracked in Nissa.
