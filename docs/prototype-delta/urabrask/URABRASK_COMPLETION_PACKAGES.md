# Urabrask Completion Packages (Prototype Delta)

Scope: land Urabrask in the prototype with BSDS‑Lite first, enabling Tamiyo to make safer decisions immediately. Use Urza `extras` to carry BSDS artefacts (no new Leyline contracts in Phase 0/1). Defer full Leyline adoption and the comprehensive crucible to later phases unless explicitly called for by the delta.

Execution order is optimized to deliver highest leverage earliest while minimizing integration risk and churn.

## Phase 0 — BSDS‑Lite (Now, Leyline‑First)

### URA0 — Leyline Contracts (Day 1) + BSDS‑Lite Schema
- Changes
  - Add canonical Leyline messages/enums under `contracts/leyline/leyline.proto`:
    - Enums: `HazardBand {LOW, MEDIUM, HIGH, CRITICAL}`, `HandlingClass {STANDARD, RESTRICTED, QUARANTINE}`, `ResourceProfile {CPU, GPU, MEMORY_HEAVY, IO_HEAVY, MIXED}`, `Provenance {URABRASK, CURATED, HEURISTIC, EXTERNAL}`.
    - Messages: `BSDS` (blueprint safety sheet), `BlueprintBenchmark` (perf summary), events `BSDSIssued`/`BSDSFailed`, `BenchmarkReport`.
  - Regenerate Python bindings (`scripts/generate_leyline.py`) and ensure imports compile across packages.
  - Define BSDS‑Lite JSON schema and examples under `docs/prototype-delta/speculative/bsds-lite/` mirroring Leyline `BSDS` (used only as an Urza extras transport in the prototype).
- Files
  - `contracts/leyline/leyline.proto` (new enums/messages)
  - `src/esper/leyline/_generated/leyline_pb2.*` (generated)
  - `docs/prototype-delta/speculative/bsds-lite/{README.md,schema.md,examples/minimal.json}`
  - `docs/prototype-delta/urabrask/bsds-lite-integration.md`
  - `docs/prototype-delta/urabrask/decision-taxonomy.md`
- Tests
  - `scripts/check_shared_types.py` passes; generated code loads.
  - JSON schema validates examples.
- Acceptance
  - Canonical Leyline contracts exist from day 1; schema/examples published.
- Estimate
  - 0.5–1 day; Complexity: Low.
- Rollback
  - Consumers simply ignore missing `extras["bsds"]`.

### URA1 — Urza Extras Persistence + Retrieval (BSDS)
- Changes
  - Standardize saving/loading `extras["bsds"]` with Urza; add doc notes; verify cache/eviction TTL behavior.
- Files
  - Docs only; code path already supports `extras`.
- Tests
  - Persist round‑trip BSDS via `UrzaLibrary.save(..., extras={"bsds": {...}})` then `get()`.
- Acceptance
  - BSDS survives DB round‑trip, cache TTL, eviction.
- Estimate
  - 0.5 day; Complexity: Low.
- Rollback
  - None (no runtime changes required).

### URA2 — Tamiyo Consumer & Gating (Prototype)
- Changes
  - Wire Tamiyo to consume `extras["bsds"]`, override risk when `risk_score` present, annotate `bsds_*`, and gate on `hazard_band`.
- Files
  - `src/esper/tamiyo/service.py` (already landed in this repo series).
- Tests
  - Unit: Urza record with BSDS produces `bsds_present` event and HIGH/CRITICAL hazard mapping as expected.
- Acceptance
  - Tamiyo decisions reflect BSDS; telemetry carries annotations; budgets unchanged.
- Estimate
  - 0.5 day; Complexity: Low.
- Rollback
  - Behavior gated by extras presence.

## Phase 0.5 — Producer Stub + Observability

### URA3 — BSDS‑Lite Producer (Heuristic CLI)
- Changes
  - Add `src/esper/urabrask/bsds.py` with `compute_bsds(descriptor, artifact_path, hints) -> dict` (heuristics: parameter size, layer types, dropout usage, stage/risk priors).
  - Add `src/esper/urabrask/service.py` minimal CLI: reads a descriptor + artifact and writes `extras["bsds"]` to Urza.
- Files
  - `src/esper/urabrask/{bsds.py,service.py}`
  - `tests/urabrask/test_bsds_lite.py`
- Tests
  - Deterministic output for a fixed descriptor; JSON serializable; attaches to Urza record.
- Acceptance
  - CLI produces BSDS for an input blueprint; Tamiyo consumes it end‑to‑end.
- Estimate
  - 1–1.5 days; Complexity: Medium.
- Rollback
  - Keep producer out‑of‑band; operators can inject BSDS manually via extras.

### URA4 — Nissa Ingest + Alerts (BSDS)
- Changes
  - Ingest BSDS/Tamiyo BSDS telemetry into Prometheus; add alerts for `CRITICAL`/`HIGH` hazards.
- Files
  - `src/esper/nissa/{observability.py,alerts.py}` (doc‑driven changes)
  - `docs/prototype-delta/urabrask/metrics.md`
- Tests
  - Unit: ingest telemetry with `bsds_hazard_*` events; alert routes recorded.
- Acceptance
  - Alerts fire; dashboards show hazard distributions.
- Estimate
  - 1 day; Complexity: Medium.
- Rollback
  - Feature‑flag new rules; keep ingestion passive.

## Phase 1 — Crucible (Minimal Hazard Battery)

### URA5 — Crucible Harness Skeleton
- Changes
  - Add `Crucible` runner with deterministic seed control, time budgets, and result bundle (logs + BSDS fields + provenance=URABRASK).
- Files
  - `src/esper/urabrask/crucible.py`
  - `tests/urabrask/test_crucible.py`
- Tests
  - Repeatable outcomes across runs; run time capped; result artifacts on disk.
- Acceptance
  - Minimal hazard battery executes and emits a BSDS dict consistent with schema.
- Estimate
  - 2–3 days; Complexity: Medium‑High.
- Rollback
  - Keep CLI to disable crucible; fall back to heuristic producer.

### URA6 — Hazard Tests v1
- Changes
  - Implement tests for: gradient instability, NaN/inf handling, OOM simulation (guarded), precision sensitivity (fp32/bf16 delta), memory watermark.
- Files
  - `src/esper/urabrask/crucible.py` (test modules)
- Tests
  - Synthetic models with fixed shapes; pass/fail + scores aggregated.
- Acceptance
  - BSDS risk bands derive from hazard outcomes with documented mapping.
- Estimate
  - 3–4 days; Complexity: High.
- Rollback
  - Scope hazard set; disable OOM test in constrained CI.

### URA7 — Benchmark Suite v1 (Optional for Prototype)
- Changes
  - Reference latency/throughput profiles; p50/p95; environment capture.
  - Store `extras["benchmarks"]` with profile summaries.
- Files
  - `src/esper/urabrask/benchmarks.py`
  - `tests/urabrask/test_benchmarks.py`
- Tests
  - Latency metrics recorded; simple regression tolerance check.
- Acceptance
  - Benchmark artefacts attached; Nissa displays trend lines.
- Estimate
  - 2 days; Complexity: Medium.
- Rollback
  - Mark as optional; skip in CPU‑only CI.

## Phase 2 — Hardening (Optional)

### URA8 — Signing + Immutability + WAL
- Changes
  - Sign BSDS payloads; append‑only WAL; verify signatures on load; mark `provenance`.
- Files
  - `src/esper/urabrask/service.py` (sign/verify)
  - `docs/prototype-delta/urabrask/README.md` (ops notes)
- Tests
  - Tamper detection; WAL recovery.
- Acceptance
  - Signature verified; recovery succeeds.
- Estimate
  - 2 days; Complexity: Medium.
- Rollback
  - Keep signature verification optional (fail‑open for prototype).

---

## Test & CI Notes
- Fast: `pytest tests/tamiyo -q`, targeted `tests/urabrask -q` as packages land.
- Integration: end‑to‑end BSDS injection → Tamiyo gating.
- Lint: `pylint --rcfile .codacy/tools-configs/pylint.rc src/esper`.
- Perf: ensure no budget regressions in Tamiyo service (step p95 ≤ 10 ms; inference p95 ≤ 45 ms).

## References
- `docs/prototype-delta/speculative/bsds-lite/README.md`
- `docs/prototype-delta/urabrask/bsds-lite-integration.md`
- `docs/prototype-delta/urabrask/metrics.md`
- `docs/prototype-delta/urabrask/decision-taxonomy.md`
- `docs/prototype-delta/urabrask/leyline-schema-draft.md`
- `docs/prototype-delta/urabrask/implementation-roadmap.md`
- `docs/design/detailed_design/07-urabrask-unified-design.md`
