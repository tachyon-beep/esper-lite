# Urabrask — Traceability Map (Intended Integration)

Design anchors
- `docs/design/detailed_design/07-urabrask-unified-design.md`
- `docs/design/detailed_design/07.1-urabrask-safety-validation.md`
- `docs/design/detailed_design/07.2-urabrask-performance-benchmarks.md`

Intended code locations (to be created when in scope)
- `src/esper/urabrask/crucible.py` — evaluation runner (hazard battery)
- `src/esper/urabrask/bsds.py` — BSDS model/helpers
- `src/esper/urabrask/benchmarks.py` — performance suite orchestration
- `src/esper/urabrask/service.py` — CLI/daemon wrapper
- `tests/urabrask/test_crucible.py` — repeatability, duration bounds
- `tests/urabrask/test_bsds_integration.py` — Urza attachment, schema checks
- `tests/urabrask/test_benchmarks.py` — baseline profiles, regressions

Contracts & telemetry (planned)
- Leyline: `BSDS`, `BlueprintBenchmark`, `HazardLevel`, `Provenance`, events `BSDSIssued/Failed`, `BenchmarkReport`.
- Metrics: `urabrask.crucible.duration_ms`, `urabrask.bsds.issued_total`, `urabrask.benchmark.latency_ms{profile}`.

