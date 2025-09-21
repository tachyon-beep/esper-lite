# Tezzeret — Traceability Map

| Design Assertion | Source | Implementation | Tests |
| --- | --- | --- | --- |
| Enumerate blueprints and compile into Urza | `06-tezzeret.md` | `src/esper/tezzeret/runner.py::TezzeretForge.run()` | `tests/tezzeret/test_runner.py::test_forge_compiles_catalog` |
| Skip existing artifacts | `06-tezzeret.md` | Forge checks Urza before compile | `tests/tezzeret/test_runner.py::test_forge_skips_existing` |
| Resume from WAL after failure | `06.1` | Forge WAL (`forge_wal.json`); compiler WAL | `tests/tezzeret/test_runner.py::test_forge_resumes_from_wal` |
| Retry failed jobs | `06.1` | Compiler retries | `tests/tezzeret/test_runner.py::test_forge_retries_failed_job` |
| torch.compile pipeline + export guards | `06.1` | `compiler.py` (to be implemented); `pytorch-2.8-upgrades.md` | — |
| Inductor cache and pre‑warm metrics | `06.1` | `src/esper/tezzeret/compiler.py::latest_catalog_update` | `tests/tezzeret/test_compiler.py::test_compiler_persists_artifact` |
| Pre‑warm percentiles at serve time | `06.1` | `src/esper/urza/library.py` stores samples; `src/esper/urza/prefetch.py` computes p50/p95 | — |
| Publish KernelCatalogUpdate to Oona | `06-tezzeret.md` | `src/esper/urza/pipeline.py::BlueprintPipeline` (catalog_notifier) | `scripts/run_demo.py` (wires Oona publisher) |
| Breakers + telemetry around compilation | `06.1` | `runner.py` (to be implemented) | — |
