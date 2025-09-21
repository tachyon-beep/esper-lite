# Urza — Traceability Map

| Design Assertion | Source | Implementation | Tests |
| --- | --- | --- | --- |
| Store blueprint metadata + artifact, recover via WAL | `08-urza.md`, `08.1` | `src/esper/urza/library.py` (save/get/WAL) | `tests/urza/test_library.py` (persist, recover) |
| Query by tier/id | `08-urza.md` | `get()`, `fetch_by_tier()`, `list_all()` | `tests/urza/test_library.py` |
| Serve kernels to Kasmina | `08-urza.md` | `src/esper/urza/runtime.py::UrzaRuntime.fetch_kernel` | `tests/kasmina/test_lifecycle.py` (Urza runtime path) |
| Pipeline compile→store | `08-urza.md` | `src/esper/urza/pipeline.py` | `tests/urza/test_pipeline.py`, `tests/integration/test_blueprint_pipeline_integration.py` |
| Multi‑tier caching & telemetry | `08.1` | — | — |

