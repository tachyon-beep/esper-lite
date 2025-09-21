# Tezzeret — Delta Matrix

Status and evidence per requirement. See rubric in `../rubric.md`.

| Area | Design Source | Expected Behaviour | Prototype Evidence | Status | Severity | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Blueprint polling & queueing | `06-tezzeret.md` | Enumerate blueprints from Urza/Karn at startup; enqueue | `src/esper/tezzeret/runner.py::TezzeretForge._discover_jobs()` | Implemented | Should‑have | Enumerates Karn; no periodic refresh. |
| Compilation pipeline (torch.compile) | `06.1` | Deterministic pipeline; resource/time metrics; strategies | `src/esper/tezzeret/compiler.py` (torch.compile + pre-warm + guards) | Implemented | Must‑have | Builds representative modules, runs torch.compile with guard spec + eager fallback. |
| Artifact guards & shape metadata | `06.1` | Export guards for runtime shape checks | `src/esper/tezzeret/compiler.py` guard spec persisted via Urza extras | Implemented | Must‑have | Guard digest derived from shape/dtype/stride metadata stored alongside artifacts. |
| Inductor cache reuse | `06.1` | Persist and reuse compiled cache where feasible | Compiler honours `inductor_cache_dir` (wraps `TORCHINDUCTOR_CACHE_DIR`) | Partially Implemented | Should‑have | Cache dir configurable; lifecycle/TTL management still pending. |
| Performance metadata | `06.1` | Record p50/p95 compile/pre‑warm latencies | `src/esper/tezzeret/compiler.py` (compile/prewarm ms in extras) | Implemented | Should‑have | Compile and pre-warm timings stored in Urza alongside catalog updates. |
| WAL & crash recovery | `06.1` | WAL for forge and compiler; resume in‑flight jobs | `runner.py` WAL for pending; `compiler.py` WAL per job; tests resume | Partially Implemented | Must‑have | JSON WAL; no CRC/O_DSYNC; minimal.
| Circuit breakers & conservative mode | `06.1` | Breakers around timeouts/resources; throttle to Fast pipeline | `src/esper/tezzeret/runner.py` (timeout breaker + eager fallback strategy) | Partially Implemented | Must‑have | Basic timeout breaker and fallback; resource-aware throttling still pending. |
| Telemetry | `06-tezzeret.md` | `tezzeret.compilation.*`, breaker state | Compiler/forge metrics snapshot; TelemetryPacket builder in forge | Partially Implemented | Should‑have | Metrics gathered and packet built; periodic Oona emission pending integration with deployment. |
| Catalog update notification (Oona) | `06-tezzeret.md` | Publish `KernelCatalogUpdate` to Oona | `src/esper/urza/pipeline.py::BlueprintPipeline` (catalog_notifier) | Implemented | Should‑have | Wired in pipeline; demo publishes updates; Urza prefetch uses metadata. |
| Resource monitoring | `06.1` | GPU utilisation ≤25 %, memory guards, TTL cleanup | — | Missing | Should‑have | Not present. |
| Signing/versioning | `06-tezzeret.md` | Sign artifacts, include version in Urza | — | Missing | Nice‑to‑have | Not present. |
| Leyline as canonical | `00-leyline` | Use Leyline messages where applicable | Indirect; uses Karn (Leyline descriptors) | Implemented | Must‑have | Descriptor inputs align to Leyline via Karn. |
