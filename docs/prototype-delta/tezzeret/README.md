# Tezzeret — Prototype Delta (Compilation Forge)

Executive summary: the prototype now executes a torch.compile pipeline that builds representative blueprint modules, exports guard metadata, primes the Inductor cache, and persists artifacts into Urza with compile/pre‑warm timings. `TezzeretForge` enumerates Karn templates, resumes via WAL on failure, and stores guard specs / fallback flags alongside each artifact, with `KernelCatalogUpdate` updates flowing to Oona. Remaining gaps include richer strategy selection, circuit breakers, streaming telemetry, and resource monitoring/TTL cleanup.

Documents in this folder:
- `delta-matrix.md` — requirement‑by‑requirement status with evidence
- `traceability-map.md` — mapping of design assertions to code/tests
- `implementation-roadmap.md` — plan to close gaps without tech debt
- `pytorch-2.8-upgrades.md` — mandatory 2.8 compilation pipeline (torch.compile/export, pre‑warm, guards)

Design sources:
- `docs/design/detailed_design/06-tezzeret-unified-design.md`
- `docs/design/detailed_design/06.1-tezzeret-compilation-internals.md`

Implementation evidence (primary):
- `src/esper/tezzeret/compiler.py`, `src/esper/tezzeret/runner.py`
- Tests: `tests/tezzeret/test_compiler.py`, `tests/tezzeret/test_runner.py`

Status (prototype)
- Implemented: catalog enumeration; per‑job WAL; stub compilation; compile + pre‑warm timing; KernelCatalogUpdate emission (via pipeline); Urza metadata persists checksum/guard_digest(prelim)/pre‑warm samples.
- Missing/Partial: torch.compile/export guards; Inductor cache reuse; breaker/telemetry around compilation; resource monitoring and TTL cleanup; signed/versioned artifacts.

Next Actions (minimal, high‑leverage)
- Torch 2.8 compile + export guards (compiler.py)
  - Wrap blueprint modules with `torch.compile(..., dynamic=True)`; pre‑warm with representative shapes.
  - Export guards with `torch.export` (shape/dtype/stride constraints) and persist alongside the artifact in Urza.
  - On failure, fall back to eager, set an `eager_fallback=true` flag in the catalog update/Urza extras.
- Inductor cache reuse (env + compiler.py)
  - Honour `TORCHINDUCTOR_CACHE_DIR` (settable via env/EsperSettings) to persist the compiled cache; log cache path in KernelCatalogUpdate.
- Breakers + conservative mode (runner.py)
  - Add a small circuit breaker around per‑job compile (threshold/timeouts); when open, throttle to a “Fast” strategy or pause new jobs with backoff.
  - Record breaker state and last error in a periodic telemetry heartbeat.
- Telemetry (runner.py and/or pipeline caller)
  - Emit `tezzeret.compilation.duration_ms{strategy}`, `tezzeret.prewarm.ms`, and breaker state via Oona’s telemetry stream at a modest interval.
- Resource monitoring + TTL (runner.py)
  - Sample GPU utilisation/memory (if CUDA available) and bound concurrency; schedule periodic TTL cleanup of any local caches/WAL debris.
- Signing/versioning (compiler.py + Urza extras)
  - Attach artifact version (semver) and a signature/HMAC (optional) in Urza’s `_urza` extras; include in KernelCatalogUpdate for traceability.

Touchpoints
- `src/esper/tezzeret/compiler.py`: implement compile/export/guards; attach metadata to `KernelCatalogUpdate`.
- `src/esper/tezzeret/runner.py`: add breakers, telemetry emission, resource checks, and simple strategy selection.
- `src/esper/urza/library.py`: store new guard/metadata fields (already handles extras); ensure retrieval to feed prefetch p50/p95.
