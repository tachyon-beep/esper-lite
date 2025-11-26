# Tezzeret — Prototype Delta (Compilation Forge)

Executive summary: the prototype now executes a torch.compile pipeline that builds representative blueprint modules, exports guard metadata, primes the Inductor cache, and persists artifacts into Urza with compile/pre‑warm timings. `TezzeretForge` enumerates Karn templates, implements a timeout breaker with eager fallback, resumes via WAL on failure, and stores guard specs / fallback flags alongside each artifact, with `KernelCatalogUpdate` updates flowing to Oona. Compiler/forge metrics are captured for Weatherlight to surface, while richer strategy selection, continuous telemetry emission, and resource monitoring/TTL cleanup remain on the roadmap.

Outstanding Items (for coders)

- Periodic telemetry emission to Oona
  - Wire `TezzeretForge.build_telemetry_packet()` into a periodic publisher (Weatherlight). Ensure breaker state elevates severity.
  - Pointers: `src/esper/tezzeret/runner.py` (packet builder, event buffer); `src/esper/weatherlight/service_runner.py` (publisher hook).

- Resource monitoring and TTL maintenance
  - Sample GPU utilisation/memory (when CUDA available) and bound compile concurrency; add TTL cleanup for Inductor cache dir and stale WAL files.
  - Pointers: `src/esper/tezzeret/runner.py` (main loop); `src/esper/tezzeret/compiler.py` (cache dir resolution).

- Inductor cache lifecycle + metrics
  - Track basic cache size/age and expose metrics; optional hit/miss counters if observable; add TTL eviction policy.
  - Pointers: `src/esper/tezzeret/compiler.py` ( `_inductor_cache`, `CompileJobConfig.inductor_cache_dir`).

- Strategy matrix expansion
  - Add Fast (reduced flags) and Emergency (CPU‑only) strategies; select based on breaker/resource state; record strategy in extras and telemetry.
  - Pointers: `src/esper/tezzeret/compiler.py::_compile_blueprint` (strategy switch), `src/esper/tezzeret/runner.py` (choose strategy).

- WAL durability
  - Add CRC and atomic O_DSYNC writes; unify compiler/forge WAL schema and cleanly resume/clear.
  - Pointers: `src/esper/tezzeret/compiler.py` (`_persist_wal/_clear_wal`), `src/esper/tezzeret/runner.py` (`_persist_pending/_load_pending_jobs`).

- Artifact signing + versioning
  - Stamp semantic version and optional signature/HMAC in Urza extras and include in `KernelCatalogUpdate`.
  - Pointers: `src/esper/urza/library.py` (extras read/write), `src/esper/tezzeret/compiler.py` (build extras), `src/esper/urza/pipeline.py`.

- Catalog refresh
  - Add periodic re‑enumeration of blueprints (or explicit trigger) beyond the startup scan.
  - Pointers: `src/esper/tezzeret/runner.py::_discover_jobs`.

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
- Implemented: catalog enumeration; per‑job WAL; torch.compile pipeline; guard spec + digest persisted; compile + pre‑warm timing; KernelCatalogUpdate emission (via pipeline); Urza extras persist checksum/guard_digest/guard_spec(_summary)/pre‑warm metrics; forge breaker with conservative mode; telemetry events and packet builder.
- Missing/Partial: Inductor cache lifecycle/TTL; periodic Oona telemetry emission (metrics snapshots exist); resource monitoring and TTL cleanup; artifact signing/versioning.

Next Actions (minimal, high‑leverage)
- Torch 2.8 export graphs (optional)
  - Current build records guard specs (shape/dtype/stride) and digest. Consider `torch.export` graph capture if/when needed for stricter verification.
  - Eager fallback strategy and flags already recorded on failure.
- Inductor cache reuse (env + compiler.py)
  - Honour `TORCHINDUCTOR_CACHE_DIR` (settable via env/EsperSettings) to persist the compiled cache; log cache path in KernelCatalogUpdate.
- Breakers + conservative mode (runner.py)
  - Present: timeout‑backed breaker with backoff and conservative strategy. Add resource‑aware throttling if GPU constraints are added.
  - Add periodic telemetry heartbeat via Oona emission.
- Telemetry (runner.py and/or pipeline caller)
  - Present: metrics snapshots + event buffer + TelemetryPacket builder. Wire periodic publish to Oona.
- Resource monitoring + TTL (runner.py)
  - Sample GPU utilisation/memory (if CUDA available) and bound concurrency; schedule periodic TTL cleanup of any local caches/WAL debris.
- Signing/versioning (compiler.py + Urza extras)
  - Attach artifact version (semver) and a signature/HMAC (optional) in Urza’s `_urza` extras; include in KernelCatalogUpdate for traceability.

Touchpoints
- `src/esper/tezzeret/compiler.py`: implement compile/export/guards; attach metadata to `KernelCatalogUpdate`.
- `src/esper/tezzeret/runner.py`: add breakers, telemetry emission, resource checks, and simple strategy selection.
- `src/esper/urza/library.py`: store new guard/metadata fields (already handles extras); ensure retrieval to feed prefetch p50/p95.
