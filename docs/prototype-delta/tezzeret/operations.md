# Tezzeret — Operations & Configuration (Prototype)

Environment variables (recommended)
- `TEZZERET_INDUCTOR_CACHE_DIR`
  - Path to a persistent directory for PyTorch Inductor’s compiled cache (wired into `EsperSettings`).
  - Rationale: enables cross‑process reuse of compiled kernels; reduces first‑run latency.
  - Example: `/var/cache/tezzeret/inductor`
- `CUDA_VISIBLE_DEVICES`
  - Pin compilation to a specific GPU or disable (for CPU‑only Emergency strategy).
- `OMP_NUM_THREADS`
  - Control CPU thread pool during compilation to stabilise timings on shared hosts.

Notes
- Tezzeret now honours `TEZZERET_INDUCTOR_CACHE_DIR`; other variables remain operational hints.
- Keep the cache on a fast local disk (NVMe). Clean periodically if not bounded by TTL.

Telemetry (Oona → Nissa) — suggested metric names
- Compilation latency (per job/strategy):
  - `tezzeret.compilation.duration_ms{strategy}`
  - `tezzeret.prewarm.ms{strategy}`
- Throughput/counters:
  - `tezzeret.jobs.total`
  - `tezzeret.jobs.failed`
  - `tezzeret.jobs.retried`
- Breaker/state:
  - `tezzeret.breaker.state` (0=closed,1=half‑open,2=open)
  - `tezzeret.breaker.open_total`
- Cache:
  - `tezzeret.inductor.cache_hits`
  - `tezzeret.inductor.cache_misses`

Event examples (TelemetryEvent description)
- `compile_started`, `compile_succeeded`, `compile_failed{reason}`
- `prewarm_completed`, `breaker_open`, `conservative_mode_enabled`

Routing
- Send Tezzeret telemetry to Oona’s telemetry stream (normal priority). Breaker transitions and repeated job failures can be sent at HIGH priority if the platform’s policy requires.
