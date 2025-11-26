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

Telemetry (current prototype)
- Available via metrics snapshot/packet (forge):
  - Compilation latency (per strategy): `tezzeret.compilation.duration_ms.{strategy}`
  - Pre‑warm latency (per strategy): `tezzeret.prewarm.ms.{strategy}`
  - Counters: `tezzeret.jobs.total`, `tezzeret.jobs.failed`, `tezzeret.jobs.retried`, `tezzeret.jobs.started`, `tezzeret.jobs.completed`
  - Breaker/state: `tezzeret.breaker.state` (0=closed,1=half‑open,2=open), `tezzeret.breaker.open_total`, `tezzeret.mode.conservative`
- Publication: TelemetryPacket builder present (`TezzeretForge.build_telemetry_packet()`); periodic Oona emission to be wired via Weatherlight.
- Not implemented: Inductor cache hit/miss metrics (planned; only cache dir is recorded today).

Event examples (TelemetryEvent description)
- `compile_started`, `compile_succeeded`, `compile_failed{reason}`
- `prewarm_completed`, `breaker_open`, `conservative_mode_enabled`

Routing
- Weatherlight can include `tezzeret.*` metrics in its periodic telemetry packet via a metrics provider; breakers should elevate severity when repeatedly opening. Periodic emission is pending.
