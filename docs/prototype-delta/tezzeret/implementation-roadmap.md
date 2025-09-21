# Tezzeret — Implementation Roadmap (Closing the Delta)

Goal: evolve the forge to match the design (Leyline‑first, no tech debt).

| Order | Theme | Key Tasks | Outcome |
| --- | --- | --- | --- |
| 1 | torch.compile pipeline | Implement Standard pipeline using torch.compile; export guards; pre‑warm representative shapes | ✅ Real compilation artefacts + guards, no runtime compile in Kasmina |
| 2 | Strategies | Add Fast (reduced flags) and Emergency (CPU‑only) strategies; optional Aggressive flag | Flexible performance envelope |
| 3 | Circuit breakers | Breakers around timeouts, GPU utilisation, memory; conservative mode throttles concurrency and strategy | Safe degradation and recovery |
| 4 | WAL durability | CRC headers and atomic O_DSYNC writes; resume incomplete jobs; unify forge+compiler WAL format | Robust crash recovery |
| 5 | Telemetry | Emit `tezzeret.compilation.duration_ms{strategy}`, pre‑warm p50/p95, breaker state; publish via Oona | Partially complete — per‑job metrics captured and `KernelCatalogUpdate` published in pipeline; streaming telemetry pending |
| 6 | Resource monitoring & TTL | Track GPU/memory; TTL cleanup on caches/queues; periodic maintenance | Bounded resource usage |
| 7 | Signing/versioning | Add artifact signatures and semantic versioning in Urza metadata; eager‑fallback flag | Integrity and traceability |

Notes
- Keep inputs as Karn’s Leyline descriptors; no dynamic blueprint generation in lite build.
- Coordinate WAL and telemetry schema with Urza consumers.

Status update
- Step 1: Implemented — torch.compile pipeline with pre‑warm; guard spec + digest persisted; fallback recorded.
- Step 2: Partially implemented — conservative strategy present; Fast/Emergency variants pending.
- Step 3: Partially implemented — timeout breaker + conservative mode present; resource‑aware throttling pending.
- Step 4: Partially implemented — JSON WALs for compiler and forge; no CRC/O_DSYNC.
- Step 5: Partially implemented — metrics snapshots + TelemetryPacket builder; periodic Oona emission pending.
- Step 6: Missing — resource monitoring & TTL cleanup not implemented.
- Step 7: Missing — artifact signing/versioning not implemented.

Acceptance Criteria
- Standard pipeline compiles catalog at startup; guards exported; failures recover via WAL and mark eager fallback.
- Breakers and conservative mode kick in under resource/time pressure.
- Telemetry present (compile/pre‑warm latency) and consumed in tests/harness.
