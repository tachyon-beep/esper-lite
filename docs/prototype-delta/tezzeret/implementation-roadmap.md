# Tezzeret — Implementation Roadmap (Closing the Delta)

Goal: evolve the forge to match the design (Leyline‑first, no tech debt).

| Order | Theme | Key Tasks | Outcome |
| --- | --- | --- | --- |
| 1 | torch.compile pipeline | Implement Standard pipeline using torch.compile; record duration and resource metrics; persist metadata | Real compilation artefacts + metrics |
| 2 | Strategies | Add Fast (reduced flags) and Emergency (CPU‑only) strategies; optional Aggressive flag | Flexible performance envelope |
| 3 | Circuit breakers | Breakers around timeouts, GPU utilisation, memory; conservative mode throttles concurrency and strategy | Safe degradation and recovery |
| 4 | WAL durability | CRC headers and atomic O_DSYNC writes; resume incomplete jobs; unify forge+compiler WAL format | Robust crash recovery |
| 5 | Telemetry | Emit `tezzeret.compilation.duration_ms{strategy}`, breaker state, conservative mode; publish via Oona | Operator visibility |
| 6 | Resource monitoring & TTL | Track GPU/memory; TTL cleanup on caches/queues; periodic maintenance | Bounded resource usage |
| 7 | Signing/versioning | Add artifact signatures and semantic versioning in Urza metadata | Integrity and traceability |

Notes
- Keep inputs as Karn’s Leyline descriptors; no dynamic blueprint generation in lite build.
- Coordinate WAL and telemetry schema with Urza consumers.

Acceptance Criteria
- Standard pipeline compiles catalog at startup; failures recover via WAL.
- Breakers and conservative mode kick in under resource/time pressure.
- Telemetry present and consumed in tests/harness.

