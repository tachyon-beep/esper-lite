# Kasmina — Delta Matrix

Status and evidence per requirement. See rubric in `../rubric.md`.

| Area | Design Source | Expected Behaviour | Prototype Evidence | Status | Severity | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Lifecycle states & gates | `02-kasmina.md` (Lifecycle), 11‑state with G0–G5 | Enforce 11 states with validation gates; transitions guarded | `src/esper/kasmina/lifecycle.py`, `src/esper/kasmina/seed_manager.py` | Missing | Must‑have | Prototype uses a reduced Leyline stage set with no gates; DORMANT/EMBARGOED/RESETTING/TERMINATED require a Leyline schema update before adoption. |
| Lifecycle fast path | `02-kasmina.md` | Allow germinating→training fast path | `tests/kasmina/test_lifecycle.py` | Implemented | Nice‑to‑have | Covered by tests. |
| Kernel execution (Urza/Tezzeret) | `02.1-kasmina-kernel-execution.md` | Load vetted kernel artefacts; async path; GPU cache | `src/esper/kasmina/seed_manager.py` (runtime.fetch_kernel) | Partially Implemented | Should‑have | Synchronous fetch with simple fallback; no GPU‑resident cache or async scheduling. |
| Fallback kernel | `02.1-kasmina-kernel-execution.md` | Identity/fallback kernel on failure or latency breach | `src/esper/kasmina/seed_manager.py` (`_load_fallback`) | Implemented | Should‑have | Budget breach triggers fallback and telemetry. |
| Gradient isolation invariant | `02-kasmina.md`, `02.1-kasmina-kernel-execution.md` | Runtime enforcement via backward hooks; `.detach()` in blending; breaker on violation | `src/esper/kasmina/seed_manager.py` | Missing | Must‑have | Only a one‑time parameter‑ID overlap sanity check exists; no hooks, no blending, no breaker. |
| Alpha blending phases | `02-kasmina.md` | Blend outputs during grafting; identity in training | — | Missing | Should‑have | No BLENDING phase or alpha schedule implemented. |
| Parameter registration | `02.3-kasmina-parameter-registration.md` | Register per‑seed params; validate updates; teacher immutability | — | Missing | Must‑have | No per‑seed registry, LR‑group mapping, or teacher immutability enforcement. |
| Memory governance | `02.2-kasmina-memory-pools.md` | TTL caches, epoch GC, KD memory budgeting | — | Missing | Should‑have | No TTL caches/GC or KD memory controls in Kasmina. |
| Safety stack (breaker/timers) | `02.4-kasmina-safety-mechanisms.md` | Circuit breakers, monotonic timing, pause/identity semantics | — | Missing | Must‑have | No breakers/timers or pause semantics. |
| torch.compile fallback | `02.4-kasmina-safety-mechanisms.md` | Monitor and fall back on instability | — | Missing | Nice‑to‑have | Not implemented. |
| Security envelope | `02-kasmina.md` | HMAC‑SHA256, nonce, freshness on critical commands | — | Missing | Must‑have | Oona supports optional HMAC on bus envelopes; Kasmina does not verify signatures/nonce/freshness on commands. |
| Telemetry pipeline | `02-kasmina.md` | Structured metrics/events, emergency bypass for critical alerts | `src/esper/core/telemetry.py`; seed_manager emissions | Partially Implemented | Should‑have | Metrics expanded, health indicators carry priority hints, rollback readiness surfaced; emergency transport still TODO. |
| Performance validation | `02.5-kasmina-performance-validation.md` | Benchmarks for kernel load, isolation overhead, Leyline limits, KD costs | — | Missing | Nice‑to‑have | No harness in prototype for Kasmina; broader profiling exists elsewhere. |
| Distributed coordination | `02-kasmina.md` | Epoch‑aligned barriers; Byzantine logging | — | Missing | Nice‑to‑have | Not present in Kasmina (low priority for single‑node prototype). |
| Knowledge distillation (C‑024) | `02-kasmina.md`, `02.1`, `02.2` | Teacher load/checkpointing; KD loss plumbing; memory budgeting | — | Missing | Nice‑to‑have | Not implemented in Kasmina (future option). |
| Rollback readiness | `02-kasmina.md` | Emit checkpoints on stateful changes; 500 ms/12 s SLA | — | Missing | Should‑have | No checkpoint emissions or SLA timing in Kasmina. |
