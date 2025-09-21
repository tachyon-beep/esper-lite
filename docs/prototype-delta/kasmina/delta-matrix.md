# Kasmina — Delta Matrix

Status and evidence per requirement. See rubric in `../rubric.md`.

| Area | Design Source | Expected Behaviour | Prototype Evidence | Status | Severity | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Lifecycle states & gates | `02-kasmina.md` (Lifecycle), legacy 11‑state with G0–G5 | Enforce 11 states with validation gates; transitions guarded | `src/esper/kasmina/lifecycle.py`, `src/esper/kasmina/seed_manager.py` | Diverges | Should‑have | Prototype uses Leyline enum stages (UNKNOWN→GERMINATING→…→CULLING→CANCELLED). No explicit G0–G5 checks or embargo/reset/terminated states. |
| Lifecycle fast path | `02-kasmina.md` | Allow germinating→training fast path | `tests/kasmina/test_lifecycle.py` | Implemented | Nice‑to‑have | Covered by tests. |
| Kernel execution (Urza/Tezzeret) | `02.1-kasmina-kernel-execution.md` | Load vetted kernel artefacts; async path; GPU cache | `src/esper/kasmina/seed_manager.py` (runtime.fetch_kernel) | Partially Implemented | Should‑have | Synchronous fetch with simple fallback; no GPU‑resident cache or async scheduling. |
| Fallback kernel | `02.1-kasmina-kernel-execution.md` | Identity/fallback kernel on failure or latency breach | `src/esper/kasmina/seed_manager.py` (`_load_fallback`) | Implemented | Should‑have | Budget breach triggers fallback and telemetry. |
| Gradient isolation invariant | `02-kasmina.md`, `02.1-kasmina-kernel-execution.md` | Runtime enforcement via backward hooks; `.detach()` in blending; breaker on violation | `src/esper/kasmina/seed_manager.py` (`_attach_kernel`) | Missing | Must‑have | Only checks parameter id overlap once; no hooks, no dot‑product check, no blending, no breaker. |
| Alpha blending phases | `02-kasmina.md` | Blend outputs during grafting; identity in training | — | Missing | Should‑have | Not present. |
| Parameter registration | `02.3-kasmina-parameter-registration.md` | Register per‑seed params; validate updates; teacher immutability | — | Missing | Must‑have | No registration, no LR group mapping, no teacher protections. |
| Memory governance | `02.2-kasmina-memory-pools.md` | TTL caches, epoch GC, KD memory budgeting | — | Missing | Should‑have | No TTL/GC in Kasmina; no KD allocation checks. |
| Safety stack (breaker/timers) | `02.4-kasmina-safety-mechanisms.md` | Circuit breakers, monotonic timing, pause/identity semantics | — | Missing | Should‑have | Only fallback kernel exists; no breaker/timer framework or pause API. |
| torch.compile fallback | `02.4-kasmina-safety-mechanisms.md` | Monitor and fall back on instability | — | Missing | Nice‑to‑have | Not implemented. |
| Security envelope | `02-kasmina.md` | HMAC‑SHA256, nonce, freshness on critical commands | `src/esper/security/signing.py` (utility only) | Partially Implemented | Should‑have | Signing helper exists, but Kasmina command path does not verify signatures/nonces. |
| Telemetry pipeline | `02-kasmina.md` | Structured metrics/events, emergency bypass for critical alerts | `src/esper/core/telemetry.py`; seed_manager emissions | Partially Implemented | Should‑have | Emits metrics/events; no emergency bypass path or priority handling. |
| Performance validation | `02.5-kasmina-performance-validation.md` | Benchmarks for kernel load, isolation overhead, Leyline limits, KD costs | — | Missing | Nice‑to‑have | No harness in prototype for Kasmina; broader profiling exists elsewhere. |
| Distributed coordination | `02-kasmina.md` | Epoch‑aligned barriers; Byzantine logging | — | Missing | Nice‑to‑have | Not present in Kasmina. |
| Knowledge distillation (C‑024) | `02-kasmina.md`, `02.1`, `02.2` | Teacher load/checkpointing; KD loss plumbing; memory budgeting | — | Missing | Nice‑to‑have | Not present in Kasmina prototype. |
| Rollback readiness | `02-kasmina.md` | Emit checkpoints on stateful changes; 500 ms/12 s SLA | — | Missing | Should‑have | Not implemented in Kasmina path. |

