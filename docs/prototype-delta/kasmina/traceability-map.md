# Kasmina — Traceability Map

Mapping of design assertions to implementation artefacts and tests.

| Design Assertion | Source | Implementation | Tests |
| --- | --- | --- | --- |
| Lifecycle uses Leyline enums as single source of truth | `docs/design/detailed_design/02-kasmina.md` | `src/esper/kasmina/lifecycle.py` | `tests/kasmina/test_lifecycle.py`, `tests/leyline/test_enum_alignment.py` |
| Seed germination→training fast path exists | `docs/design/detailed_design/02-kasmina.md` | `src/esper/kasmina/lifecycle.py` (allowed transitions) | `tests/kasmina/test_lifecycle.py` |
| Kernel fetch via Urza, with latency budget and fallback | `docs/design/detailed_design/02.1-kasmina-kernel-execution.md` | `src/esper/kasmina/seed_manager.py` (`_graft_seed`, `_load_fallback`) | `tests/kasmina/test_seed_manager.py` (latency/fallback), `tests/integration/test_control_loop.py` |
| Gradient isolation enforced at runtime | `docs/design/detailed_design/02-kasmina.md` | — | — |
| Parameter registration and update validation | `docs/design/detailed_design/02.3-kasmina-parameter-registration.md` | — | — |
| Memory TTL caches and epoch GC | `docs/design/detailed_design/02.2-kasmina-memory-pools.md` | — | — |
| Circuit breaker and monotonic timers | `docs/design/detailed_design/02.4-kasmina-safety-mechanisms.md` | — | — |
| HMAC + nonce + freshness on commands | `docs/design/detailed_design/02-kasmina.md` | Utility: `src/esper/security/signing.py`; not wired in Kasmina | — |
| Structured telemetry with health status | `docs/design/detailed_design/02-kasmina.md` | `src/esper/core/telemetry.py`; emissions in `seed_manager.py` | `tests/kasmina/test_seed_manager.py` (telemetry presence) |
| Performance validation thresholds | `docs/design/detailed_design/02.5-kasmina-performance-validation.md` | — | — |

