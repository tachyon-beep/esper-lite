# Kasmina — Prototype Delta (Execution Layer)

Executive summary: the prototype implements the Leyline 11‑state lifecycle with gate checks (G0–G5), kernel fetch via Urza with latency fallback, backward‑hook gradient isolation monitoring with `.detach()` blending, a per‑seed/teacher parameter registry, TTL memory caches, a circuit breaker + monotonic timers, and HMAC/nonce/freshness verification for commands. Structured telemetry reports seed stages, gate events, health, and priority. Remaining work includes a performance validation harness, optional GPU‑resident cache/async path, explicit emergency telemetry bypass, and KD loss/budgeting integration.

Documents in this folder:
- `delta-matrix.md` — requirement‑by‑requirement status with evidence
- `traceability-map.md` — mapping of design assertions to code/tests
- `implementation-roadmap.md` — suggested backlog to close gaps

Design sources:
- `docs/design/detailed_design/02-kasmina-unified-design.md`
- `docs/design/detailed_design/02.1-kasmina-kernel-execution.md`
- `docs/design/detailed_design/02.2-kasmina-memory-pools.md`
- `docs/design/detailed_design/02.3-kasmina-parameter-registration.md`
- `docs/design/detailed_design/02.4-kasmina-safety-mechanisms.md`
- `docs/design/detailed_design/02.5-kasmina-performance-validation.md`

Implementation evidence (primary):
- `src/esper/kasmina/lifecycle.py`
- `src/esper/kasmina/seed_manager.py`
- `src/esper/core/telemetry.py`
- `src/esper/security/signing.py`
- Tests: `tests/kasmina/*`, `tests/integration/test_control_loop.py`
