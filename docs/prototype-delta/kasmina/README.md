# Kasmina — Prototype Delta (Execution Layer)

Executive summary: the prototype implements a minimal Kasmina path that supports basic lifecycle progression using Leyline enums, kernel fetch via Urza, and structured telemetry. The core safety stack (gradient isolation hooks and circuit breakers), memory governance (TTL caches, KD budgeting), parameter registration/enforcement, and performance validation harness are not yet implemented. Security signing exists as a utility but is not wired into Kasmina command handling.

Documents in this folder:
- `delta-matrix.md` — requirement‑by‑requirement status with evidence
- `traceability-map.md` — mapping of design assertions to code/tests
- `implementation-roadmap.md` — suggested backlog to close gaps

Design sources:
- `docs/design/detailed_design/02-kasmina.md`
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

