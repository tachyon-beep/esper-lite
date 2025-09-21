# Kasmina — Traceability Map

Mapping of design assertions to implementation artefacts and tests.

| Design Assertion | Source | Implementation | Tests |
| --- | --- | --- | --- |
| Lifecycle uses Leyline enums; aligns to unified design | `docs/design/detailed_design/02-kasmina-unified-design.md` | `src/esper/kasmina/lifecycle.py` | `tests/kasmina/test_lifecycle.py` |
| Seed germination→training fast path exists | `docs/design/detailed_design/02-kasmina-unified-design.md` | `src/esper/kasmina/lifecycle.py` (allowed transitions) | `tests/kasmina/test_lifecycle.py` |
| Kernel fetch via Urza, with latency budget and fallback | `docs/design/detailed_design/02.1-kasmina-kernel-execution.md` | `src/esper/kasmina/seed_manager.py` (`_graft_seed`, `_load_fallback`) | `tests/kasmina/test_seed_manager.py` (latency/fallback), `tests/integration/test_control_loop.py` |
| Gates evaluated and failures handled | `docs/design/detailed_design/02-kasmina-unified-design.md` | `src/esper/kasmina/gates.py`, `seed_manager.py::_ensure_gate/_handle_gate_failure` | — |
| Alpha blending with host.detach() | `docs/design/detailed_design/02.1-kasmina-kernel-execution.md` | `src/esper/kasmina/blending.py`, `seed_manager.py::blend/_handle_post_transition` | `tests/kasmina/test_lifecycle.py` (alpha asserted after transitions) |
| Export seed states via Leyline messages | `docs/design/detailed_design/02-kasmina-unified-design.md` | `src/esper/kasmina/seed_manager.py` (`export_seed_states`) | Used by Tolaria state assembly (`src/esper/tolaria/trainer.py`) |
| Gradient isolation (hooks + stats) | `docs/design/detailed_design/02-kasmina-unified-design.md` | `src/esper/kasmina/isolation.py`, `seed_manager.py::_attach_kernel/isolation_stats` | `tests/kasmina/test_seed_manager.py::test_gradient_isolation_detects_overlap` (sanity) |
| Structured telemetry with health status | `docs/design/detailed_design/02-kasmina-unified-design.md` | `src/esper/core/telemetry.py`; emissions in `seed_manager.py` | `tests/kasmina/test_seed_manager.py` (telemetry presence) |
| Parameter registry & teacher immutability | `docs/design/detailed_design/02.3-kasmina-parameter-registration.md` | `src/esper/kasmina/registry.py`, `seed_manager.py::validate_parameters/register_teacher_model` | — |
| TTL memory caches (kernel/telemetry) | `docs/design/detailed_design/02.2-kasmina-memory-pools.md` | `src/esper/kasmina/memory.py`, `seed_manager.py::_emit_telemetry/_attach_kernel` | — |
| Security envelope (HMAC/nonce/freshness) | `docs/design/detailed_design/02-kasmina-unified-design.md` | `src/esper/kasmina/security.py`, `seed_manager.py::_verify_command` | `tests/kasmina/test_lifecycle.py` (signed commands) |
| Rollback payloads on failure/retire | `docs/design/detailed_design/02-kasmina-unified-design.md` | `seed_manager.py::_record_rollback/rollback_payload` | `tests/kasmina/test_lifecycle.py::test_seed_manager_grafts_and_retires_seed` |
| Performance validation thresholds | `docs/design/detailed_design/02.5-kasmina-performance-validation.md` | — | — |
