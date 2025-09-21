# Kasmina — Traceability Map

Mapping of design assertions to implementation artefacts and tests.

| Design Assertion | Source | Implementation | Tests |
| --- | --- | --- | --- |
| Lifecycle uses Leyline enums as single source of truth | `docs/design/detailed_design/02-kasmina.md` | `src/esper/kasmina/lifecycle.py` | `tests/kasmina/test_lifecycle.py`, `tests/leyline/test_enum_alignment.py` |
| Seed germination→training fast path exists | `docs/design/detailed_design/02-kasmina.md` | `src/esper/kasmina/lifecycle.py` (allowed transitions) | `tests/kasmina/test_lifecycle.py` |
| Kernel fetch via Urza, with latency budget and fallback | `docs/design/detailed_design/02.1-kasmina-kernel-execution.md` | `src/esper/kasmina/seed_manager.py` (`_graft_seed`, `_load_fallback`) | `tests/kasmina/test_seed_manager.py` (latency/fallback), `tests/integration/test_control_loop.py` |
| Gradient isolation enforced at runtime | `docs/design/detailed_design/02-kasmina.md` | `src/esper/kasmina/isolation.py`, `src/esper/kasmina/seed_manager.py` | `tests/kasmina/test_seed_manager.py::test_isolation_stats_capture_detached_blend` |
| Parameter registration and update validation | `docs/design/detailed_design/02.3-kasmina-parameter-registration.md` | `src/esper/kasmina/registry.py`, `src/esper/kasmina/seed_manager.py` | `tests/kasmina/test_seed_manager.py::test_parameter_registry_blocks_duplicate_kernel`, `tests/kasmina/test_seed_manager.py::test_register_teacher_model_blocks_updates` |
| Memory TTL caches and epoch GC | `docs/design/detailed_design/02.2-kasmina-memory-pools.md` | `src/esper/kasmina/memory.py`, `src/esper/kasmina/seed_manager.py` | `tests/kasmina/test_seed_manager.py::test_seed_manager_emits_telemetry_for_commands` |
| Circuit breaker and monotonic timers | `docs/design/detailed_design/02.4-kasmina-safety-mechanisms.md` | `src/esper/kasmina/safety.py`, `src/esper/kasmina/seed_manager.py` | `tests/kasmina/test_safety.py` |
| HMAC + nonce + freshness on commands | `docs/design/detailed_design/02-kasmina.md` | `src/esper/kasmina/security.py`, `src/esper/kasmina/seed_manager.py` | `tests/kasmina/test_seed_manager.py::test_manager_rejects_unsigned_command`, `tests/integration/test_control_loop.py` |
| Structured telemetry with health status | `docs/design/detailed_design/02-kasmina.md` | `src/esper/core/telemetry.py`; emissions in `seed_manager.py` (priority indicators, rollback events) | `tests/kasmina/test_seed_manager.py` (telemetry presence) |
| Rollback payloads recorded on transitions | `docs/design/detailed_design/02-kasmina.md` | `src/esper/kasmina/seed_manager.py` (`_record_rollback`) | `tests/kasmina/test_lifecycle.py::test_seed_manager_grafts_and_retires_seed`, `tests/kasmina/test_seed_manager.py::test_parameter_registry_blocks_duplicate_kernel` |
| Epoch coordination surfaced in exports | `docs/design/detailed_design/02-kasmina.md` | `src/esper/kasmina/seed_manager.py` (`update_epoch`, `export_seed_states`) | `tests/kasmina/test_seed_manager.py::test_update_epoch_reflected_in_export` |
| Teacher registration ready for KD | `docs/design/detailed_design/02.1-kasmina-kernel-execution.md` | `src/esper/kasmina/seed_manager.py` (`register_teacher_model`) | `tests/kasmina/test_seed_manager.py::test_register_teacher_model_blocks_updates` |
| Performance validation thresholds | `docs/design/detailed_design/02.5-kasmina-performance-validation.md` | — | — |
