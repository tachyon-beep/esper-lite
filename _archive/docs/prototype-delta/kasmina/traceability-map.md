# Kasmina — Traceability Map

Mapping of design assertions to implementation artefacts and tests.

| Design Assertion | Source | Implementation | Tests |
| --- | --- | --- | --- |
| Lifecycle uses Leyline enums; aligns to unified design | `docs/design/detailed_design/02-kasmina-unified-design.md` | `src/esper/kasmina/lifecycle.py` | `tests/kasmina/test_lifecycle.py` |
| Seed germination→training fast path exists | `docs/design/detailed_design/02-kasmina-unified-design.md` | `src/esper/kasmina/lifecycle.py` (allowed transitions) | `tests/kasmina/test_lifecycle.py` |
| Kernel fetch via Urza, with latency budget and fallback | `docs/design/detailed_design/02.1-kasmina-kernel-execution.md` | `src/esper/kasmina/seed_manager.py` (`_graft_seed`, `_load_fallback`) | `tests/kasmina/test_seed_manager.py` (latency/fallback), `tests/integration/test_control_loop.py` |
| GPU kernel cache reuse | `docs/design/detailed_design/02.1-kasmina-kernel-execution.md` | `src/esper/kasmina/kernel_cache.py`, `seed_manager.py::_graft_seed` | `tests/kasmina/test_seed_manager.py::test_gpu_cache_enables_reuse_between_seeds` |
| Oona prefetch request/ready flow | `docs/design/detailed_design/02.1-kasmina-kernel-execution.md` | `src/esper/kasmina/seed_manager.py`, `src/esper/kasmina/prefetch.py` | `tests/kasmina/test_seed_manager.py::test_prefetch_flow_attaches_kernel` |
| Gates evaluated and failures handled | `docs/design/detailed_design/02-kasmina-unified-design.md` | `src/esper/kasmina/gates.py`, `seed_manager.py::_ensure_gate/_handle_gate_failure` | — |
| Alpha blending with host.detach() | `docs/design/detailed_design/02.1-kasmina-kernel-execution.md` | `src/esper/kasmina/blending.py`, `seed_manager.py::blend/_handle_post_transition` | `tests/kasmina/test_lifecycle.py` (alpha asserted after transitions) |
| Mode selection owned by Tamiyo | `docs/design/detailed_design/03.4-tamiyo-integration-contracts.md` | Selection via `AdaptationCommand` annotations/params (prototype default if unspecified) | — |
| Per‑batch alpha advance during BLENDING | `docs/design/detailed_design/02.1-kasmina-kernel-execution.md` | `src/esper/kasmina/seed_manager.py::advance_alpha` (called from Tolaria each batch when stage==BLENDING) | `tests/tolaria/test_tolaria_trainer.py::test_tolaria_advances_alpha_during_blending` |
| Export seed states via Leyline messages | `docs/design/detailed_design/02-kasmina-unified-design.md` | `src/esper/kasmina/seed_manager.py` (`export_seed_states`) | Used by Tolaria state assembly (`src/esper/tolaria/trainer.py`) |
| Gradient isolation (hooks + stats) | `docs/design/detailed_design/02-kasmina-unified-design.md` | `src/esper/kasmina/isolation.py`, `seed_manager.py::_attach_kernel/isolation_stats` | `tests/kasmina/test_seed_manager.py::test_gradient_isolation_detects_overlap`, `tests/kasmina/test_seed_manager.py::test_isolation_breaker_escalates_after_repeated_violations` |
| Structured telemetry with health status | `docs/design/detailed_design/02-kasmina-unified-design.md` | `src/esper/core/telemetry.py`; emissions in `seed_manager.py` | `tests/kasmina/test_seed_manager.py` (telemetry presence) |
| Parameter registry & teacher immutability | `docs/design/detailed_design/02.3-kasmina-parameter-registration.md` | `src/esper/kasmina/registry.py`, `seed_manager.py::validate_parameters/register_teacher_model` | — |
| TTL memory caches, GC, emergency cleanup | `docs/design/detailed_design/02.2-kasmina-memory-pools.md` | `src/esper/kasmina/memory.py`, `seed_manager.py::update_epoch/_resume_seed` | `tests/kasmina/test_seed_manager.py::test_memory_gc_emits_telemetry_when_due`, `tests/kasmina/test_seed_manager.py::test_emergency_command_triggers_cleanup` |
| Security envelope (HMAC/nonce/freshness) | `docs/design/detailed_design/02-kasmina-unified-design.md` | `src/esper/kasmina/security.py`, `seed_manager.py::_verify_command` | `tests/kasmina/test_lifecycle.py` (signed commands) |
| Rollback payloads on failure/retire | `docs/design/detailed_design/02-kasmina-unified-design.md` | `seed_manager.py::_record_rollback/rollback_payload` | `tests/kasmina/test_lifecycle.py::test_seed_manager_grafts_and_retires_seed` |
| Pause/resume control path | `docs/design/detailed_design/02.4-kasmina-safety-mechanisms.md` | `src/esper/kasmina/seed_manager.py::_pause_seed/_resume_seed` | `tests/kasmina/test_seed_manager.py::test_pause_and_resume_cycle` |
| Performance validation thresholds | `docs/design/detailed_design/02.5-kasmina-performance-validation.md` | — | — |
