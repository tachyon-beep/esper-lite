# RC1 Status Tracker

| Work Package | Task | Owner | Status | Notes |
|--------------|------|-------|--------|-------|
| Risk Reduction | R1 – Async worker cancellation mitigation | Shared foundations lead | Complete | Async worker soak harness + integration suite verified 2025-09-27 |
| Risk Reduction | R2 – Strict dependency guard rollout | Module leads | Complete | Guard enforced across Tolaria/Tamiyo/Kasmina; telemetry verified 2025-09-27 |
| Risk Reduction | R3 – Telemetry routing load tests | Codex (Weatherlight owner) | Complete | Integration harness + CLI landed; metrics verified 2025-09-27 |
| Risk Reduction | R5 – Confidence gating logits export | Tamiyo lead | Complete | Tamiyo annotates blend mode; Kasmina enforces logits + telemetry (2025-09-27) |
| Risk Reduction | R4a – Tolaria complexity refactor | Tolaria lead | Complete | `_EpochRunner` runs the full training loop, legacy code removed, fixture parity guard in place, lint/static analysis green (2025-09-27). |
| Risk Reduction | R4b – Tamiyo risk engine refactor | Tamiyo lead | Complete | Evaluator pipeline owns all risk logic; `_apply_risk_engine` complexity A (3), fixtures + service tests green, flag removed 2025-09-27 |
| Risk Reduction | R4c – Kasmina command dispatcher refactor | Kasmina lead | Complete | Dispatcher + strict failures + telemetry updates shipped; docs & changelog sync finished 2025-09-28 |
| Shared Foundations | Async worker implementation | Codex | Complete | `src/esper/core/async_runner.py:1`; integration harness `tests/integration/test_async_worker_backpressure.py` green 2025-09-28 |
| Shared Foundations | Telemetry routing update | Codex | Complete | Weatherlight/Tamiyo priority routing verified via `tests/weatherlight/test_service_priority.py` + `tests/integration/test_weatherlight_tamiyo_emergency.py` |
| Shared Foundations | Dependency guard helper | Module leads | Complete | Guard live across Tolaria/Tamiyo/Kasmina (`src/esper/core/dependency_guard.py`, `src/esper/tolaria/trainer.py:2679`, `src/esper/tamiyo/service.py:2412`) |
| Tolaria WP-T1 | PCGrad rework & tests | Codex | Complete | Phases 0–4 delivered; PCGrad pairwise projection, dtype/device guards, and unit coverage landed 2025-09-28 |
| Tolaria WP-T1 | Weighted aggregation broadcast fix | Codex | Complete | Broadcast, validation guards, and trainer telemetry assertions in place 2025-09-28 |
| Tolaria WP-T2 | Async worker adoption | Codex | Complete | Shared worker + timeout telemetry shipped; tests logged in `CHANGELOG_RC1.md:108` |
| Tolaria WP-T3 | Rollback hardening | Codex | Complete | Shared worker cancellation, telemetry, and profiler hardening shipped (`tests/tolaria/test_rollback_cache.py`, `tests/tolaria/test_profiler.py`). |
| Tolaria WP-T4 | Telemetry/backcompat cleanup | Codex | Complete | `_optimizer_step`/`_finalize_epoch` helpers shipped; telemetry baselines + docs updated 2025-09-29. |
| Tolaria WP-T5 | Seed aggregation helper simplification | Codex | Complete | Seed aggregation snapshots + telemetry helpers landed; `_build_seed_metrics` now **A (2)**, `_emit_telemetry` **C (15)**, integration coverage updated 2025-09-30. |
| Tamiyo WP-A1 | Strict timeout behaviour | Codex | Complete | Evaluator refactor + timeout telemetry; complexity reduced to Radon A (`CHANGELOG_RC1.md:60`) |
| Tamiyo WP-A2 | Blend & ID validation | Codex | Complete | Strict command IDs default on; dependency guard + tests ensure missing seed/blueprint IDs fail fast (`src/esper/tamiyo/policy.py:902`, `src/esper/tamiyo/service.py:2408`). |
| Tamiyo WP-A3 | Telemetry completeness | Codex | Complete | Priority routing + coverage/verifier telemetry validated (`tests/tamiyo/test_service.py`, observability runbook) |
| Tamiyo WP-A4 | WAL/fsync & registry | Codex | Complete | WAL strict validation + backup/soak tooling; tests logged in `CHANGELOG_RC1.md:121` |
| Kasmina WP-K1 | Gate fallback enforcement | Kasmina lead | Complete | Dispatcher + strict failure telemetry shipped (`tests/kasmina/test_seed_manager.py`) |
| Kasmina WP-K2 | Blend telemetry/logits | Kasmina lead | Complete | Logit enforcement + blend telemetry verified (`tests/kasmina/test_blend_annotations.py`) |
| Kasmina WP-K3 | Command verifier telemetry, nonce ledger | Codex | Complete | Telemetry, nonce ledger metrics, and registry resets landed; remaining work tracked under WP-K4 |
| Kasmina WP-K4 | Prefetch worker & cache locking | Codex | Complete | Async worker spawn fix, integration suite stable, prefetch benchmark + docs landed 2025-09-28 |
| Testing & Validation | Performance harness execution | (tbd) | Complete | Harness replays captured for default/prefetch/no-compile (`baselines/perf/wp99_phase4_validation_*`); summary at `baselines/perf/wp99_phase4_validation_summary.json` (2025-10-02) |
| Tolaria WP-100 | Eager graph instrumentation | Codex | Complete | Graph pool reuse + warm-up fix the slow-start bug (~63 ms capture); alerts/dashboards updated, rollout guidance documented, baselines refreshed. |
| Testing & Validation | Telemetry verification runbook | Codex | Complete | Observability runbook documents emergency metrics & WAL ops (`docs/project/observability_runbook.md:121`) |
| Documentation | Update lint & change log | Codex | Complete | `CHANGELOG_RC1.md:108` and `lint_static_analysis.md:1` updated alongside WP closures |
| Cross-System | Performance reporting framework | Codex | Complete | WP-CS1 harness implemented (`scripts/run_rc1_harness.py`), CPU/GPU baselines stored (`.../wp_cs1_phase3[_gpu]/`), harness wired into CI matrix (`performance-harness`). |
| Kasmina WP-101 | Germination integration | Kasmina lead | Complete | Germination graft + telemetry shipped; soak/perf benchmarks captured; rollout guidance documented. |

