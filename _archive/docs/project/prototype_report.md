# Esper-Lite Prototype Report (Slice Progress)

Document status: draft updated through Slice 3 (Telemetry/Observability).

## Executive Summary

Esper-Lite delivers a streamlined control loop with integrated blueprint
pipeline, observability stack, and offline learning harness. The full demo
(`scripts/run_demo.py`) exercises Tolaria → Tamiyo → Kasmina for three epochs,
streams telemetry through Oona/Nissa, kicks off Simic PPO training, and applies
policy hot reloads. Backpressure and integration tests gate regressions, and CI
now runs matrixed suites (unit, integration, contract, serialization).

- **Slices Complete:** Environment/Contracts, Control Loop Core, Blueprint
  Pipeline, Telemetry/Observability/Safety.
- **Remaining Focus:** Prototype report expansion (this document) and any
  stretch goals (e.g., performance hardening beyond profiling harness).

## Slice Progress

### Slice 0 – Environment & Contracts
- Repository scaffold, CI bootstrap (`Add Leyline Signing` commit includes
  `.github/workflows/ci.yml`).
- Leyline proto bindings with serialization tests (`tests/leyline/…`).
- Local infrastructure scripts (`infra/docker-compose.*`).

### Slice 1 – Control Loop Core
- Tolaria trainer (`src/esper/tolaria/trainer.py`) emits state/telemetry.
- Kasmina seed manager with lifecycle enforcement (`src/esper/kasmina/seed_manager.py`).
- Tamiyo service with risk gating and field reports (`src/esper/tamiyo/service.py`).
- Control-loop integration harness (`tests/integration/test_control_loop.py`) verifies end-to-end epoch timing, command routing, and Oona publishing.

### Slice 2 – Blueprint Pipeline
- Karn catalog, Tezzeret forge (`src/esper/tezzeret/…`), Urza library with WAL.
- Tamiyo blueprint metadata awareness (blueprint annotations, risk gating).
- Integration test (`tests/integration/test_blueprint_pipeline_integration.py`) confirms Tamiyo → Karn → Tezzeret → Urza path.

### Slice 3 – Telemetry, Observability, Safety
- Telemetry schema finalised; metrics documented in `docs/design/detailed_design/00-leyline.md`.
- Nissa ingest + alerts, observability runbook.
- Oona backpressure instrumentation (`src/esper/oona/messaging.py`) with metrics snapshot.
- Operator runbook (`docs/project/operator_runbook.md`) and profiling harness (`scripts/profile_tolaria.py`).
- Leyline signing optional with env-configured secret.

## Metrics & Results

| Area | Key Result | Notes |
| --- | --- | --- |
| Control Loop | Integration test ensures telemetry/command round-trips | `tests/integration/test_control_loop.py` (lin.) |
| Blueprint | Tamiyo blueprint metadata surfaces in telemetry | `tests/integration/test_blueprint_pipeline_integration.py` |
| Observability | Alerts fire via Nissa fault drills | `tests/nissa/test_drills.py` |
| Oona | Backpressure reroutes/drops tracked via metrics snapshot | `tests/oona/test_messaging.py` |
| Profiling | `scripts/profile_tolaria.py` records epoch timings | Compare against 18 ms budget |

## Dependencies & Tooling
- CI pipeline runs on GitHub Actions; see `README.md` for test matrix.
- Security: optional message signing with `ESPER_LEYLINE_SECRET`.
- Profiling: use `docs/project/profiling.md` for instructions.

## Open Questions / Risks
- Secrets loaded via environment; consider multi-secret rollovers for zero downtime.
- Performance profiling currently manual; automation could be added for release gates.
- Prototype report will need final polish (exec summary, metrics, risk table) before presentation.

## Next Steps
1. Finalise this report with metrics snapshots (insert actual numbers prior to release).
2. Prepare demo run logs/graphs (Prometheus, profiling traces) as appendices.
3. Revisit security stretch goals (secret rotation automation) if time allows.

