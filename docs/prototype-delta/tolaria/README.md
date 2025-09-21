# Tolaria — Prototype Delta (Training Orchestrator)

Executive summary: the prototype provides a minimal training loop with epoch boundary state assembly, a Tamiyo handshake, Kasmina command application, telemetry emission, and a lightweight checkpoint/WAL for rollback. The unified learning‑rate controller, dynamic optimiser rebuilds, two‑tier rollback (500 ms/12 s), circuit‑breaker‑driven conservative mode, multi‑seed gradient aggregation, and strict end‑of‑epoch timing/timeout enforcement from the design are not yet implemented. Leyline remains the single source of truth for contracts and enums.

Documents in this folder:
- `delta-matrix.md` — requirement‑by‑requirement status with evidence
- `traceability-map.md` — mapping of design assertions to code/tests
- `implementation-roadmap.md` — suggested backlog to close gaps without tech debt
- `pytorch-2.8-upgrades.md` — mandatory training‑loop improvements (compile step, AMP, TF32, data transfer)

Design sources:
- `docs/design/detailed_design/01-tolaria-unified-design.md`
- `docs/design/detailed_design/01.1-tolaria-epoch-lifecycle.md`
- `docs/design/detailed_design/01.2-tolaria-rollback-systems.md`
- `docs/design/detailed_design/01.3-tolaria-optimizer-lr.md`
- `docs/design/detailed_design/01.4-tolaria-integration-protocols.md`

Implementation evidence (primary):
- `src/esper/tolaria/trainer.py`
- Tests: `tests/tolaria/test_tolaria_trainer.py`, `tests/integration/test_control_loop.py`
