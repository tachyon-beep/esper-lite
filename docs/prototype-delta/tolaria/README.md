# Tolaria — Prototype Delta (Training Orchestrator)

Executive summary: the prototype provides a minimal training loop with epoch boundary state assembly, a Tamiyo handshake (with timeout + fallback), Kasmina command application, telemetry emission, and a lightweight checkpoint/WAL for rollback. End‑of‑epoch budget enforcement, circuit‑breaker‑driven conservative mode, and PyTorch 2.8 upgrades (compile step, AMP, TF32, pinned memory/foreach) are implemented. Outstanding items remain: unified learning‑rate controller, dynamic optimiser rebuilds, two‑tier rollback (500 ms/12 s), emergency escalation/broadcast protocol, and multi‑seed gradient aggregation. Leyline remains the single source of truth for contracts and enums.

Documents in this folder:
- `delta-matrix.md` — requirement‑by‑requirement status with evidence
- `traceability-map.md` — mapping of design assertions to code/tests
- `implementation-roadmap.md` — suggested backlog to close gaps without tech debt
- `pytorch-2.8-upgrades.md` — mandatory training‑loop improvements (compile step, AMP, TF32, data transfer)
- Profiling helper: see `scripts/profile_tolaria.py` for a ready-made loop instrumentation harness (described in `docs/project/profiling.md`).

Design sources:
- `docs/design/detailed_design/01-tolaria-unified-design.md`
- `docs/design/detailed_design/01.1-tolaria-epoch-lifecycle.md`
- `docs/design/detailed_design/01.2-tolaria-rollback-systems.md`
- `docs/design/detailed_design/01.3-tolaria-optimizer-lr.md`
- `docs/design/detailed_design/01.4-tolaria-integration-protocols.md`

Implementation evidence (primary):
- `src/esper/tolaria/trainer.py`
- Tests: `tests/tolaria/test_tolaria_trainer.py`, `tests/integration/test_control_loop.py`

Integration guide: advance Kasmina’s alpha each batch during BLENDING
- Goal: Smooth α ramp while seeds are grafting; call into Kasmina per batch.
- Where: In `TolariaTrainer._train_single_epoch()` after backward()/optimiser step.
- How:
  - Discover seeds in BLENDING using `export_seed_states()` on the Kasmina client.
  - For each, call `advance_alpha(seed_id)` once per batch (or once per accumulation step).
- Example snippet:

  ```python
  # Inside training loop after optimizer.step()
  from esper.leyline import leyline_pb2

  exporter = getattr(self._kasmina, "export_seed_states", None)
  advancer = getattr(self._kasmina, "advance_alpha", None)
  if callable(exporter) and callable(advancer):
      try:
          for seed_state in exporter():
              if seed_state.stage == leyline_pb2.SEED_STAGE_BLENDING:
                  advancer(seed_state.seed_id)
      except Exception:
          pass  # best-effort in prototype
  ```

Notes
- If using gradient accumulation, you can call `advance_alpha` every N batches to match accumulation.
- α is managed on Kasmina’s side as a runtime buffer; changing it does not trigger graph retraces on the host.
- Two-tier rollback (fast/full) and multi-seed conflict resolution remain explicitly out of scope for this prototype; see the roadmap for forward work.
