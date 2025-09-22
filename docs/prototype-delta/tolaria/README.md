# Tolaria — Prototype Delta (Training Orchestrator)

Executive summary: the prototype provides a training loop with epoch boundary state assembly, a Tamiyo handshake (with timeout + fallback), Kasmina command application, telemetry emission, and a lightweight checkpoint/WAL for rollback. End‑of‑epoch budget enforcement, circuit‑breaker‑driven conservative mode, and PyTorch 2.8 upgrades (compile step, AMP, TF32, pinned memory/foreach) are implemented. Recent additions include a unified LR controller (constant/cosine/step with warmup), an optimizer rebuild manager with breaker guards, a two‑tier rollback (fast in‑mem snapshots + full restore with deadline), an emergency controller with L4 halt paths, seed‑aware multi‑seed gradient aggregation (registry masks + attribution split with optional PCGrad), and optional epoch‑scope profiler traces. Durability upgrades add CRC32 verification for checkpoints and WAL with atomic writes. Remaining items: cross‑process signaling/broadcast in rollback/emergency. Leyline remains the single source of truth for contracts and enums.

Outstanding Items (for coders)

- Cross‑process signaling/broadcast (rollback/emergency)
  - Shared memory primitive exists; wire it into Weatherlight broadcast and exercise end-to-end.
  - Pointers: `src/esper/tolaria/rollback.py:106` (`SharedDeadlineSignal`), `src/esper/tolaria/trainer.py:1661` (`set_shared_rollback_signal`), `src/esper/tolaria/emergency.py:38`.

- Optional: per‑layer by‑seed summaries
  - Emit compact per‑layer summaries by seed to aid Tamiyo diagnostics.
  - Keep disabled by default to avoid telemetry bloat.

- Tests & telemetry enrichment
  - Property tests for LR schedules; rollback deadline edge cases; emergency escalate/resume; add per‑feature metrics if helpful.
  - Pointers: `tests/tolaria/*` patterns.

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
- Trainer and features: `src/esper/tolaria/trainer.py`
- Aggregation: `src/esper/tolaria/aggregation.py`
- LR controller: `src/esper/tolaria/lr_controller.py`
- Optimizer manager: `src/esper/tolaria/optimizer_manager.py`
- Rollback (two‑tier + deadline): `src/esper/tolaria/rollback.py`
- Emergency controller: `src/esper/tolaria/emergency.py`
- Profiler hooks: `src/esper/tolaria/profiler.py`
- Tests: `tests/tolaria/test_tolaria_trainer.py`, `tests/integration/test_control_loop.py`,
  `tests/integration/test_profiler_trace.py`, `tests/tolaria/test_trainer_knobs.py`,
  `tests/tolaria/test_durability.py`, `tests/tolaria/test_aggregation_attribution.py`,
  `tests/tolaria/test_lr_controller.py`

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

Aggregation modes & knobs
- Modes: `TOLARIA_AGGREGATION_MODE=seed|microbatch` (default `seed`).
- Attribution: `TOLARIA_AGGREGATION_ATTRIBUTION=approx|probe|dataloader` (default `approx`).
  - `dataloader`: batches are triplets `(inputs, targets, seed_ids)` and attribution is computed from `seed_ids`.
  - `approx`: uses `Kasmina.attribute_batch(inputs, targets)` if available; otherwise proceeds without split.
  - `probe`: reserved for precise per‑seed probing (not enabled by default).
- PCGrad: `TOLARIA_PCGRAD_ENABLED=true|false` (reduces inter‑seed conflicts when enabled).

Notes
- If using gradient accumulation, you can call `advance_alpha` every N batches to match accumulation.
- α is managed on Kasmina’s side as a runtime buffer; changing it does not trigger graph retraces on the host.
- Two‑tier rollback with deadline enforcement is implemented; cross‑process signaling is deferred for the prototype.
