# Simic — Prototype Delta (Offline Policy Training)

Executive summary: the prototype implements a lightweight offline trainer (PPO‑style) and a FIFO replay buffer built from Tamiyo `FieldReport`s. It supports TTL pruning, random sampling, simple reward shaping from loss deltas/outcomes, a LoRA‑capable policy network, a minimal validation gate, and publishing `PolicyUpdate`s via Oona. The full design specifies IMPALA with V‑trace, a graph‑aware replay buffer with prioritised sampling and memory budgeting, circuit breakers and conservative mode, ingestion ack/retry semantics, and richer policy validation/versioning. Leyline remains the single source of truth for contracts.

Outstanding Items (for coders)

- IMPALA / V‑trace path (optional for prototype)
  - Add distributed learner support with V‑trace; ensure it can be disabled; document resource expectations.
  - Pointers: `docs/design/detailed_design/04.1-simic-rl-algorithms.md`, `src/esper/simic/trainer.py`.

- Graph‑aware replay buffer
  - Extend replay to store graph features/embeddings; add prioritised sampling with importance sampling weights and memory budgeting; TTL scheduler.
  - Pointers: `src/esper/simic/replay.py`, telemetry counters for size/evictions.

- Training loop hardening
  - Add AMP + GradScaler, gradient clipping, LR scheduling via UnifiedLRController; expose training telemetry (losses, KL, entropy, LR).
  - Pointers: `src/esper/simic/trainer.py`, `docs/prototype-delta/simic/pytorch-2.8-upgrades.md`.

- Policy validation + versioning
  - Expand validation suite and thresholds; attach semantic version and hash to `PolicyUpdate`; optionally sign payloads.
  - Pointers: `src/esper/simic/validation.py`, `src/esper/simic/trainer.py` (emit updates), `esper.security.signing`.

- Oona ingestion ack/retry
  - Add ack/retry semantics when ingesting `FieldReport`s from Oona; surface dropped counts and retry totals in telemetry.
  - Pointers: Oona client consume patterns; add a small ingestion loop helper.

Documents in this folder:
- `delta-matrix.md` — requirement‑by‑requirement status with evidence
- `traceability-map.md` — mapping of design assertions to code/tests
- `implementation-roadmap.md` — plan to close gaps without tech debt
- `pytorch-2.8-upgrades.md` — mandatory training‑loop improvements (compile step, AMP+Scaler, TF32, data transfer)

Design sources:
- `docs/design/detailed_design/04-simic-unified-design.md`
- `docs/design/detailed_design/04.1-simic-rl-algorithms.md`
- `docs/design/detailed_design/04.2-simic-experience-replay.md`

Implementation evidence (primary):
- `src/esper/simic/replay.py`, `src/esper/simic/trainer.py`, `src/esper/simic/validation.py`, `src/esper/simic/registry.py`
- Tests: `tests/simic/*`
