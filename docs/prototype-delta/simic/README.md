# Simic — Prototype Delta (Offline Policy Training)

Executive summary: the prototype implements a lightweight offline trainer (PPO‑style) and a FIFO replay buffer built from Tamiyo `FieldReport`s. It supports TTL pruning, random sampling, simple reward shaping from loss deltas/outcomes, a LoRA‑capable policy network, a minimal validation gate, and publishing `PolicyUpdate`s via Oona. The full design specifies IMPALA with V‑trace, a graph‑aware replay buffer with prioritised sampling and memory budgeting, circuit breakers and conservative mode, ingestion ack/retry semantics, and richer policy validation/versioning. Leyline remains the single source of truth for contracts.

Documents in this folder:
- `delta-matrix.md` — requirement‑by‑requirement status with evidence
- `traceability-map.md` — mapping of design assertions to code/tests
- `implementation-roadmap.md` — plan to close gaps without tech debt

Design sources:
- `docs/design/detailed_design/04-simic-unified-design.md`
- `docs/design/detailed_design/04.1-simic-rl-algorithms.md`
- `docs/design/detailed_design/04.2-simic-experience-replay.md`

Implementation evidence (primary):
- `src/esper/simic/replay.py`, `src/esper/simic/trainer.py`, `src/esper/simic/validation.py`, `src/esper/simic/registry.py`
- Tests: `tests/simic/*`
