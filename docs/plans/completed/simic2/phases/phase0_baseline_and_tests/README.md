# Phase 0: Baseline + Refactor Safety Rails

**Intent:** Make behavior regressions visible before we start moving code.

## What already exists (guardrails we must preserve)

These tests already lock down key vectorized-training contracts and are the primary safety rails for the refactor:

- Import isolation (prevents accidental heavy imports / cycles):
  - `tests/test_import_isolation.py`
- Public surface contracts:
  - `tests/meta/test_factored_action_contracts.py` (includes `train_ppo_vectorized` signature checks)
  - `tests/scripts/test_train.py` (CLI → `train_ppo_vectorized` kwargs mapping)
- Vectorized helper semantics:
  - `tests/simic/test_vectorized.py` (covers `_run_ppo_updates`, `_resolve_target_slot`, telemetry emission helpers, anomaly logic)
  - `tests/simic/training/test_entropy_annealing.py` (documents annealing semantics)
  - `tests/simic/rewards/escrow/test_escrow_wiring.py` (covers `_resolve_target_slot` and escrow edge cases)
  - `tests/simic/test_reward_normalizer_checkpoint.py` (resume metadata and monkeypatch seams)

Phase 0 focuses on *baseline capture* and identifying *fragile seams* before we start moving code.

## Checklist

### Baselines (record before refactor)

- Capture file sizes/LOC:
  - `wc -l src/esper/simic/training/vectorized.py`
  - `wc -l src/esper/simic/rewards/rewards.py`
  - `wc -l src/esper/simic/agent/ppo.py`
- Capture reference counts for key entrypoints (for post-refactor grep validation):
  - `rg -n "train_ppo_vectorized\\(" -S src/esper`
  - `rg -n "VectorizedEmitter" -S src/esper`
- Capture import-cycle pressure points (qualitative, but write them down):
  - lazy import sites for `get_task_spec`
  - any “import inside function to avoid circular import” notes

Use `docs/plans/planning/simic2/phases/phase0_baseline_and_tests/baseline_capture.md` as the capture template.

### Add pure unit tests (no GPU required)

Target: tests that lock down semantics of logic we plan to extract out of `vectorized.py`.

Already covered:
- `tests/simic/training/test_entropy_annealing.py` (anneal semantics)
- `tests/simic/test_vectorized.py` (helper semantics)

Potential additions (only if we hit gaps during Phase 1 extraction):
- A minimal “action validity” contract test that does *not* require a real model, once the action decode logic becomes a module-level pure function.

### Validation commands (run per PR)

- `uv run pytest -q`
- `ruff check src/ tests/`
- `mypy src/`

## Done means

- Baseline artifacts are recorded in the PR description or phase notes.
- We have a clear list of “fragile seams” (tests that monkeypatch `esper.simic.training.vectorized` internals) so Phase 1 doesn’t surprise-break unrelated tests.
