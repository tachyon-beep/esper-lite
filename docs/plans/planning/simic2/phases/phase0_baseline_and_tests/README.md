# Phase 0: Baseline + Refactor Safety Rails

**Intent:** Make behavior regressions visible before we start moving code.

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

### Add pure unit tests (no GPU required)

Target: tests that lock down semantics of logic we plan to extract out of `vectorized.py`.

- `entropy_anneal_steps` semantics:
  - `_calculate_entropy_anneal_steps(entropy_anneal_episodes, n_envs, ppo_updates_per_batch)`
  - confirm `ceil(entropy_anneal_episodes / n_envs) * max(1, ppo_updates_per_batch)`
- Action validity rules (unit-level):
  - Given `op`, slot enabled/disabled, and seed stage, ensure validity matches current rules.
  - Keep this test at the “decision policy contract” level (don’t require a real model).

### Validation commands (run per PR)

- `uv run pytest -q`
- `ruff check src/ tests/`
- `mypy src/`

## Done means

- Baseline artifacts are recorded in the PR description or phase notes.
- We have at least 2–3 pure tests that will fail on accidental semantic drift during extraction.

