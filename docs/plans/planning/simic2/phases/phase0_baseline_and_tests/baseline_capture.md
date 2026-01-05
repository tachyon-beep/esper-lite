# Phase 0 Baseline Capture (Fill In Before Refactor)

**Date:** 2026-01-06  
**Branch:** `env-refactor`  
**Commit:** `11ebe0f6d1e90a0cbd0296f65e2439123eeadbff`

## 1) LOC / size baselines

```bash
wc -l src/esper/simic/training/vectorized.py
wc -l src/esper/simic/rewards/rewards.py
wc -l src/esper/simic/agent/ppo.py
```

Paste outputs here:

- `vectorized.py`: 4408
- `rewards.py`: 1871
- `ppo.py`: 1424

## 2) Reference counts (blast radius indicators)

```bash
rg -n "train_ppo_vectorized\\(" -S src/esper tests
rg -n "import esper\\.simic\\.training\\.vectorized" -S tests
rg -n "monkeypatch\\.setattr\\(vectorized" -S tests
```

Paste key hits here:

- `train_ppo_vectorized` call sites (17):
  - `src/esper/simic/agent/ppo.py:135`
  - `src/esper/simic/training/dual_ab.py:196`
  - `src/esper/simic/training/config.py:26`
  - `src/esper/simic/training/vectorized.py:21`
  - `src/esper/simic/training/vectorized.py:549`
  - `src/esper/scripts/train.py:842`
  - `tests/cuda/test_vectorized_multi_gpu_smoke.py:97`
  - `tests/integration/test_optimizer_lifecycle.py:146`
  - `tests/integration/test_phase6_regression_baseline.py:37`
  - `tests/integration/test_sparse_training.py:21`
  - `tests/integration/test_sparse_training.py:40`
  - `tests/integration/test_vectorized_determinism.py:143`
  - `tests/scripts/test_train.py:481`
  - `tests/scripts/test_train.py:606`
  - `tests/simic/test_gpu_preload_batch_size.py:31`
  - `tests/simic/test_reward_normalizer_checkpoint.py:178`
  - `tests/telemetry/test_training_metrics.py:100`
- tests importing vectorized module directly (5):
  - `tests/simic/test_gpu_preload_batch_size.py:9`
  - `tests/simic/test_reward_normalizer_checkpoint.py:120`
  - `tests/scripts/test_train.py:484`
  - `tests/scripts/test_train.py:609`
  - `tests/integration/test_optimizer_lifecycle.py:139`
- tests monkeypatching vectorized internals (7):
  - `tests/scripts/test_train.py:486` (`train_ppo_vectorized`)
  - `tests/scripts/test_train.py:611` (`train_ppo_vectorized`)
  - `tests/simic/test_gpu_preload_batch_size.py:27` (`get_hub`)
  - `tests/simic/test_reward_normalizer_checkpoint.py:157` (`get_hub`)
  - `tests/simic/test_reward_normalizer_checkpoint.py:158` (`RewardNormalizer`)
  - `tests/simic/test_reward_normalizer_checkpoint.py:159` (`SharedBatchIterator`)
  - `tests/simic/test_reward_normalizer_checkpoint.py:160` (`PPOAgent.load`)

## 3) Import isolation (must remain green)

```bash
uv run pytest -q tests/test_import_isolation.py
```

Result (ran with workspace-local UV cache: `UV_CACHE_DIR=.uv-cache`):
- ✅ `15 passed in 8.45s`

## 4) “Fast guardrail suite” (recommended per PR during Phase 1–2)

```bash
uv run pytest -q \
  tests/test_import_isolation.py \
  tests/meta/test_factored_action_contracts.py \
  tests/simic/test_vectorized.py \
  tests/simic/training/test_entropy_annealing.py \
  tests/simic/rewards/escrow/test_escrow_wiring.py \
  tests/simic/test_reward_normalizer_checkpoint.py \
  tests/simic/test_gpu_preload_batch_size.py \
  tests/scripts/test_train.py
```

Notes:
- This is intentionally “unit-ish” and should stay reasonably fast.
- Integration/cuda tests are deferred until Phase 1 is mostly complete.

Result (ran with workspace-local UV cache: `UV_CACHE_DIR=.uv-cache`):
- ✅ `117 passed, 1 warning in 11.24s`
- Warning was a CUDA init warning on a non-CUDA environment; tests still passed.

## 5) Optional: one integration sanity after Phase 1

Pick one (depending on available hardware):

```bash
# CPU-only integration-ish coverage of fused counterfactual fidelity
uv run pytest -q -m integration tests/integration/test_fused_fidelity.py
```

or (CUDA):

```bash
uv run pytest -q tests/cuda/test_vectorized_multi_gpu_smoke.py
```

## 6) Import-cycle pressure points (notes)

- `src/esper/simic/training/vectorized.py`: lazy import of `esper.runtime.get_task_spec` to avoid cycle:
  - `runtime -> simic.rewards -> simic -> simic.training -> vectorized -> runtime`
- `src/esper/simic/training/helpers.py`: same pattern (lazy `get_task_spec` import).
- `tests/test_import_isolation.py`: enforces that importing `esper.runtime` does **not** import `esper.simic.training.vectorized`.

## 7) Static checks baseline

Ran with workspace-local UV cache: `UV_CACHE_DIR=.uv-cache`:

```bash
uv run ruff check src/ tests/
uv run mypy src/
```

Result:
- ✅ `ruff`: All checks passed
- ✅ `mypy`: Success: no issues found in 159 source files

## 8) Full default test suite baseline

Ran with workspace-local UV cache: `UV_CACHE_DIR=.uv-cache`:

```bash
uv run pytest -q
```

Result:
- ✅ `4235 passed, 34 skipped, 69 deselected, 4 xfailed, 4 warnings in 107.42s`
- Skips were CUDA/MPS availability dependent; xfails are known telemetry wiring gaps.
