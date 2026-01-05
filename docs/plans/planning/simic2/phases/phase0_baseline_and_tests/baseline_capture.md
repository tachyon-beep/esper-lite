# Phase 0 Baseline Capture (Fill In Before Refactor)

**Date:** YYYY-MM-DD  
**Branch:** `<branch>`  
**Commit:** `<git rev-parse HEAD>`

## 1) LOC / size baselines

```bash
wc -l src/esper/simic/training/vectorized.py
wc -l src/esper/simic/rewards/rewards.py
wc -l src/esper/simic/agent/ppo.py
```

Paste outputs here:

- `vectorized.py`: …
- `rewards.py`: …
- `ppo.py`: …

## 2) Reference counts (blast radius indicators)

```bash
rg -n "train_ppo_vectorized\\(" -S src/esper tests
rg -n "import esper\\.simic\\.training\\.vectorized" -S tests
rg -n "monkeypatch\\.setattr\\(vectorized" -S tests
```

Paste key hits here:

- `train_ppo_vectorized` call sites: …
- tests importing vectorized module directly: …
- tests monkeypatching vectorized internals: …

## 3) Import isolation (must remain green)

```bash
uv run pytest -q tests/test_import_isolation.py
```

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

