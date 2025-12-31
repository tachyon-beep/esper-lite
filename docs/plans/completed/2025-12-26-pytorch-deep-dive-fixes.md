# PyTorch Deep Dive Fix List

**Date:** 2025-12-26
**Branch:** `feat/tamiyo-neural-migration`
**Status:** Tier 1 & 2 Complete ✅

## Overview

Consolidated findings from 5 PyTorch specialist agent deep dives across all changed files in the Kasmina, Tamiyo, Tolaria, and Simic domains. Total: **11 HIGH**, **29 MEDIUM**, and numerous LOW severity issues identified.

Overall code quality is **excellent** - these are refinements, not fundamental problems.

---

## Tier 1: Fix Soon (Correctness/Stability) ✅ COMPLETE

These issues affect correctness or could cause silent training failures.

**All 4 fixes implemented and verified on 2025-12-26.**

### T1-1: Incorrect tensor device for LM task branch
- **File:** `src/esper/simic/training/helpers.py`
- **Lines:** 273-274
- **Severity:** HIGH
- **Issue:** `correct_batch = torch.tensor(0, device=outputs.device)` creates a scalar tensor. Should use `torch.zeros(1, device=outputs.device)` for consistency with classification branch and proper shape semantics.
- **Fix:**
  ```python
  # Before
  correct_batch = torch.tensor(0, device=outputs.device)

  # After
  correct_batch = torch.zeros(1, device=outputs.device, dtype=torch.long)
  ```

### T1-2: PBRS gamma match uses assert (can be disabled with -O)
- **File:** `src/esper/simic/rewards/rewards.py`
- **Lines:** 490-493
- **Severity:** HIGH
- **Issue:** Runtime assertion for PBRS gamma match can be disabled with Python's `-O` flag, allowing silent policy invariance violations.
- **Fix:**
  ```python
  # Before
  assert config.gamma == DEFAULT_GAMMA, "PBRS gamma must match PPO gamma"

  # After
  if config.gamma != DEFAULT_GAMMA:
      raise ValueError(
          f"PBRS gamma ({config.gamma}) must equal PPO gamma ({DEFAULT_GAMMA}) "
          "for policy invariance (Ng et al., 1999)"
      )
  ```

### T1-3: Ratio computation without numerical guard
- **File:** `src/esper/simic/agent/ppo.py`
- **Lines:** 641-642
- **Severity:** MEDIUM (upgraded from agent assessment due to silent failure mode)
- **Issue:** `torch.exp(log_probs[key] - old_log_probs[key])` can produce inf/NaN when log-prob differences are large (>88 for float32). Ratio explosion detection happens AFTER gradient damage.
- **Fix:**
  ```python
  # Before
  per_head_ratios[key] = torch.exp(log_probs[key] - old_log_probs[key])

  # After
  log_ratio = log_probs[key] - old_log_probs[key]
  log_ratio_clamped = torch.clamp(log_ratio, min=-20.0, max=20.0)
  per_head_ratios[key] = torch.exp(log_ratio_clamped)
  ```

### T1-4: initial_hidden() returns inference-mode tensors without warning
- **File:** `src/esper/tamiyo/policy/lstm_bundle.py`
- **Lines:** 258-266
- **Severity:** HIGH
- **Issue:** `@torch.inference_mode()` makes returned hidden states non-differentiable. If someone passes these to `evaluate_actions()` during PPO training, gradients won't flow. Current code works (passes `hidden=None`), but it's a footgun.
- **Fix:** Add explicit warning in docstring:
  ```python
  def initial_hidden(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
      """Get initial LSTM hidden state for rollout collection.

      WARNING: Returns inference-mode tensors that are NOT differentiable.
      For PPO training, pass hidden=None to evaluate_actions() which will
      create fresh differentiable hidden states internally.

      ...
      """
  ```

---

## Tier 2: Improve (Stability/Clarity) ✅ COMPLETE

These issues could cause subtle problems or make debugging harder.

**All 8 fixes implemented and verified on 2025-12-26.**

### T2-1: Hardcoded class count for target validation
- **File:** `src/esper/simic/training/vectorized.py`
- **Lines:** 1765-1767
- **Severity:** HIGH
- **Issue:** Target debug check assumes 10 classes (CIFAR-10). Should use `task_spec.num_classes` for task-agnostic validation.
- **Fix:** Replace hardcoded `10` with task configuration value.

### T2-2: GAE computation mixes Python scalars with tensor operations
- **File:** `src/esper/simic/agent/rollout_buffer.py`
- **Lines:** 382-408
- **Severity:** HIGH
- **Issue:** `delta + gamma * gae_lambda * next_non_terminal * last_gae` mixes Python floats with tensors. Works but is slower and can cause type promotion issues.
- **Fix:** Precompute constants as tensors:
  ```python
  gamma_t = torch.tensor(gamma, device=self.device)
  gamma_lambda = torch.tensor(gamma * gae_lambda, device=self.device)
  ```

### T2-3: Sigmoid overflow protection needed
- **File:** `src/esper/simic/rewards/rewards.py`
- **Lines:** 525-526
- **Severity:** MEDIUM
- **Issue:** `1.0 / (1.0 + math.exp(-k * x))` overflows when `k * x > 709`. For negative `total_imp`, the exp argument is positive and can overflow.
- **Fix:**
  ```python
  x = -config.attribution_sigmoid_steepness * total_imp
  attribution_discount = 1.0 / (1.0 + math.exp(min(x, 700.0)))
  ```

### T2-4: STE dtype mismatch under autocast
- **File:** `src/esper/kasmina/isolation.py`
- **Lines:** 83
- **Severity:** HIGH
- **Issue:** `seed_features - seed_features.detach()` - under autocast, the detached tensor could have a different dtype than the non-detached one.
- **Fix:**
  ```python
  def ste_forward(host_features: torch.Tensor, seed_features: torch.Tensor) -> torch.Tensor:
      if seed_features.dtype != host_features.dtype:
          seed_features = seed_features.to(host_features.dtype)
      return host_features + (seed_features - seed_features.detach())
  ```

### T2-5: BF16 backward path documentation
- **File:** `src/esper/simic/training/vectorized.py`
- **Lines:** 1424-1426
- **Severity:** HIGH
- **Issue:** When `env_state.scaler is not None`, backward uses scaled loss, otherwise raw. BF16 doesn't need scaler but the logic isn't clearly documented.
- **Fix:** Add clarifying comment explaining BF16 vs FP16 AMP paths.

### T2-6: Deprecated torch.cuda.amp import
- **File:** `src/esper/simic/training/parallel_env_state.py`
- **Lines:** 17
- **Severity:** MEDIUM
- **Issue:** `import torch.cuda.amp` is deprecated in PyTorch 2.4+. Should use `torch.amp`.
- **Fix:** Update import to `from torch.amp import GradScaler`.

### T2-7: Explained variance timing
- **File:** `src/esper/simic/agent/ppo.py`
- **Lines:** 508-513
- **Severity:** MEDIUM
- **Issue:** Explained variance is computed BEFORE updates but reported as post-update metric. This is common practice but should be documented.
- **Fix:** Add comment clarifying this measures pre-training value alignment.

### T2-8: MaskedCategorical tensor allocation in hot path
- **File:** `src/esper/tamiyo/policy/action_masks.py`
- **Lines:** 432-437
- **Severity:** MEDIUM
- **Issue:** `mask_value = torch.tensor(MASKED_LOGIT_VALUE, ...)` allocates on every `__init__`. `masked_fill` with a Python float broadcasts correctly.
- **Fix:**
  ```python
  # Before
  mask_value = torch.tensor(MASKED_LOGIT_VALUE, device=logits.device, dtype=logits.dtype)
  self.masked_logits = logits.masked_fill(~mask, mask_value)

  # After
  self.masked_logits = logits.masked_fill(~mask, MASKED_LOGIT_VALUE)
  ```

---

## Tier 3: Defer (Optimization/Cleanup)

These are optimizations that provide marginal benefit or require significant refactoring.

### T3-1: FlexAttention block mask pre-computation
- **File:** `src/esper/kasmina/blueprints/transformer.py`
- **Lines:** 232-237
- **Status:** DEFER - Already has LRU cache with 100% hit rate after warmup. Pre-computing only helps cold start.

### T3-2: Memory format consistency in Kasmina
- **Files:** `host.py:150,193`, `cnn.py:70-85`
- **Status:** DEFER - Current approach (letting host format propagate) is acceptable. Full fix requires data loader changes.

### T3-3: Hidden state cloning optimization
- **File:** `src/esper/simic/training/vectorized.py`
- **Lines:** 2345-2346, 2558-2563
- **Status:** DEFER - Use `index_copy_` instead of clone for hidden state resets. Marginal performance gain.

### T3-4: Vectorize per-slot feature extraction
- **File:** `src/esper/tamiyo/policy/features.py`
- **Lines:** 374-378, 435-488
- **Status:** DEFER - Existing TODO documents optimization path. O(slots × envs) nested loops, but runs once per rollout step.

### T3-5: Batch GPU→CPU action transfers
- **File:** `src/esper/simic/training/vectorized.py`
- **Lines:** 2376-2388
- **Status:** DEFER - Existing PERF NOTE explains tradeoff. Needs profiling to justify.

### T3-6: O(n) GPU syncs in debug telemetry
- **File:** `src/esper/simic/telemetry/debug_telemetry.py`
- **Lines:** 196-208
- **Status:** DEFER - Debug-only code. Per-layer issue identification is valuable.

### T3-7: Governor redundant detach
- **File:** `src/esper/tolaria/governor.py`
- **Lines:** 134
- **Status:** DEFER - `.detach()` is redundant before `.cpu().clone()`. Cosmetic fix.

---

## Positive Patterns Noted

The deep dives highlighted excellent patterns worth preserving:

1. **torch.compile awareness** - Correct `@torch.compiler.disable` usage, compile-friendly blend_ops.py
2. **CUDA stream management** - Proper async execution with record_stream/wait_stream in vectorized.py
3. **Deferred GPU sync** - Batch CPU transfers minimize synchronization (ppo.py:524-528)
4. **LSTM best practices** - Forget gate bias=1.0, orthogonal init, dual LayerNorm (factored_lstm.py)
5. **FP16-safe masking** - Canonical MASKED_LOGIT_VALUE from leyline
6. **Pre-allocated accumulators** - Avoid per-batch allocation churn (helpers.py:253-293)
7. **DDP symmetry documentation** - Clear requirements in slot.py docstring
8. **Non-blocking transfers with explicit sync** - governor.py:286-294

---

## Verification

After implementing fixes, run:

```bash
# Lint check
uv run ruff check src/esper/simic src/esper/kasmina src/esper/tamiyo src/esper/tolaria

# Type check
uv run mypy src/esper/simic src/esper/kasmina src/esper/tamiyo src/esper/tolaria

# Unit tests
PYTHONPATH=src uv run pytest tests/ -x -q

# Integration test (short training run)
PYTHONPATH=src uv run python -m esper.scripts.train ppo --episodes 2 --n-envs 2
```

---

## Notes

- **Tolaria dead code**: The trainer.py file has extensive dead code (duplicates vectorized.py logic). This is noted but not prioritized here—separate cleanup body of work.
- **Agent IDs for follow-up**: Kasmina (a9daa46), Tamiyo (ad1db82), Tolaria (a61901a), Simic Agent (a77b327), Simic Training (a9bb484)
