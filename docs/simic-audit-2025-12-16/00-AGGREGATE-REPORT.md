# Simic Subsystem Audit - Aggregate Report

**Date:** 2025-12-16
**Scope:** All 18 files in `src/esper/simic/`
**Auditor:** PyTorch Specialist Subagents (13 parallel audits)

---

## Executive Summary

The simic subsystem is **production-ready with targeted fixes needed**. No blocking issues prevent training, but several high-priority items risk subtle bugs or performance degradation.

| Severity | Count | Action Required |
|----------|-------|-----------------|
| CRITICAL | 1 | Fix before next major release |
| HIGH | 8 | Fix soon, risk of silent failures |
| MEDIUM | ~18 | Address during normal maintenance |
| LOW | ~25 | Opportunistic cleanup |

---

## CRITICAL Issues (P0)

### 1. Missing `collect_dual_gradients_async` Function
**File:** `gradient_collector.py`
**Lines:** N/A (missing)

The module defines `DualGradientStats` and `materialize_dual_grad_stats` but the collection function doesn't exist. The dual gradient collection is implemented **inline** in `vectorized.py` (lines 1261-1290), causing:
- Code duplication
- Inconsistent patterns vs. seed gradient collection
- Harder to test in isolation

**Fix:** Extract inline code to `collect_dual_gradients_async()` in gradient_collector.py.

---

## HIGH Priority Issues (P1)

### 1. Per-Head Advantage Masking → Gradient Starvation
**File:** `ppo.py` (Section 3.1 of report)

Causal masking in `compute_per_head_advantages` zeros out advantages for blueprint/blend heads when non-GERMINATE actions are taken. If GERMINATE is rare, these heads receive near-zero gradients, risking **policy collapse**.

**Fix:** Monitor per-head gradient magnitudes; consider entropy bonus per head.

---

### 2. Buffer/Network Hidden Dim Mismatch Risk
**File:** `ppo.py` (Section 5.1 of report)

`TamiyoRolloutBuffer` and `FactoredRecurrentActorCritic` are created with separate `lstm_hidden_dim` parameters. No validation ensures they match.

**Fix:** Add assertion in PPO `__init__` or pass single config object.

---

### 3. Global Mutable `USE_COMPILED_TRAIN_STEP` Flag
**File:** `training.py` (lines 27-72)
**Tracked:** JANK-001-P2

Module-level state affects all callers in the process. When compilation fails at import time, the flag mutates globally, creating non-deterministic behavior in multi-worker/DDP scenarios.

**Fix:** Use contextvar or pass flag explicitly.

---

### 4. `inference_mode` log_probs Misleading API
**File:** `tamiyo_network.py`

`get_action()` returns log_probs under `torch.inference_mode()` which are **non-differentiable**. Current PPO correctly uses `evaluate_actions()`, but the API is a trap for future code.

**Fix:** Return None for log_probs in get_action(), or document prominently.

---

### 5. Governor Rollback Invalidates Entire Batch
**File:** `vectorized.py`

When any single environment panics, the governor rollback invalidates the **entire batch buffer**. Safe but highly sample-inefficient.

**Fix:** Per-environment rollback tracking (architectural change).

---

### 6. `collect_seed_gradients` O(n) CUDA Syncs
**File:** `gradient_collector.py` (line 239)

When `return_enhanced=True`, iterates in Python calling `.item()` on each gradient tensor, forcing n CUDA synchronizations. Contrasts with vectorized `SeedGradientCollector.collect_async`.

**Fix:** Use `torch.stack()` + single `.tolist()` call.

---

### 7. `__init__.py` Missing Common Exports
**File:** `gradient_collector.py` + `simic/__init__.py`

Only `GradientHealthMetrics` is exported from the package. `SeedGradientCollector`, `materialize_grad_stats`, `DualGradientStats` require full import paths.

**Fix:** Add to `__all__` in `__init__.py`.

---

### 8. `_CULLABLE_STAGES` Synchronization Risk
**File:** `action_masks.py` (lines 56-64)

Hardcodes which stages allow culling, duplicating logic from `leyline/stages.py:VALID_TRANSITIONS`. If state machine evolves, silently becomes incorrect.

**Fix:** Derive programmatically from `VALID_TRANSITIONS`.

---

## MEDIUM Priority Issues (P2)

| File | Issue | Lines |
|------|-------|-------|
| ppo.py | Mixed device string vs torch.device types | various |
| ppo.py | Entropy summed across 4 heads (4x coefficient) | - |
| ppo.py | No gradient accumulation option | - |
| training.py | Device mismatch in LM path | 157 |
| training.py | Dead code `_train_one_epoch` (never called) | 96-183 |
| tamiyo_network.py | Missing dtype propagation in hidden states | - |
| tamiyo_network.py | MaskedCategorical graph breaks | - |
| tamiyo_network.py | Mask shape contract inconsistency | - |
| vectorized.py | Single GradScaler shared across devices | - |
| vectorized.py | Main function ~1900 lines | - |
| vectorized.py | Device string comparison anti-pattern | - |
| action_masks.py | `MaskedCategorical.entropy()` edge cases | - |
| action_masks.py | Many intermediate tensors in batch masks | - |
| normalization.py | Missing state_dict/load_state_dict | - |
| normalization.py | RewardNormalizer not GPU-native | - |
| gradient_collector.py | Mixed tensor/float returns in empty case | 119-126 |
| features.py | Normalization contract inconsistency | - |
| utility-files | TelemetryConfig name collision (simic vs nissa) | - |

---

## Notable Positive Findings

1. **rewards.py** - Pure Python, mathematically sound PBRS, comprehensive anti-gaming. **No issues found.**

2. **advantages.py** - Clean causal masking, correct gradient flow. **Low risk.**

3. **slots.py** - Model utility design, minimal and focused.

4. **tamiyo_buffer.py** - Correctly addresses P0 GAE interleaving bug with per-env storage.

5. **anomaly_detector.py** - Clean pure Python, phase-dependent thresholds. **Low risk.**

6. **Overall torch.compile practices** - Correct `@torch.compiler.disable` usage, proper mode selection, fused optimizer selection.

---

## Dead Code to Remove

| File | Item | Reason |
|------|------|--------|
| training.py | `_train_one_epoch` | Defined but never called |
| tamiyo_network.py | `max_entropies` dict | Unused |
| tamiyo_buffer.py | `TamiyoRolloutStep` NamedTuple | Appears unused |
| tamiyo_network.py | `import math` | Only used by dead code |

---

## Test Coverage Gaps

| File | Missing Tests |
|------|---------------|
| action_masks.py | `MaskedCategorical` direct unit tests |
| action_masks.py | `InvalidStateMachineError` handling |
| normalization.py | Checkpoint save/load (missing feature) |
| gradient_collector.py | `collect_dual_gradients` (missing function) |

---

## Recommended Fix Order

1. **P0:** Extract `collect_dual_gradients_async` from vectorized.py
2. **P1-1:** Add hidden dim validation in PPO init
3. **P1-8:** Derive `_CULLABLE_STAGES` from VALID_TRANSITIONS
4. **P1-6:** Vectorize `collect_seed_gradients` enhanced path
5. **P1-7:** Update `__init__.py` exports
6. **P2:** Remove dead code (`_train_one_epoch`, `TamiyoRolloutStep`, etc.)
7. **P2:** Resolve TelemetryConfig name collision

---

## Individual Reports

- [01-ppo.md](./01-ppo.md)
- [02-training.md](./02-training.md)
- [03-tamiyo-network.md](./03-tamiyo-network.md)
- [04-tamiyo-buffer.md](./04-tamiyo-buffer.md)
- [05-vectorized.md](./05-vectorized.md)
- [06-advantages.md](./06-advantages.md)
- [07-rewards.md](./07-rewards.md)
- [08-features.md](./08-features.md)
- [09-action-masks.md](./09-action-masks.md)
- [10-normalization.md](./10-normalization.md)
- [11-gradient-collector.md](./11-gradient-collector.md)
- [12-anomaly-detector.md](./12-anomaly-detector.md)
- [13-utility-files.md](./13-utility-files.md)
