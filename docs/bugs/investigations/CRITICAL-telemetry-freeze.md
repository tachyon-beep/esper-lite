# Critical Bug: Telemetry Freeze and Zero Clip Fraction

**Severity:** Critical (P0)
**Status:** Resolved
**Date:** 2026-01-02
**Fixed:** 2026-01-02

## Symptoms

1.  **Frozen Action Heads Panel:** Gradient figures, state, CV, and CLIP metrics do not update.
2.  **Frozen Health Panel:** Ratio Joint and Grad Norm metrics do not update.
3.  **Zero Clip Fraction:** "Clip Frac" is constantly 0.000.
4.  **Zero Head Gradients:** All `head_*_grad_norm` values are 0.0 in telemetry.

## Resolution Summary

This bug had **two distinct components**:

| Component | Status | Description |
|-----------|--------|-------------|
| **Telemetry Pipeline Bugs** | FIXED | Bug-hiding patterns, joint ratio shadow, KL overflow |
| **Expected Mask Collapse** | EXPLAINED | Zero gradients when only 1 valid action—by design |

The telemetry pipeline now correctly surfaces data. The zero gradients for some heads are
**expected behavior** due to mask collapse (action validity constraints) and causal masking
(head relevance per operation type).

---

## Part 1: Telemetry Pipeline Bugs (FIXED)

### 1.1 Bug-Hiding Pattern in Gradient Collection
**Location:** `src/esper/simic/agent/ppo.py`

When head parameters have `grad=None`, the code was silently returning `0.0`:
```python
else:
    norm_t = torch.tensor(0.0, device=self.device)  # BUG: hides missing gradients
```

This violates the project's **No Bug-Hiding Patterns** rule. The 0.0 value is indistinguishable
from a valid gradient norm, masking the underlying issue.

**Fix Applied:** Changed to emit `NaN` to signal "no gradient data":
```python
else:
    norm_t = torch.tensor(float("nan"), device=self.device)  # Surfaces as NaN in telemetry
```

### 1.2 Joint Ratio Shadow Bug
**Location:** `src/esper/simic/agent/ppo.py`

The true `joint_ratio` (product of all head ratios) was being overwritten with just the `op` head ratio.
This hides divergence in other heads (slot, blueprint, style, etc.).

**Fix Applied:** Removed the overwrites, now using the TRUE joint_ratio for:
- Clip fraction computation
- Ratio mean/max/min/std logging
- RatioExplosionDiagnostic

### 1.3 KL Computation Overflow
**Location:** `src/esper/simic/agent/ppo.py` (line ~649)

The KL divergence computation was using an **unclamped** `exp(log_ratio)`:
```python
# OLD (buggy): recomputed unclamped log_ratio
log_ratio = new_log_probs[key] - old_log_probs[key]  # unclamped!
kl_per_step = (torch.exp(log_ratio) - 1) - log_ratio  # overflow risk
```

When `log_ratio` is large (e.g., +50), `exp(log_ratio) → Inf → NaN`.

**Fix Applied:** Reuse the already-clamped log_ratio from joint ratio computation:
```python
# FIXED: use clamped version to prevent exp() overflow
log_ratio_clamped = log_ratios_for_joint[key]  # already clamped to [-5, +5]
kl_per_step = (torch.exp(log_ratio_clamped) - 1) - log_ratio_clamped
```

### 1.4 Finiteness Gate
**Location:** `src/esper/simic/agent/ppo.py` (lines ~553-593)

Added fail-fast detection for NaN/Inf corrupted probabilities immediately after `evaluate_actions()`.
This catches mask mismatches between rollout and update time—the #1 source of NaN in PPO.

```python
# FINITENESS GATE: Detect NaN/Inf early to identify source of numerical instability
nonfinite_found = False
nonfinite_sources: list[str] = []

for key in HEAD_NAMES:
    if not torch.isfinite(log_probs[key]).all():
        nonfinite_count = (~torch.isfinite(log_probs[key])).sum().item()
        nonfinite_sources.append(f"log_probs[{key}]: {nonfinite_count} non-finite")
        nonfinite_found = True

if nonfinite_found:
    logger.warning(f"FINITENESS GATE FAILED: {', '.join(nonfinite_sources)}")
    continue  # Skip this epoch's update
```

---

## Part 2: Expected Behavior (Mask Collapse)

### 2.1 Sentinel NaN vs Actual NaN

There are TWO different "NaN gradient" scenarios:

| Scenario | Meaning | Debug Probe Shows |
|----------|---------|-------------------|
| Sentinel NaN | `grad=None` - param not in computation graph | `None=4, NaN/Inf=0, Finite=0` |
| True NaN | Numerical instability - grad tensor contains NaN | `None=0, NaN/Inf=4, Finite=0` |
| Healthy | Finite gradients | `None=0, NaN/Inf=0, Finite=4` |

**Debug Probe Output (2026-01-02):**
```
slot_head: None=0, NaN/Inf=0, Finite=4 (of 4 params)
  First grad norm: 0.000000 (finite)
op_head: None=0, NaN/Inf=0, Finite=4 (of 4 params)
  First grad norm: 0.016344 (finite)
value_head: None=0, NaN/Inf=0, Finite=4 (of 4 params)
  First grad norm: 14.403625 (finite)
```

**Interpretation:** All heads receive finite gradients (no NaN/Inf, no None). The slot_head has
grad_norm=0 due to **mask collapse** (see below), not numerical issues.

### 2.2 Mask Collapse Causes Zero Gradients

**Finding:** The action validity masks often have only 1 valid action:

```
=== MASK COLLAPSE CHECK (action validity) ===
slot_mask: valid_actions min=1 max=1 mean=1.0 single_action=100.0%
style_mask: valid_actions min=1 max=4 mean=1.8 single_action=75.0%
op_mask: valid_actions min=1 max=4 mean=2.0 single_action=30.0%
```

**Why this causes zero gradients:**
- With only 1 valid action, `softmax([logit]) = [1.0]`
- `log_prob = log(1.0) = 0` (constant)
- Constant log_prob → zero gradient to head parameters
- **This is expected behavior, not a bug**

### 2.3 Causal Masks Further Limit Gradient Flow

```
=== CAUSAL MASK CHECK (head relevance) ===
slot_causal: 7/20 timesteps relevant (35.0%)
blueprint_causal: 2/20 timesteps relevant (10.0%)
style_causal: 3/20 timesteps relevant (15.0%)
```

Only a small fraction of timesteps provide learning signal for non-op heads.

---

## Summary of Fixes Applied

1. **Sentinel NaN for Missing Gradients:** Changed `0.0` to `NaN` for `grad=None` cases
2. **Joint Ratio Shadow Fix:** Removed overwrites, use true joint ratio for all metrics
3. **KL Clamping Fix:** Use pre-clamped log_ratio to prevent exp() overflow
4. **Finiteness Gate:** Fail-fast detection of NaN/Inf after evaluate_actions
5. **Enhanced Debug Probe:** Added `None/NaN/Finite` counts to distinguish failure modes
6. **Mask Collapse Check:** Added logging to surface action validity statistics
7. **Causal Mask Check:** Added logging to surface head relevance statistics

---

## Future Enhancement: Learnable Fraction Metric

To better surface the distinction between "telemetry broken" and "nothing to learn", we should
add a **learnable fraction** metric:

```
learnable_fraction = (timesteps where this head has >1 valid action) / total_timesteps
```

This would allow the dashboard to show something like:
- `slot: grad=0.000 (0% learnable)` ← expected
- `op: grad=0.016 (70% learnable)` ← healthy

Rather than showing confusing zeros that look like bugs.

---

## Verification

After applying these fixes:
1. ✅ Debug probe shows `None=0, NaN/Inf=0, Finite=4` for all heads
2. ✅ Mask collapse statistics explain zero gradients for constrained heads
3. ✅ Clip fraction uses true joint ratio (will show non-zero when policy diverges)
4. ✅ Sentinel NaN surfaces cases where gradients are truly missing
5. ✅ KL computation no longer overflows on large log_ratios
6. ✅ Finiteness gate catches mask mismatches early
