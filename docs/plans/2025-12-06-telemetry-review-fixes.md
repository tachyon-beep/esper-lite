# Telemetry Review Fixes

**Status:** All fixes pending

---

## Fix 1: Wire NaN/Inf Detection for NUMERICAL_INSTABILITY

**Found by:** Code Reviewer
**Endorsed by:** DRL Expert
**Location:** `src/esper/simic/ppo.py:448-449`

**Current code:**
```python
has_nan = False  # TODO: Wire up actual NaN detection
has_inf = False  # TODO: Wire up actual Inf detection
```

**Problem:** NUMERICAL_INSTABILITY_DETECTED event will never fire.

**Fix:** Add actual NaN/Inf detection on loss/gradients before calling `anomaly_detector.check_all()`:
```python
# Check for numerical issues in the loss
has_nan = torch.isnan(loss).any().item() if loss is not None else False
has_inf = torch.isinf(loss).any().item() if loss is not None else False
```

**Test:** Add test that triggers NaN loss and verifies NUMERICAL_INSTABILITY_DETECTED fires.

---

## Fix 2: Add Ratio Tracking to Recurrent PPO Path

**Found by:** DRL Expert
**Location:** `src/esper/simic/ppo.py:540-645` (`update_recurrent` method)

**Problem:** The recurrent update path doesn't track `ratio_max`, `ratio_min`, `ratio_std`, meaning:
- Anomaly detection is disabled for recurrent PPO
- The PPO_UPDATE_COMPLETED event has placeholder values for these metrics

**Fix:** Add ratio tracking inside the recurrent update loop:
```python
# Track ratio statistics for anomaly detection
all_ratios = []
# ... inside the batch loop after computing ratio:
all_ratios.append(ratio.detach())

# After loop, compute statistics:
if all_ratios:
    stacked = torch.cat(all_ratios)
    metrics['ratio_max'] = [stacked.max().item()]
    metrics['ratio_min'] = [stacked.min().item()]
    metrics['ratio_std'] = [stacked.std().item()]
```

**Test:** Verify recurrent PPO emits ratio_max/min/std in metrics.

---

## Fix 3: Make Progress Thresholds Configurable

**Found by:** DRL Expert
**Location:** `src/esper/simic/vectorized.py:1152-1176`

**Problem:** Hardcoded thresholds (0.5/2.0) are scale-dependent and may not generalize to:
- Tasks with different accuracy scales (0-1 vs 0-100)
- Tasks with naturally smaller deltas (fine-tuning)

**Current code:**
```python
if abs(smoothed_delta) < 0.5:  # Plateau
    ...
elif smoothed_delta < -2.0:  # Degradation
    ...
elif smoothed_delta > 2.0:  # Improvement
    ...
```

**Fix:** Add parameters to `train_ppo_vectorized()`:
```python
def train_ppo_vectorized(
    ...
    plateau_threshold: float = 0.5,
    improvement_threshold: float = 2.0,
    ...
):
```

And use them in the detection logic:
```python
if abs(smoothed_delta) < plateau_threshold:
    ...
elif smoothed_delta < -improvement_threshold:
    ...
elif smoothed_delta > improvement_threshold:
    ...
```

**Test:** Verify custom thresholds are respected.

---

## Verification

After all fixes:
```bash
PYTHONPATH=src pytest tests/ -v --tb=short
```
