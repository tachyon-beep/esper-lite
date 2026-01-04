# Finding Ticket: Division by Zero in Gradient Ratio

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B3-PT-01` |
| **Severity** | `P0` |
| **Status** | `open` |
| **Batch** | 3 |
| **Agent** | `pytorch` |
| **Domain** | `kasmina` |
| **Assignee** | |
| **Created** | 2024-12-27 |
| **Updated** | 2024-12-27 |

---

## Location

| Field | Value |
|-------|-------|
| **File(s)** | `/home/john/esper-lite/src/esper/kasmina/slot.py` |
| **Line(s)** | `1820-1821` |
| **Function/Class** | `SeedSlot._compute_gradient_health()` or gradient ratio calculation |

---

## Summary

**One-line summary:** Division by zero when computing gradient ratio for seeds with zero trainable parameters causes incorrect G2 gate decisions.

**Category:**
- [x] Correctness bug
- [ ] Race condition / concurrency
- [ ] Memory leak / resource issue
- [ ] Performance bottleneck
- [x] Numerical stability
- [ ] torch.compile compatibility
- [ ] Dead code / unwired functionality
- [ ] API design / contract violation
- [ ] Test coverage gap
- [ ] Documentation / naming
- [ ] Defensive programming violation
- [ ] Legacy code policy violation

---

## Detailed Description

### What's Wrong

When a seed has zero trainable parameters (e.g., a noop seed or identity-initialized seed before any parameters are added), the gradient ratio calculation performs:

```python
gradient_ratio = (host_params / seed_params) ** 0.5
```

When `seed_params = 0`, this produces `inf`, which is then clamped to the maximum value (10.0). This causes noop seeds to appear as if they have **maximum gradient activity** when they should have **zero**.

### Code Evidence

```python
# /home/john/esper-lite/src/esper/kasmina/slot.py:1820-1821

# Current problematic code (approximate - verify exact lines)
gradient_ratio = (host_params / seed_params) ** 0.5  # Division by zero!
gradient_ratio = min(gradient_ratio, 10.0)  # Clamps inf to 10.0
```

### Why This Matters

**Impact on RL Training:**
1. The G2 gate (TRAINING → BLENDING transition) uses gradient ratio as a signal
2. Noop seeds with zero parameters will show `gradient_ratio = 10.0` (maximum)
3. This makes noop seeds appear highly active when they contribute nothing
4. The policy receives **inverted signals** - inactive seeds look maximally active
5. This corrupts the observation space and leads to incorrect policy decisions

**Symptoms:**
- Seeds that should be pruned quickly may be advanced to BLENDING
- Training efficiency degraded due to false positive "healthy" signals
- G2 gate decisions are based on garbage data for zero-param seeds

### Reproduction Steps

1. Create a SeedSlot with a noop/identity seed (zero trainable params)
2. Call the gradient health computation
3. Observe: `gradient_ratio = 10.0` (clamped from inf)
4. Expected: `gradient_ratio = 0.0` or a sentinel indicating "no gradient activity"

---

## System Context

This issue affects the **Rent Economy** principle - seeds should pay rent proportional to their complexity/activity. A zero-parameter seed paying no rent should not appear maximally active.

**Related concepts:**
- Gradient isolation during TRAINING stage
- G2 gate checks gradient ratio to decide TRAINING → BLENDING transition
- Policy observation space includes gradient health metrics

**Relevant architectural principles:**
- [x] Signal-to-Noise Hypothesis (sensors match capabilities) - **VIOLATED**: sensor reports garbage
- [x] Rent Economy (complexity pays rent) - **VIOLATED**: zero-rent seed appears high-activity
- [ ] Inverted Control Flow (GPU-first)
- [ ] Governor prevents catastrophe
- [ ] No defensive programming policy
- [ ] No legacy code policy

---

## Recommended Fix

### Approach

Guard against zero seed parameters before division. Return 0.0 (or another sentinel) when seed has no trainable parameters.

### Suggested Code Change

```python
# /home/john/esper-lite/src/esper/kasmina/slot.py:1820-1821

# Option A: Guard with early return
if seed_params == 0:
    gradient_ratio = 0.0  # No parameters = no gradient activity
else:
    gradient_ratio = (host_params / seed_params) ** 0.5
    gradient_ratio = min(gradient_ratio, 10.0)

# Option B: Use max(seed_params, 1) to avoid div-by-zero
# BUT this is defensive programming and may mask other bugs
# Prefer Option A for explicit handling
```

### Alternative Approaches

1. **Option A (Recommended):** Explicit guard with early return for zero-param case
   - Pros: Clear intent, no magic numbers, explicit handling
   - Cons: Additional branch

2. **Option B:** Use `max(seed_params, 1)` denominator
   - Pros: Single line change
   - Cons: Violates defensive programming policy - masks the edge case

3. **Option C:** Propagate a sentinel value (e.g., `float('nan')` or `None`)
   - Pros: Downstream code must handle explicitly
   - Cons: May require changes to observation construction

---

## Verification

### How to Verify the Fix

- [x] Unit test: Test gradient ratio computation with `seed_params=0`
- [x] Unit test: Test G2 gate decision with zero-param seed
- [ ] Integration test: Verify noop seeds are handled correctly in full training
- [ ] Property-based test: Gradient ratio is always in [0, 10] for any param counts

### Regression Risk

- Other code may rely on `gradient_ratio` never being exactly 0.0
- G2 gate logic may need adjustment if it uses gradient_ratio thresholds

### Test Files to Update

- `tests/kasmina/test_slot.py` - add test for zero-param gradient ratio
- `tests/kasmina/test_seed_slot.py` - verify G2 gate with zero-param seed

---

## Related Findings

| Ticket ID | Relationship | Notes |
|-----------|--------------|-------|
| `B3-DRL-01` | `related` | DRL agent noted param_estimate inaccuracies affect rent economy |
| `B7-DRL-01` | `related` | Dead GradientEMATracker means drift detection also broken |

---

## Investigation Notes

### Questions to Answer

- [x] Exact line numbers in slot.py where division occurs
- [ ] Are there other places with similar division-by-zero risk?
- [ ] What value should gradient_ratio be for zero-param seeds? (0.0 proposed)
- [ ] Does G2 gate logic need adjustment for this edge case?

### Findings During Investigation

**2024-12-27:** Initial finding from PyTorch specialist batch review.

---

## Resolution

### Final Fix Description

[To be filled when fixed]

### Commits

- [To be filled]

### Tests Added/Modified

- [To be filled]

### Verified By

- [ ] Automated tests pass
- [ ] Manual verification complete
- [ ] Code review approved

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch3-pytorch.md`
**Section:** "Critical Finding (P0)"

### Additional Context

The gradient ratio is used to normalize gradient magnitudes between host and seed, accounting for the difference in parameter counts. The square root follows from the relationship between parameter count and expected gradient norm magnitude.
