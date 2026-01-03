# Finding Ticket: Hardcoded z-scores in is_significant()

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B5-DRL-03` |
| **Severity** | `P3` |
| **Status** | `wont-fix` |
| **Batch** | 5 |
| **Agent** | `drl` |
| **Domain** | `simic/attribution` |
| **Assignee** | |
| **Created** | 2024-12-27 |
| **Updated** | 2024-12-27 |

---

## Location

| Field | Value |
|-------|-------|
| **File(s)** | `/home/john/esper-lite/src/esper/simic/attribution/counterfactual.py` |
| **Line(s)** | `155-159` |
| **Function/Class** | `ShapleyEstimate.is_significant()` |

---

## Summary

**One-line summary:** Only supports 95% and 99% confidence with hardcoded z-scores (1.96, 2.58).

**Category:**
- [ ] Correctness bug
- [ ] Race condition / concurrency
- [ ] Memory leak / resource issue
- [ ] Performance bottleneck
- [ ] Numerical stability
- [ ] torch.compile compatibility
- [ ] Dead code / unwired functionality
- [ ] API design / contract violation
- [ ] Test coverage gap
- [x] Documentation / naming
- [ ] Defensive programming violation
- [ ] Legacy code policy violation

---

## Detailed Description

### What's Wrong

```python
def is_significant(self, confidence: float = 0.95) -> bool:
    """Check if the estimate is significantly different from zero."""
    if confidence == 0.99:
        z = 2.58
    else:
        z = 1.96  # 95% default
    return abs(self.value) > z * self.std
```

Issues:
1. The signature suggests arbitrary confidence levels are supported, but only 95% and 99% work
2. Magic numbers without source citation
3. Assumes normal approximation (valid for large samples, but Shapley often uses 20 samples)

---

## Recommended Fix

Either:

1. **Restrict to explicit options:**
```python
def is_significant(self, confidence: Literal[0.95, 0.99] = 0.95) -> bool:
    """Check significance at 95% or 99% confidence level.

    Uses normal approximation z-scores:
    - 95%: z = 1.96
    - 99%: z = 2.58
    """
    z = 2.58 if confidence == 0.99 else 1.96
    return abs(self.value) > z * self.std
```

2. **Use scipy for arbitrary levels (adds dependency):**
```python
from scipy.stats import norm
z = norm.ppf(1 - (1 - confidence) / 2)
```

3. **Add lookup table:**
```python
Z_SCORES = {0.90: 1.645, 0.95: 1.96, 0.99: 2.58}
```

---

## Verification

### How to Verify the Fix

- [ ] Choose approach (restrict signature or expand support)
- [ ] Add docstring explaining assumptions
- [ ] No functional change needed for basic case

---

## Related Findings

- B5-CR-05: Magic numbers for interaction thresholds

---

## Cross-Review (DRL Specialist)

| Field | Value |
|-------|-------|
| **Verdict** | `ENDORSE` |
| **Reviewer** | DRL Specialist |

**Evaluation:** Valid API design issue. The signature `confidence: float = 0.95` promises arbitrary confidence levels but delivers only two. For RL attribution, 90% confidence (z=1.645) is often more appropriate given the inherent noise in Shapley estimation with permutation sampling. The recommended fix using `Literal[0.95, 0.99]` is the cleanest solution since it makes the API honest without adding scipy dependency. The normal approximation caveat for n=20 samples is a valid secondary concern worth documenting.

---

## Cross-Review (PyTorch Specialist)

| Field | Value |
|-------|-------|
| **Verdict** | `ENDORSE` |
| **Reviewer** | PyTorch Specialist |
| **Date** | 2024-12-27 |

**Evaluation:** Valid API design issue. The method signature `is_significant(confidence: float = 0.95)` implies arbitrary confidence levels are supported, but the implementation only handles 0.95 and 0.99. The recommended fix using `Literal[0.95, 0.99]` is the cleanest approach as it documents the constraint in the type system. Avoid scipy dependency for this simple case. Note: with typical Shapley sample sizes (20-100), the normal approximation is borderline valid; for n < 30, a t-distribution would be more accurate, but this is a P4 refinement not needed for the current fix.

---

## Cross-Review (Code Review Specialist)

| Field | Value |
|-------|-------|
| **Verdict** | `ENDORSE` |
| **Reviewer** | Code Review Specialist |

**Evaluation:** Confirmed at lines 155-159. The function signature accepts any float but only handles 0.95 and 0.99 correctly; passing 0.90 would silently use 1.96 (the 95% z-score). Using `Literal[0.95, 0.99]` typing (option 1) is the cleanest fix as it makes the API contract explicit without adding scipy as a dependency. The lookup table approach is also acceptable for adding 0.90 support.

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch5-drl.md`
**Section:** "P3 - Code Quality/Maintainability" (ID 5.4)

---

## Resolution

### Status: WONTFIX

**Closed via Systematic Debugging investigation.**

#### Evidence Table

| Claim | Status | Evidence |
|-------|--------|----------|
| "Code at lines 155-159" | ❌ FALSE | Code shifted to lines 160-164 |
| "Uses `self.value` field" | ❌ FALSE | Actually uses `self.mean` field |
| "Only handles 0.95 and 0.99" | ✅ TRUE | Line 163: `z = 1.96 if confidence == 0.95 else 2.58` |
| "Passing 0.90 uses 1.96" | ❌ FALSE | Uses 2.58 (else clause = 99% z-score) |
| "API design issue exists" | ✅ TRUE | Signature accepts `float`, only 2 values work correctly |

#### Why This Is WONTFIX

1. **Single call site, default only:** The one call site (`counterfactual_helper.py:150`) calls `estimate.is_significant()` with no arguments. The bug where passing 0.90 silently uses the wrong z-score is unreachable in practice.

2. **UI-only impact:** The significance indicator is purely informational for the Shapley panel. It does not affect:
   - Seed pruning decisions
   - Rent economy signals
   - Training dynamics
   - Grafting/blending thresholds

3. **Statistical validity already approximate:** With n=20 samples (default), the normal approximation introduces ~10-15% error in tail probabilities. The difference between confidence levels is within this approximation error.

4. **Fix adds no behavioral value:** Using `Literal[0.95, 0.99]` would make the API "honest" but changes no behavior for the only call site.

#### Severity Assessment

- Original: P3 (documentation/code quality)
- Revised: P4 (cosmetic/theoretical)
- Resolution: WONTFIX - marginal value, zero practical impact
