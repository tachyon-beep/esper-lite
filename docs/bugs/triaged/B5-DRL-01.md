# Finding Ticket: Population vs Sample Variance in Shapley Computation

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B5-DRL-01` |
| **Severity** | `P2` |
| **Status** | `open` |
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
| **Line(s)** | `424-427` |
| **Function/Class** | `CounterfactualEngine._shapley_stats()` |

---

## Summary

**One-line summary:** Uses population variance (`/ len(values)`) instead of sample variance (`/ (len(values) - 1)`), underestimating uncertainty.

**Category:**
- [ ] Correctness bug
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

```python
variance = (
    sum((v - mean) ** 2 for v in values) / len(values)  # Population variance
    if len(values) > 1
    else 0.0
)
```

Uses `sum((v - mean) ** 2) / len(values)` which is population variance (N denominator), but the `ShapleyEstimate.std` is semantically a sample standard deviation used for confidence intervals.

### Impact

For 20 samples (typical), this underestimates true variance by ~5%. The `is_significant()` method uses this std for confidence intervals, making estimates appear more confident than they should be.

---

## Recommended Fix

Use Bessel's correction for sample variance:

```python
variance = (
    sum((v - mean) ** 2 for v in values) / (len(values) - 1)  # Sample variance
    if len(values) > 1
    else 0.0
)
```

---

## Verification

### How to Verify the Fix

- [ ] Change `/ len(values)` to `/ (len(values) - 1)`
- [ ] Verify is_significant() behavior still reasonable
- [ ] Consider adding property test for variance estimation

---

## Related Findings

- B5-CR-01: Unseeded RNG for Shapley (related Shapley computation issue)
- B5-DRL-06: Inconsistent variance estimators across codebase

---

## Cross-Review (DRL Specialist)

| Field | Value |
|-------|-------|
| **Verdict** | `ENDORSE` |
| **Reviewer** | DRL Specialist |

**Evaluation:** Correct statistical issue. With n=20 samples (default `shapley_samples`), population variance underestimates by factor n/(n-1) = 1.053, meaning std is underestimated by ~2.5%. The `is_significant()` method then uses this biased std for z-tests, making the test slightly anti-conservative. For seed pruning decisions that rely on statistical significance, this bias could lead to premature pruning of marginally beneficial seeds. Bessel's correction is the standard fix and has negligible computational cost.

---

## Cross-Review (PyTorch Specialist)

| Field | Value |
|-------|-------|
| **Verdict** | `ENDORSE` |
| **Reviewer** | PyTorch Specialist |
| **Date** | 2024-12-27 |

**Evaluation:** Confirmed the variance calculation at line 424 uses population variance (N denominator) when sample variance (N-1 denominator, Bessel's correction) is semantically correct for computing confidence intervals. For 20 samples (per the ticket), this underestimates variance by factor of 20/19 = 5.3%. The fix is a single-character change (`len(values)` to `len(values) - 1`) with no performance implications. Note: could alternatively use `torch.var(tensor, unbiased=True)` if converting to tensors, but the pure-Python approach is fine for these small sample counts.

---

## Cross-Review (Code Review Specialist)

| Field | Value |
|-------|-------|
| **Verdict** | `ENDORSE` |
| **Reviewer** | Code Review Specialist |

**Evaluation:** Confirmed at lines 423-427. The code computes `sum((v - mean) ** 2) / len(values)` which is population variance, but the `std` field is used in `is_significant()` for confidence intervals that assume sample statistics. Bessel's correction (`/ (len(values) - 1)`) is the standard fix. The 5% underestimate for n=20 is accurate; this creates slightly overconfident significance tests.

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch5-drl.md`
**Section:** "P2 - Performance/Subtle Bugs" (ID 5.2)
