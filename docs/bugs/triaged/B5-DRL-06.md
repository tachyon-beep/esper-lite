# Finding Ticket: Inconsistent Variance Estimators Across Codebase

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B5-DRL-06` |
| **Severity** | `P3` |
| **Status** | `open` |
| **Batch** | 5 |
| **Agent** | `drl` |
| **Domain** | `simic` |
| **Assignee** | |
| **Created** | 2024-12-27 |
| **Updated** | 2024-12-27 |

---

## Location

| Field | Value |
|-------|-------|
| **File(s)** | Multiple files |
| **Line(s)** | Various |
| **Function/Class** | Various |

---

## Summary

**One-line summary:** Three different variance computation approaches are used inconsistently across the codebase.

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

| Location | Formula | Use Case |
|----------|---------|----------|
| RunningMeanStd L117 | Welford M2 / count | Observation normalization |
| RewardNormalizer L217 | m2 / (count - 1) | Reward normalization |
| counterfactual.py L424 | sum((v-mean)^2) / len(values) | Shapley uncertainty |

The inconsistency between sample and population variance could lead to subtle issues in uncertainty quantification for Shapley estimates (they'll be slightly overconfident).

---

## Recommended Fix

Document which formula is used where and why:

```python
# VARIANCE ESTIMATION GUIDE:
#
# 1. RunningMeanStd: Uses Welford's algorithm with population variance (M2/count).
#    Appropriate for streaming data where we want the actual variance of seen data.
#
# 2. RewardNormalizer: Uses Bessel's correction (m2/(count-1)) for sample variance.
#    Appropriate for scalar rewards where we're estimating population variance.
#
# 3. Shapley: Uses population variance. Should probably use sample variance
#    for confidence intervals - see B5-DRL-01.
```

---

## Verification

### How to Verify the Fix

- [ ] Add documentation explaining variance formula choices
- [ ] Fix Shapley to use sample variance (B5-DRL-01)
- [ ] Consider unifying base statistics computation

---

## Related Findings

- B5-DRL-01: Population vs sample variance in Shapley
- B5-DRL-05: Count initialization differences

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch5-drl.md`
**Section:** "Cross-Cutting Integration Risks" (Variance Estimator Inconsistency)

---

## Cross-Review (Code Review Specialist)

| Field | Value |
|-------|-------|
| **Verdict** | `NEUTRAL` |
| **Reviewer** | Code Review Specialist |

**Evaluation:** Verified all three variance computations. The characterization is accurate but the practical impact is minimal. `RunningMeanStd` uses `m2/count` (population variance) which is appropriate for streaming normalization. `RewardNormalizer` uses `m2/(count-1)` (Bessel-corrected sample variance) which is statistically correct for scalar rewards. The Shapley computation at `counterfactual.py:424` uses population variance for uncertainty estimation, which may slightly underestimate confidence intervals for small sample sizes. This is a valid documentation request, but the "overconfident" claim should be quantified before elevating priority.

---

## Cross-Review (PyTorch Specialist)

| Field | Value |
|-------|-------|
| **Verdict** | `ENDORSE` |
| **Reviewer** | PyTorch Specialist |
| **Date** | 2024-12-27 |

**Evaluation:** The variance estimator differences are semantically justified. `RunningMeanStd` uses population variance (M2/count) because it normalizes streaming observations where we want the actual variance of seen data, not an estimate of population variance. `RewardNormalizer` uses sample variance (m2/(count-1)) for scalar rewards where Bessel's correction provides an unbiased estimator. The Shapley code using population variance for confidence intervals is the genuine issue (L424 should use `len(values) - 1` divisor), making this a P2 correctness concern rather than P3 documentation.

---

## Cross-Review (DRL Specialist)

| Field | Value |
|-------|-------|
| **Verdict** | `NEUTRAL` |
| **Reviewer** | DRL Specialist |

**Evaluation:** The inconsistency is real but arguably justified by different statistical contexts. `RunningMeanStd` uses population variance (M2/count) for observation normalization where we want the empirical variance of data seen so far; `RewardNormalizer` uses sample variance (m2/(count-1)) since scalar rewards are estimating an unknown population distribution. The Shapley case (lines 423-427) using population variance is borderline -- for confidence intervals, Bessel-corrected sample variance would be more conservative, but with n_samples typically around 20-100, the practical difference is negligible (~5% bias at n=20). Documentation is sufficient; unifying base computation would add complexity without meaningful statistical improvement.
