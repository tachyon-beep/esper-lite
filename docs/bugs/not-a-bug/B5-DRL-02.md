# Finding Ticket: EMA Variance is Biased Estimator

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B5-DRL-02` |
| **Severity** | `P2` |
| **Status** | `open` |
| **Batch** | 5 |
| **Agent** | `drl` |
| **Domain** | `simic/control` |
| **Assignee** | |
| **Created** | 2024-12-27 |
| **Updated** | 2024-12-27 |

---

## Location

| Field | Value |
|-------|-------|
| **File(s)** | `/home/john/esper-lite/src/esper/simic/control/normalization.py` |
| **Line(s)** | `93-105` |
| **Function/Class** | `RunningMeanStd.update()` (EMA path) |

---

## Summary

**One-line summary:** EMA variance weights recent observations more but doesn't account for effective sample size, may underestimate true variance.

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

The EMA variance formula is correctly implemented per the law of total variance:

```python
self.var = (
    self.momentum * self.var
    + (1 - self.momentum) * batch_var
    + self.momentum * (1 - self.momentum) * delta ** 2
)
```

However, this is a **biased estimator**:
- Weights recent observations more heavily
- Doesn't account for effective sample size
- Over very long runs, can underestimate true variance if the distribution has shifted

### Impact

In PPO training, underestimated variance leads to under-normalized observations, which can cause:
- Larger effective learning rates on some features
- Potential policy gradient instability

For typical training lengths (~100k steps), the bias is small. For very long runs or distribution shifts (e.g., after fossilization), the impact grows.

---

## Recommended Fix

Either:

1. **Document the limitation:**
```python
# NOTE: EMA variance is a biased estimator that weights recent observations
# more heavily. For very long runs or after distribution shifts, consider
# resetting statistics or using a longer momentum window.
```

2. **Add bias correction (optional, complex):**
```python
# Track effective sample size for bias correction
effective_n = 1 / (1 - self.momentum ** update_count)
unbiased_var = self.var * effective_n / (effective_n - 1)
```

Given complexity, documentation is likely sufficient.

---

## Verification

### How to Verify the Fix

- [ ] Add documentation about EMA variance bias
- [ ] Consider adding tests for EMA mode convergence

---

## Related Findings

- B5-CR-08: EMA delta variable naming
- B5-PT-01: EMA count tracking semantics

---

## Cross-Review (DRL Specialist)

| Field | Value |
|-------|-------|
| **Verdict** | `NEUTRAL` |
| **Reviewer** | DRL Specialist |

**Evaluation:** The EMA variance formula using law of total variance is mathematically correct for the streaming update. The "bias" concern is more subtle than stated: EMA intentionally weights recent data more heavily, which is a feature not a bug for non-stationary RL environments. The effective sample size argument applies to unweighted averages, not EMA. However, the documentation recommendation is valid since users should understand the trade-offs. No code change needed; a docstring clarification is sufficient.

---

## Cross-Review (PyTorch Specialist)

| Field | Value |
|-------|-------|
| **Verdict** | `NEUTRAL` |
| **Reviewer** | PyTorch Specialist |
| **Date** | 2024-12-27 |

**Evaluation:** The EMA variance implementation (lines 101-105) correctly applies the law of total variance formula. The "bias" mentioned is not a mathematical error but an intentional design choice: EMA weights recent observations more heavily, which is the desired behavior for handling non-stationary training distributions. PyTorch's `torch.nn.BatchNorm` uses similar EMA logic with `running_var` and faces the same trade-off. Documentation is reasonable but the bias correction formula in the ticket is incorrect (would need to track power decay properly). For RL observation normalization, the current implementation is standard practice per OpenAI baselines.

---

## Cross-Review (Code Review Specialist)

| Field | Value |
|-------|-------|
| **Verdict** | `NEUTRAL` |
| **Reviewer** | Code Review Specialist |

**Evaluation:** The analysis is technically correct but overstates practical impact. The EMA formula (lines 101-105) correctly applies the law of total variance. EMA bias is inherent to the algorithm and well-understood; in PPO normalization, recency-weighting is often desirable for non-stationary rewards. The recommended fix (documentation) is appropriate, but adding bias correction would introduce complexity with minimal benefit for typical training lengths.

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch5-drl.md`
**Section:** "P2 - Performance/Subtle Bugs" (ID 5.3)
