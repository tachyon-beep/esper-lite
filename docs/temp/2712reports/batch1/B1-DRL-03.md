# Finding Ticket: Threshold Calculation Duplicated

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B1-DRL-03` |
| **Severity** | `P3` |
| **Status** | `open` |
| **Batch** | 1 |
| **Agent** | `drl` |
| **Domain** | `tolaria` |
| **Assignee** | |
| **Created** | 2024-12-27 |
| **Updated** | 2024-12-27 |

---

## Location

| Field | Value |
|-------|-------|
| **File(s)** | `/home/john/esper-lite/src/esper/tolaria/governor.py` |
| **Line(s)** | `186-187` (check_vital_signs), also in execute_rollback |
| **Function/Class** | `TolariaGovernor.check_vital_signs()`, `TolariaGovernor.execute_rollback()` |

---

## Summary

**One-line summary:** Statistical threshold calculation (`avg + sensitivity * std`) is duplicated between two methods.

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

The threshold calculation logic appears in two places. If the formula needs to change, both locations must be updated.

### Code Evidence

```python
# Appears in check_vital_signs():
threshold = avg + self.sensitivity * std

# Also appears in execute_rollback() for telemetry:
"loss_threshold": avg + self.sensitivity * std,
```

### Why This Matters

- DRY violation - formula duplicated
- Risk of divergence if one is updated but not the other
- Minor maintenance burden

---

## Recommended Fix

### Suggested Code Change

Extract to a private method:

```python
def _compute_statistical_threshold(self) -> float:
    """Compute the statistical anomaly threshold from loss history."""
    if len(self.loss_history) < 2:
        return float('inf')
    avg = sum(self.loss_history) / len(self.loss_history)
    variance = sum((x - avg) ** 2 for x in self.loss_history) / len(self.loss_history)
    std = math.sqrt(variance)
    return avg + self.sensitivity * std
```

---

## Verification

### How to Verify the Fix

- [ ] Unit test: Verify threshold calculation is consistent
- [ ] Refactor: Extract method and update both call sites

---

## Cross-Review

| Agent | Verdict | Evaluation |
|-------|---------|------------|
| **DRL** | NEUTRAL | DRY violation is a valid maintenance concern, but the threshold formula (mean + k*std) is standard statistical practice unlikely to diverge. From an RL training stability perspective, this has no impact on gradient flow, reward signals, or learning dynamics. |
| **PyTorch** | NEUTRAL | Pure Python code refactoring with no tensor operations involved - the threshold calculation uses standard Python math on a list of floats. No torch.compile, CUDA, or memory management implications; straightforward DRY improvement with no PyTorch engineering concerns. |
| **CodeReview** | NEUTRAL | The DRY violation is real but trivial--it is a one-liner formula that is semantically clear in both contexts. Extracting to a helper method adds indirection for minimal benefit; low priority unless threshold logic becomes more complex. |

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch1-drl.md`
**Section:** "T1-G-3"
