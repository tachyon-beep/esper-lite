# Finding Ticket: check_gradient_drift() Defined but Never Called

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B7-DRL-03` |
| **Severity** | `P3` |
| **Status** | `closed` |
| **Batch** | 7 |
| **Agent** | `drl` |
| **Domain** | `simic/telemetry` |
| **Assignee** | |
| **Created** | 2024-12-27 |
| **Updated** | 2024-12-27 |

---

## Location

| Field | Value |
|-------|-------|
| **File(s)** | `/home/john/esper-lite/src/esper/simic/telemetry/anomaly_detector.py` |
| **Line(s)** | `262-295` |
| **Function/Class** | `AnomalyDetector.check_gradient_drift()` |

---

## Summary

**One-line summary:** Gradient drift check method exists but is excluded from `check_all()` aggregator.

**Category:**
- [ ] Correctness bug
- [ ] Race condition / concurrency
- [ ] Memory leak / resource issue
- [ ] Performance bottleneck
- [ ] Numerical stability
- [ ] torch.compile compatibility
- [x] Dead code / unwired functionality
- [x] API design / contract violation
- [ ] Test coverage gap
- [ ] Documentation / naming
- [ ] Defensive programming violation
- [ ] Legacy code policy violation

---

## Detailed Description

### What's Wrong

`AnomalyDetector.check_all()` (lines 297-347) aggregates multiple anomaly checks:
- `check_ratio_explosion()`
- `check_value_function()`
- `check_entropy_collapse()`
- `check_kl_divergence()`

But it does **NOT** include `check_gradient_drift()`.

```python
def check_all(self, ...) -> AnomalyReport:
    report = AnomalyReport()
    # ... checks ratio, EV, entropy, KL ...
    # NO call to check_gradient_drift()
    return report
```

This means:
1. Callers using `check_all()` miss gradient drift detection
2. The method exists but is orphaned
3. No clear API for when to call it separately

---

## Recommended Fix

**Option 1 - Include in check_all():**
```python
def check_all(
    self,
    ratio_mean: float,
    explained_variance: float,
    entropy: float,
    kl_divergence: float,
    norm_drift: float,     # Add parameter
    health_drift: float,   # Add parameter
    ...
) -> AnomalyReport:
    # ... existing checks ...
    drift_report = self.check_gradient_drift(norm_drift, health_drift)
    report.add(drift_report)
    return report
```

**Option 2 - Document as separate call:**
Add docstring explaining `check_gradient_drift()` is intentionally separate and when to use it.

---

## Verification

### How to Verify the Fix

- [ ] Either add to check_all() or document separation rationale
- [ ] Add test for gradient drift integration
- [ ] Verify callers receive drift anomalies

---

## Related Findings

- B7-DRL-01: GradientEMATracker never used (provides drift values)
- B7-CR-04: EMA bias correction (prerequisite)

---

## Cross-Review (DRL Specialist)

| Field | Value |
|-------|-------|
| **Verdict** | `ENDORSE` |
| **Reviewer** | DRL Specialist |

**Evaluation:** Gradient drift detection is valuable for catching slow training instability - should be wired up.

---

## Cross-Review (PyTorch Specialist)

| Field | Value |
|-------|-------|
| **Verdict** | `REFINE` |
| **Reviewer** | PyTorch Specialist |

**Evaluation:** Must confirm GradientEMATracker (which computes norm_drift/health_drift) is wired first.

---

## Cross-Review (Code Review Specialist)

| Field | Value |
|-------|-------|
| **Verdict** | `REFINE` |
| **Reviewer** | Code Review Specialist |

**Evaluation:** Bundle with B7-CR-04/B7-DRL-01 as coherent "gradient drift detection" effort. Fixing in isolation creates method that still can't be called with valid data.

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch7-drl.md`
**Section:** "check_gradient_drift() defined but never called"

**Report file:** `docs/temp/2712reports/batch7-codereview.md`
**Section:** "check_all does not pass entropy/kl to gradient drift check"

---

## Resolution

### Status: ALREADY FIXED

**Fixed in commit 322fb1d1 as part of B7-DRL-01.**

#### Evidence

| Claim | Status | Evidence |
|-------|--------|----------|
| "`check_gradient_drift()` exists at lines 262-295" | ✅ TRUE | Verified in `anomaly_detector.py:262-295` |
| "`check_all()` does NOT call `check_gradient_drift()`" | ✅ TRUE | Verified, but this is intentional design |
| "The method exists but is orphaned" | ❌ FALSE | Called in `vectorized.py:3590` |
| "Callers using `check_all()` miss gradient drift detection" | ❌ FALSE | `vectorized.py` calls it separately and merges results |

#### Why Separation from check_all() is Intentional

The `check_gradient_drift()` method is intentionally kept separate from `check_all()` because:

1. **Different data sources**: `check_all()` takes standard PPO update metrics (ratios, EV, entropy, KL). `check_gradient_drift()` requires drift metrics from `GradientEMATracker` - a separate component.

2. **Conditional availability**: The tracker may not always be present (`if grad_ema_tracker is not None`).

3. **Fixed by B7-DRL-01**: Commit `322fb1d1` wired up the gradient drift detection as part of the B7-DRL-01 fix. The code at `vectorized.py:3588-3597` calls `check_gradient_drift()` and merges results into the anomaly report.

#### Severity Confirmation

- Original: P3 (dead code / unwired functionality)
- Revised: P3 → Closed (functionality was wired up by B7-DRL-01)
- Resolution: ALREADY FIXED
