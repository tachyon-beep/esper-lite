# Finding Ticket: check_gradient_drift() Defined but Never Called

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B7-DRL-03` |
| **Severity** | `P3` |
| **Status** | `open` |
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
