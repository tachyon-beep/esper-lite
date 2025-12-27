# Finding Ticket: Anomaly Samples Not Added to History During Consecutive Panics

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B1-DRL-02` |
| **Severity** | `P2` |
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
| **Line(s)** | `200-209` |
| **Function/Class** | `TolariaGovernor.check_vital_signs()` |

---

## Summary

**One-line summary:** When anomaly is detected, current loss is not added to history, causing history to become stale during consecutive panics.

**Category:**
- [ ] Correctness bug
- [ ] Race condition / concurrency
- [ ] Memory leak / resource issue
- [ ] Performance bottleneck
- [ ] Numerical stability
- [ ] torch.compile compatibility
- [ ] Dead code / unwired functionality
- [x] API design / contract violation
- [ ] Test coverage gap
- [ ] Documentation / naming
- [ ] Defensive programming violation
- [ ] Legacy code policy violation

---

## Detailed Description

### What's Wrong

When `is_anomaly` is True, the current loss is NOT added to `loss_history`. The history append only happens in the `else` branch (line 209). This means during consecutive anomalies, the history remains stale.

### Code Evidence

```python
# /home/john/esper-lite/src/esper/tolaria/governor.py:200-209

if is_anomaly:
    self.consecutive_panics += 1
    # Note: loss is NOT added to history here
    if self.consecutive_panics >= self.panic_threshold:
        return GovernorReport(rollback_required=True, ...)
    return GovernorReport(rollback_required=False, ...)
else:
    self.consecutive_panics = 0
    self.loss_history.append(loss)  # Only added when NOT anomaly
```

### Why This Matters

**Scenario:** Agent causes several consecutive borderline anomalies (panics) but recovers before rollback threshold:

1. Normal training: history = [1.0, 1.1, 0.9, 1.0]
2. Panic 1 (loss=5.0): consecutive_panics=1, history unchanged
3. Panic 2 (loss=4.5): consecutive_panics=2, history unchanged
4. Recovery (loss=1.2): consecutive_panics=0, history = [1.0, 1.1, 0.9, 1.0, 1.2]

The anomalous losses (5.0, 4.5) are never recorded. This is arguably intentional (don't corrupt history with outliers), but could cause:

- Lingering sensitivity: thresholds remain based on old "good" data
- No record of near-misses for telemetry/debugging
- If agent oscillates between anomaly and normal, history may never update

---

## System Context

The history is used to compute statistical thresholds (mean + N*std). Keeping it "clean" of anomalies might be intentional to maintain stable thresholds.

---

## Recommended Fix

### Option A: Document as intentional (Recommended if correct)

Add a comment explaining why anomaly losses are excluded:
```python
if is_anomaly:
    self.consecutive_panics += 1
    # Intentionally NOT adding anomaly loss to history to keep
    # statistical thresholds based on "healthy" training samples
```

### Option B: Track anomaly history separately

```python
if is_anomaly:
    self.consecutive_panics += 1
    self.anomaly_history.append(loss)  # Separate tracking for debugging
else:
    self.consecutive_panics = 0
    self.loss_history.append(loss)
```

### Option C: Add with flag for post-mortem analysis

```python
self.loss_history.append((loss, is_anomaly))  # Include all, flag anomalies
```

---

## Verification

### How to Verify the Fix

- [ ] Unit test: Verify history behavior during consecutive panics
- [ ] Manual: Check if current behavior is intentional

### Test Files to Update

- `tests/tolaria/test_governor.py` - add test for history during consecutive panics

---

## Cross-Review

| Agent | Verdict | Evaluation |
|-------|---------|------------|
| **DRL** | ENDORSE | Excluding anomaly losses from history is actually correct behavior for robust threshold estimation (contaminated statistics would inflate variance, triggering fewer alarms). Option A (document as intentional) is sufficient; however, Option B (separate anomaly_history) adds value for post-mortem debugging without corrupting the adaptive thresholds used for anomaly detection. |
| **PyTorch** | NEUTRAL | Statistical history tracking for anomaly detection is pure Python/RL logic with no tensor operations or compile implications. The fix options (documenting behavior or adding separate history) have no impact on CUDA memory, torch.compile, or gradient flow. |
| **CodeReview** | NEUTRAL | The observation is accurate but the current behavior is defensible as intentional - anomaly losses SHOULD be excluded to keep thresholds stable. Option A (add clarifying comment) is the right fix; this is a documentation gap, not a bug. Re-categorize from "API design" to "Documentation". |

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch1-drl.md`
**Section:** "T1-G-2"
