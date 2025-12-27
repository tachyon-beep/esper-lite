# Finding Ticket: Stabilization Edge Case - 50% Loss Increase Can Count

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B9-DRL-02` |
| **Severity** | `P1` |
| **Status** | `open` |
| **Batch** | 9 |
| **Agent** | `drl` |
| **Domain** | `tamiyo/tracker` |
| **Assignee** | |
| **Created** | 2025-12-27 |
| **Updated** | 2025-12-27 |

---

## Location

| Field | Value |
|-------|-------|
| **File(s)** | `/home/john/esper-lite/src/esper/tamiyo/tracker.py` |
| **Line(s)** | `133-137` |
| **Function/Class** | `SignalTracker._update_stabilization()` |

---

## Summary

**One-line summary:** Stabilization check allows up to 50% loss increase while still counting toward stable epochs.

**Category:**
- [x] Correctness bug
- [ ] Race condition / concurrency
- [ ] Memory leak / resource issue
- [ ] Performance bottleneck
- [ ] Numerical stability
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
# Lines 133-137 (approximate)
if loss_delta >= 0:  # Prevents regression from counting
    return  # Epoch doesn't count toward stabilization

# But the condition `val_loss < self._prev_loss * 1.5` allows up to 50% increase
```

The check `loss_delta >= 0` prevents regression epochs from counting as stable. However, there's a subtle interaction:

1. If `relative_improvement` is negative (loss increased) but `loss_delta` is exactly 0.0 (no change), the epoch still counts
2. The condition `val_loss < self._prev_loss * 1.5` allows up to 50% loss increase while still counting toward stabilization if other conditions pass

### Impact

- **Premature stabilization**: Host could be declared "stabilized" during a regression phase
- **Incorrect germination timing**: Tamiyo may germinate seeds during unstable training
- **Training instability**: Seeds germinated during host instability may fail

---

## Recommended Fix

Tighten the stabilization criteria:

```python
# Option 1: Stricter loss delta check
if loss_delta > 0:  # Any increase disqualifies (was >= 0)
    return

# Option 2: Lower the threshold
if val_loss > self._prev_loss * 1.05:  # Only 5% regression allowed
    return
```

Or document that the 50% threshold is intentional (perhaps for noisy training runs).

---

## Verification

### How to Verify the Fix

- [ ] Add test case for edge case: loss_delta = 0.0 exactly
- [ ] Add test case for edge case: loss increased by 49% (should not count as stable)
- [ ] Verify stabilization behavior in noisy training scenarios
- [ ] Document intended threshold in docstring

---

## Related Findings

- B9-CR-02: best_val_loss naming (related TrainingMetrics field)

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch9-drl.md`
**Section:** "T1 - Stabilization can trigger on regression epochs in edge case"
