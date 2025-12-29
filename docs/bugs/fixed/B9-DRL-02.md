# Finding Ticket: Stabilization Edge Case - 50% Loss Increase Can Count

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B9-DRL-02` |
| **Severity** | `P1` |
| **Status** | `closed` |
| **Batch** | 9 |
| **Agent** | `drl` |
| **Domain** | `tamiyo/tracker` |
| **Assignee** | |
| **Created** | 2025-12-27 |
| **Updated** | 2025-12-29 |

---

## Resolution

**Status:** FIXED

**Root Cause:** The stabilization logic had two flaws:

1. **Too Strict:** `loss_delta >= 0` rejected ANY loss increase, even normal PPO stochastic noise (+1-2% fluctuation). This could prevent germination indefinitely in noisy training.

2. **Too Loose:** The fallback `val_loss < self._prev_loss * 1.5` allowed up to 50% regression, but this was dead code since condition 1 already rejected any increase.

**Fix Applied - Symmetric Stability Window:**

Replaced the old logic with a symmetric threshold approach:

```python
# Old (flawed):
is_stable_epoch = (
    loss_delta >= 0 and  # Too strict - rejects all noise
    relative_improvement < self.stabilization_threshold and
    val_loss < self._prev_loss * 1.5  # Dead code
)

# New (B9-DRL-02):
is_stable_epoch = (
    relative_improvement > -self.regression_threshold and  # -5%
    relative_improvement < self.stabilization_threshold    # +3%
)
```

The symmetric window:
- **Explosive growth (>3% improvement):** NOT stable (blocks germination during rapid training)
- **Divergence (>5% regression):** NOT stable (catches failing training)
- **Plateau/noise (-5% to +3%):** Stable (tolerates PPO variance)

**Verification:**
- 328 tamiyo tests pass
- 15 stabilization tests pass (6 new tests for edge cases)

---

## Location

| Field | Value |
|-------|-------|
| **File(s)** | `/home/john/esper-lite/src/esper/tamiyo/tracker.py` |
| **Line(s)** | `131-144` |
| **Function/Class** | `SignalTracker.update()` |

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

The old stabilization logic had conflicting design:

```python
# Lines 133-137 (old)
is_stable_epoch = (
    loss_delta >= 0 and  # Rejects ANY regression (too strict)
    relative_improvement < self.stabilization_threshold and
    val_loss < self._prev_loss * 1.5  # Allows 50% regression (dead code)
)
```

**Impact in PPO training:**
- Normal stochastic variance (+1-2% loss fluctuation) would reset the stable counter
- This could delay or prevent germination indefinitely
- Seeds might never get a chance to train on a "stable" host

### Impact

- **Premature stabilization blocking:** Host could be blocked from germination indefinitely due to normal noise
- **Incorrect germination timing:** Tamiyo may never allow germination in noisy training
- **Training inefficiency:** Seeds wait longer than necessary before starting

---

## Tests Added

New test class `TestSymmetricStabilityWindow` with 6 tests:

1. `test_small_regression_counts_as_stable` - 2% regression counts (PPO noise)
2. `test_large_regression_resets_counter` - 10% regression resets (divergence)
3. `test_boundary_at_five_percent_regression` - Exact boundary behavior
4. `test_unchanged_loss_counts_as_stable` - Plateau counts as stable
5. `test_stabilization_with_noise_tolerance` - Noisy but bounded epochs stabilize
6. `test_custom_regression_threshold` - Custom threshold respected

---

## Related Findings

- B9-CR-02: best_val_loss naming (related TrainingMetrics field)

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch9-drl.md`
**Section:** "T1 - Stabilization can trigger on regression epochs in edge case"
