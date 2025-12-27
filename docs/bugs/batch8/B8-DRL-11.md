# Finding Ticket: Contribution Velocity EMA Decay is Hardcoded

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B8-DRL-11` |
| **Severity** | `P3` |
| **Status** | `open` |
| **Batch** | 8 |
| **Agent** | `drl` |
| **Domain** | `simic/training` |
| **Assignee** | |
| **Created** | 2025-12-27 |
| **Updated** | 2025-12-27 |

---

## Location

| Field | Value |
|-------|-------|
| **File(s)** | `/home/john/esper-lite/src/esper/simic/training/vectorized.py` |
| **Line(s)** | `2142-2150` |
| **Function/Class** | `train_ppo_vectorized()` |

---

## Summary

**One-line summary:** Contribution velocity EMA uses fixed 0.7 decay, not configurable for different episode lengths.

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

```python
# Lines 2142-2150
# Contribution velocity uses fixed EMA decay
contribution_velocity = 0.7 * contribution_velocity + 0.3 * delta
```

The 0.7/0.3 decay values are:
1. Hardcoded, not configurable
2. Tuned for 25-epoch episodes (default)
3. May not be appropriate for other episode lengths

For short episodes (10 epochs), 0.7 decay gives too much weight to old values.
For long episodes (100 epochs), recent changes get too much weight.

### Impact

- **Suboptimal for non-default configs**: EMA window doesn't match episode length
- **Hidden coupling**: Episode length affects contribution velocity interpretation
- **Not tunable**: Can't adjust for different task dynamics

---

## Recommended Fix

Make decay configurable or compute from episode length:

```python
# Option 1: Add to TrainingConfig
contribution_velocity_decay: float = 0.7

# Option 2: Compute from episode length
# For N epochs, use decay that gives ~95% weight to last N/2 epochs
decay = 0.5 ** (2 / max_epochs)  # Half-life of N/2 epochs
```

---

## Verification

### How to Verify the Fix

- [ ] Add contribution_velocity_decay to TrainingConfig
- [ ] Or compute from max_epochs with documented formula
- [ ] Test with different episode lengths

---

## Related Findings

- B8-CR-09: Magic number for gradient EMA (similar hardcoded constant)

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch8-drl.md`
**Section:** "C8-27 - Contribution velocity EMA decay is hardcoded"
