# Finding Ticket: Action Validity Checked Twice (Parse and Execution)

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B8-DRL-08` |
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
| **Line(s)** | `1302-1304, 2524-2817, 2756-2757` |
| **Function/Class** | `_parse_sampled_action()`, action execution block |

---

## Summary

**One-line summary:** Action preconditions checked in parse phase and again in execution phase, creating code duplication.

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
# In _parse_sampled_action() (lines 1302-1304):
if seed_age < MIN_PRUNE_AGE:
    # Mark action as invalid for reward

# In action execution block (lines 2756-2757):
if seed_age < MIN_PRUNE_AGE:
    action_success = False  # Duplicate check!
```

Similar duplication exists for other preconditions (slot occupancy, seed stage, etc.).

### Rationale

The code is defensive against state changes between action selection and execution. However, this creates:
1. **Maintenance burden**: Two places to update if threshold changes
2. **Code duplication**: Same logic repeated
3. **Inconsistency risk**: If one is updated but not the other

### Impact

- **Correctness maintained** (the duplication is intentional defense)
- **Maintenance burden** increases
- **Threshold drift** possible if one location is updated but not the other

---

## Recommended Fix

**Option 1 - Single source of truth:**
Extract precondition checks to shared functions:

```python
def can_prune(seed_age: int) -> bool:
    return seed_age >= MIN_PRUNE_AGE

# Use in both parse and execution:
if not can_prune(seed_age):
    ...
```

**Option 2 - Document the intentional duplication:**
Add a comment explaining why both checks exist and that they must stay in sync.

---

## Verification

### How to Verify the Fix

- [ ] Extract precondition checks to shared functions
- [ ] Update both parse and execution to use shared functions
- [ ] Add test that verifies consistency

---

## Related Findings

- C8-23 in DRL report: MIN_PRUNE_AGE check duplicated

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch8-drl.md`
**Section:** "C8-21 - Action validity vs action success mismatch" and "C8-23 - PRUNE age check duplication"
