# Finding Ticket: Fused Validation Creates Persistent Alpha Schedule Side Effect

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B8-DRL-10` |
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
| **Line(s)** | `2036-2050` |
| **Function/Class** | `process_fused_val_batch()` |

---

## Summary

**One-line summary:** Fused validation creates alpha_schedule as side effect that persists beyond validation phase.

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
# Lines 2036-2050 (in fused validation)
if slot_id: 0.0 in configs and slot is TRAINING stage with GATE alpha:
    # Creates alpha_schedule if one doesn't exist!
    slot.alpha_schedule = ...
```

The fused validation phase modifies slot state (creating alpha_schedule) as a side effect. This state persists beyond the validation phase.

### Impact

- **Hidden state mutation**: Validation (read-only expectation) modifies state
- **Unexpected behavior**: Slot may behave differently after validation
- **Hard to debug**: Side effect is non-obvious

---

## Recommended Fix

**Option 1 - Use temporary alpha for validation only:**
```python
# Don't modify slot state, use local alpha for this validation pass
validation_alpha = compute_validation_alpha(slot, config)
# Use validation_alpha without storing to slot
```

**Option 2 - Document the side effect:**
```python
# NOTE: This validation pass may create alpha_schedule for TRAINING/GATE seeds.
# This is intentional to enable future blending but is a side effect.
```

---

## Verification

### How to Verify the Fix

- [ ] Determine if side effect is intentional design
- [ ] If not: use temporary alpha for validation
- [ ] If yes: document the side effect clearly

---

## Related Findings

None.

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch8-drl.md`
**Section:** "C8-26 - Fused forward alpha override semantics"
