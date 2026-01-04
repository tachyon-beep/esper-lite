# Finding Ticket: Hardcoded slot_idx=0 in Heuristic Path Breaks Multi-Slot

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B8-DRL-06` |
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
| **File(s)** | `/home/john/esper-lite/src/esper/simic/training/helpers.py` |
| **Line(s)** | `311-381` |
| **Function/Class** | `_convert_flat_to_factored()` |

---

## Summary

**One-line summary:** Heuristic policy action conversion hardcodes `slot_idx=0`, breaking multi-slot support.

**Category:**
- [x] Correctness bug
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
# In _convert_flat_to_factored():
return FactoredAction(
    blueprint=blueprint,
    slot_idx=0,  # Hardcoded!
    # ...
)
```

The heuristic policy path always targets slot index 0, regardless of:
1. Which slot the seed is actually in
2. How many slots exist
3. What the heuristic policy intended to target

### Impact

- **Multi-slot broken**: Heuristic training only ever affects slot 0
- **Unfair comparison**: Heuristic vs learned policy comparison is invalid for multi-slot configs
- **Silent failure**: No error raised, just wrong behavior

---

## Recommended Fix

Pass slot information to the conversion function:

```python
def _convert_flat_to_factored(
    flat_action: FlatAction,
    slot_idx: int,  # Add parameter
    topology: SeedTopology,
) -> FactoredAction:
    # ...
    return FactoredAction(
        blueprint=blueprint,
        slot_idx=slot_idx,  # Use passed value
        # ...
    )
```

Or derive slot_idx from the action context if available.

---

## Verification

### How to Verify the Fix

- [ ] Add slot_idx parameter to _convert_flat_to_factored
- [ ] Update all callsites to pass correct slot
- [ ] Add test for multi-slot heuristic training
- [ ] Verify actions target correct slots

---

## Related Findings

- B8-CR-01: Silent NOOP fallback in same function

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch8-drl.md`
**Section:** "C8-08 - _convert_flat_to_factored() hardcodes slot_idx=0"
