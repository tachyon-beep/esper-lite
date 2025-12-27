# Finding Ticket: get_interaction_terms Returns Empty Before Computation

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B5-DRL-10` |
| **Severity** | `P4` |
| **Status** | `open` |
| **Batch** | 5 |
| **Agent** | `drl` |
| **Domain** | `simic/attribution` |
| **Assignee** | |
| **Created** | 2024-12-27 |
| **Updated** | 2024-12-27 |

---

## Location

| Field | Value |
|-------|-------|
| **File(s)** | `/home/john/esper-lite/src/esper/simic/attribution/counterfactual_helper.py` |
| **Line(s)** | `164-168` |
| **Function/Class** | `CounterfactualHelper.get_interaction_terms()` |

---

## Summary

**One-line summary:** Returns empty dict before any computation is done, could be confusing for callers.

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

```python
def get_interaction_terms(self) -> dict[tuple[str, str], InteractionTerm]:
    if self._last_matrix is None:
        return {}  # Silently returns empty
```

Callers may not realize that an empty dict means "no computation done yet" vs "no interactions found".

---

## Recommended Fix

Either:

1. **Document the behavior:**
```python
def get_interaction_terms(self) -> dict[tuple[str, str], InteractionTerm]:
    """Get interaction terms from the last computation.

    Returns:
        Dict of (slot_a, slot_b) -> InteractionTerm.
        Returns empty dict if no computation has been performed yet.
    """
```

2. **Raise if called too early:**
```python
def get_interaction_terms(self) -> dict[tuple[str, str], InteractionTerm]:
    if self._last_matrix is None:
        raise RuntimeError(
            "get_interaction_terms() called before any computation. "
            "Call compute_contributions() first."
        )
```

---

## Verification

### How to Verify the Fix

- [ ] Add docstring or raise on premature call
- [ ] No functional change for normal usage

---

## Related Findings

None.

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch5-drl.md`
**Section:** "P4 - Style/Minor" (ID 5.12)

---

## Cross-Review (DRL Specialist)

| Field | Value |
|-------|-------|
| **Verdict** | `ENDORSE` |
| **Reviewer** | DRL Specialist |

**Evaluation:** Valid documentation concern. In RL training loops, distinguishing "no computation done" from "computation found no interactions" is important for debugging attribution issues. I favor Option 2 (raise RuntimeError) as it follows fail-fast principles and prevents silent bugs where callers assume an empty dict means "no interactions detected" when it actually means "forgot to call compute_contributions() first."

---

## Cross-Review (Code Review Specialist)

| Field | Value |
|-------|-------|
| **Verdict** | `NEUTRAL` |
| **Reviewer** | Code Review Specialist |

**Evaluation:** The finding is accurate - `get_interaction_terms()` silently returns an empty dict when `_last_matrix` is None. However, examining the actual code at line 164-168 in `counterfactual_helper.py`, the pattern `if not self._last_matrix: return {}` is consistent with the method's accessor role. The recommended docstring clarification (Option 1) is the appropriate fix; raising an exception (Option 2) would be overly aggressive for a getter method. This is a genuine documentation gap but correctly categorized as P4.

---

## Cross-Review (PyTorch Specialist)

| Field | Value |
|-------|-------|
| **Verdict** | `ENDORSE` |
| **Reviewer** | PyTorch Specialist |
| **Date** | 2024-12-27 |

**Evaluation:** The documentation concern is valid. I reviewed the actual implementation at lines 158-168 and confirm the method silently returns an empty dict when `_last_matrix` is None. From a PyTorch conventions perspective, raising RuntimeError (option 2) aligns better with how PyTorch's own APIs handle "operation requires prerequisite" scenarios (e.g., accessing `.grad` before backward). The docstring approach is acceptable but less defensive.
