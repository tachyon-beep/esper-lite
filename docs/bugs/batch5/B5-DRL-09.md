# Finding Ticket: Inconsistent Division Operators in Shapley

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B5-DRL-09` |
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
| **File(s)** | `/home/john/esper-lite/src/esper/simic/attribution/counterfactual.py` |
| **Line(s)** | `351, 352` |
| **Function/Class** | `CounterfactualEngine._generate_shapley_configs()` |

---

## Summary

**One-line summary:** Inconsistent `n_samples / 2` vs `n_samples // 2` for integer division.

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
# Line 351
something = n_samples / 2  # Float division

# Line 352
for _ in range(n_samples // 2):  # Integer division
```

Both work correctly (the float gets truncated by `range()`), but the inconsistency is style noise.

---

## Recommended Fix

Use consistent integer division:

```python
half_samples = n_samples // 2
for _ in range(half_samples):
    ...
```

---

## Verification

### How to Verify the Fix

- [ ] Standardize on `//` for integer division
- [ ] No functional change

---

## Related Findings

None.

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch5-drl.md`
**Section:** "P4 - Style/Minor" (ID 5.10)

---

## Cross-Review (Code Review Specialist)

| Field | Value |
|-------|-------|
| **Verdict** | `OBJECT` |
| **Reviewer** | Code Review Specialist |

**Evaluation:** The ticket claims lines 351-352 show inconsistent `/` vs `//` usage, but inspection of `counterfactual.py:351` shows only `n_samples // 2` in the loop (line 351: `for _ in range(n_samples // 2):`). There is no `n_samples / 2` float division in the adjacent code. This finding appears to be a false positive based on stale or incorrect line references. Recommend closing as invalid unless the original reporter can identify the actual location of the inconsistency.

---

## Cross-Review (PyTorch Specialist)

| Field | Value |
|-------|-------|
| **Verdict** | `OBJECT` |
| **Reviewer** | PyTorch Specialist |
| **Date** | 2024-12-27 |

**Evaluation:** The ticket appears to be based on an incorrect reading of the code. Looking at L351-352 in `_generate_shapley_configs`, there is only one division operation: `n_samples // 2` (integer division) in the `range()` call. There is no `n_samples / 2` float division on L351. The code uses `//` consistently. This ticket should be closed as invalid unless the line numbers refer to a different version of the file.

---

## Cross-Review (DRL Specialist)

| Field | Value |
|-------|-------|
| **Verdict** | `OBJECT` |
| **Reviewer** | DRL Specialist |

**Evaluation:** Ticket is based on stale information. Reviewing `counterfactual.py` lines 351-352, the current code reads `for _ in range(n_samples // 2):` which already uses integer division consistently. There is no `n_samples / 2` on line 351 -- the ticket appears to reference an earlier version that has since been corrected. Recommend closing as "already fixed" or "cannot reproduce."
