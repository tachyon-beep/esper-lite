# Finding Ticket: match-case Default Passes Unknown Gates

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B3-DRL-19` |
| **Severity** | `P3` |
| **Status** | `open` |
| **Batch** | 3 |
| **Agent** | `drl` |
| **Domain** | `kasmina` |
| **Assignee** | |
| **Created** | 2024-12-27 |
| **Updated** | 2024-12-27 |

---

## Location

| Field | Value |
|-------|-------|
| **File(s)** | `/home/john/esper-lite/src/esper/kasmina/slot.py` |
| **Line(s)** | `672-688` |
| **Function/Class** | `QualityGates._check_gate()` |

---

## Summary

**One-line summary:** The match statement for gate levels has a default case that returns `passed=True`, which could accidentally advance seeds if a new gate level is added.

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

The match statement dispatches to individual gate checks. If a new gate level is added to Leyline but not handled here, the default case returns `passed=True`, potentially advancing seeds through unchecked gates.

### Code Evidence

```python
# /home/john/esper-lite/src/esper/kasmina/slot.py:672-688

match gate:
    case GateLevel.G0:
        return self._check_g0(...)
    case GateLevel.G1:
        return self._check_g1(...)
    # ... other gates
    case _:
        return GateResult(passed=True, ...)  # Dangerous default!
```

---

## Recommended Fix

Raise an error for unknown gates:

```python
case _:
    raise ValueError(f"Unknown gate level: {gate}")
```

---

## Verification

### How to Verify the Fix

- [ ] Add test that unknown gate level raises ValueError
- [ ] Audit Leyline for potential new gate additions

---

## Related Findings

None.

---

## Cross-Review

| Verdict | **ENDORSE** |
|---------|-------------|

**Evaluation:** Clear correctness bug. Lines 686-688 show unknown gates silently pass with `passed=True`. If GateLevel enum is extended in Leyline without updating `check_gate()`, seeds would skip validation entirely. Should raise `ValueError` per fail-fast principles.

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch3-drl.md`
**Section:** "P3 - Code Quality" (B3-SLOT-07)
