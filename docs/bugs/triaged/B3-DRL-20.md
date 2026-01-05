# Finding Ticket: current_alpha Naming Ambiguous with Gated Blends

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B3-DRL-20` |
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
| **Line(s)** | `143-149` |
| **Function/Class** | `SeedMetrics.current_alpha` |

---

## Summary

**One-line summary:** `current_alpha` represents blending schedule progress, not actual per-sample alpha for gated blends, which could cause confusion.

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

The `SeedMetrics.current_alpha` field represents the scheduled alpha value (blending progress), not the actual per-sample alpha used in gated blend modes. This is correctly documented in comments but the field name is ambiguous.

### Code Evidence

```python
# /home/john/esper-lite/src/esper/kasmina/slot.py:143-149

current_alpha: float = 0.0
# Note: This is the scheduled alpha, not the per-sample gated alpha
```

### Why This Matters

- RL observation construction uses this value
- Could be misinterpreted as actual blend weight
- Source of confusion for future maintainers

---

## Recommended Fix

Consider renaming to clarify:

```python
blending_progress: float = 0.0  # OR
alpha_schedule_progress: float = 0.0
```

Or add more explicit documentation.

---

## Verification

### How to Verify the Fix

- [ ] Update field name and all references
- [ ] Verify RL observation construction uses correct semantics

---

## Related Findings

None.

---

## Cross-Review

| Reviewer | Verdict | Evaluation |
|----------|---------|------------|
| DRL Expert | **OBJECT** | The existing comment (lines 143-148) already explains the semantics comprehensively: `current_alpha` represents controllable schedule progress, not emergent gate behavior. This is *intentional* for credit assignment. Renaming would obscure the deliberate design choice that the agent controls the timeline, not per-sample gates. |
| PyTorch Specialist | **NEUTRAL** | The field is well-documented with a 6-line comment block (lines 143-148) explaining the semantics. The concern is cosmetic; the code is self-documenting for anyone reading the definition. |
| Code Review Specialist | **NEUTRAL** | The field already has a detailed comment (lines 143-148) explaining the semantics. Renaming would require updating RL observation tensors and could introduce bugs; current documentation is sufficient for P3 priority. |

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch3-drl.md`
**Section:** "P3 - Code Quality" (B3-SLOT-09)
