# Finding Ticket: get_host_parameters Name-Based Filtering Is Fragile

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B2-DRL-12` |
| **Severity** | `P3` |
| **Status** | `open` |
| **Batch** | 2 |
| **Agent** | `drl` |
| **Domain** | `kasmina` |
| **Assignee** | |
| **Created** | 2024-12-27 |
| **Updated** | 2024-12-27 |

---

## Location

| Field | Value |
|-------|-------|
| **File(s)** | `/home/john/esper-lite/src/esper/kasmina/host.py` |
| **Line(s)** | `642-647` |
| **Function/Class** | `MorphogeneticModel.get_host_parameters()` |

---

## Summary

**One-line summary:** Filters parameters by checking if "slots" in name - any refactoring could break this fragile substring match.

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

`get_host_parameters()` filters out seed slot parameters by checking if "slots" appears in the parameter name. This works because SeedSlots are registered via `nn.ModuleDict` named `seed_slots`, but:

1. If `seed_slots` is renamed, this breaks silently
2. If a host parameter contains "slots" in its name (unlikely but possible), it's incorrectly excluded

### Code Evidence

```python
# /home/john/esper-lite/src/esper/kasmina/host.py:642-647

def get_host_parameters(self) -> Iterator[torch.nn.Parameter]:
    """Yield only host parameters, excluding seed slots."""
    for name, param in self.named_parameters():
        if "slots" not in name:  # Fragile substring match
            yield param
```

---

## Recommended Fix

Use explicit exclusion via `self.seed_slots.named_parameters()` set difference:

```python
def get_host_parameters(self) -> Iterator[torch.nn.Parameter]:
    """Yield only host parameters, excluding seed slots."""
    slot_param_ids = {id(p) for p in self.seed_slots.parameters()}
    for param in self.parameters():
        if id(param) not in slot_param_ids:
            yield param
```

---

## Verification

### How to Verify the Fix

- [ ] Verify exclusion works with explicit set difference
- [ ] Test edge case where host param name contains "slots"

---

## Related Findings

| Ticket ID | Relationship | Notes |
|-----------|--------------|-------|
| `B2-PT-08` | `related` | PT also noted same fragility |

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch2-drl.md`
**Section:** "File-by-File Analysis" - host.py - B2-12

---

## Cross-Review: PyTorch Specialist

| Verdict | ENDORSE |
|---------|---------|

The proposed `id(param)` set-difference approach is the idiomatic PyTorch pattern for parameter exclusion and is torch.compile-safe since it operates at optimizer setup time.
String-based filtering via `"slots" in name` is fragile; the fix correctly uses parameter identity which survives refactors and avoids false positives from naming collisions.

---

## Cross-Review: DRL Specialist

| Verdict | ENDORSE |
|---------|---------|

Agree with fragility concern. The code iterates over `self.host.named_parameters()` and filters by "slots" substring, but `seed_slots` lives on `MorphogeneticModel`, not on `self.host`.
Current code is actually *correct* (slots won't appear in host params) but for the wrong reason. The recommended id-based set-difference fix is cleaner and makes intent explicit. Low priority but worth fixing for maintainability.

---

## Cross-Review: Code Review Specialist

| Verdict | OBJECT |
|---------|--------|

The finding misstates the code: `get_host_parameters` iterates `self.host.named_parameters()` (backbone only), not `self.named_parameters()`.
Since seed_slots live on MorphogeneticModel (not host backbone), "slots" never appears in host param names - the substring check is redundant but harmless.
