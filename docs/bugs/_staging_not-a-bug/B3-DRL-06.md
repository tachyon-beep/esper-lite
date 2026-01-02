# Finding Ticket: actual_param_count Allocates Full Module

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B3-DRL-06` |
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
| **File(s)** | `/home/john/esper-lite/src/esper/kasmina/blueprints/registry.py` |
| **Line(s)** | `47-50` |
| **Function/Class** | `BlueprintSpec.actual_param_count()` |

---

## Summary

**One-line summary:** `actual_param_count()` instantiates a full module just to count parameters, then discards it.

**Category:**
- [ ] Correctness bug
- [ ] Race condition / concurrency
- [ ] Memory leak / resource issue
- [x] Performance bottleneck
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

The `actual_param_count` method creates a full module instance just to count its parameters, then immediately discards it. For large blueprints (e.g., transformer MLP with dim=4096), this allocates significant memory.

### Code Evidence

```python
# /home/john/esper-lite/src/esper/kasmina/blueprints/registry.py:47-50

def actual_param_count(self, dim: int) -> int:
    module = self.factory(dim)
    return sum(p.numel() for p in module.parameters())
```

### Impact

- Unnecessary GPU/CPU memory allocation
- GC pressure if called frequently
- Could be slow for large modules

---

## Recommended Fix

Consider caching the computed param count or computing it lazily:

```python
@lru_cache(maxsize=16)
def actual_param_count(self, dim: int) -> int:
    module = self.factory(dim)
    count = sum(p.numel() for p in module.parameters())
    del module  # Explicit cleanup
    return count
```

---

## Verification

### How to Verify the Fix

- [ ] Profile memory usage of registry operations
- [ ] Add benchmark for large-dim param counting

---

## Related Findings

None.

---

## Cross-Review

| Reviewer | Verdict | Evaluation |
|----------|---------|------------|
| **DRL Specialist** | **NEUTRAL** | `actual_param_count` is called once per blueprint registration, not during training loops. The allocation is trivial and GC handles cleanup; lru_cache adds complexity without measurable benefit. |
| **PyTorch Specialist** | **NEUTRAL** | Module instantiation is CPU-bound (no GPU allocation until `.to(device)`). `lru_cache` on methods requires self in key; `del module` is redundant since GC handles it after method return. |
| **Code Review Specialist** | **NEUTRAL** | Proposed fix is incompatible with frozen dataclass (no `@lru_cache` on instance methods). Low-priority given infrequent invocation. |

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch3-drl.md`
**Section:** "P3 - Code Quality" (B3-REG-01)
