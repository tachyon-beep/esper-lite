# Finding Ticket: Factory Kwargs Not Validated at Registration

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B3-DRL-07` |
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
| **Line(s)** | `103-118` |
| **Function/Class** | `BlueprintRegistry.create()` |

---

## Summary

**One-line summary:** The `create()` method passes `**kwargs` to factory but factory signatures vary, making it hard to validate kwargs at registration time.

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

Different blueprint factories accept different kwargs:
- Some take `reduction` (SE attention)
- Some take `rank` (LoRA)
- Some take `checkpoint` (MLP with activation checkpointing)

The registry's `create(**kwargs)` passes all kwargs blindly, and invalid kwargs only fail at instantiation time.

### Code Evidence

```python
# /home/john/esper-lite/src/esper/kasmina/blueprints/registry.py:103-118

def create(
    self,
    name: str,
    dim: int,
    topology: str | None = None,
    **kwargs: Any,
) -> nn.Module:
    spec = self._get_spec(name, topology)
    return spec.factory(dim, **kwargs)  # Type safety gap here
```

### Why This Matters

- Invalid kwargs produce runtime errors, not registration-time errors
- Difficult to discover valid kwargs for each blueprint
- Type safety gap in the registry API

---

## Recommended Fix

Consider documenting valid kwargs per blueprint or adding a `valid_kwargs` field to `BlueprintSpec`:

```python
@dataclass
class BlueprintSpec:
    name: str
    topology: str
    factory: BlueprintFactory
    param_estimate: int
    valid_kwargs: frozenset[str] = frozenset()  # e.g., {"reduction", "checkpoint"}
```

---

## Verification

### How to Verify the Fix

- [ ] Audit all blueprint factories for their kwargs
- [ ] Add validation in create() if valid_kwargs is specified
- [ ] Update tests to cover invalid kwargs

---

## Related Findings

None.

---

## Cross-Review

| Reviewer | Verdict | Evaluation |
|----------|---------|------------|
| **DRL Specialist** | **OBJECT** | Adding `valid_kwargs` metadata creates maintenance burden without preventing runtime errors (still need try/except). The existing docstring in `create()` (lines 110-112) already documents valid kwargs per blueprint. |
| **PyTorch Specialist** | **OBJECT** | This is documentation, not API violation. PyTorch itself uses `**kwargs` in factory patterns (e.g., `torch.optim.Adam`); clear TypeError at instantiation is acceptable. Prefer improving existing docstrings. |
| **Code Review Specialist** | **OBJECT** | Category mislabeled as "API design violation" - this is standard factory pattern. Docstring (lines 106-112) already documents kwargs. Close as won't-fix. |

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch3-drl.md`
**Section:** "P3 - Code Quality" (B3-REG-03)
