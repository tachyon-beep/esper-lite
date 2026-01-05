# Finding Ticket: BlendCatalog._algorithms Is Mutable Class Variable

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B2-DRL-14` |
| **Severity** | `P4` |
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
| **File(s)** | `/home/john/esper-lite/src/esper/kasmina/blending.py` |
| **Line(s)** | `203-205` |
| **Function/Class** | `BlendCatalog._algorithms` |

---

## Summary

**One-line summary:** Class-level mutable dict could be modified at runtime, breaking catalog immutability.

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

`BlendCatalog._algorithms` is a class-level mutable dict. While there's no code that modifies it at runtime, the mutability is a design smell.

---

## Recommended Fix

Use `@functools.cache` on a factory method, or consider `frozendict` pattern:

```python
class BlendCatalog:
    @staticmethod
    @functools.cache
    def _get_algorithms() -> dict[str, type[BlendAlgorithm]]:
        return {"gated": GatedBlend}
```

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch2-drl.md`
**Section:** "File-by-File Analysis" - blending.py - B2-14

---

## Cross-Review: PyTorch Specialist

| Verdict | NEUTRAL |
|---------|---------|

Class-level mutable dicts are a standard Python pattern for registries; the `@functools.cache` alternative adds indirection without real benefit since the dict is never mutated.
No torch.compile implications since `BlendCatalog` is used at module construction time, not in forward passes; this is stylistic preference rather than a functional issue.

---

## Cross-Review: DRL Specialist

| Verdict | OBJECT |
|---------|--------|

This is a standard registry pattern, not a bug. The `_algorithms` dict is a class-level constant populated at module load; no runtime mutation occurs.
Using `@functools.cache` on a factory method adds complexity and indirection for zero practical benefit. Python lacks `frozendict` in stdlib, and third-party deps for this are overkill. The "design smell" framing is unfounded; this is idiomatic Python for registries. Recommend closing as won't-fix.

---

## Cross-Review: Code Review Specialist

| Verdict | NEUTRAL |
|---------|---------|

While `_algorithms` is technically mutable, no runtime code modifies it and Python lacks built-in `frozendict`.
The `@functools.cache` pattern adds complexity for a purely theoretical threat; acceptable as-is given P4 severity.
