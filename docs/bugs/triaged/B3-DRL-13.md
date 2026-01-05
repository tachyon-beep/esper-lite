# Finding Ticket: torch._foreach_norm Is Private API

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B3-DRL-13` |
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
| **File(s)** | `/home/john/esper-lite/src/esper/kasmina/isolation.py` |
| **Line(s)** | `172, 186` |
| **Function/Class** | `GradientHealthMonitor` async methods |

---

## Summary

**One-line summary:** `torch._foreach_norm` is a private API (underscore prefix) that could break in future PyTorch versions.

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

The code uses `torch._foreach_norm` for batched norm computation. While stable since PyTorch 1.9, it's still underscore-prefixed (private).

### Code Evidence

```python
# /home/john/esper-lite/src/esper/kasmina/isolation.py:172,186

norms = torch._foreach_norm(host_grads)
```

### Why This Is Acceptable

- Performance benefit for batched norm computation
- Stable since PyTorch 1.9 per comment
- Per project guidelines, we don't add legacy compatibility code

---

## Recommended Fix

Add version documentation but accept the risk:

```python
# NOTE: Private API, stable since PyTorch 1.9.
# If this breaks in future PyTorch, fallback is:
#   norms = [g.norm() for g in host_grads]
norms = torch._foreach_norm(host_grads)
```

---

## Verification

### How to Verify the Fix

- [ ] Verify availability in target PyTorch version
- [ ] Add CI test that catches import failures

---

## Related Findings

- B3-PT-03: torch._dynamo.disable private API (similar concern)

---

## Cross-Review: PyTorch Expert

| Verdict | Evaluation |
|---------|------------|
| **ENDORSE** | `torch._foreach_norm` is stable since PyTorch 1.9 and used internally by `clip_grad_norm_`. The O(1) kernel launch benefit (vs O(n) per-tensor norms) is significant for gradient health monitoring. The existing code comment already documents the fallback. Risk is low; keep as-is with documentation. |

## Cross-Review: Code Reviewer

| Verdict | Evaluation |
|---------|------------|
| **NEUTRAL** | Valid concern, but code already documents the private API and fallback (lines 168-171). Per project policy, we avoid compatibility shims - existing docs are sufficient. |

## Cross-Review: DRL Specialist

| Verdict | Evaluation |
|---------|------------|
| **NEUTRAL** | The private API concern is valid but overstated; the existing comment at lines 168-170 already documents this with a fallback path. The performance benefit for G2 gate gradient monitoring is real (O(1) vs O(n) kernel launches). |

**Recommendation:** Accept P3. The code already has the recommended documentation. No further action needed unless CI test for import availability is desired.

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch3-drl.md`
**Section:** "P3 - Code Quality" (B3-ISO-02)
