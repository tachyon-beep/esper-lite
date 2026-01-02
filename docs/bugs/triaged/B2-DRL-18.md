# Finding Ticket: Property/Method Distinction Unclear in HostProtocol

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B2-DRL-18` |
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
| **File(s)** | `/home/john/esper-lite/src/esper/kasmina/protocol.py` |
| **Line(s)** | `35-43` |
| **Function/Class** | `HostProtocol` |

---

## Summary

**One-line summary:** `injection_points` and `segment_channels` are properties while `injection_specs()` is a method - naming doesn't indicate which is cached vs computed.

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

The Protocol defines some members as properties and others as methods. The naming doesn't clearly indicate:
- Which are cached (cheap to access repeatedly)
- Which compute new objects each call (allocation cost)

### Why This Is Low Priority

The Protocol documentation could clarify this, but it doesn't affect correctness.

---

## Recommended Fix

Add docstring guidance on performance expectations:

```python
class HostProtocol(Protocol):
    """Protocol for pluggable host networks.

    Properties (cached, cheap to access):
        injection_points: Segment indices where seeds can attach
        segment_channels: Channel counts at each segment

    Methods (may allocate):
        injection_specs(): Creates new InjectionSpec objects each call
    """
```

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch2-drl.md`
**Section:** "Severity-Tagged Findings Summary" - B2-18

---

## Cross-Review

| Verdict | Reviewer | Domain |
|---------|----------|--------|
| **ENDORSE** | DRL Specialist | Deep RL |
| **ENDORSE** | PyTorch Specialist | PyTorch Engineering |

**DRL Evaluation:** Valid documentation improvement; properties vs methods should indicate allocation cost.
The suggested docstring separates cached properties from allocating methods - low-effort, high-clarity fix.

**PyTorch Evaluation:** The property/method distinction follows Python convention: properties for cached/cheap accessors, methods for operations that may allocate. The suggested docstring improvement is reasonable low-effort documentation. No torch.compile or performance concerns - purely a documentation enhancement at P4 priority.
