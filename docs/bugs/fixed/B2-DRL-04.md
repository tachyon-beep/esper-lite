# Finding Ticket: Cached Properties on Mutable State (DRL Perspective)

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B2-DRL-04` |
| **Severity** | `P2` |
| **Status** | `closed` |
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
| **Line(s)** | `102-113` |
| **Function/Class** | `CNNHost.segment_channels`, `CNNHost._segment_to_block` |

---

## Summary

**One-line summary:** `@functools.cached_property` on segment_channels/segment_to_block assumes blocks are immutable after init.

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

`segment_channels` and `_segment_to_block` use `@functools.cached_property` but depend on `injection_specs()` which creates new objects each call. If the host is modified after init (e.g., blocks added/removed), cached values become stale.

### Code Evidence

```python
# /home/john/esper-lite/src/esper/kasmina/host.py:102-113

@functools.cached_property
def segment_channels(self) -> list[int]:
    """Channel count at each injection point."""
    return [spec.channels for spec in self.injection_specs()]

@functools.cached_property
def _segment_to_block(self) -> dict[int, int]:
    """Map segment index to block index."""
    return {spec.segment: i for i, spec in enumerate(self.injection_specs())}
```

### DRL Impact

- If host architecture changes during training (not currently supported), cached values would be wrong
- Slot injection would target wrong segments
- Training would silently corrupt

### Current Mitigation

The protocol doesn't support dynamic block modification. Caching is safe in practice.

---

## Recommended Fix

### Option A: Document immutability assumption

```python
@functools.cached_property
def segment_channels(self) -> list[int]:
    """Channel count at each injection point.

    Note: Assumes blocks are immutable after construction.
    Do not modify self.blocks after init or cache will be stale.
    """
```

### Option B: Add cache invalidation

```python
def _invalidate_caches(self) -> None:
    """Clear cached segment info. Call after modifying blocks."""
    if 'segment_channels' in self.__dict__:
        del self.__dict__['segment_channels']
    if '_segment_to_block' in self.__dict__:
        del self.__dict__['_segment_to_block']
```

---

## Verification

### How to Verify the Fix

- [ ] Add test verifying cache consistency
- [ ] Document immutability requirement

---

## Related Findings

| Ticket ID | Relationship | Notes |
|-----------|--------------|-------|
| `B2-CR-02` | `duplicate` | Primary ticket from CodeReview perspective |

---

## Resolution

**Status:** Fixed by B2-CR-02

**Fix:** Documented architecture immutability invariant at protocol, class, and property levels.

See B2-CR-02 for full resolution details.

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch2-drl.md`
**Section:** "File-by-File Analysis" - host.py - B2-04

---

## Cross-Review: DRL Specialist

| Field | Value |
|-------|-------|
| **Verdict** | `NEUTRAL` |
| **Reviewer** | `drl` |

**Evaluation:** No current DRL impact since architecture is frozen during training. Option A (document immutability) is sufficient; cache invalidation (Option B) adds complexity for a non-existent dynamic architecture use case.

---

## Cross-Review: Code Review Specialist

| Field | Value |
|-------|-------|
| **Verdict** | `NEUTRAL` |
| **Reviewer** | `codereview` |

**Evaluation:** Reviewed host.py: `segment_channels` and `_segment_to_block` derive from `self.blocks` which is set in `__init__` and never mutated. Caching is safe by design. Option A (document immutability) is sufficient.

---

## Cross-Review: PyTorch Specialist

| Field | Value |
|-------|-------|
| **Verdict** | `ENDORSE` |
| **Reviewer** | `pytorch` |

**Evaluation:** `@functools.cached_property` is safe here only because nn.Module architecture is frozen after `__init__` - this is a PyTorch invariant that the caching correctly exploits.
Option A (documentation) is sufficient since dynamic block modification would violate nn.Module contracts (parameter registration, state_dict) well before cache staleness matters.
