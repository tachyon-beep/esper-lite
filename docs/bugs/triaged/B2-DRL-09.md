# Finding Ticket: Thread-Local Cache Never Cleaned in Practice

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B2-DRL-09` |
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
| **File(s)** | `/home/john/esper-lite/src/esper/kasmina/blending.py` |
| **Line(s)** | `93-104` |
| **Function/Class** | `BlendAlgorithm.reset_cache()` |

---

## Summary

**One-line summary:** `reset_cache()` method exists but is never called in the codebase - thread-local cache accumulates in long-running training.

**Category:**
- [ ] Correctness bug
- [ ] Race condition / concurrency
- [x] Memory leak / resource issue
- [ ] Performance bottleneck
- [ ] Numerical stability
- [ ] torch.compile compatibility
- [x] Dead code / unwired functionality
- [ ] API design / contract violation
- [ ] Test coverage gap
- [ ] Documentation / naming
- [ ] Defensive programming violation
- [ ] Legacy code policy violation

---

## Detailed Description

### What's Wrong

`BlendAlgorithm` has a `reset_cache()` method for clearing thread-local alpha tensor caches, but it's never called anywhere in the codebase. In long-running training with DataParallel, each worker thread accumulates cache entries.

### Code Evidence

```python
# /home/john/esper-lite/src/esper/kasmina/blending.py:93-104

def reset_cache(self) -> None:
    """Clear thread-local alpha tensor cache.

    Call at epoch boundaries in long-running training to prevent
    memory accumulation across DataParallel workers.
    """
    self._alpha_cache_local.cache = None  # Only clears calling thread
```

### Impact

- Memory accumulation over epochs
- Each worker thread holds cached tensors indefinitely
- Impact is small (single scalar tensor per thread) but technically a leak

---

## Recommended Fix

### Option A: Call reset_cache() at epoch boundaries

In the training loop:

```python
def on_epoch_end():
    for slot in model.seed_slots.values():
        if hasattr(slot._blend_algorithm, 'reset_cache'):
            slot._blend_algorithm.reset_cache()
```

### Option B: Document when to call

If manual cleanup is expected, document clearly in docstring.

---

## Verification

### How to Verify the Fix

- [ ] Add call to reset_cache() in training loop
- [ ] Profile memory usage across epochs
- [ ] Verify cache is properly cleared

---

## Related Findings

| Ticket ID | Relationship | Notes |
|-----------|--------------|-------|
| `B2-PT-01` | `duplicate` | Primary ticket - P1 severity |
| `B2-CR-01` | `related` | CodeReview perspective on same issue |

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch2-drl.md`
**Section:** "File-by-File Analysis" - blending.py - B2-09

---

## Cross-Review: PyTorch Specialist

| Verdict | **ENDORSE** |
|---------|-------------|
| **Evaluation** | Thread-local storage in PyTorch DataParallel is a known footgun; each worker thread persists until DataLoader shutdown. The cache holds a single scalar tensor per thread (negligible memory), but the pattern is sloppy. Wire `reset_cache()` to epoch boundaries or switch to `WeakValueDictionary` keyed on device. |

---

## Cross-Review: Code Review Specialist

| Verdict | **NEUTRAL** |
|---------|-------------|
| **Evaluation** | Confirmed: `reset_cache()` at line 93 exists but grep shows zero call sites in the codebase; however, cache holds only a single scalar tensor per thread. Memory impact is negligible (bytes, not megabytes); wiring cleanup call adds complexity disproportionate to the actual resource savings.

---

## Cross-Review: DRL Specialist

| Verdict | **ENDORSE** |
|---------|-------------|
| **Evaluation** | Confirmed: grep shows `reset_cache()` is only defined in blending.py, never called. For RL training runs spanning thousands of epochs with DataParallel, each worker thread accumulates a cached tensor indefinitely. The leak is small (one scalar tensor per thread) but violates resource hygiene. Wire `reset_cache()` into `step_epoch()` or document that callers must invoke it. |
