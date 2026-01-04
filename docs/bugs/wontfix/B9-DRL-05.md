# Finding Ticket: Hidden State Detachment is Caller Responsibility

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B9-DRL-05` |
| **Severity** | `P3` |
| **Status** | `wont-fix` |
| **Batch** | 9 |
| **Agent** | `drl` |
| **Domain** | `tamiyo/networks` |
| **Assignee** | |
| **Created** | 2025-12-27 |
| **Updated** | 2025-12-27 |

---

## Location

| Field | Value |
|-------|-------|
| **File(s)** | `/home/john/esper-lite/src/esper/tamiyo/networks/factored_lstm.py` |
| **Line(s)** | `266-283` |
| **Function/Class** | `FactoredRecurrentActorCritic.get_initial_hidden()` |

---

## Summary

**One-line summary:** LSTM hidden state detachment at episode boundaries is caller's responsibility, easy to forget.

**Category:**
- [ ] Correctness bug
- [ ] Race condition / concurrency
- [x] Memory leak / resource issue
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

The docstring in `get_initial_hidden` correctly explains the need to detach hidden states at episode boundaries:

```python
"""
MEMORY MANAGEMENT - Hidden State Detachment:
--------------------------------------------
LSTM hidden states carry gradient graphs. To prevent memory leaks during
training, callers MUST detach hidden states at episode boundaries:

    hidden = (h.detach(), c.detach())  # Break gradient graph
...
"""
```

However:

1. This is documentation only - no enforcement
2. If caller forgets to detach:
   - BPTT extends across episode boundaries (incorrect gradients)
   - Memory grows unbounded (gradient graphs accumulate)
   - OOM after ~100-1000 episodes

### Current Mitigation

The training loop in `vectorized.py` does correctly detach (lines 2934-2935). But this is easy to forget when writing new training loops or testing.

---

## Recommended Fix

Add a convenience method that documents and performs detachment:

```python
@staticmethod
def detach_hidden(hidden: HiddenState) -> HiddenState:
    """Detach hidden state gradient graphs at episode boundaries.

    MUST be called when an episode ends to prevent:
    1. BPTT extending across episodes (incorrect gradients)
    2. Unbounded memory growth (OOM after ~100-1000 eps)

    Args:
        hidden: LSTM hidden state tuple (h, c)

    Returns:
        Detached hidden state (no gradient graph)
    """
    h, c = hidden
    return (h.detach(), c.detach())
```

Or add auto-detach option to get_initial_hidden:

```python
def get_initial_hidden(
    self,
    batch_size: int = 1,
    device: torch.device | None = None,
    detach_from: HiddenState | None = None,  # If provided, detach instead of zeros
) -> HiddenState:
    ...
```

---

## Verification

### How to Verify the Fix

- [ ] Add helper method for detachment
- [ ] Consider adding memory growth test for episode boundaries
- [ ] Verify all training loops use proper detachment

---

## Related Findings

- B9-DRL-06: No option to skip hidden state in evaluate_actions (related hidden state API)

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch9-drl.md`
**Section:** "N3 - Hidden state detachment is caller responsibility"

---

## Resolution

### Status: WONTFIX

**Closed via Systematic Debugging investigation.**

#### Evidence

| Claim | Status | Evidence |
|-------|--------|----------|
| "Docstring explains detachment requirement" | ✅ TRUE | Lines 360-374 comprehensive documentation |
| "vectorized.py correctly detaches" | ✅ TRUE | Lines 3235-3236 call `.detach()` |
| "Memory leaks if caller forgets" | ✅ TRUE | Standard PyTorch LSTM behavior |
| "Easy to forget" | ⚠️ SUBJECTIVE | True of many PyTorch patterns |

#### Why This Is Acceptable Design

1. **Comprehensive documentation**: The docstring at lines 360-374 clearly explains:
   - Why detachment is needed
   - What happens if you forget (OOM after ~100-1000 episodes)
   - That vectorized.py handles it automatically

2. **Standard PyTorch idiom**: LSTM hidden state detachment is a well-known pattern. Similar "caller responsibility" patterns exist throughout PyTorch:
   - `model.eval()` before inference
   - `with torch.no_grad()` for inference
   - `.backward()` before optimizer step

3. **Main use case handled**: The training loop in `vectorized.py` correctly detaches hidden states at episode boundaries (lines 3235-3236).

4. **Helper method doesn't prevent misuse**: Adding `detach_hidden()` would be ergonomic but wouldn't prevent callers from forgetting to call it - same as the current pattern.

#### Severity Downgrade

- Original: P3 (API design / memory leak potential)
- Revised: WONTFIX (documented behavior, standard PyTorch idiom)
- Resolution: No action required
