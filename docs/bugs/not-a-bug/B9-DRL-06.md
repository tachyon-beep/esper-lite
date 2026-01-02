# Finding Ticket: No Option to Skip Hidden State in evaluate_actions

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B9-DRL-06` |
| **Severity** | `P3` |
| **Status** | `open` |
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
| **Line(s)** | `533-632` |
| **Function/Class** | `FactoredRecurrentActorCritic.evaluate_actions()` |

---

## Summary

**One-line summary:** evaluate_actions always returns hidden state even when caller doesn't need it (PPO updates).

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

```python
# Line 632 (approximate)
return (log_probs, entropy, value, hidden)  # hidden always computed and returned
```

The `evaluate_actions` method always computes and returns the LSTM hidden state as part of its tuple. However:

1. During PPO updates, the hidden state is typically not needed (we're evaluating stored trajectories)
2. The forward pass through LSTM still computes hidden state (unavoidable)
3. But returning it in the tuple prevents optimization and adds to return overhead

### Impact

- **Minor overhead**: Tuple construction and return with unused tensor
- **Memory**: Hidden state tensors kept alive longer than necessary
- **API complexity**: Caller must unpack unused value

---

## Recommended Fix

Add optional parameter to skip hidden state return:

```python
def evaluate_actions(
    self,
    states: torch.Tensor,
    actions: FactoredAction,
    hidden: HiddenState,
    masks: ActionMasks,
    return_hidden: bool = True,  # Default True for backwards compat
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, HiddenState | None]:
    """
    ...
    Args:
        return_hidden: If False, returns None for hidden state (saves memory in PPO updates)
    """
    # ... forward pass (hidden computed anyway)

    if return_hidden:
        return (log_probs, entropy, value, new_hidden)
    else:
        return (log_probs, entropy, value, None)
```

Or return a namedtuple/dataclass where hidden is Optional:

```python
@dataclass
class EvaluateResult:
    log_probs: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor
    hidden: HiddenState | None = None
```

---

## Verification

### How to Verify the Fix

- [ ] Add return_hidden parameter
- [ ] Update PPO training loop to use return_hidden=False
- [ ] Benchmark memory impact (likely minimal)

---

## Related Findings

- B9-DRL-05: Hidden state detachment (related hidden state API)

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch9-drl.md`
**Section:** "N4 - Missing `return_hidden` for `evaluate_actions`"
