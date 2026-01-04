# Finding Ticket: TamiyoDecision.confidence Field is Unused

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B9-DRL-08` |
| **Severity** | `P4` |
| **Status** | `open` |
| **Batch** | 9 |
| **Agent** | `drl` |
| **Domain** | `tamiyo/decisions` |
| **Assignee** | |
| **Created** | 2025-12-27 |
| **Updated** | 2025-12-27 |

---

## Location

| Field | Value |
|-------|-------|
| **File(s)** | `/home/john/esper-lite/src/esper/tamiyo/decisions.py` |
| **Line(s)** | `32` |
| **Function/Class** | `TamiyoDecision` |

---

## Summary

**One-line summary:** The `confidence` field is set but never read by downstream consumers.

**Category:**
- [ ] Correctness bug
- [ ] Race condition / concurrency
- [ ] Memory leak / resource issue
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

```python
# Line 32
@dataclass
class TamiyoDecision:
    action: ActionType
    target_seed_id: str | None = None
    reason: str = ""
    confidence: float = 1.0  # <-- Set but never read
```

The heuristic policy sets `confidence` values:
- `min(1.0, signals.metrics.plateau_epochs / 5.0)` for germination
- `min(1.0, improvement / 5.0)` for fossilization

But no downstream consumer reads this field:
- Training loop doesn't use it for action weighting
- Telemetry doesn't log it
- No exploration modification based on confidence

### Impact

- **Dead code**: Computation done but result unused
- **API confusion**: Field exists but serves no purpose
- **Potential feature**: Could be useful for exploration or logging

---

## Recommended Fix

Either:

1. **Remove the field** if no plans to use it:
```python
@dataclass
class TamiyoDecision:
    action: ActionType
    target_seed_id: str | None = None
    reason: str = ""
    # confidence removed - was unused
```

2. **Document intended use** and wire it up:
```python
# In telemetry/logging
logger.info(f"Decision: {decision.action} (confidence={decision.confidence:.2f})")

# Or for exploration (neural policy)
if decision.confidence < 0.5:
    # Consider alternative action
```

3. **Keep for future use** with TODO:
```python
confidence: float = 1.0  # TODO: [FUTURE FUNCTIONALITY] Wire to telemetry/exploration
```

---

## Verification

### How to Verify the Fix

- [ ] Grep for confidence field usage
- [ ] Decide: remove, wire up, or document as future work
- [ ] Implement decision

---

## Related Findings

- B9-DRL-07: Action introspection helpers (related TamiyoDecision code)

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch9-drl.md`
**Section:** "D2 - confidence field is set but never read"
