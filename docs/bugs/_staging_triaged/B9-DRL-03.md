# Finding Ticket: Missing None Validation for Improvement in HOLDING Decision

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B9-DRL-03` |
| **Severity** | `P2` |
| **Status** | `open` |
| **Batch** | 9 |
| **Agent** | `drl` |
| **Domain** | `tamiyo/heuristic` |
| **Assignee** | |
| **Created** | 2025-12-27 |
| **Updated** | 2025-12-27 |

---

## Location

| Field | Value |
|-------|-------|
| **File(s)** | `/home/john/esper-lite/src/esper/tamiyo/heuristic.py` |
| **Line(s)** | `269` |
| **Function/Class** | `HeuristicTamiyo._decide_seed_management()` |

---

## Summary

**One-line summary:** `improvement` can be `None` when both metrics are missing, causing TypeError on comparison.

**Category:**
- [x] Correctness bug
- [ ] Race condition / concurrency
- [ ] Memory leak / resource issue
- [ ] Performance bottleneck
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
# Line 269 (approximate)
# In HOLDING stage decision logic:
if improvement > self.config.min_improvement_to_fossilize:
    # ... fossilize decision
```

The variable `improvement` is derived from either `contribution` or `total_improvement`. If both are `None`:

```python
improvement = contribution if contribution is not None else total_improvement
# improvement is now None
if improvement > self.config.min_improvement_to_fossilize:  # TypeError!
```

### Impact

- **Runtime crash**: TypeError when comparing None to float
- **Silent failure**: If upstream is expected to always provide metrics but doesn't
- **Training interruption**: Crash during decision-making halts training

### Current Mitigation

The code currently relies on at least one metric being populated by upstream callers. This appears to be enforced in practice, but there's no defensive validation.

---

## Recommended Fix

Add explicit validation:

```python
# Option 1: Default to 0.0 (conservative - don't fossilize without evidence)
improvement = contribution if contribution is not None else total_improvement
if improvement is None:
    improvement = 0.0  # No improvement data = no fossilization

# Option 2: Skip decision if no data
if improvement is None:
    return TamiyoDecision(action=Action.WAIT, reason="No improvement data")

# Option 3: Assert upstream contract
assert improvement is not None, "HOLDING seeds must have improvement metric"
```

---

## Verification

### How to Verify the Fix

- [ ] Add test case where both contribution and total_improvement are None
- [ ] Verify decision behavior matches intended semantics
- [ ] Document expected upstream contract

---

## Related Findings

- B9-DRL-01: Blueprint selection fallback (related heuristic logic)

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch9-drl.md`
**Section:** "H2 - Missing validation for `improvement` in HOLDING decision"
