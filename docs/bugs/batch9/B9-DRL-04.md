# Finding Ticket: Summary Seed Selection Prefers Lower Counterfactual

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B9-DRL-04` |
| **Severity** | `P2` |
| **Status** | `open` |
| **Batch** | 9 |
| **Agent** | `drl` |
| **Domain** | `tamiyo/tracker` |
| **Assignee** | |
| **Created** | 2025-12-27 |
| **Updated** | 2025-12-27 |

---

## Location

| Field | Value |
|-------|-------|
| **File(s)** | `/home/john/esper-lite/src/esper/tamiyo/tracker.py` |
| **Line(s)** | `205-211` |
| **Function/Class** | `SignalTracker._select_summary_seed()` |

---

## Summary

**One-line summary:** The `summary_key` function uses `-counterfactual` which prefers seeds with lower (more negative) contribution.

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

```python
# Lines 205-211
def summary_key(seed):
    return (
        -seed.stage.value,      # Highest stage first
        -seed.alpha,            # Highest alpha first
        -counterfactual,        # Most NEGATIVE counterfactual first (?)
        seed.seed_id,           # Deterministic tie-break
    )
```

The negation on counterfactual (`-counterfactual`) for tie-breaking means the selection prefers seeds with *lower* (more negative) counterfactual contribution. The comment says "(safety)" but the rationale is unclear.

Typically, we'd want to prioritize seeds with *higher* positive contribution for the summary, as these are the "best performing" seeds.

### Impact

- **Counterintuitive summary**: Summary seed may be the worst performer
- **Misleading signals**: TrainingSignals.seed_* fields show low-performer metrics
- **Policy confusion**: Neural policy observing summary may get wrong signals

---

## Recommended Fix

Either:

1. **Flip the sign** if positive contribution should be preferred:
```python
-counterfactual  # Remove negation, or use +counterfactual
```

2. **Document the rationale** if current behavior is intentional:
```python
-counterfactual,  # Prefer lower contribution for conservative summary (safety: report worst case)
```

3. **Add both** - summarize best and worst:
```python
@property
def best_seed(self) -> SeedInfo:
    ...
@property
def worst_seed(self) -> SeedInfo:
    ...
```

---

## Verification

### How to Verify the Fix

- [ ] Clarify intended semantics with domain expert
- [ ] Document selection rule in TrainingSignals docstring
- [ ] Add test verifying selection behavior

---

## Related Findings

- B9-CR-02: best_val_loss naming (related tracker semantics)

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch9-drl.md`
**Section:** "T2 - Summary seed selection prefers lower counterfactual (counterintuitive)"
