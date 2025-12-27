# Finding Ticket: Metric Aggregation Type Handling Is Confusing

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B4-DRL-05` |
| **Severity** | `P3` |
| **Status** | `open` |
| **Batch** | 4 |
| **Agent** | `drl` |
| **Domain** | `simic` |
| **Assignee** | |
| **Created** | 2024-12-27 |
| **Updated** | 2024-12-27 |

---

## Location

| Field | Value |
|-------|-------|
| **File(s)** | `/home/john/esper-lite/src/esper/simic/agent/ppo.py` |
| **Line(s)** | `840-851` |
| **Function/Class** | `PPOAgent.update()` |

---

## Summary

**One-line summary:** Metric aggregation converts `list[float]` to `float` for scalars but keeps dicts as-is - the mixed return type is confusing.

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

The metric aggregation at the end of `update()`:

```python
# Lines 840-851
for key in first_metrics:
    values = [m[key] for m in all_metrics if key in m]
    if not values:
        continue
    if isinstance(values[0], (int, float)):
        # Aggregate scalars by averaging
        result[key] = sum(values) / len(values)
    else:
        # Keep dicts as-is (e.g., head_entropies)
        result[key] = values[-1]
```

The return type is `PPOUpdateMetrics` which has `total=False`, but:
1. Some values are `float` (averaged scalars)
2. Some values are `dict[str, list[float]]` (kept as-is from last epoch)

### Why This Is Confusing

- Type annotation says `PPOUpdateMetrics` but actual structure is heterogeneous
- Callers need to know which keys are averaged vs last-value
- The docstring doesn't clarify this behavior

---

## Recommended Fix

Document the aggregation behavior in the docstring:

```python
def update(self, data: dict[str, torch.Tensor]) -> PPOUpdateMetrics:
    """Run PPO update on collected rollout data.

    Returns:
        PPOUpdateMetrics containing:
        - Scalar metrics (policy_loss, value_loss, entropy, etc.): averaged across epochs
        - Dict metrics (head_entropies, head_grad_norms): from final epoch only
    """
```

Or consider splitting into two return values:
```python
AggregatedMetrics = TypedDict('AggregatedMetrics', {
    'policy_loss': float,
    'value_loss': float,
    # ... all scalar fields
})

FinalEpochMetrics = TypedDict('FinalEpochMetrics', {
    'head_entropies': dict[str, list[float]],
    'head_grad_norms': dict[str, list[float]],
})
```

---

## Verification

### How to Verify the Fix

- [ ] Add documentation clarifying aggregation behavior
- [ ] Consider splitting return type if complexity warrants

---

## Related Findings

None.

---

## Cross-Review

| Agent | Verdict | Evaluation |
|-------|---------|------------|
| **PyTorch** | NEUTRAL | Documentation concern only - the mixed aggregation (scalar averaging vs dict passthrough) is semantically correct for PPO metrics. No tensor operation or memory implications; TypedDict heterogeneity does not affect runtime behavior. |
| **DRL** | ENDORSE | Mixed aggregation (averaged scalars vs. last-epoch dicts) is standard PPO practice: losses should be averaged for stable logging, but per-head entropy/grad-norm histories need final-epoch values for debugging training dynamics. The fix is documentation, not restructuring. |
| **CodeReview** | ENDORSE | The mixed aggregation strategy (average for scalars, last-value for dicts) is not documented in the return type or docstring. Lines 846-851 show dict payloads taking `first` value while scalars are averaged - this semantic difference should be explicit. The docstring fix is the minimal appropriate solution; the TypedDict split is overkill for a P3. |
