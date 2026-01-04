# Finding Ticket: Credit Assignment Timing - Shapley Computed at Episode End

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B5-DRL-07` |
| **Severity** | `P3` |
| **Status** | `open` |
| **Batch** | 5 |
| **Agent** | `drl` |
| **Domain** | `simic/attribution` |
| **Assignee** | |
| **Created** | 2024-12-27 |
| **Updated** | 2024-12-27 |

---

## Location

| Field | Value |
|-------|-------|
| **File(s)** | `/home/john/esper-lite/src/esper/simic/attribution/counterfactual.py` |
| **Line(s)** | N/A (design consideration) |
| **Function/Class** | `CounterfactualEngine` |

---

## Summary

**One-line summary:** Shapley values computed at episode end may be noisy during early training when host is adapting.

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

Shapley values are computed at episode end. During early training when the host network is rapidly adapting, these values are noisy because:

1. The host's baseline performance changes between episodes
2. Seeds that were beneficial may become neutral as host learns
3. High variance in Shapley estimates makes lifecycle decisions unreliable

### Impact

Early-stage lifecycle decisions (prune/fossilize) may be based on noisy attribution data, leading to suboptimal seed selection.

---

## Recommended Fix

Consider temporal smoothing:

```python
# Option 1: EMA of Shapley values across episodes
class CounterfactualHelper:
    def __init__(self, ...):
        self._shapley_ema: dict[str, float] = {}
        self._ema_momentum = 0.9

    def _process_matrix(self, matrix, slot_ids):
        shapley = self.engine.compute_shapley_values(matrix)
        for slot_id, estimate in shapley.items():
            if slot_id in self._shapley_ema:
                self._shapley_ema[slot_id] = (
                    self._ema_momentum * self._shapley_ema[slot_id]
                    + (1 - self._ema_momentum) * estimate.value
                )
            else:
                self._shapley_ema[slot_id] = estimate.value
```

Or document the limitation:
```python
# NOTE: Shapley values are single-episode estimates. During early training,
# these values have high variance. Lifecycle decisions should use
# aggregated metrics over multiple episodes.
```

---

## Verification

### How to Verify the Fix

- [ ] Consider implementing EMA smoothing
- [ ] At minimum, add documentation about timing considerations

---

## Related Findings

- B5-CR-01: Unseeded RNG adds to Shapley variance

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch5-drl.md`
**Section:** "RL-Specific Observations" (Credit assignment timing)

---

## Cross-Review (Code Review Specialist)

| Field | Value |
|-------|-------|
| **Verdict** | `NEUTRAL` |
| **Reviewer** | Code Review Specialist |

**Evaluation:** This is a valid design consideration rather than a code defect. The `CounterfactualEngine` is stateless by design (per docstring: "doesn't store state -- that's the collector's job"), so temporal smoothing would need to be implemented in the caller (likely `CounterfactualHelper`). The recommended EMA fix is architecturally appropriate. However, the ticket should clarify whether early-training noise is causing observable harm to lifecycle decisions in practice, or if this is speculative. P3 severity seems appropriate for a potential enhancement rather than a known problem.

---

## Cross-Review (PyTorch Specialist)

| Field | Value |
|-------|-------|
| **Verdict** | `NEUTRAL` |
| **Reviewer** | PyTorch Specialist |
| **Date** | 2024-12-27 |

**Evaluation:** This is a valid RL design consideration but not a PyTorch implementation issue. EMA smoothing as proposed would work fine with torch tensors, though the current implementation uses Python floats in the Shapley computation path. The real question is whether lifecycle decisions should be deferred until host training stabilizes (warmup period) rather than smoothing noisy early signals. Recommend deferring to DRL specialist for the algorithm design choice; PyTorch implementation would be straightforward either way.

---

## Cross-Review (DRL Specialist)

| Field | Value |
|-------|-------|
| **Verdict** | `ENDORSE` |
| **Reviewer** | DRL Specialist |

**Evaluation:** This is a genuine concern in credit assignment literature -- non-stationarity from host adaptation creates a moving-baseline problem where Shapley values become noisy estimates of a non-stationary quantity. The EMA smoothing proposal (momentum=0.9) is the standard remedy and is cheap to implement. However, lifecycle decisions should already aggregate over multiple episodes per the existing design; verify that prune/fossilize thresholds use smoothed or aggregated Shapley values rather than single-episode point estimates. If they do, this becomes documentation-only; if not, promote to P2.
