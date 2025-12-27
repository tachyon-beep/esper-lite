# Finding Ticket: TransformerNormSeed param_estimate Fixed

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B3-DRL-11` |
| **Severity** | `P3` |
| **Status** | `open` |
| **Batch** | 3 |
| **Agent** | `drl` |
| **Domain** | `kasmina` |
| **Assignee** | |
| **Created** | 2024-12-27 |
| **Updated** | 2024-12-27 |

---

## Location

| Field | Value |
|-------|-------|
| **File(s)** | `/home/john/esper-lite/src/esper/kasmina/blueprints/transformer.py` |
| **Line(s)** | `24-38` |
| **Function/Class** | `TransformerNormSeed` blueprint |

---

## Summary

**One-line summary:** TransformerNormSeed `param_estimate=800` is reasonable for typical dims but doesn't scale (LayerNorm has 2*dim params).

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

The estimate of 800 is close for dim=384 (`2*384 + 1 = 769`) but doesn't scale with dimension.

### Formula

LayerNorm(dim) has `2*dim` parameters (gamma + beta) plus 1 scale = `2*dim + 1`

### Impact

Lower severity than CNN attention seed because the error is small and scales linearly, not quadratically.

---

## Recommended Fix

Document or make dynamic:

```python
param_estimate=800  # Approximate for dim~384; actual is 2*dim+1
```

---

## Verification

### How to Verify the Fix

- [ ] Add test comparing estimate vs actual

---

## Related Findings

- B3-DRL-01: NormSeed param_estimate incorrect
- B3-DRL-02: AttentionSeed param_estimate doesn't scale

---

## Cross-Review: PyTorch Expert

| Verdict | Evaluation |
|---------|------------|
| **NEUTRAL** | Estimate 800 for dim=384 yields 769 actual (2*384+1) - 4% error is acceptable for allocation heuristics. LayerNorm is cheap; this estimate affects only capacity planning, not correctness. Add clarifying comment but low priority. |

## Cross-Review: Code Reviewer

| Verdict | Evaluation |
|---------|------------|
| **NEUTRAL** | Agree with PT expert - 4% overestimate is acceptable for allocation planning. The param_estimate is a hint, not a constraint; a clarifying comment suffices. |

## Cross-Review: DRL Specialist

| Verdict | Evaluation |
|---------|------------|
| **ENDORSE** | The `param_estimate` is used for blueprint selection heuristics not memory allocation; a fixed 800 is within 5% for dim=384 (actual 769) and the error is O(dim) not O(dim^2), making this low-impact for G2 gate decisions. |

**Recommendation:** Accept P3 severity. Add inline comment `# Approx for dim~384; actual is 2*dim+1` rather than dynamic calculation.

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch3-drl.md`
**Section:** "P3 - Code Quality" (B3-TFM-05)
