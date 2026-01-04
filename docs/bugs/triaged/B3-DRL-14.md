# Finding Ticket: Gradient Ratio Normalization May Disadvantage Small Seeds

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B3-DRL-14` |
| **Severity** | `P2` |
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
| **File(s)** | `/home/john/esper-lite/src/esper/kasmina/slot.py` |
| **Line(s)** | `1798-1827` |
| **Function/Class** | `SeedSlot` gradient ratio computation |

---

## Summary

**One-line summary:** The gradient ratio formula `(seed_norm / host_norm) * sqrt(host_params / seed_params)` assumes equal per-parameter gradient magnitude, which may disadvantage certain architectures.

**Category:**
- [ ] Correctness bug
- [ ] Race condition / concurrency
- [ ] Memory leak / resource issue
- [x] Performance bottleneck
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

The normalization formula scales up small seeds and down large ones using `sqrt(host_params / seed_params)`. However, this assumes equal gradient magnitude per parameter, which isn't true across architectures.

### Example

A small attention seed may have naturally higher per-param gradients than a large MLP. The sqrt scaling could cause the G2 gate to prefer certain architectures over others for non-fundamental reasons.

### Code Evidence

```python
# /home/john/esper-lite/src/esper/kasmina/slot.py:1820-1821

normalization_factor = (host_params / seed_params) ** 0.5
ratio = raw_ratio * normalization_factor
```

### Impact

- G2 gate decisions may have architecture bias
- Small but architecturally valuable seeds may fail G2
- Large inefficient seeds may pass G2

---

## Recommended Fix

Consider alternative normalization strategies:

1. **Per-layer normalization**: Normalize within layer types
2. **Gradient-per-parameter**: Use `sum(|g|) / num_params` instead of norm
3. **Architecture-specific thresholds**: Different G2 thresholds per blueprint type

Document current behavior as a known limitation.

---

## Verification

### How to Verify the Fix

- [ ] Compare G2 pass rates across different blueprint types
- [ ] Analyze gradient magnitude distribution per architecture
- [ ] A/B test different normalization strategies

---

## Related Findings

- B3-PT-01: Division by zero in gradient ratio for noop seeds (related edge case)

---

## Cross-Review: PyTorch Expert

| Verdict | Evaluation |
|---------|------------|
| **NEUTRAL** | The sqrt scaling `(host_params/seed_params)^0.5` is a reasonable heuristic for L2-norm-based gradient comparison across different parameter counts. Alternative: use mean gradient magnitude `sum(|g|)/n` for true per-param comparison. However, this is an RL design decision (G2 gate policy), not a PyTorch correctness issue. Defer to DRL expert for threshold tuning. |

## Cross-Review: Code Reviewer

| Verdict | Evaluation |
|---------|------------|
| **ENDORSE** | Valid concern - sqrt scaling assumes uniform per-param gradient magnitude which doesn't hold across architectures (attention vs LayerNorm). Needs empirical G2 pass rate analysis by blueprint type before tuning. |

## Cross-Review: DRL Specialist

| Verdict | Evaluation |
|---------|------------|
| **ENDORSE** | The sqrt scaling is principled (L2 norm scales with sqrt(n) for random vectors), but the ticket correctly identifies that gradient magnitude varies by architecture. This is a real concern for G2 gate fairness across blueprint types in multi-seed RL scenarios. |

**Recommendation:** Accept P2. The concern is theoretically valid. Suggested mitigation: track gradient-ratio distributions per blueprint type in telemetry before changing the formula. A/B testing normalization strategies requires empirical data we don't have yet.

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch3-drl.md`
**Section:** "P2 - Performance/Safety" (B3-SLOT-01)
