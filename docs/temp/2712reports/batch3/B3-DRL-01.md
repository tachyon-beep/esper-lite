# Finding Ticket: NormSeed param_estimate Incorrect

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B3-DRL-01` |
| **Severity** | `P1` |
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
| **File(s)** | `/home/john/esper-lite/src/esper/kasmina/blueprints/cnn.py` |
| **Line(s)** | `93-108` |
| **Function/Class** | `NormSeed` blueprint registration |

---

## Summary

**One-line summary:** NormSeed `param_estimate=100` is incorrect - actual is 129 for 64 channels (29% error), affecting rent economy signals.

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

The `param_estimate` for NormSeed is fixed at 100, but actual parameter count is `2*channels + 1` (GroupNorm gamma/beta + scale). For 64 channels: `2*64 + 1 = 129`, a 29% error.

### Why This Matters

The rent economy uses parameter counts as a cost signal to the policy. Inaccurate estimates could:
- Mislead the policy about the true cost of architectural choices
- Create reward hacking opportunities where underestimated seeds appear cheaper
- Cause inconsistent rent penalty computation

### Code Evidence

```python
# param_estimate=100 is fixed, but actual varies with channels
# For 64 channels: GroupNorm(groups, 64) = 2*64 + 1 scale = 129 params
```

---

## Recommended Fix

Either:
1. Make `param_estimate` a function of `dim` rather than a constant
2. Always use `actual_param_count()` in reward computation (with caching)

```python
# Option 1: Dynamic estimate
param_estimate=lambda dim: 2 * dim + 1

# Option 2: Document the fixed estimate
param_estimate=100  # Approximate for typical 64-channel case; use actual_param_count() for precise rent
```

---

## Verification

### How to Verify the Fix

- [ ] Add test comparing param_estimate vs actual_param_count for all blueprints
- [ ] Verify rent economy uses consistent parameter counts
- [ ] Test policy behavior with corrected estimates

---

## Related Findings

- B3-DRL-02: AttentionSeed param_estimate doesn't scale (65x error at high dim)
- B3-DRL-03: DepthwiseSeed param_estimate fixed vs scaling
- B3-DRL-11: TransformerNormSeed param_estimate fixed
- B3-CR-16: param_estimate accuracy documentation

---

## Cross-Review

| Reviewer | Verdict | Evaluation |
|----------|---------|------------|
| `code-review` | **OBJECT** | Severity overstated. The formula `2*channels + 1` is correct (GroupNorm gamma/beta + scale param), but P1 implies production impact. The rent economy uses estimates for policy guidance, not billing; 29% error at 64ch is tolerable. Recommend P3 with consolidated fix across B3-DRL-01/02/03 using dynamic `param_estimate` callbacks. |
| `drl-specialist` | **ENDORSE** | Confirmed: GroupNorm(n_groups, 64) = 2*64 gamma/beta + 1 scale = 129 params. 29% error distorts rent economy cost signals, potentially biasing policy toward NormSeeds. P1 severity appropriate given rent economy is a core training signal. |
| `pytorch` | **ENDORSE** | Confirmed correctness issue. GroupNorm(g, C) has exactly `2*C` affine params (gamma/beta), plus 1 for the learned scale, totaling `2*channels + 1`. For 64 channels: 129 actual vs 100 estimated = 29% error. This directly affects rent economy cost signals; fix should compute `2 * dim + 1` dynamically. |

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch3-drl.md`
**Section:** "P1 - Correctness"
