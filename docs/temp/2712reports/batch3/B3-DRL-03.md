# Finding Ticket: DepthwiseSeed param_estimate Fixed vs Scaling

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B3-DRL-03` |
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
| **File(s)** | `/home/john/esper-lite/src/esper/kasmina/blueprints/cnn.py` |
| **Line(s)** | `166-186` |
| **Function/Class** | `DepthwiseSeed` blueprint |

---

## Summary

**One-line summary:** DepthwiseSeed `param_estimate=4800` is correct for dim=64 but formula is `channels*9 + channels² + 2*channels`.

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

The `param_estimate=4800` is correct for 64 channels:
- Depthwise conv(64,64,3,3) = 64*9 = 576
- Pointwise conv(64,64) = 64² = 4096
- GroupNorm ≈ 2*64 = 128
- Total: 576 + 4096 + 128 = 4800 ✓

But the estimate is presented as fixed when it scales with channels.

### Impact

Lower severity than attention seed because the scaling is more linear, but still affects rent economy for non-standard dimensions.

---

## Recommended Fix

Document the formula or make it dynamic:

```python
# Formula: channels*9 + channels² + 2*channels
param_estimate=4800  # For dim=64; scales quadratically with channels
```

---

## Verification

### How to Verify the Fix

- [ ] Add docstring explaining the formula
- [ ] Consider dynamic estimate for large channels

---

## Related Findings

- B3-DRL-01: NormSeed param_estimate incorrect
- B3-DRL-02: AttentionSeed param_estimate doesn't scale

---

## Cross-Review

| Reviewer | Verdict | Evaluation |
|----------|---------|------------|
| `code-review` | **NEUTRAL** | Formula in ticket is correct: depthwise (c*9) + pointwise (c^2) + GroupNorm (2c) = c*9 + c^2 + 2c. For 64ch: 576 + 4096 + 128 = 4800. Ticket marked P3/Documentation but the actual code is correct; this is purely a "formula should be documented" issue. Consider consolidating with B3-DRL-01/02 as a single "dynamic param_estimate" refactor. |
| `drl-specialist` | **ENDORSE** | Estimate is exact for 64 channels but formula shows O(channels^2) scaling via pointwise conv. Lower severity than B3-DRL-02 since depthwise is less likely selected at high dims, but documenting the formula prevents future confusion. |
| `pytorch` | **NEUTRAL** | Formula confirmed: depthwise Conv2d(C,C,3,groups=C) = `C*9`, pointwise Conv2d(C,C,1) = `C*C`, GroupNorm(g,C) = `2*C`. Total = `C^2 + 11*C`. For 64ch: 4800 exact. The `C^2` term dominates at higher dims, so documenting the formula is useful but not urgent if channels are fixed at 64. |

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch3-drl.md`
**Section:** "P3 - Code Quality"
