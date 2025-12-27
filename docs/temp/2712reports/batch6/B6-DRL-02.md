# Finding Ticket: PBRS Telescoping Comment Misleading for Gamma < 1

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B6-DRL-02` |
| **Severity** | `P2` |
| **Status** | `open` |
| **Batch** | 6 |
| **Agent** | `drl` |
| **Domain** | `simic/rewards` |
| **Assignee** | |
| **Created** | 2024-12-27 |
| **Updated** | 2024-12-27 |

---

## Location

| Field | Value |
|-------|-------|
| **File(s)** | `/home/john/esper-lite/src/esper/simic/rewards/rewards.py` |
| **Line(s)** | `70-114` |
| **Function/Class** | Module-level comment block |

---

## Summary

**One-line summary:** PBRS comment claims "telescoping" but gamma < 1 breaks exact cancellation; comment should clarify.

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

The comment block describes PBRS (Potential-Based Reward Shaping) and claims "telescoping" where shaped rewards cancel out over an episode. However, with gamma=0.995 < 1:

**Exact PBRS formula:** `F(s, s') = gamma * phi(s') - phi(s)`

**Telescoping only works perfectly for gamma=1:**
```
Sum of F over episode = gamma^T * phi(s_T) - phi(s_0)
```

With gamma=0.995 and T=25: gamma^25 = 0.882, so there's ~12% "leakage" from the telescoping property.

The property tests acknowledge this with relaxed tolerance (`tolerance = 2.0 + 1.0 * T`), but the comment doesn't clarify that exact telescoping requires gamma=1.

---

## Recommended Fix

Clarify in the comment block:

```python
# PBRS Telescoping Property:
# For gamma=1.0: sum of shaped rewards exactly equals phi(s_T) - phi(s_0)
# For gamma<1.0: introduces bounded error of (1-gamma^T) * phi(s_T)
#                With gamma=0.995, T=25: ~12% approximation error
#                This is acceptable and tested (see test_pbrs_properties.py)
```

---

## Verification

### How to Verify the Fix

- [ ] Update comment to clarify gamma < 1 behavior
- [ ] No code change needed
- [ ] Documentation-only fix

---

## Related Findings

- B6-PT-01: PBRS warning continues with potentially wrong values

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch6-drl.md`
**Section:** "P2-2: PBRS Telescoping with gamma < 1"

---

## Cross-Review (PyTorch Specialist)

| Field | Value |
|-------|-------|
| **Verdict** | `ENDORSE` |
| **Reviewer** | PyTorch Specialist |
| **Date** | 2024-12-27 |

**Evaluation:** The gamma^T calculation (0.995^25 = 0.882) is numerically stable with no precision concerns in float32 or float64. The ~12% "leakage" the ticket identifies is a fundamental property of discounted PBRS, not a numerical artifact. From a PyTorch perspective, clarifying this in documentation prevents future maintainers from incorrectly "fixing" the approximation error by adjusting tolerances or adding numerical corrections. The relaxed test tolerance is appropriate given the mathematical reality.

---

## Cross-Review (DRL Specialist)

| Field | Value |
|-------|-------|
| **Verdict** | `ENDORSE` |
| **Reviewer** | DRL Specialist |

**Evaluation:** The analysis is technically correct. PBRS guarantees policy invariance for gamma=1 (Ng et al., 1999), but with gamma<1 the telescoping introduces bounded error proportional to (1-gamma^T). With gamma=0.995 and T=25, the ~12% leakage is non-trivial but acceptable in practice since the shaped rewards still provide useful learning signal without introducing optimality-changing bias. However, I would **downgrade to P3**: this is a documentation clarification, the code behavior is correct and the tests already account for this with relaxed tolerances. The comment update is valuable for future maintainers but not urgent.

---

## Cross-Review (Code Review Specialist)

| Field | Value |
|-------|-------|
| **Verdict** | `NEUTRAL` |
| **Reviewer** | Code Review Specialist |

**Evaluation:** The technical analysis is correct: with gamma=0.995, exact telescoping only holds for discounted returns, not undiscounted sums. However, reviewing lines 70-114 shows the existing comment (lines 78-81) already clarifies this: "The undiscounted sum of per-step PBRS bonuses differs from this value when gamma < 1, but the effect on optimal actions is unchanged." The existing documentation is accurate but could be more explicit about the ~12% error magnitude. This is a minor documentation enhancement, not a P2 issue.