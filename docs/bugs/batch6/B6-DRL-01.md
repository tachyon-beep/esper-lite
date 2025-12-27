# Finding Ticket: Terminal Bonus Goodhart Risk - Same Bonus for 1% and 10% Improvement

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B6-DRL-01` |
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
| **Line(s)** | `790-802` |
| **Function/Class** | `compute_contribution_reward()` |

---

## Summary

**One-line summary:** Terminal fossilize bonus gives same +3.0 for 1% improvement seed as 10% improvement seed (Goodhart risk).

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
if epoch == max_epochs and not config.disable_terminal_reward:
    terminal_bonus = val_acc * config.terminal_acc_weight
    fossilize_terminal_bonus = num_contributing_fossilized * config.fossilize_terminal_scale
    terminal_bonus += fossilize_terminal_bonus
    reward += terminal_bonus
```

The `fossilize_terminal_scale=3.0` creates an incentive to maximize fossilized count. While `num_contributing_fossilized` filters for seeds with `total_improvement >= DEFAULT_MIN_FOSSILIZE_CONTRIBUTION` (1.0%), this threshold is binary:

- A seed that barely passes (1.0% improvement) gets +3.0
- A seed with 10% improvement also gets +3.0

This creates a Goodhart incentive to fossilize marginal seeds rather than high-value ones.

---

## Recommended Fix

Scale terminal bonus proportionally to total_improvement:

```python
# Instead of flat bonus per fossilized seed
fossilize_terminal_bonus = sum(
    min(
        seed.total_improvement * config.fossilize_contribution_scale,
        config.fossilize_terminal_scale
    )
    for seed in contributing_fossilized
)
```

This would make the terminal bonus proportional to actual value added, with a cap.

---

## Verification

### How to Verify the Fix

- [ ] Modify terminal bonus to scale with improvement
- [ ] Add property test verifying proportional scaling
- [ ] Verify anti-farming property still holds

---

## Related Findings

None.

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch6-drl.md`
**Section:** "P2-1: Terminal Bonus Analysis"

---

## Cross-Review (PyTorch Specialist)

| Field | Value |
|-------|-------|
| **Verdict** | `ENDORSE` |
| **Reviewer** | PyTorch Specialist |
| **Date** | 2024-12-27 |

**Evaluation:** The recommended fix introduces a `sum()` over potentially many seeds with `min()` clamping per seed. From a numerical stability perspective, this is safe since we are summing bounded positive floats. When these rewards are later converted to tensors at the training loop, there is no risk of overflow or precision loss. The proportional scaling is mathematically sound and the cap prevents reward explosion. No torch.compile concerns since this is pure Python float arithmetic.

---

## Cross-Review (DRL Specialist)

| Field | Value |
|-------|-------|
| **Verdict** | `ENDORSE` |
| **Reviewer** | DRL Specialist |

**Evaluation:** This is a valid Goodhart risk. Flat bonuses for binary thresholds create incentives to satisfy the threshold minimally rather than maximize the underlying objective (Ng et al., 1999 discusses similar specification gaming). The proposed fix to scale proportionally with total_improvement (capped) aligns rewards with actual value delivered, improving credit assignment. The P2 severity is appropriate since this could lead to suboptimal fossilization decisions over extended training, though the 1% threshold provides some protection against truly marginal seeds.

---

## Cross-Review (Code Review Specialist)

| Field | Value |
|-------|-------|
| **Verdict** | `ENDORSE` |
| **Reviewer** | Code Review Specialist |

**Evaluation:** Valid Goodhart risk identification. The code at line 797 applies a flat `fossilize_terminal_scale` (3.0) per contributing seed regardless of improvement magnitude. The recommended fix to scale proportionally with `total_improvement` (with a cap) is clean and aligns with the project's anti-gaming design philosophy. The fix should be implemented in conjunction with property tests to verify proportional scaling.