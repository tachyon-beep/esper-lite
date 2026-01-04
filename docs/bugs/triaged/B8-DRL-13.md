# Finding Ticket: A/B Winner Determination Lacks Statistical Significance

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B8-DRL-13` |
| **Severity** | `P4` |
| **Status** | `open` |
| **Batch** | 8 |
| **Agent** | `drl` |
| **Domain** | `simic/training` |
| **Assignee** | |
| **Created** | 2025-12-27 |
| **Updated** | 2025-12-27 |

---

## Location

| Field | Value |
|-------|-------|
| **File(s)** | `/home/john/esper-lite/src/esper/simic/training/dual_ab.py` |
| **Line(s)** | `232-294` |
| **Function/Class** | `_print_dual_ab_comparison()` |

---

## Summary

**One-line summary:** Winner is determined by simple final accuracy comparison without statistical significance testing.

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
# Lines 232-294 (simplified)
final_acc_a = history_a[-1]['avg_accuracy']
final_acc_b = history_b[-1]['avg_accuracy']
winner = "A" if final_acc_a > final_acc_b else "B"
print(f"Winner: Group {winner}")
```

This determination:
1. Uses single final value, not distribution of episode rewards
2. No confidence interval or p-value reported
3. Can declare winner even when difference is within noise

### Impact

- **False conclusions**: May select wrong reward mode due to noise
- **No confidence measure**: Can't assess reliability of winner
- **Scientific rigor**: Doesn't meet standards for A/B testing

---

## Recommended Fix

Add statistical significance testing:

```python
from scipy.stats import mannwhitneyu

# Compare episode reward distributions
stat, p_value = mannwhitneyu(
    rewards_a[-10:],  # Last 10 episodes
    rewards_b[-10:],
    alternative='two-sided'
)

# Report with confidence
mean_a, std_a = np.mean(rewards_a[-10:]), np.std(rewards_a[-10:])
mean_b, std_b = np.mean(rewards_b[-10:]), np.std(rewards_b[-10:])

print(f"Group A: {mean_a:.2f} ± {1.96*std_a/np.sqrt(10):.2f} (95% CI)")
print(f"Group B: {mean_b:.2f} ± {1.96*std_b/np.sqrt(10):.2f} (95% CI)")
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    winner = "A" if mean_a > mean_b else "B"
    print(f"Winner: Group {winner} (statistically significant)")
else:
    print("No statistically significant difference")
```

---

## Verification

### How to Verify the Fix

- [ ] Add statistical significance testing (Mann-Whitney U or t-test)
- [ ] Report confidence intervals
- [ ] Add warning when difference is not significant

---

## Related Findings

- B8-DRL-03: Sequential training bias (confounds A/B comparison)
- B8-DRL-12: MD5 seed offset fragility

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch8-drl.md`
**Section:** "C8-06 - _print_dual_ab_comparison() uses simple final accuracy"
