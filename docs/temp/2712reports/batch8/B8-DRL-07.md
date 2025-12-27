# Finding Ticket: Hindsight Credit Cap Applied Before Accumulation

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B8-DRL-07` |
| **Severity** | `P3` |
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
| **File(s)** | `/home/john/esper-lite/src/esper/simic/training/vectorized.py` |
| **Line(s)** | `2734-2736` |
| **Function/Class** | `train_ppo_vectorized()` |

---

## Summary

**One-line summary:** Hindsight credit is capped per-fossilization, but accumulated total can exceed cap.

**Category:**
- [ ] Correctness bug
- [ ] Race condition / concurrency
- [ ] Memory leak / resource issue
- [ ] Performance bottleneck
- [x] Numerical stability
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

```python
# Line 2734-2736
total_credit = min(total_credit, MAX_HINDSIGHT_CREDIT)
pending_hindsight_credit[env_idx] += total_credit
```

The cap is applied to `total_credit` BEFORE adding to the pending accumulator. If multiple fossilizations occur in the same episode:
1. Fossilization 1: 0.8 credit (capped from 1.2) → pending = 0.8
2. Fossilization 2: 0.8 credit (capped from 1.5) → pending = 1.6 (exceeds 1.0 cap!)

### Impact

- **Reward explosion**: Multiple fossilizations can accumulate unbounded credit
- **Training instability**: Large credit values bias the policy
- **Cap ineffective**: The intended safeguard is bypassed

---

## Recommended Fix

Cap at the point of application, not accumulation:

```python
# When applying to reward (line 2649):
hindsight_credit_applied = min(
    pending_hindsight_credit[env_idx],
    MAX_HINDSIGHT_CREDIT
)
reward += hindsight_credit_applied
pending_hindsight_credit[env_idx] = 0  # Reset after applying
```

Or cap the accumulated total:

```python
# After accumulation:
pending_hindsight_credit[env_idx] = min(
    pending_hindsight_credit[env_idx] + total_credit,
    MAX_HINDSIGHT_CREDIT
)
```

---

## Verification

### How to Verify the Fix

- [ ] Cap at application point or on accumulated total
- [ ] Add test with multiple fossilizations per episode
- [ ] Verify credit never exceeds MAX_HINDSIGHT_CREDIT

---

## Related Findings

None.

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch8-drl.md`
**Section:** "C8-20 - Hindsight credit cap applied before accumulation"
