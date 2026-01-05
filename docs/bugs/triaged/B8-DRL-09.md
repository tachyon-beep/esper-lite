# Finding Ticket: reward_mode_per_env Removed (Obsolete)

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B8-DRL-09` |
| **Severity** | `P3` |
| **Status** | `closed` |
| **Batch** | 8 |
| **Agent** | `drl` |
| **Domain** | `simic/training` |
| **Assignee** | |
| **Created** | 2025-12-27 |
| **Updated** | 2026-01-04 |

---

## Location

| Field | Value |
|-------|-------|
| **File(s)** | `src/esper/simic/training/config.py` |
| **Line(s)** | `N/A` |
| **Function/Class** | `TrainingConfig` |

---

## Summary

**One-line summary:** `reward_mode_per_env` (mixed reward-mode A/B) was removed; true A/B uses `--dual-ab` with separate policies.

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

This ticket is obsolete: the `reward_mode_per_env` surface no longer exists, so the validation gap cannot occur.

```python
# Line 433-445 (validation)
if self.reward_mode_per_env is not None:
    if len(self.reward_mode_per_env) != self.num_envs:
        raise ValueError(...)
    for mode in self.reward_mode_per_env:
        if not isinstance(mode, RewardMode):
            raise ValueError(...)
    # Missing: validation that modes are compatible with reward_family!
```

If `reward_family == RewardFamily.LOSS` but `reward_mode_per_env` contains `RewardMode.SPARSE`, the config validation passes but runtime will fail when computing rewards.

### Impact

- **Delayed error**: Fails at runtime instead of config creation
- **Confusing error message**: Runtime error won't point to config mismatch
- **Wasted computation**: Training starts before config error is caught

---

## Resolution

The mixed-mode A/B mechanism was removed. Use the dual-policy A/B runner (`--dual-ab`) to compare reward modes without shared normalization/state.

---

## Verification

### How to Verify the Fix

- [x] Confirm `reward_mode_per_env` is removed from `TrainingConfig`
- [x] Confirm the CLI rejects `--ab-test`

---

## Related Findings

- C8-01 in DRL report: chunk_length constraint (also config validation)

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch8-drl.md`
**Section:** "C8-03 - reward_mode_per_env not validated against reward_family"
