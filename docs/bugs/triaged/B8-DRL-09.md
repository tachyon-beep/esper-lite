# Finding Ticket: reward_mode_per_env Not Validated Against reward_family

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B8-DRL-09` |
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
| **File(s)** | `/home/john/esper-lite/src/esper/simic/training/config.py` |
| **Line(s)** | `433-445` |
| **Function/Class** | `TrainingConfig._validate()` |

---

## Summary

**One-line summary:** `reward_mode_per_env` validates length and type but not compatibility with `reward_family`.

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

## Recommended Fix

Add cross-validation:

```python
if self.reward_mode_per_env is not None:
    # ... existing checks ...

    # Validate compatibility with reward_family
    valid_modes = get_valid_modes_for_family(self.reward_family)
    for mode in self.reward_mode_per_env:
        if mode not in valid_modes:
            raise ValueError(
                f"RewardMode.{mode.name} is not valid for "
                f"RewardFamily.{self.reward_family.name}. "
                f"Valid modes: {[m.name for m in valid_modes]}"
            )
```

---

## Verification

### How to Verify the Fix

- [ ] Add reward_family/mode compatibility validation
- [ ] Add test for incompatible mode/family combinations
- [ ] Verify helpful error message is raised

---

## Related Findings

- C8-01 in DRL report: chunk_length constraint (also config validation)

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch8-drl.md`
**Section:** "C8-03 - reward_mode_per_env not validated against reward_family"
