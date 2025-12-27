# Finding Ticket: stage_bonus Field Never Set in Telemetry

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B6-DRL-04` |
| **Severity** | `P3` |
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
| **File(s)** | `/home/john/esper-lite/src/esper/simic/rewards/reward_telemetry.py` |
| **Line(s)** | `37` |
| **Function/Class** | `RewardComponentsTelemetry` |

---

## Summary

**One-line summary:** `stage_bonus` field is defined but never set by any reward function; always 0.0.

**Category:**
- [ ] Correctness bug
- [ ] Race condition / concurrency
- [ ] Memory leak / resource issue
- [ ] Performance bottleneck
- [ ] Numerical stability
- [ ] torch.compile compatibility
- [x] Dead code / unwired functionality
- [ ] API design / contract violation
- [ ] Test coverage gap
- [ ] Documentation / naming
- [ ] Defensive programming violation
- [ ] Legacy code policy violation

---

## Detailed Description

### What's Wrong

```python
@dataclass(slots=True)
class RewardComponentsTelemetry:
    # ...
    stage_bonus: float = 0.0  # Never set anywhere
```

The `stage_bonus` field exists in the telemetry dataclass but:
1. No reward function sets it
2. It's always 0.0
3. It's included in `shaped_reward_ratio` calculation, adding noise

This appears to be dead code from an older reward design.

---

## Recommended Fix

Either:

**Option 1 - Remove (preferred per No Legacy Code Policy):**
```python
# Remove stage_bonus from dataclass
# Remove from shaped_reward_ratio calculation
# Remove from to_dict()
```

**Option 2 - Set appropriately:**
If there's a valid use case, set it in `compute_contribution_reward`:
```python
components.stage_bonus = some_value
```

---

## Verification

### How to Verify the Fix

- [ ] Search codebase for stage_bonus assignments
- [ ] If none found, remove the field
- [ ] Update shaped_reward_ratio calculation
- [ ] Update to_dict() method

---

## Related Findings

- B6-PT-02: shaped_reward_ratio includes unset fields
- B6-PT-04: base_acc_delta is also dead code
- B6-PT-05: hindsight_credit is also dead code

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch6-drl.md`
**Section:** "P3-4: stage_bonus field never set"

**Also identified by:**
- PyTorch Report P3-3

---

## Cross-Review (PyTorch Specialist)

| Field | Value |
|-------|-------|
| **Verdict** | `ENDORSE` |
| **Reviewer** | PyTorch Specialist |
| **Date** | 2024-12-27 |

**Evaluation:** Dead fields in telemetry dataclasses add noise to reward analysis and can mislead debugging efforts when investigating training instability. The `shaped_reward_ratio` calculation including a perpetual 0.0 value slightly distorts the ratio. Per the No Legacy Code Policy, removal is the correct fix. From a PyTorch perspective, eliminating dead fields reduces memory footprint when telemetry is batched and simplifies any future tensor conversion of telemetry data for TensorBoard logging.

---

## Cross-Review (DRL Specialist)

| Field | Value |
|-------|-------|
| **Verdict** | `ENDORSE` |
| **Reviewer** | DRL Specialist |

**Evaluation:** Dead fields in reward telemetry are problematic for two DRL-specific reasons: (1) they add noise to the shaped_reward_ratio metric which may be used for reward diagnostics, and (2) they make it harder to reason about which reward components are actually driving policy updates. Per the No Legacy Code Policy, removal is correct. However, verify whether `stage_bonus` was intended for seed lifecycle stage rewards (e.g., bonus for advancing from TRAINING to BLENDING) before deleting, as this could be a missing feature rather than dead code.

---

## Cross-Review (Code Review Specialist)

| Field | Value |
|-------|-------|
| **Verdict** | `ENDORSE` |
| **Reviewer** | Code Review Specialist |

**Evaluation:** Verified dead code. The `stage_bonus` field at line 37 of `reward_telemetry.py` defaults to 0.0 and grep confirms no assignment anywhere in the codebase. Per the No Legacy Code Policy, Option 1 (removal) is the correct action. The field should be deleted along with any references in `shaped_reward_ratio` and `to_dict()`. This is a straightforward cleanup aligned with project policy to delete unused code completely.