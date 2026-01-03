# Finding Ticket: Synergy Bonus Not Protected by Anti-Stacking Logic

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B6-DRL-03` |
| **Severity** | `P3` |
| **Status** | `closed` |
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
| **Line(s)** | `714-720` |
| **Function/Class** | `compute_contribution_reward()` |

---

## Summary

**One-line summary:** Synergy bonus is added to reward without anti-stacking protection, allowing ransomware seeds to receive it.

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

The reward function has anti-stacking logic that skips `ratio_penalty` and `holding_warning` when `attribution_discount < 0.5`. However, `synergy_bonus` is added without this protection:

```python
# Anti-stacking: skip ratio_penalty when attribution already zeroed
if attribution_discount >= 0.5:
    ratio_penalty = _compute_ratio_penalty(...)

# ...

# Synergy bonus added WITHOUT anti-stacking check
synergy_bonus = _compute_synergy_bonus(...)
reward += synergy_bonus  # Ransomware seed could still receive this
```

A ransomware seed (negative total_improvement, high contribution) could still receive synergy bonus if it has positive `interaction_sum` with other seeds.

---

## Recommended Fix

Gate synergy bonus on positive attribution:

```python
# Only award synergy bonus if seed is providing legitimate value
if attribution_discount >= 0.5 and bounded_attribution > 0:
    synergy_bonus = _compute_synergy_bonus(
        interaction_sum=seed_info.interaction_sum,
        boost_received=seed_info.boost_received,
        synergy_weight=config.synergy_weight,
    )
    reward += synergy_bonus
```

---

## Verification

### How to Verify the Fix

- [ ] Add anti-stacking gate to synergy bonus
- [ ] Add property test: ransomware seed gets synergy_bonus=0
- [ ] Verify existing synergy tests still pass

---

## Related Findings

- B6-CR-01: Component sum tests missing synergy_bonus
- B6-CR-02: boost_received parameter unused

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch6-drl.md`
**Section:** "P3-3: Synergy Bonus Not Protected by Anti-Stacking"

---

## Cross-Review (PyTorch Specialist)

| Field | Value |
|-------|-------|
| **Verdict** | `ENDORSE` |
| **Reviewer** | PyTorch Specialist |
| **Date** | 2024-12-27 |

**Evaluation:** The proposed conditional gating adds a branch but this occurs in pure Python before tensor conversion. From a torch.compile perspective, reward computation happens outside the compiled graph (rewards are computed as floats, then converted to tensors for the training loop). The fix has no impact on compilation, memory, or numerical stability. The logic change is sound and prevents reward signal corruption that could destabilize policy gradient updates downstream.

---

## Cross-Review (DRL Specialist)

| Field | Value |
|-------|-------|
| **Verdict** | `ENDORSE` |
| **Reviewer** | DRL Specialist |

**Evaluation:** This is a genuine reward shaping vulnerability. Ransomware seeds (negative total_improvement despite high contribution) receiving synergy bonus creates a reward hacking vector where seeds could game the interaction_sum metric while providing net-negative value. This violates the anti-stacking design intent and introduces misaligned credit assignment. The proposed fix gating synergy on `attribution_discount >= 0.5 and bounded_attribution > 0` correctly closes the loophole. Consider **upgrading to P2** since this enables a concrete reward hacking path that could compound during extended training.

---

## Cross-Review (Code Review Specialist)

| Field | Value |
|-------|-------|
| **Verdict** | `ENDORSE` |
| **Reviewer** | Code Review Specialist |

**Evaluation:** Valid finding with clear code evidence. Lines 714-721 show `synergy_bonus` is computed and added to reward unconditionally when `seed_info is not None`, without the `attribution_discount >= 0.5` gate applied to `ratio_penalty` and `holding_warning`. The recommended fix to gate synergy bonus on positive attribution is consistent with the anti-stacking design pattern already established in the function. Should be fixed to prevent ransomware seeds from receiving synergy rewards.

---

## Resolution

### Status: FIXED

**Fixed by adding anti-stacking gate to synergy bonus computation.**

#### The Fix (rewards.py line 724)

```python
# Before:
if seed_info is not None:
    synergy_bonus = _compute_synergy_bonus(seed_info.interaction_sum)

# After:
if seed_info is not None and attribution_discount >= 0.5 and bounded_attribution > 0:
    synergy_bonus = _compute_synergy_bonus(seed_info.interaction_sum)
```

#### DRL Expert Review

The fix was reviewed and **APPROVED** by the DRL expert agent:
- `attribution_discount >= 0.5` threshold is correct (matches ratio_penalty)
- `bounded_attribution > 0` check ensures genuine contribution
- No edge cases where legitimate seeds lose synergy bonus incorrectly
- `disable_anti_gaming` flag not needed (synergy is scaffolding, not penalty)

#### Tests Added

- `TestRansomwarePattern.test_ransomware_no_synergy_bonus` - property test verifying ransomware seeds receive `synergy_bonus == 0.0`

#### Verification Checklist

- [x] Add anti-stacking gate to synergy bonus
- [x] Add property test: ransomware seed gets synergy_bonus=0
- [x] Verify existing synergy tests still pass (6/6 passed)

#### Severity Confirmation

- Original: P3 (API design / contract violation)
- Confirmed: P3 (appropriate for this reward hacking vulnerability)