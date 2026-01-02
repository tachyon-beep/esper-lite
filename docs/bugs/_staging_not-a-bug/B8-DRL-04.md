# Finding Ticket: Rollback Buffer Handling After Transitions Added

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B8-DRL-04` |
| **Severity** | `P2` |
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
| **Line(s)** | `3183-3188` |
| **Function/Class** | `train_ppo_vectorized()` |

---

## Summary

**One-line summary:** When governor rollback occurs, buffer is cleared AFTER transitions were added, leaving stale GAE values.

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

```python
# Transitions are added (lines 3141-3175)
agent.buffer.add_transition(...)

# Later, if governor panic occurs:
if panic:
    agent.buffer.clear_env(env_idx)  # Clears AFTER adding
```

The problem:
1. Transitions from the rolled-back environment are added to the buffer
2. GAE (Generalized Advantage Estimation) is computed using values from that environment
3. THEN the buffer is cleared due to rollback
4. But GAE values for OTHER environments may have been computed using the now-invalid data

### Impact

- **Stale advantages**: PPO updates use incorrect advantage estimates
- **Learning instability**: Policy gradient has wrong sign/magnitude
- **Intermittent**: Only occurs during governor interventions (safety mechanism)

---

## Recommended Fix

**Option 1 - Don't add transitions from rolled-back envs:**
```python
# Mark env as invalid BEFORE adding transitions
env_panicked = check_panic(...)
if not env_panicked:
    agent.buffer.add_transition(...)
```

**Option 2 - Recompute GAE after rollback:**
```python
if any_rollback:
    agent.buffer.recompute_advantages_excluding(rolled_back_envs)
```

**Option 3 - Flag transitions for exclusion:**
```python
agent.buffer.add_transition(..., valid=not env_panicked)
# PPO update skips invalid transitions
```

---

## Verification

### How to Verify the Fix

- [ ] Identify all rollback trigger points
- [ ] Ensure buffer consistency before and after rollback
- [ ] Add test with simulated governor panic
- [ ] Verify GAE values are consistent after rollback

---

## Related Findings

None.

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch8-drl.md`
**Section:** "C8-29 - Rollback clears buffer after transitions added"

**Note:** Governor rollback is a rare safety mechanism, so this may not affect typical training runs. However, the correctness concern remains.
