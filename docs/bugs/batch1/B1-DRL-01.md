# Finding Ticket: Death Penalty Never Injected into PPO Buffer

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B1-DRL-01` |
| **Severity** | `P1` |
| **Status** | `open` |
| **Batch** | 1 |
| **Agent** | `drl` |
| **Domain** | `tolaria` / `simic` |
| **Assignee** | |
| **Created** | 2024-12-27 |
| **Updated** | 2024-12-27 |

---

## Location

| Field | Value |
|-------|-------|
| **File(s)** | `/home/john/esper-lite/src/esper/tolaria/governor.py`, `/home/john/esper-lite/src/esper/simic/training/vectorized.py` |
| **Line(s)** | governor.py: `338-340`, vectorized.py: `2547-2552`, `3186-3188` |
| **Function/Class** | `TolariaGovernor.get_punishment_reward()`, `train_ppo_vectorized()` |

---

## Summary

**One-line summary:** The `death_penalty` punishment reward is defined but never injected into the PPO buffer - the RL agent cannot learn to avoid rollback.

**Category:**
- [x] Correctness bug
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

The Governor has a `death_penalty` parameter (default 10.0) and a `get_punishment_reward()` method that returns `-death_penalty`. The intent is clearly to punish the RL agent when its actions cause catastrophic training failure.

However, when rollback occurs:
1. `execute_rollback()` is called to restore model state
2. The PPO buffer for that environment is **completely cleared**
3. The death penalty is **never injected** into the buffer
4. The agent never sees any negative reward signal from the rollback

### Code Evidence

```python
# /home/john/esper-lite/src/esper/tolaria/governor.py:338-340
def get_punishment_reward(self) -> float:
    """Return the punishment reward for a catastrophic failure."""
    return -self.death_penalty  # Returns -10.0 by default

# /home/john/esper-lite/src/esper/simic/training/vectorized.py:2547-2552
if env_idx in governor_panic_envs:
    env_state.governor.execute_rollback(
        env_id=env_idx, optimizer=env_state.host_optimizer
    )
    env_rollback_occurred[env_idx] = True

# Later at lines 3186-3188
if rollback_env_indices:
    for env_idx in rollback_env_indices:
        agent.buffer.clear_env(env_idx)  # All transitions discarded!
```

### Why This Matters

**Credit Assignment Failure:**
- The agent cannot learn the causal relationship between actions and rollback
- The `death_penalty` parameter is effectively dead code
- Actions that cause catastrophic failure have NO negative signal

**Sample Inefficiency:**
- All training data from the episode is discarded
- No learning signal is extracted from the failure

**Potential Reward Hacking:**
- Agent might discover actions that are locally "good" but cause eventual collapse
- Without punishment, agent has no incentive to avoid these patterns

---

## System Context

This violates the **Governor prevents catastrophe** principle. The Governor should not just prevent catastrophe - it should teach the RL agent to avoid catastrophe through negative reward signals.

**Relevant architectural principles:**
- [ ] Signal-to-Noise Hypothesis (sensors match capabilities)
- [ ] Rent Economy (complexity pays rent)
- [ ] Inverted Control Flow (GPU-first)
- [x] Governor prevents catastrophe - **PARTIALLY VIOLATED**
- [ ] No defensive programming policy
- [ ] No legacy code policy

---

## Recommended Fix

### Option A: Inject terminal punishment before clearing (Recommended)

```python
# /home/john/esper-lite/src/esper/simic/training/vectorized.py

if rollback_env_indices:
    for env_idx in rollback_env_indices:
        punishment = env_states[env_idx].governor.get_punishment_reward()
        agent.buffer.inject_terminal_reward(env_idx, punishment, done=True)
    for env_idx in rollback_env_indices:
        agent.buffer.clear_env(env_idx)
```

Note: This requires adding an `inject_terminal_reward()` method to the rollout buffer.

### Option B: Keep last N transitions with punishment

Instead of clearing entirely, keep the last few transitions before rollback and attach the death penalty as a terminal reward. This preserves some credit assignment signal.

### Option C: Model rollback as explicit terminal action

Treat rollback as an automatic terminal action with fixed negative reward, separate from the agent's chosen action.

---

## Verification

### How to Verify the Fix

- [ ] Unit test: Verify `get_punishment_reward()` returns expected value
- [ ] Unit test: Verify buffer contains punishment reward after rollback
- [ ] Integration test: Train with intentional rollback triggers, verify agent learns to avoid
- [ ] Check: `death_penalty` parameter is no longer dead code

### Regression Risk

- Changing buffer behavior after rollback could affect training dynamics
- Need to verify PPO update handles the modified buffer correctly

### Test Files to Update

- `tests/tolaria/test_governor.py` - test `get_punishment_reward()` integration
- `tests/simic/test_rollout_buffer.py` - test `inject_terminal_reward()` if added

---

## Related Findings

| Ticket ID | Relationship | Notes |
|-----------|--------------|-------|
| `B1-PT-01` | `related` | Optimizer state after rollback is also questionable |

---

## Cross-Review

| Agent | Verdict | Evaluation |
|-------|---------|------------|
| **DRL** | ENDORSE | This is a critical credit assignment failure - the agent receives zero negative signal for actions causing catastrophic rollback, completely breaking the intended punishment mechanism. Option A (inject terminal punishment before clearing) is the correct fix; without it, the death_penalty parameter is dead code and the agent cannot learn to avoid collapse-inducing policies. |
| **PyTorch** | ENDORSE | Critical RL correctness issue but also a PyTorch concern: clearing the buffer discards all tensor data without extracting the learning signal. The `inject_terminal_reward()` approach should ensure proper tensor device placement and avoid graph breaks if buffer operations are compiled. |
| **CodeReview** | ENDORSE | Critical finding - `get_punishment_reward()` is indeed dead code since `clear_env()` discards all transitions without injecting the death penalty. This violates "Governor prevents catastrophe" principle; the RL agent genuinely cannot learn to avoid rollback-causing behaviors. P1 is warranted. |

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch1-drl.md`
**Section:** "T1-G-1" and "Critical: Death Penalty Not Injected into PPO Buffer"
