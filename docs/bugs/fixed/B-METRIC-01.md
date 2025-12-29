# Fix: Rollback Penalty Not Reflected in Episode Rewards

---

## Resolution Summary

| Field | Value |
|-------|-------|
| **Ticket ID** | `B-METRIC-01` |
| **Status** | `fixed` |
| **Severity** | `P2` |
| **Fix Date** | 2025-12-30 |
| **Domain** | `simic/training` |

---

## Problem

When a governor rollback occurs due to catastrophic failure:

1. **PPO learned from:** `agent.buffer.mark_terminal_with_penalty(env_idx, normalized_penalty)`
   - Penalty injected into buffer, done=True set, GAE handles terminal properly

2. **Metrics computed from:** `sum(env_state.episode_rewards)`
   - This list NEVER received the rollback penalty
   - Fed into: `EpisodeOutcome`, A/B history, stability score, dashboard rewards

**Result:** After a catastrophic rollback, the agent learns "that action was terrible" but metrics show "episode went fine!" - causing divergence between training signal and reported performance.

---

## Fix

Added penalty to `episode_rewards` after injecting into buffer:

```python
if rollback_env_indices:
    for env_idx in rollback_env_indices:
        penalty = env_states[env_idx].governor.get_punishment_reward()
        normalized_penalty = reward_normalizer.normalize_only(penalty)
        agent.buffer.mark_terminal_with_penalty(env_idx, normalized_penalty)
        # B-METRIC-01 fix: Reflect penalty in episode_rewards so metrics
        # (EpisodeOutcome, A/B history, stability) match what PPO learned.
        env_states[env_idx].episode_rewards.append(normalized_penalty)
```

---

## Verification

- [x] Penalty now appears in `episode_rewards`
- [x] `EpisodeOutcome.episode_reward` reflects catastrophic failures
- [x] A/B history includes rollback penalties
- [x] Stability score calculation includes rollback impact
- [x] All existing tests pass

---

## Impact

| Metric | Before | After |
|--------|--------|-------|
| Episode reward after rollback | Inflated (missing penalty) | Accurate (includes penalty) |
| Pareto analysis | Overly optimistic | Reflects true performance |
| A/B testing | Biased toward rollback-prone configs | Fair comparison |
| Stability score | Ignored catastrophic events | Accounts for variance from rollbacks |

---

## Files Changed

| File | Change |
|------|--------|
| `src/esper/simic/training/vectorized.py` | Append normalized_penalty to episode_rewards after buffer injection |
