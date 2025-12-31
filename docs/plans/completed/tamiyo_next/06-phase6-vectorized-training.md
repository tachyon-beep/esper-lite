### Phase 6: Vectorized Training Integration

**Goal:** Wire up the new feature extraction and network in the training loop.

**⚠️ RISK: VERY HIGH** — This phase threads new tuple obs + blueprint indices through the hottest, largest module.

**Files:**

- `src/esper/simic/training/vectorized.py`
- `src/esper/simic/agent/rollout_buffer.py` (schema change!)
- `src/esper/simic/agent/ppo.py` (buffer consumption)

#### 6a. Rollout Buffer Schema Changes

**Current state:** `TamiyoRolloutBuffer` stores `states: torch.Tensor` as a single `[num_envs, max_steps, state_dim]` tensor.

**Required change:** Add `blueprint_indices` storage:

```python
# In TamiyoRolloutBuffer.__post_init__()
self.blueprint_indices = torch.zeros(
    n, m, self.num_slots, dtype=torch.int64, device=device
)

# In add() method - new parameter
blueprint_indices: torch.Tensor,  # [num_slots], int64
# ...
self.blueprint_indices[env_id, step_idx] = blueprint_indices

# In get_batched_sequences() - add to return dict
"blueprint_indices": self.blueprint_indices.to(device, non_blocking=nb),
```

**Also update:** `TamiyoRolloutStep` NamedTuple to include `blueprint_indices` field.

**Note on op storage:** The existing `op_actions` field already stores the sampled op from each rollout step. This is the op used for value conditioning (see Phase 4e/4f). During PPO update, `batch["op_actions"]` becomes `actions["op"]` which `evaluate_actions()` uses for `Q(s, stored_op)`. No additional storage needed.

#### 6a½. Bootstrap Value Computation (CRITICAL FIX REQUIRED)

> ⚠️ **BUG: Current bootstrap uses `deterministic=True` which is WRONG for Q(s,op) critics**
>
> The current code at `vectorized.py:3212-3219` uses:
> ```python
> bootstrap_result = agent.policy.get_action(
>     post_action_features_normalized,
>     masks=post_masks_batch,
>     hidden=batched_lstm_hidden,
>     deterministic=True,  # <-- BUG!
> )
> ```
>
> With Q(s,op) critics, this creates an **optimistic bias**:
> - Rollout stores `Q(s, sampled_op)` where op is sampled stochastically
> - Bootstrap computes `Q(s, argmax_op)` which is always the "best" action
> - Mixing these creates inconsistent value targets for GAE

**Required fix:**

```python
# Bootstrap at truncation: sample op stochastically (NOT deterministic)
# This ensures bootstrap Q(s_T, sampled_op_T) matches rollout Q(s_t, sampled_op_t)
bootstrap_result = agent.policy.get_action(
    post_action_features_normalized,
    masks=post_masks_batch,
    hidden=batched_lstm_hidden,
    deterministic=False,  # CRITICAL: Sample op, don't use argmax
)
bootstrap_value = bootstrap_result.value  # Q(s_T, sampled_op_T)
```

#### 6a¾. Done vs Truncated Semantics (CRITICAL FIX REQUIRED)

> ⚠️ **BUG: Current code conflates done and truncated**
>
> The current code at `vectorized.py:3003` sets:
> ```python
> done = epoch == max_epochs
> truncated = done  # <-- BUG! "truncated = episode ended" is wrong
> ```
>
> This conflates "episode ended" with "time-limit truncation". With Q(s,op) critics,
> bootstrapping on genuine terminals biases GAE targets.

**Required fix - Gymnasium-compliant semantics:**

| Condition | done | truncated | bootstrap |
|-----------|------|-----------|-----------|
| Intermediate step | `False` | `False` | N/A |
| Natural termination (goal/failure) | `True` | `False` | `0.0` |
| Time limit (max_epochs) | `False` | `True` | `Q(s_next, sampled_op)` |

```python
# At episode end:
is_natural_terminal = (some_terminal_condition)  # e.g., catastrophic failure
is_time_limit = (epoch == max_epochs) and not is_natural_terminal

# Gymnasium semantics: done=True only for natural terminals, NOT time limits
done = is_natural_terminal
truncated = is_time_limit  # ONLY time limit, not natural terminals

# In GAE computation:
if truncated:
    bootstrap = Q(s_next, sampled_op)  # Time limit: bootstrap from next state
else:
    bootstrap = 0.0  # Natural terminal: no future returns
```

**Test coverage:** See `test_done_vs_truncated_at_max_epochs()` in Phase 7g (line 2815).

#### 6b. Obs V3 State Tracking in ParallelEnvState

**IMPORTANT:** This project uses `ParallelEnvState` (in `simic/training/parallel_env_state.py`), NOT a separate `EnvState` class. All Obs V3 state tracking fields go here.

Add to `ParallelEnvState`:

```python
from esper.leyline import LifecycleOp

@dataclass
class ParallelEnvState:
    # ... existing fields (signals, reports, hidden, etc.) ...

    # === Obs V3 Action Feedback (7 dims: 1 success + 6 op one-hot) ===
    last_action_success: bool = True
    last_action_op: int = LifecycleOp.WAIT.value  # 0

    # === Obs V3 Gradient Health Tracking (for gradient_health_prev feature) ===
    # Maps slot_id -> previous epoch's gradient_health value
    # Used in Phase 2c feature extraction
    gradient_health_prev: dict[str, float] = field(default_factory=dict)

    # === Obs V3 Counterfactual Freshness Tracking ===
    # Maps slot_id -> epochs since last counterfactual measurement
    # Incremented each epoch, reset to 0 when counterfactual is computed
    epochs_since_counterfactual: dict[str, int] = field(default_factory=dict)
```

**State Update Contract:**

| Field | When to Read | When to Update | Initial Value |
|-------|--------------|----------------|---------------|
| `last_action_success` | Feature extraction (Phase 2b) | After action execution, before next step | `True` |
| `last_action_op` | Feature extraction (Phase 2b) | After action execution, before next step | `LifecycleOp.WAIT.value` (0) |
| `gradient_health_prev[slot_id]` | Feature extraction (Phase 2c) | After epoch completion, from `report.telemetry.gradient_health` | `1.0` (assume healthy for new slots) |
| `epochs_since_counterfactual[slot_id]` | Feature extraction (Phase 2c) | Increment each epoch; reset to 0 after counterfactual computed | `0` for newly germinated slots |

**Action Success Definition:**

```python
# In vectorized.py, after action execution:
# NOTE: ActionResult (from tamiyo.policy.types) does NOT have germination_succeeded etc.
# Action success must be determined from the environment state changes, not ActionResult.
# This is pseudocode showing the LOGIC, not the actual type signature.

def _determine_action_success(action: dict, env_state: ParallelEnvState) -> bool:
    """Determine if the action succeeded for action feedback feature.

    NOTE: Success is determined by comparing environment state before/after action,
    NOT from ActionResult (which only contains policy outputs: action, log_prob, value).
    """
    op = LifecycleOp(action["op"])

    if op == LifecycleOp.WAIT:
        return True  # WAIT always "succeeds"
    elif op == LifecycleOp.GERMINATE:
        # Check if a new seed appeared in the target slot
        return env_state.seeds_created > previous_seeds_created
    elif op == LifecycleOp.SET_ALPHA_TARGET:
        return True  # Alpha changes always succeed
    elif op == LifecycleOp.PRUNE:
        # Check if seed was actually pruned from slot
        return slot_is_now_empty
    elif op == LifecycleOp.FOSSILIZE:
        # Check if fossilization occurred
        return env_state.seeds_fossilized > previous_fossilized
    else:
        return True  # Unknown ops default to success

# Update state after action
env_state.last_action_op = action["op"]
env_state.last_action_success = _determine_action_success(action, env_state)
```

**Gradient Health Prev Update:**

```python
# After epoch completion (in vectorized.py step loop):
for slot_id, report in slot_reports.items():
    if report.telemetry is not None:
        env_state.gradient_health_prev[slot_id] = report.telemetry.gradient_health
    # Note: Keep stale values for inactive slots until slot is cleared

# When slot becomes PRUNED (NOT fossilized - fossilized is still active per is_active_stage):
# FOSSILIZED seeds remain in the forward pass, so keep tracking them.
# Only PRUNED/EMBARGOED/RESETTING/DORMANT are truly inactive.
if not is_active_stage(report.stage) and slot_id in env_state.gradient_health_prev:
    del env_state.gradient_health_prev[slot_id]
```

**Counterfactual Freshness Update:**

```python
# After each epoch (in vectorized.py step loop):
for slot_id in active_slot_ids:
    env_state.epochs_since_counterfactual[slot_id] = (
        env_state.epochs_since_counterfactual.get(slot_id, 0) + 1
    )

# When slot is germinated:
env_state.epochs_since_counterfactual[slot_id] = 0

# When slot becomes inactive:
if slot_id in env_state.epochs_since_counterfactual:
    del env_state.epochs_since_counterfactual[slot_id]
```

**Counterfactual Reset Integration Pattern:**

The reset happens in `vectorized.py` immediately after a successful `compute_contributions()` call:

```python
# In vectorized.py, after Shapley computation (around the counterfactual helper call):
if env_state.counterfactual_helper is not None:
    try:
        contributions = env_state.counterfactual_helper.compute_contributions(...)
        # Reset freshness counter for all slots that received fresh counterfactual
        for slot_id in active_slot_ids:
            env_state.epochs_since_counterfactual[slot_id] = 0
    except Exception as e:
        _logger.warning(f"Shapley failed for env {env_idx}: {e}")
        # Don't reset counters on failure - stale values decay naturally
```

**Integration Notes:**
- The reset is triggered by successful completion of `compute_contributions()`, not by the attribution module itself
- Failed computations leave counters unchanged (stale values decay via gamma)
- This keeps all state management in `vectorized.py` rather than coupling attribution to ParallelEnvState

#### 6c. Update Feature Extraction Call

```python
# Was:
obs = batch_obs_to_features(signals, reports, use_telemetry=True, ...)

# Now (clean replacement, not new function name):
obs, blueprint_indices = batch_obs_to_features(signals, reports, slot_config, device)
```

Note: The function signature changes from returning `Tensor` to returning `tuple[Tensor, Tensor]`.

#### 6d. Update Policy Forward Calls

Pass `blueprint_indices` to all `policy.forward()` and `policy.get_action()` calls:

```python
# Was:
action, value, log_probs, hidden = policy(state, hidden, masks)

# Now:
action, value, log_probs, hidden = policy(state, blueprint_indices, hidden, masks)
```

#### 6e. Update Buffer add() Calls

Thread blueprint_indices through the rollout collection loop:

```python
buffer.add(
    env_id=env_id,
    state=obs[env_id],
    blueprint_indices=blueprint_indices[env_id],  # NEW
    slot_action=action["slot"],
    # ... rest unchanged ...
)
```

#### 6f. Update PPO Consumption

In `ppo.py`, the `_ppo_update()` method must pass blueprint_indices from buffer to `evaluate_actions()`:

```python
batch = buffer.get_batched_sequences(device)
actions = {
    "op": batch["op_actions"],
    "slot": batch["slot_actions"],
    # ... other action heads ...
}
# evaluate_actions() extracts stored_op from actions["op"] internally
# (see Phase 4f) - no separate sampled_op parameter needed
result = policy.evaluate_actions(
    batch["states"],
    batch["blueprint_indices"],  # NEW
    actions,
    masks,
    initial_hidden,
)
```

---

