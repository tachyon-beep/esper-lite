# Multi-Slot Reward Attribution Fix

## Problem Statement

The multi-slot action execution is correctly wired (sampled slot used for germinate/fossilize/cull), but **reward attribution always uses `slots[0]`** regardless of which slot was sampled.

### Bug Locations

1. **Counterfactual computation** (lines 845-846, 882-883):
   ```python
   target_slot = slots[0]  # BUG: Always first slot
   with env_state.model.seed_slots[target_slot].force_alpha(0.0):
   ```

2. **Telemetry sync** (lines 908-909):
   ```python
   target_slot = slots[0]  # BUG: Always first slot
   seed_state = model.seed_slots[target_slot].state
   ```

3. **Feature extraction loop** (line 951):
   ```python
   target_slot = slots[0]  # BUG: Always first slot
   ```

### Impact

When running with multiple enabled slots (`--slots early,mid,late`):
- The slot action selection learns to pick a slot
- But reward signal always comes from EARLY slot
- Credit assignment is broken for multi-slot learning

## Fix Design

### Core Insight

Counterfactual must be computed **per active slot** before action sampling, then we use the **sampled slot's counterfactual** for reward computation.

### Changes Required

#### 1. Change `baseline_accs` from scalar to dict

**Before:**
```python
baseline_accs = [None] * envs_this_batch
```

**After:**
```python
# baseline_accs[env_idx][slot_id] = accuracy with that slot's seed disabled
baseline_accs = [{} for _ in range(envs_this_batch)]
```

#### 2. Compute counterfactual for each active slot

**Before:**
```python
if i in envs_needing_counterfactual:
    target_slot = slots[0]
    with env_state.model.seed_slots[target_slot].force_alpha(0.0):
        _, cf_correct_tensor, cf_total = process_val_batch(...)
```

**After:**
```python
if i in envs_needing_counterfactual:
    for slot_id in slots:
        if env_state.model.has_active_seed_in_slot(slot_id):
            with env_state.model.seed_slots[slot_id].force_alpha(0.0):
                _, cf_correct_tensor, cf_total = process_val_batch(...)
            # Store per-slot baseline
            baseline_accs[i][slot_id] = 100.0 * cf_correct_tensor.item() / max(cf_total, 1)
```

Wait - this has a sync issue. We need to accumulate across batches, not per-batch. Let me revise.

#### 2 (Revised). Per-slot counterfactual accumulators

**Add per-slot accumulators to ParallelEnvState:**
```python
# In init_accumulators():
self.cf_correct_accums = {slot_id: torch.zeros(1, device=device) for slot_id in slots}
self.cf_totals = {slot_id: 0 for slot_id in slots}
```

**During validation loop:**
```python
if i in envs_needing_counterfactual:
    for slot_id in slots:
        if env_state.model.has_active_seed_in_slot(slot_id):
            with env_state.model.seed_slots[slot_id].force_alpha(0.0):
                _, cf_correct_tensor, cf_total = process_val_batch(...)
            with stream_ctx:
                env_state.cf_correct_accums[slot_id].add_(cf_correct_tensor)
            env_state.cf_totals[slot_id] += cf_total
```

**After sync, compute per-slot baselines:**
```python
for i in envs_needing_counterfactual:
    for slot_id in slots:
        if env_states[i].cf_totals.get(slot_id, 0) > 0:
            baseline_accs[i][slot_id] = (
                100.0 * env_states[i].cf_correct_accums[slot_id].item()
                / env_states[i].cf_totals[slot_id]
            )
```

#### 3. Use sampled slot's counterfactual for reward

**After action sampling (line ~1150):**
```python
# Get counterfactual for the SAMPLED slot (not slots[0])
seed_contribution = None
if target_slot in baseline_accs[env_idx]:
    seed_contribution = env_state.val_acc - baseline_accs[env_idx][target_slot]
```

#### 4. Telemetry sync uses all active slots (or sampled slot post-action)

The telemetry sync at lines 908-935 happens BEFORE action sampling. Options:
- **Option A**: Sync telemetry for ALL active slots (small overhead)
- **Option B**: Keep as-is (telemetry is informational, not used for reward)

Recommend **Option B** for now - the critical fix is reward attribution.

#### 5. Feature extraction at line 951

This is used for `signal_tracker.update()` which feeds into `signals.metrics.accuracy_delta`. This delta is used in reward computation.

The signal tracker already receives `active_seeds` as a list. The accuracy delta is a global metric (model accuracy change), not per-slot. So this is fine.

## Implementation Checklist

1. [ ] Add `cf_correct_accums: dict[str, Tensor]` to `ParallelEnvState`
2. [ ] Add `cf_totals: dict[str, int]` to `ParallelEnvState`
3. [ ] Initialize in `init_accumulators()` with `slots` parameter
4. [ ] Change `baseline_accs` from `list[float | None]` to `list[dict[str, float]]`
5. [ ] Update counterfactual loop to iterate over active slots
6. [ ] Update baseline computation after sync to use per-slot accumulators
7. [ ] Update reward computation to use `baseline_accs[env_idx][target_slot]`
8. [ ] Reset per-slot accumulators at epoch start (already done for single accumulator)

## Testing

1. Run existing tests to verify no regression
2. Add test for multi-slot counterfactual computation
3. Verify reward uses correct slot's contribution with logging

## Performance Impact

- **Worst case**: 3x counterfactual GPU work (if all 3 slots have active seeds)
- **Typical case**: 1x counterfactual GPU work (only 1 slot active)
- **Memory**: 3 additional scalar accumulators per env (negligible)
