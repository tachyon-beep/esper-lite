# Phase 4 Integration Analysis

**Created:** 2025-12-31
**Purpose:** Document call sites, data flow, and migration requirements for Phase 4

---

## Call Site Audit

### 1. Direct Network Usage

**File:** `src/esper/tamiyo/policy/lstm_bundle.py`

| Method | Line | Current Signature | Required Changes |
|--------|------|-------------------|------------------|
| `get_action()` | 93-106 | `get_action(features, hidden, slot_mask=..., blueprint_mask=..., ...)` | Add `blueprint_indices` parameter |
| `forward()` | 152-163 | `forward(features, hidden, slot_mask=..., blueprint_mask=..., ...)` | Add `blueprint_indices` parameter |
| `evaluate_actions()` | 196-208 | `evaluate_actions(features, actions, slot_mask=..., blueprint_mask=..., ...)` | Add `blueprint_indices` parameter |
| `get_value()` | 240-257 | `get_value(features, hidden)` | Add `blueprint_indices` parameter |

**Impact:** `LSTMPolicyBundle` is the primary wrapper. ALL methods that call the network must be updated.

---

### 2. Indirect Usage via PolicyBundle

**File:** `src/esper/simic/training/vectorized.py`

| Location | Line | Current Call | Required Changes |
|----------|------|--------------|------------------|
| Rollout collection | 2498 | `agent.policy.get_action(states_batch_normalized, masks=masks_batch, hidden=...)` | Pass `blueprint_indices=blueprint_indices_batch` |
| Bootstrap value | 3233 | `agent.policy.get_action(post_action_features_normalized, masks=post_masks_batch, hidden=...)` | Pass `blueprint_indices=post_blueprint_indices` |

**Critical Discovery:** Line 2461-2462 has a TODO comment:
```python
# TODO: [Phase 3] Pass blueprint_indices_batch to policy network for embedding lookup
# For now, store for future use - network currently ignores this parameter
```

**Good News:** The data is already being extracted! `blueprint_indices_batch` exists but isn't passed to the network yet.

---

### 3. PPO Agent Integration

**File:** `src/esper/simic/agent/ppo.py`

| Location | Line | Method | Required Changes |
|----------|------|--------|------------------|
| PPO update | 499 | `self.policy.evaluate_actions(states, actions, masks, hidden)` | Add `blueprint_indices` from buffer |

**Buffer Implication:** The rollout buffer must store `blueprint_indices` alongside states.

---

## Blueprint Indices Data Flow

### Phase 2 Output (Already Implemented ✓)

**File:** `src/esper/tamiyo/policy/features.py`

```python
def batch_obs_to_features(
    batch_signals: list[TrainingSignals],
    batch_slot_reports: list[dict[str, SeedStateReport]],
    batch_env_states: list[ParallelEnvState],
    slot_config: SlotConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns: (obs [batch, obs_dim], blueprint_indices [batch, num_slots])"""
    # ... extraction logic ...
    return obs, blueprint_indices  # Line 572
```

**Status:** ✅ Complete - Phase 2 implementation ready

---

### Rollout Collection (Needs Update)

**File:** `src/esper/simic/training/vectorized.py` Line 2454-2462

**Current:**
```python
states_batch, blueprint_indices_batch = batch_obs_to_features(
    batch_signals=all_signals,
    batch_slot_reports=all_slot_reports,
    batch_env_states=env_states,
    slot_config=slot_config,
    device=torch.device(device),
)
# TODO: [Phase 3] Pass blueprint_indices_batch to policy network
```

**Required Change:**
```python
# Line 2498: Add blueprint_indices parameter
action_result = agent.policy.get_action(
    states_batch_normalized,
    blueprint_indices=blueprint_indices_batch,  # NEW
    masks=masks_batch,
    hidden=batched_lstm_hidden,
    deterministic=False,
)
```

---

### Bootstrap Value Computation (Needs Update)

**File:** `src/esper/simic/training/vectorized.py` Line 3190-3240

**Current Challenge:** Bootstrap is computed AFTER the environment step, so we need to:
1. Extract features from `post_action_state`
2. Get `post_blueprint_indices` from feature extraction
3. Pass to `get_action()` for bootstrap value

**Required Changes:**

```python
# Around line 3220: Extract features for bootstrap (currently missing blueprint_indices)
post_action_features, post_blueprint_indices = batch_obs_to_features(
    batch_signals=[env_state.training_signals],
    batch_slot_reports=[env_state.slot_reports],
    batch_env_states=[env_state],
    slot_config=slot_config,
    device=torch.device(device),
)

# Line 3233: Pass blueprint_indices to get_action
bootstrap_result = agent.policy.get_action(
    post_action_features_normalized,
    blueprint_indices=post_blueprint_indices,  # NEW
    masks=post_masks_batch,
    hidden=batched_lstm_hidden,
    deterministic=True,
)
```

---

### PPO Update (Needs Buffer Schema Update)

**File:** `src/esper/simic/agent/ppo.py` Line 499

**Current:**
```python
result = self.policy.evaluate_actions(
    states,      # From buffer
    actions,     # From buffer
    masks,       # From buffer
    hidden,      # From buffer
)
```

**Required:**
```python
result = self.policy.evaluate_actions(
    states,              # From buffer
    blueprint_indices,   # NEW - from buffer
    actions,             # From buffer
    masks,               # From buffer
    hidden,              # From buffer
)
```

**Buffer Schema Impact:** `TamiyoRolloutBuffer` must add a `blueprint_indices` field.

---

## Rollout Buffer Schema Changes

**File:** `src/esper/simic/agent/rollout_buffer.py`

### Required Changes

1. **Add field to buffer:**
```python
@dataclass
class TamiyoRolloutBuffer:
    obs: torch.Tensor                    # [num_steps, num_envs, obs_dim]
    blueprint_indices: torch.Tensor      # NEW: [num_steps, num_envs, num_slots]
    actions: dict[str, torch.Tensor]     # Existing
    # ... other fields ...
```

2. **Update `add()` method:**
```python
def add(
    self,
    obs: torch.Tensor,
    blueprint_indices: torch.Tensor,     # NEW parameter
    action: dict[str, torch.Tensor],
    # ... other params ...
):
    self.blueprint_indices[self.pos] = blueprint_indices.clone()
    # ... existing logic ...
```

3. **Update `get_batched_sequences()` method:**
```python
def get_batched_sequences(self, ...):
    # ... existing logic ...
    return {
        "obs": batched_obs,
        "blueprint_indices": batched_blueprint_indices,  # NEW
        "actions": batched_actions,
        # ... other fields ...
    }
```

---

## Test Migration Plan

### Tests That Will Break

**File:** `tests/simic/test_tamiyo_network.py`

| Test | Line | Issue | Fix |
|------|------|-------|-----|
| `test_forward_pass` | 27 | Missing `blueprint_indices` parameter | Add `bp_idx = torch.randint(0, 13, (2, 5, 3))` |
| `test_hidden_state_propagation` | 62 | Missing `blueprint_indices` | Same as above |
| `test_action_masking` | 88 | Missing `blueprint_indices` + mask format changed | Add bp_idx + update mask dict format |
| `test_get_action` | 111 | Missing `blueprint_indices` | Add bp_idx parameter |
| `test_evaluate_actions` | 145 | Missing `blueprint_indices` | Add bp_idx parameter |

**File:** `tests/tamiyo/policy/test_lstm_bundle.py`

| Test | Line | Issue | Fix |
|------|------|-------|-----|
| `test_get_action` | 77 | Bundle must pass blueprint_indices to network | Update bundle implementation first |
| `test_evaluate_actions` | 112 | Same as above | Same as above |

### New Tests Needed

1. **Test op-conditioned value correctness:**
```python
def test_op_conditioned_value():
    """Verify value = Q(s, sampled_op) during forward pass."""
    net = FactoredRecurrentActorCritic(state_dim=114, num_slots=3)
    state = torch.randn(2, 5, 114)
    bp_idx = torch.randint(0, 13, (2, 5, 3))

    out = net(state, bp_idx)
    # Verify value was computed with sampled_op
    assert out.sampled_op is not None
    assert out.value.shape == (2, 5)
```

2. **Test stored_op value consistency:**
```python
def test_stored_op_value_consistency():
    """Verify evaluate_actions uses stored_op for value."""
    net = FactoredRecurrentActorCritic(state_dim=114, num_slots=3)
    state = torch.randn(2, 5, 114)
    bp_idx = torch.randint(0, 13, (2, 5, 3))

    # Collect actions during forward
    fwd_out = net(state, bp_idx)
    stored_op = fwd_out.sampled_op

    # Evaluate with stored actions
    actions = {"op": stored_op, ...}  # Use stored op
    eval_out = net.evaluate_actions(state, bp_idx, actions)

    # Value should be consistent when same op is used
    assert torch.allclose(fwd_out.value, eval_out.value, atol=1e-5)
```

3. **Test blueprint embedding integration:**
```python
def test_blueprint_embedding_integration():
    """Verify blueprint embeddings flow through network correctly."""
    net = FactoredRecurrentActorCritic(state_dim=114, num_slots=3)

    # Test with active blueprints
    bp_idx_active = torch.tensor([[0, 2, 5], [1, 3, 7]])  # batch=2, slots=3
    state = torch.randn(2, 1, 114)
    out1 = net(state, bp_idx_active)

    # Test with inactive slots (-1)
    bp_idx_inactive = torch.tensor([[-1, -1, -1], [0, -1, 2]])
    out2 = net(state, bp_idx_inactive)

    # Verify different blueprint indices produce different outputs
    assert not torch.allclose(out1.value, out2.value)
```

---

## Bootstrap Value: Resolving the Contradiction

### The Confusion

Phase 4 plan has contradictory guidance:

- **Line 24-32:** "Update `get_action()` for op-conditioned value" ✓
- **Line 203:** "Do NOT use for bootstrap at truncation!" ✗
- **Line 219:** "Use forward() instead" ❌ (forward doesn't sample actions?)

### The Truth

**Bootstrap SHOULD use `get_action()`** because:

1. Bootstrap needs `Q(s_{T+1}, op_{T+1})` where `op_{T+1}` is what the policy would choose
2. `get_action()` naturally:
   - Samples `op` from the policy (or uses argmax if deterministic=True)
   - Computes `Q(s, sampled_op)` via `_compute_value(lstm_out, sampled_op)`
   - Returns `sampled_op` so it can be stored if needed

3. Current code (line 3233) is CORRECT:
```python
bootstrap_result = agent.policy.get_action(
    post_action_features_normalized,
    masks=post_masks_batch,
    hidden=batched_lstm_hidden,
    deterministic=True,  # Use deterministic=True for stable bootstrap
)
bootstrap_values = bootstrap_result.value.cpu().tolist()
```

### What `get_value()` Is For

The plan's `get_value()` method is ONLY for debugging scenarios like:
- Visualizing Q(s, op) landscape for different ops
- Comparing value estimates between ops
- Unit testing value head behavior

**It should NOT be used in the training loop at all.**

### Resolution

**Delete section 4g from the plan** OR clearly label it "DEBUG ONLY - NOT FOR TRAINING."

The bootstrap implementation doesn't need to change - `get_action()` with op-conditioning already works correctly.

---

## NamedTuple vs TypedDict Assessment

### Current Implementation

**File:** `src/esper/tamiyo/networks/factored_lstm.py` Line 66

```python
class _ForwardOutput(TypedDict):
    """Typed dict for forward() return value - enables mypy to track per-key types."""
    slot_logits: torch.Tensor
    blueprint_logits: torch.Tensor
    # ... 8 more fields ...
```

**Access Pattern:** Dict-style: `output["slot_logits"]`

### Proposed Change

```python
class ForwardOutput(NamedTuple):
    """Output from forward() during rollout."""
    op_logits: torch.Tensor
    slot_logits: torch.Tensor
    # ... 8 more fields ...
```

**Access Pattern:** Attribute-style: `output.slot_logits`

### Analysis

#### NamedTuple Pros
- Immutable (prevents accidental modification)
- Slightly faster field access (~5% in microbenchmarks)
- Can't add unexpected keys
- Better repr() for debugging

#### NamedTuple Cons
- **Breaking change:** All 30+ call sites in `factored_lstm.py` must change from `output["key"]` to `output.key`
- **Breaking change:** All test files must update access patterns
- **Breaking change:** Any external code that accesses network outputs

#### TypedDict Pros
- **No breaking changes** - existing code continues to work
- Compatible with dict-based code (e.g., passing `**output` to functions)
- Easier gradual typing migration

#### TypedDict Cons
- Mutable (can add keys at runtime, though TypedDict prevents this statically)
- Slightly slower dict access overhead

### torch.compile Compatibility

**Both are compatible** with torch.compile/TorchScript:
- NamedTuple: Fully supported, fields traced as constants
- TypedDict: Supported in Torch 2.x (Dynamo traces dict keys as constants)

### Recommendation

**KEEP TypedDict** unless there's a strong performance justification.

**Rationale:**
1. No measurable performance benefit in practice (network forward pass dominates)
2. Avoids breaking 30+ call sites
3. Avoids updating all tests
4. Existing TypedDict works fine and is already well-typed

**If changing:** Require torch.compile benchmark showing >5% speedup on actual training loop, not microbenchmarks.

---

## Summary

### Critical Findings

1. ✅ **blueprint_indices data already extracted** (vectorized.py:2454) - just needs to be passed
2. ✅ **Integration points identified** - 3 files need updates
3. ✅ **Buffer schema clear** - add `blueprint_indices` field
4. ✅ **Bootstrap works correctly** - no changes needed to bootstrap logic
5. ⚠️ **NamedTuple change unjustified** - recommend keeping TypedDict

### Files Requiring Changes

| File | Changes Required |
|------|------------------|
| `lstm_bundle.py` | Add `blueprint_indices` parameter to 4 methods |
| `vectorized.py` | Pass `blueprint_indices_batch` at 2 call sites |
| `rollout_buffer.py` | Add `blueprint_indices` field + update add/get methods |
| `ppo.py` | Pass `blueprint_indices` from buffer to evaluate_actions |
| `factored_lstm.py` | Implement Phase 4 architecture (already documented) |
| `test_tamiyo_network.py` | Update 5 tests with blueprint_indices |
| `test_lstm_bundle.py` | Update 2 tests with blueprint_indices |

### Next Steps

1. Update Phase 4 plan with this integration analysis
2. Remove contradictory `get_value()` section OR clearly mark DEBUG ONLY
3. Change TypedDict to NamedTuple **only if** performance justification provided
4. Proceed with implementation following updated plan
