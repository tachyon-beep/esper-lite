# Phase 4 Implementation Addendum

**Created:** 2025-12-31
**Purpose:** Critical integration details for Phase 4 implementation

**See Also:** `phase4-integration-analysis.md` for complete analysis

---

## Pre-Implementation Checklist

Before starting Phase 4 implementation, confirm:

- [x] Phase 2 complete (Obs V3 feature extraction returns `(obs, blueprint_indices)`)
- [x] Phase 3 complete (BlueprintEmbedding module implemented)
- [ ] Read this addendum in full
- [ ] Read `phase4-integration-analysis.md` for detailed call site audit

---

## Critical Corrections to Original Plan

### 1. blueprint_indices Is Already Extracted ✓

**GOOD NEWS:** Phase 2 already implemented the tuple return!

**File:** `src/esper/simic/training/vectorized.py` Line 2454
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

**Status:** Data exists, just needs to be passed to the network.

---

### 2. Bootstrap Does NOT Need `get_value()` Method

**REMOVE** section 4g from the original plan (`get_value()` method).

**Why:** The current bootstrap implementation is CORRECT:

```python
# vectorized.py line 3233
bootstrap_result = agent.policy.get_action(
    post_action_features_normalized,
    blueprint_indices=post_blueprint_indices,  # ADD THIS
    masks=post_masks_batch,
    hidden=batched_lstm_hidden,
    deterministic=True,
)
```

`get_action()` naturally:
1. Samples (or argmax) the op
2. Computes `Q(s, sampled_op)` via `_compute_value(lstm_out, sampled_op)`
3. Returns the correct bootstrap value

**No additional method needed.** If you want `get_value()` for debugging, add it later.

---

### 3. Keep TypedDict (Don't Change to NamedTuple)

**RECOMMENDATION:** Keep existing `TypedDict` for `_ForwardOutput`.

**Rationale:**
- NamedTuple change breaks 30+ call sites in `factored_lstm.py`
- No measurable performance benefit in practice
- Both are torch.compile compatible
- TypedDict works fine and avoids migration churn

**If changing:** Require torch.compile benchmark showing >5% training speedup.

---

## Integration Requirements

### Required File Changes

| File | What to Change | Priority |
|------|----------------|----------|
| `factored_lstm.py` | Implement Phase 4 architecture | **P0** (core task) |
| `lstm_bundle.py` | Add `blueprint_indices` param to 4 methods | **P0** (wrapper) |
| `vectorized.py` | Pass `blueprint_indices_batch` at 2 call sites | **P0** (integration) |
| `rollout_buffer.py` | Add `blueprint_indices` field | **P1** (buffer schema) |
| `ppo.py` | Pass `blueprint_indices` to evaluate_actions | **P1** (PPO update) |
| `test_tamiyo_network.py` | Update 5 broken tests | **P2** (tests) |
| `test_lstm_bundle.py` | Update 2 broken tests | **P2** (tests) |

---

## Implementation Order (Detailed)

### Step 1: Update Network Architecture (`factored_lstm.py`)

Follow original Phase 4 plan sections 4a-4f:
- Add `BlueprintEmbedding` to `__init__`
- Update `forward()` signature: add `blueprint_indices` parameter
- Update `evaluate_actions()` signature: add `blueprint_indices` parameter
- Implement op-conditioned value head
- Update output types (keep TypedDict unless justified otherwise)

**Key signature change:**
```python
def forward(
    self,
    state: torch.Tensor,              # [batch, seq, 114]
    blueprint_indices: torch.Tensor,  # NEW: [batch, seq, num_slots]
    hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    # ... masks ...
) -> _ForwardOutput:  # KEEP TypedDict for now
```

---

### Step 2: Update Wrapper (`lstm_bundle.py`)

Update 4 methods to pass `blueprint_indices`:

```python
def get_action(
    self,
    features: torch.Tensor,
    blueprint_indices: torch.Tensor,  # NEW
    masks: dict[str, torch.Tensor],
    hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    deterministic: bool = False,
) -> ActionResult:
    result = self._network.get_action(
        features,
        blueprint_indices,  # NEW
        hidden,
        slot_mask=masks.get("slot"),
        # ... other masks ...
    )
    return ActionResult(...)
```

**Repeat for:** `forward()`, `evaluate_actions()`, `get_value()`

---

### Step 3: Update Rollout Collection (`vectorized.py`)

**Location 1:** Line 2498 (action sampling)
```python
# BEFORE:
action_result = agent.policy.get_action(
    states_batch_normalized,
    masks=masks_batch,
    hidden=batched_lstm_hidden,
)

# AFTER:
action_result = agent.policy.get_action(
    states_batch_normalized,
    blueprint_indices=blueprint_indices_batch,  # NEW
    masks=masks_batch,
    hidden=batched_lstm_hidden,
)
```

**Location 2:** Line 3233 (bootstrap)
```python
# First, extract blueprint indices for post-action state
# (around line 3220, after feature extraction)
post_action_features, post_blueprint_indices = batch_obs_to_features(
    batch_signals=[env_state.training_signals],
    batch_slot_reports=[env_state.slot_reports],
    batch_env_states=[env_state],
    slot_config=slot_config,
    device=torch.device(device),
)

# Then pass to get_action
bootstrap_result = agent.policy.get_action(
    post_action_features_normalized,
    blueprint_indices=post_blueprint_indices,  # NEW
    masks=post_masks_batch,
    hidden=batched_lstm_hidden,
    deterministic=True,
)
```

---

### Step 4: Update Rollout Buffer (`rollout_buffer.py`)

**Add field:**
```python
@dataclass
class TamiyoRolloutBuffer:
    obs: torch.Tensor                    # [num_steps, num_envs, obs_dim]
    blueprint_indices: torch.Tensor      # NEW: [num_steps, num_envs, num_slots]
    actions: dict[str, torch.Tensor]
    # ... other fields ...
```

**Update `add()` method:**
```python
def add(
    self,
    obs: torch.Tensor,
    blueprint_indices: torch.Tensor,     # NEW
    action: dict[str, torch.Tensor],
    # ... other params ...
):
    self.blueprint_indices[self.pos] = blueprint_indices.clone()
    # ... existing logic ...
```

**Update `get_batched_sequences()`:**
```python
def get_batched_sequences(self, ...):
    # ... existing logic ...
    yield {
        "obs": batched_obs,
        "blueprint_indices": batched_blueprint_indices,  # NEW
        "actions": batched_actions,
        # ... other fields ...
    }
```

---

### Step 5: Update PPO Update (`ppo.py`)

**Location:** Line 499

```python
# BEFORE:
result = self.policy.evaluate_actions(
    states,
    actions,
    masks,
    hidden,
)

# AFTER:
result = self.policy.evaluate_actions(
    states,
    blueprint_indices,  # NEW - from buffer batch
    actions,
    masks,
    hidden,
)
```

---

### Step 6: Fix Broken Tests

**File:** `tests/simic/test_tamiyo_network.py`

All tests need `blueprint_indices` parameter added:

```python
def test_forward_pass():
    net = FactoredRecurrentActorCritic(state_dim=114, num_slots=3)
    state = torch.randn(2, 5, 114)
    bp_idx = torch.randint(0, 13, (2, 5, 3))  # NEW
    output = net(state, bp_idx)  # Pass bp_idx
    # ... assertions ...
```

**Apply same pattern to:**
- `test_hidden_state_propagation`
- `test_action_masking`
- `test_get_action`
- `test_evaluate_actions`

---

## Validation Strategy

### After Step 1 (Network):
```bash
PYTHONPATH=src python -c "
from esper.tamiyo.networks import FactoredRecurrentActorCritic
import torch

net = FactoredRecurrentActorCritic(state_dim=114, num_slots=3)
state = torch.randn(2, 5, 114)
bp_idx = torch.randint(0, 13, (2, 5, 3))

out = net(state, bp_idx)
print(f'Output value shape: {out[\"value\"].shape}')  # [2, 5]
print('✓ Network forward pass works')
"
```

### After Step 2 (Wrapper):
```bash
PYTHONPATH=src python -c "
from esper.tamiyo.policy import create_policy
import torch

policy = create_policy('lstm', feature_dim=114)
features = torch.randn(2, 114)
bp_idx = torch.randint(0, 13, (2, 3))
masks = {}  # Empty masks for test

result = policy.get_action(features, bp_idx, masks)
print(f'Action result value shape: {result.value.shape}')
print('✓ Wrapper works')
"
```

### After Steps 3-5 (Integration):
```bash
# Run a short training episode
PYTHONPATH=src python -m esper.scripts.train ppo --episodes 1 --max-epochs 10
```

If this completes without errors, integration is correct.

---

## Common Pitfalls

### 1. Forgetting to Update Bootstrap Blueprint Indices

**Error:** Bootstrap uses old features without blueprint indices
**Symptom:** `TypeError: forward() missing required argument: 'blueprint_indices'`
**Fix:** Extract `post_blueprint_indices` from `batch_obs_to_features()` (Step 3, Location 2)

### 2. Buffer Schema Mismatch

**Error:** Buffer doesn't store blueprint indices
**Symptom:** KeyError in PPO update when trying to retrieve blueprint_indices
**Fix:** Update rollout buffer schema (Step 4)

### 3. Test Imports Out of Date

**Error:** Tests import old network signature
**Symptom:** Tests fail with missing argument errors
**Fix:** Update all test files to pass blueprint_indices (Step 6)

---

## New Tests Required

Add these to `tests/simic/test_tamiyo_network.py`:

### 1. Op-Conditioned Value Test
```python
def test_op_conditioned_value_forward():
    """Verify forward() computes Q(s, sampled_op)."""
    net = FactoredRecurrentActorCritic(state_dim=114, num_slots=3)
    state = torch.randn(2, 5, 114)
    bp_idx = torch.randint(0, 13, (2, 5, 3))

    out = net(state, bp_idx)

    # Should have sampled op
    assert "sampled_op" in out or hasattr(out, "sampled_op")
    # Value should be conditioned on that op
    assert out["value"].shape == (2, 5)
```

### 2. Stored Op Consistency Test
```python
def test_stored_op_value_consistency():
    """Verify evaluate_actions uses stored_op."""
    net = FactoredRecurrentActorCritic(state_dim=114, num_slots=3)
    state = torch.randn(2, 5, 114)
    bp_idx = torch.randint(0, 13, (2, 5, 3))

    # Get actions from forward
    fwd_out = net(state, bp_idx)
    stored_op = fwd_out.get("sampled_op") or fwd_out.sampled_op

    # Build actions dict with stored op
    actions = {
        "op": stored_op,
        "slot": torch.randint(0, 3, (2, 5)),
        # ... other actions ...
    }

    # Evaluate with stored actions
    eval_log_probs, eval_value, eval_entropy, _ = net.evaluate_actions(
        state, bp_idx, actions
    )

    # Values should match when same op is used
    # (may differ slightly due to floating point, hence atol=1e-5)
    assert torch.allclose(fwd_out["value"], eval_value, atol=1e-5)
```

### 3. Blueprint Embedding Integration Test
```python
def test_blueprint_embedding_shapes():
    """Verify blueprint embeddings integrate correctly."""
    net = FactoredRecurrentActorCritic(state_dim=114, num_slots=3)

    # Test with various blueprint index patterns
    bp_idx_active = torch.tensor([[0, 2, 5], [1, 3, 7]])
    bp_idx_inactive = torch.tensor([[-1, -1, -1], [0, -1, 2]])

    state = torch.randn(2, 1, 114)

    out1 = net(state, bp_idx_active)
    out2 = net(state, bp_idx_inactive)

    # Both should produce valid outputs
    assert out1["value"].shape == (2, 1)
    assert out2["value"].shape == (2, 1)

    # Different indices should produce different values
    assert not torch.allclose(out1["value"], out2["value"])
```

---

## Success Criteria

Phase 4 is complete when:

- [ ] Network accepts `blueprint_indices` parameter
- [ ] BlueprintEmbedding integrated into forward pass
- [ ] Op-conditioned value head implemented (`_compute_value()` helper)
- [ ] Wrapper (`lstm_bundle.py`) passes `blueprint_indices` to network
- [ ] Rollout collection passes `blueprint_indices_batch` to policy
- [ ] Bootstrap extracts and passes `post_blueprint_indices`
- [ ] Rollout buffer stores `blueprint_indices`
- [ ] PPO update retrieves `blueprint_indices` from buffer
- [ ] All existing tests pass
- [ ] 3 new op-conditioning tests added and passing
- [ ] Short training run (10 epochs) completes without errors

---

## Rollback Plan

If Phase 4 breaks training:

1. **Revert to Phase 3:** `git checkout HEAD~1`
2. **Identify issue:** Check error logs for which step failed
3. **Fix forward:** Address the specific issue (likely buffer schema or call site)
4. **Test incrementally:** Validate each step before proceeding

**Do NOT introduce compatibility layers.** Fix the integration properly.
