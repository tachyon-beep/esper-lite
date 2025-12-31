### Phase 8: Cleanup (After Validation)

**Goal:** Verify no stale code remains. Since we used clean replacement (not dual paths), this phase is mainly verification.

#### 8a. Verify Old Code Deleted

Ensure old feature/network code was deleted during implementation:

```bash
# Should return no matches if clean replacement was done correctly
git grep "use_telemetry"  # Old flag
git grep "FactoredRecurrentActorCritic" | grep -v V2  # Old network class (if renamed)
```

#### 8b. Verify Tests Updated

Ensure tests use new API signatures:

```bash
# Check tests aren't importing old functions
git grep "batch_obs_to_features.*use_telemetry" tests/
```

#### 8c. Remove Any Deferred Cleanup Markers

If any `TODO(obs-v3)` or `TODO(policy-v2)` markers were added during implementation, resolve them:

```bash
git grep "TODO(obs-v3)"
git grep "TODO(policy-v2)"
```

---

## Quick Reference: File → Changes

| File | Phase | Key Changes |
|------|-------|-------------|
| `leyline/__init__.py` | 1 | Dimension constants (512), BLUEPRINT_NULL_INDEX, DEFAULT_BLUEPRINT_EMBED_DIM, NUM_BLUEPRINTS |
| `simic/training/parallel_env_state.py` | 2a½ | Add Obs V3 state tracking fields (action feedback, gradient_health_prev, epochs_since_counterfactual) |
| `tamiyo/policy/features.py` | 2 | Clean replacement, raw history (10 dims), gradient_health_prev, device-keyed cache, gamma-matched counterfactual, returns `(obs[121], bp_idx)` |
| `tamiyo/networks/factored_lstm.py` | 3, 4 | BlueprintEmbedding, ForwardOutput/EvaluateOutput types, 3-layer blueprint head, op-conditioned Q(s,op) critic, `_compute_value()` helper |
| `simic/agent/ppo.py` | 5, 6f | Differential entropy, evaluate_actions with blueprint_indices |
| `simic/agent/rollout_buffer.py` | 6a | Add `blueprint_indices` tensor, update `add()` and `get_batched_sequences()` |
| `simic/training/vectorized.py` | 6 | Action feedback tracking, gradient_health_prev tracking, new feature extraction API, blueprint_indices plumbing |

---

## Gotchas & Warnings

### 1. Blueprint Index dtype

Must be `torch.int64` for `nn.Embedding`. NumPy extraction should use `np.int64`.

### 2. Op-Conditioning During Bootstrap (P1 from DRL Review)

The **same conditioning pattern** must be used for ALL three value computations:
1. Value stored during rollout collection → `Q(s, sampled_op)`
2. Value computed during PPO update → `Q(s, stored_op)` where `stored_op == sampled_op` from rollout
3. Value bootstrap at episode truncation → `Q(s_T, sampled_op_T)`

**At truncation, where does `sampled_op_T` come from?**

The agent hasn't taken an action in the terminal state `s_T` yet. To get the op for conditioning:

```python
# In truncation handling (vectorized.py):
# 1. Extract features for final state
final_obs, final_blueprint_indices = batch_obs_to_features(final_signals, final_reports, ...)

# 2. Run forward() to sample what op WOULD have been taken
#    (this also gives us the correctly conditioned value)
with torch.no_grad():
    forward_out = policy(final_obs, final_blueprint_indices, hidden)
    bootstrap_value = forward_out.value  # Already Q(s_T, sampled_op_T)
```

**Why not use the last stored op?** Using `op_{T-1}` (from the previous transition) would condition on a potentially different operation, creating a mismatch. The policy might choose differently in state `s_T`.

**If any path uses different conditioning (e.g., expected value E[Q(s,op)]), advantage estimates will be biased.**

### 3. Blueprint Head Initialization

Apply `gain=0.01` ONLY to the final Linear layer. Intermediate layers use `sqrt(2)`.

### 4. Action Feedback Initial State

First step of episode has no "previous action". Initialize:

- `last_action_success = True`
- `last_action_op = LifecycleOp.WAIT.value`

### 5. Telemetry Toggle Removal

The `use_telemetry` flag no longer exists in V3. Remove from CLI args and config.

### 6. Feature Size Mismatch

V3 feature extraction returns 121 dims. Network expects 133 (121 + 12 from embeddings).
The embedding concatenation happens INSIDE the network, not in feature extraction.

### 7. Op-Conditioning Consistency (Q(s,op) Pattern)

**Terminology note:** This is an **action-conditioned baseline** (not SARSA). The distinction:
- SARSA: On-policy TD learning for Q-values with temporal difference updates
- This design: Using Q(s,op) as a variance-reduced baseline for policy gradient (advantage actor-critic)

The conditioning is the same (value depends on action taken), but the learning algorithm is different.

Use hard one-hot conditioning in **both** rollout and PPO update:

```python
# WRONG - different conditioning creates GAE mismatch
# rollout:  softmax(op_logits).detach()  # soft
# update:   one_hot(sampled_op)          # hard

# CORRECT - same conditioning in both
op_one_hot = F.one_hot(sampled_op, num_classes=NUM_OPS).float()
value = value_head(cat(lstm_out, op_one_hot))
```

This treats the critic as `Q(s, op)` rather than `V(s)`. The value stored during rollout matches exactly what's trained during update — no advantage estimation mismatch.

### 8. Enum Normalization Strategy

> ⚠️ **CRITICAL: Most enums do NOT start at 0 and SeedStage has gaps**
>
> The codebase enums have non-sequential values:
> - `AlphaMode`: HOLD=0, UP=1, DOWN=2 — **starts at 0, OK**
> - `AlphaAlgorithm`: ADD=1, MULTIPLY=2, GATE=3 — **starts at 1, NOT 0**
> - `AlphaCurve`: LINEAR=1, COSINE=2, SIGMOID=3 — **starts at 1, NOT 0**
> - `SeedStage`: UNKNOWN=0, DORMANT=1, ..., HOLDING=6, FOSSILIZED=7, ..., RESETTING=10 — **gap at 5, max is 10**
>
> **Source files:**
> - `src/esper/leyline/alpha.py` — AlphaMode, AlphaAlgorithm, AlphaCurve
> - `src/esper/leyline/stages.py` — SeedStage

**Decision:** Use explicit index mappings (NOT `.value / max_value`).

The `.value / (len(Enum) - 1)` pattern only works for enums that start at 0 with contiguous values. Most of ours don't.

```python
# WRONG - would produce incorrect normalization for ADD=1, MULTIPLY=2, GATE=3:
# algo.value / (len(AlphaAlgorithm) - 1) = 1/2 for ADD, 2/2 for MULTIPLY, 3/2(!) for GATE
alpha_algorithm_norm = algo.value / (len(AlphaAlgorithm) - 1)  # DON'T DO THIS

# CORRECT - explicit index mapping:
_ALGO_TO_INDEX = {AlphaAlgorithm.ADD: 0, AlphaAlgorithm.MULTIPLY: 1, AlphaAlgorithm.GATE: 2}
alpha_algorithm_norm = _ALGO_TO_INDEX[algo] / (len(_ALGO_TO_INDEX) - 1)

# AlphaMode is the ONLY one where .value / max works (HOLD=0, UP=1, DOWN=2):
alpha_mode_norm = mode.value / (len(AlphaMode) - 1)  # OK for AlphaMode only

# AlphaCurve needs explicit mapping (LINEAR=1, COSINE=2, SIGMOID=3):
_CURVE_TO_INDEX = {AlphaCurve.LINEAR: 0, AlphaCurve.COSINE: 1, AlphaCurve.SIGMOID: 2}
alpha_curve_norm = _CURVE_TO_INDEX[curve] / (len(_CURVE_TO_INDEX) - 1)

# SeedStage: Use STAGE_TO_INDEX from stage_schema.py (handles gap at value 5):
from esper.leyline.stage_schema import STAGE_TO_INDEX, NUM_STAGES
stage_idx = STAGE_TO_INDEX[stage.value]
stage_norm = stage_idx / (NUM_STAGES - 1)
```

**Key insight:** The codebase already has `STAGE_TO_INDEX` in `stage_schema.py` for exactly this reason — SeedStage has a gap at value 5 (retired SHADOWING stage).

### 9. Inactive Slot Stage Encoding

Inactive slots (where `is_active_stage(report.stage)` returns False) must have **all-zeros** for stage one-hot, NOT stage 0. The vectorized one-hot helper must mask by validity:

> **Note:** `SeedStateReport` does NOT have an `is_active` field. Use `is_active_stage(report.stage)` from `esper.leyline.stages` instead.

```python
def _vectorized_one_hot(indices, table):
    valid_mask = indices >= 0  # -1 for inactive
    safe_indices = indices.clamp(min=0)
    one_hot = table[safe_indices]
    return one_hot * valid_mask.unsqueeze(-1).float()  # Zeros out inactive
```

### 10. Bootstrap Blueprint Indices (P1 from DRL Review)

At episode truncation, the bootstrap value computation needs **both** the final observation AND the final blueprint_indices:

```python
# In truncation handling (vectorized.py)
final_obs, final_blueprint_indices = batch_obs_to_features(final_signals, final_reports, ...)

# Use forward() to get correctly conditioned value (see Gotcha #2)
with torch.no_grad():
    forward_out = policy(final_obs, final_blueprint_indices, hidden)
    bootstrap_value = forward_out.value
```

If blueprint_indices are missing at truncation, the bootstrap value will be computed incorrectly.

### 11. Gradient Health Prev Tracking

The `gradient_health_prev` feature requires tracking the previous epoch's gradient health in `ParallelEnvState`. See **Phase 6b** for the complete state tracking contract including:
- Field definition in `ParallelEnvState`
- When to read (feature extraction)
- When to update (after epoch completion)
- Initial value (1.0 for new slots)
- Cleanup (delete on slot deactivation)

### 12. Action Feedback & Counterfactual Tracking

Both `last_action_success`/`last_action_op` (action feedback) and `epochs_since_counterfactual` require tracking in `ParallelEnvState`. See **Phase 6b** for:
- Field definitions
- Action success definition per `LifecycleOp`
- Update timing (after action execution, after epoch completion)
- Initial values

### 13. History Padding at Episode Start

At episode start, loss/accuracy history may be shorter than 5 epochs. Use left-padding with zeros:

```python
def _pad_history(history: list[float], length: int = 5) -> list[float]:
    if len(history) >= length:
        return history[-length:]
    return [0.0] * (length - len(history)) + history
```

This prevents `statistics.stdev()` crashes and gives the LSTM a consistent 5-element window.

---

## Success Criteria

1. ✅ All unit tests pass
2. ✅ Smoke test training completes without crashes
3. ✅ V3 learning curves match or exceed V2
4. ✅ Feature extraction is 5x+ faster than V2
5. ✅ Explained variance improves (less negative)
6. ✅ Multi-generation scaffolding credit assignment works
7. ✅ Per-head entropy stable for sparse heads (blueprint, tempo)
