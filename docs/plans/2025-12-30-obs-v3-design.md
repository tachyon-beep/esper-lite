# Obs 3.0 Design: Tamiyo Observation Space Overhaul

**Status:** Approved
**Date:** 2025-12-30
**Authors:** Claude (with DRL Expert, PyTorch Expert, Exploration Agent)

## Summary

Redesign Tamiyo's observation space to eliminate duplication, normalize all features, add missing signals, and improve construction efficiency.

| Metric | V2 (Current) | V3 (Proposed) | Change |
|--------|--------------|---------------|--------|
| Total dimensions | 218 | 117 (→124 with Policy V2) | -46% (→-43%) |
| Duplicated features | 27 dims | 0 | Eliminated |
| Unbounded features | 5+ | 0 | All bounded (see note) |
| Construction | Nested Python loops | Vectorized + cached tables | ~10x faster |

**Note on "All bounded":** Log-scale features (`loss_norm`, `loss_volatility`, `best_loss_norm`) can slightly exceed 1.0 for extreme values (e.g., loss > 10). This is acceptable as values remain bounded (~1.3 max) and maintain meaningful gradients. If strict [0,1] is needed, add final clamp.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| History handling | Compress to trend + volatility | Saves 6 dims, LSTM handles fine-grained patterns |
| Blueprint encoding | Learned 4-dim embeddings | Model discovers semantic relationships |
| Telemetry toggle | Remove, merge unique fields | Gradient health signals always valuable |
| Counterfactual freshness | Include as decay signal | Reliability indicator for contribution feature |

## Feature Specification

### Base Features (18 dims)

| Idx | Feature | Formula | Range | Purpose |
|-----|---------|---------|-------|---------|
| 0 | `epoch_progress` | `epoch / max_epochs` | [0,1] | Temporal position |
| 1 | `loss_norm` | `log(1 + val_loss) / log(11)` | [0,~1.3] | Task objective (log-scale) |
| 2 | `loss_delta_norm` | `clamp(loss_delta, -2, 2) / 2` | [-1,1] | Improvement signal |
| 3 | `accuracy_norm` | `val_accuracy / 100` | [0,1] | Task metric |
| 4 | `accuracy_delta_norm` | `clamp(acc_delta, -10, 10) / 10` | [-1,1] | Improvement signal |
| 5 | `plateau_norm` | `clamp(plateau_epochs, 0, 20) / 20` | [0,1] | Stagnation detection |
| 6 | `host_stabilized` | `1.0 if stabilized else 0.0` | {0,1} | Germination gate |
| 7 | `loss_trend` | `mean(loss_delta_history[-5:])` norm | [-1,1] | Compressed history |
| 8 | `loss_volatility` | `log(1 + std(loss_history[-5:])) / log(3)` | [0,~1] | Compressed history (log-scale) |
| 9 | `accuracy_trend` | `mean(acc_delta_history[-5:])` norm | [-1,1] | Compressed history |
| 10 | `accuracy_volatility` | `std(acc_history[-5:]) / 30` | [0,1] | Compressed history |
| 11 | `best_accuracy_norm` | `best_val_accuracy / 100` | [0,1] | Progress tracking |
| 12 | `best_loss_norm` | `log(1 + best_val_loss) / log(11)` | [0,~1.3] | Progress tracking (log-scale) |
| 13 | `params_utilization` | `log(1 + params) / log(1 + budget)` | [0,1] | Resource awareness |
| 14 | `slot_utilization` | `num_active / num_slots` | [0,1] | Germination opportunity |
| 15 | `num_training_norm` | `count(TRAINING) / num_slots` | [0,1] | Stage distribution |
| 16 | `num_blending_norm` | `count(BLENDING) / num_slots` | [0,1] | Stage distribution |
| 17 | `num_holding_norm` | `count(HOLDING) / num_slots` | [0,1] | Stage distribution |

### Per-Slot Features (29 dims × num_slots)

| Idx | Feature | Formula | Range | Purpose |
|-----|---------|---------|-------|---------|
| 0 | `is_active` | `1.0 if seed else 0.0` | {0,1} | Slot occupancy |
| 1-10 | `stage_one_hot` | One-hot of SeedStage | {0,1}^10 | State machine position |
| 11 | `alpha` | Direct | [0,1] | Current blend weight |
| 12 | `alpha_target` | Direct | [0,1] | Controller target |
| 13 | `alpha_mode_norm` | `_MODE_INDEX[mode] / (len(_MODE_INDEX) - 1)` | [0,1] | HOLD/RAMP/STEP (see note) |
| 14 | `alpha_velocity` | `clamp(velocity, -1, 1)` | [-1,1] | Rate of change |
| 15 | `time_to_target_norm` | `clamp(time, 0, max) / max` | [0,1] | Schedule progress |
| 16 | `alpha_algorithm_norm` | `(algo - min) / range` | [0,1] | ADD/MULTIPLY/GATE |
| 17 | `contribution_norm` | `clamp(contrib, -10, 10) / 10` | [-1,1] | Counterfactual attribution |
| 18 | `contribution_velocity_norm` | `clamp(vel, -10, 10) / 10` | [-1,1] | Trend for fossilize timing |
| 19 | `epochs_in_stage_norm` | `clamp(epochs, 0, 50) / 50` | [0,1] | Lifecycle timing |
| 20 | `gradient_norm_norm` | `clamp(grad, 0, 10) / 10` | [0,1] | Training health |
| 21 | `gradient_health` | Direct | [0,1] | Composite health score |
| 22 | `has_vanishing` | Binary | {0,1} | Prune signal |
| 23 | `has_exploding` | Binary | {0,1} | Prune signal |
| 24 | `interaction_sum_norm` | `clamp(sum, 0, 10) / 10` | [0,1] | Total synergy |
| 25 | `boost_received_norm` | `clamp(max, 0, 5) / 5` | [0,1] | Strongest interaction |
| 26 | `upstream_alpha_norm` | `clamp(sum, 0, 3) / 3` | [0,1] | Position context |
| 27 | `downstream_alpha_norm` | `clamp(sum, 0, 3) / 3` | [0,1] | Position context |
| 28 | `counterfactual_fresh` | `0.8 ** epochs_since_cf` | [0,1] | Attribution reliability |

**⚠️ Implementation Notes:**

1. **Enum normalization robustness (alpha_mode_norm, alpha_algorithm_norm):** Use explicit index mappings rather than `.value / max_value`. This prevents silent bugs if enum values change or are reordered:
   ```python
   _MODE_INDEX = {AlphaMode.HOLD: 0, AlphaMode.RAMP: 1, AlphaMode.STEP: 2}
   alpha_mode_norm = _MODE_INDEX[mode] / (len(_MODE_INDEX) - 1)
   ```

2. **Inactive slot stage encoding:** Inactive slots (`is_active=0`) must have **all-zeros** for stage one-hot, distinct from any valid stage. The vectorized one-hot helper must respect this:
   ```python
   def _vectorized_one_hot(indices: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
       valid_mask = indices >= 0  # -1 for inactive
       safe_indices = indices.clamp(min=0)
       one_hot = table.to(indices.device)[safe_indices]
       return one_hot * valid_mask.unsqueeze(-1).float()  # Zeros out inactive
   ```

### Blueprint Embeddings (4 dims × num_slots, separate)

Blueprint indices are passed to an `nn.Embedding(14, 4)` layer inside the policy network:
- 13 blueprint types + 1 null embedding for inactive slots
- Embedding learns semantic relationships between architectures
- Output: `[batch, num_slots, 4]` → flattened to `[batch, 12]`

### Total Dimensions

```
Base:              18
Slot 0:            29
Slot 1:            29
Slot 2:            29
Blueprint embed:   12  (4 × 3 slots)
─────────────────────
Total:            117
```

## Architecture Changes

### New BlueprintEmbedding Module

```python
class BlueprintEmbedding(nn.Module):
    def __init__(self, num_blueprints: int = 13, embed_dim: int = 4):
        super().__init__()
        # Index 13 = null embedding for inactive slots
        self.embedding = nn.Embedding(num_blueprints + 1, embed_dim)
        self.null_idx = num_blueprints

    def forward(self, blueprint_indices: torch.Tensor) -> torch.Tensor:
        # Map -1 (inactive) to null index
        safe_idx = blueprint_indices.clone()
        safe_idx[blueprint_indices < 0] = self.null_idx
        return self.embedding(safe_idx)
```

### Updated Policy Network

```python
class FactoredRecurrentActorCriticV3(nn.Module):
    def __init__(self, state_dim: int = 112, ...):
        # state_dim = 25 base (with action feedback) + 29*3 slots = 112
        # (Without action feedback: 18 + 29*3 = 105)
        # blueprint handled separately by embedding
        self.blueprint_embed = BlueprintEmbedding(13, 4)
        self.feature_net = nn.Linear(state_dim + 12, feature_dim)  # 112 + 12 = 124

    def forward(self, obs, blueprint_indices, hidden):
        bp_emb = self.blueprint_embed(blueprint_indices).view(obs.shape[0], -1)
        full_obs = torch.cat([obs, bp_emb], dim=-1)
        # ... rest of forward pass
```

**Dimension breakdown (with Policy V2 action feedback):**
- Base features: 18 original + 7 action feedback = 25
- Slot features: 29 × 3 = 87
- Non-blueprint obs: 25 + 87 = 112
- Blueprint embeddings: 4 × 3 = 12 (added inside network)
- Total network input: 112 + 12 = 124

### Feature Extraction API Change

```python
# V2 (current)
obs = batch_obs_to_features(signals, reports, use_telemetry=True, ...)
# Returns: [batch, 218]

# V3 (new, with Policy V2 action feedback)
obs, blueprint_idx = batch_obs_to_features_v3(signals, reports, ...)
# Returns: ([batch, 112], [batch, num_slots])
# Note: 112 = 25 base + 29*3 slots (excludes blueprint, includes action feedback)
```

## Vectorization Strategy

### Cached One-Hot Tables

```python
# Module-level constants (computed once at import)
_STAGE_ONE_HOT_TABLE = torch.eye(10, dtype=torch.float32)

def _vectorized_one_hot(indices: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    valid_mask = indices >= 0
    safe_indices = indices.clamp(min=0)
    one_hot = table.to(indices.device)[safe_indices]
    return one_hot * valid_mask.unsqueeze(-1).float()
```

### Pre-extraction Pattern

```python
def _extract_slot_arrays(batch_reports, slot_config) -> dict[str, np.ndarray]:
    # Single array assignment per slot instead of 25+ individual writes
    for env_idx, reports in enumerate(batch_reports):
        for slot_idx, slot_id in enumerate(slot_config.slot_ids):
            if report := reports.get(slot_id):
                scalars[env_idx, slot_idx] = _pack_scalars(report)  # One assignment
    return {"scalars": scalars, "stage_indices": stage_idx, ...}
```

## Migration Plan

### Phase 1: Feature Extraction
- Add `_extract_base_features_v3()`
- Add `_extract_slot_arrays()` with scalar packing
- Add `_vectorized_one_hot()` with cached tables
- Add `batch_obs_to_features_v3()` returning `(obs, blueprint_idx)`
- Unit tests for dimensions and value ranges

### Phase 2: Network Architecture
- Create `BlueprintEmbedding` module
- Create `FactoredRecurrentActorCriticV3`
- Update forward() signature
- Unit tests for shape correctness

### Phase 3: Integration
- Update `vectorized.py` to use V3 extraction
- Pass `blueprint_indices` to network
- Remove `--use-telemetry` CLI flag
- Update `get_feature_size()` for V3
- Integration tests

### Phase 4: Validation
- Sanity check: verify obs values in expected ranges
- Learning curve comparison: V2 vs V3
- Performance profiling

## Clean Replacement Strategy

Per `CLAUDE.md` no-legacy policy, we do **clean replacement**—no dual paths:

- Delete old feature extraction code as you implement V3
- No `_OBS_VERSION` toggle
- Rollback via git branch if needed

**The old observation space is part of why Tamiyo doesn't work.** No value in keeping it.

## Files to Modify

| File | Changes |
|------|---------|
| `src/esper/tamiyo/policy/features.py` | Replace feature extraction functions |
| `src/esper/tamiyo/networks/factored_lstm.py` | Add `BlueprintEmbedding`, new network |
| `src/esper/simic/agent/rollout_buffer.py` | Add `blueprint_indices` storage |
| `src/esper/simic/training/vectorized.py` | Update extraction calls, pass blueprint_idx |
| CLI args | Remove `--use-telemetry` flag |

## Breaking Changes

- **Checkpoint incompatibility:** V2 checkpoints cannot load into V3 networks
- **API change:** Feature extraction returns tuple `(obs, blueprint_indices)`
- **Telemetry toggle removed:** Always includes gradient health signals

## Success Criteria

1. V3 learning curves match or exceed V2 sample efficiency
2. Feature extraction 5x+ faster than V2 (profiled)
3. All features in documented ranges (unit tested)
4. No stale V2 code paths remain (clean replacement)

## Expert Reviews

### External Review (2025-12-30)

Identified three strategic constraints:

1. **LSTM Burden:** History compression forces LSTM to remember rather than see. **Verdict:** Good trade for recurrent policy.
2. **Linear Clamp Risk:** Original `clamp(loss, 0, 10) / 10` saturates on high-loss tasks. **Resolution:** Switched to log-scale normalization.
3. **Breaking Change Reality:** V2 checkpoints incompatible. **Verdict:** Acceptable, retrain from scratch.

### DRL Expert Sign-off (2025-12-30)

**Status:** APPROVED

| Concern | Verdict | Notes |
|---------|---------|-------|
| LSTM burden | Approved | 128 hidden size sufficient for 5-8 epoch decisions |
| Linear clamp | Approved with change | Switched to `log(1 + loss) / log(11)` |

Additional recommendations:
- Blueprint embedding init: `nn.init.normal_(weight, std=0.02)`
- Monitor feature correlations during training
- Ablation plan: V3 full vs V3+raw history vs V3+log norm

### PyTorch Expert Sign-off (2025-12-30)

**Status:** APPROVED

| Aspect | Verdict | Notes |
|--------|---------|-------|
| NumPy pre-extraction | Approved | Correct approach for minimizing Python overhead |
| Cached one-hot tables | Approved | Device handling is sound |
| Blueprint embedding in network | Approved | Clean separation, torch.compile friendly |
| torch.compile compatibility | Approved | No graph break risks |

Additional recommendations:
- Pre-allocate NumPy buffers to avoid per-step allocation
- Use `torch.int64` for blueprint indices (required by `nn.Embedding`)
- Benchmark with `torch.profiler` before/after
