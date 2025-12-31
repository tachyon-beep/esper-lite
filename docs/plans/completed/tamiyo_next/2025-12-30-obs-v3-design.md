# Obs 3.0 Design: Tamiyo Observation Space Overhaul

**Status:** Approved (Updated with P1 Decisions)
**Date:** 2025-12-30
**Authors:** Claude (with DRL Expert, PyTorch Expert, Exploration Agent)

> **P1 Update (2025-12-30):** Network dimensions increased from 256 to 512.
> See `tamiyo_next.md` for rationale. This document focuses on observation space;
> network architecture details in `2025-12-30-policy-v2-design.md`.

## Summary

Redesign Tamiyo's observation space to eliminate duplication, normalize all features, add missing signals, and improve construction efficiency.

| Metric | V2 (Current) | V3 (Proposed) | Change |
|--------|--------------|---------------|--------|
| Total dimensions | 218 | 133 (with Policy V2) | -39% |
| Duplicated features | 27 dims | 0 | Eliminated |
| Unbounded features | 5+ | 3 | log-scale loss features need clamp (see note) |
| Construction | Nested Python loops | Vectorized + cached tables | ~10x faster |

**Note on log-scale features:** Features using `log(1+loss)/log(11)` (`loss_norm`, `loss_history`, `best_loss_norm`) are UNBOUNDED—for loss=100, result is ~1.92; for loss=1000, result is ~2.88. **Recommendation:** Add `.clamp(max=2.0)` to bound these features in practice, or accept that early training with high losses may have features > 1.0.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| History handling | **Keep raw history (10 dims)** | LSTM needs sequence shape for credit assignment; trend+volatility loses temporal causality |
| Blueprint encoding | Learned 4-dim embeddings | Model discovers semantic relationships |
| Telemetry toggle | Remove, merge unique fields | Gradient health signals always valuable |
| Counterfactual freshness | Gamma-matched decay (0.995^epochs) | Matches PPO discount horizon; 0.8^epochs was too aggressive |
| Gradient trend | Add `gradient_health_prev` | Enables LSTM to distinguish transient vs sustained gradient issues |

## Feature Specification

### Base Features (24 dims)

| Idx | Feature | Formula | Range | Purpose |
|-----|---------|---------|-------|---------|
| 0 | `epoch_progress` | `epoch / max_epochs` | [0,1] | Temporal position |
| 1 | `loss_norm` | `log(1 + val_loss) / log(11)` | [0,~1.3] | Task objective (log-scale) |
| 2 | `loss_delta_norm` | `clamp(loss_delta, -2, 2) / 2` | [-1,1] | Improvement signal |
| 3 | `accuracy_norm` | `val_accuracy / 100` | [0,1] | Task metric |
| 4 | `accuracy_delta_norm` | `clamp(acc_delta, -10, 10) / 10` | [-1,1] | Improvement signal |
| 5 | `plateau_norm` | `clamp(plateau_epochs, 0, 20) / 20` | [0,1] | Stagnation detection |
| 6 | `host_stabilized` | `1.0 if stabilized else 0.0` | {0,1} | Germination gate |
| 7-11 | `loss_history[0:5]` | `log(1 + loss_i) / log(11)`, padded | [0,~1.3] | Raw loss trajectory (5 dims) |
| 12-16 | `acc_history[0:5]` | `acc_i / 100`, padded | [0,1] | Raw accuracy trajectory (5 dims) |
| 17 | `best_accuracy_norm` | `best_val_accuracy / 100` | [0,1] | Progress tracking |
| 18 | `best_loss_norm` | `log(1 + best_val_loss) / log(11)` | [0,~1.3] | Progress tracking (log-scale) |
| 19 | `params_utilization` | `log(1 + params) / log(1 + budget)` | [0,1] | Resource awareness |
| 20 | `slot_utilization` | `num_active / num_slots` | [0,1] | Germination opportunity |
| 21 | `num_training_norm` | `count(TRAINING) / num_slots` | [0,1] | Stage distribution |
| 22 | `num_blending_norm` | `count(BLENDING) / num_slots` | [0,1] | Stage distribution |
| 23 | `num_holding_norm` | `count(HOLDING) / num_slots` | [0,1] | Stage distribution |

**History padding:** Use `_pad_history(history, 5)` to left-pad short histories with zeros. This prevents `statistics.stdev()` crashes at episode start.

### Per-Slot Features (30 dims × num_slots)

| Idx | Feature | Formula | Range | Purpose |
|-----|---------|---------|-------|---------|
| 0 | `is_active` | `1.0 if seed else 0.0` | {0,1} | Slot occupancy |
| 1-10 | `stage_one_hot` | One-hot of SeedStage | {0,1}^10 | State machine position |
| 11 | `alpha` | Direct | [0,1] | Current blend weight |
| 12 | `alpha_target` | Direct | [0,1] | Controller target |
| 13 | `alpha_mode_norm` | `_MODE_INDEX[mode] / (len(_MODE_INDEX) - 1)` | [0,1] | HOLD/RAMP/STEP (see note) |
| 14 | `alpha_velocity` | `clamp(velocity, -1, 1)` | [-1,1] | Rate of change |
| 15 | `time_to_target_norm` | `clamp(time, 0, max) / max` | [0,1] | Schedule progress |
| 16 | `alpha_algorithm_norm` | `_ALGO_INDEX[algo] / (len(_ALGO_INDEX) - 1)` | [0,1] | ADD/MULTIPLY/GATE (explicit index) |
| 17 | `contribution_norm` | `clamp(contrib, -10, 10) / 10` | [-1,1] | Counterfactual attribution |
| 18 | `contribution_velocity_norm` | `clamp(vel, -10, 10) / 10` | [-1,1] | Trend for fossilize timing |
| 19 | `epochs_in_stage_norm` | `clamp(epochs, 0, MAX_EPOCHS_IN_STAGE) / MAX_EPOCHS_IN_STAGE` | [0,1] | Lifecycle timing |
| 20 | `gradient_norm_norm` | `clamp(grad, 0, 10) / 10` | [0,1] | Training health |
| 21 | `gradient_health` | Direct | [0,1] | Composite health score |
| 22 | `gradient_health_prev` | Previous epoch's gradient_health | [0,1] | **NEW:** Enables trend detection |
| 23 | `has_vanishing` | Binary | {0,1} | Prune signal |
| 24 | `has_exploding` | Binary | {0,1} | Prune signal |
| 25 | `interaction_sum_norm` | `clamp(sum, 0, 10) / 10` | [0,1] | Total synergy |
| 26 | `boost_received_norm` | `clamp(max, 0, 5) / 5` | [0,1] | Strongest interaction |
| 27 | `upstream_alpha_norm` | `clamp(sum, 0, 3) / 3` | [0,1] | Position context |
| 28 | `downstream_alpha_norm` | `clamp(sum, 0, 3) / 3` | [0,1] | Position context |
| 29 | `counterfactual_fresh` | `DEFAULT_GAMMA ** epochs_since_cf` | [0,1] | Attribution reliability (gamma-matched) |

**Counterfactual decay rationale:** Using `DEFAULT_GAMMA` (0.995) matches the PPO credit horizon. With 0.995^10 ≈ 0.95, measurements remain useful for ~138 epochs before dropping below 0.5 (ln(0.5)/ln(0.995) ≈ 138). The previous 0.8^epochs decay was too aggressive (0.8^10 = 0.1).

**⚠️ Implementation Notes:**

1. **Enum normalization (alpha_mode_norm, alpha_algorithm_norm):** Use `.value / (len(Enum) - 1)` for simplicity, but this REQUIRES enum values to be sequential starting from 0:
   ```python
   # Works because AlphaMode.HOLD=0, RAMP=1, STEP=2 (sequential)
   alpha_mode_norm = mode.value / (len(AlphaMode) - 1)
   ```
   If enum values become non-sequential, switch to explicit `_MODE_INDEX` mapping.

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
Base:              24  (was 18; +6 for raw history instead of compressed)
Action feedback:    7  (last_action_success + op one-hot) — see note below
Slot 0:            30  (was 29; +1 for gradient_health_prev)
Slot 1:            30
Slot 2:            30
─────────────────────
Non-blueprint obs: 121
Blueprint embed:    12  (4 × 3 slots, added inside network)
─────────────────────
Total network:    133
```

**Note on Action Feedback:** The 7-dim action feedback (indices 24-30) is defined in `2025-12-30-policy-v2-design.md`, not this document. It consists of:
- `last_action_success` (1 dim): Whether the previous action executed successfully
- `last_action_op` (6 dims): One-hot encoding of the previous operation

See `2025-12-30-policy-v2-design.md` "Action Feedback Features" section for full specification.

## Architecture Changes

### New BlueprintEmbedding Module

```python
from esper.leyline import NUM_BLUEPRINTS, BLUEPRINT_NULL_INDEX, DEFAULT_BLUEPRINT_EMBED_DIM

class BlueprintEmbedding(nn.Module):
    def __init__(
        self,
        num_blueprints: int = NUM_BLUEPRINTS,
        embed_dim: int = DEFAULT_BLUEPRINT_EMBED_DIM,
    ):
        super().__init__()
        # Index 13 = null embedding for inactive slots
        self.embedding = nn.Embedding(num_blueprints + 1, embed_dim)
        nn.init.normal_(self.embedding.weight, std=0.02)

        # Register null index as buffer: moves with module.to(device), no grad, in state_dict
        self.register_buffer(
            '_null_idx',
            torch.tensor(BLUEPRINT_NULL_INDEX, dtype=torch.int64)
        )

    def forward(self, blueprint_indices: torch.Tensor) -> torch.Tensor:
        # Map -1 (inactive) to null index WITHOUT per-call allocation
        # _null_idx is already on correct device via module.to(device)
        safe_idx = torch.where(blueprint_indices < 0, self._null_idx, blueprint_indices)
        return self.embedding(safe_idx)
```

**Note:** Avoid `.clone()` in forward pass—it allocates a new tensor every call. Use `torch.where()` instead.

### Updated Policy Network

```python
class FactoredRecurrentActorCriticV2(nn.Module):
    def __init__(self, state_dim: int = 121, feature_dim: int = 512, ...):
        # state_dim = 31 base (24 + 7 action feedback) + 30*3 slots = 121
        # blueprint handled separately by embedding
        self.blueprint_embed = BlueprintEmbedding()
        self.feature_net = nn.Linear(state_dim + 12, feature_dim)  # 121 + 12 = 133 → 512

    def forward(self, obs, blueprint_indices, hidden):
        bp_emb = self.blueprint_embed(blueprint_indices).view(obs.shape[0], -1)
        full_obs = torch.cat([obs, bp_emb], dim=-1)  # [batch, seq, 133]
        features = self.feature_net(full_obs)  # [batch, seq, 512]
        # ... rest of forward pass
```

**Dimension breakdown (with raw history + gradient_health_prev):**
- Base features: 24 (includes raw 10-dim history) + 7 action feedback = 31
- Slot features: 30 × 3 = 90 (includes gradient_health_prev)
- Non-blueprint obs: 31 + 90 = 121
- Blueprint embeddings: 4 × 3 = 12 (added inside network)
- Total network input: 121 + 12 = 133 → 512 via feature_net

### Feature Extraction API Change

```python
# V2 (current)
obs = batch_obs_to_features(signals, reports, use_telemetry=True, ...)
# Returns: [batch, 218]

# V3 (new, clean replacement - no _v3 suffix)
obs, blueprint_idx = batch_obs_to_features(signals, reports, slot_config, device)
# Returns: ([batch, 121], [batch, num_slots])
# Note: 121 = 31 base (24 + 7 action feedback) + 30*3 slots (excludes blueprint)
```

## Vectorization Strategy

### Cached One-Hot Tables

```python
# Module-level constants (computed once at import)
# IMPORTANT: Use leyline constant, NOT hard-coded 10 (prevents cardinality drift)
from esper.leyline.stage_schema import NUM_STAGES
_STAGE_ONE_HOT_TABLE = torch.eye(NUM_STAGES, dtype=torch.float32)

# Device-keyed cache to avoid per-step .to(device) allocations
_DEVICE_CACHE: dict[torch.device, torch.Tensor] = {}

def _normalize_device(device: torch.device | str) -> torch.device:
    """Normalize device to canonical form.

    Handles:
    - "cuda" or torch.device("cuda") → torch.device("cuda", current_device)
    - "cpu" or torch.device("cpu") → torch.device("cpu")
    - "cuda:0" → torch.device("cuda", 0)

    Prevents duplicate cache entries: torch.device('cuda') and
    torch.device('cuda:0') hash differently but refer to the same GPU.
    """
    if isinstance(device, str):
        device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device

def _get_cached_table(device: torch.device) -> torch.Tensor:
    """Get stage one-hot table for device, caching to avoid repeated transfers."""
    device = _normalize_device(device)  # Prevent cuda vs cuda:0 duplication
    if device not in _DEVICE_CACHE:
        _DEVICE_CACHE[device] = _STAGE_ONE_HOT_TABLE.to(device)
    return _DEVICE_CACHE[device]

def _vectorized_one_hot(indices: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Vectorized one-hot encoding with device-cached lookup table."""
    table = _get_cached_table(device)
    valid_mask = indices >= 0  # -1 marks inactive slots
    safe_indices = indices.clamp(min=0)
    one_hot = table[safe_indices]
    return one_hot * valid_mask.unsqueeze(-1).float()  # Zeros out inactive
```

**⚠️ Why device-keyed cache:** The original `.to(indices.device)` call allocates a new tensor on every step. With ~500 steps/episode, this adds significant overhead. The cache ensures one-time transfer per device.

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
- Replace `batch_obs_to_features()` to return `(obs, blueprint_idx)` tuple
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

**Status:** APPROVED (updated for 512 hidden)

| Concern | Verdict | Notes |
|---------|---------|-------|
| LSTM burden | Approved | 512 hidden size for 150-epoch 3-seed sequential scaffolding |
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
