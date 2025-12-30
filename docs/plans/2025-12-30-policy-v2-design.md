# Policy V2 Design: Architecture and Training Enhancements

**Status:** Approved
**Date:** 2025-12-30
**Authors:** Claude (with DRL Expert, PyTorch Expert)
**Depends on:** Obs V3 Design (2025-12-30-obs-v3-design.md)

## Summary

Enhance Tamiyo's policy architecture and training configuration to better support multi-generation seed scaffolding patterns with 50+ epoch decision horizons.

| Metric | V1 (Current) | V2 (Proposed) | Change |
|--------|--------------|---------------|--------|
| LSTM hidden dim | 128 | 256 | +100% capacity |
| Feature dim | 128 | 256 | Matched to LSTM |
| Total params | ~227K | ~830K | +266% |
| Observation dims | 117 | 124 | +7 (action feedback) |
| Blueprint head layers | 2 | 3 | +1 layer |
| Critic conditioning | None | Op-conditioned | +6 input dims |

## Design Rationale

### Core Insight: Multi-Generation Scaffolding

Test runs show that multi-generation seed scaffolding is a **common, high-value pattern**:

```
Seed A germinates → trains → scaffolds Seed B → Seed A pruned → Seed B succeeds
```

This creates 50+ epoch decision horizons where:
- Credit assignment spans multiple seed lifecycles
- The LSTM must track "who helped whom" across generations
- Value estimation depends on scaffold relationships, not just current state

### Why Current Architecture is Limiting

| Issue | Current (V1) | Impact |
|-------|--------------|--------|
| **No temporal enrichment** | input_dim (117) ≈ hidden_dim (128) | LSTM maintains but doesn't enrich |
| **Sparse head training** | Blueprint head trains 18% of steps | Under-explored during GERMINATE |
| **Value aliasing** | Same state → same value for all ops | Can't distinguish FOSSILIZE vs PRUNE value |
| **Short memory assumption** | Designed for 25-epoch horizons | Struggles with 50+ epoch scaffolding |

## Architecture Changes

### 1. LSTM Hidden Dimension: 128 → 256

**Rationale:**
- Current 128 provides no temporal enrichment (input ≈ hidden)
- 50+ epoch scaffolding chains need ~64 dims per slot for working memory
- 256 provides 2x capacity for temporal pattern learning

**Parameter impact:**
```
LSTM: 4 * (256*256 + 256^2 + 256) = ~525K params (was ~132K)
```

### 2. Feature Dimension: 128 → 256

**Rationale:** Match feature_dim to lstm_hidden_dim for consistent information flow.

```python
# Updated architecture
self.feature_net = nn.Sequential(
    nn.Linear(state_dim, 256),    # Was 128
    nn.LayerNorm(256),
    nn.ReLU(),
)
```

### 3. Blueprint Head: 2-layer → 3-layer

**Rationale:** Blueprint selection is the most consequential decision (determines seed architecture for entire lifecycle). More capacity helps discriminate between 13 blueprint types.

```python
# V1 (current)
self.blueprint_head = nn.Sequential(
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 13),
)

# V2 (proposed)
self.blueprint_head = nn.Sequential(
    nn.Linear(256, 256),   # Full width first layer
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 13),
)
```

**Parameter impact:** +~40K params

### 4. Operation-Conditioned Critic

**Rationale:** Same state can have different values depending on intended operation. "State S with FOSSILIZE" ≠ "State S with PRUNE" — especially when scaffold relationships exist.

```python
# V1 (current)
self.value_head = nn.Sequential(
    nn.Linear(lstm_hidden_dim, head_hidden),
    nn.ReLU(),
    nn.Linear(head_hidden, 1),
)

# V2 (proposed)
self.value_head = nn.Sequential(
    nn.Linear(lstm_hidden_dim + NUM_OPS, head_hidden),  # +6 for one-hot op
    nn.ReLU(),
    nn.Linear(head_hidden, 1),
)
```

**Forward pass change:**
```python
def forward(self, state, hidden, masks, op_for_value=None):
    # ... LSTM processing ...

    if op_for_value is not None:
        # During PPO update: condition on sampled op (hard one-hot)
        op_one_hot = F.one_hot(op_for_value, num_classes=NUM_OPS).float()
        value_input = torch.cat([lstm_out, op_one_hot], dim=-1)
    else:
        # During rollout: use expected value over ops (soft probs)
        # CRITICAL: .detach() prevents value loss from backprop into actor head
        op_probs = F.softmax(op_logits, dim=-1).detach()
        value_input = torch.cat([lstm_out, op_probs], dim=-1)

    value = self.value_head(value_input)
```

**⚠️ Gradient Isolation Warning:** Without `.detach()`, value loss would backpropagate through `op_logits`, training the actor to help the critic rather than to select good actions. This coupling destabilizes PPO.

**Parameter impact:** +~1K params

### 5. Other Head Dimensions

All other heads scale with LSTM hidden dim:

```python
head_hidden = lstm_hidden_dim // 2  # 128 (was 64)

# Each head: Linear(256, 128) → ReLU → Linear(128, head_dim)
```

## Observation Changes (Obs V3 Amendment)

### Action Feedback Features (+7 dims)

Add to base features in Obs V3:

| Idx | Feature | Formula | Range | Purpose |
|-----|---------|---------|-------|---------|
| 18 | `last_action_success` | `1.0 if executed else 0.0` | {0,1} | Immediate action feedback |
| 19-24 | `last_action_op` | One-hot of previous op | {0,1}^6 | Context for next decision |

**Updated Obs V3 totals:**
```
Base:              18 + 7 = 25
Slot 0:            29
Slot 1:            29
Slot 2:            29
Blueprint embed:   12
─────────────────────────
Total:            124 dims (was 117)
```

### Implementation

In `_extract_base_features_v3()`:
```python
# Action feedback (new in V2)
last_action_success = env_state.last_action_success  # bool → float
last_op_one_hot = F.one_hot(env_state.last_action_op, num_classes=6)

base_features = torch.cat([
    # ... existing 18 features ...
    last_action_success.unsqueeze(-1),
    last_op_one_hot,
], dim=-1)
```

## Training Changes

### 1. Differential Entropy Coefficients

**Rationale:** Sparse heads (blueprint, tempo) receive fewer training signals due to causal masking. Boost their entropy to maintain exploration when causally relevant.

```python
# In PPO config or __init__
entropy_coef_per_head = {
    "op": 1.0,           # Always active (100%)
    "slot": 1.0,         # Usually active (~60%)
    "blueprint": 1.5,    # GERMINATE only (~18%) — needs boost
    "style": 1.2,        # GERMINATE + SET_ALPHA (~22%)
    "tempo": 1.5,        # GERMINATE only (~18%) — needs boost
    "alpha_target": 1.2, # GERMINATE + SET_ALPHA (~22%)
    "alpha_speed": 1.3,  # SET_ALPHA + PRUNE (~19%)
    "alpha_curve": 1.3,  # SET_ALPHA + PRUNE (~19%)
}
```

**Usage in PPO update:**
```python
for head, entropy in per_head_entropy.items():
    coef = entropy_coef_per_head.get(head, 1.0)
    total_entropy_loss -= base_entropy_coef * coef * entropy.mean()
```

### 2. Blueprint Embedding Initialization

For the new `BlueprintEmbedding` in Obs V3, use small initialization:

```python
def __init__(self, num_blueprints: int = 13, embed_dim: int = 4):
    super().__init__()
    self.embedding = nn.Embedding(num_blueprints + 1, embed_dim)
    nn.init.normal_(self.embedding.weight, std=0.02)  # Small init
```

## Network Architecture Summary

```
Inputs:
  obs: [batch, seq, 112]              ← Non-blueprint features (25 base + 29×3 slots)
  blueprint_indices: [batch, seq, 3]  ← Integer indices for embedding lookup
    │
    ▼
BlueprintEmbedding(indices) → [batch, seq, 3, 4] → flatten → [batch, seq, 12]
    │
    ▼
Concat(obs, bp_emb) → [batch, seq, 112 + 12] = [batch, seq, 124]
    │
    ▼
Feature Net: Linear(124, 256) → LayerNorm → ReLU
    │
    ▼
LSTM(256, 256, num_layers=1, batch_first=True)
    │
    ▼
Post-LSTM LayerNorm(256)
    │
    ├─→ slot_head:         Linear(256,128) → ReLU → Linear(128, 3)
    ├─→ blueprint_head:    Linear(256,256) → ReLU → Linear(256,128) → ReLU → Linear(128, 13)  [3-layer]
    ├─→ style_head:        Linear(256,128) → ReLU → Linear(128, 4)
    ├─→ tempo_head:        Linear(256,128) → ReLU → Linear(128, 3)
    ├─→ alpha_target_head: Linear(256,128) → ReLU → Linear(128, 3)
    ├─→ alpha_speed_head:  Linear(256,128) → ReLU → Linear(128, 4)
    ├─→ alpha_curve_head:  Linear(256,128) → ReLU → Linear(128, 5)
    ├─→ op_head:           Linear(256,128) → ReLU → Linear(128, 6)
    │
    └─→ value_head:        Linear(256+6, 128) → ReLU → Linear(128, 1)  [op-conditioned]
```

## Parameter Count

| Component | V1 Params | V2 Params | Delta |
|-----------|-----------|-----------|-------|
| feature_net | ~18K | ~33K | +15K |
| LSTM | ~132K | ~528K | +396K |
| lstm_ln | ~0.3K | ~0.5K | +0.2K |
| blueprint_head | ~9K | ~100K | +91K |
| other 7 heads | ~60K | ~137K | +77K |
| value_head | ~8K | ~34K | +26K |
| **Total** | **~227K** | **~830K** | **+603K** |

**Memory estimate:**
- Parameters: 830K × 4 bytes = 3.3 MB
- Gradients: 830K × 4 bytes = 3.3 MB
- Optimizer state (Adam m+v): 830K × 8 bytes = 6.6 MB
- **Total model memory:** ~13 MB (trivial)

## Migration Plan

### Phase 1: Observation Update
1. Add `last_action_success` and `last_action_op` to env state tracking
2. Update `_extract_base_features_v3()` to include action feedback
3. Update `get_feature_size()` to return 124

### Phase 2: Network Architecture
1. Update `DEFAULT_LSTM_HIDDEN_DIM` to 256 in leyline
2. Update `DEFAULT_FEATURE_DIM` to 256 in leyline
3. Modify `FactoredRecurrentActorCriticV3` with:
   - 3-layer blueprint head
   - Op-conditioned value head
4. Update initialization code

### Phase 3: Training Config
1. Add `entropy_coef_per_head` to PPO config
2. Update PPO loss computation to use differential coefficients
3. Add blueprint embedding small init

### Phase 4: Validation
1. Verify forward pass shapes
2. Profile memory and speed
3. Compare learning curves: V1 vs V2
4. Monitor multi-generation scaffolding credit assignment

## Files to Modify

| File | Changes |
|------|---------|
| `src/esper/leyline/__init__.py` | Update DEFAULT_LSTM_HIDDEN_DIM, DEFAULT_FEATURE_DIM |
| `src/esper/tamiyo/policy/features.py` | Add action feedback to V3 extraction |
| `src/esper/tamiyo/networks/factored_lstm.py` | 3-layer blueprint head, op-conditioned critic |
| `src/esper/simic/agent/ppo.py` | Differential entropy coefficients |
| `src/esper/simic/training/vectorized.py` | Track last_action_success, last_action_op |

## Breaking Changes

- **Checkpoint incompatibility:** V1 checkpoints cannot load into V2 networks (different dimensions)
- **API change:** Value head forward requires op conditioning
- **Observation size:** 117 → 124 dims

## V1 Retention (Temporary)

Similar to Obs V3, keep V1 code paths until V2 is validated:

```python
# In leyline/__init__.py
_POLICY_VERSION = "v2"  # Toggle for testing

DEFAULT_LSTM_HIDDEN_DIM = 256 if _POLICY_VERSION == "v2" else 128
DEFAULT_FEATURE_DIM = 256 if _POLICY_VERSION == "v2" else 128

# TODO(policy-v2): Delete V1 code paths after validation
```

## Success Criteria

1. V2 learning curves match or exceed V1 on standard benchmarks
2. **Improved scaffolding credit assignment** — verify via hindsight reward attribution
3. Value explained variance improves (currently negative)
4. Blueprint head entropy maintained during GERMINATE (not collapsing)
5. Training speed within 2x of V1 (acceptable given 3x params)

## Monitoring Points

1. **Multi-generation credit:** Track `pending_hindsight_credit` distribution
2. **LSTM cell state magnitude:** Monitor for drift in long sequences
3. **Per-head gradient norms:** Ensure blueprint head gradients are healthy
4. **Op distribution:** Verify policy doesn't collapse to WAIT-only
5. **Value loss by op type:** Op-conditioned critic should show lower variance

## Future Considerations

### Multi-Generation Scaffolding Beyond 50 Epochs
- Consider LSTM hidden_dim 384
- Consider 2-layer LSTM with residual connections
- Consider attention over historical scaffold relationships

### Scaling to 5 Slots

**Observation impact:** 124 dims → 190 dims (+66)

| Component | 3 Slots | 5 Slots |
|-----------|---------|---------|
| Per-slot features | 87 | 145 |
| Blueprint embeddings | 12 | 20 |
| Total | 124 | 190 |

**LSTM capacity per slot:**

| Hidden Dim | 3 Slots | 5 Slots | Verdict |
|------------|---------|---------|---------|
| 256 | ~85 dims/slot | ~51 dims/slot | Tight for 5 slots |
| 384 | ~128 dims/slot | ~77 dims/slot | **Recommended for 5 slots** |
| 512 | ~170 dims/slot | ~102 dims/slot | Overkill unless needed |

**Scaling path:**
1. Current (3 slots): LSTM hidden_dim = 256
2. 5 slots: Increase to hidden_dim = 384
3. 5 slots + complex scaffolding: Consider 512 or 2-layer LSTM

**Note:** The feature_dim should always match lstm_hidden_dim for consistent information flow. When scaling to 384/512, update both.

## Expert Reviews

### DRL Expert Sign-off (2025-12-30)

**Status:** APPROVED

| Component | Verdict | Notes |
|-----------|---------|-------|
| LSTM 256 hidden | Approved | Adequate for 50+ epoch horizons with forget gate bias=1 |
| 3-layer blueprint head | Approved | Keep gain=0.01 for final layer only, sqrt(2) for intermediate |
| Op-conditioned critic | Approved | Use sampled op for terminal bootstrap |
| Differential entropy | Approved | Coefficients well-reasoned for sparse head training |
| Action feedback obs | Approved | Monitor for shortcut learning |

**Required clarifications:**
1. Value bootstrap at episode end: use sampled op from final step (not expected value)
2. Blueprint head init: apply gain=0.01 only to output layer, sqrt(2) to intermediate layers

**Monitoring recommendations:**
- Track LSTM cell state magnitude for drift beyond epoch 30-40
- Monitor per-head entropy to detect collapse in sparse heads
- Track explained_variance — should improve from current negative values

### PyTorch Expert Sign-off (2025-12-30)

**Status:** APPROVED

| Component | Verdict | Notes |
|-----------|---------|-------|
| Orthogonal initialization | Approved | Scales correctly to 256-dim |
| Op-conditioned critic | Approved | Soft probs during rollout, hard one-hot during update |
| torch.compile | Approved | No graph break concerns |
| Memory | Approved | ~13 MB total with optimizer (trivial) |
| Mixed precision | Approved | BF16 recommended on Ampere+ GPUs |

**Implementation notes:**
1. Op-conditioning approach:
   - Rollout (get_action): `value_input = cat(lstm_out, softmax(op_logits).detach())` — soft, **detached**
   - PPO update: `value_input = cat(lstm_out, one_hot(op_action))` — hard
2. **CRITICAL:** The `.detach()` on softmax during rollout prevents value loss from backpropagating into actor head, which would destabilize training
3. Current orthogonal init is appropriate — no changes needed for larger dims
4. Use `torch.amp.autocast(dtype=torch.bfloat16)` for training on Ampere+ GPUs
