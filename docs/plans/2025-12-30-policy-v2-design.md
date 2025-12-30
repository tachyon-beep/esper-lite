# Policy V2 Design: Architecture and Training Enhancements

**Status:** Approved (Updated with P1 Decisions)
**Date:** 2025-12-30
**Authors:** Claude (with DRL Expert, PyTorch Expert)
**Depends on:** Obs V3 Design (2025-12-30-obs-v3-design.md)

> **P1 Update (2025-12-30):** Dimensions increased from 256 to 512 based on specialist review.
> 150-epoch sequential 3-seed scaffolding requires larger hidden state to prevent
> "Catastrophic Overwrite" of archival seed memories. See `tamiyo_next.md` for rationale.

## Summary

Enhance Tamiyo's policy architecture and training configuration to better support multi-generation seed scaffolding patterns with 50+ epoch decision horizons.

| Metric | V1 (Current) | V2 (Proposed) | Change |
|--------|--------------|---------------|--------|
| LSTM hidden dim | 128 | 512 | +300% capacity |
| Feature dim | 128 | 512 | Matched to LSTM |
| Total params | ~227K | ~2.1M | +825% |
| Observation dims | 218 | 133 | -39% (Obs V3 consolidation) |
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

### 1. LSTM Hidden Dimension: 128 → 512

**Rationale:**
- Current 128 provides no temporal enrichment (input ≈ hidden)
- 150-epoch sequential 3-seed scaffolding needs larger working memory
- 512 provides sufficient "archival slots" to prevent Catastrophic Overwrite
- Power-of-2 optimal for GPU tensor cores (divisible by 8 for AMP)

**Parameter impact:**
```
LSTM: 4 * (512*512 + 512^2 + 512) = ~2.1M params (was ~132K)
```

### 2. Feature Dimension: 128 → 512

**Rationale:** Match feature_dim to lstm_hidden_dim for consistent information flow. Prevents information bottleneck before LSTM.

```python
# Updated architecture
self.feature_net = nn.Sequential(
    nn.Linear(state_dim, 512),    # Was 128
    nn.LayerNorm(512),
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
    nn.Linear(512, 512),   # Full width first layer
    nn.ReLU(),
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Linear(128, 13),
)
```

**Parameter impact:** +~330K params

### 4. Operation-Conditioned Critic

**Rationale:** Same state can have different values depending on intended operation. "State S with FOSSILIZE" ≠ "State S with PRUNE" — especially when scaffold relationships exist.

```python
# V1 (current)
self.value_head = nn.Sequential(
    nn.Linear(lstm_hidden_dim, head_hidden),  # 128 → 64
    nn.ReLU(),
    nn.Linear(head_hidden, 1),
)

# V2 (proposed)
self.value_head = nn.Sequential(
    nn.Linear(lstm_hidden_dim + NUM_OPS, head_hidden),  # 512+6 → 256
    nn.ReLU(),
    nn.Linear(head_hidden, 1),
)
```

**Forward pass change:**
```python
def forward(self, state, hidden, masks, sampled_op: torch.Tensor):
    # ... LSTM processing ...

    # Op-conditioned critic: always condition on sampled op (hard one-hot)
    # This treats the critic as Q(s, op) — a lightweight action-value baseline
    op_one_hot = F.one_hot(sampled_op, num_classes=NUM_OPS).float()
    value_input = torch.cat([lstm_out, op_one_hot], dim=-1)
    value = self.value_head(value_input)
```

**Design choice: Q(s, op) pattern**

We use hard one-hot conditioning in **both** rollout and PPO update for consistency:
- Rollout: `value = Q(s, sampled_op)` — value stored in buffer matches op taken
- PPO update: same `Q(s, sampled_op)` — no mismatch in advantage estimation

This avoids the alternative (soft probs during rollout, hard one-hot during update) which creates GAE mismatch because the value function trained differs from the one used during rollout.

For logging/monitoring, expected value can be computed separately:
```python
# Optional: compute V(s) = E[Q(s,op)] for monitoring only
with torch.no_grad():
    op_probs = F.softmax(op_logits, dim=-1)
    q_all = torch.stack([value_head(cat(lstm_out, one_hot(i))) for i in range(NUM_OPS)])
    expected_value = (op_probs * q_all).sum(dim=-1)
```

**Parameter impact:** +~1K params

### 5. Other Head Dimensions

All other heads scale with LSTM hidden dim:

```python
head_hidden = lstm_hidden_dim // 2  # 256 (was 64)

# Each head: Linear(512, 256) → ReLU → Linear(256, head_dim)
```

## Observation Changes (Obs V3 Amendment)

### Action Feedback Features (+7 dims)

Add to base features in Obs V3:

| Idx | Feature | Formula | Range | Purpose |
|-----|---------|---------|-------|---------|
| 24 | `last_action_success` | `1.0 if executed else 0.0` | {0,1} | Immediate action feedback |
| 25-30 | `last_action_op` | One-hot of previous op | {0,1}^6 | Context for next decision |

**Note:** Indices 0-23 are the base features from Obs V3 (see `obs-v3-design.md`). Action feedback is appended at indices 24-30.

**Updated Obs V3 totals:**
```
Base:              24 (includes 10-dim raw history)
Action feedback:    7 (success + op one-hot)
Slot 0:            30 (includes gradient_health_prev)
Slot 1:            30
Slot 2:            30
─────────────────────────
Non-blueprint obs: 121
Blueprint embed:   12 (4 × 3, added inside network)
─────────────────────────
Total network:    133 dims
```

### Implementation

In `_extract_base_features_v3()`:
```python
# Action feedback (new in V2) - appended after 24 base features
last_action_success = float(env_state.last_action_success)  # bool → float
last_op_one_hot = F.one_hot(torch.tensor(env_state.last_action_op), num_classes=6).float()

base_features = torch.cat([
    # ... existing 24 base features (indices 0-23) ...
    torch.tensor([last_action_success]),  # index 24
    last_op_one_hot,                       # indices 25-30
], dim=-1)  # Total: 31 dims
```

## Training Changes

### 1. Differential Entropy Coefficients

**Rationale:** Sparse heads (blueprint, tempo) receive fewer training signals due to causal masking. Boost their entropy to maintain exploration when causally relevant.

```python
# In PPO config or __init__
entropy_coef_per_head = {
    "op": 1.0,           # Always active (100%)
    "slot": 1.0,         # Usually active (~60%)
    "blueprint": 1.3,    # GERMINATE only (~18%) — needs boost
    "style": 1.2,        # GERMINATE + SET_ALPHA (~22%)
    "tempo": 1.3,        # GERMINATE only (~18%) — needs boost
    "alpha_target": 1.2, # GERMINATE + SET_ALPHA (~22%)
    "alpha_speed": 1.2,  # SET_ALPHA + PRUNE (~19%)
    "alpha_curve": 1.2,  # SET_ALPHA + PRUNE (~19%)
}
# Note: Start conservative (1.2-1.3x), tune empirically if heads collapse
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
  obs: [batch, seq, 121]              ← Non-blueprint features (31 base + 30×3 slots)
  blueprint_indices: [batch, seq, 3]  ← Integer indices for embedding lookup
    │
    ▼
BlueprintEmbedding(indices) → [batch, seq, 3, 4] → flatten → [batch, seq, 12]
    │
    ▼
Concat(obs, bp_emb) → [batch, seq, 121 + 12] = [batch, seq, 133]
    │
    ▼
Feature Net: Linear(133, 512) → LayerNorm → ReLU
    │
    ▼
LSTM(512, 512, num_layers=1, batch_first=True)
    │
    ▼
Post-LSTM LayerNorm(512)
    │
    ├─→ slot_head:         Linear(512,256) → ReLU → Linear(256, 3)
    ├─→ blueprint_head:    Linear(512,512) → ReLU → Linear(512,128) → ReLU → Linear(128, 13)  [3-layer]
    ├─→ style_head:        Linear(512,256) → ReLU → Linear(256, 4)
    ├─→ tempo_head:        Linear(512,256) → ReLU → Linear(256, 3)
    ├─→ alpha_target_head: Linear(512,256) → ReLU → Linear(256, 3)
    ├─→ alpha_speed_head:  Linear(512,256) → ReLU → Linear(256, 4)
    ├─→ alpha_curve_head:  Linear(512,256) → ReLU → Linear(256, 5)
    ├─→ op_head:           Linear(512,256) → ReLU → Linear(256, 6)
    │
    └─→ value_head:        Linear(512+6, 256) → ReLU → Linear(256, 1)  [op-conditioned]
```

## Parameter Count

| Component | V1 Params | V2 Params | Formula |
|-----------|-----------|-----------|---------|
| feature_net | ~18K | ~69K | Linear(133,512) + LN(512) |
| LSTM | ~132K | ~2.1M | 4×512×512 + 4×512² + 8×512 |
| lstm_ln | ~0.3K | ~1K | LN(512) |
| blueprint_head | ~9K | ~330K | 512→512→128→13 (3-layer) |
| 7 standard heads | ~60K | ~920K | 7 × (512→256→k), ~130K each |
| value_head | ~8K | ~134K | 518→256→1 |
| blueprint_embed | - | ~0.1K | Embedding(14, 4) |
| **Total** | **~227K** | **~2.1M** | **+825%** |

**Verification:** Run `sum(p.numel() for p in model.parameters())` on the instantiated network. The above is hand-calculated; implementation may vary slightly.

**Memory estimate:**
- Parameters: 2.1M × 4 bytes = 8.4 MB
- Gradients: 2.1M × 4 bytes = 8.4 MB
- Optimizer state (Adam m+v): 2.1M × 8 bytes = 16.8 MB
- **Total model memory:** ~34 MB (still trivial)

## Migration Plan

### Phase 1: Observation Update
1. Add `last_action_success` and `last_action_op` to env state tracking
2. Update `_extract_base_features_v3()` to include action feedback
3. Update `get_feature_size()` to return 121 (non-blueprint obs; network adds 12 for embeddings)

### Phase 2: Network Architecture
1. Update `DEFAULT_LSTM_HIDDEN_DIM` to 512 in leyline ✓ (already done)
2. Update `DEFAULT_FEATURE_DIM` to 512 in leyline ✓ (already done)
3. Modify `FactoredRecurrentActorCriticV2` with:
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
- **Observation size:** 218 → 133 dims (Obs V3 consolidation)

## Clean Replacement Strategy

Per `CLAUDE.md` no-legacy policy, we do **clean replacement**—no dual paths:

```python
# In leyline/__init__.py - direct values, no conditionals
DEFAULT_LSTM_HIDDEN_DIM = 512
DEFAULT_FEATURE_DIM = 512
```

Delete old network code as you implement V2. Rollback via git branch if needed.

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

### Multi-Generation Scaffolding Beyond 150 Epochs
- Consider 2-layer LSTM with residual connections for 200+ epoch horizons
- Consider attention over historical scaffold relationships for complex synergy patterns

### Scaling to 5 Slots

**Observation impact:** 133 dims → 193 dims (+60)

| Component | 3 Slots | 5 Slots |
|-----------|---------|---------|
| Per-slot features | 90 | 150 |
| Blueprint embeddings | 12 | 20 |
| Total | 133 | 193 |

**LSTM capacity per slot (with 512 hidden):**

| Hidden Dim | 3 Slots | 5 Slots | Verdict |
|------------|---------|---------|---------|
| 512 | ~170 dims/slot | ~102 dims/slot | **Current design, adequate for 5 slots** |
| 768 | ~256 dims/slot | ~154 dims/slot | Consider for complex 5-slot scaffolding |

**Scaling path:**
1. Current (3 slots): LSTM hidden_dim = 512 ✓
2. 5 slots: 512 should be adequate for most scenarios
3. 5 slots + very complex scaffolding: Consider 768 or 2-layer LSTM

**Note:** The feature_dim should always match lstm_hidden_dim for consistent information flow.

## Expert Reviews

### DRL Expert Sign-off (2025-12-30)

**Status:** APPROVED (updated for 512 hidden)

| Component | Verdict | Notes |
|-----------|---------|-------|
| LSTM 512 hidden | Approved | Required for 150-epoch 3-seed sequential scaffolding; prevents Catastrophic Overwrite |
| 3-layer blueprint head | Approved | Keep gain=0.01 for final layer only, sqrt(2) for intermediate |
| Op-conditioned critic | Approved | Use sampled op for terminal bootstrap |
| Differential entropy | Approved | Coefficients well-reasoned for sparse head training |
| Action feedback obs | Approved | Monitor for shortcut learning |

**Required clarifications:**
1. Value bootstrap at episode end: use sampled op from final step (not expected value)
2. Blueprint head init: apply gain=0.01 only to output layer, sqrt(2) to intermediate layers

**Monitoring recommendations:**
- Track LSTM cell state magnitude for drift beyond epoch 100-120 (longer horizon now)
- Monitor per-head entropy to detect collapse in sparse heads
- Track explained_variance — should improve from current negative values

### PyTorch Expert Sign-off (2025-12-30)

**Status:** APPROVED (updated for 512 hidden)

| Component | Verdict | Notes |
|-----------|---------|-------|
| Orthogonal initialization | Approved | Scales correctly to 512-dim |
| Op-conditioned critic | Approved | Hard one-hot in both rollout and update for consistency |
| torch.compile | Approved | No graph break concerns |
| Memory | Approved | ~34 MB total with optimizer (trivial) |
| Mixed precision | Approved | BF16 recommended on Ampere+ GPUs |

**Implementation notes:**
1. Op-conditioning approach (Q(s,op) pattern):
   - Rollout: `value_input = cat(lstm_out, one_hot(sampled_op))` — hard
   - PPO update: `value_input = cat(lstm_out, one_hot(sampled_op))` — same hard one-hot
   - **Consistency:** Both use the same conditioning to avoid GAE mismatch
2. Current orthogonal init is appropriate — no changes needed for larger dims
3. Use `torch.amp.autocast(dtype=torch.bfloat16)` for training on Ampere+ GPUs
4. **Param count verification:** Run `sum(p.numel() for p in model.parameters())` after instantiation
