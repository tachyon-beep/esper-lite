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

**Note:** Indices 0-23 are the base features from Obs V3 (see `2025-12-30-obs-v3-design.md`). Action feedback is appended at indices 24-30.

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

## Multi-Head PPO Validation Checklist

This section documents potential silent bugs in the multi-head action architecture that require explicit validation.

### 1. Per-Head Advantage Masking Verification

**Risk:** When `op=WAIT`, other heads (slot, blueprint, etc.) should be masked to stable "noop" values. The current design uses blueprint index 0 as "noop". If masking fails, blueprint head will "flap around unpredictably" — outputting arbitrary values that receive no gradient but pollute the action distribution.

**Unit test template:**

```python
def test_advantage_masking_wait_op():
    """Verify that non-op heads receive zero advantage when op=WAIT."""
    batch = make_test_batch(ops=[OpType.WAIT] * 32)

    advantages = compute_per_head_advantages(batch)

    # Op head should have non-zero advantage
    assert advantages["op"].abs().sum() > 0

    # All other heads must have exactly zero advantage during WAIT
    for head in ["slot", "blueprint", "style", "tempo", "alpha_target", "alpha_speed", "alpha_curve"]:
        assert advantages[head].abs().sum() == 0, f"{head} got non-zero advantage during WAIT"


def test_blueprint_noop_during_non_germinate():
    """Verify blueprint head outputs index 0 (noop) when not germinating."""
    policy = make_policy()
    state = make_state(can_germinate=False)

    with torch.no_grad():
        action = policy.get_action(state)

    # If op is not GERMINATE, blueprint should be masked to 0
    if action.op != OpType.GERMINATE:
        assert action.blueprint_idx == 0, "Blueprint should be noop (0) when not germinating"
```

### 2. ADVANCE Op Gradient Flow Test

**Risk:** The code specifies blueprint head gets advantage only when `op==GERMINATE`, slot head gets advantage for any op except `WAIT`. `ADVANCE` uses only slot head, but there's no explicit `is_advance` mask — it relies on implicit `~is_wait`. This could allow style/alpha heads to receive spurious gradients during `ADVANCE`.

**Validation:**

```python
def test_advance_op_gradient_isolation():
    """Verify style/alpha heads receive zero advantage during ADVANCE."""
    batch = make_test_batch(ops=[OpType.ADVANCE] * 32)

    advantages = compute_per_head_advantages(batch)

    # During ADVANCE: only op and slot heads should have advantage
    assert advantages["op"].abs().sum() > 0
    assert advantages["slot"].abs().sum() > 0

    # Style/alpha heads must have exactly zero advantage during ADVANCE
    for head in ["style", "alpha_target", "alpha_speed", "alpha_curve"]:
        assert advantages[head].abs().sum() == 0, f"{head} got spurious advantage during ADVANCE"

    # Blueprint and tempo also zero (only active during GERMINATE)
    assert advantages["blueprint"].abs().sum() == 0
    assert advantages["tempo"].abs().sum() == 0
```

**Implementation note:** The advantage computation should use explicit op-type masks:

```python
is_germinate = (ops == OpType.GERMINATE)
is_set_alpha = (ops == OpType.SET_ALPHA)
is_wait = (ops == OpType.WAIT)

# Blueprint head: only GERMINATE
advantage_mask["blueprint"] = is_germinate

# Style head: GERMINATE or SET_ALPHA
advantage_mask["style"] = is_germinate | is_set_alpha

# Slot head: everything except WAIT
advantage_mask["slot"] = ~is_wait
```

### 3. Joint Policy-Value Consistency

**Risk:** `V(s)` is unconditioned on action, but our `Q(s, op)` is conditioned on op. If certain ops consistently lead to large positive/negative advantages, the baseline may be systematically misestimating value.

**Monitoring:**

```python
def monitor_advantage_bias_by_op(batch):
    """Detect systematic advantage bias by operation type."""
    for op in OpType:
        op_mask = batch.ops == op
        if op_mask.sum() == 0:
            continue

        op_advantages = batch.advantages[op_mask]
        mean_adv = op_advantages.mean().item()
        std_adv = op_advantages.std().item()

        # Alert if any op has persistent bias > 1 std
        if abs(mean_adv) > std_adv and op_mask.sum() > 100:
            log.warning(
                f"Op {op.name} shows systematic advantage bias: "
                f"mean={mean_adv:.3f}, std={std_adv:.3f}, n={op_mask.sum()}"
            )
```

**Expected behavior:** Advantages should be zero-mean per op over a training run. Persistent bias indicates:
- Value function is underestimating certain op outcomes
- Op selection policy is suboptimal (keep choosing low-value ops)
- Reward attribution is incorrectly assigned

### 4. Sparse Head Entropy Monitoring

**Risk:** Sparse heads (blueprint ~18% active, tempo ~18% active) receive fewer training signals due to causal masking. They could become deterministic simply due to lack of feedback, collapsing to a single action before being meaningfully explored.

**Expected entropy values per head:**

| Head | N Actions | Max Entropy (ln N) | Warning Threshold | Critical Threshold |
|------|-----------|-------------------|-------------------|-------------------|
| op | 6 | 1.79 | < 1.0 | < 0.5 |
| slot | 3 | 1.10 | < 0.6 | < 0.3 |
| blueprint | 13 | 2.56 | < 1.5 | < 0.7 |
| style | 4 | 1.39 | < 0.8 | < 0.4 |
| tempo | 3 | 1.10 | < 0.6 | < 0.3 |
| alpha_target | 3 | 1.10 | < 0.6 | < 0.3 |
| alpha_speed | 4 | 1.39 | < 0.8 | < 0.4 |
| alpha_curve | 5 | 1.61 | < 0.9 | < 0.5 |

**Monitoring implementation:**

```python
def monitor_sparse_head_entropy(policy, telemetry):
    """Track per-head entropy with sparse-head-aware thresholds."""
    ENTROPY_THRESHOLDS = {
        "blueprint": {"warn": 1.5, "critical": 0.7, "max": 2.56},
        "tempo": {"warn": 0.6, "critical": 0.3, "max": 1.10},
        "style": {"warn": 0.8, "critical": 0.4, "max": 1.39},
        "alpha_target": {"warn": 0.6, "critical": 0.3, "max": 1.10},
    }

    for head, thresholds in ENTROPY_THRESHOLDS.items():
        entropy = telemetry.get(f"entropy/{head}", 0.0)
        usage_pct = telemetry.get(f"head_usage/{head}", 0.0)

        # A head's entropy dropping to zero before meaningful use indicates collapse
        if entropy < thresholds["critical"]:
            log.error(
                f"CRITICAL: {head} entropy collapsed to {entropy:.3f} "
                f"(max={thresholds['max']:.2f}, usage={usage_pct:.1%})"
            )
        elif entropy < thresholds["warn"]:
            log.warning(
                f"WARNING: {head} entropy low at {entropy:.3f} "
                f"(max={thresholds['max']:.2f}, usage={usage_pct:.1%})"
            )

        # Special check: entropy collapse before meaningful training
        if entropy < thresholds["warn"] and usage_pct < 0.10:
            log.error(
                f"SILENT BUG: {head} collapsed (entropy={entropy:.3f}) "
                f"before meaningful use (usage={usage_pct:.1%})"
            )
```

**Minimum entropy coefficients:** The differential entropy coefficients in the Training Changes section (1.3x for blueprint/tempo, 1.2x for style/alpha heads) are designed to counteract this sparsity. If entropy still drops below warning thresholds, consider:
1. Increasing the multiplier (1.3 → 1.5)
2. Adding entropy bonus only when head is actually used
3. Using separate optimizers with different learning rates per head

### 5. Additional Multi-Head PPO Checks

These additional checks address subtle failure modes identified during DRL expert review.

#### 5.1 Log Probability Computation - Finite Check

**Risk:** Masked heads may produce `-inf` log probabilities if masked actions are treated as "out-of-distribution" rather than valid fallback values. When a head is masked, the action should be set to index 0 (noop), which is a valid action—not an impossible one.

**Validation:**

```python
def test_masked_head_log_probs_finite():
    """Verify masked heads produce finite log probs, not -inf."""
    policy = make_policy()
    batch = make_test_batch(ops=[OpType.WAIT] * 32)

    # During WAIT, only op head is active; others are masked to index 0
    log_probs = policy.evaluate_actions(batch.states, batch.actions, batch.masks)

    # All log probs must be finite (masked heads use index 0, which is valid)
    for head, lp in log_probs.items():
        assert torch.isfinite(lp).all(), (
            f"{head} produced non-finite log probs: "
            f"min={lp.min():.3f}, max={lp.max():.3f}, "
            f"inf_count={(~torch.isfinite(lp)).sum()}"
        )

    # Specifically check that masked heads don't have -inf
    for head in ["slot", "blueprint", "style", "tempo"]:
        if head in log_probs:
            assert (log_probs[head] > -1e10).all(), (
                f"{head} log prob is effectively -inf during WAIT: {log_probs[head].min():.3f}"
            )
```

**Root cause when this fails:** The action passed to `log_prob()` is not index 0 for masked heads, or the distribution was computed with an action mask that excludes index 0.

#### 5.2 Entropy Gradient to Masked Heads

**Risk:** Masked heads receive zero advantage (correctly), but they should still receive gradient flow via the entropy term. Without entropy gradients, masked heads will drift—their distribution slowly becomes arbitrary because nothing anchors them.

**Validation:**

```python
def test_masked_head_receives_entropy_gradient():
    """Verify masked heads receive gradient via entropy term even with zero advantage."""
    policy = make_policy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    # Create batch where blueprint head is always masked (no GERMINATE ops)
    batch = make_test_batch(ops=[OpType.WAIT, OpType.ADVANCE, OpType.SET_ALPHA] * 10)

    # Store initial blueprint head weights
    bp_weight_before = policy.blueprint_head[-1].weight.clone()

    # Compute loss with entropy term
    _, log_probs, entropies, _ = policy.evaluate_actions(batch.states, batch.actions, batch.masks)

    # Advantage is zero for blueprint (masked), but entropy should be non-zero
    advantage_loss = torch.tensor(0.0)  # Blueprint advantage is zero
    entropy_loss = -0.01 * entropies["blueprint"].mean()  # Entropy term still applies

    total_loss = advantage_loss + entropy_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # Blueprint head weights should have changed due to entropy gradient
    bp_weight_after = policy.blueprint_head[-1].weight
    weight_delta = (bp_weight_after - bp_weight_before).abs().sum()

    assert weight_delta > 1e-8, (
        f"Blueprint head received no gradient despite entropy term: "
        f"weight_delta={weight_delta:.2e}"
    )
```

**Implementation requirement:** Entropy must be computed for ALL heads every step, not just causally-active ones. The entropy loss is:

```python
# Correct: compute entropy for all heads
total_entropy = sum(ent.mean() for ent in per_head_entropy.values())

# Wrong: only compute entropy for active heads
# total_entropy = sum(ent[mask].mean() for head, ent in per_head_entropy.items() if mask.any())
```

#### 5.3 Action Mask Propagation Consistency

**Risk:** If the action masks used during rollout differ from those used during PPO update, you get "impossible action" log probabilities. The rollout might sample action=5 when the update's mask says action=5 is invalid, producing `-inf` log prob and NaN gradients.

**Validation:**

```python
def test_action_mask_consistency():
    """Verify rollout masks match update masks exactly."""
    env = make_env()
    policy = make_policy()

    # Collect rollout with masks
    states, actions, masks_rollout = [], [], []
    state = env.reset()
    for _ in range(100):
        mask = env.get_action_mask()
        action = policy.get_action(state, mask)

        # Store the mask used during rollout
        masks_rollout.append(mask.clone())
        states.append(state.clone())
        actions.append(action.clone())

        state, _, done, _ = env.step(action)
        if done:
            state = env.reset()

    # Reconstruct masks during update (simulating what PPO does)
    for i, (state, action, mask_rollout) in enumerate(zip(states, actions, masks_rollout)):
        mask_update = env.get_action_mask_for_state(state)  # Recompute mask

        # Masks must be identical
        for head in mask_rollout.keys():
            assert torch.equal(mask_rollout[head], mask_update[head]), (
                f"Mask mismatch at step {i} for {head}: "
                f"rollout={mask_rollout[head].tolist()}, "
                f"update={mask_update[head].tolist()}"
            )

        # Verify sampled action is valid under both masks
        for head, act in action.items():
            if not mask_rollout[head][act]:
                raise AssertionError(
                    f"Step {i}: {head} action {act} was sampled but is invalid under rollout mask"
                )
```

**Implementation pattern:** Store masks in the rollout buffer alongside actions, then use the STORED masks during PPO update—don't recompute them from state.

```python
# In rollout collection
buffer.add(state=state, action=action, mask=mask, ...)  # Store mask

# In PPO update
for batch in buffer.get_batches():
    log_probs = policy.evaluate_actions(batch.states, batch.actions, batch.masks)  # Use stored mask
```

#### 5.4 Per-Head Ratio Explosion Detection

**Risk:** In multi-head PPO, the joint ratio (product of per-head ratios) can look healthy while individual heads have exploded. A single head with ratio=10 multiplied by seven heads with ratio=0.9 gives joint ratio ~4.8—potentially clipped and looking "fine"—but that one head is unstable.

**Monitoring:**

```python
def monitor_per_head_ratios(old_log_probs: dict, new_log_probs: dict, telemetry):
    """Detect per-head ratio explosion before joint ratio shows symptoms."""
    RATIO_WARN_THRESHOLD = 3.0   # log(ratio) > 1.1
    RATIO_CRIT_THRESHOLD = 10.0  # log(ratio) > 2.3

    for head in old_log_probs.keys():
        log_ratio = new_log_probs[head] - old_log_probs[head]
        ratio = log_ratio.exp()

        max_ratio = ratio.max().item()
        min_ratio = ratio.min().item()
        mean_ratio = ratio.mean().item()

        telemetry.log(f"ppo/ratio_{head}_max", max_ratio)
        telemetry.log(f"ppo/ratio_{head}_min", min_ratio)
        telemetry.log(f"ppo/ratio_{head}_mean", mean_ratio)

        if max_ratio > RATIO_CRIT_THRESHOLD:
            log.error(
                f"CRITICAL: {head} ratio exploded to {max_ratio:.1f} "
                f"(mean={mean_ratio:.2f}, min={min_ratio:.3f})"
            )
        elif max_ratio > RATIO_WARN_THRESHOLD:
            log.warning(
                f"WARNING: {head} ratio elevated at {max_ratio:.1f} "
                f"(mean={mean_ratio:.2f})"
            )

        # Also check for ratio collapse (policy changed too little)
        if max_ratio < 1.01 and min_ratio > 0.99:
            log.info(f"INFO: {head} ratio near 1.0 — policy unchanged this update")
```

**Implementation note:** Log per-head ratios separately, not just the joint. When debugging training instability, check these metrics first.

**Joint Ratio Clipping Test (DRL Review Addition):**

With 8 heads, individual per-head ratios can look healthy while the joint ratio (product) exceeds PPO's clip range. This test verifies joint ratio behavior:

```python
def test_multi_head_joint_ratio_clipping():
    """Verify joint ratio (product of per-head ratios) respects PPO clip range.

    BUG PATTERN: Each head at ratio=1.15 (within typical [0.8, 1.2] clip range)
    produces joint ratio = 1.15^8 ≈ 3.06 (well outside clip range).

    This test ensures:
    1. Joint ratio is computed correctly as product of per-head ratios
    2. Clipping behavior is well-defined for multi-head policies
    3. No silent gradient issues from out-of-range joint ratios
    """
    import torch
    from esper.simic.agent.ppo import compute_ppo_loss

    NUM_HEADS = 8
    CLIP_EPSILON = 0.2  # Standard PPO clip range [0.8, 1.2]

    # Scenario 1: All heads have moderate ratio (individual OK, joint may exceed)
    per_head_log_ratios = {
        f"head_{i}": torch.log(torch.tensor([1.15])) for i in range(NUM_HEADS)
    }

    # Compute joint ratio
    joint_log_ratio = sum(per_head_log_ratios.values())
    joint_ratio = joint_log_ratio.exp().item()

    print(f"Per-head ratio: 1.15 each")
    print(f"Joint ratio: {joint_ratio:.2f} = 1.15^{NUM_HEADS}")
    print(f"Clip range: [{1-CLIP_EPSILON:.1f}, {1+CLIP_EPSILON:.1f}]")

    # Joint ratio is outside clip range
    assert joint_ratio > 1 + CLIP_EPSILON, (
        f"Test setup error: joint ratio {joint_ratio:.2f} should exceed clip upper bound"
    )

    # Verify PPO loss handles this correctly (clips, doesn't NaN/explode)
    advantages = torch.tensor([1.0])  # Positive advantage
    old_log_probs = {f"head_{i}": torch.tensor([0.0]) for i in range(NUM_HEADS)}
    new_log_probs = per_head_log_ratios

    # This should clip the ratio, not explode
    loss = compute_ppo_loss(
        old_log_probs=old_log_probs,
        new_log_probs=new_log_probs,
        advantages=advantages,
        clip_epsilon=CLIP_EPSILON,
    )

    assert torch.isfinite(loss), f"PPO loss is non-finite: {loss}"
    assert loss.abs() < 100, f"PPO loss seems too large: {loss} (possible clipping failure)"

    print(f"✓ Joint ratio clipping works correctly (loss={loss.item():.4f})")


def test_joint_ratio_per_head_decomposition():
    """Verify joint ratio equals product of per-head ratios.

    This is a sanity check that the multi-head log-prob summation
    correctly corresponds to the product of individual ratios.
    """
    import torch

    # Random per-head log probs
    torch.manual_seed(42)
    old_log_probs = {f"head_{i}": torch.randn(32) for i in range(8)}
    new_log_probs = {f"head_{i}": torch.randn(32) for i in range(8)}

    # Compute joint ratio via log-space sum (numerically stable)
    joint_log_ratio = sum(
        new_log_probs[h] - old_log_probs[h] for h in old_log_probs.keys()
    )
    joint_ratio_from_sum = joint_log_ratio.exp()

    # Compute joint ratio via direct product (less stable, for verification)
    per_head_ratios = [
        (new_log_probs[h] - old_log_probs[h]).exp() for h in old_log_probs.keys()
    ]
    joint_ratio_from_prod = torch.stack(per_head_ratios).prod(dim=0)

    # Should be equal (within floating point tolerance)
    assert torch.allclose(joint_ratio_from_sum, joint_ratio_from_prod, rtol=1e-4), (
        f"Joint ratio mismatch: sum-based={joint_ratio_from_sum.mean():.4f}, "
        f"prod-based={joint_ratio_from_prod.mean():.4f}"
    )

    print("✓ Joint ratio decomposition verified")
```

#### 5.5 Q(s,op) Variance Monitoring

**Risk:** The op-conditioned critic `Q(s, op)` may ignore the op input entirely, effectively collapsing to `V(s)`. This defeats the purpose of op-conditioning. Symptom: `Q(s, op=GERMINATE) ≈ Q(s, op=PRUNE) ≈ Q(s, op=WAIT)` for all states.

**Validation:**

```python
def monitor_critic_op_sensitivity(policy, states: torch.Tensor, telemetry):
    """Verify critic actually uses op conditioning—low variance indicates it's ignoring op."""
    NUM_OPS = 6

    with torch.no_grad():
        # Compute Q(s, op) for all ops
        q_values = []
        for op_idx in range(NUM_OPS):
            op_one_hot = F.one_hot(
                torch.full((states.size(0),), op_idx, dtype=torch.long),
                num_classes=NUM_OPS
            ).float()
            value_input = torch.cat([states, op_one_hot], dim=-1)
            q = policy.value_head(value_input)
            q_values.append(q)

        # q_values is a list of [batch, 1] tensors (one per op)
        # torch.cat along dim=-1 produces [batch, num_ops]
        q_matrix = torch.cat(q_values, dim=-1)

        # Per-state variance across ops
        per_state_var = q_matrix.var(dim=-1)  # [batch]
        mean_var = per_state_var.mean().item()

        # Per-op mean across states
        per_op_mean = q_matrix.mean(dim=0)  # [num_ops]
        op_spread = (per_op_mean.max() - per_op_mean.min()).item()

        telemetry.log("critic/q_variance_across_ops", mean_var)
        telemetry.log("critic/q_op_spread", op_spread)

        # Warning thresholds (tune empirically)
        if mean_var < 0.01:
            log.warning(
                f"Critic shows low Q-variance across ops: {mean_var:.4f}. "
                f"Op conditioning may be ineffective."
            )
        if op_spread < 0.1:
            log.warning(
                f"Critic Q-values nearly identical across ops (spread={op_spread:.3f}). "
                f"Consider increasing op embedding influence or critic capacity."
            )

        return {
            "q_variance_across_ops": mean_var,
            "q_op_spread": op_spread,
            "q_per_op_mean": per_op_mean.tolist(),
        }
```

**Root cause when this fails:**
1. Op one-hot signal is too weak relative to state features (512 dims vs 6 dims)
2. Critic has learned that op doesn't matter for value prediction (possible if rewards are op-agnostic)
3. Op one-hot is being zeroed somewhere in the forward pass

**Remediation options:**
1. Add a learnable op embedding instead of raw one-hot
2. Use a separate "op encoder" MLP before concatenation
3. Check reward structure—if same reward for all ops, critic correctly ignores op

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
