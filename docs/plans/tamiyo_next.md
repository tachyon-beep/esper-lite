# Tamiyo Next: Obs V3 + Policy V2 Implementation Guide

**Status:** Ready for Implementation
**Date:** 2025-12-30
**Prerequisites:** This document consolidates the approved designs from:

- `2025-12-30-obs-v3-design.md` — Observation space overhaul
- `2025-12-30-policy-v2-design.md` — Architecture and training enhancements

YOU MUST READ BOTH THIS DOCUMENT AND THE RELEVANT PRERESQUISITE DESIGNS IN FULL BEFORE IMPLEMENTING. DO IT NOW. NO EXCEPTIONS.

## Summary

| Aspect | Before | After |
|--------|--------|-------|
| Observation dims | 218 | 124 |
| Model params | ~227K | ~930K |
| LSTM hidden | 128 | 256 |
| Feature dim | 128 | 256 |
| Decision horizon | ~25 epochs | 50+ epochs |
| Value aliasing | Present | Solved |

## Before You Start

### 1. Understand the Breaking Changes

**Checkpoint Incompatibility:** V1 checkpoints will NOT load into V2 networks due to:

- Different observation dimensions (218 → 124)
- Different LSTM hidden size (128 → 256)
- Different head input dimensions
- New value head architecture (op-conditioned)

**Action:** Plan to train from scratch. No migration path exists.

### 2. Verify Current Test Suite Passes

```bash
PYTHONPATH=src uv run pytest tests/tamiyo/ -v
PYTHONPATH=src uv run pytest tests/simic/ -v
```

All tests should pass before starting. Any failures are pre-existing issues.

### 3. Create a Development Branch

```bash
git checkout -b tamiyo-v2 quality-sprint
```

### 4. Understand the Version Toggle Pattern

Both Obs V3 and Policy V2 use a version toggle for safe rollback:

```python
# In features.py
_OBS_VERSION = "v3"  # Toggle between "v2" and "v3"

# In leyline/__init__.py
_POLICY_VERSION = "v2"  # Toggle between "v1" and "v2"
```

Keep V1/V2 code paths until V3/V2 is validated. Delete after validation.

### 5. Key Files to Read First

| File | Purpose |
|------|---------|
| `src/esper/tamiyo/policy/features.py` | Current feature extraction (V2) |
| `src/esper/tamiyo/networks/factored_lstm.py` | Current network architecture |
| `src/esper/leyline/__init__.py` | Default dimensions and constants |
| `src/esper/simic/agent/ppo.py` | PPO training loop |
| `src/esper/simic/training/vectorized.py` | Rollout collection |

---

## Implementation Order

### Phase 1: Leyline Constants (Foundation)

**Goal:** Update shared constants that everything depends on.

**Files:**

- `src/esper/leyline/__init__.py`

**Changes:**

```python
# Add version toggle
_POLICY_VERSION = "v2"

# Update defaults (conditional on version)
DEFAULT_LSTM_HIDDEN_DIM = 256 if _POLICY_VERSION == "v2" else 128
DEFAULT_FEATURE_DIM = 256 if _POLICY_VERSION == "v2" else 128

# Add new constants for action feedback
NUM_OPS = 6  # For one-hot encoding
```

**Validation:**

```bash
python -c "from esper.leyline import DEFAULT_LSTM_HIDDEN_DIM; print(DEFAULT_LSTM_HIDDEN_DIM)"
# Should print: 256
```

---

### Phase 2: Obs V3 Feature Extraction

**Goal:** Implement new observation structure without network changes.

**Files:**

- `src/esper/tamiyo/policy/features.py`

**Sub-phases:**

#### 2a. Add Version Toggle and Imports

```python
_OBS_VERSION = "v3"

# Add for blueprint embedding support
from esper.leyline import NUM_OPS
```

#### 2b. Implement Base Feature Extraction V3

Add `_extract_base_features_v3()` with:

- Log-scale loss normalization: `log(1 + loss) / log(11)`
- Compressed history: trend + volatility (4 dims instead of 10)
- Stage distribution: `num_training_norm`, `num_blending_norm`, `num_holding_norm`
- Host stabilized flag
- **Action feedback:** `last_action_success`, `last_action_op` (7 dims)

**Key formula changes:**

```python
# Loss normalization (was: clamp/10)
loss_norm = math.log(1 + val_loss) / math.log(11)

# History compression (was: 5 raw values each)
loss_trend = sum(loss_delta_history[-5:]) / max(len(loss_delta_history[-5:]), 1)
loss_volatility = math.log(1 + statistics.stdev(loss_history[-5:])) / math.log(3) if len(loss_history) > 1 else 0.0
```

#### 2c. Implement Slot Feature Extraction V3

Add `_extract_slot_features_v3()` with:

- Merged telemetry fields (gradient_norm, gradient_health, has_vanishing, has_exploding)
- `epochs_in_stage_norm` (was missing normalization)
- `counterfactual_fresh` decay signal
- **No blueprint one-hot** (moved to embedding)

#### 2d. Implement Vectorized Construction

Add pre-extraction pattern:

```python
def _extract_slot_arrays(batch_reports, slot_config) -> dict[str, np.ndarray]:
    """Pre-extract slot data into NumPy arrays for vectorized processing."""
    # Single array assignment per slot instead of 25+ individual writes
```

Add cached one-hot tables:

```python
_STAGE_ONE_HOT_TABLE = torch.eye(10, dtype=torch.float32)

def _vectorized_one_hot(indices: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """Lookup-based one-hot encoding."""
```

#### 2e. Implement batch_obs_to_features_v3()

New function returning `(obs, blueprint_indices)` tuple:

```python
def batch_obs_to_features_v3(
    batch_signals: list[TrainingSignals],
    batch_slot_reports: list[dict[str, SeedStateReport]],
    slot_config: SlotConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        obs: [batch, 112] - base + slot features (no blueprint)
        blueprint_indices: [batch, num_slots] - for embedding lookup
    """
```

#### 2f. Update get_feature_size()

```python
def get_feature_size_v3(slot_config: SlotConfig) -> int:
    """V3 feature size: 25 base + 29*slots = 112 for 3 slots.

    Note: Blueprint embeddings (4*slots=12) added by network, not here.
    Total input to network: 112 + 12 = 124.
    """
    return 25 + (29 * slot_config.num_slots)
```

**Validation:**

```bash
PYTHONPATH=src python -c "
from esper.tamiyo.policy.features import get_feature_size_v3
from esper.leyline import SlotConfig
config = SlotConfig.default()
print(f'Feature size: {get_feature_size_v3(config)}')  # Should be 112
print(f'With embeddings: {get_feature_size_v3(config) + 4 * config.num_slots}')  # Should be 124
"
```

---

### Phase 3: Blueprint Embedding Module

**Goal:** Create the learned embedding for blueprints.

**Files:**

- `src/esper/tamiyo/networks/factored_lstm.py` (or new file)

**Implementation:**

```python
class BlueprintEmbedding(nn.Module):
    """Learned blueprint embeddings for Obs V3."""

    def __init__(self, num_blueprints: int = 13, embed_dim: int = 4):
        super().__init__()
        # Index 13 = null embedding for inactive slots
        self.embedding = nn.Embedding(num_blueprints + 1, embed_dim)
        self.null_idx = num_blueprints

        # Small initialization per DRL expert recommendation
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, blueprint_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            blueprint_indices: [batch, num_slots] with -1 for inactive
        Returns:
            [batch, num_slots, embed_dim]
        """
        safe_idx = blueprint_indices.clone()
        safe_idx[blueprint_indices < 0] = self.null_idx
        return self.embedding(safe_idx)
```

**Validation:**

```bash
PYTHONPATH=src python -c "
import torch
from esper.tamiyo.networks.factored_lstm import BlueprintEmbedding
emb = BlueprintEmbedding()
idx = torch.tensor([[0, 2, -1], [5, -1, 3]])  # batch=2, slots=3
out = emb(idx)
print(f'Output shape: {out.shape}')  # Should be [2, 3, 4]
"
```

---

### Phase 4: Policy Network V2

**Goal:** Implement the new network architecture.

**Files:**

- `src/esper/tamiyo/networks/factored_lstm.py`

**Sub-phases:**

#### 4a. Create FactoredRecurrentActorCriticV2 Class

New class (don't modify V1 yet) with:

- `feature_dim = 256`
- `lstm_hidden_dim = 256`
- `BlueprintEmbedding` module
- 3-layer blueprint head
- Op-conditioned value head

#### 4b. Implement 3-Layer Blueprint Head

```python
self.blueprint_head = nn.Sequential(
    nn.Linear(lstm_hidden_dim, lstm_hidden_dim),  # 256 → 256
    nn.ReLU(),
    nn.Linear(lstm_hidden_dim, head_hidden),      # 256 → 128
    nn.ReLU(),
    nn.Linear(head_hidden, num_blueprints),       # 128 → 13
)
```

**Initialization:** Apply `gain=0.01` only to final layer, `sqrt(2)` to intermediate.

#### 4c. Implement Op-Conditioned Value Head

```python
self.value_head = nn.Sequential(
    nn.Linear(lstm_hidden_dim + NUM_OPS, head_hidden),  # 256+6 → 128
    nn.ReLU(),
    nn.Linear(head_hidden, 1),
)
```

#### 4d. Update forward() Signature

```python
def forward(
    self,
    state: torch.Tensor,           # [batch, seq, 112] - features without blueprint
    blueprint_indices: torch.Tensor,  # [batch, seq, num_slots] - for embedding
    hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    op_for_value: torch.Tensor | None = None,  # For hard conditioning during update
    # ... masks ...
) -> ForwardOutput:
```

#### 4e. Implement Op-Conditioning Logic (Q(s,op) Pattern)

```python
# Compute op logits and sample action
op_logits = self.op_head(lstm_out)
op_dist = Categorical(logits=op_logits)
sampled_op = op_dist.sample()

# Op-conditioned critic: always use hard one-hot of sampled op
# This treats the critic as Q(s, op) — consistent in both rollout and update
op_one_hot = F.one_hot(sampled_op, num_classes=NUM_OPS).float()
value_input = torch.cat([lstm_out, op_one_hot], dim=-1)
value = self.value_head(value_input)
```

**Design rationale:** We use hard one-hot conditioning in **both** rollout and PPO update:
- Rollout: `Q(s, sampled_op)` — value stored matches the op actually taken
- PPO update: same `Q(s, sampled_op)` — no GAE mismatch

This avoids the "soft probs during rollout, hard one-hot during update" pattern which creates advantage estimation inconsistencies.

**Validation:**

```bash
PYTHONPATH=src python -c "
import torch
from esper.tamiyo.networks.factored_lstm import FactoredRecurrentActorCriticV2
net = FactoredRecurrentActorCriticV2(state_dim=112, num_slots=3)
state = torch.randn(2, 5, 112)  # batch=2, seq=5
bp_idx = torch.randint(0, 13, (2, 5, 3))
out = net(state, bp_idx)
print(f'Op logits: {out[\"op_logits\"].shape}')  # [2, 5, 6]
print(f'Value: {out[\"value\"].shape}')  # [2, 5]
print(f'Params: {sum(p.numel() for p in net.parameters())}')  # ~930K
"
```

---

### Phase 5: Training Configuration Updates

**Goal:** Update PPO for differential entropy and new forward signature.

**Files:**

- `src/esper/simic/agent/ppo.py`

#### 5a. Add Differential Entropy Coefficients

```python
# In PPOConfig or __init__
ENTROPY_COEF_PER_HEAD = {
    "op": 1.0,
    "slot": 1.0,
    "blueprint": 1.5,
    "style": 1.2,
    "tempo": 1.5,
    "alpha_target": 1.2,
    "alpha_speed": 1.3,
    "alpha_curve": 1.3,
}
```

#### 5b. Update Entropy Loss Computation

```python
# In PPO update
for head, entropy in per_head_entropy.items():
    coef = ENTROPY_COEF_PER_HEAD.get(head, 1.0)
    total_entropy_loss -= base_entropy_coef * coef * entropy.mean()
```

#### 5c. Update evaluate_actions() Call

Pass the sampled op for value conditioning (same as rollout for consistency):

```python
result = self.policy.evaluate_actions(
    states,
    blueprint_indices,  # New
    actions,
    masks,
    hidden,
    sampled_op=actions["op"],  # Same hard conditioning as rollout
)
```

**Note:** The `sampled_op` is the same action that was taken during rollout. This ensures the Q(s,op) value trained matches exactly what was stored in the buffer.

---

### Phase 6: Vectorized Training Integration

**Goal:** Wire up the new feature extraction and network in the training loop.

**Files:**

- `src/esper/simic/training/vectorized.py`

#### 6a. Track Action Feedback State

Add to environment state:

```python
@dataclass
class EnvState:
    # ... existing fields ...
    last_action_success: bool = True
    last_action_op: int = 0  # LifecycleOp.WAIT
```

Update after each action execution.

#### 6b. Update Feature Extraction Call

```python
# Was:
obs = batch_obs_to_features(signals, reports, use_telemetry=True, ...)

# Now:
obs, blueprint_indices = batch_obs_to_features_v3(signals, reports, slot_config, device)
```

#### 6c. Update Policy Forward Calls

Pass `blueprint_indices` to all `policy.forward()` and `policy.get_action()` calls.

#### 6d. Store Blueprint Indices in Rollout Buffer

Add `blueprint_indices` to the rollout buffer alongside states.

---

### Phase 7: Validation & Testing

**Goal:** Verify the implementation works end-to-end.

#### 7a. Unit Tests

```bash
# Feature extraction
PYTHONPATH=src uv run pytest tests/tamiyo/policy/test_features.py -v

# Network architecture
PYTHONPATH=src uv run pytest tests/tamiyo/networks/test_factored_lstm.py -v

# PPO updates
PYTHONPATH=src uv run pytest tests/simic/agent/test_ppo.py -v
```

#### 7b. Smoke Test Training

```bash
# Short training run to verify no crashes
PYTHONPATH=src uv run python -m esper.scripts.train ppo \
    --episodes 5 \
    --max-epochs 10 \
    --seed 42
```

#### 7c. Verify Observation Ranges

Add temporary logging to verify all features are in expected ranges:

```python
# In batch_obs_to_features_v3
assert (obs >= -2).all() and (obs <= 2).all(), f"Obs out of range: min={obs.min()}, max={obs.max()}"
```

#### 7d. Profile Performance

```bash
# Compare V2 vs V3 feature extraction speed
PYTHONPATH=src python -m esper.scripts.profile_features --version v3
```

#### 7e. Compare Learning Curves

Run identical experiments with V2 (old) and V3 (new):

```bash
# V2 baseline
PYTHONPATH=src uv run python -m esper.scripts.train ppo --obs-version v2 --policy-version v1 --seed 42

# V3 + Policy V2
PYTHONPATH=src uv run python -m esper.scripts.train ppo --obs-version v3 --policy-version v2 --seed 42
```

---

### Phase 8: Cleanup (After Validation)

**Goal:** Remove V1/V2 code paths once V3 is proven.

#### 8a. Delete Old Code

Search for and remove:

```bash
git grep "TODO(obs-v3)"
git grep "TODO(policy-v2)"
git grep "_OBS_VERSION"
git grep "_POLICY_VERSION"
```

#### 8b. Remove Version Toggles

Make V3/V2 the only path. Delete:

- `batch_obs_to_features_v2()`
- `FactoredRecurrentActorCriticV1` (if separate class)
- `_OBS_VERSION` and `_POLICY_VERSION` constants

#### 8c. Update Tests

Remove any V1/V2 specific test cases.

---

## Quick Reference: File → Changes

| File | Phase | Key Changes |
|------|-------|-------------|
| `leyline/__init__.py` | 1 | Dimension constants, version toggle |
| `tamiyo/policy/features.py` | 2 | V3 extraction, vectorization, log normalization |
| `tamiyo/networks/factored_lstm.py` | 3, 4 | BlueprintEmbedding, V2 network, op-conditioned critic |
| `simic/agent/ppo.py` | 5 | Differential entropy, evaluate_actions update |
| `simic/training/vectorized.py` | 6 | Action feedback tracking, V3 integration |

---

## Gotchas & Warnings

### 1. Blueprint Index dtype

Must be `torch.int64` for `nn.Embedding`. NumPy extraction should use `np.int64`.

### 2. Op-Conditioning During Bootstrap

At episode end, use sampled op from final step for value bootstrap, NOT expected value.

### 3. Blueprint Head Initialization

Apply `gain=0.01` ONLY to the final Linear layer. Intermediate layers use `sqrt(2)`.

### 4. Action Feedback Initial State

First step of episode has no "previous action". Initialize:

- `last_action_success = True`
- `last_action_op = LifecycleOp.WAIT.value`

### 5. Telemetry Toggle Removal

The `use_telemetry` flag no longer exists in V3. Remove from CLI args and config.

### 6. Feature Size Mismatch

V3 feature extraction returns 112 dims. Network expects 124 (112 + 12 from embeddings).
The embedding concatenation happens INSIDE the network, not in feature extraction.

### 7. Op-Conditioning Consistency (Q(s,op) Pattern)

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

### 8. Enum Normalization Robustness

Use explicit index mappings for enum features, not `.value / max_value`:

```python
# FRAGILE - breaks if enum values change
alpha_mode_norm = mode.value / max_mode

# ROBUST - explicit mapping
_MODE_INDEX = {AlphaMode.HOLD: 0, AlphaMode.RAMP: 1, AlphaMode.STEP: 2}
alpha_mode_norm = _MODE_INDEX[mode] / (len(_MODE_INDEX) - 1)
```

### 9. Inactive Slot Stage Encoding

Inactive slots (`is_active=0`) must have **all-zeros** for stage one-hot, NOT stage 0. The vectorized one-hot helper must mask by validity:

```python
def _vectorized_one_hot(indices, table):
    valid_mask = indices >= 0  # -1 for inactive
    safe_indices = indices.clamp(min=0)
    one_hot = table[safe_indices]
    return one_hot * valid_mask.unsqueeze(-1).float()  # Zeros out inactive
```

---

## Success Criteria

1. ✅ All unit tests pass
2. ✅ Smoke test training completes without crashes
3. ✅ V3 learning curves match or exceed V2
4. ✅ Feature extraction is 5x+ faster than V2
5. ✅ Explained variance improves (less negative)
6. ✅ Multi-generation scaffolding credit assignment works
7. ✅ Per-head entropy stable for sparse heads (blueprint, tempo)
