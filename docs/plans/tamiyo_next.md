# Tamiyo Next: Obs V3 + Policy V2 Implementation Guide

**Status:** Ready for Implementation
**Date:** 2025-12-30
**Prerequisites:** This document consolidates the approved designs from:

- `2025-12-30-obs-v3-design.md` — Observation space overhaul
- `2025-12-30-policy-v2-design.md` — Architecture and training enhancements

YOU MUST READ BOTH THIS DOCUMENT AND THE RELEVANT PRERESQUISITE DESIGNS IN FULL BEFORE IMPLEMENTING. DO IT NOW. NO EXCEPTIONS.

---

## Risk Assessment

| Phase | Risk | Complexity | Key Hazards |
|-------|------|------------|-------------|
| 1: Leyline Constants | Low | Low | Most constants already exist; mainly dimension updates |
| 2: Obs V3 Extraction | **High** | **High** | Feature spec correctness, enum cardinalities, inactive-slot masking, CPU↔GPU copies |
| 3: Blueprint Embedding | Med | Med | dtype/int64, -1 handling, shape flattening |
| 4: Policy V2 + Q(s,op) | **High** | **High** | Critic semantics become SARSA-like Q(s,op); bootstrap + GAE consistency |
| 5: PPO Entropy | Med | Med-High | Stability tuning, per-head causal masking/entropy |
| 6: Vectorized Integration | **Very High** | **Very High** | Rollout buffer schema changes, (obs, blueprint_indices) tuple through hot path |
| 7: Validation | Med | Risk-reducing | Verifies correctness; surface area for catching bugs |
| 8: Cleanup | Med | Med | Risk of leaving stale code; follow CLAUDE.md no-legacy policy |

**Top Technical Risks:**
1. **Op-conditioned value head:** Rollout value, truncation bootstrap_value, and PPO update must all condition on the same op
2. **Feature vectorization/perf:** Cached one-hot tables with `.to(device)` can accidentally allocate per-step
3. **Enum/cardinality drift:** Hard-coding `torch.eye(10)` for stages assumes fixed cardinality—use leyline constants
4. **API propagation:** `(obs, blueprint_indices)` touches network forward, rollout collection, buffer storage, PPO update

## Summary

| Aspect | Before | After |
|--------|--------|-------|
| Observation dims | 218 | 133 |
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

### 4. Clean Replacement Strategy (No Dual Paths)

Per `CLAUDE.md` no-legacy policy, we do **clean replacement**, NOT dual-path version toggles:

- **Delete** old Tamiyo network/feature code as you implement V2
- **No** `_OBS_VERSION` or `_POLICY_VERSION` toggles
- **No** backwards compatibility shims
- Rollback via git branch, not code toggles

**The old Tamiyo doesn't work.** That's why we're here. There's no value in keeping it.

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

**What already exists (don't re-add):**

- `NUM_OPS = 6` ✓ (in `leyline/factored_actions.py`, re-exported)
- `NUM_STAGES = 10` ✓ (in `leyline/stage_schema.py`, re-exported)
- `NUM_BLUEPRINTS = 13`, `NUM_STYLES`, `NUM_TEMPO`, etc. ✓

**Changes:**

```python
# ⚠️ WARNING: Changing dimension constants IMMEDIATELY breaks checkpoint compatibility.
# ALL V1 checkpoints (LSTM_HIDDEN=128) will fail to load. Train from scratch.

# Update existing dimension constants (clean replacement)
DEFAULT_LSTM_HIDDEN_DIM = 256  # Was 128 - supports 50+ epoch decision horizon
DEFAULT_FEATURE_DIM = 256      # Was 128 - matches LSTM dim, prevents bottleneck

# New constants for blueprint embedding
BLUEPRINT_NULL_INDEX: int = 13  # Index for inactive slot embedding (== NUM_BLUEPRINTS)
DEFAULT_BLUEPRINT_EMBED_DIM: int = 4  # Learned blueprint representation dimension
```

**Rationale (from specialist reviews):**
- **256 hidden dim:** For LSTM credit assignment, hidden dim scales with `O(horizon × info_per_step)`. With 3 slots needing ~10-20 bits each, 256 provides ~85 dims/slot.
- **256 is power-of-2:** Optimal for GPU tensor cores (divisible by 8 for AMP).
- **Feature dim = LSTM dim:** Prevents information bottleneck before LSTM.
- **BLUEPRINT_NULL_INDEX:** Prevents magic number 13 scattered across features.py and factored_lstm.py.

**Validation:**

```bash
PYTHONPATH=src python -c "
from esper.leyline import (
    DEFAULT_LSTM_HIDDEN_DIM, DEFAULT_FEATURE_DIM,
    NUM_OPS, NUM_STAGES, NUM_BLUEPRINTS
)
print(f'LSTM={DEFAULT_LSTM_HIDDEN_DIM}, FEATURE={DEFAULT_FEATURE_DIM}')
print(f'OPS={NUM_OPS}, STAGES={NUM_STAGES}, BLUEPRINTS={NUM_BLUEPRINTS}')
"
# Should print:
# LSTM=256, FEATURE=256
# OPS=6, STAGES=10, BLUEPRINTS=13
```

---

### Phase 2: Obs V3 Feature Extraction

**Goal:** Implement new observation structure without network changes.

**Files:**

- `src/esper/tamiyo/policy/features.py`

**Sub-phases:**

#### 2a. Replace Imports (Clean Replacement)

Delete old V2 feature functions as you implement V3. No version toggle.

```python
# In features.py - ensure these imports exist
from esper.leyline import NUM_OPS, NUM_STAGES
```

#### 2b. Implement Base Feature Extraction V3

Add `_extract_base_features_v3()` with:

- Log-scale loss normalization: `log(1 + loss) / log(11)`
- **Raw history (10 dims):** 5 loss + 5 accuracy values, log-normalized and left-padded
- Stage distribution: `num_training_norm`, `num_blending_norm`, `num_holding_norm`
- Host stabilized flag
- **Action feedback:** `last_action_success`, `last_action_op` (7 dims)

**Base features total: 24 dims** (was 18 in V2)

**Key formula changes:**

```python
# Loss normalization (was: clamp/10)
loss_norm = math.log(1 + val_loss) / math.log(11)

# Raw history with padding (10 dims total)
# Left-pad short histories with zeros to ensure consistent length
def _pad_history(history: list[float], length: int = 5) -> list[float]:
    """Left-pad history to fixed length."""
    if len(history) >= length:
        return history[-length:]
    return [0.0] * (length - len(history)) + history

loss_history_norm = [math.log(1 + x) / math.log(11) for x in _pad_history(loss_history, 5)]
acc_history_norm = [x / 100.0 for x in _pad_history(acc_history, 5)]
```

**Why raw history (from DRL review):** Trend+volatility compression loses temporal causality. The LSTM needs the raw sequence shape for credit assignment—which epoch's loss delta matters for which decision? 6 dims of "savings" isn't worth degraded policy learning.

#### 2c. Implement Slot Feature Extraction V3

Add `_extract_slot_features_v3()` with:

- Merged telemetry fields (gradient_norm, gradient_health, has_vanishing, has_exploding)
- **`gradient_health_prev`:** Previous epoch's gradient_health (enables LSTM trend detection)
- `epochs_in_stage_norm` (was missing normalization)
- **`counterfactual_fresh`:** `DEFAULT_GAMMA ** epochs_since_cf` (gamma-matched decay)
- **No blueprint one-hot** (moved to embedding)

**Per-slot features total: 30 dims** (was 29 in V2)

```python
from esper.leyline import DEFAULT_GAMMA

# Counterfactual freshness (gamma-matched decay)
# With DEFAULT_GAMMA=0.995, signal stays >0.5 for ~50 epochs
counterfactual_fresh = DEFAULT_GAMMA ** epochs_since_counterfactual

# Gradient trend signal
gradient_health_prev = previous_epoch_gradient_health  # Track in EnvState
```

**Why gamma-matched decay (from DRL review):** The old 0.8^epochs decayed too fast—0.8^10 = 0.1, making counterfactual estimates unreliable after just 10 epochs. Using DEFAULT_GAMMA (0.995) aligns with PPO's credit horizon: 0.995^10 ≈ 0.95, staying useful for ~50 epochs.

#### 2d. Implement Vectorized Construction

Add pre-extraction pattern:

```python
def _extract_slot_arrays(batch_reports, slot_config) -> dict[str, np.ndarray]:
    """Pre-extract slot data into NumPy arrays for vectorized processing."""
    # Single array assignment per slot instead of 25+ individual writes
```

Add cached one-hot tables with **device-keyed cache**:

```python
from esper.leyline import NUM_STAGES

# Module-level constants
_STAGE_ONE_HOT_TABLE = torch.eye(NUM_STAGES, dtype=torch.float32)

# Device-keyed cache to avoid per-step .to(device) allocations
_DEVICE_CACHE: dict[torch.device, torch.Tensor] = {}

def _get_cached_table(device: torch.device) -> torch.Tensor:
    """Get stage one-hot table for device, caching to avoid repeated transfers."""
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

**Why device-keyed cache (from PyTorch review):** The naive `.to(indices.device)` allocates a new tensor every step. With ~500 steps/episode across 4 environments, that's 2000 allocations/episode. The cache ensures one-time transfer per device.

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
        obs: [batch, 121] - base (31) + slot features (30*3) (no blueprint)
        blueprint_indices: [batch, num_slots] - for embedding lookup, dtype=int64
    """
```

#### 2f. Update get_feature_size()

Replace existing function (no `_v3` suffix—this is the new implementation):

```python
def get_feature_size(slot_config: SlotConfig) -> int:
    """Feature size: 31 base (24 + 7 action feedback) + 30*slots = 121 for 3 slots.

    Note: Blueprint embeddings (4*slots=12) added by network, not here.
    Total input to network: 121 + 12 = 133.
    """
    return 31 + (30 * slot_config.num_slots)
```

**Dimension breakdown:**
- Base features: 24 (includes 10-dim raw history)
- Action feedback: 7 (success + op one-hot)
- Per-slot: 30 (includes gradient_health_prev)
- Non-blueprint obs: 31 + 90 = 121
- With embeddings: 121 + 12 = 133

**Validation:**

```bash
PYTHONPATH=src python -c "
from esper.tamiyo.policy.features import get_feature_size
from esper.leyline.slot_config import SlotConfig
config = SlotConfig.default()
print(f'Feature size: {get_feature_size(config)}')  # Should be 121
print(f'With embeddings: {get_feature_size(config) + 4 * config.num_slots}')  # Should be 133
"
```

---

### Phase 3: Blueprint Embedding Module

**Goal:** Create the learned embedding for blueprints.

**Files:**

- `src/esper/tamiyo/networks/factored_lstm.py` (or new file)

**Implementation:**

```python
from esper.leyline import (
    NUM_BLUEPRINTS,
    BLUEPRINT_NULL_INDEX,
    DEFAULT_BLUEPRINT_EMBED_DIM,
)

class BlueprintEmbedding(nn.Module):
    """Learned blueprint embeddings for Obs V3."""

    def __init__(
        self,
        num_blueprints: int = NUM_BLUEPRINTS,
        embed_dim: int = DEFAULT_BLUEPRINT_EMBED_DIM,
    ):
        super().__init__()
        # Index 13 = null embedding for inactive slots (from leyline)
        self.embedding = nn.Embedding(num_blueprints + 1, embed_dim)
        self.null_idx = BLUEPRINT_NULL_INDEX

        # Small initialization per DRL expert recommendation
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, blueprint_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            blueprint_indices: [batch, num_slots] with -1 for inactive
        Returns:
            [batch, num_slots, embed_dim]
        """
        # Use torch.where to avoid .clone() allocation in hot path
        safe_idx = torch.where(
            blueprint_indices < 0,
            torch.tensor(self.null_idx, device=blueprint_indices.device, dtype=blueprint_indices.dtype),
            blueprint_indices
        )
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
    state: torch.Tensor,           # [batch, seq, 121] - features without blueprint
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
net = FactoredRecurrentActorCriticV2(state_dim=121, num_slots=3)
state = torch.randn(2, 5, 121)  # batch=2, seq=5 (121 non-blueprint dims)
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

**⚠️ RISK: VERY HIGH** — This phase threads new tuple obs + blueprint indices through the hottest, largest module.

**Files:**

- `src/esper/simic/training/vectorized.py`
- `src/esper/simic/agent/rollout_buffer.py` (schema change!)
- `src/esper/simic/agent/ppo.py` (buffer consumption)

#### 6a. Rollout Buffer Schema Changes

**Current state:** `TamiyoRolloutBuffer` stores `states: torch.Tensor` as a single `[num_envs, max_steps, state_dim]` tensor.

**Required change:** Add `blueprint_indices` storage:

```python
# In TamiyoRolloutBuffer.__post_init__()
self.blueprint_indices = torch.zeros(
    n, m, self.num_slots, dtype=torch.int64, device=device
)

# In add() method - new parameter
blueprint_indices: torch.Tensor,  # [num_slots], int64
# ...
self.blueprint_indices[env_id, step_idx] = blueprint_indices

# In get_batched_sequences() - add to return dict
"blueprint_indices": self.blueprint_indices.to(device, non_blocking=nb),
```

**Also update:** `TamiyoRolloutStep` NamedTuple to include `blueprint_indices` field.

#### 6b. Track Action Feedback State

Add to environment state:

```python
@dataclass
class EnvState:
    # ... existing fields ...
    last_action_success: bool = True
    last_action_op: int = 0  # LifecycleOp.WAIT
```

Update after each action execution.

#### 6c. Update Feature Extraction Call

```python
# Was:
obs = batch_obs_to_features(signals, reports, use_telemetry=True, ...)

# Now (clean replacement, not new function name):
obs, blueprint_indices = batch_obs_to_features(signals, reports, slot_config, device)
```

Note: The function signature changes from returning `Tensor` to returning `tuple[Tensor, Tensor]`.

#### 6d. Update Policy Forward Calls

Pass `blueprint_indices` to all `policy.forward()` and `policy.get_action()` calls:

```python
# Was:
action, value, log_probs, hidden = policy(state, hidden, masks)

# Now:
action, value, log_probs, hidden = policy(state, blueprint_indices, hidden, masks)
```

#### 6e. Update Buffer add() Calls

Thread blueprint_indices through the rollout collection loop:

```python
buffer.add(
    env_id=env_id,
    state=obs[env_id],
    blueprint_indices=blueprint_indices[env_id],  # NEW
    slot_action=action["slot"],
    # ... rest unchanged ...
)
```

#### 6f. Update PPO Consumption

In `ppo.py`, the `_ppo_update()` method must pass blueprint_indices from buffer to `evaluate_actions()`:

```python
batch = buffer.get_batched_sequences(device)
# ...
result = policy.evaluate_actions(
    batch["states"],
    batch["blueprint_indices"],  # NEW
    actions,
    masks,
    initial_hidden,
    sampled_op=batch["op_actions"],
)

---

### Phase 7: Validation & Testing

**Goal:** Verify the implementation works end-to-end.

#### 7a. Unit Tests

```bash
# Feature extraction (update test file for new API)
PYTHONPATH=src uv run pytest tests/tamiyo/policy/test_features.py -v

# Network architecture
PYTHONPATH=src uv run pytest tests/simic/test_tamiyo_network.py -v

# PPO updates
PYTHONPATH=src uv run pytest tests/simic/test_ppo.py -v

# Buffer schema
PYTHONPATH=src uv run pytest tests/simic/test_tamiyo_buffer.py -v
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

Add temporary assertion in feature extraction to verify all features are in expected ranges:

```python
# In batch_obs_to_features
assert (obs >= -2).all() and (obs <= 2).all(), f"Obs out of range: min={obs.min()}, max={obs.max()}"
```

Remove assertion after validation passes.

#### 7d. Profile Performance

Add timing to feature extraction (no existing profile script for features):

```python
# Ad-hoc profiling during development
import time
start = time.perf_counter()
for _ in range(100):
    obs, bp_idx = batch_obs_to_features(signals, reports, slot_config, device)
elapsed = (time.perf_counter() - start) / 100
print(f"Feature extraction: {elapsed*1000:.2f}ms/batch")
```

Target: < 1ms per batch for 4 environments.

#### 7e. Learning Curve Comparison

Since there's no version toggle, comparison is via git branches:

```bash
# Current branch has new implementation
PYTHONPATH=src uv run python -m esper.scripts.train ppo --episodes 100 --seed 42

# Compare metrics in TensorBoard/telemetry against pre-V2 runs
```

Note: Checkpoints are incompatible, so comparison is learning curve shape, not resumed training.

---

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
| `leyline/__init__.py` | 1 | Dimension constants (256), BLUEPRINT_NULL_INDEX, DEFAULT_BLUEPRINT_EMBED_DIM |
| `tamiyo/policy/features.py` | 2 | Clean replacement, raw history (10 dims), gradient_health_prev, device-keyed cache, gamma-matched counterfactual, returns `(obs[121], bp_idx)` |
| `tamiyo/networks/factored_lstm.py` | 3, 4 | BlueprintEmbedding (torch.where), V2 network, op-conditioned Q(s,op) critic |
| `simic/agent/ppo.py` | 5, 6f | Differential entropy, evaluate_actions with blueprint_indices |
| `simic/agent/rollout_buffer.py` | 6a | Add `blueprint_indices` tensor, update `add()` and `get_batched_sequences()` |
| `simic/training/vectorized.py` | 6 | Action feedback tracking, gradient_health_prev tracking in EnvState, new feature extraction API, blueprint_indices plumbing |

---

## Gotchas & Warnings

### 1. Blueprint Index dtype

Must be `torch.int64` for `nn.Embedding`. NumPy extraction should use `np.int64`.

### 2. Op-Conditioning During Bootstrap (P1 from DRL Review)

The **same sampled_op** must be used for ALL three value computations:
1. Value stored during rollout collection
2. Value computed during PPO update (for GAE)
3. Value bootstrap at episode truncation

```python
# At truncation, compute bootstrap value with SAME conditioning as rollout
bootstrap_value = value_head(cat(lstm_out, F.one_hot(final_sampled_op, NUM_OPS).float()))
```

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

### 10. Bootstrap Blueprint Indices (P1 from DRL Review)

At episode truncation, the bootstrap value computation needs **both** the final observation AND the final blueprint_indices. Ensure the rollout buffer stores blueprint_indices for the final observation:

```python
# In truncation handling (vectorized.py)
final_obs, final_blueprint_indices = batch_obs_to_features(final_signals, final_reports, ...)
bootstrap_value = policy.get_value(final_obs, final_blueprint_indices, hidden, final_sampled_op)
```

If blueprint_indices are missing at truncation, the bootstrap value will be computed incorrectly.

### 11. Gradient Health Prev Tracking

The `gradient_health_prev` feature requires tracking the previous epoch's gradient health in `EnvState`:

```python
@dataclass
class EnvState:
    # ... existing fields ...
    gradient_health_prev: dict[str, float] = field(default_factory=dict)  # slot_id → prev value
```

At each step:
1. Read `gradient_health_prev[slot_id]` for feature extraction
2. After step, update `gradient_health_prev[slot_id] = current_gradient_health`

First epoch of each slot should use `gradient_health_prev = 1.0` (assume healthy until proven otherwise).

### 12. History Padding at Episode Start

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
