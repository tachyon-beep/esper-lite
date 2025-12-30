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
| Model params | ~227K | ~2.1M |
| LSTM hidden | 128 | 512 |
| Feature dim | 128 | 512 |
| Episode length | 25 epochs | 150 epochs |
| Decision horizon | ~25 epochs | 150+ epochs (3-seed sequential) |
| Value aliasing | Present | Solved |

**Why 512 hidden dim (not 256):**

With 150-epoch sequential scaffolding, the LSTM must maintain "archival memories" of earlier seeds while processing current ones. At epoch 140:
- Seed A: "I planted a CONV_HEAVY at epoch 5. It's fossilized and providing base structure."
- Seed B: "I planted ATTENTION at epoch 55. It fossilized at epoch 105, interacting with Seed A."
- Seed C: "I'm currently tuning DEPTHWISE to synergize with A+B."

**256 risks "Catastrophic Overwrite"** — as the LSTM processes Seed C's noisy gradients, it might evict the memory of what Seed A actually is. 512 provides the scratchpad space to keep archival memories safe.

**PPO "Horizon Cut" Bridge:** The LSTM hidden state is the only thing connecting step 150 to step 1 across rollout boundaries. Larger hidden state = more robust long-term state representation for the value function.

## Before You Start

### 1. Understand the Breaking Changes

**Checkpoint Incompatibility:** V1 checkpoints will NOT load into V2 networks due to:

- Different observation dimensions (218 → 133)
- Different LSTM hidden size (128 → 512)
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

### Phase 1: Leyline Constants (Foundation) ✅ COMPLETE

**Status:** Already merged to `quality-sprint` branch.

**Goal:** Update shared constants that everything depends on.

**Files:**

- `src/esper/leyline/__init__.py`

**Already implemented (verify with validation command below):**

- `DEFAULT_LSTM_HIDDEN_DIM = 512` (line 103)
- `DEFAULT_FEATURE_DIM = 512` (line 418)
- `BLUEPRINT_NULL_INDEX = 13` (line 432)
- `DEFAULT_BLUEPRINT_EMBED_DIM = 4` (line 438)
- `NUM_BLUEPRINTS = 13` (line 426)
- `NUM_OPS = 6`, `NUM_STAGES = 10`, etc.

**Validation (run to confirm):**

```bash
PYTHONPATH=src python -c "
from esper.leyline import (
    DEFAULT_LSTM_HIDDEN_DIM, DEFAULT_FEATURE_DIM,
    DEFAULT_EPISODE_LENGTH, NUM_OPS, NUM_STAGES, NUM_BLUEPRINTS,
    BLUEPRINT_NULL_INDEX, DEFAULT_BLUEPRINT_EMBED_DIM
)
print(f'LSTM={DEFAULT_LSTM_HIDDEN_DIM}, FEATURE={DEFAULT_FEATURE_DIM}')
print(f'EPISODE_LENGTH={DEFAULT_EPISODE_LENGTH}')
print(f'OPS={NUM_OPS}, STAGES={NUM_STAGES}, BLUEPRINTS={NUM_BLUEPRINTS}')
print(f'NULL_INDEX={BLUEPRINT_NULL_INDEX}, EMBED_DIM={DEFAULT_BLUEPRINT_EMBED_DIM}')
"
# Should print:
# LSTM=512, FEATURE=512
# EPISODE_LENGTH=150
# OPS=6, STAGES=10, BLUEPRINTS=13
# NULL_INDEX=13, EMBED_DIM=4
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

#### 2a½. Add State Tracking Fields to ParallelEnvState

**File:** `src/esper/simic/training/parallel_env_state.py`

Before implementing feature extraction, add the fields that features will read from. The *update logic* for these fields is in Phase 6b, but the field definitions must exist now.

```python
from esper.leyline import LifecycleOp

@dataclass
class ParallelEnvState:
    # ... existing fields ...

    # === Obs V3 Action Feedback (7 dims: 1 success + 6 op one-hot) ===
    last_action_success: bool = True
    last_action_op: int = LifecycleOp.WAIT.value  # 0

    # === Obs V3 Gradient Health Tracking (for gradient_health_prev feature) ===
    # Maps slot_id -> previous epoch's gradient_health value
    gradient_health_prev: dict[str, float] = field(default_factory=dict)

    # === Obs V3 Counterfactual Freshness Tracking ===
    # Maps slot_id -> epochs since last counterfactual measurement
    epochs_since_counterfactual: dict[str, int] = field(default_factory=dict)
```

**Initial Values Contract:**

| Field | Initial Value | Rationale |
|-------|---------------|-----------|
| `last_action_success` | `True` | First step has no prior action to fail |
| `last_action_op` | `LifecycleOp.WAIT.value` (0) | Neutral "no action yet" |
| `gradient_health_prev[slot_id]` | `1.0` | Assume healthy for new slots |
| `epochs_since_counterfactual[slot_id]` | `0` | Fresh when slot germinates |

**Note:** Phase 6b covers *when and how* to update these fields. For now, just add the field definitions with their default values.

#### 2b. Implement Base Feature Extraction V3

Add `_extract_base_features_v3()` with:

- Log-scale loss normalization: `log(1 + loss) / log(11)`
- **Raw history (10 dims):** 5 loss + 5 accuracy values, log-normalized and left-padded (from `TrainingSignals.loss_history` and `TrainingSignals.accuracy_history`)
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

> **State Tracking:** The `gradient_health_prev` and `epochs_since_counterfactual` values
> require tracking in `ParallelEnvState`. See **Phase 6b** for the complete state tracking
> contract including field definitions, update timing, and initial values.

```python
from esper.leyline import DEFAULT_GAMMA

# Counterfactual freshness (gamma-matched decay)
# With DEFAULT_GAMMA=0.995, signal stays >0.5 for ~50 epochs
counterfactual_fresh = DEFAULT_GAMMA ** epochs_since_counterfactual

# Gradient trend signal
gradient_health_prev = previous_epoch_gradient_health  # Track in ParallelEnvState
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

def _normalize_device(device: torch.device) -> torch.device:
    """Normalize device to canonical form (cuda -> cuda:0).

    This prevents duplicate cache entries for the same physical device.
    torch.device("cuda") != torch.device("cuda:0") but they're the same GPU.
    """
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device

def _get_cached_table(device: torch.device) -> torch.Tensor:
    """Get stage one-hot table for device, caching to avoid repeated transfers."""
    device = _normalize_device(device)
    if device not in _DEVICE_CACHE:
        _DEVICE_CACHE[device] = _STAGE_ONE_HOT_TABLE.to(device)
    return _DEVICE_CACHE[device]

def _vectorized_one_hot(indices: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Vectorized one-hot encoding with device-cached lookup table.

    Args:
        indices: Stage indices of shape [..., num_slots] where -1 marks inactive.
                 Supports arbitrary leading batch dimensions.
        device: Target device for output tensor.

    Returns:
        One-hot encoded tensor of shape [..., num_slots, NUM_STAGES].
        Inactive slots (index -1) are all zeros.
    """
    table = _get_cached_table(device)
    valid_mask = indices >= 0  # -1 marks inactive slots
    safe_indices = indices.clamp(min=0)
    one_hot = table[safe_indices]
    return one_hot * valid_mask.unsqueeze(-1).float()  # Zeros out inactive
```

**Why device-keyed cache (from PyTorch review):** The naive `.to(indices.device)` allocates a new tensor every step. With ~500 steps/episode across 4 environments, that's 2000 allocations/episode. The cache ensures one-time transfer per device.

**Why device normalization:** `torch.device("cuda")` and `torch.device("cuda:0")` hash differently but represent the same GPU on single-GPU systems. Without normalization, the cache could create duplicate tensors for the same physical device, wasting VRAM.

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
    # ... feature extraction ...

    # CRITICAL: nn.Embedding requires int64 (LongTensor).
    # NumPy may default to int32 on some platforms - be explicit!
    bp_indices = np.zeros((n_envs, num_slots), dtype=np.int64)
    for env_idx, reports in enumerate(batch_slot_reports):
        for slot_idx, slot_id in enumerate(slot_config.slot_ids):
            if report := reports.get(slot_id):
                bp_indices[env_idx, slot_idx] = report.blueprint_index
            else:
                bp_indices[env_idx, slot_idx] = -1  # Inactive slot

    blueprint_indices = torch.from_numpy(bp_indices).to(device)
    # tensor is already int64 from numpy dtype
    return obs, blueprint_indices
```

**⚠️ dtype Gotcha:** `nn.Embedding` raises `RuntimeError: Expected tensor for argument #1 'indices' to have scalar type Long` if given int32. Always use `np.int64` or `torch.int64`.

#### 2f. Update get_feature_size()

Replace existing function (no `_v3` suffix—this is the new implementation):

```python
def get_feature_size(slot_config: SlotConfig) -> int:
    """Feature size excluding blueprint embeddings (added by network).

    Breakdown:
        Base features:     24 dims (epoch, loss, accuracy, history, stage counts, etc.)
        Action feedback:    7 dims (last_action_success + last_action_op one-hot)
        Per-slot features: 30 dims × num_slots (stage, alpha, gradient health, etc.)

    For 3 slots: 24 + 7 + (30 × 3) = 31 + 90 = 121 dims

    Note: Blueprint embeddings (4 × num_slots = 12 for 3 slots) are added
    inside the network via BlueprintEmbedding, making total network input 133.
    """
    BASE_FEATURES = 24
    ACTION_FEEDBACK = 7
    SLOT_FEATURES = 30
    return BASE_FEATURES + ACTION_FEEDBACK + (SLOT_FEATURES * slot_config.num_slots)
```

**Dimension breakdown (for reference):**

| Component | Dims | Notes |
|-----------|------|-------|
| Base features | 24 | epoch, loss, accuracy, 10-dim raw history, stage counts |
| Action feedback | 7 | last_action_success (1) + last_action_op one-hot (6) |
| Per-slot | 30 × 3 = 90 | stage, alpha, gradient health, contribution, etc. |
| **Non-blueprint obs** | **121** | Returned by `get_feature_size()` |
| Blueprint embeddings | 4 × 3 = 12 | Added inside network |
| **Total network input** | **133** | —

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

- `src/esper/tamiyo/networks/factored_lstm.py`

> **Decision:** Keep `BlueprintEmbedding` in `factored_lstm.py` since it's intimately tied to
> the network architecture and will be used directly in the network's `__init__` and `forward`.

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

        # Small initialization per DRL expert recommendation
        nn.init.normal_(self.embedding.weight, std=0.02)

        # Register null index as buffer: moves with module.to(device), no grad, in state_dict
        # This avoids per-forward-call tensor allocation that torch.tensor() would cause
        self.register_buffer(
            '_null_idx',
            torch.tensor(BLUEPRINT_NULL_INDEX, dtype=torch.int64)
        )

    def forward(self, blueprint_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            blueprint_indices: [batch, num_slots] with -1 for inactive, dtype=int64
        Returns:
            [batch, num_slots, embed_dim]
        """
        # _null_idx is already on correct device via module.to(device)
        safe_idx = torch.where(blueprint_indices < 0, self._null_idx, blueprint_indices)
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

#### 4a. Replace FactoredRecurrentActorCritic with V2

Per clean replacement policy, delete the old class and create the new one:

- `feature_dim = 512`
- `lstm_hidden_dim = 512`
- `head_hidden = lstm_hidden_dim // 2 = 256`
- `BlueprintEmbedding` module
- 3-layer blueprint head
- Op-conditioned value head

**Note:** If you need to reference old code during implementation, use `git show HEAD~1:path/to/file`.

#### 4b. Implement 3-Layer Blueprint Head

```python
self.blueprint_head = nn.Sequential(
    nn.Linear(lstm_hidden_dim, lstm_hidden_dim),  # 512 → 512
    nn.ReLU(),
    nn.Linear(lstm_hidden_dim, head_hidden),      # 512 → 256
    nn.ReLU(),
    nn.Linear(head_hidden, num_blueprints),       # 256 → 13
)

# Initialization: gain=0.01 for final layer, sqrt(2) for intermediate
def _init_blueprint_head(self):
    """Apply orthogonal init with appropriate gains to blueprint head."""
    layers = [m for m in self.blueprint_head if isinstance(m, nn.Linear)]
    for i, layer in enumerate(layers):
        if i == len(layers) - 1:  # Final layer
            nn.init.orthogonal_(layer.weight, gain=0.01)
        else:  # Intermediate layers
            nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
        nn.init.zeros_(layer.bias)
```

#### 4c. Implement Op-Conditioned Value Head

```python
self.value_head = nn.Sequential(
    nn.Linear(lstm_hidden_dim + NUM_OPS, head_hidden),  # 512+6 → 256
    nn.ReLU(),
    nn.Linear(head_hidden, 1),                          # 256 → 1
)

def _compute_value(self, lstm_out: torch.Tensor, op: torch.Tensor) -> torch.Tensor:
    """Shared helper: compute Q(s, op) value conditioned on operation.

    Used by both forward() (with sampled op) and evaluate_actions() (with stored op).
    """
    op_one_hot = F.one_hot(op, num_classes=NUM_OPS).float()
    value_input = torch.cat([lstm_out, op_one_hot], dim=-1)
    return self.value_head(value_input).squeeze(-1)
```

#### 4d. Define Output Types

Use `NamedTuple` for strict typing and torch.jit compatibility:

```python
from typing import NamedTuple

class ForwardOutput(NamedTuple):
    """Output from forward() during rollout."""
    op_logits: torch.Tensor           # [batch, seq, NUM_OPS]
    slot_logits: torch.Tensor         # [batch, seq, num_slots]
    blueprint_logits: torch.Tensor    # [batch, seq, NUM_BLUEPRINTS]
    style_logits: torch.Tensor        # [batch, seq, num_styles]
    tempo_logits: torch.Tensor        # [batch, seq, num_tempos]
    alpha_target_logits: torch.Tensor # [batch, seq, num_targets]
    alpha_speed_logits: torch.Tensor  # [batch, seq, num_speeds]
    alpha_curve_logits: torch.Tensor  # [batch, seq, num_curves]
    value: torch.Tensor               # [batch, seq] - Q(s, sampled_op)
    sampled_op: torch.Tensor          # [batch, seq] - the op used for value conditioning
    hidden: tuple[torch.Tensor, torch.Tensor]  # LSTM state

class EvaluateOutput(NamedTuple):
    """Output from evaluate_actions() during PPO update."""
    log_probs: dict[str, torch.Tensor]  # Per-head log probs of stored actions
    entropy: dict[str, torch.Tensor]     # Per-head entropy
    value: torch.Tensor                  # [batch, seq] - Q(s, stored_op)
```

#### 4e. Implement forward() for Rollout

```python
def forward(
    self,
    state: torch.Tensor,              # [batch, seq, 121]
    blueprint_indices: torch.Tensor,  # [batch, seq, num_slots]
    hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    action_mask: torch.Tensor | None = None,
) -> ForwardOutput:
    """Rollout path: sample actions, compute Q(s, sampled_op).

    This is used during data collection. The value returned is conditioned
    on the sampled op and should be stored in the rollout buffer.
    """
    # ... feature processing and LSTM ...

    # Sample op from policy
    op_logits = self.op_head(lstm_out)
    op_dist = Categorical(logits=op_logits)
    sampled_op = op_dist.sample()

    # Value conditioned on sampled op (what we store in buffer)
    value = self._compute_value(lstm_out, sampled_op)

    return ForwardOutput(
        op_logits=op_logits,
        # ... other logits ...
        value=value,
        sampled_op=sampled_op,
        hidden=new_hidden,
    )
```

#### 4f. Implement evaluate_actions() for PPO Update

```python
def evaluate_actions(
    self,
    state: torch.Tensor,              # [batch, seq, 121]
    blueprint_indices: torch.Tensor,  # [batch, seq, num_slots]
    actions: dict[str, torch.Tensor], # Stored actions from buffer
    hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    action_mask: torch.Tensor | None = None,
) -> EvaluateOutput:
    """PPO update path: compute log_probs, entropy, Q(s, stored_op).

    The value is conditioned on the STORED op (actions["op"]), ensuring
    consistency with what was stored during rollout.
    """
    # ... feature processing and LSTM (same as forward) ...

    # Compute distributions for all heads
    op_logits = self.op_head(lstm_out)
    op_dist = Categorical(logits=op_logits)

    # Log prob of the STORED action (not a fresh sample)
    stored_op = actions["op"]
    op_log_prob = op_dist.log_prob(stored_op)

    # Value conditioned on STORED op (must match what was stored)
    value = self._compute_value(lstm_out, stored_op)

    # Compute log_probs and entropy for all heads
    log_probs = {"op": op_log_prob, ...}
    entropy = {"op": op_dist.entropy(), ...}

    return EvaluateOutput(log_probs=log_probs, entropy=entropy, value=value)
```

#### 4g. Implement get_value() for Bootstrap

```python
def get_value(
    self,
    state: torch.Tensor,              # [batch, 1, 121] - single step
    blueprint_indices: torch.Tensor,  # [batch, 1, num_slots]
    hidden: tuple[torch.Tensor, torch.Tensor],
    sampled_op: torch.Tensor,         # [batch] - op from final step
) -> torch.Tensor:
    """Compute bootstrap value at episode truncation.

    Used when episode is truncated (not terminal). Must condition on
    the same op that would have been taken, for GAE consistency.
    """
    # ... feature processing and LSTM ...
    return self._compute_value(lstm_out, sampled_op)
```

**Design rationale:** We use hard one-hot conditioning in **both** rollout and PPO update:
- `forward()`: `Q(s, sampled_op)` — value stored matches the op sampled
- `evaluate_actions()`: `Q(s, stored_op)` — stored_op == sampled_op from rollout
- `get_value()`: `Q(s, sampled_op)` — for bootstrap at truncation

This ensures the value function trained matches what was stored, avoiding GAE mismatch.

**Validation:**

```bash
PYTHONPATH=src python -c "
import torch
from esper.tamiyo.networks.factored_lstm import (
    FactoredRecurrentActorCritic, ForwardOutput, EvaluateOutput
)
net = FactoredRecurrentActorCritic(state_dim=121, num_slots=3)
state = torch.randn(2, 5, 121)  # batch=2, seq=5
bp_idx = torch.randint(0, 13, (2, 5, 3))

# Test forward
out = net(state, bp_idx)
assert isinstance(out, ForwardOutput)
print(f'Op logits: {out.op_logits.shape}')  # [2, 5, 6]
print(f'Value: {out.value.shape}')          # [2, 5]
print(f'Sampled op: {out.sampled_op.shape}')  # [2, 5]

# Test evaluate_actions
actions = {'op': torch.randint(0, 6, (2, 5)), 'slot': torch.randint(0, 3, (2, 5)), ...}
eval_out = net.evaluate_actions(state, bp_idx, actions)
assert isinstance(eval_out, EvaluateOutput)
print(f'Value (eval): {eval_out.value.shape}')  # [2, 5]

print(f'Params: {sum(p.numel() for p in net.parameters())}')  # ~2.1M
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

#### 6b. Obs V3 State Tracking in ParallelEnvState

**IMPORTANT:** This project uses `ParallelEnvState` (in `simic/training/parallel_env_state.py`), NOT a separate `EnvState` class. All Obs V3 state tracking fields go here.

Add to `ParallelEnvState`:

```python
from esper.leyline import LifecycleOp

@dataclass
class ParallelEnvState:
    # ... existing fields (signals, reports, hidden, etc.) ...

    # === Obs V3 Action Feedback (7 dims: 1 success + 6 op one-hot) ===
    last_action_success: bool = True
    last_action_op: int = LifecycleOp.WAIT.value  # 0

    # === Obs V3 Gradient Health Tracking (for gradient_health_prev feature) ===
    # Maps slot_id -> previous epoch's gradient_health value
    # Used in Phase 2c feature extraction
    gradient_health_prev: dict[str, float] = field(default_factory=dict)

    # === Obs V3 Counterfactual Freshness Tracking ===
    # Maps slot_id -> epochs since last counterfactual measurement
    # Incremented each epoch, reset to 0 when counterfactual is computed
    epochs_since_counterfactual: dict[str, int] = field(default_factory=dict)
```

**State Update Contract:**

| Field | When to Read | When to Update | Initial Value |
|-------|--------------|----------------|---------------|
| `last_action_success` | Feature extraction (Phase 2b) | After action execution, before next step | `True` |
| `last_action_op` | Feature extraction (Phase 2b) | After action execution, before next step | `LifecycleOp.WAIT.value` (0) |
| `gradient_health_prev[slot_id]` | Feature extraction (Phase 2c) | After epoch completion, from `report.telemetry.gradient_health` | `1.0` (assume healthy for new slots) |
| `epochs_since_counterfactual[slot_id]` | Feature extraction (Phase 2c) | Increment each epoch; reset to 0 after counterfactual computed | `0` for newly germinated slots |

**Action Success Definition:**

```python
# In vectorized.py, after action execution:
def _determine_action_success(action: dict, result: ActionResult) -> bool:
    """Determine if the action succeeded for action feedback feature."""
    op = LifecycleOp(action["op"])

    if op == LifecycleOp.WAIT:
        return True  # WAIT always "succeeds"
    elif op == LifecycleOp.GERMINATE:
        return result.germination_succeeded  # Did seed actually germinate?
    elif op == LifecycleOp.SET_ALPHA:
        return True  # Alpha changes always succeed
    elif op == LifecycleOp.PRUNE:
        return result.pruned  # Did the prune actually happen?
    elif op == LifecycleOp.FOSSILIZE:
        return result.fossilized  # Did fossilization succeed?
    else:
        return True  # Unknown ops default to success

# Update state after action
env_state.last_action_op = action["op"]
env_state.last_action_success = _determine_action_success(action, result)
```

**Gradient Health Prev Update:**

```python
# After epoch completion (in vectorized.py step loop):
for slot_id, report in slot_reports.items():
    if report.telemetry is not None:
        env_state.gradient_health_prev[slot_id] = report.telemetry.gradient_health
    # Note: Keep stale values for inactive slots until slot is cleared

# When slot becomes inactive (PRUNED/FOSSILIZED):
if slot_id in env_state.gradient_health_prev:
    del env_state.gradient_health_prev[slot_id]
```

**Counterfactual Freshness Update:**

```python
# After each epoch (in vectorized.py step loop):
for slot_id in active_slot_ids:
    env_state.epochs_since_counterfactual[slot_id] = (
        env_state.epochs_since_counterfactual.get(slot_id, 0) + 1
    )

# When counterfactual is computed (in attribution module):
env_state.epochs_since_counterfactual[slot_id] = 0

# When slot is germinated:
env_state.epochs_since_counterfactual[slot_id] = 0

# When slot becomes inactive:
if slot_id in env_state.epochs_since_counterfactual:
    del env_state.epochs_since_counterfactual[slot_id]
```

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
| `leyline/__init__.py` | 1 | Dimension constants (512), BLUEPRINT_NULL_INDEX, DEFAULT_BLUEPRINT_EMBED_DIM, NUM_BLUEPRINTS |
| `simic/training/parallel_env_state.py` | 2a½ | Add Obs V3 state tracking fields (action feedback, gradient_health_prev, epochs_since_counterfactual) |
| `tamiyo/policy/features.py` | 2 | Clean replacement, raw history (10 dims), gradient_health_prev, device-keyed cache, gamma-matched counterfactual, returns `(obs[121], bp_idx)` |
| `tamiyo/networks/factored_lstm.py` | 3, 4 | BlueprintEmbedding, ForwardOutput/EvaluateOutput types, 3-layer blueprint head, op-conditioned Q(s,op) critic, `_compute_value()` helper |
| `simic/agent/ppo.py` | 5, 6f | Differential entropy, evaluate_actions with blueprint_indices |
| `simic/agent/rollout_buffer.py` | 6a | Add `blueprint_indices` tensor, add `sampled_op` storage, update `add()` and `get_batched_sequences()` |
| `simic/training/vectorized.py` | 6 | Action feedback tracking, gradient_health_prev tracking, new feature extraction API, blueprint_indices plumbing |

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

### 8. Enum Normalization Strategy

**Decision:** Keep `.value / max_value` pattern for enum normalization (simpler, auto-adapts to new enum values).

```python
# Current approach (keeping this)
alpha_mode_norm = mode.value / (len(AlphaMode) - 1)
alpha_algorithm_norm = algo.value / (len(AlphaAlgorithm) - 1)
```

**⚠️ ASSUMPTION:** This requires enum values to be sequential starting from 0:
- `AlphaMode.HOLD = 0, RAMP = 1, STEP = 2` ✓
- `AlphaAlgorithm.ADD = 0, MULTIPLY = 1, GATE = 2` ✓
- `SeedStage.DORMANT = 0, ... , FOSSILIZED = 9` ✓

**If this assumption breaks** (non-sequential values, gaps, or values > len-1), switch to explicit mapping:
```python
_MODE_INDEX = {AlphaMode.HOLD: 0, AlphaMode.RAMP: 1, AlphaMode.STEP: 2}
alpha_mode_norm = _MODE_INDEX[mode] / (len(_MODE_INDEX) - 1)
```

**Maintenance note:** When adding new enum values, ensure they follow the sequential pattern OR update all normalization code.

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
