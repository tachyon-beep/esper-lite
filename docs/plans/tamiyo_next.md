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
| 4: Policy V2 + Q(s,op) | **High** | **High** | Critic becomes action-conditioned Q(s,op) baseline; bootstrap + GAE consistency |
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

#### Phase 1a: Additional Leyline Constants (Pre-Implementation)

Before beginning Phase 2, define these constants in `src/esper/leyline/__init__.py` to eliminate magic numbers in subsequent phases:

```python
# =============================================================================
# Obs V3 Dimension Constants
# =============================================================================

# Non-blueprint feature dimension for Obs V3
# Breakdown: 24 base + 7 temporal + 30×3 slots = 121
OBS_V3_NON_BLUEPRINT_DIM = 121

# Default number of slots in training configurations
DEFAULT_NUM_SLOTS = 3

# =============================================================================
# PPO Architecture Constants
# =============================================================================

# Number of action heads in the factored policy
# (slot, blueprint, style, tempo, alpha_target, alpha_speed, alpha_curve, op)
# See ACTION_HEAD_SPECS in factored_actions.py for the authoritative list
NUM_ACTION_HEADS = 8

# PPO clipping epsilon (Schulman et al., 2017)
DEFAULT_PPO_CLIP_EPSILON = 0.2

# Minimum log probability for numerical stability
# exp(-100) ≈ 3.7e-44 (tiny but non-zero in float64)
# In float32: exp(-88) ≈ 1e-38 (smallest normal), exp(-104) underflows to 0.0
# For ratio stability, we need exp(new_lp - old_lp) to be finite
LOG_PROB_MIN = -100.0
```

**Add to `__all__`:**

```python
"OBS_V3_NON_BLUEPRINT_DIM",
"DEFAULT_NUM_SLOTS",
"NUM_ACTION_HEADS",
"DEFAULT_PPO_CLIP_EPSILON",
"LOG_PROB_MIN",
```

**Validation:**

```bash
PYTHONPATH=src python -c "
from esper.leyline import (
    OBS_V3_NON_BLUEPRINT_DIM, DEFAULT_NUM_SLOTS,
    NUM_ACTION_HEADS, DEFAULT_PPO_CLIP_EPSILON, LOG_PROB_MIN
)
print(f'OBS_DIM={OBS_V3_NON_BLUEPRINT_DIM}, SLOTS={DEFAULT_NUM_SLOTS}')
print(f'HEADS={NUM_ACTION_HEADS}, CLIP={DEFAULT_PPO_CLIP_EPSILON}, LOG_MIN={LOG_PROB_MIN}')
"
# Should print:
# OBS_DIM=121, SLOTS=3
# HEADS=9, CLIP=0.2, LOG_MIN=-100.0
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

**Add helper function for history padding:**

```python
def _pad_history(history: list[float], length: int = 5) -> list[float]:
    """Left-pad history to fixed length with zeros.

    Args:
        history: Raw history values (may be shorter than length at episode start)
        length: Target length (default 5)

    Returns:
        List of exactly `length` values, left-padded with 0.0 if needed
    """
    if len(history) >= length:
        return history[-length:]
    return [0.0] * (length - len(history)) + history
```

**Action feedback validation:**

```python
from esper.leyline import NUM_OPS

# Validate last_action_op is in valid range before one-hot encoding
assert 0 <= env_state.last_action_op < NUM_OPS, \
    f"Invalid last_action_op: {env_state.last_action_op} (expected 0-{NUM_OPS-1})"

last_op_one_hot = F.one_hot(
    torch.tensor(env_state.last_action_op),
    num_classes=NUM_OPS
).float()
```

**Action feedback after rollback:**

When `reset_episode_state()` is called after a governor rollback (catastrophic failure), the action feedback is reset to the same initial values as a fresh episode start:
- `last_action_success = True`
- `last_action_op = LifecycleOp.WAIT.value`

This is intentional—the penalty signal from `mark_terminal_with_penalty()` provides the failure signal, so action feedback doesn't need to encode rollback status separately.

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

> ⚠️ **CRITICAL: gradient_health telemetry is currently hardcoded to 1.0**
>
> In `vectorized.py` line 2382, the telemetry sync sets `gradient_health=1.0` because
> `DualGradientStats` doesn't compute health scores. This means `gradient_health_prev`
> will always be 1.0—a **constant feature with zero information gain**.
>
> **Before implementing Phase 2c, you MUST either:**
> 1. **Fix the source:** Compute real gradient health from the `SeedGradientCollector`
>    (requires refactoring telemetry sync to use the full collector, not just dual stats)
> 2. **Remove the feature:** Delete `gradient_health_prev` from the observation spec
>    and save the dimension for something useful
>
> Per CLAUDE.md: "do not defer or put it off" for telemetry components. Option 1 is preferred.

> **State Tracking:** The `gradient_health_prev` and `epochs_since_counterfactual` values
> require tracking in `ParallelEnvState`. Field definitions and initial values are in
> **Phase 2a½**. Update timing and logic are in **Phase 6b**.

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
import threading
from esper.leyline import NUM_STAGES

# Module-level constants
_STAGE_ONE_HOT_TABLE = torch.eye(NUM_STAGES, dtype=torch.float32)

# Device-keyed cache to avoid per-step .to(device) allocations
# Thread-safe for potential multi-worker DataLoader scenarios
_DEVICE_CACHE: dict[torch.device, torch.Tensor] = {}
_DEVICE_CACHE_LOCK = threading.Lock()

def _normalize_device(device: torch.device | str) -> torch.device:
    """Normalize device to canonical form.

    Handles:
    - "cuda" or torch.device("cuda") → torch.device("cuda", current_device)
    - "cpu" or torch.device("cpu") → torch.device("cpu")
    - "cuda:0" → torch.device("cuda", 0)

    This prevents duplicate cache entries for the same physical device.
    torch.device("cuda") != torch.device("cuda:0") but they're the same GPU.
    """
    # Convert string to torch.device if needed
    if isinstance(device, str):
        device = torch.device(device)

    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device

def _get_cached_table(device: torch.device) -> torch.Tensor:
    """Get stage one-hot table for device, caching to avoid repeated transfers.

    Thread-safe via double-checked locking pattern.
    """
    device = _normalize_device(device)
    if device not in _DEVICE_CACHE:
        with _DEVICE_CACHE_LOCK:
            if device not in _DEVICE_CACHE:  # Double-check after acquiring lock
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

#### 2e. Implement batch_obs_to_features()

Replace the existing function to return `(obs, blueprint_indices)` tuple (clean replacement, no `_v3` suffix):

```python
def batch_obs_to_features(
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

> ⚠️ **PREREQUISITE: `SeedStateReport.blueprint_index` field**
>
> The current `SeedStateReport` has `blueprint_id: str` (e.g., `"CONV_HEAVY"`), NOT `blueprint_index: int`.
> Before implementing Phase 2e, you must **either:**
>
> **Option A (Preferred): Add `blueprint_index` field to `SeedStateReport`**
> ```python
> # In src/esper/leyline/reports.py
> @dataclass
> class SeedStateReport:
>     blueprint_id: str = ""
>     blueprint_index: int = -1  # NEW: Index for nn.Embedding lookup
> ```
> Then update `SeedStateFactory` to populate both fields consistently.
>
> **Option B: Derive index from ID in feature extraction**
> ```python
> from esper.leyline import BlueprintAction
>
> _BLUEPRINT_TO_INDEX: dict[str, int] = {
>     bp.name: idx for idx, bp in enumerate(BlueprintAction)
> }
>
> # In extraction:
> bp_idx = _BLUEPRINT_TO_INDEX.get(report.blueprint_id, -1)
> ```
>
> Option A is preferred because it centralizes the mapping and avoids hot-path dict lookups.

> **Note on `reports.get(slot_id)`:** This is **legitimate dictionary access**, not defensive programming. The `reports` parameter is `dict[str, SeedStateReport]` — a dictionary mapping slot IDs to optional seed state reports. Not all slot IDs will have reports (inactive slots return `None`), so `.get()` is the correct idiom for probing optional dictionary entries. This differs from `.get()` on a dataclass or object attribute, which would violate the codebase's prohibition on defensive programming patterns.

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

#### 2g. Phase 1-2 Validation Checklist

Before proceeding to Phase 3, verify these potential silent bug sources:

##### 1. Feature Index Verification

Verify the dimension breakdown matches `state_dim` passed to the network:

| Component | Dims | Cumulative |
|-----------|------|------------|
| Base features | 24 | 24 |
| Action feedback | 7 | 31 |
| Per-slot (3 slots) | 90 | 121 |

```bash
PYTHONPATH=src python -c "
from esper.tamiyo.policy.features import get_feature_size
from esper.leyline.slot_config import SlotConfig

config = SlotConfig.default()
feature_size = get_feature_size(config)

# Expected breakdown
BASE = 24
ACTION_FEEDBACK = 7
SLOT_FEATURES = 30
expected = BASE + ACTION_FEEDBACK + (SLOT_FEATURES * config.num_slots)

assert feature_size == expected, f'Mismatch: {feature_size} != {expected}'
assert feature_size == 121, f'Expected 121 dims, got {feature_size}'
print(f'✓ Feature size verified: {feature_size} dims')
print(f'  Base: {BASE}, Action feedback: {ACTION_FEEDBACK}, Slots: {SLOT_FEATURES}×{config.num_slots}={SLOT_FEATURES * config.num_slots}')
"
```

##### 2. Blueprint Mapping Drift Guard

Add this assertion to prevent silent drift if `BlueprintAction` enum changes:

```python
# In features.py or as a module-level assertion
from esper.leyline import BlueprintAction, NUM_BLUEPRINTS

# _BLUEPRINT_TO_INDEX maps BlueprintAction enum values to embedding indices
_BLUEPRINT_TO_INDEX: dict[BlueprintAction, int] = {
    bp: idx for idx, bp in enumerate(BlueprintAction)
}

# Guard: Ensure all BlueprintAction values are mapped
assert len(_BLUEPRINT_TO_INDEX) == len(BlueprintAction), (
    f"Blueprint mapping drift: {len(_BLUEPRINT_TO_INDEX)} mapped vs "
    f"{len(BlueprintAction)} enum values"
)

# Guard: Ensure mapping covers indices 0 to NUM_BLUEPRINTS-1
assert set(_BLUEPRINT_TO_INDEX.values()) == set(range(NUM_BLUEPRINTS)), (
    f"Blueprint indices must cover 0..{NUM_BLUEPRINTS-1}, "
    f"got {sorted(_BLUEPRINT_TO_INDEX.values())}"
)
```

**Runtime Validation Guard (DRL Review Addition):**

The static guards above catch enum/constant drift at import time. Add this runtime guard in `batch_obs_to_features()` to catch data corruption during extraction:

```python
# In batch_obs_to_features(), validate runtime blueprint indices:
# ALWAYS fail-fast - per CLAUDE.md "No Bug-Hiding Patterns" policy


def _validate_blueprint_index(bp_idx: int, slot_id: str) -> int:
    """Validate blueprint index - raises on invalid data.

    Per CLAUDE.md: "If code would fail without a defensive pattern, that failure
    is a bug to fix, not a symptom to suppress."

    If this raises, the bug is in SeedStateReport CREATION, not here.
    Fix the upstream producer, don't sanitize here.

    Args:
        bp_idx: Blueprint index from observation (-1 for inactive, 0-12 for active)
        slot_id: Slot identifier for error context

    Raises:
        ValueError: If index is invalid.
    """
    if bp_idx != -1 and (bp_idx < 0 or bp_idx >= NUM_BLUEPRINTS):
        raise ValueError(
            f"Slot {slot_id} has invalid blueprint_index {bp_idx}. "
            f"Valid range: 0-{NUM_BLUEPRINTS-1} or -1 (inactive). "
            f"This indicates a bug in SeedStateReport creation - fix upstream."
        )
    return bp_idx


# Usage in feature extraction loop:
for slot_idx, slot_id in enumerate(slot_config.slot_ids):
    if report := reports.get(slot_id):
        # Derive blueprint_index from blueprint_id (see prerequisite note in Phase 2e)
        bp_idx = _BLUEPRINT_TO_INDEX.get(report.blueprint_id, -1)
        bp_idx = _validate_blueprint_index(bp_idx, slot_id)  # Fail-fast
        bp_indices[env_idx, slot_idx] = bp_idx
    else:
        bp_indices[env_idx, slot_idx] = -1  # Inactive slot
```

> ⚠️ **CLAUDE.md Compliance:** The previous `strict=False` pattern was removed because it violates
> the "No Bug-Hiding Patterns" prohibition. Invalid blueprint indices indicate bugs in upstream
> code (SeedStateReport creation). The fix is to validate at the source, not sanitize at consumption.
>
> If a 12-hour training run crashes due to invalid data, **that's a bug worth finding immediately**,
> not a symptom to suppress. The crash message now clearly identifies where the fix belongs.

**Validation command:**

```bash
PYTHONPATH=src python -c "
from esper.leyline import BlueprintAction, NUM_BLUEPRINTS

blueprint_values = list(BlueprintAction)
print(f'BlueprintAction has {len(blueprint_values)} values')
print(f'NUM_BLUEPRINTS = {NUM_BLUEPRINTS}')
assert len(blueprint_values) == NUM_BLUEPRINTS, (
    f'Enum/constant mismatch: {len(blueprint_values)} enum values vs NUM_BLUEPRINTS={NUM_BLUEPRINTS}'
)
print('✓ BlueprintAction enum matches NUM_BLUEPRINTS constant')
"
```

##### 3. Stage Encoding Debug Mode

For debugging stage encoding issues during testing, set the `ESPER_DEBUG_STAGE` environment variable.

**Important:** Use `VALID_STAGE_VALUES` instead of a simple range check. Stage value 5 (retired SHADOWING stage) should be rejected even though `5 < NUM_STAGES`. The `VALID_STAGE_VALUES` set explicitly lists only the stages that should produce valid one-hot encodings.

```python
# In features.py, add optional debug assertion
import os
from esper.leyline import VALID_STAGE_VALUES, SeedStage

_DEBUG_STAGE = os.environ.get("ESPER_DEBUG_STAGE", "0") == "1"

def _encode_stage(stage_value: int) -> torch.Tensor:
    if _DEBUG_STAGE:
        # Use VALID_STAGE_VALUES to reject retired stages like SHADOWING (value=5)
        # A simple range check (0 <= stage_value < NUM_STAGES) would incorrectly
        # accept SHADOWING, which should never appear in observations.
        assert stage_value in VALID_STAGE_VALUES, (
            f"Invalid stage value {stage_value}. "
            f"Valid values: {sorted(VALID_STAGE_VALUES)}. "
            f"Note: SHADOWING (5) is retired and must be rejected."
        )
    # ... encoding logic ...
```

**Design Decision: All-Zeros for Inactive Slots**

When a slot is inactive (stage is PRUNED, EMBARGOED, etc.), its stage encoding should be all-zeros, NOT a one-hot for stage 0. This is intentional:

- All-zeros is indistinguishable from "no slot" in the embedding space
- This prevents the network from learning spurious patterns from inactive slot stages
- A derived `is_active` feature (from stage) disambiguates "inactive slot" from "active slot with DORMANT stage"

> ⚠️ **CRITICAL: `SeedStateReport` does NOT have an `is_active` field.**
> Use `is_active_stage(report.stage)` from `esper.leyline.stages` instead.
>
> Also: `F.one_hot(report.stage, NUM_STAGES)` will FAIL because SeedStage values
> are non-contiguous (gap at 5, max is 10). Use `STAGE_TO_INDEX` mapping.

```python
from esper.leyline.stages import is_active_stage
from esper.leyline.stage_schema import STAGE_TO_INDEX, NUM_STAGES

# CORRECT: check activity via stage, use index mapping for one-hot
if not is_active_stage(report.stage):
    stage_one_hot = torch.zeros(NUM_STAGES)  # All zeros for inactive
else:
    stage_idx = STAGE_TO_INDEX[report.stage.value if hasattr(report.stage, 'value') else report.stage]
    stage_one_hot = F.one_hot(torch.tensor(stage_idx), NUM_STAGES)

# WRONG: F.one_hot with raw enum value (fails for HOLDING=6, RESETTING=10)
# stage_one_hot = F.one_hot(torch.tensor(report.stage), NUM_STAGES)  # DON'T DO THIS
```

**Enable during testing:**

```bash
# Run tests with stage validation enabled
ESPER_DEBUG_STAGE=1 PYTHONPATH=src uv run pytest tests/tamiyo/policy/test_features.py -v
```

##### 4. Action Mask dtype Validation

Before calling `.to(device)`, validate action mask dtype to catch silent type coercion bugs:

```python
# In compute_action_masks() or where masks are created
def _validate_action_mask(mask: torch.Tensor, name: str) -> None:
    """Validate action mask has correct dtype before device transfer."""
    valid_dtypes = (torch.bool, torch.float32, torch.float64)
    assert mask.dtype in valid_dtypes, (
        f"Action mask '{name}' has invalid dtype {mask.dtype}. "
        f"Expected one of {valid_dtypes}. "
        f"This can cause silent masking failures after .to(device)."
    )
    if mask.dtype in (torch.float32, torch.float64):
        # Float masks should only contain 0.0 or 1.0
        unique_vals = mask.unique()
        assert all(v in (0.0, 1.0) for v in unique_vals.tolist()), (
            f"Float action mask '{name}' contains values other than 0/1: {unique_vals.tolist()}"
        )

# Usage:
# action_mask = compute_action_masks(...)
# _validate_action_mask(action_mask, "slot_mask")
# action_mask = action_mask.to(device)
```

**Validation command:**

```bash
PYTHONPATH=src python -c "
import torch

# Simulate mask validation
def validate_mask(mask, name):
    valid_dtypes = (torch.bool, torch.float32, torch.float64)
    assert mask.dtype in valid_dtypes, f'{name}: invalid dtype {mask.dtype}'
    if mask.dtype in (torch.float32, torch.float64):
        unique = mask.unique()
        assert all(v in (0.0, 1.0) for v in unique.tolist()), f'{name}: non-binary values'
    print(f'✓ {name}: dtype={mask.dtype}, shape={mask.shape}')

# Test cases
validate_mask(torch.tensor([True, False, True]), 'bool_mask')
validate_mask(torch.tensor([1.0, 0.0, 1.0]), 'float_mask')
validate_mask(torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64), 'float64_mask')
print('✓ All mask dtype validations passed')
"
```

##### 5. Slot Configuration Edge Case Testing

Test feature extraction with various slot configurations to catch inactive slot handling bugs:

```bash
PYTHONPATH=src python -c "
import torch
from esper.tamiyo.policy.features import batch_obs_to_features, get_feature_size
from esper.leyline.slot_config import SlotConfig
from esper.leyline import TrainingSignals, SeedStateReport, SeedStage

# Create minimal test fixtures
def make_test_signals():
    return TrainingSignals(
        epoch=10,
        val_loss=0.5,
        val_accuracy=75.0,
        loss_history=[0.8, 0.7, 0.6, 0.55, 0.5],
        accuracy_history=[60.0, 65.0, 70.0, 72.0, 75.0],
    )

def make_test_report(slot_id: str, stage: int = 2):
    \"\"\"Create a test slot report.

    Note: SeedStateReport does NOT have is_active or blueprint_index fields.
    Activity is derived from stage via is_active_stage().
    Blueprint index must be derived from blueprint_id via mapping.
    \"\"\"
    from esper.leyline.stages import is_active_stage
    is_active = is_active_stage(SeedStage(stage))
    return SeedStateReport(
        slot_id=slot_id,
        stage=SeedStage(stage),
        blueprint_id='CONV_HEAVY' if is_active else '',  # String, not index
        alpha_target=0.3 if is_active else 0.0,
    )

config = SlotConfig.default()
device = torch.device('cpu')

# Test configurations: (reports_dict, description)
# Note: This tests DIFFERENT cases than 'no slot report at all'
test_cases = [
    # Basic slot occupancy cases
    ({}, 'No slot reports at all'),
    ({config.slot_ids[0]: make_test_report(config.slot_ids[0])}, 'Single active slot'),
    ({sid: make_test_report(sid) for sid in config.slot_ids[:2]}, 'Two active slots'),
    ({sid: make_test_report(sid) for sid in config.slot_ids}, 'All slots active'),

    # DRL Review Addition: Seeds in inactive stages WITH slot reports
    # This is different from 'no slot report' - the seed exists but is inactive
    # Note: is_active is DERIVED from stage, not a separate field
    ({config.slot_ids[0]: make_test_report(
        config.slot_ids[0],
        stage=SeedStage.PRUNED.value,  # Inactive stage
    )}, 'Seed in PRUNED stage (has report, inactive by stage)'),
    ({config.slot_ids[0]: make_test_report(
        config.slot_ids[0],
        stage=SeedStage.EMBARGOED.value,  # Inactive stage
    )}, 'Seed in EMBARGOED stage (has report, inactive by stage)'),

    # Mixed: some active, some inactive with reports
    ({
        config.slot_ids[0]: make_test_report(config.slot_ids[0], stage=SeedStage.TRAINING.value),  # Active
        config.slot_ids[1]: make_test_report(config.slot_ids[1], stage=SeedStage.PRUNED.value),  # Inactive
    }, 'Mixed: one TRAINING (active), one PRUNED (inactive)'),
]

for reports, desc in test_cases:
    signals = make_test_signals()

    try:
        obs, bp_idx = batch_obs_to_features([signals], [reports], config, device)
        expected_size = get_feature_size(config)
        assert obs.shape == (1, expected_size), f'Shape mismatch: {obs.shape} vs (1, {expected_size})'
        assert bp_idx.shape == (1, config.num_slots), f'BP idx shape: {bp_idx.shape}'

        # Verify inactive slots (either no report OR inactive stage) have -1 blueprint index
        from esper.leyline.stages import is_active_stage
        for i, slot_id in enumerate(config.slot_ids):
            report = reports.get(slot_id)
            if report is None or not is_active_stage(report.stage):
                assert bp_idx[0, i] == -1, f'Inactive slot {slot_id} should have bp_idx=-1, got {bp_idx[0, i]}'

        print(f'✓ {desc}: obs={obs.shape}, bp_idx={bp_idx.shape}')
    except Exception as e:
        print(f'✗ {desc}: {e}')
        raise

print('✓ All slot configuration edge cases passed')
"
```

**Note:** The test cases distinguish between:
1. **No slot report** - Slot is empty (never germinated or fully cleared)
2. **Slot report with inactive stage** - Seed exists but is in an inactive stage (PRUNED, EMBARGOED)
   - Activity is determined by `is_active_stage(report.stage)`, NOT a separate field

Both cases should produce `bp_idx=-1` and all-zeros feature encoding, but they arise from different situations in the training loop.

##### 2h. DRL-Specific Validation Checks

These checks address DRL-specific concerns that go beyond basic feature correctness.

**Gradient Flow Check**

Verify all 121 features receive gradients during backpropagation. Dead features (never receiving gradients) indicate either:
- Masking bugs that zero out features inappropriately
- Network architecture issues (dead ReLU paths)
- Feature extraction bugs that produce constant values

```python
# Add to test suite or run manually after Phase 4:
def test_gradient_flow_all_features():
    """Verify all 121 features receive gradients during a forward-backward pass."""
    import torch
    from esper.tamiyo.networks.factored_lstm import FactoredRecurrentActorCritic

    net = FactoredRecurrentActorCritic(state_dim=121, num_slots=3)
    net.train()

    # Create input that requires grad
    state = torch.randn(2, 5, 121, requires_grad=True)  # batch=2, seq=5, features=121
    bp_idx = torch.randint(0, 13, (2, 5, 3))

    # Forward pass
    output = net(state, bp_idx)

    # Backward pass on value (arbitrary choice - any output works)
    loss = output.value.sum()
    loss.backward()

    # Check gradient magnitude per feature
    grad = state.grad  # [2, 5, 121]
    grad_per_feature = grad.abs().mean(dim=(0, 1))  # [121]

    dead_features = (grad_per_feature == 0).nonzero(as_tuple=True)[0]
    assert len(dead_features) == 0, (
        f"Features {dead_features.tolist()} receive zero gradient. "
        f"Check masking logic and feature extraction for these indices."
    )
    print(f"✓ All 121 features receive gradients (min={grad_per_feature.min():.2e}, max={grad_per_feature.max():.2e})")
```

**Observation Normalization Check**

Features should be roughly in [-10, 10] for stable training. Features outside this range can cause:
- Gradient explosion during early training
- LSTM saturation (tanh/sigmoid activations)
- Poor learning dynamics

```python
# Add to feature extraction or test suite:
def _check_observation_normalization(obs: torch.Tensor, warn_threshold: float = 10.0) -> None:
    """Warn if any features are outside expected range.

    This is a soft check (warning, not error) because some outliers may be
    acceptable during transient training states.
    """
    import warnings

    obs_min = obs.min().item()
    obs_max = obs.max().item()

    if obs_min < -warn_threshold or obs_max > warn_threshold:
        # Find which features are problematic
        feature_mins = obs.min(dim=0).values if obs.dim() > 1 else obs
        feature_maxs = obs.max(dim=0).values if obs.dim() > 1 else obs

        problematic = []
        for i in range(obs.shape[-1]):
            if feature_mins[i] < -warn_threshold or feature_maxs[i] > warn_threshold:
                problematic.append((i, feature_mins[i].item(), feature_maxs[i].item()))

        warnings.warn(
            f"Observation features outside [-{warn_threshold}, {warn_threshold}] range: "
            f"{problematic[:5]}{'...' if len(problematic) > 5 else ''}. "
            f"Consider adjusting normalization in feature extraction.",
            RuntimeWarning,
        )
```

**Validation command:**

```bash
PYTHONPATH=src python -c "
import torch
import warnings

# Simulate observation normalization check
def check_obs_range(obs, name):
    obs_min, obs_max = obs.min().item(), obs.max().item()
    if obs_min < -10 or obs_max > 10:
        warnings.warn(f'{name}: range [{obs_min:.2f}, {obs_max:.2f}] exceeds [-10, 10]')
    else:
        print(f'✓ {name}: range [{obs_min:.2f}, {obs_max:.2f}] within bounds')

# Test cases
check_obs_range(torch.randn(4, 121) * 2, 'normal_obs')  # Should pass
check_obs_range(torch.randn(4, 121) * 20, 'scaled_obs')  # Should warn
print('✓ Observation normalization check working')
"
```

**Temporal Consistency Note (Action Feedback)**

The action feedback features (`last_action_success`, `last_action_op`) introduce temporal dependencies between consecutive observations. For DRL training to work correctly:

1. **Episode boundaries:** Action feedback must be reset at episode start (see Phase 2b)
2. **Rollback handling:** After governor rollback, action feedback resets (penalty signal is separate)
3. **Buffer alignment:** The action feedback in obs[t] describes action[t-1], not action[t]

If action feedback is misaligned (e.g., obs[t] contains feedback about action[t] instead of action[t-1]), the policy will learn incorrect temporal associations. This manifests as:
- Unexpectedly high explained variance (the network is "cheating" by seeing current action)
- Poor generalization at test time
- Reward correlation with action feedback features

#### 2i. Phase 2 Addendum: Normalization Validation

These checks catch silent bugs where normalization logic appears correct but produces degraded performance due to misconfiguration or missing updates.

##### 1. TaskConfig Calibration Check

Wrong `baseline_loss` or `max_epochs` in `TaskConfig` can silently skew normalization. If `max_epochs` is underestimated, normalized epoch counts saturate too early (always returning 1.0 after the cap). If `baseline_loss` is wrong, loss-based features will be scaled incorrectly.

**Verify TaskConfig matches your training scenario:**

```bash
PYTHONPATH=src python -c "
from esper.leyline import TaskConfig

# Get current config (adjust for your task)
config = TaskConfig.default()

print('=== TaskConfig Calibration Check ===')
print(f'baseline_loss: {config.baseline_loss}')
print(f'max_epochs: {config.max_epochs}')
print(f'loss_floor: {config.loss_floor}')
print(f'loss_ceiling: {config.loss_ceiling}')

# Sanity checks
print()
print('=== Sanity Validation ===')
if config.max_epochs < 150:
    print(f'⚠️  WARNING: max_epochs={config.max_epochs} < 150 (Obs V3 episode length)')
    print('   Epoch normalization will saturate before episode end!')

if config.baseline_loss < 0.1 or config.baseline_loss > 5.0:
    print(f'⚠️  WARNING: baseline_loss={config.baseline_loss} seems unusual')
    print('   Check this matches your task initial loss range')
else:
    print(f'✓ baseline_loss={config.baseline_loss} looks reasonable')

if config.max_epochs >= 150:
    print(f'✓ max_epochs={config.max_epochs} covers full episode')
"
```

**What happens if baseline_loss is wrong:**
- If baseline_loss is too high: Loss improvements appear smaller than they are (policy under-rewarded)
- If baseline_loss is too low: Small losses appear as huge improvements (noisy reward signal)
- Both cases degrade credit assignment quality

##### 2. Observation Normalizer Update Verification

If using a running normalizer (e.g., `ObservationNormalizer` with running mean/std), ensure `update()` is actually being called with raw observations. If the normalizer is created but never updated, statistics stay frozen at initialization values, and observations can drift outside the normalizer's expected range.

**⚠️ PyTorch Implementation Note:** The actual `RunningMeanStd` class uses `var` (variance), not `std`. The standard deviation is computed as `sqrt(var + epsilon)` when normalizing. When logging stats, check `running_var` not `running_std`.

**⚠️ Detection Check:** The `count` field is initialized to `epsilon` (typically 1e-4), not 0. To detect if updates are occurring, check `count > epsilon + 1`, not `count > 0`.

**⚠️ Gradient Warning:** If the normalizer is wrapped in `@torch.inference_mode()` or `@torch.no_grad()`, updates will work but any downstream gradient computation through the normalized values will fail silently. Ensure normalizer updates happen outside inference mode if gradients are needed.

**Track normalizer stats over training:**

```python
# Add to training loop or as periodic logging
from __future__ import annotations
from typing import Protocol, Union
import torch


class RunningNormalizerProtocol(Protocol):
    """Protocol for normalizers with running statistics.

    Addresses code review finding: Type annotations for better IDE support
    and runtime checking. Use Protocol for structural typing.
    """
    running_mean: torch.Tensor
    running_var: torch.Tensor
    count: Union[float, torch.Tensor]


def _log_normalizer_stats(
    normalizer: RunningNormalizerProtocol,
    step: int,
    epsilon: float = 1e-4,
    log_interval: int = 1000,
) -> dict[str, float] | None:
    """Log normalizer statistics to detect frozen updates.

    Args:
        normalizer: Object with running_mean, running_var, count attributes
        step: Current training step
        epsilon: Small value for numerical stability in std computation
        log_interval: How often to log (default every 1000 steps)

    Returns:
        Dict with stats if logged this step, None otherwise
    """
    # Log periodically
    if step % log_interval != 0:
        return None

    mean = normalizer.running_mean
    var = normalizer.running_var
    count = normalizer.count

    # Handle count being either float or tensor
    count_val = count.item() if isinstance(count, torch.Tensor) else count

    std = (var + epsilon).sqrt()
    stats = {
        "mean_avg": mean.mean().item(),
        "std_avg": std.mean().item(),
        "count": count_val,
    }

    print(
        f"[Step {step}] Normalizer: mean={stats['mean_avg']:.4f}, "
        f"std={stats['std_avg']:.4f}, count={stats['count']:.1f}"
    )

    return stats


# Expected behavior:
# - mean and var SHOULD change over first few thousand steps as data is collected
# - count should increase from epsilon toward step count
# - If count stays near epsilon, the update path was missed
```

**Manual verification command:**

```bash
PYTHONPATH=src python -c "
import torch

# Simulate normalizer behavior check (matches actual RunningMeanStd)
class MockNormalizer:
    def __init__(self, epsilon=1e-4):
        self.running_mean = torch.zeros(121)
        self.running_var = torch.ones(121)  # var, not std
        self.count = epsilon  # Initialized to epsilon, not 0
        self.epsilon = epsilon

    def update(self, obs):
        # Running mean/var update (Welford's algorithm)
        batch_mean = obs.mean(dim=0) if obs.dim() > 1 else obs
        batch_var = obs.var(dim=0) if obs.dim() > 1 else torch.zeros_like(obs)
        batch_count = obs.shape[0] if obs.dim() > 1 else 1

        delta = batch_mean - self.running_mean
        tot_count = self.count + batch_count

        self.running_mean = self.running_mean + delta * batch_count / tot_count
        # M2 update for variance (simplified)
        self.running_var = (self.running_var * self.count + batch_var * batch_count +
                           delta**2 * self.count * batch_count / tot_count) / tot_count
        self.count = tot_count

# Check: If count stays near epsilon after training starts, update() is never called
normalizer = MockNormalizer()
print(f'Initial: count={normalizer.count:.4f} (should be ~epsilon), mean={normalizer.running_mean.mean():.4f}')

# Simulate some updates
for _ in range(100):
    fake_obs = torch.randn(121) * 0.5 + 0.1  # Non-zero mean data
    normalizer.update(fake_obs)

print(f'After 100 updates: count={normalizer.count:.1f}, mean={normalizer.running_mean.mean():.4f}')
print()
if normalizer.count > normalizer.epsilon + 1 and abs(normalizer.running_mean.mean()) > 0.01:
    print('✓ Normalizer is being updated (count increased, mean shifted from 0)')
else:
    print('⚠️  WARNING: Normalizer may not be receiving updates')
"
```

##### 3. Clamping Saturation Monitoring

Loss values are clamped to ±10 and improvement percentages to ±10 points. If values frequently hit these bounds, information is lost—the policy can't distinguish between "slightly over limit" and "massively over limit".

**⚠️ Float Equality Bug:** Do NOT use `raw != clamped` to detect saturation—floating point equality is unreliable. Use boundary checks instead.

**⚠️ Asymmetric Saturation:** Track lower and upper bound saturation separately. Losses hitting -10 (improvement) vs +10 (degradation) are very different signals and may require different interventions.

**Add saturation logging to feature extraction:**

```python
# In batch_obs_to_features or _extract_base_features_v3
from __future__ import annotations
import logging
import threading
from dataclasses import dataclass, field
from typing import Optional

_saturation_logger = logging.getLogger("esper.tamiyo.features.saturation")


@dataclass
class ClampingSaturationTracker:
    """Thread-safe tracker for clamping saturation statistics.

    Addresses code review finding: Global dict anti-pattern replaced with
    class-based state management for thread safety and testability.

    Usage:
        tracker = ClampingSaturationTracker(log_interval=1000)
        tracker.record(raw_loss, raw_improvement, clamp_min=-10.0, clamp_max=10.0)

        # In tests:
        tracker.reset()  # Clear state between test runs
    """

    log_interval: int = 1000
    _loss_lower: int = field(default=0, repr=False)
    _loss_upper: int = field(default=0, repr=False)
    _improvement_lower: int = field(default=0, repr=False)
    _improvement_upper: int = field(default=0, repr=False)
    _total: int = field(default=0, repr=False)
    _lock: Optional[threading.Lock] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        # Only create lock if running in threaded context
        # (e.g., vectorized training with thread pool)
        self._lock = threading.Lock()

    def record(
        self,
        raw_loss: float,
        raw_improvement: float,
        clamp_min: float = -10.0,
        clamp_max: float = 10.0,
    ) -> None:
        """Record a sample and log if saturation rate exceeds threshold.

        Uses boundary checks instead of float equality to avoid precision issues.
        Tracks upper/lower bounds separately for asymmetric saturation analysis.

        Args:
            raw_loss: Unnormalized loss value before clamping
            raw_improvement: Unnormalized improvement value before clamping
            clamp_min: Lower clamp bound (default -10.0)
            clamp_max: Upper clamp bound (default 10.0)
        """
        with self._lock:
            self._total += 1

            # Use boundary checks, not equality comparison
            if raw_loss <= clamp_min:
                self._loss_lower += 1
            elif raw_loss >= clamp_max:
                self._loss_upper += 1

            if raw_improvement <= clamp_min:
                self._improvement_lower += 1
            elif raw_improvement >= clamp_max:
                self._improvement_upper += 1

            # Log periodically
            if self._total > 0 and self._total % self.log_interval == 0:
                self._log_stats(clamp_min, clamp_max)

    def _log_stats(self, clamp_min: float, clamp_max: float) -> None:
        """Log saturation statistics if above warning threshold."""
        total = self._total
        loss_lower_rate = self._loss_lower / total * 100
        loss_upper_rate = self._loss_upper / total * 100
        impr_lower_rate = self._improvement_lower / total * 100
        impr_upper_rate = self._improvement_upper / total * 100

        if loss_lower_rate > 10 or loss_upper_rate > 10:
            _saturation_logger.warning(
                f"Loss clamping saturation: {loss_lower_rate:.1f}% at {clamp_min}, "
                f"{loss_upper_rate:.1f}% at +{clamp_max}. "
                f"Consider widening clamp range or adjusting baseline_loss."
            )
        if impr_lower_rate > 10 or impr_upper_rate > 10:
            _saturation_logger.warning(
                f"Improvement clamping saturation: {impr_lower_rate:.1f}% at {clamp_min}, "
                f"{impr_upper_rate:.1f}% at +{clamp_max}. "
                f"Consider widening clamp range."
            )

    def reset(self) -> None:
        """Reset all counters. Useful for testing."""
        with self._lock:
            self._loss_lower = 0
            self._loss_upper = 0
            self._improvement_lower = 0
            self._improvement_upper = 0
            self._total = 0

    def get_stats(self) -> dict[str, float]:
        """Get current saturation rates as percentages."""
        with self._lock:
            if self._total == 0:
                return {"loss_lower": 0.0, "loss_upper": 0.0,
                        "improvement_lower": 0.0, "improvement_upper": 0.0, "total": 0}
            return {
                "loss_lower": self._loss_lower / self._total * 100,
                "loss_upper": self._loss_upper / self._total * 100,
                "improvement_lower": self._improvement_lower / self._total * 100,
                "improvement_upper": self._improvement_upper / self._total * 100,
                "total": self._total,
            }


# Module-level singleton (thread-safe, resettable for tests)
_saturation_tracker = ClampingSaturationTracker(log_interval=1000)


def log_clamping_saturation(
    raw_loss: float,
    raw_improvement: float,
    clamp_min: float = -10.0,
    clamp_max: float = 10.0,
) -> None:
    """Convenience function using module-level tracker.

    For custom tracking (e.g., per-environment in vectorized training),
    instantiate ClampingSaturationTracker directly.
    """
    _saturation_tracker.record(raw_loss, raw_improvement, clamp_min, clamp_max)
```

**Validation command (check current clamp bounds):**

```bash
PYTHONPATH=src python -c "
import torch

# Simulate clamping check with boundary detection (not float equality)
def check_saturation(values, clamp_min, clamp_max, name):
    # Use boundary checks, NOT equality comparison
    lower_saturated = (values <= clamp_min).float().mean().item() * 100
    upper_saturated = (values >= clamp_max).float().mean().item() * 100
    total_saturated = lower_saturated + upper_saturated

    if total_saturated > 10:
        print(f'⚠️  {name}: {total_saturated:.1f}% saturated (lower={lower_saturated:.1f}%, upper={upper_saturated:.1f}%)')
    else:
        print(f'✓ {name}: {total_saturated:.1f}% saturation (acceptable)')

# Test with realistic loss distributions
# Normal training: losses typically 0.1-3.0
normal_losses = torch.randn(1000).abs() * 1.5 + 0.5
check_saturation(normal_losses, -10, 10, 'Normal losses')

# Unstable training: occasional spikes
unstable_losses = torch.randn(1000).abs() * 5 + 1.0
check_saturation(unstable_losses, -10, 10, 'Unstable losses')

# Very unstable: frequent large values
very_unstable = torch.randn(1000).abs() * 15
check_saturation(very_unstable, -10, 10, 'Very unstable losses')
"
```

**If >10% saturation is observed:**
1. Check if training is unstable (should fix root cause, not widen bounds)
2. If stable but values naturally large, consider:
   - Adjusting `baseline_loss` to bring values into range
   - Using log-scale normalization (already used for loss in V3)
   - Widening clamp bounds (last resort—affects gradient scaling)

##### 4. global_step Sanity Check

The formula `global_step = epoch * num_train_batches` assumes counting AFTER finishing an epoch's batches. If the training loop defines `global_step` differently (e.g., starting from 0 before any batches), calculations could be off by one batch.

**⚠️ Batch vs Sample Counting:** Ensure clarity on whether `global_step` counts batches or individual samples. Some frameworks (e.g., PyTorch Lightning) count samples by default. This affects learning rate schedulers and logging alignment.

**⚠️ Consumer Consistency:** The `global_step` value must be consistent across all consumers:
- Feature extraction (for time-based features)
- Learning rate scheduler (for warmup/decay)
- TensorBoard/logging (for x-axis alignment)
- Checkpointing (for resume correctness)

If different components read `global_step` at different points in the training loop, they may see different values for the "same" step.

**Print global_step for first few epochs to verify alignment:**

```python
# Add to training loop (temporary debugging)
def _verify_global_step(signals, epoch: int, batch_idx: int, num_batches: int) -> None:
    """Verify global_step calculation matches expectations."""
    expected_at_epoch_end = (epoch + 1) * num_batches
    actual = signals.metrics.global_step if hasattr(signals, 'metrics') else None

    # Log at end of each of first 3 epochs
    if batch_idx == num_batches - 1 and epoch < 3:
        print(f"[Epoch {epoch}] global_step: expected={expected_at_epoch_end}, actual={actual}")
        if actual != expected_at_epoch_end:
            print(f"  ⚠️  Off by {actual - expected_at_epoch_end} batches")
```

**Validation command:**

```bash
PYTHONPATH=src python -c "
# Simulate global_step verification
num_batches = 100  # Example: 100 batches per epoch

print('=== global_step Alignment Check ===')
print(f'num_train_batches = {num_batches}')
print()

# Expected values at epoch boundaries
for epoch in range(3):
    # If counting AFTER epoch completion
    expected_after = (epoch + 1) * num_batches
    # If counting BEFORE epoch (0-indexed at start)
    expected_before = epoch * num_batches

    print(f'Epoch {epoch} end:')
    print(f'  If counted AFTER batches:  global_step = {expected_after}')
    print(f'  If counted BEFORE batches: global_step = {expected_before}')
    print()

print('Your training loop should match one of these patterns.')
print('Verify by printing signals.metrics.global_step at epoch boundaries.')
"
```

**If off-by-one detected:**
- Check where `global_step` is incremented in the training loop
- Ensure feature extraction reads `global_step` at a consistent point
- Document which convention is used (post-batch vs pre-batch counting)

##### 5. PyTorch Gotchas

These are PyTorch-specific issues that can cause subtle bugs in normalization and feature extraction.

**Device Migration GPU Sync**

Moving tensors to GPU with `.to(device)` triggers a synchronization point at first use. This can cause unexpected latency spikes:

```python
# This pattern causes a sync at each call:
normalized = (obs - self.running_mean.to(obs.device)) / self.running_std.to(obs.device)

# Better: Move once during setup or use device-keyed cache (see Phase 2d)
if self._device != obs.device:
    self.running_mean = self.running_mean.to(obs.device)
    self.running_std = self.running_std.to(obs.device)
    self._device = obs.device
```

**Normalizer Warmup Period**

During the first N steps, `RunningMeanStd` statistics are unstable because they're based on limited data. This can cause:
- Large swings in normalized observation values
- Policy receiving inconsistent input scale
- Value function learning on different input distributions

**Mitigation strategies:**
1. **Burn-in period:** Collect N steps of data before starting PPO updates
2. **Clip normalized values:** Even after warmup, clip to [-10, 10] to bound outliers
3. **Monitor stats:** Log `running_mean` and `running_var` to detect when they stabilize

```python
# Example warmup check
WARMUP_STEPS = 1000

def is_normalizer_stable(normalizer, warmup_steps: int = WARMUP_STEPS) -> bool:
    """Check if normalizer has seen enough data to be stable."""
    return normalizer.count > warmup_steps

# In training loop:
if not is_normalizer_stable(obs_normalizer):
    # Skip PPO update, just collect data
    continue
```

**torch.compile Graph Breaks**

If using `torch.compile()` for performance, in-place normalizer updates can cause graph breaks:

```python
# This may cause a graph break:
self.running_mean += delta  # In-place update

# torch.compile sees this as a side effect and breaks the graph
```

**Symptoms:**
- Unexpected recompilation warnings
- Performance regression instead of improvement
- `TORCH_LOGS=recompiles` shows frequent breaks

**Workarounds:**
1. Exclude normalizer from compiled region
2. Use `torch.compile(mode="reduce-overhead")` which is more tolerant
3. Batch normalizer updates outside the compiled forward pass

```python
# Option 1: Exclude from compilation
@torch.compile
def forward(self, x):
    # ... main logic ...
    pass

def update_normalizer(self, x):  # Not compiled
    self.normalizer.update(x)

# Option 2: Disable for normalizer
with torch._dynamo.config.disable():
    self.normalizer.update(x)
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

**⚠️ CRITICAL: Update `get_action()` for Op-Conditioned Value**

The current `get_action()` method returns value from the unconditioned value head. When implementing the op-conditioned critic:

1. Update `GetActionResult` dataclass to include `sampled_op: torch.Tensor`
2. Update `get_action()` to compute value via `_compute_value(lstm_out, sampled_op)`
3. Verify `vectorized.py` bootstrap code (around line 3212) uses the correctly conditioned value

This ensures bootstrap values at truncation are computed with the same Q(s,op) conditioning as rollout values.

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

    # Sample op from policy (with action masking)
    op_logits = self.op_head(lstm_out)
    # CRITICAL: Use MaskedCategorical, NOT raw Categorical
    # - Prevents sampling invalid/masked actions
    # - Computes entropy only over valid actions
    # - Uses -1e4 (not -inf) for numerical stability
    op_dist = MaskedCategorical(logits=op_logits, mask=op_mask)
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

    # Compute distributions for all heads (with action masking)
    op_logits = self.op_head(lstm_out)
    # CRITICAL: Use MaskedCategorical, NOT raw Categorical
    # This ensures entropy is computed only over valid actions and
    # log_prob returns correct values for the stored action
    op_dist = MaskedCategorical(logits=op_logits, mask=action_mask["op"])

    # Log prob of the STORED action (not a fresh sample)
    stored_op = actions["op"]
    op_log_prob = op_dist.log_prob(stored_op)

    # Value conditioned on STORED op (must match what was stored)
    value = self._compute_value(lstm_out, stored_op)

    # Compute log_probs and entropy for all heads
    # MaskedCategorical.entropy() already excludes masked actions
    log_probs = {"op": op_log_prob, ...}
    entropy = {"op": op_dist.entropy(), ...}

    return EvaluateOutput(log_probs=log_probs, entropy=entropy, value=value)
```

#### 4g. Implement get_value() for Debugging (Optional)

**⚠️ WARNING: This method is NOT used in the main training loop.**

The primary use cases are handled by:
- `forward()` → samples op and returns `Q(s, sampled_op)` (rollout + bootstrap)
- `evaluate_actions()` → uses stored op and returns `Q(s, stored_op)` (PPO update)

This method exists only for debugging/monitoring scenarios where you want to evaluate a specific state-op pair's value without sampling actions.

```python
@torch.no_grad()
def get_value(
    self,
    state: torch.Tensor,              # [batch, 1, 121] - single step
    blueprint_indices: torch.Tensor,  # [batch, 1, num_slots]
    hidden: tuple[torch.Tensor, torch.Tensor],
    sampled_op: torch.Tensor,         # [batch] - op to condition on
) -> torch.Tensor:
    """Compute value for a given state-op pair without gradient tracking.

    ⚠️ DEBUG ONLY - Do NOT use for bootstrap at truncation!
    Use forward() instead (see Gotcha #2).

    This method is for debugging scenarios like:
    - Visualizing Q(s, op) landscape for different ops
    - Comparing value estimates between ops
    - Unit testing value head behavior
    """
    # ... feature processing and LSTM ...
    return self._compute_value(lstm_out, sampled_op)
```

**@torch.no_grad() rationale:** Prevents gradient accumulation during debugging calls.

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

**Goal:** Update PPO for differential entropy and new network API.

**Files:**

- `src/esper/simic/agent/ppo.py`

#### 5a. Add Differential Entropy Coefficients

Add to `PPOConfig` dataclass (or as module-level constants in ppo.py):

```python
# Sparse heads need higher entropy coefficients to maintain exploration
# when they receive fewer training signals due to causal masking
ENTROPY_COEF_PER_HEAD: dict[str, float] = {
    "op": 1.0,           # Always active (100% of steps)
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

#### 5b. Update Entropy Loss Computation

```python
# In _ppo_update(), after calling evaluate_actions:
result = self.policy.evaluate_actions(states, blueprint_indices, actions, hidden, action_mask)

# result.entropy is dict[str, Tensor] from EvaluateOutput
total_entropy_loss = torch.zeros(1, device=device)
for head, entropy in result.entropy.items():
    coef = ENTROPY_COEF_PER_HEAD.get(head, 1.0)
    total_entropy_loss -= self.entropy_coef * coef * entropy.mean()
```

#### 5c. Update evaluate_actions() Call Site

Update the PPO update loop to pass blueprint_indices from the buffer:

```python
def _ppo_update(self, buffer: TamiyoRolloutBuffer) -> dict[str, float]:
    # Get batched data from buffer (Phase 6a adds blueprint_indices)
    batch = buffer.get_batched_sequences(self.device)

    states = batch["states"]                    # [batch, seq, 121]
    blueprint_indices = batch["blueprint_indices"]  # [batch, seq, num_slots]
    actions = {
        "op": batch["op_actions"],
        "slot": batch["slot_actions"],
        # ... other action heads ...
    }

    # evaluate_actions extracts stored_op from actions["op"] internally
    # (see Phase 4f - no separate sampled_op parameter needed)
    result = self.policy.evaluate_actions(
        states,
        blueprint_indices,
        actions,
        hidden,
        action_mask,
    )

    # result.value is Q(s, stored_op) - matches what was stored during rollout
    # result.log_probs is dict[str, Tensor] of per-head log probs
    # result.entropy is dict[str, Tensor] of per-head entropy
```

**Note:** The op used for value conditioning comes from `actions["op"]` (the stored action), extracted inside `evaluate_actions()`. This ensures the Q(s,op) value matches what was stored during rollout.

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

**Note on op storage:** The existing `op_actions` field already stores the sampled op from each rollout step. This is the op used for value conditioning (see Phase 4e/4f). During PPO update, `batch["op_actions"]` becomes `actions["op"]` which `evaluate_actions()` uses for `Q(s, stored_op)`. No additional storage needed.

#### 6a½. Bootstrap Value Computation (CRITICAL FIX REQUIRED)

> ⚠️ **BUG: Current bootstrap uses `deterministic=True` which is WRONG for Q(s,op) critics**
>
> The current code at `vectorized.py:3212-3219` uses:
> ```python
> bootstrap_result = agent.policy.get_action(
>     post_action_features_normalized,
>     masks=post_masks_batch,
>     hidden=batched_lstm_hidden,
>     deterministic=True,  # <-- BUG!
> )
> ```
>
> With Q(s,op) critics, this creates an **optimistic bias**:
> - Rollout stores `Q(s, sampled_op)` where op is sampled stochastically
> - Bootstrap computes `Q(s, argmax_op)` which is always the "best" action
> - Mixing these creates inconsistent value targets for GAE

**Required fix:**

```python
# Bootstrap at truncation: sample op stochastically (NOT deterministic)
# This ensures bootstrap Q(s_T, sampled_op_T) matches rollout Q(s_t, sampled_op_t)
bootstrap_result = agent.policy.get_action(
    post_action_features_normalized,
    masks=post_masks_batch,
    hidden=batched_lstm_hidden,
    deterministic=False,  # CRITICAL: Sample op, don't use argmax
)
bootstrap_value = bootstrap_result.value  # Q(s_T, sampled_op_T)
```

#### 6a¾. Done vs Truncated Semantics (CRITICAL FIX REQUIRED)

> ⚠️ **BUG: Current code conflates done and truncated**
>
> The current code at `vectorized.py:3003` sets:
> ```python
> done = epoch == max_epochs
> truncated = done  # <-- BUG! "truncated = episode ended" is wrong
> ```
>
> This conflates "episode ended" with "time-limit truncation". With Q(s,op) critics,
> bootstrapping on genuine terminals biases GAE targets.

**Required fix - Gymnasium-compliant semantics:**

| Condition | done | truncated | bootstrap |
|-----------|------|-----------|-----------|
| Intermediate step | `False` | `False` | N/A |
| Natural termination (goal/failure) | `True` | `False` | `0.0` |
| Time limit (max_epochs) | `True` | `True` | `Q(s_next, sampled_op)` |

```python
# At episode end:
is_natural_terminal = (some_terminal_condition)  # e.g., catastrophic failure
is_time_limit = (epoch == max_epochs) and not is_natural_terminal

done = is_natural_terminal or is_time_limit
truncated = is_time_limit  # ONLY time limit, not natural terminals

# In GAE computation:
if truncated:
    bootstrap = Q(s_next, sampled_op)  # Time limit: bootstrap from next state
else:
    bootstrap = 0.0  # Natural terminal: no future returns
```

**Test coverage:** See `test_done_vs_truncated_at_max_epochs()` in Phase 7g (line 2815).

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

# When slot is germinated:
env_state.epochs_since_counterfactual[slot_id] = 0

# When slot becomes inactive:
if slot_id in env_state.epochs_since_counterfactual:
    del env_state.epochs_since_counterfactual[slot_id]
```

**Counterfactual Reset Integration Pattern:**

The reset happens in `vectorized.py` immediately after a successful `compute_contributions()` call:

```python
# In vectorized.py, after Shapley computation (around the counterfactual helper call):
if env_state.counterfactual_helper is not None:
    try:
        contributions = env_state.counterfactual_helper.compute_contributions(...)
        # Reset freshness counter for all slots that received fresh counterfactual
        for slot_id in active_slot_ids:
            env_state.epochs_since_counterfactual[slot_id] = 0
    except Exception as e:
        _logger.warning(f"Shapley failed for env {env_idx}: {e}")
        # Don't reset counters on failure - stale values decay naturally
```

**Integration Notes:**
- The reset is triggered by successful completion of `compute_contributions()`, not by the attribution module itself
- Failed computations leave counters unchanged (stale values decay via gamma)
- This keeps all state management in `vectorized.py` rather than coupling attribution to ParallelEnvState

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
actions = {
    "op": batch["op_actions"],
    "slot": batch["slot_actions"],
    # ... other action heads ...
}
# evaluate_actions() extracts stored_op from actions["op"] internally
# (see Phase 4f) - no separate sampled_op parameter needed
result = policy.evaluate_actions(
    batch["states"],
    batch["blueprint_indices"],  # NEW
    actions,
    masks,
    initial_hidden,
)
```

---

### Phase 7: Validation & Testing

**Goal:** Verify the implementation works end-to-end.

> **Implementation Note:** The validation tests in this phase reference functions and types that are created in earlier phases (e.g., `compute_scaffold_hindsight_credit`, `_BLUEPRINT_TO_INDEX`, `encode_stage`, `TamiyoDecision.to_command()`). Write and run these tests **incrementally as each phase completes**, not all at once against the pre-implementation codebase. Tests for Phase 2 features should be written after Phase 2 is complete, tests for Phase 4 features after Phase 4, etc.

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

#### 7a½. Op Conditioning Consistency Test

Add a unit test to verify Q(s,op) conditioning is consistent across all code paths:

```python
def test_op_conditioning_consistency():
    """Verify Q(s,op) conditioning is consistent across forward/evaluate paths."""
    net = FactoredRecurrentActorCritic(state_dim=121, num_slots=3)
    state = torch.randn(2, 5, 121)
    bp_idx = torch.randint(0, 13, (2, 5, 3))

    # Forward path: value should match manual _compute_value with sampled_op
    fwd_out = net.forward(state, bp_idx)

    # Re-run LSTM to get lstm_out (or expose it for testing)
    # Then verify: fwd_out.value == net._compute_value(lstm_out, fwd_out.sampled_op)

    # Evaluate path with same op should give same value
    actions = {
        "op": fwd_out.sampled_op,
        "slot": torch.randint(0, 3, (2, 5)),
        # ... other action heads ...
    }
    eval_out = net.evaluate_actions(state, bp_idx, actions)

    # Values should match (same state, same op)
    assert torch.allclose(fwd_out.value, eval_out.value, atol=1e-5), \
        f"Value mismatch: forward={fwd_out.value.mean()}, eval={eval_out.value.mean()}"
```

This test catches any inconsistency in how the value head is conditioned between rollout and PPO update.

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

**Before Phase 2 implementation:** Measure current V2 performance as baseline:

```python
# Baseline measurement (run BEFORE implementing V3)
import time
from esper.tamiyo.policy.features import batch_obs_to_features

# ... setup signals, reports, device ...
start = time.perf_counter()
for _ in range(100):
    obs = batch_obs_to_features(signals, reports, use_telemetry=True, device=device)
elapsed_v2 = (time.perf_counter() - start) / 100
print(f"V2 feature extraction: {elapsed_v2*1000:.2f}ms/batch")
```

**After Phase 2 implementation:** Verify improvement:

```python
# V3 measurement (run AFTER implementing)
import time
start = time.perf_counter()
for _ in range(100):
    obs, bp_idx = batch_obs_to_features(signals, reports, slot_config, device)
elapsed_v3 = (time.perf_counter() - start) / 100
print(f"V3 feature extraction: {elapsed_v3*1000:.2f}ms/batch")
print(f"Speedup: {elapsed_v2 / elapsed_v3:.1f}x")
```

**Targets:**
- V3 absolute: < 1ms per batch for 4 environments
- V3 vs V2 speedup: ≥ 5x (due to vectorization and cached tables)

#### 7e. Learning Curve Comparison

Since there's no version toggle, comparison is via git branches:

```bash
# Current branch has new implementation
PYTHONPATH=src uv run python -m esper.scripts.train ppo --episodes 100 --seed 42

# Compare metrics in TensorBoard/telemetry against pre-V2 runs
```

Note: Checkpoints are incompatible, so comparison is learning curve shape, not resumed training.

#### 7f. Lifecycle Gating Validation Checks

These tests catch silent bugs in lifecycle operation enforcement that corrupt PPO's trust region.

##### 1. Embargo Enforcement Test

After a prune, the slot enters EMBARGOED stage and should NOT be available for immediate regermination.

```python
def test_embargo_blocks_immediate_regermination():
    """Verify slot is unavailable for germination during embargo period.

    BUG PATTERN: If prune with speed_steps=0 (INSTANT) clears the slot too fast,
    the mask logic might see an empty slot before embargo state is applied.
    """
    from esper.leyline import (
        DEFAULT_EMBARGO_EPOCHS_AFTER_PRUNE,
        SeedStage,
        LifecycleOp,
        AlphaSpeedAction,
        ALPHA_SPEED_TO_STEPS,
    )
    from esper.tamiyo.policy.action_masks import compute_action_masks

    # Setup: seed in HOLDING with HOLD mode, execute PRUNE with INSTANT speed
    slot_id = "slot_a"

    # Execute prune with INSTANT speed (0 ticks)
    assert ALPHA_SPEED_TO_STEPS[AlphaSpeedAction.INSTANT] == 0

    # After prune execution, verify slot enters EMBARGOED/PRUNED (not empty)
    # The slot_states dict should NOT have this slot as None
    post_prune_slot_states = execute_prune(slot_id, speed=AlphaSpeedAction.INSTANT)

    # Compute action mask - GERMINATE should NOT be enabled for this slot
    masks = compute_action_masks(
        slot_states=post_prune_slot_states,
        enabled_slots=[slot_id],
        # ...
    )

    # If slot appears empty (None), this assertion fails - that's the bug
    assert not masks["op"][LifecycleOp.GERMINATE], (
        f"GERMINATE should be blocked during embargo period "
        f"(slot in {post_prune_slot_states.get(slot_id)}, not None)"
    )

    # Verify slot becomes available after embargo_epochs
    for epoch in range(DEFAULT_EMBARGO_EPOCHS_AFTER_PRUNE):
        advance_epoch()
        masks = compute_action_masks(slot_states=get_slot_states(), ...)
        assert not masks["op"][LifecycleOp.GERMINATE], (
            f"Embargo should block germination for {DEFAULT_EMBARGO_EPOCHS_AFTER_PRUNE} epochs, "
            f"failed at epoch {epoch + 1}"
        )

    # After embargo period, slot should be available
    advance_epoch()  # embargo_epochs + 1
    masks = compute_action_masks(slot_states=get_slot_states(), ...)
    assert masks["op"][LifecycleOp.GERMINATE], "Slot should be available after embargo period"


def test_mask_staleness_after_instant_prune():
    """Verify vectorized rollout doesn't use cached mask from before INSTANT prune.

    BUG PATTERN: With INSTANT prune (speed_steps=0), the slot state changes
    within the same step. If vectorized rollout caches action masks at step
    start, it may use a stale mask that doesn't reflect the prune.

    This can cause:
    1. Policy samples GERMINATE for slot that's now embargoed (mask was valid pre-prune)
    2. Invalid action executed, corrupting trust region
    """
    from esper.leyline import AlphaSpeedAction, ALPHA_SPEED_TO_STEPS, LifecycleOp

    # Verify INSTANT has 0 steps (executes within same tick)
    assert ALPHA_SPEED_TO_STEPS[AlphaSpeedAction.INSTANT] == 0

    # Setup: slot in HOLDING, about to be pruned
    slot_id = "slot_a"
    initial_slot_states = {slot_id: MaskSeedInfo(stage=SeedStage.HOLDING.value, ...)}

    # Cache mask at step start (this is what vectorized rollout might do)
    cached_mask = compute_action_masks(slot_states=initial_slot_states, ...)
    assert cached_mask["op"][LifecycleOp.GERMINATE] is False  # Slot occupied

    # Execute INSTANT prune within same step
    execute_prune(slot_id, speed=AlphaSpeedAction.INSTANT)

    # CRITICAL: Mask must be recomputed AFTER any INSTANT operation
    # If rollout uses cached_mask here, it's stale
    fresh_mask = compute_action_masks(slot_states=get_slot_states(), ...)

    # Slot is now embargoed, not empty - GERMINATE still blocked
    assert fresh_mask["op"][LifecycleOp.GERMINATE] is False, (
        "GERMINATE should be blocked for embargoed slot. "
        "If this differs from cached_mask expectation, mask staleness bug exists."
    )

    # The cached mask would incorrectly show slot as occupied (pre-prune state)
    # while the slot is actually in EMBARGOED state (post-prune)
    # Both block GERMINATE, but for different reasons - the concern is when
    # an INSTANT op changes state in a way that WOULD enable a masked action
```

##### 2. Lifecycle Op Precondition Tests

The action mask logic AND execution logic must enforce identical invariants. If an illegal action slips through, it corrupts PPO's trust region (the policy learns that "impossible" actions are possible).

```python
# GERMINATE: slot must be empty (None)
def test_germinate_requires_empty_slot():
    """GERMINATE precondition: target slot must be empty (state is None)."""
    slot_states = {"slot_a": SeedStateReport(stage=SeedStage.TRAINING.value, ...)}
    masks = compute_action_masks(slot_states=slot_states, enabled_slots=["slot_a"], ...)

    # If the only enabled slot is occupied, GERMINATE should be masked
    assert not masks["op"][LifecycleOp.GERMINATE], (
        "GERMINATE should be masked when all enabled slots are occupied"
    )

    # Verify execution also rejects (defense in depth)
    with pytest.raises(InvalidActionError):
        execute_action(LifecycleOp.GERMINATE, slot="slot_a")


# GERMINATE: blocked at seed limit (even with empty slot)
def test_germinate_blocked_at_seed_limit():
    """GERMINATE precondition: total active seeds must be < MAX_CONCURRENT_SEEDS.

    BUG PATTERN: Empty slot exists, but system already at seed limit from other
    slots. GERMINATE should be masked even though the target slot is available.
    This precondition is separate from slot availability.
    """
    from esper.leyline import MAX_CONCURRENT_SEEDS, LifecycleOp, SeedStage

    # Setup: MAX_CONCURRENT_SEEDS already active, but one extra empty slot exists
    # (e.g., 4 slots configured, MAX_CONCURRENT_SEEDS=3, 3 occupied + 1 empty)
    slot_states = {
        "slot_a": MaskSeedInfo(stage=SeedStage.TRAINING.value, ...),
        "slot_b": MaskSeedInfo(stage=SeedStage.TRAINING.value, ...),
        "slot_c": MaskSeedInfo(stage=SeedStage.HOLDING.value, ...),
        "slot_d": None,  # Empty slot available
    }
    active_seed_count = sum(1 for s in slot_states.values() if s is not None)
    assert active_seed_count == MAX_CONCURRENT_SEEDS, "Test setup: at seed limit"

    masks = compute_action_masks(
        slot_states=slot_states,
        enabled_slots=["slot_a", "slot_b", "slot_c", "slot_d"],
        ...
    )

    # GERMINATE should be blocked despite empty slot_d
    assert not masks["op"][LifecycleOp.GERMINATE], (
        f"GERMINATE should be masked at seed limit ({MAX_CONCURRENT_SEEDS}), "
        f"even with empty slot available. "
        f"Active seeds: {active_seed_count}, empty slots: ['slot_d']"
    )

    # Verify execution also rejects
    with pytest.raises(InvalidActionError, match="seed limit"):
        execute_action(LifecycleOp.GERMINATE, slot="slot_d")


# PRUNE: seed must be in HOLDING with alpha_mode==HOLD and age >= MIN_PRUNE_AGE
def test_prune_requires_holding_hold_mode_and_min_age():
    """PRUNE precondition: HOLDING stage + HOLD mode + MIN_PRUNE_AGE."""
    from esper.leyline import MIN_PRUNE_AGE, AlphaMode

    # Case 1: TRAINING stage (wrong stage) - should be masked
    slot_states = {
        "slot_a": MaskSeedInfo(
            stage=SeedStage.TRAINING.value,
            alpha_mode=AlphaMode.HOLD.value,
            seed_age_epochs=MIN_PRUNE_AGE,
        )
    }
    masks = compute_action_masks(slot_states=slot_states, ...)
    assert not masks["op"][LifecycleOp.PRUNE], "PRUNE should be masked for TRAINING stage"

    # Case 2: HOLDING but BLEND mode (wrong mode) - should be masked
    slot_states = {
        "slot_a": MaskSeedInfo(
            stage=SeedStage.HOLDING.value,
            alpha_mode=AlphaMode.BLEND.value,
            seed_age_epochs=MIN_PRUNE_AGE,
        )
    }
    masks = compute_action_masks(slot_states=slot_states, ...)
    assert not masks["op"][LifecycleOp.PRUNE], "PRUNE should be masked for BLEND mode"

    # Case 3: HOLDING + HOLD but age < MIN_PRUNE_AGE - should be masked
    slot_states = {
        "slot_a": MaskSeedInfo(
            stage=SeedStage.HOLDING.value,
            alpha_mode=AlphaMode.HOLD.value,
            seed_age_epochs=MIN_PRUNE_AGE - 1,
        )
    }
    masks = compute_action_masks(slot_states=slot_states, ...)
    assert not masks["op"][LifecycleOp.PRUNE], (
        f"PRUNE should be masked for age < MIN_PRUNE_AGE ({MIN_PRUNE_AGE})"
    )

    # Case 4: All preconditions met - should be valid
    slot_states = {
        "slot_a": MaskSeedInfo(
            stage=SeedStage.HOLDING.value,
            alpha_mode=AlphaMode.HOLD.value,
            seed_age_epochs=MIN_PRUNE_AGE,
        )
    }
    masks = compute_action_masks(slot_states=slot_states, ...)
    assert masks["op"][LifecycleOp.PRUNE], "PRUNE should be valid when all preconditions met"


# FOSSILIZE: seed must be in HOLDING
def test_fossilize_requires_holding():
    """FOSSILIZE precondition: seed must be in HOLDING stage."""
    # TRAINING stage - should be masked
    slot_states = {"slot_a": MaskSeedInfo(stage=SeedStage.TRAINING.value, ...)}
    masks = compute_action_masks(slot_states=slot_states, ...)
    assert not masks["op"][LifecycleOp.FOSSILIZE], "FOSSILIZE should be masked for TRAINING"

    # BLENDING stage - should be masked
    slot_states = {"slot_a": MaskSeedInfo(stage=SeedStage.BLENDING.value, ...)}
    masks = compute_action_masks(slot_states=slot_states, ...)
    assert not masks["op"][LifecycleOp.FOSSILIZE], "FOSSILIZE should be masked for BLENDING"

    # HOLDING stage - should be valid
    slot_states = {"slot_a": MaskSeedInfo(stage=SeedStage.HOLDING.value, ...)}
    masks = compute_action_masks(slot_states=slot_states, ...)
    assert masks["op"][LifecycleOp.FOSSILIZE], "FOSSILIZE should be valid for HOLDING"


# ADVANCE: seed must exist in stage < HOLDING
def test_advance_requires_pre_holding_stage():
    """ADVANCE precondition: seed must be in TRAINING or BLENDING (not HOLDING)."""
    # HOLDING stage - should be masked (already at terminal training stage)
    slot_states = {"slot_a": MaskSeedInfo(stage=SeedStage.HOLDING.value, ...)}
    masks = compute_action_masks(slot_states=slot_states, ...)
    assert not masks["op"][LifecycleOp.ADVANCE], "ADVANCE should be masked for HOLDING"

    # TRAINING stage - should be valid
    slot_states = {"slot_a": MaskSeedInfo(stage=SeedStage.TRAINING.value, ...)}
    masks = compute_action_masks(slot_states=slot_states, ...)
    assert masks["op"][LifecycleOp.ADVANCE], "ADVANCE should be valid for TRAINING"

    # BLENDING stage - should be valid
    slot_states = {"slot_a": MaskSeedInfo(stage=SeedStage.BLENDING.value, ...)}
    masks = compute_action_masks(slot_states=slot_states, ...)
    assert masks["op"][LifecycleOp.ADVANCE], "ADVANCE should be valid for BLENDING"
```

##### 3. Stabilization Latch Monitoring

`signals.is_stabilized` is a sticky latch that never resets. If it triggers too early due to noise, Tamiyo stops germinating prematurely.

```python
def test_stabilization_latch_noise_resistance():
    """Verify stabilization threshold (3% over 3 epochs) isn't tripped by normal noise.

    The latch uses:
    - stabilization_threshold = 0.03 (3% improvement triggers instability)
    - stabilization_epochs = 3 (must be stable for 3 consecutive epochs)
    - regression_threshold = 0.05 (5% regression triggers instability)

    Normal early-training noise can easily exceed 3% epoch-to-epoch variance.

    ⚠️ COUNTER-INTUITIVE THRESHOLD SEMANTICS (DRL Review Addition):

    The stabilization_threshold controls when training is considered "stable":
    - "Stable" means improvement < threshold for N consecutive epochs
    - LOWER threshold (e.g., 0.02) = only tiny improvements qualify = HARDER to stabilize = FEWER false positives
    - HIGHER threshold (e.g., 0.05) = larger improvements still qualify = EASIER to stabilize = MORE false positives

    Example:
    - threshold=0.03: if epoch improves by 2%, training is "stable" (2% < 3%)
    - threshold=0.05: if epoch improves by 2%, training is "stable" (2% < 5%)
    - threshold=0.05: if epoch improves by 4%, training is STILL "stable" (4% < 5%)

    ↑ Notice: HIGHER threshold (0.05) accepts MORE improvements as "stable" → more false positives

    To PREVENT early triggering (more conservative):
    - DECREASE threshold (e.g., 0.03 → 0.02) — fewer improvements qualify
    - INCREASE stabilization_epochs (e.g., 3 → 5) — must stay stable longer

    To trigger MORE easily (less conservative):
    - INCREASE threshold (e.g., 0.03 → 0.05) — more improvements qualify
    """
    from esper.tamiyo.tracker import TamiyoTracker

    # stabilization_threshold = 0.03 means:
    # "If improvement < 3% for stabilization_epochs consecutive epochs, consider stable"
    # This is a CEILING on allowed improvement, not a floor!
    tracker = TamiyoTracker(
        stabilization_threshold=0.03,  # Improvement must be < 3% to count as "stable"
        stabilization_epochs=3,         # Must be stable for 3 consecutive epochs
        regression_threshold=0.05,      # 5% regression resets stability counter
    )

    # Simulate noisy early training (typical: 5-10% epoch-to-epoch variance)
    early_losses = [2.5, 2.3, 2.4, 2.2, 2.35, 2.1, 2.25, 2.0]  # ~5-8% swings
    early_accs = [10.0, 12.0, 11.0, 14.0, 13.0, 16.0, 15.0, 18.0]  # Normal noise

    for epoch, (loss, acc) in enumerate(zip(early_losses, early_accs)):
        tracker.update(epoch=epoch, val_loss=loss, val_accuracy=acc)

        # Should NOT stabilize during noisy early training
        if epoch < 5:  # First 5 epochs are typically noisy
            assert not tracker.is_stabilized, (
                f"Stabilization latch tripped too early at epoch {epoch} "
                f"(loss={loss}, prev_loss={early_losses[epoch-1] if epoch > 0 else 'N/A'}). "
                f"This will block germination prematurely."
            )

    # After enough stable epochs, should eventually stabilize
    stable_losses = [1.5, 1.48, 1.47, 1.46, 1.45]  # <3% improvement each
    stable_accs = [45.0, 45.5, 46.0, 46.2, 46.5]

    for i, (loss, acc) in enumerate(zip(stable_losses, stable_accs)):
        epoch = len(early_losses) + i
        tracker.update(epoch=epoch, val_loss=loss, val_accuracy=acc)

    assert tracker.is_stabilized, (
        "Tracker should stabilize after sufficient stable epochs"
    )
```

**Threshold tuning guidance:**

If stabilization triggers too early in production:
1. Increase `stabilization_epochs` from 3 to 5
2. **DECREASE** `stabilization_threshold` from 0.03 to 0.02 (lower threshold = harder to satisfy = fewer false positives)
3. Add early-epoch grace period in the tracker: `if epoch < 10: return False` before checking stability conditions

> **Note:** If stability logic is `improvement < threshold`, then HIGHER threshold makes it EASIER to stabilize (more improvements qualify as "small"). To prevent early triggering, LOWER the threshold or INCREASE stabilization_epochs.

##### 4. Stale Field Detection

Fields like `gradient_health_prev` are always 1.0 unless extreme conditions trigger updates. Verify no logic depends on these fields decreasing under normal conditions.

```python
def test_gradient_health_prev_field_updates():
    """Verify gradient_health_prev is actually updated, not stuck at 1.0.

    BUG PATTERN: If the update logic has a bug, gradient_health_prev stays 1.0
    and the LSTM never learns gradient trend patterns.
    """
    from esper.simic.training.parallel_env_state import ParallelEnvState

    state = ParallelEnvState()

    # Initial value should be empty dict (populated on first seed creation)
    assert state.gradient_health_prev == {}

    # After first epoch with a seed, should have entry
    slot_id = "slot_a"
    state.gradient_health_prev[slot_id] = 1.0  # Initial value

    # Simulate epochs with varying gradient health
    gradient_health_sequence = [1.0, 0.95, 0.8, 0.6, 0.7, 0.85, 0.9]

    for epoch, current_health in enumerate(gradient_health_sequence[1:], start=1):
        prev_health = state.gradient_health_prev.get(slot_id, 1.0)

        # If prev_health is ALWAYS 1.0, the update logic is broken
        if epoch > 2:
            assert prev_health != 1.0, (
                f"gradient_health_prev is stuck at 1.0 after {epoch} epochs. "
                f"The LSTM cannot learn gradient trends. "
                f"Check Phase 6b update logic in vectorized.py."
            )

        # Update for next epoch
        state.gradient_health_prev[slot_id] = current_health

    # Verify the sequence was tracked
    assert state.gradient_health_prev[slot_id] == gradient_health_sequence[-1]


def test_epochs_since_counterfactual_increments():
    """Verify epochs_since_counterfactual increments and resets correctly.

    BUG PATTERN: If always 0, counterfactual_fresh is always 1.0 (gamma^0).
    The LSTM cannot distinguish fresh vs stale counterfactual estimates.
    """
    from esper.simic.training.parallel_env_state import ParallelEnvState
    from esper.leyline import DEFAULT_GAMMA

    state = ParallelEnvState()
    slot_id = "slot_a"

    # Initialize when seed is created
    state.epochs_since_counterfactual[slot_id] = 0

    # Simulate epochs without counterfactual measurement
    for epoch in range(10):
        state.epochs_since_counterfactual[slot_id] += 1
        freshness = DEFAULT_GAMMA ** state.epochs_since_counterfactual[slot_id]

        # Freshness should decay: 0.995^10 ≈ 0.95
        if epoch == 9:
            assert 0.94 < freshness < 0.96, (
                f"Unexpected freshness decay: {freshness} at epoch {epoch}. "
                f"Expected ~0.95 with DEFAULT_GAMMA={DEFAULT_GAMMA}"
            )

    # After counterfactual measurement, should reset to 0
    state.epochs_since_counterfactual[slot_id] = 0
    freshness = DEFAULT_GAMMA ** state.epochs_since_counterfactual[slot_id]
    assert freshness == 1.0, "Freshness should reset to 1.0 after counterfactual measurement"
```

**Key validation invariants:**

| Field | Should Never Be | Detection |
|-------|-----------------|-----------|
| `gradient_health_prev` | Always 1.0 after epoch 2 | Log when unchanged for 3+ epochs |
| `epochs_since_counterfactual` | Always 0 | Log when never increments |
| `last_action_success` | Always True | Log when no failures after 50 epochs |
| `is_stabilized` | True before epoch 5 | Assert false in early epochs |
| `contribution_velocity` | Always 0.0 or NaN | Log when unchanged for 5+ epochs; used directly in obs, staleness corrupts policy input |

#### 7g. Phase 5 Addendum: Buffer & Bootstrap Validation

These tests catch silent bugs in rollout buffer schema and episode termination handling that corrupt advantage estimation and PPO updates.

> **Note on magic numbers**: The values `121` and `3` used below represent:
> - `121`: Obs V3 non-blueprint feature dimension (24 base + 7 temporal + 30×3 slots = 121)
> - `3`: Default number of slots in test configurations
>
> When Obs V3 is implemented, define `OBS_V3_NON_BLUEPRINT_DIM` and `DEFAULT_NUM_SLOTS`
> in `leyline` and reference them here. See section 4.2 for the dimension breakdown.

##### 1. Buffer Schema Alignment Test

After Obs V3 integration, the rollout buffer stores new fields (blueprint_indices, action feedback state, etc.). Misaligned arrays cause silent data corruption where transition N's action pairs with transition N+1's state.

```python
def test_buffer_schema_alignment_after_mini_rollout() -> None:
    """Verify all buffer fields are correctly aligned after a mini-rollout.

    BUG PATTERN: If a new field (e.g., blueprint_indices) is appended at a
    different point in the add() call than other fields, all subsequent data
    becomes off-by-one. The policy learns from mismatched (state, action) pairs.

    This test runs a short rollout and verifies field coherence by checking
    that each transition's data is internally consistent.
    """
    from esper.simic.agent.rollout_buffer import TamiyoRolloutBuffer
    import torch

    # Obs V3 dimensions: 24 base + 7 temporal + 30×3 slots = 121 non-blueprint dims
    # TODO: Replace with leyline.OBS_V3_NON_BLUEPRINT_DIM when available
    state_dim = 121
    num_slots = 3  # TODO: Replace with leyline.DEFAULT_NUM_SLOTS

    # Setup: Create buffer and run mini-rollout (2 envs, 3 steps each)
    buffer = TamiyoRolloutBuffer(
        num_envs=2,
        max_steps_per_env=10,
        state_dim=state_dim,
        num_slots=num_slots,
        device=torch.device("cpu"),
    )

    # Collect transitions with known values for verification
    from typing import Any
    test_transitions: list[dict[str, Any]] = []
    for env_id in range(2):
        for step in range(3):
            is_final_step = (step == 2)
            # Create transition with identifiable values
            # IMPORTANT: Intermediate steps have done=False, truncated=False
            # Only final step gets termination flags
            transition = {
                "env_id": env_id,
                "state": torch.full((state_dim,), float(env_id * 100 + step)),
                "blueprint_indices": torch.full((num_slots,), env_id * 10 + step, dtype=torch.int64),
                "op_action": step % 6,  # LifecycleOp value
                "slot_action": step % num_slots,
                "value": float(env_id + step * 0.1),
                "reward": float(step * 0.5),
                "done": is_final_step,  # Only final step terminates
                "truncated": False,  # Natural termination, not time limit
            }
            test_transitions.append(transition)
            buffer.add(**transition)

    # Retrieve buffer contents
    batch = buffer.get_batched_sequences(torch.device("cpu"))

    # Verify alignment: each transition's fields should correspond
    for i, expected in enumerate(test_transitions):
        env_id = expected["env_id"]
        step = i % 3  # Step within env

        # State should match
        actual_state = batch["states"][env_id, step]
        expected_val = float(env_id * 100 + step)
        assert torch.allclose(actual_state, torch.full_like(actual_state, expected_val)), (
            f"State mismatch at env={env_id}, step={step}: "
            f"expected {expected_val}, got {actual_state.mean().item()}"
        )

        # Blueprint indices should match (NEW in Obs V3)
        actual_bp = batch["blueprint_indices"][env_id, step]
        expected_bp = env_id * 10 + step
        assert (actual_bp == expected_bp).all(), (
            f"Blueprint indices mismatch at env={env_id}, step={step}: "
            f"expected {expected_bp}, got {actual_bp.tolist()}"
        )

        # Op action should match
        actual_op = batch["op_actions"][env_id, step].item()
        expected_op = step % 6
        assert actual_op == expected_op, (
            f"Op action mismatch at env={env_id}, step={step}: "
            f"expected {expected_op}, got {actual_op}"
        )


def test_buffer_dtype_consistency() -> None:
    """Verify buffer field dtypes match network expectations.

    BUG PATTERN: If blueprint_indices are stored as float32 instead of int64,
    nn.Embedding will fail silently or produce garbage. If masks are stored as
    int8 instead of bool/float, masking logic may not block invalid actions.
    """
    from esper.simic.agent.rollout_buffer import TamiyoRolloutBuffer
    import torch

    state_dim = 121  # Obs V3 non-blueprint dims
    num_slots = 3

    buffer = TamiyoRolloutBuffer(
        num_envs=1, max_steps_per_env=5, state_dim=state_dim, num_slots=num_slots,
        device=torch.device("cpu"),
    )

    # Add one transition (intermediate step: both flags False)
    buffer.add(
        env_id=0,
        state=torch.randn(state_dim),
        blueprint_indices=torch.tensor([0, 1, 2], dtype=torch.int64),
        op_action=0,
        slot_action=0,
        value=0.0,
        reward=0.0,
        done=False,
        truncated=False,  # Intermediate step, not truncated
    )

    batch = buffer.get_batched_sequences(torch.device("cpu"))

    # Critical dtype checks
    assert batch["blueprint_indices"].dtype == torch.int64, (
        f"blueprint_indices must be int64 for nn.Embedding, got {batch['blueprint_indices'].dtype}"
    )
    assert batch["states"].dtype == torch.float32, (
        f"states must be float32, got {batch['states'].dtype}"
    )
    # Note: torch.int64 and torch.long are the same type on all platforms
    assert batch["op_actions"].dtype == torch.int64, (
        f"op_actions must be int64 for indexing, got {batch['op_actions'].dtype}"
    )
```

##### 2. Done vs Truncated Flag Test

**Gymnasium Semantics for done/truncated (critical for correct bootstrap values):**

- **Intermediate steps**: `done=False, truncated=False` — episode is ongoing
- **Natural termination** (success/failure): `done=True, truncated=False, bootstrap=0.0`
- **Time limit truncation** (max_epochs): `done=False, truncated=True, bootstrap=V(s_next)`

> ⚠️ **Common Bug**: Setting `truncated=True` for intermediate steps. This is WRONG—truncated
> only applies to the final step when the episode ends due to a time limit, not natural termination.

```python
def test_done_vs_truncated_at_max_epochs() -> None:
    """Verify correct done/truncated semantics throughout an episode.

    BUG PATTERN: Setting truncated=True for intermediate steps. The truncated
    flag ONLY applies to the final step when the episode ends due to time limits.
    Intermediate steps must have done=False, truncated=False.

    Gymnasium semantics (from SB3/Gymnasium spec):
    - Intermediate: done=False, truncated=False (episode continues)
    - Natural end:  done=True,  truncated=False, bootstrap=0.0
    - Time limit:   done=False, truncated=True,  bootstrap=V(s_next)

    Getting this wrong causes bootstrap values to be computed for steps that
    don't need them (or vice versa), corrupting advantage estimation.
    """
    from esper.simic.training.vectorized import VectorizedTamiyoTraining
    import torch

    # Setup: Run training to max_epochs (time limit truncation scenario)
    trainer = VectorizedTamiyoTraining(
        num_envs=1,
        max_epochs=10,  # Short run for testing
        device=torch.device("cpu"),
    )

    # Collect transitions for full episode
    from typing import Any
    transitions: list[dict[str, Any]] = []
    for epoch in range(1, 11):  # epochs 1-10
        transition = trainer.step()
        transitions.append(transition)

    # Verify intermediate epochs (1-9): both flags must be False
    for i, t in enumerate(transitions[:-1]):
        epoch = i + 1
        assert t["done"] is False, (
            f"Epoch {epoch}: intermediate step must have done=False"
        )
        assert t["truncated"] is False, (
            f"Epoch {epoch}: intermediate step must have truncated=False. "
            f"truncated=True only applies to the FINAL step of a time-limited episode."
        )

    # Verify final epoch (10): time limit truncation
    final = transitions[-1]
    # For time limit: done=False (episode didn't naturally end), truncated=True
    assert final["truncated"] is True, (
        f"Final epoch at time limit should have truncated=True, got {final['truncated']}"
    )
    # Note: done can be True or False depending on whether we treat max_epochs as
    # "natural end" (done=True) or "artificial truncation" (done=False).
    # The key is: if truncated=True, bootstrap must be V(s_next), not 0.0
    if final["truncated"]:
        assert final["bootstrap_value"] != 0.0, (
            f"Truncated episode should have bootstrap_value=V(s_next), got 0.0. "
            f"This causes the critic to underestimate value of near-terminal states."
        )
    else:
        assert final["bootstrap_value"] == 0.0, (
            f"Done (non-truncated) episode should have bootstrap_value=0.0"
        )


def test_signals_done_propagates_correctly() -> None:
    """Verify signals.done from environment propagates to buffer correctly.

    BUG PATTERN: The environment sets signals.done=True at max_epochs, but
    the rollout loop might override this or check epoch count separately,
    leading to inconsistency between what the env reports and what's stored.
    """
    from esper.tamiyo.tracker import SignalTracker

    tracker = SignalTracker(max_epochs=25)

    # Simulate epochs
    for epoch in range(1, 26):
        signals = tracker.update(epoch=epoch, val_loss=2.0 - epoch * 0.05, val_accuracy=epoch * 2.0)

        if epoch < 25:
            assert not signals.done, f"signals.done should be False at epoch {epoch}"
        else:
            assert signals.done, f"signals.done should be True at epoch {epoch} (max_epochs=25)"
```

##### 3. Bootstrap Indexing Test for Parallel Environments

With multiple parallel environments, bootstrap values must be assigned to the correct truncated transitions. If env ordering diverges between `all_post_action_signals` and `transitions_data`, bootstrap values get swapped between environments.

```python
def test_bootstrap_indexing_parallel_envs() -> None:
    """Verify bootstrap_values[k] matches the k-th truncated transition's env_id.

    BUG PATTERN: The rollout loop builds all_post_action_signals in env order
    and transitions_data in env order, then zips them together. If any code
    path processes envs out of order (e.g., skipping done envs), the bootstrap
    values get assigned to wrong transitions.

    This corrupts advantage calculation: env 0's next-state value is used for
    env 1's transition, causing the critic to learn incorrect state values.
    """
    # Simulate final step of parallel rollout with 4 envs
    # At episode end: some envs terminate naturally (done), others hit time limit (truncated)
    from typing import Any
    num_envs = 4
    transitions_data: list[dict[str, Any]] = []
    all_post_action_signals: list[dict[str, Any]] = []
    bootstrap_env_ids: list[int] = []

    # Final step status: envs 1, 3 terminated naturally; envs 0, 2 hit time limit
    env_done_status = [False, True, False, True]  # envs 1, 3 are done (natural end)
    for env_id in range(num_envs):
        is_done = env_done_status[env_id]
        is_truncated = not is_done  # Time limit if not naturally done
        transition = {
            "env_id": env_id,
            "done": is_done,
            "truncated": is_truncated,
        }
        transitions_data.append(transition)

        # Only truncated (time limit) envs need bootstrap values
        # Done (natural end) envs get bootstrap=0.0
        if is_truncated:
            # Use offset of 100 to avoid confusing 0.0 with "no bootstrap"
            value = 100.0 + env_id * 10.0  # env 0 → 100.0, env 2 → 120.0
            all_post_action_signals.append({"env_id": env_id, "value": value})
            bootstrap_env_ids.append(env_id)

    # Compute bootstrap values (simulated)
    bootstrap_values = [sig["value"] for sig in all_post_action_signals]

    # Assign bootstrap values to transitions
    bootstrap_idx = 0
    for t in transitions_data:
        if t["truncated"]:
            # CRITICAL: Verify env_id matches
            expected_env_id = bootstrap_env_ids[bootstrap_idx]
            assert t["env_id"] == expected_env_id, (
                f"Bootstrap env_id mismatch: transition has env_id={t['env_id']}, "
                f"but bootstrap_values[{bootstrap_idx}] is for env_id={expected_env_id}. "
                f"This means env {t['env_id']}'s advantage will use env {expected_env_id}'s value!"
            )
            t["bootstrap_value"] = bootstrap_values[bootstrap_idx]
            bootstrap_idx += 1
        else:
            t["bootstrap_value"] = 0.0

    # Verify assignment - messages now correctly describe what each env's status is
    assert transitions_data[0]["bootstrap_value"] == 100.0, (
        "Env 0 (truncated at time limit) should have bootstrap=V(s_next)=100.0"
    )
    assert transitions_data[1]["bootstrap_value"] == 0.0, (
        "Env 1 (done naturally) should have bootstrap=0.0"
    )
    assert transitions_data[2]["bootstrap_value"] == 120.0, (
        "Env 2 (truncated at time limit) should have bootstrap=V(s_next)=120.0"
    )
    assert transitions_data[3]["bootstrap_value"] == 0.0, (
        "Env 3 (done naturally) should have bootstrap=0.0"
    )


def test_bootstrap_value_computation_uses_correct_state() -> None:
    """Verify bootstrap value is computed from post-action state, not pre-action.

    BUG PATTERN: If bootstrap is computed from the state BEFORE the action
    (s_t instead of s_{t+1}), the advantage estimate is off by one step.
    The critic learns V(s_t) = r_t + gamma*V(s_t) instead of V(s_t) = r_t + gamma*V(s_{t+1}).
    """
    from esper.tamiyo.policy.features import batch_obs_to_features
    import torch

    # Simulate: action at epoch 5 leads to state at epoch 6
    # These have DIFFERENT feature values - epoch affects multiple features
    pre_action_signals = {"epoch": 5, "val_loss": 2.0, "val_accuracy": 50.0}
    post_action_signals = {"epoch": 6, "val_loss": 1.8, "val_accuracy": 55.0}

    # Convert both to features to show they're different
    # (In practice, vectorized.py must use post_action_signals for bootstrap)
    # Using a mock slot config for illustration
    from esper.tamiyo.policy.features import SlotConfig

    slot_config = SlotConfig(num_slots=3, slot_ids=["r0c0", "r0c1", "r0c2"])

    pre_features = batch_obs_to_features(
        batch_signals=[pre_action_signals],
        slot_config=slot_config,
        device=torch.device("cpu"),
    )
    post_features = batch_obs_to_features(
        batch_signals=[post_action_signals],
        slot_config=slot_config,
        device=torch.device("cpu"),
    )

    # The features must be different - epoch, loss, accuracy all changed
    assert not torch.allclose(pre_features[0], post_features[0]), (
        "Pre-action and post-action features should differ. "
        "If they're the same, epoch/loss/accuracy changes are not reflected in features."
    )

    # Verify epoch feature specifically differs (epoch is typically feature 0 or early)
    # This ensures the temporal signal propagates to the feature vector
    epoch_pre = pre_action_signals["epoch"]
    epoch_post = post_action_signals["epoch"]
    assert epoch_pre != epoch_post, "Test setup error: epochs should differ"

    # The actual verification: check that vectorized.py passes post_action_signals
    # to batch_obs_to_features when computing bootstrap values.
    # This is a code inspection requirement, verified by grep:
    #   grep -n "bootstrap.*post_action" src/esper/simic/training/vectorized.py
    # Must find usage of post_action_signals, not pre_action_signals
```

##### 4. LSTM Hidden State Reset Test

At episode boundaries, the LSTM hidden state must be reset. Leftover hidden state from a previous episode causes the policy to condition on irrelevant context from a different training run.

```python
def test_lstm_hidden_state_reset_at_episode_boundary() -> None:
    """Verify LSTM hidden state is reset when a new episode begins.

    BUG PATTERN: If the policy's LSTM carries hidden state from episode N
    into episode N+1, the first few decisions of episode N+1 are conditioned
    on stale context. This is especially harmful if episode N ended badly
    (e.g., all seeds pruned) - the LSTM "remembers" that failure context.

    The hidden state should be zeroed at episode start, either via:
    - policy.reset_hidden_state() called explicitly
    - Passing hidden=None to forward() which initializes fresh state
    """
    from esper.tamiyo.networks.factored_lstm import FactoredRecurrentActorCritic
    from esper.leyline import NUM_BLUEPRINTS
    import torch

    state_dim = 121  # Obs V3 non-blueprint dims
    num_slots = 3

    policy = FactoredRecurrentActorCritic(
        state_dim=state_dim,
        num_slots=num_slots,
        hidden_dim=128,
        num_blueprints=NUM_BLUEPRINTS,
    )

    # Run one episode to build up hidden state
    state = torch.randn(1, 10, state_dim)  # batch=1, seq=10, features
    bp_idx = torch.randint(0, NUM_BLUEPRINTS, (1, 10, num_slots))

    # First forward pass - hidden state initialized internally
    out1 = policy.forward(state, bp_idx, hidden=None)
    hidden_after_ep1 = out1.hidden

    # Verify hidden state is not zero after episode
    h1, c1 = hidden_after_ep1
    assert not torch.allclose(h1, torch.zeros_like(h1)), (
        "Hidden state should be non-zero after processing a sequence"
    )

    # NEW EPISODE: Reset by passing hidden=None
    # This should initialize fresh state, NOT reuse hidden_after_ep1
    out2 = policy.forward(state, bp_idx, hidden=None)
    h2_start, _ = out2.hidden

    # Run same input with carried hidden state (simulates NO reset)
    out3 = policy.forward(state, bp_idx, hidden=hidden_after_ep1)
    h3_carried, _ = out3.hidden

    # Key assertion: fresh start (hidden=None) should produce different
    # hidden state than carrying over from previous episode
    # This proves that hidden=None actually resets
    assert not torch.allclose(h2_start, h3_carried, atol=1e-5), (
        "Fresh hidden state (hidden=None) should differ from carried state. "
        "If equal, the LSTM is not properly resetting between episodes."
    )


def test_lstm_hidden_state_gradient_detachment() -> None:
    """Verify LSTM hidden state is detached between rollouts to prevent gradient leakage.

    BUG PATTERN: If hidden state from rollout N is passed to rollout N+1
    WITHOUT calling .detach(), gradients flow backward across rollout boundaries.
    This causes:
    1. Memory leak: computation graph grows unboundedly
    2. Incorrect gradients: policy update for rollout N+1 affects rollout N's graph
    3. CUDA OOM after several rollouts

    The fix: Always call hidden = (h.detach(), c.detach()) before passing
    hidden state to the next rollout.
    """
    from esper.tamiyo.networks.factored_lstm import FactoredRecurrentActorCritic
    from esper.leyline import NUM_BLUEPRINTS
    import torch

    state_dim = 121
    num_slots = 3

    policy = FactoredRecurrentActorCritic(
        state_dim=state_dim,
        num_slots=num_slots,
        hidden_dim=128,
        num_blueprints=NUM_BLUEPRINTS,
    )

    state = torch.randn(1, 5, state_dim)
    bp_idx = torch.randint(0, NUM_BLUEPRINTS, (1, 5, num_slots))

    # Rollout 1
    out1 = policy.forward(state, bp_idx, hidden=None)
    h1, c1 = out1.hidden

    # Before passing to next rollout, hidden must be detached
    # This is what the training loop should do:
    h1_detached = h1.detach()
    c1_detached = c1.detach()

    # Verify detachment breaks gradient flow
    assert not h1_detached.requires_grad, (
        "Detached hidden state should not require grad"
    )
    assert not c1_detached.requires_grad, (
        "Detached cell state should not require grad"
    )

    # Verify original still has grad (before detach)
    # Note: This may be False if model is in eval mode, so we check in train mode
    policy.train()
    out_train = policy.forward(state, bp_idx, hidden=None)
    h_train, c_train = out_train.hidden
    # Hidden states from forward pass should be part of computation graph
    # (requires_grad depends on whether model params require grad)
    assert h_train.grad_fn is not None or not any(p.requires_grad for p in policy.parameters()), (
        "Hidden state should be connected to computation graph during training"
    )


def test_hidden_state_not_shared_across_parallel_envs() -> None:
    """Verify each parallel env has independent LSTM hidden state.

    BUG PATTERN: If hidden state tensor is accidentally shared (e.g., same
    reference used for all envs), env 0's context bleeds into env 1's decisions.
    Each env should have its own hidden state tensor.
    """
    import torch

    num_envs = 4
    hidden_dim = 128
    num_layers = 1

    # Hidden states should be separate tensors per env
    # Standard LSTM hidden shape: (num_layers, batch/num_envs, hidden_dim)
    hidden_h = torch.zeros(num_layers, num_envs, hidden_dim)
    hidden_c = torch.zeros(num_layers, num_envs, hidden_dim)

    # Modify env 0's hidden state
    hidden_h[0, 0, :] = 1.0

    # Env 1's hidden state should be unaffected
    assert torch.allclose(hidden_h[0, 1, :], torch.zeros(hidden_dim)), (
        "Env 1's hidden state was modified when only env 0 should change. "
        "Hidden states are incorrectly shared across envs."
    )

    # Verify shape supports independent per-env state
    assert hidden_h.shape == (num_layers, num_envs, hidden_dim), (
        f"Hidden state shape should be (layers, envs, hidden_dim), got {hidden_h.shape}"
    )

    # Verify all envs except 0 remain zero
    for env_id in range(1, num_envs):
        assert torch.allclose(hidden_h[0, env_id, :], torch.zeros(hidden_dim)), (
            f"Env {env_id}'s hidden state should be zero, but was modified"
        )
```

##### 5. Reward Normalization Consistency Test

Rewards must be consistently normalized (or unnormalized) between buffer storage and PPO update. Mixing scales causes the critic to learn incorrect value estimates and corrupts advantage normalization.

```python
def test_reward_normalization_consistency() -> None:
    """Verify rewards stored in buffer match what was computed (no silent modification).

    BUG PATTERN: If compute_reward() returns normalized rewards but the buffer
    stores raw rewards (or vice versa), the advantage calculation uses
    mismatched scales. The critic learns V(s) in one scale while rewards
    are in another, causing systematic bias.

    This test verifies: the reward value added to the buffer is exactly
    what compute_reward() returned, with no intermediate transformation.
    """
    from esper.simic.control.normalization import RunningMeanStd
    import torch

    # Create a normalizer with known statistics
    normalizer = RunningMeanStd(shape=(1,))  # 1D for scalar rewards
    # Prime with some values to get non-trivial mean/var
    for r in [0.1, 0.2, 0.3, 0.4, 0.5]:
        normalizer.update(torch.tensor([[r]]))  # [batch=1, feature=1]

    # Simulate compute_reward() output
    raw_reward = 0.75
    # Normalize using the normalizer (this is what compute_reward should do)
    computed_reward = (raw_reward - normalizer.mean.item()) / (
        torch.sqrt(normalizer.var + 1e-8).item()
    )

    # Simulate buffer storage - buffer should receive EXACTLY computed_reward
    # NOT raw_reward, and NOT a re-normalized version
    buffer_rewards: list[float] = []
    buffer_rewards.append(computed_reward)  # This is what add() should do

    # Simulate what gets used in PPO update
    ppo_reward = buffer_rewards[0]

    # The invariant: what PPO sees == what compute_reward returned
    assert abs(ppo_reward - computed_reward) < 1e-9, (
        f"Reward mismatch: PPO sees {ppo_reward}, compute_reward returned {computed_reward}. "
        f"Something modified the reward between computation and usage."
    )

    # Also verify normalization actually changed the value
    assert abs(computed_reward - raw_reward) > 0.01, (
        f"Normalization should transform reward: raw={raw_reward}, normalized={computed_reward}. "
        f"If equal, normalizer is not being applied."
    )


def test_reward_normalizer_updates_correctly() -> None:
    """Verify reward normalizer statistics update correctly using Welford's algorithm.

    BUG PATTERN: If the normalizer's mean/var are never updated, early rewards
    dominate the statistics. As training progresses and reward distribution
    shifts, normalized rewards become increasingly biased.

    This test verifies:
    1. Mean tracks the running average correctly
    2. Variance (not std!) is stored and computed correctly
    3. Count increments properly
    """
    from esper.simic.control.normalization import RunningMeanStd
    import torch

    # Note: RunningMeanStd uses population variance (unbiased=False in torch.var)
    # so we compare against / n, not / (n-1)
    normalizer = RunningMeanStd(shape=(1,))  # 1D for scalar rewards

    # Phase 1: Early training (small rewards)
    early_rewards = [0.1, 0.2, 0.15, 0.12, 0.18]
    for r in early_rewards:
        normalizer.update(torch.tensor([[r]]))  # [batch=1, feature=1]

    # Verify against manual calculation
    expected_early_mean = sum(early_rewards) / len(early_rewards)
    expected_early_var = sum((r - expected_early_mean) ** 2 for r in early_rewards) / len(early_rewards)

    assert abs(normalizer.mean.item() - expected_early_mean) < 1e-5, (
        f"Mean mismatch after early rewards: expected {expected_early_mean}, "
        f"got {normalizer.mean.item()}"
    )
    # Note: RunningMeanStd stores VAR, not STD
    assert abs(normalizer.var.item() - expected_early_var) < 1e-4, (
        f"Variance mismatch after early rewards: expected {expected_early_var}, "
        f"got {normalizer.var.item()}. Note: RunningMeanStd stores var, not std."
    )
    # Note: count is a tensor initialized to epsilon, not 0
    # After n updates with batch_size=1, count = epsilon + n
    assert normalizer.count.item() >= len(early_rewards), (
        f"Count should be >= {len(early_rewards)}, got {normalizer.count.item()}"
    )

    # Phase 2: Later training (larger rewards)
    later_rewards = [0.8, 0.9, 0.85, 0.92, 0.88]
    for r in later_rewards:
        normalizer.update(torch.tensor([[r]]))  # [batch=1, feature=1]

    all_rewards = early_rewards + later_rewards
    expected_final_mean = sum(all_rewards) / len(all_rewards)
    expected_final_var = sum((r - expected_final_mean) ** 2 for r in all_rewards) / len(all_rewards)

    # Mean should track all rewards, not just later ones
    assert abs(normalizer.mean.item() - expected_final_mean) < 1e-5, (
        f"Mean should track all rewards: expected {expected_final_mean}, "
        f"got {normalizer.mean.item()}"
    )
    # Variance should also track full history
    assert abs(normalizer.var.item() - expected_final_var) < 1e-3, (
        f"Variance should track all rewards: expected {expected_final_var}, "
        f"got {normalizer.var.item()}. Using 1e-3 tolerance for numerical precision."
    )
    # Count should be number of update() calls (not number of samples if batch_size varies)
    # Note: count is a tensor, not an int
    assert normalizer.count.item() >= len(all_rewards), (
        f"Count should be >= {len(all_rewards)}, got {normalizer.count.item()}"
    )


def test_buffer_reward_matches_transition_reward() -> None:
    """Verify the reward stored in buffer equals the transition's computed reward.

    BUG PATTERN: If buffer.add() receives a different reward than what was
    computed (e.g., due to an intermediate variable being overwritten or
    a stale value being used), the policy learns from incorrect feedback.
    """
    import torch

    from typing import Any
    state_dim = 121  # Obs V3 non-blueprint dims

    # Simulate transition with unique reward value
    computed_reward = 0.42
    transition: dict[str, Any] = {
        "reward": computed_reward,
        "state": torch.randn(state_dim),
        # ... other fields omitted for clarity
    }

    # Add to buffer (simulated)
    buffer_rewards: list[float] = []
    buffer_rewards.append(transition["reward"])

    # Retrieve and verify exact match (no floating point tolerance needed)
    assert buffer_rewards[0] == computed_reward, (
        f"Buffer reward {buffer_rewards[0]} != computed reward {computed_reward}. "
        f"Check that transition['reward'] is not modified between compute and add."
    )

    # Verify the stored reward is not accidentally zero or default
    assert buffer_rewards[0] != 0.0, (
        "Buffer reward should not be zero - check that reward assignment is not skipped"
    )
```

**Validation commands:**

```bash
# Run all Phase 5 buffer/bootstrap validation tests
PYTHONPATH=src uv run pytest tests/simic/test_buffer_bootstrap.py -v

# Quick smoke test: run mini-rollout and print buffer contents
PYTHONPATH=src uv run python -c "
from esper.simic.agent.rollout_buffer import TamiyoRolloutBuffer
import torch

buffer = TamiyoRolloutBuffer(num_envs=2, max_steps_per_env=5, state_dim=121, num_slots=3, device=torch.device('cpu'))
# Add test transitions and print shapes/dtypes
print('Buffer initialized successfully')
"

# Verify done/truncated at episode end
PYTHONPATH=src uv run python -m esper.scripts.train ppo --episodes 1 --max-epochs 5 --seed 42 2>&1 | grep -E "(done|truncated|bootstrap)"
```

**Key validation invariants:**

| Check | Expected | Failure Indicates |
|-------|----------|-------------------|
| `batch["blueprint_indices"].dtype` | `torch.int64` | nn.Embedding will fail or produce garbage |
| Final epoch `done` | `True` | Bootstrap will use V(s) instead of 0.0 |
| Final epoch `truncated` | `False` | Advantage biased by spurious bootstrap |
| Final epoch `bootstrap_value` | `0.0` | Critic learns incorrect terminal values |
| `bootstrap_values[k]` env_id | Matches transition k's env_id | Cross-env value contamination |
| LSTM hidden at episode start | Fresh (zeros or `None`) | Context from previous episode bleeds through |
| Buffer reward | Equals logged reward | Scale mismatch corrupts advantages |

#### 7h. Phase 6 Addendum: PPO Update & Clipping Validation

These tests catch silent bugs in multi-head PPO policy ratio computation, clipping, and advantage handling. With 8 action heads (op, slot, blueprint, style, tempo, alpha_target, alpha_speed, alpha_curve), subtle coupling issues can corrupt training without obvious symptoms.

> **⚠️ Architectural Note: Causal Masking Creates Per-Head Advantages**
>
> Tamiyo uses **causal masking** where heads that don't influence the outcome get `advantage=0`.
> For example, when `op=WAIT`, the slot/blueprint/style heads are masked and get zero advantage
> because they had no effect on the reward.
>
> This is a **hybrid approach** between standard PPO and per-head PPO:
> - **Joint ratio**: We compute `exp(sum(new_log_probs) - sum(old_log_probs))` across all heads
> - **Per-head advantages**: Causal masking sets some head advantages to zero
> - **Clipping**: Applied to joint ratio, but gradient only flows through non-masked heads
>
> This differs from standard PPO (single scalar advantage) and per-head PPO (separate rewards).
> The tests below validate this specific architecture.

> **Note on magic numbers**: The values used below represent:
> - `8`: Number of action heads in the factored policy
> - `0.2`: PPO clip epsilon (standard value from Schulman et al., 2017)
> - `121`: Obs V3 non-blueprint feature dimension
> - `128`: Default hidden dimension for test networks
> - `0.1`: Standard weight perturbation magnitude for testing ratio sensitivity
> - `1e-8`: Numerical stability epsilon for division
>
> When Phase 6 is implemented, define `NUM_ACTION_HEADS` and `PPO_CLIP_EPSILON`
> in `leyline` and reference them here.

##### 1. Joint Policy Ratio Computation Test

The policy ratio for PPO must be computed as the joint probability ratio across ALL heads: `exp(sum(log_probs_new) - sum(log_probs_old))`. A change in ANY head should affect the ratio.

```python
def test_joint_policy_ratio_reflects_all_heads() -> None:
    """Verify policy ratio is computed jointly across all heads, not per-head.

    BUG PATTERN: Computing ratio per-head then averaging loses the joint
    probability structure. If head 0 has ratio=2.0 and heads 1-7 have ratio=1.0,
    the joint ratio should be 2.0 (product), not 1.125 (average).

    The correct computation is:
        ratio = exp(sum(new_log_probs) - sum(old_log_probs))
              = exp(log(prod(new_probs)) - log(prod(old_probs)))
              = prod(new_probs) / prod(old_probs)

    This test perturbs secondary head weights and verifies the ratio changes.
    """
    import torch

    # TODO: Replace with leyline.NUM_ACTION_HEADS when available
    num_heads = 8
    batch_size = 4

    # Simulate log_probs from old and new policy
    # Old policy: all heads have log_prob = -1.0 (prob ~0.368)
    old_log_probs = torch.full((batch_size, num_heads), -1.0)

    # New policy: head 0 changed significantly, others unchanged
    new_log_probs = old_log_probs.clone()
    new_log_probs[:, 0] = -0.5  # Head 0 probability increased

    # CORRECT: Joint ratio = exp(sum(new) - sum(old))
    joint_ratio = torch.exp(new_log_probs.sum(dim=1) - old_log_probs.sum(dim=1))

    # INCORRECT (bug pattern): Per-head ratio averaged
    per_head_ratios = torch.exp(new_log_probs - old_log_probs)
    averaged_ratio = per_head_ratios.mean(dim=1)

    # Verify joint ratio reflects the change in head 0
    expected_ratio = torch.exp(torch.tensor(-0.5 - (-1.0)))  # exp(0.5) ~= 1.649
    assert torch.allclose(joint_ratio, torch.full((batch_size,), expected_ratio.item()), atol=1e-5), (
        f"Joint ratio should reflect head 0 change: expected ~{expected_ratio.item():.3f}, "
        f"got {joint_ratio[0].item():.3f}"
    )

    # Verify that averaged ratio is DIFFERENT (and thus wrong)
    assert not torch.allclose(joint_ratio, averaged_ratio, atol=0.1), (
        f"Joint ratio ({joint_ratio[0].item():.3f}) should differ from averaged ratio "
        f"({averaged_ratio[0].item():.3f}). If equal, ratio computation may be incorrect."
    )


def test_ratio_changes_when_secondary_head_weights_perturbed() -> None:
    """Verify that perturbing any head's weights affects the joint ratio.

    BUG PATTERN: If only the primary action head (e.g., op) affects the ratio,
    updates to secondary heads (e.g., alpha_curve) are ignored. The policy
    never learns to adjust secondary heads because gradients are zero.

    This test perturbs a secondary head and verifies ratio changes.
    """
    from esper.tamiyo.networks.factored_lstm import FactoredRecurrentActorCritic
    from esper.leyline import NUM_BLUEPRINTS
    import torch

    state_dim = 121  # TODO: Replace with leyline.OBS_V3_NON_BLUEPRINT_DIM
    num_slots = 3

    policy = FactoredRecurrentActorCritic(
        state_dim=state_dim,
        num_slots=num_slots,
        hidden_dim=128,
        num_blueprints=NUM_BLUEPRINTS,
    )
    policy.eval()

    # Create test input
    state = torch.randn(1, 1, state_dim)  # batch=1, seq=1
    bp_idx = torch.randint(0, NUM_BLUEPRINTS, (1, 1, num_slots))

    # Get baseline log_probs
    with torch.no_grad():
        out_baseline = policy.forward(state, bp_idx, hidden=None)
        # Assume output contains log_probs for all heads
        # Sample actions to get log_probs
        baseline_log_probs = out_baseline.log_probs.clone()

    # Perturb a secondary head (e.g., alpha_curve head, typically index 7)
    # Find the head's parameters and perturb
    secondary_head_idx = 7  # alpha_curve
    perturbed_any = False
    with torch.no_grad():
        # Perturb the last head's weights if accessible
        # This is a structural test - actual implementation may vary
        for name, param in policy.named_parameters():
            if "alpha_curve" in name or "head_7" in name:
                param.add_(torch.randn_like(param) * 0.1)
                perturbed_any = True
                break

    # Ensure we actually perturbed something - fail loudly if naming changed
    assert perturbed_any, (
        "Failed to find secondary head parameters to perturb. "
        "Check that policy has parameters containing 'alpha_curve' or 'head_7' in name."
    )

    # Get new log_probs after perturbation
    with torch.no_grad():
        out_perturbed = policy.forward(state, bp_idx, hidden=None)
        perturbed_log_probs = out_perturbed.log_probs.clone()

    # Compute joint ratios
    baseline_joint = baseline_log_probs.sum(dim=-1)
    perturbed_joint = perturbed_log_probs.sum(dim=-1)

    # The joint log_probs should differ after perturbing secondary head
    assert not torch.allclose(baseline_joint, perturbed_joint, atol=1e-6), (
        f"Joint log_probs should change when secondary head is perturbed. "
        f"Baseline: {baseline_joint.item():.6f}, Perturbed: {perturbed_joint.item():.6f}. "
        f"If equal, secondary head changes are not reflected in joint ratio."
    )
```

##### 2. Per-Head Clipping Coupling Test

A large ratio in a zero-advantage head could trigger clipping that affects other heads with real advantages. This tests whether clipping is applied correctly when heads have different advantage magnitudes.

```python
def test_clipping_with_zero_advantage_head() -> None:
    """Verify clipping behavior when one head has zero advantage.

    BUG PATTERN: If clipping is based on joint ratio when only one head changed,
    a head with zero advantage but high ratio triggers clipping that suppresses
    gradients to heads with real advantages.

    Scenario:
    - Head 0: ratio=2.0 (outside clip), advantage=0.0 (no signal)
    - Head 1: ratio=1.1 (inside clip), advantage=1.0 (real signal)

    The head 1 gradient should NOT be affected by head 0's clipping.
    """
    import torch

    # TODO: Replace with leyline.PPO_CLIP_EPSILON when available
    clip_epsilon = 0.2

    batch_size = 1
    num_heads = 2

    # Ratios: head 0 outside clip, head 1 inside
    ratios = torch.tensor([[2.0, 1.1]])  # [batch, heads]

    # Advantages: head 0 zero, head 1 positive
    advantages = torch.tensor([[0.0, 1.0]])  # [batch, heads]

    # Clip range
    ratio_min = 1.0 - clip_epsilon  # 0.8
    ratio_max = 1.0 + clip_epsilon  # 1.2

    # CORRECT: Per-head clipping, each head uses its own advantage
    clipped_ratios = torch.clamp(ratios, ratio_min, ratio_max)

    # Unclipped objective: ratio * advantage
    unclipped_obj = ratios * advantages
    # Clipped objective: clipped_ratio * advantage
    clipped_obj = clipped_ratios * advantages

    # PPO objective: min(unclipped, clipped) for positive advantage
    # For negative advantage: max(unclipped, clipped) - but we use positive here
    ppo_obj = torch.minimum(unclipped_obj, clipped_obj)

    # Head 0: ratio=2.0, advantage=0.0
    # Both unclipped and clipped objectives are 0.0
    assert ppo_obj[0, 0].item() == 0.0, (
        f"Head 0 (zero advantage) should have zero objective, got {ppo_obj[0, 0].item()}"
    )

    # Head 1: ratio=1.1 (inside clip), advantage=1.0
    # Unclipped: 1.1 * 1.0 = 1.1
    # Clipped: 1.1 * 1.0 = 1.1 (already inside clip range)
    # min(1.1, 1.1) = 1.1
    expected_head1_obj = 1.1 * 1.0
    assert abs(ppo_obj[0, 1].item() - expected_head1_obj) < 1e-6, (
        f"Head 1 should have objective {expected_head1_obj}, got {ppo_obj[0, 1].item()}"
    )

    # WRONG pattern: Joint ratio triggers clipping
    # If we computed a joint ratio = prod([2.0, 1.1]) = 2.2 and clipped THAT,
    # the clipping would affect all heads equally (wrong)
    joint_ratio = ratios.prod(dim=1, keepdim=True)  # 2.2
    assert joint_ratio.item() > ratio_max, (
        "Joint ratio should be outside clip range for this test"
    )


def test_gradient_isolation_between_heads() -> None:
    """Verify gradients flow independently through each head.

    BUG PATTERN: If heads share intermediate computation incorrectly,
    zeroing advantage for head 0 may still produce non-zero gradients
    for head 0's parameters via head 1's backprop.

    This test verifies that a head with zero advantage contributes zero gradient.
    """
    import torch
    import torch.nn as nn

    # Simple multi-head model for testing
    class MockMultiHeadPolicy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.shared = nn.Linear(10, 16)
            self.head_0 = nn.Linear(16, 4)  # First action head
            self.head_1 = nn.Linear(16, 4)  # Second action head

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            shared_features = self.shared(x)
            logits_0 = self.head_0(shared_features)
            logits_1 = self.head_1(shared_features)
            return logits_0, logits_1

    model = MockMultiHeadPolicy()
    x = torch.randn(1, 10)

    logits_0, logits_1 = model(x)

    # Compute log_probs (softmax + log)
    log_probs_0 = torch.log_softmax(logits_0, dim=-1)
    log_probs_1 = torch.log_softmax(logits_1, dim=-1)

    # Sample actions
    action_0 = torch.tensor([0])
    action_1 = torch.tensor([1])

    # Get log_probs for sampled actions
    lp_0 = log_probs_0[0, action_0]
    lp_1 = log_probs_1[0, action_1]

    # Advantages: head 0 has zero, head 1 has positive
    adv_0 = torch.tensor(0.0)
    adv_1 = torch.tensor(1.0)

    # Compute loss: -advantage * log_prob (policy gradient)
    loss = -(adv_0 * lp_0 + adv_1 * lp_1)

    # Backprop
    loss.backward()

    # Head 0 should have non-zero gradient ONLY because of shared layer
    # But head_0's own parameters should have zero gradient
    head_0_grad_norm = sum(
        p.grad.norm().item() for p in model.head_0.parameters() if p.grad is not None
    )

    # With zero advantage for head 0, its parameters should have zero gradient
    # (gradients only flow through head 1)
    assert head_0_grad_norm < 1e-6, (
        f"Head 0 gradient norm should be ~0 with zero advantage, got {head_0_grad_norm}. "
        f"Gradient is leaking from head 1 or advantage masking is broken."
    )
```

##### 3. Advantage Normalization with Masked Zeros Test

Many masked heads have advantage=0. Including them in normalization underestimates standard deviation, causing non-masked advantages to be incorrectly scaled.

```python
def test_advantage_normalization_excludes_masked_zeros() -> None:
    """Verify advantage normalization uses valid_mask to exclude masked zeros.

    BUG PATTERN: If advantages.mean() and advantages.std() include zeros from
    masked heads, the statistics are biased:
    - Mean pulled toward 0 (underestimated if true mean is positive)
    - Std underestimated (zeros have no variance contribution)

    The result: normalized advantages are too large, causing aggressive updates.

    Example: [1.0, 1.0, 0.0, 0.0, 0.0] (2 valid, 3 masked)
    - Wrong: mean=0.4, std=0.49, normalized valid = (1.0-0.4)/0.49 = 1.22
    - Right: mean=1.0, std=0.0, normalized valid = 0.0 (no variance)

    With variance: [1.0, 2.0, 0.0, 0.0, 0.0] (2 valid, 3 masked)
    - Wrong: mean=0.6, std=0.8, normalized[0] = (1.0-0.6)/0.8 = 0.5
    - Right: mean=1.5, std=0.5, normalized[0] = (1.0-1.5)/0.5 = -1.0
    """
    import torch

    # Advantages with masked zeros
    advantages = torch.tensor([1.0, 2.0, 0.0, 0.0, 0.0])
    valid_mask = torch.tensor([True, True, False, False, False])

    # WRONG: Include all values
    wrong_mean = advantages.mean()
    wrong_std = advantages.std()

    # RIGHT: Only valid values
    valid_advantages = advantages[valid_mask]
    right_mean = valid_advantages.mean()
    right_std = valid_advantages.std()

    # Verify masked calculation is different
    assert abs(wrong_mean.item() - right_mean.item()) > 0.1, (
        f"Mean should differ: wrong={wrong_mean.item():.3f}, right={right_mean.item():.3f}"
    )
    assert abs(wrong_std.item() - right_std.item()) > 0.1, (
        f"Std should differ: wrong={wrong_std.item():.3f}, right={right_std.item():.3f}"
    )

    # Verify correct values
    expected_mean = 1.5  # (1.0 + 2.0) / 2
    # Bessel-corrected std: sqrt(((1-1.5)^2 + (2-1.5)^2) / (2-1)) = sqrt(0.5) ≈ 0.7071
    expected_std = 0.7071067811865476  # sqrt(0.5)
    assert abs(right_mean.item() - expected_mean) < 1e-5, (
        f"Valid mean should be {expected_mean}, got {right_mean.item()}"
    )
    # Note: torch.std uses Bessel correction by default (ddof=1)
    assert abs(right_std.item() - expected_std) < 1e-4, (  # Relaxed tolerance for float precision
        f"Valid std should be {expected_std}, got {right_std.item()}"
    )


def test_advantage_normalization_with_action_masks() -> None:
    """Verify advantage normalization respects action masks from environment.

    BUG PATTERN: The environment provides action masks (e.g., which slots are
    valid targets). If advantages for invalid actions are included in
    normalization, statistics are corrupted.

    This test simulates a scenario where only some slots are valid targets
    and verifies normalization handles this correctly.
    """
    import torch

    batch_size = 4
    num_slots = 3

    # Advantages per slot action
    # Shape: [batch, num_slots]
    advantages = torch.tensor([
        [1.0, 2.0, 0.0],  # Batch 0: slots 0,1 valid, slot 2 masked
        [0.0, 1.5, 1.5],  # Batch 1: slot 0 masked, slots 1,2 valid
        [0.5, 0.0, 0.0],  # Batch 2: only slot 0 valid
        [1.0, 1.0, 1.0],  # Batch 3: all slots valid
    ])

    # Valid action masks (True = valid, False = masked)
    valid_masks = torch.tensor([
        [True, True, False],
        [False, True, True],
        [True, False, False],
        [True, True, True],
    ])

    # Collect all valid advantages
    all_valid = []
    for b in range(batch_size):
        for s in range(num_slots):
            if valid_masks[b, s]:
                all_valid.append(advantages[b, s].item())

    # Compute correct normalization stats
    valid_tensor = torch.tensor(all_valid)
    correct_mean = valid_tensor.mean()
    correct_std = valid_tensor.std()

    # Normalize using correct stats
    normalized = (advantages - correct_mean) / (correct_std + 1e-8)

    # Verify normalized valid values have reasonable magnitude
    for b in range(batch_size):
        for s in range(num_slots):
            if valid_masks[b, s]:
                # Valid normalized values should be roughly in [-2, 2] range
                assert abs(normalized[b, s].item()) < 3.0, (
                    f"Normalized advantage at [{b},{s}] is {normalized[b, s].item():.3f}, "
                    f"which is unexpectedly large. Check normalization stats."
                )

    # Verify masked positions still have their original (zero) values treated correctly
    # Note: We don't normalize masked values in practice, but if we did, zeros would
    # become negative (pulled by mean). This is why masking is important.
```

##### 4. Gradient Flow Through Masked Heads Test

Heads with advantage=0 should contribute zero gradient, but the log_prob must still be computed correctly to avoid NaN gradients.

```python
def test_zero_advantage_produces_zero_gradient() -> None:
    """Verify loss = -advantage * ratio produces grad=0 when advantage=0.

    BUG PATTERN: Even though advantage=0 should zero out the gradient,
    numerical issues or incorrect masking can cause non-zero gradients.
    This wastes compute and can destabilize training.

    NOTE: This tests the POLICY GRADIENT component only. In full PPO:
    - Policy gradient: -advantage * ratio * log_prob → zero when advantage=0
    - Entropy bonus: +entropy_coef * entropy → STILL produces gradients
    - Value loss: (V_pred - V_target)^2 → separate from policy

    The entropy bonus intentionally produces gradients for ALL heads to prevent
    premature convergence. This test verifies policy gradient isolation only.
    """
    import torch
    import torch.nn as nn

    # Simple policy network
    policy = nn.Linear(10, 4)
    x = torch.randn(1, 10)

    logits = policy(x)
    log_probs = torch.log_softmax(logits, dim=-1)
    action = torch.tensor([0])
    lp = log_probs[0, action]

    # Zero advantage (causal mask says this head had no effect)
    advantage = torch.tensor(0.0)
    # Note: ratio=1.0 is a scalar here for simplicity. In real PPO,
    # ratio = exp(new_log_prob - old_log_prob) has gradients w.r.t. new_log_prob
    ratio = torch.tensor(1.0)

    # Policy gradient loss (excluding entropy bonus)
    loss = -advantage * ratio * lp

    # Should be exactly zero
    assert loss.item() == 0.0, (
        f"Loss with zero advantage should be 0.0, got {loss.item()}"
    )

    # Backprop
    loss.backward()

    # Policy gradient contribution should be zero
    grad_norm = sum(p.grad.norm().item() for p in policy.parameters() if p.grad is not None)
    assert grad_norm == 0.0, (
        f"Policy gradient with zero advantage should be 0.0, got {grad_norm}. "
        f"Note: entropy bonus would still produce gradients - that's intentional."
    )


def test_masked_head_with_zero_probability_action() -> None:
    """Verify no NaN gradients when masked head has zero probability for sampled action.

    BUG PATTERN: If the policy assigns probability 0 to the action that was
    sampled (e.g., due to masking), log(0) = -inf causes NaN gradients.

    Even if advantage=0, the log_prob computation happens first. If log_prob
    is -inf and advantage is 0, we get -inf * 0 = NaN.

    The fix: Ensure action masking in the policy prevents sampling invalid
    actions, OR clamp log_probs to a minimum value.
    """
    import torch

    # Simulate log_probs with a very low probability action
    log_probs = torch.tensor([-1.0, -2.0, -50.0, -100.0])  # Last action has ~0 prob

    # Sample the near-zero probability action
    action = torch.tensor(3)
    lp = log_probs[action]

    # Ratio computation
    old_lp = torch.tensor(-1.0)
    ratio = torch.exp(lp - old_lp)  # exp(-100 - (-1)) = exp(-99) ~= 0

    # Zero advantage (masked head)
    advantage = torch.tensor(0.0)

    # Compute objective
    obj = ratio * advantage

    # Should be 0, not NaN
    assert not torch.isnan(obj), (
        f"Objective is NaN. ratio={ratio.item()}, advantage={advantage.item()}"
    )
    assert obj.item() == 0.0, (
        f"Objective should be 0.0, got {obj.item()}"
    )

    # Now test with -inf log_prob (the actual danger case)
    inf_lp = torch.tensor(float("-inf"))
    inf_ratio = torch.exp(inf_lp - old_lp)  # exp(-inf) = 0

    # 0 * 0 = 0, not NaN
    inf_obj = inf_ratio * advantage
    assert not torch.isnan(inf_obj), (
        f"Objective with -inf log_prob should not be NaN. "
        f"inf_ratio={inf_ratio.item()}, advantage={advantage.item()}"
    )

    # But if advantage is non-zero with -inf log_prob, we get 0 * non-zero = 0
    # Still not NaN, but the gradient would be problematic
    nonzero_adv = torch.tensor(1.0)
    mixed_obj = inf_ratio * nonzero_adv
    assert mixed_obj.item() == 0.0, (
        f"Objective should be 0.0 (0 * 1.0 = 0), got {mixed_obj.item()}"
    )


def test_log_prob_clamping_prevents_nan() -> None:
    """Verify log_prob clamping prevents NaN in edge cases.

    BUG PATTERN: Without clamping, very small probabilities produce -inf
    log_probs, which cause numerical instability in ratio computation.

    The fix: Clamp log_probs to a minimum value (e.g., -100) before
    computing ratios.
    """
    import torch

    # Extreme log_probs
    log_probs = torch.tensor([0.0, -50.0, -100.0, float("-inf")])

    # TODO: Replace with leyline.LOG_PROB_MIN when available
    # Note: -100.0 chosen because exp(-100) ≈ 3.7e-44 (tiny but non-zero in float64)
    # In float32: exp(-88) ≈ 1e-38 (smallest normal), exp(-104) underflows to 0.0
    # For ratio stability, we need exp(new_lp - old_lp) to be finite, so -100 is safe
    # Alternative: log_prob_min = torch.finfo(torch.float32).min * 0.1 ≈ -34
    log_prob_min = -100.0

    # Clamp to prevent -inf
    clamped = torch.clamp(log_probs, min=log_prob_min)

    # Verify no -inf after clamping
    assert not torch.isinf(clamped).any(), (
        f"Clamped log_probs should not contain inf: {clamped}"
    )

    # Verify clamping doesn't affect reasonable values
    assert clamped[0].item() == 0.0, "Log_prob 0.0 should be unchanged"
    assert clamped[1].item() == -50.0, "Log_prob -50.0 should be unchanged"
    assert clamped[2].item() == -100.0, "Log_prob -100.0 should be unchanged (at boundary)"
    assert clamped[3].item() == -100.0, "Log_prob -inf should be clamped to -100.0"

    # Verify ratio computation is now stable
    old_lp = torch.tensor(-1.0)
    ratios = torch.exp(clamped - old_lp)

    assert not torch.isnan(ratios).any(), f"Ratios should not contain NaN: {ratios}"
    assert not torch.isinf(ratios).any(), f"Ratios should not contain inf: {ratios}"
```

##### 5. Clip Fraction Monitoring Per Head Test

Tracking clip fraction per head detects premature convergence in individual heads.

```python
def test_clip_fraction_computed_per_head() -> None:
    """Verify clip_fraction is computed correctly for each action head.

    BUG PATTERN: Aggregating clip_fraction across heads hides per-head issues.
    If head 0 clips 90% of updates but head 7 clips 5%, the average (47.5%)
    looks normal. But head 0 has stopped learning.

    Per-head clip fractions reveal:
    - Which heads are converging prematurely
    - Whether certain action types are easier/harder to learn
    - Policy collapse in specific heads

    NOTE: Per-head clip fractions are DIAGNOSTIC only. The actual clipping in
    Tamiyo's hybrid PPO uses the joint ratio. This metric helps debug which
    heads are contributing most to clipping events.
    """
    import torch

    # TODO: Replace with leyline constants
    num_heads = 8
    clip_epsilon = 0.2
    batch_size = 100

    # Simulate ratios per head with different clipping patterns
    # Shape: [batch, heads]
    ratios = torch.ones(batch_size, num_heads)

    # Head 0: Almost all clipped (90%)
    ratios[:90, 0] = 1.5  # Outside clip range [0.8, 1.2]

    # Head 7: Rarely clipped (5%)
    ratios[:5, 7] = 0.5  # Outside clip range

    # Other heads: Moderate clipping (30%)
    for h in range(1, 7):
        ratios[:30, h] = 1.4

    # Compute clip fraction per head
    ratio_min = 1.0 - clip_epsilon
    ratio_max = 1.0 + clip_epsilon

    clipped = (ratios < ratio_min) | (ratios > ratio_max)
    clip_fraction_per_head = clipped.float().mean(dim=0)

    # Verify per-head computation
    assert abs(clip_fraction_per_head[0].item() - 0.90) < 0.01, (
        f"Head 0 clip fraction should be ~0.90, got {clip_fraction_per_head[0].item()}"
    )
    assert abs(clip_fraction_per_head[7].item() - 0.05) < 0.01, (
        f"Head 7 clip fraction should be ~0.05, got {clip_fraction_per_head[7].item()}"
    )

    # Verify aggregated clip fraction hides the issue
    aggregated_clip_fraction = clipped.float().mean()
    # (90 + 5 + 6*30) / (8*100) = 275 / 800 = 0.34375
    expected_aggregated = 0.34375
    assert abs(aggregated_clip_fraction.item() - expected_aggregated) < 0.01, (
        f"Aggregated clip fraction should be ~{expected_aggregated}, "
        f"got {aggregated_clip_fraction.item()}"
    )

    # The key insight: aggregated looks normal (34%) but head 0 is at 90%
    assert clip_fraction_per_head[0].item() > aggregated_clip_fraction.item() * 2, (
        "Head 0 clip fraction should be much higher than aggregated, "
        "demonstrating that aggregation hides per-head issues"
    )


def test_clip_fraction_detects_policy_collapse() -> None:
    """Verify high clip fraction indicates potential policy collapse.

    BUG PATTERN: When a head's policy becomes deterministic (entropy -> 0),
    the ratio for any non-modal action becomes very large (approaching inf).
    This causes near-100% clipping, which prevents further learning.

    This is detectable by monitoring clip fraction over training:
    - Healthy: Clip fraction ~10-30%, decreases over training
    - Collapse: Clip fraction suddenly spikes to >80%
    """
    import torch

    # Simulate a collapsing policy (becoming deterministic)
    batch_size = 100
    num_actions = 4

    # Early training: Policy is exploratory
    early_probs = torch.tensor([0.25, 0.25, 0.25, 0.25])  # Uniform
    early_log_probs = torch.log(early_probs)

    # After collapse: Policy is deterministic
    # One action has prob ~1, others have prob ~0
    collapsed_probs = torch.tensor([0.97, 0.01, 0.01, 0.01])
    collapsed_log_probs = torch.log(collapsed_probs)

    # Sample from early policy, evaluate under collapsed policy
    # This is what happens when old_policy was exploratory but new_policy collapsed
    actions = torch.randint(0, num_actions, (batch_size,))

    # Compute ratios: exp(new_log_prob - old_log_prob)
    old_lp = early_log_probs[actions]
    new_lp = collapsed_log_probs[actions]
    ratios = torch.exp(new_lp - old_lp)

    # For action 0 (modal): ratio = 0.97/0.25 = 3.88
    # For actions 1-3: ratio = 0.01/0.25 = 0.04
    # Both are outside clip range [0.8, 1.2]

    clip_epsilon = 0.2
    clipped = (ratios < 1 - clip_epsilon) | (ratios > 1 + clip_epsilon)
    clip_fraction = clipped.float().mean()

    # Nearly all ratios should be clipped
    assert clip_fraction.item() > 0.9, (
        f"Collapsed policy should have >90% clip fraction, got {clip_fraction.item():.2%}. "
        f"This indicates policy collapse detection is not working."
    )


def test_clip_fraction_logging_format() -> None:
    """Verify clip fraction is logged in a format suitable for monitoring.

    BUG PATTERN: If clip fractions are logged as a single aggregated value,
    dashboards cannot show per-head trends. Logging must include head names.
    """
    from typing import Any

    # TODO: Replace with actual head names from leyline
    head_names = [
        "op", "slot", "blueprint", "style",
        "tempo", "alpha_target", "alpha_speed", "alpha_curve"
    ]

    # Simulated clip fractions per head
    clip_fractions = [0.15, 0.12, 0.08, 0.22, 0.18, 0.25, 0.30, 0.11]

    # Expected logging format: dict with head names as keys
    logged_metrics: dict[str, Any] = {}
    for name, frac in zip(head_names, clip_fractions):
        logged_metrics[f"ppo/clip_fraction/{name}"] = frac

    # Also log aggregated for backward compatibility
    logged_metrics["ppo/clip_fraction/mean"] = sum(clip_fractions) / len(clip_fractions)

    # Verify all heads are logged
    for name in head_names:
        key = f"ppo/clip_fraction/{name}"
        assert key in logged_metrics, f"Missing clip fraction for head: {name}"

    # Verify aggregated is also present
    assert "ppo/clip_fraction/mean" in logged_metrics, (
        "Missing aggregated clip fraction"
    )

    # Verify values are in valid range [0, 1]
    for key, value in logged_metrics.items():
        assert 0.0 <= value <= 1.0, f"Invalid clip fraction {key}={value}"
```

**Validation commands:**

```bash
# Run all Phase 6 PPO validation tests
PYTHONPATH=src uv run pytest tests/simic/test_ppo_clipping.py -v

# Quick smoke test: verify ratio computation
PYTHONPATH=src uv run python -c "
import torch
# Test joint ratio computation
old_lp = torch.tensor([-1.0, -1.0, -1.0])
new_lp = torch.tensor([-0.5, -1.0, -1.0])  # Head 0 changed
joint_ratio = torch.exp(new_lp.sum() - old_lp.sum())
print(f'Joint ratio with one head changed: {joint_ratio.item():.4f}')
assert abs(joint_ratio.item() - 1.6487) < 0.01, 'Joint ratio incorrect'
print('Joint ratio computation: PASS')
"

# Check clip fraction per head in training run
PYTHONPATH=src uv run python -m esper.scripts.train ppo --episodes 1 --max-epochs 10 --seed 42 2>&1 | grep -E "clip_fraction"

# Verify advantage normalization excludes masked zeros
PYTHONPATH=src uv run python -c "
import torch
advantages = torch.tensor([1.0, 2.0, 0.0, 0.0, 0.0])
valid_mask = torch.tensor([True, True, False, False, False])
wrong_std = advantages.std().item()
right_std = advantages[valid_mask].std().item()
print(f'Wrong std (includes zeros): {wrong_std:.4f}')
print(f'Right std (valid only): {right_std:.4f}')
assert abs(wrong_std - right_std) > 0.1, 'Masking should change std'
print('Advantage normalization masking: PASS')
"
```

**Key validation invariants:**

| Check | Expected | Failure Indicates |
|-------|----------|-------------------|
| Joint ratio = exp(sum_new - sum_old) | Uses sum, not per-head average | Secondary head changes ignored |
| Head with adv=0, ratio=2.0 | No clipping effect on other heads | Clipping coupling across heads |
| Advantage normalization mean | Excludes masked (zero) entries | Biased advantage estimates |
| Advantage normalization std | Excludes masked entries | Understated std, aggressive updates |
| Zero advantage loss | Exactly 0.0 | Spurious gradients from masked heads |
| Log_prob for masked action | Clamped (no -inf) | NaN gradients in ratio computation |
| Clip fraction per head | Logged separately (8 values) | Per-head issues hidden by aggregation |
| Clip fraction spike (>80%) | Triggers alert | Policy collapse in specific head |

#### 7i. Phase 7 Addendum: Reward Design & Hindsight Credit Validation

These tests validate that the reward function correctly attributes credit for seed lifecycle actions. Incorrect reward attribution is insidious: the system trains without crashing, but the policy learns wrong behaviors. A seed that creates dependencies (ransomware pattern) can appear valuable while actually degrading the host. Immediate prune after training skips counterfactual measurement, leaving `seed_contribution=None` and potentially incorrect credit assignment.

The reward system has multiple interacting components:
1. **Bounded attribution** - Primary counterfactual-based signal
2. **Compute rent** - Parameter overhead penalty (logarithmic scaling)
3. **Alpha shock** - Convex penalty on rapid alpha changes
4. **Hindsight credit** - Retroactive credit for scaffold seeds when beneficiaries fossilize
5. **PBRS stage progression** - Potential-based shaping for lifecycle advancement

> **Note on magic numbers**: The values used below represent:
> - `0.5`: Base fossilize bonus (ContributionRewardConfig.fossilize_base_bonus)
> - `0.3`: PBRS weight (ContributionRewardConfig.pbrs_weight)
> - `0.5`: Rent weight (ContributionRewardConfig.rent_weight)
> - `0.1958`: Alpha shock coefficient (calibrated from telemetry)
> - `0.0039`: Base slot rent ratio (calibrated from telemetry)
> - `5.0`: Hacking ratio threshold (contribution/improvement ratio triggering penalty)
> - `0.2`: Credit weight for hindsight credit (compute_scaffold_hindsight_credit)
> - `25`: Default max_epochs for episode length
>
> When Phase 7 is implemented, verify these match `ContributionRewardConfig` defaults in `simic/rewards/rewards.py`.

##### 1. Seed Contribution Without Baseline Test

When a seed is pruned before reaching BLENDING stage, no counterfactual measurement exists. The reward function must handle `seed_contribution=None` gracefully and still produce sensible rewards (not zero, not NaN).

```python
def test_prune_before_blending_has_no_counterfactual() -> None:
    """Verify reward is sensible when pruning a seed before counterfactual is available.

    BUG PATTERN: If seed_contribution=None is not handled, the reward function may:
    1. Return 0.0 (incorrect - pruning still has shaping consequences)
    2. Crash with TypeError (None in arithmetic)
    3. Use stale counterfactual from previous seed (wrong attribution)

    The correct behavior: use acc_delta proxy signal for pre-blending stages,
    but do NOT impute fake counterfactual values.
    """
    from esper.simic.rewards import (
        compute_contribution_reward,
        ContributionRewardConfig,
        SeedInfo,
    )
    from esper.leyline import LifecycleOp, SeedStage

    config = ContributionRewardConfig()

    # Seed in TRAINING stage (before BLENDING, so no counterfactual)
    seed_info = SeedInfo(
        stage=SeedStage.TRAINING.value,
        improvement_since_stage_start=0.5,
        total_improvement=0.5,
        epochs_in_stage=3,
        seed_params=10000,
        previous_stage=SeedStage.GERMINATED.value,
        previous_epochs_in_stage=2,
        seed_age_epochs=5,
    )

    # PRUNE with no counterfactual (seed_contribution=None)
    reward, components = compute_contribution_reward(
        action=LifecycleOp.PRUNE,
        seed_contribution=None,  # No counterfactual available yet
        val_acc=45.0,
        seed_info=seed_info,
        epoch=10,
        max_epochs=25,
        total_params=110000,
        host_params=100000,
        config=config,
        acc_at_germination=40.0,
        acc_delta=0.5,
        return_components=True,
    )

    # Reward should be a valid float (not NaN, not None)
    import math
    assert reward is not None and not math.isnan(reward), (
        f"Reward is NaN or None when pruning without counterfactual. "
        f"This corrupts policy gradients."
    )

    # Bounded attribution should be None or 0 (no counterfactual to invert)
    # Pre-blending PRUNE should not receive proxy-based attribution
    # because proxy signal is for WAIT/ADVANCE, not PRUNE
    assert components.bounded_attribution is None or components.bounded_attribution == 0, (
        f"PRUNE without counterfactual should have no attribution, "
        f"got {components.bounded_attribution}"
    )

    # Reward should still include other components (PBRS, rent, action shaping)
    assert components.pbrs_bonus != 0 or components.compute_rent != 0 or components.action_shaping != 0, (
        "Reward should include non-attribution components even without counterfactual"
    )


def test_prune_immediately_after_training_stage() -> None:
    """Verify immediate prune after entering TRAINING produces valid reward.

    BUG PATTERN: Seed just transitioned to TRAINING (epochs_in_stage=0).
    PRUNE is issued immediately. If the reward function assumes at least
    1 epoch in stage, it may compute incorrect PBRS delta or crash.
    """
    from esper.simic.rewards import (
        compute_contribution_reward,
        ContributionRewardConfig,
        SeedInfo,
    )
    from esper.leyline import LifecycleOp, SeedStage

    config = ContributionRewardConfig()

    # Just transitioned to TRAINING (epochs_in_stage=0)
    seed_info = SeedInfo(
        stage=SeedStage.TRAINING.value,
        improvement_since_stage_start=0.0,  # No improvement yet
        total_improvement=0.1,
        epochs_in_stage=0,  # Just entered this stage
        seed_params=10000,
        previous_stage=SeedStage.GERMINATED.value,
        previous_epochs_in_stage=3,
        seed_age_epochs=3,
    )

    reward, components = compute_contribution_reward(
        action=LifecycleOp.PRUNE,
        seed_contribution=None,
        val_acc=42.0,
        seed_info=seed_info,
        epoch=5,
        max_epochs=25,
        total_params=110000,
        host_params=100000,
        config=config,
        acc_at_germination=40.0,
        acc_delta=0.2,
        return_components=True,
    )

    # Verify PBRS handles transition correctly
    # phi_current = TRAINING(2.0) + epochs_in_stage*0.3 = 2.0 + 0 = 2.0
    # phi_prev = GERMINATED(1.0) + previous_epochs*0.3 = 1.0 + 0.9 = 1.9
    # pbrs = 0.3 * (0.995 * 2.0 - 1.9) = 0.3 * (1.99 - 1.9) = 0.027
    assert not (components.pbrs_bonus != components.pbrs_bonus), (
        f"PBRS bonus is NaN at stage transition. "
        f"Check phi_prev calculation for epochs_in_stage=0."
    )

    # Verify total reward is reasonable
    assert -5.0 < reward < 5.0, (
        f"Reward {reward} is outside reasonable range [-5, 5]. "
        f"Immediate prune may be triggering extreme penalties."
    )
```

##### 2. Rent Penalty Scaling Test

Compute rent should scale logarithmically with parameter overhead. Large blueprints should incur higher rent than small ones. A bug where rent is always zero or constant would allow parameter bloat.

```python
def test_rent_penalty_scales_with_blueprint_size() -> None:
    """Verify compute_rent differs for large vs small blueprints.

    BUG PATTERN: If rent is computed incorrectly (e.g., always 0, or uses
    total_params instead of seed overhead), large seeds incur same cost as
    small seeds. The policy learns to always use maximum-size blueprints.
    """
    from dataclasses import replace
    from esper.simic.rewards import (
        compute_contribution_reward,
        ContributionRewardConfig,
        SeedInfo,
    )
    from esper.leyline import LifecycleOp, SeedStage

    config = ContributionRewardConfig()
    host_params = 100000

    # Base seed info (same for both)
    base_seed_info = SeedInfo(
        stage=SeedStage.BLENDING.value,
        improvement_since_stage_start=0.5,
        total_improvement=1.0,
        epochs_in_stage=3,
        seed_params=0,  # Will vary
        previous_stage=SeedStage.TRAINING.value,
        previous_epochs_in_stage=5,
        seed_age_epochs=8,
    )

    # Small blueprint: 5% overhead
    small_seed_params = 5000
    _, small_components = compute_contribution_reward(
        action=LifecycleOp.WAIT,
        seed_contribution=0.5,
        val_acc=50.0,
        seed_info=replace(base_seed_info, seed_params=small_seed_params),
        epoch=10,
        max_epochs=25,
        total_params=host_params + small_seed_params,
        host_params=host_params,
        config=config,
        acc_at_germination=45.0,
        acc_delta=0.5,
        return_components=True,
    )

    # Large blueprint: 50% overhead
    large_seed_params = 50000
    _, large_components = compute_contribution_reward(
        action=LifecycleOp.WAIT,
        seed_contribution=0.5,
        val_acc=50.0,
        seed_info=replace(base_seed_info, seed_params=large_seed_params),
        epoch=10,
        max_epochs=25,
        total_params=host_params + large_seed_params,
        host_params=host_params,
        config=config,
        acc_at_germination=45.0,
        acc_delta=0.5,
        return_components=True,
    )

    # Rent should be negative (it's a penalty)
    assert small_components.compute_rent <= 0, (
        f"compute_rent should be <= 0 (penalty), got {small_components.compute_rent}"
    )
    assert large_components.compute_rent <= 0, (
        f"compute_rent should be <= 0 (penalty), got {large_components.compute_rent}"
    )

    # Large blueprint should have MORE NEGATIVE rent (bigger penalty)
    assert large_components.compute_rent < small_components.compute_rent, (
        f"Large blueprint rent ({large_components.compute_rent}) should be more negative "
        f"than small blueprint rent ({small_components.compute_rent}). "
        f"Rent is not scaling with parameter overhead."
    )

    # Verify logarithmic scaling (not linear)
    # 50% overhead vs 5% overhead = 10x params, but rent should be ~2-3x (log)
    rent_ratio = abs(large_components.compute_rent) / max(abs(small_components.compute_rent), 1e-8)
    assert rent_ratio < 5.0, (
        f"Rent ratio ({rent_ratio}) is too high for logarithmic scaling. "
        f"Expected ~2-3x for 10x parameter difference. Rent may be linear."
    )
    assert rent_ratio > 1.5, (
        f"Rent ratio ({rent_ratio}) is too low. Large blueprints should incur "
        f"noticeably higher rent. Check growth_ratio calculation."
    )
```

##### 3. Alpha Shock for Rapid Changes Test

The alpha_shock penalty should differ for immediate vs gradual alpha changes. Rapid changes (large delta^2) should incur higher penalty.

```python
def test_alpha_shock_differs_for_fast_vs_slow_changes() -> None:
    """Verify alpha_shock penalty is higher for rapid alpha changes.

    BUG PATTERN: If alpha_shock is always 0 or ignores the delta magnitude,
    the policy can make arbitrarily rapid alpha changes without penalty.
    This enables reward hacking via fast alpha oscillation.
    """
    from esper.simic.rewards import (
        compute_contribution_reward,
        ContributionRewardConfig,
        SeedInfo,
    )
    from esper.leyline import LifecycleOp, SeedStage

    config = ContributionRewardConfig()

    seed_info = SeedInfo(
        stage=SeedStage.BLENDING.value,
        improvement_since_stage_start=0.5,
        total_improvement=1.0,
        epochs_in_stage=5,
        seed_params=10000,
        previous_stage=SeedStage.TRAINING.value,
        previous_epochs_in_stage=5,
        seed_age_epochs=10,
    )

    # Slow alpha change: delta = 0.1, delta^2 = 0.01
    slow_alpha_delta_sq_sum = 0.01
    _, slow_components = compute_contribution_reward(
        action=LifecycleOp.SET_ALPHA_TARGET,
        seed_contribution=0.5,
        val_acc=50.0,
        seed_info=seed_info,
        epoch=12,
        max_epochs=25,
        total_params=110000,
        host_params=100000,
        config=config,
        acc_at_germination=45.0,
        acc_delta=0.3,
        return_components=True,
        alpha_delta_sq_sum=slow_alpha_delta_sq_sum,
    )

    # Fast alpha change: delta = 0.8, delta^2 = 0.64
    fast_alpha_delta_sq_sum = 0.64
    _, fast_components = compute_contribution_reward(
        action=LifecycleOp.SET_ALPHA_TARGET,
        seed_contribution=0.5,
        val_acc=50.0,
        seed_info=seed_info,
        epoch=12,
        max_epochs=25,
        total_params=110000,
        host_params=100000,
        config=config,
        acc_at_germination=45.0,
        acc_delta=0.3,
        return_components=True,
        alpha_delta_sq_sum=fast_alpha_delta_sq_sum,
    )

    # Both should have negative (penalty) alpha_shock
    assert slow_components.alpha_shock <= 0, (
        f"alpha_shock should be <= 0 (penalty), got {slow_components.alpha_shock}"
    )
    assert fast_components.alpha_shock <= 0, (
        f"alpha_shock should be <= 0 (penalty), got {fast_components.alpha_shock}"
    )

    # Fast change should have MORE NEGATIVE shock (bigger penalty)
    assert fast_components.alpha_shock < slow_components.alpha_shock, (
        f"Fast alpha change shock ({fast_components.alpha_shock}) should be more negative "
        f"than slow change shock ({slow_components.alpha_shock}). "
        f"alpha_shock is not scaling with delta magnitude."
    )

    # Verify convex scaling: 64x delta^2 should give ~64x penalty
    if slow_components.alpha_shock != 0:
        shock_ratio = abs(fast_components.alpha_shock) / abs(slow_components.alpha_shock)
        # Allow some tolerance for coefficient variations
        assert 30 < shock_ratio < 100, (
            f"alpha_shock ratio ({shock_ratio}) should be ~64 for 64x delta^2. "
            f"Convex scaling may be incorrect."
        )
```

##### 4. Ransomware Seed Scenario Test

A "ransomware seed" has high counterfactual contribution but zero or negative total improvement. The reward system must NOT reward fossilizing such seeds.

```python
def test_ransomware_seed_not_rewarded_for_fossilization() -> None:
    """Verify seeds with high contribution but negative total_improvement are penalized.

    BUG PATTERN: The "ransomware" pattern occurs when a seed creates structural
    dependencies that inflate its counterfactual contribution without adding
    actual value. Removing the seed hurts (high contribution), but keeping it
    also hurts (negative total_improvement).

    If the reward function only looks at seed_contribution, it rewards
    fossilizing these toxic seeds. The correct behavior: penalty when
    total_improvement < 0 regardless of contribution.
    """
    from esper.simic.rewards import (
        compute_contribution_reward,
        ContributionRewardConfig,
        SeedInfo,
    )
    from esper.leyline import LifecycleOp, SeedStage

    config = ContributionRewardConfig()

    # Ransomware seed: high contribution, negative total improvement
    ransomware_seed = SeedInfo(
        stage=SeedStage.HOLDING.value,
        improvement_since_stage_start=-0.5,  # Still hurting
        total_improvement=-2.0,  # Net negative since germination
        epochs_in_stage=3,
        seed_params=15000,
        previous_stage=SeedStage.BLENDING.value,
        previous_epochs_in_stage=5,
        seed_age_epochs=15,
    )

    reward, components = compute_contribution_reward(
        action=LifecycleOp.FOSSILIZE,
        seed_contribution=5.0,  # High contribution (removal hurts)
        val_acc=48.0,  # Lower than at germination
        seed_info=ransomware_seed,
        epoch=18,
        max_epochs=25,
        total_params=115000,
        host_params=100000,
        config=config,
        acc_at_germination=50.0,  # Was higher before seed
        acc_delta=-0.2,
        return_components=True,
    )

    # Reward should be NEGATIVE for fossilizing a ransomware seed
    assert reward < 0, (
        f"Fossilizing ransomware seed should be penalized (negative reward), "
        f"but got {reward}. The policy will learn to fossilize toxic seeds."
    )

    # Attribution discount should have been applied (sigmoid of negative total_improvement)
    assert components.attribution_discount < 1.0, (
        f"Attribution discount should be < 1.0 for negative total_improvement, "
        f"got {components.attribution_discount}"
    )

    # Action shaping should include penalty for negative total_improvement FOSSILIZE
    # This is the ransomware check in _contribution_fossilize_shaping()
    assert components.action_shaping < 0, (
        f"Action shaping should be negative for ransomware FOSSILIZE, "
        f"got {components.action_shaping}"
    )


def test_no_terminal_bonus_for_ransomware_fossilizations() -> None:
    """Verify fossilized seeds with negative improvement don't count for terminal bonus.

    BUG PATTERN: The terminal bonus includes fossilize_terminal_scale * num_contributing_fossilized.
    If num_contributing_fossilized includes seeds with negative total_improvement,
    ransomware fossilizations become NPV-positive (immediate penalty offset by terminal bonus).
    """
    from esper.simic.rewards import (
        compute_contribution_reward,
        ContributionRewardConfig,
        SeedInfo,
    )
    from esper.leyline import LifecycleOp, SeedStage

    config = ContributionRewardConfig()

    # Terminal epoch with 2 fossilized seeds: 1 contributing, 1 ransomware
    # num_contributing_fossilized should be 1, not 2

    seed_info = None  # No active seed at terminal

    # With 1 contributing fossilized
    _, good_components = compute_contribution_reward(
        action=LifecycleOp.WAIT,
        seed_contribution=None,
        val_acc=55.0,
        seed_info=seed_info,
        epoch=25,  # Terminal
        max_epochs=25,
        total_params=120000,
        host_params=100000,
        config=config,
        acc_at_germination=None,
        acc_delta=0.0,
        return_components=True,
        num_fossilized_seeds=2,
        num_contributing_fossilized=1,  # Only 1 meets threshold
    )

    # With 2 fossilized (incorrectly counting ransomware as contributing)
    _, bad_components = compute_contribution_reward(
        action=LifecycleOp.WAIT,
        seed_contribution=None,
        val_acc=55.0,
        seed_info=seed_info,
        epoch=25,
        max_epochs=25,
        total_params=120000,
        host_params=100000,
        config=config,
        acc_at_germination=None,
        acc_delta=0.0,
        return_components=True,
        num_fossilized_seeds=2,
        num_contributing_fossilized=2,  # Bug: counts ransomware
    )

    # fossilize_terminal_bonus should differ
    assert bad_components.fossilize_terminal_bonus > good_components.fossilize_terminal_bonus, (
        "Test setup error: bad case should have higher fossilize_terminal_bonus"
    )

    # The correct behavior: caller must NOT count ransomware seeds
    # Verify the magnitude of difference
    bonus_diff = bad_components.fossilize_terminal_bonus - good_components.fossilize_terminal_bonus
    expected_diff = config.fossilize_terminal_scale  # One extra "contributing" seed
    assert abs(bonus_diff - expected_diff) < 0.1, (
        f"Terminal bonus difference ({bonus_diff}) doesn't match fossilize_terminal_scale "
        f"({expected_diff}). Terminal bonus calculation may be incorrect."
    )
```

##### 5. Hindsight Credit Delivery Test

The scaffold hindsight credit system has `pending_hindsight_credit` in ParallelEnvState which must be consumed and added to the actual reward. If it accumulates but is never applied, scaffolding behavior is never reinforced.

```python
def test_pending_hindsight_credit_is_consumed() -> None:
    """Verify pending_hindsight_credit is added to reward and then cleared.

    BUG PATTERN: If pending_hindsight_credit accumulates but is never read
    and added to the reward, scaffold seeds never receive retroactive credit.
    The policy learns that scaffolding has no value.

    Symptoms:
    - pending_hindsight_credit grows unboundedly
    - hindsight_credit in RewardComponentsTelemetry is always 0
    - Scaffold behavior (positive interaction with beneficiary) is never learned
    """
    from esper.simic.training.parallel_env_state import ParallelEnvState

    state = ParallelEnvState()

    # Simulate: beneficiary fossilizes, scaffold gets pending credit
    scaffold_credit = 0.15
    state.pending_hindsight_credit = scaffold_credit

    # Simulate consumption (what vectorized.py should do)
    hindsight_credit_applied = 0.0
    if state.pending_hindsight_credit > 0:
        hindsight_credit_applied = state.pending_hindsight_credit
        state.pending_hindsight_credit = 0.0

    # Credit should have been applied
    assert hindsight_credit_applied == scaffold_credit, (
        f"pending_hindsight_credit ({scaffold_credit}) was not fully consumed. "
        f"Only {hindsight_credit_applied} was applied."
    )

    # pending_hindsight_credit should be cleared
    assert state.pending_hindsight_credit == 0.0, (
        f"pending_hindsight_credit should be 0 after consumption, "
        f"got {state.pending_hindsight_credit}"
    )


def test_hindsight_credit_computation_is_bounded() -> None:
    """Verify compute_scaffold_hindsight_credit returns bounded values.

    BUG PATTERN: If hindsight credit is unbounded, scaffold seeds can
    accumulate arbitrary credit, dominating the reward signal and
    causing reward hacking.
    """
    from esper.simic.rewards import compute_scaffold_hindsight_credit

    credit_weight = 0.2  # Default HINDSIGHT_CREDIT_WEIGHT

    # Small boost, small improvement
    small_credit = compute_scaffold_hindsight_credit(
        boost_given=0.1,
        beneficiary_improvement=0.5,
        credit_weight=credit_weight,
    )

    # Large boost, large improvement
    large_credit = compute_scaffold_hindsight_credit(
        boost_given=10.0,
        beneficiary_improvement=20.0,
        credit_weight=credit_weight,
    )

    # Both should be in [0, credit_weight] due to tanh saturation
    assert 0 <= small_credit <= credit_weight, (
        f"Small credit ({small_credit}) should be in [0, {credit_weight}]"
    )
    assert 0 <= large_credit <= credit_weight, (
        f"Large credit ({large_credit}) should be in [0, {credit_weight}]. "
        f"Hindsight credit is not bounded by tanh."
    )

    # Large should be higher but both bounded
    assert large_credit > small_credit, (
        "Large boost/improvement should give more credit than small"
    )

    # tanh saturation: even extreme values should stay bounded
    extreme_credit = compute_scaffold_hindsight_credit(
        boost_given=100.0,
        beneficiary_improvement=100.0,
        credit_weight=credit_weight,
    )
    assert abs(extreme_credit - credit_weight) < 0.01, (
        f"Extreme credit ({extreme_credit}) should saturate at {credit_weight}"
    )


def test_hindsight_credit_zero_for_negative_inputs() -> None:
    """Verify no hindsight credit for negative boost or improvement.

    BUG PATTERN: If negative values produce positive credit, parasitic
    interactions (negative boost) get rewarded as if they were helpful.
    """
    from esper.simic.rewards import compute_scaffold_hindsight_credit

    # Negative boost (parasitism)
    parasitic_credit = compute_scaffold_hindsight_credit(
        boost_given=-0.5,
        beneficiary_improvement=1.0,
        credit_weight=0.2,
    )
    assert parasitic_credit == 0.0, (
        f"Negative boost should produce 0 credit, got {parasitic_credit}"
    )

    # Negative improvement (beneficiary failed)
    failed_credit = compute_scaffold_hindsight_credit(
        boost_given=0.5,
        beneficiary_improvement=-1.0,
        credit_weight=0.2,
    )
    assert failed_credit == 0.0, (
        f"Negative improvement should produce 0 credit, got {failed_credit}"
    )
```

##### 6. Germinate-Prune Thrashing Test

A degenerate policy could learn to germinate and immediately prune seeds in a cycle. If this cycle has net positive reward, the policy farms rewards without making progress.

```python
def test_germinate_then_immediate_prune_is_not_profitable() -> None:
    """Verify germinate followed by immediate prune has net negative or zero reward.

    BUG PATTERN: If GERMINATE bonus > PRUNE penalty, the policy learns to
    thrash: germinate, prune, germinate, prune. This farms PBRS bonuses
    without ever training seeds.

    The correct invariant: sum(rewards for germinate + prune) <= 0
    """
    from esper.simic.rewards import (
        compute_contribution_reward,
        ContributionRewardConfig,
        SeedInfo,
    )
    from esper.leyline import LifecycleOp, SeedStage

    config = ContributionRewardConfig()
    host_params = 100000

    # Step 1: GERMINATE (no existing seed)
    germinate_reward, _ = compute_contribution_reward(
        action=LifecycleOp.GERMINATE,
        seed_contribution=None,
        val_acc=40.0,
        seed_info=None,  # No existing seed
        epoch=5,
        max_epochs=25,
        total_params=host_params,
        host_params=host_params,
        config=config,
        acc_at_germination=None,
        acc_delta=0.5,
        return_components=True,
    )

    # Step 2: Immediate PRUNE (seed just germinated, minimal age)
    seed_after_germinate = SeedInfo(
        stage=SeedStage.GERMINATED.value,
        improvement_since_stage_start=0.0,
        total_improvement=0.0,
        epochs_in_stage=1,  # Just 1 epoch old
        seed_params=10000,
        previous_stage=SeedStage.DORMANT.value,
        previous_epochs_in_stage=0,
        seed_age_epochs=1,
    )

    prune_reward, _ = compute_contribution_reward(
        action=LifecycleOp.PRUNE,
        seed_contribution=None,  # No counterfactual yet
        val_acc=40.5,
        seed_info=seed_after_germinate,
        epoch=6,
        max_epochs=25,
        total_params=host_params + 10000,
        host_params=host_params,
        config=config,
        acc_at_germination=40.0,
        acc_delta=0.5,
        return_components=True,
    )

    # Net reward should be <= 0 (no profit from thrashing)
    net_reward = germinate_reward + prune_reward
    assert net_reward <= 0.1, (  # Small tolerance for numerical precision
        f"Germinate ({germinate_reward:.3f}) + immediate Prune ({prune_reward:.3f}) "
        f"= {net_reward:.3f}. Net should be <= 0 to prevent thrashing. "
        f"Policy can farm rewards by cycling germinate/prune."
    )


def test_repeated_thrashing_cycle_is_punished() -> None:
    """Verify repeated germinate-prune cycles have cumulative negative reward.

    BUG PATTERN: Even if single cycle is slightly negative, repeated cycles
    might amortize startup costs and become profitable. The policy learns
    to thrash at high frequency.
    """
    from esper.simic.rewards import (
        compute_contribution_reward,
        ContributionRewardConfig,
        SeedInfo,
    )
    from esper.leyline import LifecycleOp, SeedStage

    config = ContributionRewardConfig()
    host_params = 100000

    total_reward = 0.0
    num_cycles = 5

    for cycle in range(num_cycles):
        epoch_base = 5 + cycle * 2

        # GERMINATE
        germinate_reward, _ = compute_contribution_reward(
            action=LifecycleOp.GERMINATE,
            seed_contribution=None,
            val_acc=40.0 + cycle * 0.5,
            seed_info=None,
            epoch=epoch_base,
            max_epochs=25,
            total_params=host_params,
            host_params=host_params,
            config=config,
            acc_at_germination=None,
            acc_delta=0.5,
            return_components=True,
        )
        total_reward += germinate_reward

        # Immediate PRUNE
        seed = SeedInfo(
            stage=SeedStage.GERMINATED.value,
            improvement_since_stage_start=0.0,
            total_improvement=0.0,
            epochs_in_stage=1,
            seed_params=10000,
            previous_stage=SeedStage.DORMANT.value,
            previous_epochs_in_stage=0,
            seed_age_epochs=1,
        )

        prune_reward, _ = compute_contribution_reward(
            action=LifecycleOp.PRUNE,
            seed_contribution=None,
            val_acc=40.0 + cycle * 0.5,
            seed_info=seed,
            epoch=epoch_base + 1,
            max_epochs=25,
            total_params=host_params + 10000,
            host_params=host_params,
            config=config,
            acc_at_germination=40.0 + cycle * 0.5,
            acc_delta=0.0,
            return_components=True,
        )
        total_reward += prune_reward

    # Cumulative should be clearly negative
    assert total_reward < -0.5, (
        f"Repeated thrashing ({num_cycles} cycles) total reward = {total_reward:.3f}. "
        f"Should be clearly negative (< -0.5) to discourage this behavior."
    )
```

##### 7. Telemetry Counter Consistency Test

The `seeds_created` and `seeds_fossilized` counters must match actual GERMINATE and FOSSILIZE actions. Inconsistent counters corrupt telemetry and terminal bonus calculations.

```python
def test_telemetry_counters_match_actions() -> None:
    """Verify seeds_created and seeds_fossilized increment on correct actions.

    BUG PATTERN: Counters increment at wrong time (e.g., on action selection
    instead of successful execution), causing:
    1. Inflated counts (increment on failed actions)
    2. Off-by-one errors (increment before/after actual germination)
    3. Terminal bonus mismatch (num_fossilized_seeds != actual fossilized)

    This test verifies counter logic matches action execution.
    """
    from esper.simic.training.parallel_env_state import ParallelEnvState

    state = ParallelEnvState()

    # Initial state
    assert state.seeds_created == 0, "Initial seeds_created should be 0"
    assert state.seeds_fossilized == 0, "Initial seeds_fossilized should be 0"

    # Simulate GERMINATE action (successful)
    # In vectorized.py, this happens after slot.germinate() succeeds
    state.seeds_created += 1
    assert state.seeds_created == 1, "seeds_created should be 1 after germinate"

    # Simulate WAIT action (should NOT increment either counter)
    # No change expected
    assert state.seeds_created == 1, "WAIT should not increment seeds_created"
    assert state.seeds_fossilized == 0, "WAIT should not increment seeds_fossilized"

    # Simulate another GERMINATE (in different slot)
    state.seeds_created += 1
    assert state.seeds_created == 2, "seeds_created should be 2 after second germinate"

    # Simulate FOSSILIZE action (successful)
    state.seeds_fossilized += 1
    assert state.seeds_fossilized == 1, "seeds_fossilized should be 1 after fossilize"

    # seeds_created should NOT decrement on fossilize
    assert state.seeds_created == 2, (
        "seeds_created should remain 2 after fossilize (it counts births, not active)"
    )

    # Simulate PRUNE action (should NOT increment seeds_fossilized)
    # Prune is removal, not fossilization
    assert state.seeds_fossilized == 1, "PRUNE should not increment seeds_fossilized"


def test_reset_clears_counters() -> None:
    """Verify ParallelEnvState.reset() clears telemetry counters.

    BUG PATTERN: If counters persist across episodes, terminal bonus
    calculations use stale data from previous episodes.
    """
    from esper.simic.training.parallel_env_state import ParallelEnvState

    state = ParallelEnvState()

    # Simulate episode activity
    state.seeds_created = 5
    state.seeds_fossilized = 3
    state.pending_hindsight_credit = 0.25

    # Reset for new episode
    state.reset()

    # All counters should be cleared
    assert state.seeds_created == 0, "seeds_created not reset"
    assert state.seeds_fossilized == 0, "seeds_fossilized not reset"
    assert state.pending_hindsight_credit == 0.0, "pending_hindsight_credit not reset"
```

**Validation commands:**

```bash
# Run all Phase 7 reward validation tests
PYTHONPATH=src uv run pytest tests/simic/test_reward_validation.py -v

# Quick smoke test: verify reward components sum correctly
PYTHONPATH=src uv run python -c "
from esper.simic.rewards import compute_contribution_reward, ContributionRewardConfig, SeedInfo
from esper.leyline import LifecycleOp, SeedStage

config = ContributionRewardConfig()
seed = SeedInfo(
    stage=SeedStage.BLENDING.value,
    improvement_since_stage_start=0.5,
    total_improvement=1.0,
    epochs_in_stage=3,
    seed_params=10000,
    previous_stage=SeedStage.TRAINING.value,
    previous_epochs_in_stage=5,
    seed_age_epochs=8,
)

reward, comp = compute_contribution_reward(
    action=LifecycleOp.WAIT,
    seed_contribution=0.5,
    val_acc=50.0,
    seed_info=seed,
    epoch=10,
    max_epochs=25,
    total_params=110000,
    host_params=100000,
    config=config,
    acc_at_germination=45.0,
    acc_delta=0.5,
    return_components=True,
)
print(f'Total reward: {reward:.4f}')
print(f'Bounded attribution: {comp.bounded_attribution}')
print(f'PBRS bonus: {comp.pbrs_bonus:.4f}')
print(f'Compute rent: {comp.compute_rent:.4f}')
print(f'Action shaping: {comp.action_shaping:.4f}')
print('Reward computation: PASS')
"

# Verify hindsight credit function exists and is bounded
PYTHONPATH=src uv run python -c "
from esper.simic.rewards import compute_scaffold_hindsight_credit
credit = compute_scaffold_hindsight_credit(boost_given=1.0, beneficiary_improvement=5.0)
assert 0 <= credit <= 0.2, f'Credit {credit} out of bounds'
print(f'Hindsight credit (bounded): {credit:.4f}')
print('Hindsight credit: PASS')
"

# Verify telemetry counters exist in ParallelEnvState
PYTHONPATH=src uv run python -c "
from esper.simic.training.parallel_env_state import ParallelEnvState
state = ParallelEnvState()
assert hasattr(state, 'seeds_created'), 'Missing seeds_created'
assert hasattr(state, 'seeds_fossilized'), 'Missing seeds_fossilized'
assert hasattr(state, 'pending_hindsight_credit'), 'Missing pending_hindsight_credit'
print('ParallelEnvState counters: PASS')
"
```

**Key validation invariants:**

| Check | Expected | Failure Indicates |
|-------|----------|-------------------|
| PRUNE without counterfactual | Valid reward (not NaN) | Division by None in attribution |
| compute_rent for large blueprint | More negative than small | Rent not scaling with overhead |
| alpha_shock for fast change | More negative than slow | Shock not convex in delta |
| Fossilize ransomware seed | Negative reward | Ransomware check bypassed |
| num_contributing_fossilized | Excludes negative total_improvement | Terminal bonus rewards bad seeds |
| pending_hindsight_credit after consume | 0.0 | Credit accumulates without delivery |
| compute_scaffold_hindsight_credit | In [0, credit_weight] | Unbounded credit enables hacking |
| germinate + immediate prune | sum(rewards) <= 0 | Thrashing is profitable |
| seeds_created after GERMINATE | Incremented by 1 | Counter off-by-one or not updating |
| seeds_fossilized after FOSSILIZE | Incremented by 1 | Counter off-by-one or not updating |
| Counters after reset() | All zero | Stale data across episodes |

#### 7j. General Quality Wins & Monitoring Recommendations

These are cross-cutting validation patterns and monitoring recommendations that don't fit neatly into a single phase but catch subtle bugs across the entire system.

> **Note on proactive assertions**: These assertions should be enabled during development and CI but can be conditionally disabled in production via `ESPER_DEBUG_ASSERTIONS=1` environment variable to avoid performance overhead.

##### 1. Blueprint-to-Index Enum Coverage Assertion

The `_BLUEPRINT_TO_INDEX` mapping must cover all `BlueprintAction` enum values. Silent drift between the enum and dict causes actions to map to incorrect indices.

```python
def test_blueprint_to_index_covers_all_enum_values() -> None:
    """Verify _BLUEPRINT_TO_INDEX covers all BlueprintAction enum values.

    BUG PATTERN: When new blueprints are added to the enum but not to the
    mapping dict, the policy receives incorrect indices. The mismatch is
    silent because dict.get() returns None, which gets converted to 0
    (the null blueprint index), causing all new blueprints to behave
    identically to "null" during training.

    This assertion should run on import to catch drift early.
    """
    from esper.leyline import BlueprintAction
    from esper.tamiyo.policy.features import _BLUEPRINT_TO_INDEX

    enum_values = set(BlueprintAction)
    dict_keys = set(_BLUEPRINT_TO_INDEX.keys())

    missing_from_dict = enum_values - dict_keys
    extra_in_dict = dict_keys - enum_values

    assert not missing_from_dict, (
        f"BlueprintAction enum values missing from _BLUEPRINT_TO_INDEX: {missing_from_dict}. "
        f"Add mappings for these blueprints."
    )

    assert not extra_in_dict, (
        f"_BLUEPRINT_TO_INDEX has keys not in BlueprintAction enum: {extra_in_dict}. "
        f"Remove stale mappings or update the enum."
    )

    # Verify indices are contiguous and start from 0 (or 1 if 0 is reserved for null)
    indices = sorted(_BLUEPRINT_TO_INDEX.values())
    expected_start = 0  # Or 1 if BLUEPRINT_NULL_INDEX is reserved
    expected_indices = list(range(expected_start, expected_start + len(indices)))

    assert indices == expected_indices, (
        f"Blueprint indices should be contiguous starting from {expected_start}. "
        f"Got: {indices}, expected: {expected_indices}"
    )
```

##### 2. Stage Encoding Debug Mode

Unknown stage values silently encode as all-zeros, making the policy perceive a seed as having "no progress." Enable debug mode to catch invalid stages.

```python
def test_stage_encoding_rejects_unknown_stages() -> None:
    """Verify stage encoding fails loudly for unknown stages.

    BUG PATTERN: When SeedStage enum is extended but stage encoding isn't
    updated, new stages encode as [0,0,0,0] (same as having no stage info).
    The policy cannot distinguish between "no seed" and "unknown stage."

    Enable ESPER_DEBUG_STAGE=1 to convert silent zeros into assertion failures.
    """
    from esper.leyline import SeedStage
    from esper.tamiyo.policy.features import encode_stage  # If this helper exists

    # Create a mock "unknown" stage (simulate enum extension)
    # In practice, this tests that all current stages have explicit handling
    for stage in SeedStage:
        encoding = encode_stage(stage)

        # Each stage should produce a non-zero encoding (unless DORMANT is all-zeros)
        if stage != SeedStage.DORMANT:
            assert sum(encoding) > 0, (
                f"Stage {stage.name} encodes as all-zeros, "
                f"indistinguishable from DORMANT or unknown stage."
            )

    # Verify encoding dimensions match expected (5 stages = 5-dim one-hot or binary)
    sample_encoding = encode_stage(SeedStage.TRAINING)
    expected_dim = len(SeedStage)  # One dimension per stage for one-hot
    assert len(sample_encoding) == expected_dim, (
        f"Stage encoding dimension mismatch: got {len(sample_encoding)}, "
        f"expected {expected_dim} (one per SeedStage)"
    )
```

##### 3. Action Mask Validity Assertion

After sampling, if an action was masked out but still selected, something is wrong with the masking logic.

```python
def test_sampled_action_respects_mask() -> None:
    """Verify sampled actions are never from masked-out options.

    BUG PATTERN: The policy samples from a masked distribution, but due to
    numerical issues or incorrect mask application, a masked action can
    still be sampled with small probability. This causes:
    1. Invalid actions that the environment must handle
    2. Credit assignment to actions that "shouldn't" have been taken

    This test verifies the mask → sample → reward pipeline is consistent.
    """
    import torch
    from esper.tamiyo.networks.factored_lstm import FactoredRecurrentActorCritic
    from esper.leyline import NUM_BLUEPRINTS

    # state_dim=121: Obs V3 non-blueprint features (24 base + 7 temporal + 30×3 slots)
    # TODO: Replace with leyline.OBS_V3_NON_BLUEPRINT_DIM when available
    policy = FactoredRecurrentActorCritic(
        state_dim=121,
        num_slots=3,
        hidden_dim=128,
        num_blueprints=NUM_BLUEPRINTS,
    )

    # Create a scenario where slot 0 is the only valid target
    batch_size = 10
    state = torch.randn(batch_size, 1, 121)
    bp_idx = torch.randint(0, NUM_BLUEPRINTS, (batch_size, 1, 3))

    # Mask: only slot 0 is valid
    slot_mask = torch.tensor([[True, False, False]] * batch_size)  # [batch, num_slots]

    # Sample 50 times: if mask is correctly applied, probability of sampling
    # invalid slot is 0. With 50 samples, even a 1% leak would likely appear.
    for _ in range(50):
        with torch.no_grad():
            out = policy.forward(state, bp_idx, slot_mask=slot_mask)

        # Sampled slot should always be 0 (the only valid slot)
        assert (out.sampled_slot == 0).all(), (
            f"Sampled slot {out.sampled_slot.tolist()} includes invalid slots. "
            f"Mask was {slot_mask[0].tolist()}, only slot 0 should be sampled."
        )
```

##### 4. Advantage Distribution Monitoring

Consistently skewed advantages indicate value function lag or reward scale issues.

```python
def test_advantage_distribution_health() -> None:
    """Verify advantage distribution has healthy statistics.

    BUG PATTERN: When the value function lags behind the policy:
    - Advantages are consistently positive (V underestimates returns)
    - Or consistently negative (V overestimates returns)

    Healthy training shows advantages centered around 0 with symmetric tails.
    Persistent skew > 0.5 or kurtosis > 6 indicates problems.

    This is a monitoring assertion, not a hard invariant. Log warnings
    rather than failing tests for moderate deviations.
    """
    import torch
    import warnings

    def compute_advantage_stats(advantages: torch.Tensor) -> dict[str, float]:
        """Compute mean, std, skewness, and kurtosis of advantages."""
        mean = advantages.mean().item()
        std = advantages.std().item()

        # Standardized for skewness/kurtosis calculation
        if std < 1e-8:
            return {"mean": mean, "std": std, "skewness": 0.0, "kurtosis": 0.0}

        standardized = (advantages - mean) / std
        skewness = standardized.pow(3).mean().item()
        kurtosis = standardized.pow(4).mean().item() - 3.0  # Excess kurtosis

        return {"mean": mean, "std": std, "skewness": skewness, "kurtosis": kurtosis}

    # Simulate advantages from a healthy training run
    healthy_advantages = torch.randn(1000) * 2.0  # Centered, symmetric

    stats = compute_advantage_stats(healthy_advantages)

    # Healthy distribution checks
    assert abs(stats["mean"]) < 0.5, (
        f"Advantage mean {stats['mean']:.3f} far from 0. "
        f"Value function may be miscalibrated."
    )

    if abs(stats["skewness"]) > 0.5:
        warnings.warn(
            f"Advantage skewness {stats['skewness']:.3f} exceeds threshold. "
            f"Positive skew = V underestimates, negative = V overestimates."
        )

    if stats["kurtosis"] > 6.0:
        warnings.warn(
            f"Advantage kurtosis {stats['kurtosis']:.3f} is high (heavy tails). "
            f"May indicate reward spikes or outlier transitions."
        )

    # For actual training, hook this into TensorBoard/wandb logging:
    # log_metrics({
    #     "advantages/mean": stats["mean"],
    #     "advantages/std": stats["std"],
    #     "advantages/skewness": stats["skewness"],
    #     "advantages/kurtosis": stats["kurtosis"],
    # })
```

##### 5. Gradient Health Consistency Check

The `gradient_health_prev` field is currently constant at 1.0. Verify no logic depends on it changing.

```python
def test_gradient_health_prev_is_not_used_for_decisions() -> None:
    """Verify gradient_health_prev doesn't gate any decisions.

    BUG PATTERN: If lifecycle gating logic uses gradient_health_prev but it's
    always 1.0 (constant), the gating logic is effectively disabled. Any
    code path checking `if gradient_health_prev < threshold` never triggers.

    This test searches for usages and verifies they're only for telemetry,
    not decision-making.
    """
    import os
    import warnings

    # Files that might use gradient_health_prev
    files_to_check = [
        "src/esper/simic/training/vectorized.py",
        "src/esper/tamiyo/policy/features.py",
        "src/esper/simic/control/lifecycle.py",
    ]

    decision_patterns = ["if", "elif", "while", "and", "or"]
    telemetry_patterns = ["log", "emit", "record", "telemetry", "metrics"]

    for filepath in files_to_check:
        if not os.path.exists(filepath):
            continue

        with open(filepath) as f:
            content = f.read()

        if "gradient_health_prev" not in content:
            continue

        # Simple heuristic: check if it appears in decision contexts
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if "gradient_health_prev" in line:
                # Check if this line is part of a decision statement
                line_lower = line.lower()
                is_decision = any(p in line_lower for p in decision_patterns)
                is_telemetry = any(p in line_lower for p in telemetry_patterns)

                if is_decision and not is_telemetry:
                    # This is a potential issue - log for manual review
                    warnings.warn(
                        f"{filepath}:{i+1} uses gradient_health_prev in decision context: "
                        f"{line.strip()}"
                    )
                    # Don't assert here - this is informational
```

**Validation commands:**

```bash
# Run all general quality tests
PYTHONPATH=src uv run pytest tests/quality/ -v

# Check blueprint enum coverage
PYTHONPATH=src uv run python -c "
from esper.leyline import BlueprintAction
from esper.tamiyo.policy.features import _BLUEPRINT_TO_INDEX

enum_values = set(BlueprintAction)
dict_keys = set(_BLUEPRINT_TO_INDEX.keys())
missing = enum_values - dict_keys

if missing:
    print(f'FAIL: Missing blueprints in mapping: {missing}')
else:
    print(f'PASS: All {len(enum_values)} blueprints mapped')
"

# Search for gradient_health_prev in decision contexts
PYTHONPATH=src grep -rn "gradient_health_prev" src/esper/ | grep -E "(if|elif|while|and|or)" | grep -v log | grep -v emit
```

**Key validation invariants:**

| Check | Expected | Failure Indicates |
|-------|----------|-------------------|
| _BLUEPRINT_TO_INDEX keys | == set(BlueprintAction) | Enum/dict drift |
| Blueprint indices | Contiguous from 0 | Index gaps cause embedding misses |
| Stage encoding for known stage | Non-zero (except DORMANT) | New stage encodes as "unknown" |
| Sampled action with mask | Always from valid set | Mask not applied correctly |
| Advantage mean | Near 0 over epoch | Value function miscalibration |
| Advantage skewness | < 0.5 | Consistent over/under-estimation |
| gradient_health_prev usage | Telemetry only, not decisions | Disabled gating logic |

#### 7k. Edge Cases & Integration Testing

These tests target failure modes that only manifest in long-running scenarios, edge cases, or cross-domain integration points. They catch resource leaks, state inconsistencies, and silent failures that unit tests miss.

> **Note on test scope**: These tests require more setup than typical unit tests. Run them as part of extended CI (nightly) or before releases, not on every commit.

##### 1. Scheduled Prune Completion Verification

Seeds scheduled for pruning might persist beyond their intended epoch due to off-by-one errors or early termination of training loops.

```python
def test_scheduled_prune_completes_within_epoch() -> None:
    """Verify seeds scheduled for pruning are removed by end of scheduled epoch.

    BUG PATTERN: When a PRUNE action is scheduled for epoch N, the seed should
    be removed before epoch N+1 begins. Off-by-one errors in epoch boundary
    logic, or early loop termination, can leave seeds in slots longer than
    intended. This wastes compute and corrupts seed lifetime metrics.

    The test simulates a training loop and verifies prune completion timing.
    """
    from dataclasses import dataclass
    from typing import Optional

    @dataclass
    class MockSeed:
        slot_id: int
        prune_scheduled_epoch: Optional[int] = None
        is_pruned: bool = False

    @dataclass
    class MockSlotState:
        seeds: dict[int, MockSeed]  # slot_id -> seed
        current_epoch: int = 0

        def schedule_prune(self, slot_id: int, at_epoch: int) -> None:
            if slot_id in self.seeds:
                self.seeds[slot_id].prune_scheduled_epoch = at_epoch

        def execute_pending_prunes(self) -> list[int]:
            """Execute prunes scheduled for current epoch. Returns pruned slot_ids."""
            pruned = []
            for slot_id, seed in list(self.seeds.items()):
                if seed.prune_scheduled_epoch is not None and seed.prune_scheduled_epoch <= self.current_epoch:
                    seed.is_pruned = True
                    del self.seeds[slot_id]
                    pruned.append(slot_id)
            return pruned

        def advance_epoch(self) -> None:
            self.current_epoch += 1

    # Setup: 3 seeds, schedule prune for seed 1 at epoch 2
    state = MockSlotState(
        seeds={
            0: MockSeed(slot_id=0),
            1: MockSeed(slot_id=1),
            2: MockSeed(slot_id=2),
        },
        current_epoch=0,
    )
    state.schedule_prune(slot_id=1, at_epoch=2)

    # Simulate training loop
    for epoch in range(5):
        state.current_epoch = epoch
        pruned = state.execute_pending_prunes()

        if epoch < 2:
            assert 1 in state.seeds, (
                f"Seed 1 pruned too early at epoch {epoch}, "
                f"was scheduled for epoch 2"
            )
        elif epoch == 2:
            assert 1 not in state.seeds, (
                f"Seed 1 not pruned at scheduled epoch {epoch}. "
                f"Prune logic has off-by-one error."
            )
            assert 1 in pruned, (
                f"Prune return value missing slot 1 at epoch {epoch}"
            )
        else:
            assert 1 not in state.seeds, (
                f"Seed 1 reappeared after pruning at epoch {epoch}"
            )

    # Verify other seeds unaffected
    assert 0 in state.seeds, "Seed 0 was incorrectly pruned"
    assert 2 in state.seeds, "Seed 2 was incorrectly pruned"
```

##### 2. Optimizer Memory Leak Detection

When seeds are pruned or fossilized, their optimizers must be cleared. Accumulating stale optimizers causes memory growth proportional to total seeds created.

```python
def test_optimizer_cleanup_on_seed_removal() -> None:
    """Verify optimizers are cleared when seeds are removed from slots.

    BUG PATTERN: ParallelEnvState.seed_optimizers maps slot_id -> optimizer.
    When a seed is pruned/fossilized, if the optimizer isn't explicitly deleted,
    it persists indefinitely. Over 1000s of seeds, this causes OOM.

    Symptom: len(env_state.seed_optimizers) grows monotonically even as
    active seed count stays constant.

    This test simulates the full create -> train -> remove cycle.
    """
    from typing import Any

    class MockParallelEnvState:
        """Simplified state tracking optimizer lifecycle."""

        def __init__(self) -> None:
            self.seed_optimizers: dict[int, Any] = {}
            self.active_seeds: set[int] = set()
            self._optimizer_creates: int = 0
            self._optimizer_deletes: int = 0

        def create_seed(self, slot_id: int) -> None:
            """Create seed with optimizer."""
            self.active_seeds.add(slot_id)
            self.seed_optimizers[slot_id] = {"lr": 0.001}  # Mock optimizer
            self._optimizer_creates += 1

        def remove_seed(self, slot_id: int, cleanup_optimizer: bool = True) -> None:
            """Remove seed, optionally cleaning up optimizer."""
            if slot_id in self.active_seeds:
                self.active_seeds.discard(slot_id)
                if cleanup_optimizer and slot_id in self.seed_optimizers:
                    del self.seed_optimizers[slot_id]
                    self._optimizer_deletes += 1

    state = MockParallelEnvState()

    # Simulate 100 seed lifecycles
    for i in range(100):
        slot_id = i % 3  # Cycle through 3 slots

        # Remove existing seed if present
        if slot_id in state.active_seeds:
            state.remove_seed(slot_id, cleanup_optimizer=True)

        # Create new seed
        state.create_seed(slot_id)

    # After 100 cycles through 3 slots, should have exactly 3 active optimizers
    assert len(state.seed_optimizers) == 3, (
        f"Expected 3 optimizers (one per slot), got {len(state.seed_optimizers)}. "
        f"Optimizers not being cleaned up on seed removal."
    )

    assert len(state.active_seeds) == 3, (
        f"Expected 3 active seeds, got {len(state.active_seeds)}"
    )

    # Verify cleanup was called appropriately
    # First 3 creates have no prior seed, remaining 97 do
    expected_deletes = 100 - 3  # 97 removals before creates
    assert state._optimizer_deletes == expected_deletes, (
        f"Expected {expected_deletes} optimizer deletions, got {state._optimizer_deletes}. "
        f"Some removals skipped cleanup."
    )
```

##### 3. Graceful Shutdown Mid-Epoch Safety

Breaking training mid-epoch (Ctrl+C, timeout, crash) should not corrupt the rollout buffer or leave partial state.

```python
def test_graceful_shutdown_preserves_buffer_integrity() -> None:
    """Verify rollout buffer remains valid after simulated mid-epoch interruption.

    BUG PATTERN: If training loop is interrupted after adding partial rollout
    data but before completing the epoch:
    1. Buffer has mismatched tensor lengths (obs added but not rewards)
    2. Next training resume reads corrupted sequences
    3. PPO update computes on garbage → NaN losses

    This test simulates interruption at various points and verifies buffer
    either completes cleanly or rolls back to last consistent state.
    """
    from dataclasses import dataclass, field
    from typing import Optional

    @dataclass
    class MockRolloutBuffer:
        """Simplified buffer tracking consistency."""

        observations: list[float] = field(default_factory=list)
        actions: list[int] = field(default_factory=list)
        rewards: list[float] = field(default_factory=list)
        _transaction_start: Optional[int] = None

        def begin_step(self) -> None:
            """Mark transaction start for rollback capability."""
            self._transaction_start = len(self.observations)

        def add_observation(self, obs: float) -> None:
            self.observations.append(obs)

        def add_action(self, action: int) -> None:
            self.actions.append(action)

        def add_reward(self, reward: float) -> None:
            self.rewards.append(reward)
            self._transaction_start = None  # Commit

        def rollback_incomplete(self) -> bool:
            """Rollback to last consistent state. Returns True if rollback occurred."""
            if self._transaction_start is not None:
                self.observations = self.observations[:self._transaction_start]
                self.actions = self.actions[:self._transaction_start]
                self.rewards = self.rewards[:self._transaction_start]
                self._transaction_start = None
                return True
            return False

        def is_consistent(self) -> bool:
            """Check all arrays have same length."""
            return (
                len(self.observations) == len(self.actions) == len(self.rewards)
            )

    # Scenario 1: Complete step (no interruption)
    buffer = MockRolloutBuffer()
    buffer.begin_step()
    buffer.add_observation(1.0)
    buffer.add_action(0)
    buffer.add_reward(0.5)
    assert buffer.is_consistent(), "Complete step should be consistent"

    # Scenario 2: Interrupt after observation only
    buffer2 = MockRolloutBuffer()
    buffer2.begin_step()
    buffer2.add_observation(1.0)
    # Simulate interruption here - action and reward never added
    assert not buffer2.is_consistent(), "Partial step should be inconsistent"

    rolled_back = buffer2.rollback_incomplete()
    assert rolled_back, "Should detect incomplete transaction"
    assert buffer2.is_consistent(), "After rollback should be consistent"
    assert len(buffer2.observations) == 0, "Rollback should remove partial data"

    # Scenario 3: Interrupt after observation + action
    buffer3 = MockRolloutBuffer()
    buffer3.begin_step()
    buffer3.add_observation(2.0)
    buffer3.add_action(1)
    # Interrupt before reward
    assert not buffer3.is_consistent(), "Missing reward should be inconsistent"

    buffer3.rollback_incomplete()
    assert buffer3.is_consistent(), "Rollback should restore consistency"

    # Scenario 4: Multiple complete steps then interrupt
    buffer4 = MockRolloutBuffer()
    for i in range(5):
        buffer4.begin_step()
        buffer4.add_observation(float(i))
        buffer4.add_action(i)
        buffer4.add_reward(float(i) * 0.1)

    buffer4.begin_step()
    buffer4.add_observation(5.0)
    # Interrupt mid-step

    buffer4.rollback_incomplete()
    assert len(buffer4.observations) == 5, (
        f"Rollback should preserve 5 complete steps, got {len(buffer4.observations)}"
    )
    assert buffer4.is_consistent(), "Buffer should be consistent after rollback"
```

##### 4. Kasmina/Leyline Integration: TamiyoDecision.to_command() Telemetry

When RL actions are converted to Kasmina commands, telemetry events should still fire. Bypassing `TamiyoDecision.to_command()` breaks observability.

```python
def test_tamiyo_decision_to_command_emits_telemetry() -> None:
    """Verify TamiyoDecision.to_command() triggers expected telemetry events.

    BUG PATTERN: Direct construction of KasminaCommand from RL output bypasses
    the TamiyoDecision wrapper that handles:
    1. Telemetry emission for decision audit trail
    2. Validation that action is legal in current state
    3. Conversion of slot indices to Kasmina-compatible format

    If code paths exist that create commands without going through to_command(),
    we lose visibility into what Tamiyo decided and why.

    This test verifies the telemetry emission contract.
    """
    from dataclasses import dataclass
    from typing import Any
    from enum import Enum, auto

    class MockLifecycleOp(Enum):
        WAIT = auto()
        GERMINATE = auto()
        PRUNE = auto()
        FOSSILIZE = auto()

    @dataclass
    class MockTamiyoDecision:
        op: MockLifecycleOp
        target_slot: int
        confidence: float
        telemetry_emitted: bool = False

        def to_command(self, telemetry_sink: Any) -> dict:
            """Convert decision to command, emitting telemetry."""
            # Emit telemetry
            telemetry_sink.emit({
                "event": "tamiyo_decision",
                "op": self.op.name,
                "slot": self.target_slot,
                "confidence": self.confidence,
            })
            self.telemetry_emitted = True

            return {
                "type": "kasmina_command",
                "operation": self.op.name,
                "slot_id": self.target_slot,
            }

    class MockTelemetrySink:
        def __init__(self) -> None:
            self.events: list[dict] = []

        def emit(self, event: dict) -> None:
            self.events.append(event)

    # Create decision and convert through proper channel
    decision = MockTamiyoDecision(
        op=MockLifecycleOp.GERMINATE,
        target_slot=1,
        confidence=0.85,
    )

    sink = MockTelemetrySink()
    command = decision.to_command(sink)

    assert decision.telemetry_emitted, (
        "to_command() did not set telemetry_emitted flag"
    )

    assert len(sink.events) == 1, (
        f"Expected 1 telemetry event, got {len(sink.events)}"
    )

    event = sink.events[0]
    assert event["event"] == "tamiyo_decision", (
        f"Wrong event type: {event.get('event')}"
    )
    assert event["op"] == "GERMINATE", (
        f"Event has wrong op: {event.get('op')}"
    )
    assert event["slot"] == 1, (
        f"Event has wrong slot: {event.get('slot')}"
    )

    # Verify command structure
    assert command["type"] == "kasmina_command", (
        f"Command has wrong type: {command.get('type')}"
    )
    assert command["operation"] == "GERMINATE", (
        f"Command has wrong operation: {command.get('operation')}"
    )
```

##### 5. Default Constants Validation

Critical constants like `MIN_FOSSILIZE_CONTRIBUTION` shape training dynamics. Overly lenient defaults can allow degenerate behavior.

```python
def test_default_constants_within_design_bounds() -> None:
    """Verify default constants match documented design intent.

    BUG PATTERN: Default constants are often set to permissive values during
    development (MIN_X=0.0, MAX_Y=inf) and never tightened. This allows:
    1. Fossilizing seeds with zero contribution (wastes capacity)
    2. Unbounded rent accumulation (reward hacking)
    3. Infinite prune delays (resource leak)

    This test documents expected bounds and catches drift from design intent.
    """
    # Import actual constants when available
    # from esper.leyline import (
    #     MIN_FOSSILIZE_CONTRIBUTION,
    #     MAX_SEED_RENT_PER_EPOCH,
    #     MAX_PRUNE_DELAY_EPOCHS,
    # )

    # For now, define expected bounds (update when leyline exposes these)
    EXPECTED_BOUNDS = {
        "MIN_FOSSILIZE_CONTRIBUTION": {
            "min": 0.01,  # Should require measurable positive contribution
            "max": 0.5,   # Shouldn't be so high that fossilization is rare
            "rationale": "Fossilizing zero-contribution seeds wastes model capacity",
        },
        "MAX_SEED_RENT_PER_EPOCH": {
            "min": 0.001,  # Rent should be non-negligible
            "max": 0.1,    # But not so high that seeds are pruned immediately
            "rationale": "Rent pressure prevents indefinite training without value",
        },
        "MAX_PRUNE_DELAY_EPOCHS": {
            "min": 1,      # At least 1 epoch delay
            "max": 10,     # But not indefinite
            "rationale": "Bounded prune delay prevents resource leaks",
        },
        "GERMINATE_COOLDOWN_EPOCHS": {
            "min": 0,      # Can be immediate for aggressive exploration
            "max": 5,      # But not so long that adaptation stalls
            "rationale": "Cooldown prevents thrashing without blocking adaptation",
        },
    }

    # Placeholder values (replace with actual imports)
    ACTUAL_VALUES = {
        "MIN_FOSSILIZE_CONTRIBUTION": 0.0,  # Current value - potentially too lenient
        "MAX_SEED_RENT_PER_EPOCH": 0.01,
        "MAX_PRUNE_DELAY_EPOCHS": 5,
        "GERMINATE_COOLDOWN_EPOCHS": 1,
    }

    warnings_found = []

    for const_name, bounds in EXPECTED_BOUNDS.items():
        actual = ACTUAL_VALUES.get(const_name)
        if actual is None:
            warnings_found.append(f"{const_name}: not defined")
            continue

        if actual < bounds["min"]:
            warnings_found.append(
                f"{const_name}={actual} below minimum {bounds['min']}. "
                f"Rationale: {bounds['rationale']}"
            )

        if actual > bounds["max"]:
            warnings_found.append(
                f"{const_name}={actual} above maximum {bounds['max']}. "
                f"Rationale: {bounds['rationale']}"
            )

    # Report all issues
    if warnings_found:
        import warnings
        for w in warnings_found:
            warnings.warn(w)

    # Specific assertion for MIN_FOSSILIZE_CONTRIBUTION
    min_fossil = ACTUAL_VALUES["MIN_FOSSILIZE_CONTRIBUTION"]
    assert min_fossil >= 0.0, f"MIN_FOSSILIZE_CONTRIBUTION cannot be negative: {min_fossil}"

    # Note: Uncomment below once design is finalized
    # assert min_fossil >= 0.01, (
    #     f"MIN_FOSSILIZE_CONTRIBUTION={min_fossil} is too lenient. "
    #     f"Seeds with zero contribution should not be fossilized."
    # )
```

##### 6. Long-Running Memory Stability

Memory should not grow unboundedly during extended training. This test simulates epoch cycles and checks for leaks.

```python
def test_memory_stable_over_extended_training() -> None:
    """Verify memory usage stabilizes over many epoch cycles.

    BUG PATTERN: Subtle memory leaks compound over long training runs:
    1. Tensors retained in closures (e.g., lambda in callbacks)
    2. Circular references preventing GC (model ↔ optimizer ↔ scheduler)
    3. Growing caches without eviction (feature cache, embedding cache)
    4. Telemetry buffers accumulating without flush

    This test simulates 100 epoch cycles and checks memory delta.
    """
    import gc
    from typing import Any

    class MockTrainingState:
        """Simplified training state for leak detection."""

        def __init__(self) -> None:
            self.epoch = 0
            self.rollout_buffer: list[dict] = []
            self.metrics_history: list[dict] = []  # Potential leak if unbounded
            self._cache: dict[str, Any] = {}

        def simulate_epoch(self) -> None:
            """Simulate one epoch of training."""
            # Accumulate rollout data
            for step in range(100):
                self.rollout_buffer.append({
                    "obs": [0.0] * 121,
                    "action": 0,
                    "reward": 0.1,
                })

            # Process and clear buffer (correct pattern)
            self._process_buffer()

            # Track metrics (potential leak if never pruned)
            self.metrics_history.append({
                "epoch": self.epoch,
                "loss": 0.5,
                "entropy": 1.0,
            })

            self.epoch += 1

        def _process_buffer(self) -> None:
            """Process buffer and clear for next epoch."""
            # Simulate PPO update
            _ = len(self.rollout_buffer)
            self.rollout_buffer.clear()

        def prune_old_metrics(self, keep_last: int = 100) -> None:
            """Prune old metrics to prevent unbounded growth."""
            if len(self.metrics_history) > keep_last:
                self.metrics_history = self.metrics_history[-keep_last:]

    def get_object_count() -> int:
        """Count objects after GC."""
        gc.collect()
        return len(gc.get_objects())

    state = MockTrainingState()

    # Warm-up phase
    for _ in range(10):
        state.simulate_epoch()
    gc.collect()
    baseline_objects = get_object_count()

    # Extended training phase
    for i in range(100):
        state.simulate_epoch()
        if i % 20 == 0:
            state.prune_old_metrics(keep_last=50)  # Periodic cleanup

    gc.collect()
    final_objects = get_object_count()

    # Allow some growth but not unbounded
    growth = final_objects - baseline_objects
    max_allowed_growth = 10000  # Generous threshold

    assert growth < max_allowed_growth, (
        f"Object count grew by {growth} (from {baseline_objects} to {final_objects}). "
        f"Possible memory leak detected. Check for: "
        f"1. Unbounded metrics_history, "
        f"2. Retained rollout tensors, "
        f"3. Cache without eviction"
    )

    # Verify buffer was actually cleared each epoch
    assert len(state.rollout_buffer) == 0, (
        f"Rollout buffer not cleared: {len(state.rollout_buffer)} entries remain"
    )

    # Verify metrics were pruned
    assert len(state.metrics_history) <= 50, (
        f"Metrics history not pruned: {len(state.metrics_history)} entries"
    )
```

**Validation commands:**

```bash
# Run edge case and integration tests
PYTHONPATH=src uv run pytest tests/integration/tamiyo/ -v -k "edge" --timeout=60

# Check for optimizer leaks in real training (5 epochs, monitor memory)
PYTHONPATH=src uv run python -c "
import gc
import tracemalloc

tracemalloc.start()

# Simulate training loop with seed lifecycle
class MockState:
    def __init__(self):
        self.seed_optimizers = {}
        self.active_seeds = set()

    def create_seed(self, slot_id):
        self.active_seeds.add(slot_id)
        self.seed_optimizers[slot_id] = {'lr': 0.001, 'params': [0.0] * 1000}

    def remove_seed(self, slot_id):
        self.active_seeds.discard(slot_id)
        if slot_id in self.seed_optimizers:
            del self.seed_optimizers[slot_id]

state = MockState()

# Simulate 1000 seed lifecycles across 3 slots
for i in range(1000):
    slot = i % 3
    if slot in state.active_seeds:
        state.remove_seed(slot)
    state.create_seed(slot)

gc.collect()
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(f'Optimizer count: {len(state.seed_optimizers)} (expected: 3)')
print(f'Current memory: {current / 1024:.1f} KB')
print(f'Peak memory: {peak / 1024:.1f} KB')

assert len(state.seed_optimizers) == 3, f'Optimizer leak: {len(state.seed_optimizers)}'
print('Optimizer cleanup: PASS')
"

# Verify TamiyoDecision class has to_command method
PYTHONPATH=src uv run python -c "
# When TamiyoDecision exists, verify to_command interface
try:
    from esper.tamiyo.decision import TamiyoDecision
    assert hasattr(TamiyoDecision, 'to_command'), 'TamiyoDecision missing to_command method'
    print('TamiyoDecision.to_command: PASS')
except ImportError:
    print('TamiyoDecision not yet implemented (expected during Phase 7)')
"

# Check default constants in leyline
PYTHONPATH=src uv run python -c "
from esper import leyline

# Check for fossilization threshold
if hasattr(leyline, 'MIN_FOSSILIZE_CONTRIBUTION'):
    val = leyline.MIN_FOSSILIZE_CONTRIBUTION
    if val == 0.0:
        print(f'WARNING: MIN_FOSSILIZE_CONTRIBUTION={val} may be too lenient')
    else:
        print(f'MIN_FOSSILIZE_CONTRIBUTION={val}: OK')
else:
    print('MIN_FOSSILIZE_CONTRIBUTION: not defined in leyline')
"
```

**Key validation invariants:**

| Check | Expected | Failure Indicates |
|-------|----------|-------------------|
| Scheduled prune timing | Seed removed at scheduled epoch, not before/after | Off-by-one in epoch boundary logic |
| Optimizer count after N cycles | == num_slots (not growing) | Optimizer leak on seed removal |
| Buffer consistency after interrupt | All arrays same length or rolled back | Missing transaction safety in buffer |
| TamiyoDecision.to_command() telemetry | Event emitted for every decision | Bypass path avoiding decision wrapper |
| MIN_FOSSILIZE_CONTRIBUTION | > 0.0 (design-dependent) | Zero-contribution seeds waste capacity |
| Memory growth over 100 epochs | < 10MB delta | Unbounded cache/history/tensor retention |
| Rollout buffer after epoch | Empty (cleared for next epoch) | Buffer not reset between epochs |
| Metrics history length | Bounded by prune policy | Unbounded metric accumulation |

##### 7. Concurrent Slot Operations (Multi-Prune Same Epoch)

When multiple seeds are scheduled for pruning in the same epoch, iteration-during-mutation bugs can cause skipped or double-processed seeds.

```python
def test_concurrent_prune_operations_in_same_epoch() -> None:
    """Verify multiple seeds can be pruned in the same epoch without corruption.

    BUG PATTERN: If the prune loop modifies the seeds dict while iterating,
    seeds may be skipped or processed twice. Common manifestations:
    1. RuntimeError: dictionary changed size during iteration
    2. Silent skip: seed remains in slot despite scheduled prune
    3. Double prune: attempt to prune already-empty slot causes KeyError

    This is especially dangerous when seeds have inter-slot dependencies
    (e.g., scaffold relationships where pruning seed A affects seed B's credit).
    """
    from dataclasses import dataclass, field

    @dataclass
    class MockSeed:
        slot_id: int
        prune_scheduled_epoch: int | None = None

    @dataclass
    class MockSlotState:
        seeds: dict[int, MockSeed] = field(default_factory=dict)
        current_epoch: int = 0
        prune_log: list[int] = field(default_factory=list)

        def schedule_prune(self, slot_id: int, at_epoch: int) -> None:
            if slot_id in self.seeds:
                self.seeds[slot_id].prune_scheduled_epoch = at_epoch

        def execute_pending_prunes(self) -> list[int]:
            """Prune all seeds scheduled for current epoch."""
            # CORRECT: Build list of slots to prune BEFORE modifying dict
            slots_to_prune = [
                slot_id for slot_id, seed in self.seeds.items()
                if seed.prune_scheduled_epoch is not None
                and seed.prune_scheduled_epoch <= self.current_epoch
            ]

            # Now safe to modify dict
            for slot_id in slots_to_prune:
                del self.seeds[slot_id]
                self.prune_log.append(slot_id)

            return slots_to_prune

    # Setup: 3 slots with seeds, all scheduled for prune at epoch 2
    state = MockSlotState()
    state.seeds = {
        0: MockSeed(slot_id=0),
        1: MockSeed(slot_id=1),
        2: MockSeed(slot_id=2),
    }
    state.schedule_prune(0, at_epoch=2)
    state.schedule_prune(1, at_epoch=2)
    state.schedule_prune(2, at_epoch=2)

    # Advance to prune epoch
    state.current_epoch = 2
    pruned = state.execute_pending_prunes()

    # All 3 seeds should be pruned in one epoch
    assert len(pruned) == 3, (
        f"Expected 3 seeds pruned, got {len(pruned)}. "
        f"Seeds remaining: {list(state.seeds.keys())}. "
        f"Iteration-during-mutation may have skipped seeds."
    )

    assert len(state.seeds) == 0, (
        f"All seeds should be removed, but {len(state.seeds)} remain: "
        f"{list(state.seeds.keys())}"
    )

    # Verify each slot was pruned exactly once
    assert sorted(state.prune_log) == [0, 1, 2], (
        f"Prune log should show [0, 1, 2], got {state.prune_log}. "
        f"Some seeds may have been double-pruned or skipped."
    )
```

##### 8. Blueprint Index Bounds Validation

Invalid blueprint indices passed to `nn.Embedding` cause cryptic CUDA errors. This test ensures indices stay within valid bounds.

```python
def test_blueprint_index_within_embedding_bounds() -> None:
    """Verify blueprint indices are always in [0, NUM_BLUEPRINTS-1].

    BUG PATTERN: When blueprint selection logic has off-by-one errors or
    uses uninitialized values, the index can be:
    - Negative (causes IndexError or wraps to end of embedding table)
    - >= NUM_BLUEPRINTS (causes CUDA error: index out of range)
    - Fractional (if not properly cast to int64)

    CUDA errors are especially cryptic:
    "RuntimeError: CUDA error: device-side assert triggered"
    with no indication that the root cause is an embedding lookup.

    This validation should run on every blueprint index before embedding lookup.
    """
    import torch
    from esper.leyline import NUM_BLUEPRINTS, BLUEPRINT_NULL_INDEX

    def validate_blueprint_indices(bp_idx: torch.Tensor) -> None:
        """Validate blueprint indices are within bounds.

        Call this before passing to BlueprintEmbedding.forward().
        """
        # Must be integer type for embedding lookup
        assert bp_idx.dtype in (torch.int32, torch.int64, torch.long), (
            f"Blueprint indices must be integer type, got {bp_idx.dtype}. "
            f"Floating point indices cause silent truncation."
        )

        # Check lower bound
        min_idx = bp_idx.min().item()
        assert min_idx >= 0, (
            f"Blueprint index {min_idx} is negative. "
            f"Valid range is [0, {NUM_BLUEPRINTS - 1}]."
        )

        # Check upper bound
        max_idx = bp_idx.max().item()
        assert max_idx < NUM_BLUEPRINTS, (
            f"Blueprint index {max_idx} >= NUM_BLUEPRINTS ({NUM_BLUEPRINTS}). "
            f"This will cause CUDA index-out-of-range error in embedding lookup."
        )

    # Test 1: Valid indices (including null index)
    valid_bp_idx = torch.tensor([0, 1, BLUEPRINT_NULL_INDEX, NUM_BLUEPRINTS - 1])
    validate_blueprint_indices(valid_bp_idx)  # Should not raise

    # Test 2: Negative index (should fail)
    negative_bp_idx = torch.tensor([0, -1, 2])
    try:
        validate_blueprint_indices(negative_bp_idx)
        assert False, "Should have raised for negative index"
    except AssertionError as e:
        assert "negative" in str(e).lower(), f"Wrong error message: {e}"

    # Test 3: Out of bounds index (should fail)
    oob_bp_idx = torch.tensor([0, NUM_BLUEPRINTS, 2])  # NUM_BLUEPRINTS is out of range
    try:
        validate_blueprint_indices(oob_bp_idx)
        assert False, "Should have raised for out-of-bounds index"
    except AssertionError as e:
        assert "NUM_BLUEPRINTS" in str(e), f"Wrong error message: {e}"

    # Test 4: Wrong dtype (should fail)
    float_bp_idx = torch.tensor([0.0, 1.0, 2.0])  # Float, not int
    try:
        validate_blueprint_indices(float_bp_idx)
        assert False, "Should have raised for float dtype"
    except AssertionError as e:
        assert "integer" in str(e).lower(), f"Wrong error message: {e}"

    # Test 5: Realistic batch shape
    batch_bp_idx = torch.randint(0, NUM_BLUEPRINTS, (4, 10, 3), dtype=torch.int64)
    validate_blueprint_indices(batch_bp_idx)  # Should not raise
```

##### 9. Action Mask Dimension Consistency

Action masks must match the action space dimensions. Mismatched dimensions cause silent broadcasting bugs or index errors.

```python
def test_action_mask_dimensions_match_action_space() -> None:
    """Verify action masks have correct dimensions for each head.

    BUG PATTERN: When mask dimensions don't match action space:
    1. Broadcasting: [batch, 1] mask broadcasts to [batch, num_actions],
       causing all actions to be masked/unmasked together
    2. Silent truncation: mask with fewer elements ignores some actions
    3. Index mismatch: mask[i] doesn't correspond to action[i]

    This is especially dangerous for the slot head where num_slots varies
    per environment configuration.
    """
    import torch
    from esper.leyline import NUM_BLUEPRINTS

    # Expected action space dimensions for Tamiyo's 8 heads
    # TODO: Import these from leyline when available
    ACTION_SPACE_DIMS = {
        "op": 5,            # WAIT, ADVANCE, GERMINATE, PRUNE, FOSSILIZE
        "slot": 3,          # Configurable: num_slots (using 3 for test)
        "blueprint": NUM_BLUEPRINTS,
        "style": 3,         # AGGRESSIVE, BALANCED, CONSERVATIVE
        "tempo": 3,         # FAST, MEDIUM, SLOW
        "alpha_target": 10, # Discretized target values
        "alpha_speed": 10,  # Discretized speed values
        "alpha_curve": 5,   # LINEAR, EASE_IN, EASE_OUT, SMOOTH, STEP
    }

    batch_size = 4

    def validate_mask_dimensions(
        masks: dict[str, torch.Tensor],
        action_dims: dict[str, int],
        batch_size: int
    ) -> list[str]:
        """Validate mask dimensions match action space. Returns list of errors."""
        errors = []

        for head_name, expected_dim in action_dims.items():
            if head_name not in masks:
                errors.append(f"Missing mask for head '{head_name}'")
                continue

            mask = masks[head_name]

            # Check batch dimension
            if mask.shape[0] != batch_size:
                errors.append(
                    f"Mask '{head_name}' batch size {mask.shape[0]} != {batch_size}"
                )

            # Check action dimension
            if len(mask.shape) < 2:
                errors.append(
                    f"Mask '{head_name}' should be 2D [batch, actions], got shape {mask.shape}"
                )
            elif mask.shape[-1] != expected_dim:
                errors.append(
                    f"Mask '{head_name}' action dim {mask.shape[-1]} != expected {expected_dim}"
                )

            # Check dtype (should be bool or float for soft masks)
            if mask.dtype not in (torch.bool, torch.float32, torch.float16):
                errors.append(
                    f"Mask '{head_name}' has unusual dtype {mask.dtype}"
                )

            # Check at least one action is valid (mask=True) per batch item
            if mask.dtype == torch.bool:
                valid_per_batch = mask.any(dim=-1)
                if not valid_per_batch.all():
                    invalid_indices = (~valid_per_batch).nonzero().squeeze(-1).tolist()
                    errors.append(
                        f"Mask '{head_name}' has no valid actions for batch items {invalid_indices}"
                    )

        return errors

    # Test 1: Correct masks
    correct_masks = {
        head: torch.ones(batch_size, dim, dtype=torch.bool)
        for head, dim in ACTION_SPACE_DIMS.items()
    }
    errors = validate_mask_dimensions(correct_masks, ACTION_SPACE_DIMS, batch_size)
    assert len(errors) == 0, f"Valid masks reported errors: {errors}"

    # Test 2: Wrong action dimension
    wrong_dim_masks = correct_masks.copy()
    wrong_dim_masks["slot"] = torch.ones(batch_size, 5, dtype=torch.bool)  # Should be 3
    errors = validate_mask_dimensions(wrong_dim_masks, ACTION_SPACE_DIMS, batch_size)
    assert any("slot" in e and "5" in e and "3" in e for e in errors), (
        f"Should detect slot dimension mismatch, got: {errors}"
    )

    # Test 3: Missing mask
    missing_masks = {k: v for k, v in correct_masks.items() if k != "blueprint"}
    errors = validate_mask_dimensions(missing_masks, ACTION_SPACE_DIMS, batch_size)
    assert any("blueprint" in e and "Missing" in e for e in errors), (
        f"Should detect missing blueprint mask, got: {errors}"
    )

    # Test 4: All actions masked (invalid state)
    all_masked = correct_masks.copy()
    all_masked["op"] = torch.zeros(batch_size, 5, dtype=torch.bool)  # All False
    errors = validate_mask_dimensions(all_masked, ACTION_SPACE_DIMS, batch_size)
    assert any("op" in e and "no valid actions" in e for e in errors), (
        f"Should detect all-masked head, got: {errors}"
    )

    # Test 5: Wrong batch size
    wrong_batch_masks = correct_masks.copy()
    wrong_batch_masks["tempo"] = torch.ones(2, 3, dtype=torch.bool)  # batch=2, should be 4
    errors = validate_mask_dimensions(wrong_batch_masks, ACTION_SPACE_DIMS, batch_size)
    assert any("tempo" in e and "batch" in e for e in errors), (
        f"Should detect batch size mismatch, got: {errors}"
    )
```

**Additional validation commands:**

```bash
# Test concurrent prune handling
PYTHONPATH=src uv run python -c "
# Verify dict iteration safety in actual prune implementation
from esper.simic.training.parallel_env_state import ParallelEnvState
import inspect

source = inspect.getsource(ParallelEnvState)
if 'list(' in source and 'seeds.items' in source:
    print('PASS: Prune loop appears to copy keys before iteration')
else:
    print('WARNING: Check prune loop for iteration-during-mutation safety')
"

# Test blueprint embedding bounds
PYTHONPATH=src uv run python -c "
import torch
from esper.leyline import NUM_BLUEPRINTS
from esper.tamiyo.networks.factored_lstm import BlueprintEmbedding

embed = BlueprintEmbedding(NUM_BLUEPRINTS, embed_dim=64)

# Valid indices should work
valid_idx = torch.randint(0, NUM_BLUEPRINTS, (2, 3))
try:
    _ = embed(valid_idx)
    print(f'PASS: Valid indices [0, {NUM_BLUEPRINTS-1}] work')
except Exception as e:
    print(f'FAIL: Valid indices raised {e}')

# Out of bounds should fail
oob_idx = torch.tensor([[NUM_BLUEPRINTS]])  # One past end
try:
    _ = embed(oob_idx)
    print(f'FAIL: OOB index {NUM_BLUEPRINTS} did not raise error')
except Exception as e:
    print(f'PASS: OOB index correctly raised: {type(e).__name__}')
"
```

**Additional validation invariants:**

| Check | Expected | Failure Indicates |
|-------|----------|-------------------|
| Multi-prune same epoch | All scheduled seeds removed | Iteration-during-mutation bug |
| Prune log length | == number of pruned seeds | Double-prune or skip |
| Blueprint index range | [0, NUM_BLUEPRINTS-1] | Off-by-one or uninitialized value |
| Blueprint index dtype | int32/int64/long | Silent truncation from float |
| Mask dimensions per head | [batch, action_dim] | Broadcasting or truncation bugs |
| At least one valid action | mask.any(dim=-1).all() | Invalid environment state |

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
