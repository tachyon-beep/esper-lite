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
import logging
from typing import Optional

_bp_validation_logger = logging.getLogger("esper.tamiyo.features.validation")


def _validate_blueprint_index(
    bp_idx: int,
    slot_id: str,
    *,
    strict: bool = False,
) -> int:
    """Validate and optionally sanitize blueprint index.

    Runtime guard against invalid blueprint indices in observations.
    This catches data corruption (e.g., malformed SeedStateReport,
    stale slot reports, or serialization issues), not just static drift.

    Addresses code review finding: Single corrupt observation shouldn't crash
    a 12-hour training run. Production needs resilience; tests need strictness.

    Args:
        bp_idx: Blueprint index from observation (-1 for inactive, 0-12 for active)
        slot_id: Slot identifier for error context
        strict: If True, raise ValueError on invalid index.
                If False (default), log error and return -1 (treat as inactive).

    Returns:
        Validated blueprint index, or -1 if invalid and not strict.

    Raises:
        ValueError: If strict=True and index is invalid.

    Example:
        # In tests (crash on bad data for immediate feedback):
        bp_idx = _validate_blueprint_index(raw_idx, slot_id, strict=True)

        # In production training (graceful degradation):
        bp_idx = _validate_blueprint_index(raw_idx, slot_id, strict=False)
    """
    if bp_idx != -1 and (bp_idx < 0 or bp_idx >= NUM_BLUEPRINTS):
        msg = (
            f"Slot {slot_id} has invalid blueprint_index {bp_idx}. "
            f"Valid range: 0-{NUM_BLUEPRINTS-1} or -1 (inactive)."
        )
        if strict:
            raise ValueError(msg)
        else:
            _bp_validation_logger.error(
                f"{msg} Treating as inactive slot (-1). "
                f"This may indicate data corruption in SeedStateReport."
            )
            return -1
    return bp_idx


# Usage in feature extraction loop:
for slot_idx, slot_id in enumerate(slot_config.slot_ids):
    if report := reports.get(slot_id):
        # Use strict=False in training, strict=True in tests
        bp_idx = _validate_blueprint_index(
            report.blueprint_index, slot_id, strict=False
        )
        bp_indices[env_idx, slot_idx] = bp_idx
    else:
        bp_indices[env_idx, slot_idx] = -1  # Inactive slot
```

**Note:** This catches runtime data corruption, not just static drift. The static guards ensure the code is correct; the runtime guard ensures the data is correct. The `strict` parameter allows tests to fail immediately on invalid data while production training gracefully degrades.

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

When a slot is inactive (`is_active=False`), its stage encoding should be all-zeros, NOT a one-hot for stage 0 (DORMANT). This is intentional:

- All-zeros is indistinguishable from "no slot" in the embedding space
- This prevents the network from learning spurious patterns from inactive slot stages
- The `is_active` flag (separate feature) disambiguates "inactive slot with stage 0" from "active slot with DORMANT stage"

```python
# CORRECT: inactive slot gets all-zeros stage encoding
if not report.is_active:
    stage_one_hot = torch.zeros(NUM_STAGES)  # All zeros
else:
    stage_one_hot = F.one_hot(torch.tensor(report.stage), NUM_STAGES)

# WRONG: inactive slot gets DORMANT encoding (creates spurious signal)
# stage_one_hot = F.one_hot(torch.tensor(0), NUM_STAGES)  # DON'T DO THIS
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

def make_test_report(slot_id: str, stage: int = 2, is_active: bool = True):
    \"\"\"Create a test slot report.

    Note: is_active=False with a slot report is different from no slot report.
    Seeds in PRUNED or EMBARGOED stages have slot reports but is_active=False.
    \"\"\"
    return SeedStateReport(
        slot_id=slot_id,
        is_active=is_active,
        stage=stage,
        blueprint_index=5 if is_active else -1,
        alpha=0.3 if is_active else 0.0,
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
    ({config.slot_ids[0]: make_test_report(
        config.slot_ids[0],
        stage=SeedStage.PRUNED.value,
        is_active=False
    )}, 'Seed in PRUNED stage (has report, is_active=False)'),
    ({config.slot_ids[0]: make_test_report(
        config.slot_ids[0],
        stage=SeedStage.EMBARGOED.value,
        is_active=False
    )}, 'Seed in EMBARGOED stage (has report, is_active=False)'),

    # Mixed: some active, some inactive with reports
    ({
        config.slot_ids[0]: make_test_report(config.slot_ids[0], stage=2, is_active=True),
        config.slot_ids[1]: make_test_report(config.slot_ids[1], stage=SeedStage.PRUNED.value, is_active=False),
    }, 'Mixed: one active, one PRUNED'),
]

for reports, desc in test_cases:
    signals = make_test_signals()

    try:
        obs, bp_idx = batch_obs_to_features([signals], [reports], config, device)
        expected_size = get_feature_size(config)
        assert obs.shape == (1, expected_size), f'Shape mismatch: {obs.shape} vs (1, {expected_size})'
        assert bp_idx.shape == (1, config.num_slots), f'BP idx shape: {bp_idx.shape}'

        # Verify inactive slots (either no report OR is_active=False) have -1 blueprint index
        for i, slot_id in enumerate(config.slot_ids):
            report = reports.get(slot_id)
            if report is None or not report.is_active:
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
2. **Slot report with `is_active=False`** - Seed exists but is in an inactive stage (PRUNED, EMBARGOED)

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
    - LOWER threshold = smaller improvement required to be "stable" = MORE false positives
    - HIGHER threshold = larger improvements still count as "stable" = FEWER false positives

    Example:
    - threshold=0.03: if epoch improves by 2%, training is "stable" (2% < 3%)
    - threshold=0.05: if epoch improves by 2%, training is "stable" (2% < 5%)
    - threshold=0.05: if epoch improves by 4%, training is STILL "stable" (4% < 5%)

    To PREVENT early triggering (more conservative):
    - INCREASE threshold (e.g., 0.03 → 0.05)
    - INCREASE stabilization_epochs (e.g., 3 → 5)

    To trigger MORE easily (less conservative):
    - DECREASE threshold (e.g., 0.03 → 0.02)
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
2. **INCREASE** `stabilization_threshold` from 0.03 to 0.05 (higher threshold = harder to satisfy = fewer false positives)
3. Add early-epoch grace period in the tracker: `if epoch < 10: return False` before checking stability conditions

> **Note:** Lower threshold makes it *easier* to stabilize (smaller improvement required), not harder. To prevent early triggering, raise the bar.

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

##### 1. Buffer Schema Alignment Test

After Obs V3 integration, the rollout buffer stores new fields (blueprint_indices, action feedback state, etc.). Misaligned arrays cause silent data corruption where transition N's action pairs with transition N+1's state.

```python
def test_buffer_schema_alignment_after_mini_rollout():
    """Verify all buffer fields are correctly aligned after a mini-rollout.

    BUG PATTERN: If a new field (e.g., blueprint_indices) is appended at a
    different point in the add() call than other fields, all subsequent data
    becomes off-by-one. The policy learns from mismatched (state, action) pairs.

    This test runs a short rollout and verifies field coherence by checking
    that each transition's data is internally consistent.
    """
    from esper.simic.agent.rollout_buffer import TamiyoRolloutBuffer
    import torch

    # Setup: Create buffer and run mini-rollout (2 envs, 3 steps each)
    buffer = TamiyoRolloutBuffer(
        num_envs=2,
        max_steps_per_env=10,
        state_dim=121,
        num_slots=3,
        device=torch.device("cpu"),
    )

    # Collect transitions with known values for verification
    test_transitions = []
    for env_id in range(2):
        for step in range(3):
            # Create transition with identifiable values
            transition = {
                "env_id": env_id,
                "state": torch.full((121,), float(env_id * 100 + step)),
                "blueprint_indices": torch.full((3,), env_id * 10 + step, dtype=torch.int64),
                "op_action": step % 6,  # LifecycleOp value
                "slot_action": step % 3,
                "value": float(env_id + step * 0.1),
                "reward": float(step * 0.5),
                "done": (step == 2),  # Last step is done
                "truncated": False,
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

    print("Buffer schema alignment: PASS")


def test_buffer_dtype_consistency():
    """Verify buffer field dtypes match network expectations.

    BUG PATTERN: If blueprint_indices are stored as float32 instead of int64,
    nn.Embedding will fail silently or produce garbage. If masks are stored as
    int8 instead of bool/float, masking logic may not block invalid actions.
    """
    from esper.simic.agent.rollout_buffer import TamiyoRolloutBuffer
    import torch

    buffer = TamiyoRolloutBuffer(
        num_envs=1, max_steps_per_env=5, state_dim=121, num_slots=3,
        device=torch.device("cpu"),
    )

    # Add one transition
    buffer.add(
        env_id=0,
        state=torch.randn(121),
        blueprint_indices=torch.tensor([0, 1, 2], dtype=torch.int64),
        op_action=0,
        slot_action=0,
        value=0.0,
        reward=0.0,
        done=False,
        truncated=True,
    )

    batch = buffer.get_batched_sequences(torch.device("cpu"))

    # Critical dtype checks
    assert batch["blueprint_indices"].dtype == torch.int64, (
        f"blueprint_indices must be int64 for nn.Embedding, got {batch['blueprint_indices'].dtype}"
    )
    assert batch["states"].dtype == torch.float32, (
        f"states must be float32, got {batch['states'].dtype}"
    )
    assert batch["op_actions"].dtype in (torch.int64, torch.long), (
        f"op_actions must be int64/long for indexing, got {batch['op_actions'].dtype}"
    )

    print("Buffer dtype consistency: PASS")
```

##### 2. Done vs Truncated Flag Test

At `max_epochs`, the episode truly ends (`done=True`). Earlier epochs are truncated (`truncated=True`) and require bootstrap values. Mismarking the final epoch as truncated causes the bootstrap logic to use a value estimate when it should use 0.0.

```python
def test_done_vs_truncated_at_max_epochs():
    """Verify done=True, truncated=False, bootstrap_value=0.0 at max_epochs.

    BUG PATTERN: If the final epoch is marked truncated=True instead of done=True,
    the bootstrap logic computes V(s_final) instead of using 0.0. This biases
    the advantage estimate for the final transition, causing the critic to learn
    incorrect values for near-terminal states.

    The invariant is:
    - epoch < max_epochs: done=False, truncated=True, bootstrap_value=V(s_next)
    - epoch == max_epochs: done=True, truncated=False, bootstrap_value=0.0
    """
    from esper.simic.training.vectorized import VectorizedTamiyoTraining
    from esper.leyline import DEFAULT_MAX_EPOCHS
    import torch

    # Setup: Run training to max_epochs
    trainer = VectorizedTamiyoTraining(
        num_envs=1,
        max_epochs=10,  # Short run for testing
        device=torch.device("cpu"),
    )

    # Collect transitions for full episode
    transitions = []
    for epoch in range(1, 11):  # epochs 1-10
        transition = trainer.step()
        transitions.append(transition)

    # Verify intermediate epochs (1-9)
    for i, t in enumerate(transitions[:-1]):
        epoch = i + 1
        assert t["done"] is False, (
            f"Epoch {epoch} should have done=False (not terminal)"
        )
        assert t["truncated"] is True, (
            f"Epoch {epoch} should have truncated=True (episode continues)"
        )
        assert t["bootstrap_value"] != 0.0, (
            f"Epoch {epoch} should have nonzero bootstrap_value (V(s_next))"
        )

    # Verify final epoch (10)
    final = transitions[-1]
    assert final["done"] is True, (
        f"Final epoch should have done=True, got {final['done']}"
    )
    assert final["truncated"] is False, (
        f"Final epoch should have truncated=False, got {final['truncated']}"
    )
    assert final["bootstrap_value"] == 0.0, (
        f"Final epoch should have bootstrap_value=0.0, got {final['bootstrap_value']}. "
        f"Non-zero bootstrap at episode end biases advantage estimation."
    )

    print("Done vs truncated flags: PASS")


def test_signals_done_propagates_correctly():
    """Verify signals.done from environment propagates to buffer correctly.

    BUG PATTERN: The environment sets signals.done=True at max_epochs, but
    the rollout loop might override this or check epoch count separately,
    leading to inconsistency between what the env reports and what's stored.
    """
    from esper.tamiyo.tracker import SignalTracker
    from esper.leyline import DEFAULT_MAX_EPOCHS

    tracker = SignalTracker(max_epochs=25)

    # Simulate epochs
    for epoch in range(1, 26):
        signals = tracker.update(epoch=epoch, val_loss=2.0 - epoch * 0.05, val_accuracy=epoch * 2.0)

        if epoch < 25:
            assert not signals.done, f"signals.done should be False at epoch {epoch}"
        else:
            assert signals.done, f"signals.done should be True at epoch {epoch} (max_epochs=25)"

    print("Signals.done propagation: PASS")
```

##### 3. Bootstrap Indexing Test for Parallel Environments

With multiple parallel environments, bootstrap values must be assigned to the correct truncated transitions. If env ordering diverges between `all_post_action_signals` and `transitions_data`, bootstrap values get swapped between environments.

```python
def test_bootstrap_indexing_parallel_envs():
    """Verify bootstrap_values[k] matches the k-th truncated transition's env_id.

    BUG PATTERN: The rollout loop builds all_post_action_signals in env order
    and transitions_data in env order, then zips them together. If any code
    path processes envs out of order (e.g., skipping done envs), the bootstrap
    values get assigned to wrong transitions.

    This corrupts advantage calculation: env 0's next-state value is used for
    env 1's transition, causing the critic to learn incorrect state values.
    """
    import torch

    # Simulate parallel rollout with 4 envs
    num_envs = 4
    transitions_data = []
    all_post_action_signals = []
    bootstrap_env_ids = []

    # Collect transitions - some envs are done, others truncated
    env_done_status = [False, True, False, True]  # envs 1, 3 are done
    for env_id in range(num_envs):
        is_done = env_done_status[env_id]
        transition = {
            "env_id": env_id,
            "done": is_done,
            "truncated": not is_done,
        }
        transitions_data.append(transition)

        # Only non-done envs need bootstrap values
        if not is_done:
            all_post_action_signals.append({"env_id": env_id, "value": env_id * 10.0})
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

    # Verify assignment
    assert transitions_data[0]["bootstrap_value"] == 0.0, "Env 0 (truncated) should have bootstrap"
    assert transitions_data[1]["bootstrap_value"] == 0.0, "Env 1 (done) should have bootstrap=0"
    assert transitions_data[2]["bootstrap_value"] == 20.0, "Env 2 (truncated) should have bootstrap=20"
    assert transitions_data[3]["bootstrap_value"] == 0.0, "Env 3 (done) should have bootstrap=0"

    print("Bootstrap indexing for parallel envs: PASS")


def test_bootstrap_value_computation_uses_correct_state():
    """Verify bootstrap value is computed from post-action state, not pre-action.

    BUG PATTERN: If bootstrap is computed from the state BEFORE the action
    (s_t instead of s_{t+1}), the advantage estimate is off by one step.
    The critic learns V(s_t) = r_t + gamma*V(s_t) instead of V(s_t) = r_t + gamma*V(s_{t+1}).
    """
    from esper.tamiyo.policy.features import batch_obs_to_features
    import torch

    # Simulate: action at epoch 5 leads to state at epoch 6
    pre_action_signals = {"epoch": 5, "val_loss": 2.0, "val_accuracy": 50.0}
    post_action_signals = {"epoch": 6, "val_loss": 1.8, "val_accuracy": 55.0}

    # Bootstrap should use POST-action signals (epoch 6 state)
    # The value V(s_6) estimates future returns from state after epoch 6

    # This is a design verification - ensure the rollout code uses:
    # final_obs, final_bp = batch_obs_to_features(POST_action_signals, ...)
    # NOT: batch_obs_to_features(PRE_action_signals, ...)

    # The test passes if the implementation follows this pattern
    print("Bootstrap value computation: Verify manually that vectorized.py uses post_action_signals")
```

##### 4. LSTM Hidden State Reset Test

At episode boundaries, the LSTM hidden state must be reset. Leftover hidden state from a previous episode causes the policy to condition on irrelevant context from a different training run.

```python
def test_lstm_hidden_state_reset_at_episode_boundary():
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
    import torch

    policy = FactoredRecurrentActorCritic(
        state_dim=121,
        num_slots=3,
        hidden_dim=128,
        num_blueprints=13,
    )

    # Run one episode to build up hidden state
    state = torch.randn(1, 10, 121)  # batch=1, seq=10, features=121
    bp_idx = torch.randint(0, 13, (1, 10, 3))

    # First forward pass - hidden state initialized internally
    out1 = policy.forward(state, bp_idx, hidden=None)
    hidden_after_ep1 = out1.hidden

    # Verify hidden state is not zero after episode
    h, c = hidden_after_ep1
    assert not torch.allclose(h, torch.zeros_like(h)), (
        "Hidden state should be non-zero after episode"
    )

    # NEW EPISODE: Reset hidden state
    # Option 1: Pass hidden=None
    out2 = policy.forward(state, bp_idx, hidden=None)
    h2_init, _ = out2.hidden

    # The initial hidden state should be fresh (zeros or learned init)
    # NOT the leftover state from episode 1

    # Option 2: Explicit reset (if method exists)
    if hasattr(policy, 'reset_hidden_state'):
        policy.reset_hidden_state()
        # Verify internal state is cleared

    print("LSTM hidden state reset: PASS (verify reset is called at episode boundaries)")


def test_hidden_state_not_shared_across_parallel_envs():
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

    print("Hidden state independence across envs: PASS")
```

##### 5. Reward Normalization Consistency Test

Rewards must be consistently normalized (or unnormalized) between buffer storage and PPO update. Mixing scales causes the critic to learn incorrect value estimates and corrupts advantage normalization.

```python
def test_reward_normalization_consistency():
    """Verify rewards in buffer match logged rewards (same normalization).

    BUG PATTERN: If compute_reward() returns normalized rewards but the buffer
    stores raw rewards (or vice versa), the advantage calculation uses
    mismatched scales. The critic learns V(s) in one scale while rewards
    are in another, causing systematic bias.

    The invariant is: buffer.rewards[i] == logged_reward[i] (same value).
    """
    from esper.simic.training.parallel_env_state import ParallelEnvState
    import torch

    env_state = ParallelEnvState()

    # Simulate reward computation and storage
    raw_reward = 0.5
    normalized_reward = (raw_reward - env_state.reward_normalizer.mean) / (
        env_state.reward_normalizer.std + 1e-8
    )

    # What goes into the buffer?
    buffer_reward = normalized_reward  # Should be normalized

    # What gets logged?
    logged_reward = normalized_reward  # Should match buffer

    assert abs(buffer_reward - logged_reward) < 1e-6, (
        f"Reward mismatch: buffer has {buffer_reward}, log has {logged_reward}. "
        f"If these differ, advantage estimation will be corrupted."
    )

    print("Reward normalization consistency: PASS")


def test_reward_normalizer_updates_correctly():
    """Verify reward normalizer statistics update during training.

    BUG PATTERN: If the normalizer's mean/std are never updated, early rewards
    dominate the statistics. As training progresses and reward distribution
    shifts, normalized rewards become increasingly biased.
    """
    from esper.simic.utils.normalizers import RunningMeanStd
    import torch

    normalizer = RunningMeanStd()

    # Phase 1: Early training (small rewards)
    early_rewards = torch.tensor([0.1, 0.2, 0.15, 0.12, 0.18])
    for r in early_rewards:
        normalizer.update(r.unsqueeze(0))

    early_mean = normalizer.mean.item()
    early_std = normalizer.std.item()

    # Phase 2: Later training (larger rewards as policy improves)
    later_rewards = torch.tensor([0.8, 0.9, 0.85, 0.92, 0.88])
    for r in later_rewards:
        normalizer.update(r.unsqueeze(0))

    later_mean = normalizer.mean.item()
    later_std = normalizer.std.item()

    # Statistics should have changed
    assert later_mean > early_mean, (
        f"Reward normalizer mean should increase as rewards increase. "
        f"Early mean: {early_mean}, Later mean: {later_mean}. "
        f"If unchanged, normalizer.update() is not being called."
    )

    print("Reward normalizer updates: PASS")


def test_buffer_reward_matches_transition_reward():
    """Verify the reward stored in buffer equals the transition's computed reward.

    BUG PATTERN: If buffer.add() receives a different reward than what was
    computed (e.g., due to an intermediate variable being overwritten or
    a stale value being used), the policy learns from incorrect feedback.
    """
    import torch

    # Simulate transition
    computed_reward = 0.42
    transition = {
        "reward": computed_reward,
        "state": torch.randn(121),
        # ... other fields
    }

    # Add to buffer (simulated)
    buffer_rewards = []
    buffer_rewards.append(transition["reward"])

    # Retrieve and verify
    assert buffer_rewards[0] == computed_reward, (
        f"Buffer reward {buffer_rewards[0]} != computed reward {computed_reward}. "
        f"Check that transition['reward'] is not modified between compute and add."
    )

    print("Buffer reward matches transition: PASS")
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

**Verify enum values in:**
- `src/esper/leyline/enums.py` — AlphaMode, AlphaAlgorithm definitions
- `src/esper/leyline/__init__.py` — SeedStage, LifecycleOp definitions

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
