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
# With DEFAULT_GAMMA=0.995, signal stays >0.5 for ~138 epochs
# Math: 0.995^t = 0.5 → t = ln(0.5)/ln(0.995) ≈ 138
counterfactual_fresh = DEFAULT_GAMMA ** epochs_since_counterfactual

# Gradient trend signal
gradient_health_prev = previous_epoch_gradient_health  # Track in ParallelEnvState
```

**Why gamma-matched decay (from DRL review):** The old 0.8^epochs decayed too fast—0.8^10 = 0.1, making counterfactual estimates unreliable after just 10 epochs. Using DEFAULT_GAMMA (0.995) aligns with PPO's credit horizon: 0.995^10 ≈ 0.95, staying useful for ~138 epochs before dropping below 0.5 (ln(0.5)/ln(0.995) ≈ 138).

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
> The current `SeedStateReport` has `blueprint_id: str` (e.g., `"conv_heavy"`), NOT `blueprint_index: int`.
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
> Then update the `SeedState.to_report()` method in `src/esper/kasmina/slot.py` to populate both fields consistently.
>
> **Option B: Derive index from ID in feature extraction**
> ```python
> from esper.leyline import BlueprintAction
>
> _BLUEPRINT_TO_INDEX: dict[str, int] = {
>     bp.to_blueprint_id(): idx for idx, bp in enumerate(BlueprintAction)
> }
>
> # In extraction (FAIL-FAST version - don't silently convert unknown to -1):
> bp_idx = _BLUEPRINT_TO_INDEX[report.blueprint_id]  # Raises KeyError if unknown
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
        # FAIL-FAST: Don't use .get(..., -1) which silently converts unknown IDs to inactive
        if report.blueprint_id not in _BLUEPRINT_TO_INDEX:
            raise ValueError(f"Unknown blueprint_id '{report.blueprint_id}' for slot {slot_id}")
        bp_idx = _BLUEPRINT_TO_INDEX[report.blueprint_id]
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
    # report.stage is always SeedStage enum - access .value directly (no hasattr check needed)
    stage_idx = STAGE_TO_INDEX[report.stage.value]
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
    """Validate action mask has correct dtype before device transfer.

    IMPORTANT: MaskedCategorical expects BOOLEAN masks (True=valid, False=invalid).
    See action_masks.py line 415: "mask: Boolean mask, True = valid, False = invalid"
    """
    # MaskedCategorical requires boolean masks - do NOT accept float
    assert mask.dtype == torch.bool, (
        f"Action mask '{name}' has invalid dtype {mask.dtype}. "
        f"MaskedCategorical requires torch.bool (True=valid, False=invalid). "
        f"Use .bool() to convert if needed."
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

# Validate mask is boolean (MaskedCategorical requires ~mask which needs bool)
def validate_mask(mask, name):
    assert mask.dtype == torch.bool, (
        f'{name}: invalid dtype {mask.dtype}. '
        f'MaskedCategorical requires torch.bool (uses ~mask internally).'
    )
    print(f'✓ {name}: dtype={mask.dtype}, shape={mask.shape}')

# Test cases - only boolean masks are valid
validate_mask(torch.tensor([True, False, True]), 'bool_mask')
print('✓ Boolean mask validation passed')

# These SHOULD fail - float masks are NOT valid for MaskedCategorical:
# validate_mask(torch.tensor([1.0, 0.0, 1.0]), 'float_mask')  # Would fail
# validate_mask(torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64), 'float64_mask')  # Would fail
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
    # NOTE: TrainingSignals uses nested TrainingMetrics, NOT flat fields
    from esper.leyline.signals import TrainingMetrics
    return TrainingSignals(
        metrics=TrainingMetrics(
            epoch=10,
            val_loss=0.5,
            val_accuracy=75.0,
        ),
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
    # TrainingSignals always has metrics field (default_factory) - access directly
    actual = signals.metrics.global_step

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

