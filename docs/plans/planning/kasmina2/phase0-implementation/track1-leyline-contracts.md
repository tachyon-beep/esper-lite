# Track 1: Leyline Contracts

**Priority:** Critical (blocks all other tracks)
**Estimated Effort:** 1-2 days

## Overview

Leyline is the "DNA" of Esper — all shared types, enums, and contracts live here. These tickets establish the type-level foundation for internal ops.

---

## L1: Add `SeedInternalKind` Enum

**File:** `src/esper/leyline/seed_internal.py` (new file)

### Specification

```python
from enum import IntEnum

class SeedInternalKind(IntEnum):
    """Kind of internal microstructure a seed exposes.

    This enum identifies the type of internal structure, which determines:
    - What `internal_level` means
    - How `level_to_structure(level)` maps to active components
    - Which telemetry fields are relevant
    """
    NONE = 0              # No internal structure (legacy seeds)
    CONV_LADDER = 1       # CNN block-count ladder (Phase 0)
    # Reserved for future phases:
    # LORA_RANK_LADDER = 2      # Phase 1
    # CHANNEL_GROUP_LADDER = 3  # Phase 2
    # ATTN_HEAD_GROUP_LADDER = 4  # Phase 3
```

### Acceptance Criteria
- [ ] Enum defined with `NONE` and `CONV_LADDER`
- [ ] Exported from `src/esper/leyline/__init__.py`
- [ ] Reserved values documented for future phases

---

## L2: Extend `SeedStateReport` with Internal Fields

**File:** `src/esper/leyline/reports.py`

### Specification

Add to `SeedStateReport` dataclass:

```python
@dataclass(frozen=True, slots=True)
class SeedStateReport:
    # ... existing fields ...

    # Internal microstructure fields (Phase 0)
    internal_kind: SeedInternalKind = SeedInternalKind.NONE
    internal_level: int = 0  # Range: [0..internal_max_level]
    internal_max_level: int = 1  # Invariant: >= 1 (avoids division by zero)
    internal_active_params: int = 0  # Trainable params at current level

    def __post_init__(self) -> None:
        """Validate internal field invariants (per Python specialist review)."""
        if self.internal_max_level < 1:
            raise ValueError(f"internal_max_level must be >= 1, got {self.internal_max_level}")
        if not (0 <= self.internal_level <= self.internal_max_level):
            raise ValueError(
                f"internal_level ({self.internal_level}) must be in range "
                f"[0, {self.internal_max_level}]"
            )
        if not isinstance(self.internal_kind, SeedInternalKind):
            raise TypeError(
                f"internal_kind must be SeedInternalKind, got {type(self.internal_kind)}"
            )
```

### Contract Invariants
- `0 <= internal_level <= internal_max_level`
- `internal_max_level >= 1` always (even when `internal_kind == NONE`)
- When `internal_kind == NONE`: `internal_level == 0`, `internal_max_level == 1`

### Acceptance Criteria
- [ ] Fields added to `SeedStateReport`
- [ ] Default values set correctly
- [ ] **Invariants enforced in `__post_init__`** (per Python specialist review)
- [ ] Import `SeedInternalKind` from `seed_internal.py`

---

## L3: Add `GROW_INTERNAL`, `SHRINK_INTERNAL` to `LifecycleOp`

**File:** `src/esper/leyline/factored_actions.py`

### Specification

```python
class LifecycleOp(IntEnum):
    NOOP = 0
    GERMINATE = 1
    PRUNE = 2
    FOSSILIZE = 3
    # New internal ops (Phase 0)
    GROW_INTERNAL = 4    # Increase internal_level by 1
    SHRINK_INTERNAL = 5  # Decrease internal_level by 1
```

### Semantics
- `GROW_INTERNAL`: `internal_level = min(internal_level + 1, internal_max_level)`
- `SHRINK_INTERNAL`: `internal_level = max(internal_level - 1, 0)`
- Both are no-ops if already at boundary (masked out by action masks)

### Acceptance Criteria
- [ ] Ops added to `LifecycleOp` enum
- [ ] `NUM_OPS` constant updated (or derived)
- [ ] Downstream imports work without change

---

## L4: Add `CONV_LADDER` to `BlueprintAction`

**File:** `src/esper/leyline/factored_actions.py`

### Specification

```python
class BlueprintAction(IntEnum):
    # ... existing values ...
    CONV_LADDER = 13  # Microstructured CNN ladder (Phase 0)

    def to_blueprint_id(self) -> str:
        mapping = {
            # ... existing mappings ...
            13: "conv_ladder",
        }
        return mapping[self.value]
```

### Acceptance Criteria
- [ ] `CONV_LADDER` added to enum
- [ ] `to_blueprint_id()` mapping updated
- [ ] Added to `CNN_BLUEPRINTS` set for action masking
- [ ] `BLUEPRINT_IDS`, `BLUEPRINT_ID_TO_INDEX` updated in `__init__.py`

---

## L5: Update Causal Masks for Internal Ops

**File:** `src/esper/leyline/causal_masks.py`

### Specification

For `GROW_INTERNAL` and `SHRINK_INTERNAL`:
- **Relevant heads:** `op`, `slot`
- **Irrelevant heads:** `blueprint`, `style`, `tempo`, `alpha_target`, `alpha_speed`, `alpha_curve`

```python
# In the causal relevance mapping:
LifecycleOp.GROW_INTERNAL: {"op", "slot"},
LifecycleOp.SHRINK_INTERNAL: {"op", "slot"},
```

### Acceptance Criteria
- [ ] Causal masks updated for both ops
- [ ] Only `op` and `slot` heads are relevant
- [ ] Tests verify mask correctness

---

## L6: Add `SEED_INTERNAL_LEVEL_CHANGED` Telemetry Event

**File:** `src/esper/leyline/telemetry.py`

### Specification

```python
class TelemetryEventType(IntEnum):
    # ... existing events ...
    SEED_INTERNAL_LEVEL_CHANGED = 15  # New

@dataclass(frozen=True, slots=True)
class SeedInternalLevelChangedPayload:
    """Payload for SEED_INTERNAL_LEVEL_CHANGED event.

    Note: env_id is injected by the telemetry emission layer (emit_with_env_context),
    not set by Kasmina directly. This avoids -1 sentinel patterns per project guidelines.
    """
    slot_id: str
    env_id: int  # Injected by telemetry emission layer
    blueprint_id: str
    internal_kind: int  # SeedInternalKind.value
    from_level: int
    to_level: int
    max_level: int
    active_params: int  # Params after level change
```

### env_id Injection Pattern (per Python specialist review)

The `env_id` field is populated by the telemetry emission layer, NOT by Kasmina:

```python
# In Kasmina (slot.py) - create payload WITHOUT env_id
payload = SeedInternalLevelChangedPayload(
    slot_id=self.slot_id,
    env_id=0,  # Placeholder - will be overwritten
    # ... other fields ...
)

# The telemetry layer's emit_with_env_context() sets the actual env_id
# This avoids -1 sentinel patterns that mask bugs
```

### Acceptance Criteria
- [ ] Event type added to enum
- [ ] Payload dataclass defined with all fields
- [ ] **env_id injected by telemetry layer, not Kasmina** (per Python specialist review)
- [ ] Payload registered in event dispatch mapping
- [ ] Exported from `__init__.py`

---

## L7: Make Obs Dims Derived from Field Lists

**File:** `src/esper/leyline/__init__.py`

### Specification

Replace hardcoded constants with derived values:

```python
# Base feature fields (order matters for unpacking)
BASE_FEATURE_FIELDS: list[str] = [
    "loss_delta",
    "accuracy",
    # ... enumerate all base fields ...
    # last_action_op one-hot is derived from NUM_OPS
]

# Per-slot feature fields
SLOT_FEATURE_FIELDS: list[str] = [
    "stage_norm",
    "alpha",
    "epochs_in_stage_norm",
    # ... existing fields ...
    "internal_level_norm",  # NEW for Phase 0
]

# Derived dimensions (single source of truth)
NUM_OPS = len(LifecycleOp)
OBS_V3_BASE_FEATURE_SIZE = len(BASE_FEATURE_FIELDS) + NUM_OPS  # +NUM_OPS for one-hot
OBS_V3_SLOT_FEATURE_SIZE = len(SLOT_FEATURE_FIELDS)
# ... other derived dims ...
```

### Contract
- **No manual updates to dimension constants**
- Add fields to lists; dims derive automatically
- Shape assertions fail fast on mismatch

### Acceptance Criteria
- [ ] Field lists defined as source of truth
- [ ] Dimension constants derived from lists
- [ ] `internal_level_norm` added to slot fields
- [ ] Downstream assertions use derived dims
- [ ] **REQUIRED: Dimension validation test** (per Python specialist review)

### Required Validation Test (per Python specialist review)

This test is CRITICAL to prevent field list drift from feature extraction:

```python
# In tests/leyline/test_observation_dims.py
def test_slot_feature_fields_match_extraction():
    """Ensure SLOT_FEATURE_FIELDS matches actual feature extraction."""
    from esper.tamiyo.policy.features import extract_slot_features
    from esper.leyline import SLOT_FEATURE_FIELDS, OBS_V3_SLOT_FEATURE_SIZE

    # Create dummy report with known values
    report = create_test_seed_state_report()
    features = extract_slot_features(report)

    expected_dim = len(SLOT_FEATURE_FIELDS)
    assert features.shape[-1] == expected_dim, (
        f"Feature extraction produces {features.shape[-1]} dims, "
        f"but SLOT_FEATURE_FIELDS implies {expected_dim}"
    )
    assert expected_dim == OBS_V3_SLOT_FEATURE_SIZE
```

---

## Testing Requirements

### Unit Tests
- `tests/leyline/test_seed_internal.py` (new): Test `SeedInternalKind` enum values
- `tests/leyline/test_factored_actions.py`: Update for new ops and blueprint
- `tests/leyline/test_causal_masks.py`: Verify mask correctness for internal ops

### Property Tests
- Verify `internal_level` invariants hold across all valid states
- Verify causal masks are complete (all ops covered)

### Integration
- Verify imports work from all consuming modules
- Verify telemetry event can be serialized/deserialized
