# Track 3: Tamiyo Policy

**Priority:** High (blocks training integration)
**Estimated Effort:** 1-2 days
**Dependencies:** Track 1 (L2, L3, L7), Track 2 (K2)

## Overview

Tamiyo needs to observe internal microstructure state and have appropriate action masks for the new internal ops. This track wires the observation and action spaces.

---

## T1: Add `internal_level_norm` to Feature Extraction

**File:** `src/esper/tamiyo/policy/features.py`

### Specification

Add the normalized internal level to per-slot features:

```python
def extract_slot_features(report: SeedStateReport) -> torch.Tensor:
    """Extract per-slot features from a seed state report.

    Returns tensor of shape [SLOT_FEATURE_SIZE].
    """
    features = [
        # ... existing features ...

        # Internal microstructure (Phase 0)
        # Normalized to [0, 1] range
        report.internal_level / max(report.internal_max_level, 1),  # internal_level_norm
    ]

    return torch.tensor(features, dtype=torch.float32)
```

### Field Position

The `internal_level_norm` field must be added to `SLOT_FEATURE_FIELDS` in Leyline (see L7):

```python
SLOT_FEATURE_FIELDS: list[str] = [
    # ... existing fields ...
    "internal_level_norm",  # NEW: Position determined by list order
]
```

### Normalization Contract

- `internal_level_norm = internal_level / internal_max_level`
- Range: `[0.0, 1.0]`
- When `internal_kind == NONE`: value is `0.0` (since `internal_level == 0`, `internal_max_level == 1`)

### Acceptance Criteria
- [ ] `internal_level_norm` extracted from `SeedStateReport`
- [ ] Correctly normalized to `[0, 1]` range
- [ ] Division-by-zero safe (max_level >= 1 invariant)
- [ ] Feature tensor shape matches `OBS_V3_SLOT_FEATURE_SIZE`
- [ ] **CRITICAL**: `OBS_V3_SLOT_FEATURE_SIZE` updated from 31 to 32 in Leyline (coordinate with L7)
- [ ] Both `extract_slot_features()` and `batch_obs_to_features()` updated

---

## T2: Add Action Masks for Internal Ops (Stage-Gated)

**File:** `src/esper/tamiyo/policy/action_masks.py`

### Specification

Add mask computation for `GROW_INTERNAL` and `SHRINK_INTERNAL`:

```python
# Module-level constants for hot-path optimization (per PyTorch specialist review)
_INTERNAL_OPS_STAGES: frozenset[int] = frozenset({
    SeedStage.TRAINING.value,
    SeedStage.BLENDING.value,
    SeedStage.HOLDING.value,
})


def compute_lifecycle_op_mask(
    report: SeedStateReport,
    slot_config: SlotConfig,
) -> torch.Tensor:
    """Compute action mask for lifecycle ops.

    Returns tensor of shape [NUM_OPS] with 1.0 for valid ops, 0.0 for invalid.
    """
    mask = torch.zeros(NUM_OPS, dtype=torch.float32)

    # ... existing op masks ...

    # Internal ops mask (Phase 0)
    # Valid during TRAINING, BLENDING, HOLDING stages
    # Uses .value for faster int comparison in hot path
    internal_ops_valid = report.stage.value in _INTERNAL_OPS_STAGES

    # GROW_INTERNAL: valid if has structure AND not at max level
    if internal_ops_valid and report.internal_kind != SeedInternalKind.NONE:
        if report.internal_level < report.internal_max_level:
            mask[LifecycleOp.GROW_INTERNAL] = 1.0

        # SHRINK_INTERNAL: valid if has structure AND above level 0
        if report.internal_level > 0:
            mask[LifecycleOp.SHRINK_INTERNAL] = 1.0

    return mask
```

### Stage Gating Rules

| Stage | GROW_INTERNAL | SHRINK_INTERNAL |
|-------|---------------|-----------------|
| DORMANT | ❌ | ❌ |
| GERMINATING | ❌ | ❌ |
| TRAINING | ✅ | ✅ |
| BLENDING | ✅ | ✅ |
| HOLDING | ✅ | ✅ |
| FOSSILIZED | ❌ | ❌ |
| PRUNED | ❌ | ❌ |

### Boundary Conditions

- `GROW_INTERNAL` masked out when `internal_level == internal_max_level`
- `SHRINK_INTERNAL` masked out when `internal_level == 0`
- Both masked out when `internal_kind == NONE` (no microstructure)

### Acceptance Criteria
- [ ] `GROW_INTERNAL` correctly masked by stage and level
- [ ] `SHRINK_INTERNAL` correctly masked by stage and level
- [ ] Both masked for non-microstructured seeds
- [ ] Mask tensor shape matches `NUM_OPS`
- [ ] Unit tests verify all stage combinations

---

## T3: Update Feature Net Input Dim Assertions

**File:** `src/esper/tamiyo/networks/factored_lstm.py`

### Specification

Update input dimension assertions to use derived constants:

```python
from esper.leyline import (
    OBS_V3_BASE_FEATURE_SIZE,
    OBS_V3_SLOT_FEATURE_SIZE,
    NUM_SLOTS,
)

class FactoredLSTMPolicy(nn.Module):
    def __init__(
        self,
        base_feature_size: int = OBS_V3_BASE_FEATURE_SIZE,
        slot_feature_size: int = OBS_V3_SLOT_FEATURE_SIZE,
        num_slots: int = NUM_SLOTS,
        # ... other params ...
    ):
        super().__init__()

        # Assertions use derived dims
        assert base_feature_size == OBS_V3_BASE_FEATURE_SIZE, (
            f"base_feature_size mismatch: {base_feature_size} vs {OBS_V3_BASE_FEATURE_SIZE}"
        )
        assert slot_feature_size == OBS_V3_SLOT_FEATURE_SIZE, (
            f"slot_feature_size mismatch: {slot_feature_size} vs {OBS_V3_SLOT_FEATURE_SIZE}"
        )

        # ... rest of init ...
```

### Shape Validation Pattern (per PyTorch specialist review)

Use isolated validation function with `@torch.compiler.disable` to avoid graph breaks:

```python
@torch.compiler.disable
def _validate_obs_shapes(
    base_features: torch.Tensor,
    slot_features: torch.Tensor,
) -> None:
    """Validate observation shapes (disabled during torch.compile).

    Called only when MaskedCategorical.validate is True (development mode).
    """
    if base_features.shape[-1] != OBS_V3_BASE_FEATURE_SIZE:
        raise ValueError(
            f"base_features shape[-1] = {base_features.shape[-1]}, "
            f"expected {OBS_V3_BASE_FEATURE_SIZE}"
        )
    if slot_features.shape[-1] != OBS_V3_SLOT_FEATURE_SIZE:
        raise ValueError(
            f"slot_features shape[-1] = {slot_features.shape[-1]}, "
            f"expected {OBS_V3_SLOT_FEATURE_SIZE}"
        )


def forward(self, obs: dict[str, torch.Tensor]) -> PolicyOutput:
    base_features = obs["base_features"]
    slot_features = obs["slot_features"]

    # Shape validation (toggleable, compile-safe)
    if MaskedCategorical.validate:
        _validate_obs_shapes(base_features, slot_features)

    # ... rest of forward ...
```

**Note:** Plain `assert` in forward causes graph breaks under torch.compile. Using a separate
`@torch.compiler.disable` function allows toggling validation without compile overhead.

### Acceptance Criteria
- [ ] Input dims use derived constants from Leyline
- [ ] Assertions fail-fast on shape mismatch
- [ ] No hardcoded dimension values
- [ ] Tests verify shape validation catches mismatches

---

## Testing Requirements

### Unit Tests (`tests/tamiyo/policy/`)

**test_features.py:**
```python
def test_internal_level_norm_in_features():
    """Verify internal_level_norm is extracted correctly."""
    report = SeedStateReport(
        # ... standard fields ...
        internal_kind=SeedInternalKind.CONV_LADDER,
        internal_level=2,
        internal_max_level=4,
    )
    features = extract_slot_features(report)

    # Find internal_level_norm position
    idx = SLOT_FEATURE_FIELDS.index("internal_level_norm")
    assert features[idx] == pytest.approx(0.5)  # 2/4 = 0.5

def test_internal_level_norm_no_structure():
    """Verify internal_level_norm is 0.0 for non-microstructured seeds."""
    report = SeedStateReport(
        internal_kind=SeedInternalKind.NONE,
        internal_level=0,
        internal_max_level=1,
    )
    features = extract_slot_features(report)
    idx = SLOT_FEATURE_FIELDS.index("internal_level_norm")
    assert features[idx] == 0.0

def test_feature_tensor_shape():
    """Verify feature tensor has correct shape."""
    report = create_test_report()
    features = extract_slot_features(report)
    assert features.shape == (OBS_V3_SLOT_FEATURE_SIZE,)
```

**test_action_masks.py:**
```python
@pytest.mark.parametrize("stage,expected", [
    (SeedStage.DORMANT, False),
    (SeedStage.TRAINING, True),
    (SeedStage.BLENDING, True),
    (SeedStage.HOLDING, True),
    (SeedStage.FOSSILIZED, False),
])
def test_internal_ops_stage_gating(stage, expected):
    """Verify internal ops are gated by stage."""
    report = SeedStateReport(
        stage=stage,
        internal_kind=SeedInternalKind.CONV_LADDER,
        internal_level=2,
        internal_max_level=4,
    )
    mask = compute_lifecycle_op_mask(report, config)

    if expected:
        assert mask[LifecycleOp.GROW_INTERNAL] == 1.0
        assert mask[LifecycleOp.SHRINK_INTERNAL] == 1.0
    else:
        assert mask[LifecycleOp.GROW_INTERNAL] == 0.0
        assert mask[LifecycleOp.SHRINK_INTERNAL] == 0.0

def test_grow_masked_at_max_level():
    """Verify GROW_INTERNAL masked when at max level."""
    report = SeedStateReport(
        stage=SeedStage.TRAINING,
        internal_kind=SeedInternalKind.CONV_LADDER,
        internal_level=4,
        internal_max_level=4,
    )
    mask = compute_lifecycle_op_mask(report, config)
    assert mask[LifecycleOp.GROW_INTERNAL] == 0.0
    assert mask[LifecycleOp.SHRINK_INTERNAL] == 1.0

def test_shrink_masked_at_zero():
    """Verify SHRINK_INTERNAL masked when at level 0."""
    report = SeedStateReport(
        stage=SeedStage.TRAINING,
        internal_kind=SeedInternalKind.CONV_LADDER,
        internal_level=0,
        internal_max_level=4,
    )
    mask = compute_lifecycle_op_mask(report, config)
    assert mask[LifecycleOp.GROW_INTERNAL] == 1.0
    assert mask[LifecycleOp.SHRINK_INTERNAL] == 0.0
```

**test_factored_lstm.py:**
```python
def test_input_dim_assertion():
    """Verify input dim assertions catch mismatches."""
    with pytest.raises(AssertionError):
        FactoredLSTMPolicy(
            base_feature_size=OBS_V3_BASE_FEATURE_SIZE + 1,  # Wrong!
        )

def test_forward_shape_validation():
    """Verify forward pass validates input shapes."""
    policy = FactoredLSTMPolicy()
    bad_obs = {
        "base_features": torch.randn(1, OBS_V3_BASE_FEATURE_SIZE + 1),
        "slot_features": torch.randn(1, NUM_SLOTS, OBS_V3_SLOT_FEATURE_SIZE),
    }
    with pytest.raises(AssertionError):
        policy(bad_obs)
```
