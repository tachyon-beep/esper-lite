# `.get()` Hot Path Remediation Plan

**Created**: 2025-12-25
**Status**: Draft
**Context**: Following typed telemetry migration, explore agents found critical `.get()` patterns in hot path code that silently mask missing required fields.

## Problem Statement

Two critical code paths use `.get()` with defaults for fields that are **required by contract**:

### 1. rewards.py:1492-1494 — `compute_seed_potential()`

```python
def compute_seed_potential(obs: dict) -> float:
    """...
    Args:
        obs: Observation dictionary with has_active_seed, seed_stage, seed_epochs_in_stage
    """
    has_active = obs.get("has_active_seed", 0)      # REQUIRED per docstring!
    seed_stage = obs.get("seed_stage", 0)           # REQUIRED per docstring!
    epochs_in_stage = obs.get("seed_epochs_in_stage", 0)  # REQUIRED per docstring!
```

**Impact**: If any caller passes an observation missing these keys, the function silently returns 0.0 instead of failing. This masks bugs where reward shaping is silently disabled.

### 2. features.py:243 — `extract_tamiyo_features()`

```python
for slot_id in slot_config.slot_ids:
    slot = obs.get('slots', {}).get(slot_id, {})  # Silent cascade failure!
    alpha = safe(slot.get("alpha", 0.0), ...)     # Defaults to 0
    alpha_target = safe(slot.get("alpha_target", alpha), ...)  # Defaults
    # ... 15+ more fields, ALL defaulting to zero
```

**Impact**: If `obs['slots']` is missing or a slot_id is missing, the entire slot's 39-feature vector silently becomes all zeros. Tamiyo's policy sees a "dead slot" instead of an error.

## Design Decision

**Fail loudly when required fields are missing.** This follows the project's "No Legacy Code Policy" and the user's explicit instruction: "always fail loudly when someone is doing something dumb."

### Option 1: Direct Key Access (Chosen)

Replace `.get()` with direct `obs[key]` access for required fields. Missing keys raise `KeyError` immediately.

**Pros:**
- Zero runtime overhead
- Clear stack trace pointing to the exact missing field
- Works with existing dict-based obs

**Cons:**
- KeyError message less informative than custom exception

### Option 2: TypedDict + Validation

Define `SeedObservation` and `SlotObservation` TypedDicts, add runtime validation.

**Pros:**
- IDE autocomplete and mypy checking
- Self-documenting contracts

**Cons:**
- Additional runtime overhead (validation)
- Requires updating all call sites to pass typed dicts

### Option 3: Protocol + Dataclass

Define Protocol for observation access, use dataclasses at call sites.

**Pros:**
- Full type safety
- Immutable if frozen

**Cons:**
- Major refactor of observation construction
- Overkill for hot path code

**Decision**: Option 1 (Direct Key Access) for immediate fix. The fields are already documented as required; we just need to enforce it.

## Implementation Tasks

### Task 1: Fix `compute_seed_potential()` in rewards.py

**File**: `src/esper/simic/rewards/rewards.py`

**Change**:
```python
def compute_seed_potential(obs: dict) -> float:
    """Compute potential value Phi(s) based on seed state.

    Args:
        obs: Observation dictionary. Required keys:
            - has_active_seed: int (0 or 1)
            - seed_stage: int (SeedStage enum value)
            - seed_epochs_in_stage: int (epochs spent in current stage)

    Raises:
        KeyError: If any required key is missing (indicates caller bug)
    """
    has_active = obs["has_active_seed"]
    seed_stage = obs["seed_stage"]
    epochs_in_stage = obs["seed_epochs_in_stage"]

    # ... rest unchanged
```

**Test**: Add test that missing keys raise KeyError:
```python
def test_missing_keys_raise_keyerror(self):
    """Missing required keys should fail loudly."""
    with pytest.raises(KeyError, match="has_active_seed"):
        compute_seed_potential({})

    with pytest.raises(KeyError, match="seed_stage"):
        compute_seed_potential({"has_active_seed": 1})
```

### Task 2: Fix `extract_tamiyo_features()` slot access in features.py

**File**: `src/esper/tamiyo/policy/features.py`

**Change**:
```python
# Before
slot = obs.get('slots', {}).get(slot_id, {})

# After
slots = obs["slots"]  # KeyError if missing 'slots'
slot = slots[slot_id]  # KeyError if slot_id missing
```

**Decision on slot fields**: The individual slot fields (alpha, alpha_target, etc.) have legitimate defaults of 0.0 for newly created slots. Keep `.get()` for these, but document which are truly optional vs which should always exist.

**Test**: Add test that missing slots dict raises KeyError:
```python
def test_missing_slots_raises_keyerror(self):
    """Observations without 'slots' key should fail loudly."""
    obs = _make_minimal_obs()
    del obs["slots"]

    with pytest.raises(KeyError, match="slots"):
        extract_tamiyo_features(obs)
```

### Task 3: Audit slot field requirements

**Analysis needed**: Which slot fields are truly optional (with sensible defaults) vs which indicate a bug if missing?

**Likely required** (presence indicates active slot):
- `stage` - must exist to determine slot state
- `is_active` - explicitly tracked

**Truly optional** (have sensible defaults):
- `alpha`, `alpha_target` - default 0.0 for inactive/new slots
- `alpha_steps_*` - default 0 for no scheduled transition
- Topology features - default 0 for no interactions

### Task 4: Update tests to always provide required fields

Search for tests that construct observation dicts and ensure they provide all required fields. Any test that relies on defaults is implicitly testing incorrect behavior.

**Files to check**:
- `tests/simic/test_rewards.py` - already provides all fields ✓
- `tests/tamiyo/policy/test_features.py` - check slot construction
- `tests/simic/properties/test_pbrs_properties.py` - already provides all fields ✓

### Task 5: Add type hints for documentation

Add TypedDict definitions (not for runtime validation, just documentation):

```python
# In leyline/signals.py or leyline/types.py

class SeedObservationFields(TypedDict, total=True):
    """Required fields for seed potential calculation."""
    has_active_seed: int  # 0 or 1
    seed_stage: int       # SeedStage enum value
    seed_epochs_in_stage: int

class SlotObservationFields(TypedDict, total=False):
    """Slot-level observation fields. All optional with defaults."""
    alpha: float          # Default: 0.0
    alpha_target: float   # Default: alpha
    alpha_mode: int       # Default: AlphaMode.HOLD.value
    stage: int            # Default: 0 (inactive)
    # ... etc
```

This provides IDE hints and documentation without runtime overhead.

## Verification

1. **Unit tests pass**: `pytest tests/simic/test_rewards.py tests/tamiyo/policy/test_features.py`
2. **Integration tests pass**: `pytest tests/integration/`
3. **No silent failures in training**: Run a short training loop and verify telemetry shows expected reward shaping
4. **Mypy**: `mypy src/esper/simic/rewards/rewards.py src/esper/tamiyo/policy/features.py`

## Rollout Risk

**Low**. If any caller was passing incomplete observations, they would already be getting incorrect behavior (silent zeros). Failing loudly exposes bugs, doesn't create them.

**Migration path**: None needed. Breaking on missing required fields is the correct behavior.

## Future Considerations

If we see repeated KeyError issues, consider:
1. Adding a `validate_observation()` helper with informative error messages
2. Using TypedDict for stronger typing at construction sites
3. Moving to structured observation dataclasses (larger refactor)

For now, direct key access is sufficient - it's zero overhead and catches bugs at the exact point of misuse.
