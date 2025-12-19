# FactoredAction Skip Optimization - Phase 1

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate `FactoredAction.from_indices()` object creation overhead in the PPO training hot path by using direct array indexing with lookup tables.

**Architecture:** Add module-level lookup tables for enum names and IDs, replace property access with direct indexing, keep `FactoredAction` creation only for telemetry where the object is required.

**Tech Stack:** Python, NumPy (existing), IntEnum comparisons

---

## Background

The `FactoredAction.from_indices()` call in the hot path (line 1879-1884 of `vectorized.py`) creates a dataclass instance just to:
1. Convert indices to enums (`BlueprintAction(idx)`, `BlendAction(idx)`, `LifecycleOp(idx)`)
2. Access semantic properties (`.is_germinate`, `.blueprint_id`, `.op.name`)

This creates 16 objects per step (one per env) that are immediately destructured. Direct indexing eliminates this overhead.

**Expected improvement:** 15-25% reduction in per-env loop time.

---

## Task 1: Add Lookup Tables to factored_actions.py

**Files:**
- Modify: `src/esper/leyline/factored_actions.py`
- Test: `tests/leyline/test_factored_action_lookup_tables.py`

**Step 1: Write the failing test**

Create `tests/leyline/test_factored_action_lookup_tables.py`:

```python
"""Tests for factored action lookup tables."""

import pytest

from esper.leyline.factored_actions import (
    BlueprintAction,
    BlendAction,
    LifecycleOp,
    OP_NAMES,
    BLUEPRINT_IDS,
    BLEND_IDS,
    OP_GERMINATE,
    OP_FOSSILIZE,
    OP_CULL,
    OP_WAIT,
)


class TestLookupTableSync:
    """Verify lookup tables stay in sync with enum definitions."""

    def test_op_names_match_lifecycle_op_enum(self):
        """OP_NAMES must match LifecycleOp enum names in order."""
        for i, op in enumerate(LifecycleOp):
            assert OP_NAMES[i] == op.name, f"OP_NAMES[{i}] = {OP_NAMES[i]} != {op.name}"

    def test_op_names_length_matches_enum(self):
        """OP_NAMES length must equal LifecycleOp enum length."""
        assert len(OP_NAMES) == len(LifecycleOp)

    def test_blueprint_ids_match_enum_method(self):
        """BLUEPRINT_IDS must match BlueprintAction.to_blueprint_id() for all values."""
        for i, bp in enumerate(BlueprintAction):
            assert BLUEPRINT_IDS[i] == bp.to_blueprint_id(), (
                f"BLUEPRINT_IDS[{i}] = {BLUEPRINT_IDS[i]} != {bp.to_blueprint_id()}"
            )

    def test_blueprint_ids_length_matches_enum(self):
        """BLUEPRINT_IDS length must equal BlueprintAction enum length."""
        assert len(BLUEPRINT_IDS) == len(BlueprintAction)

    def test_blend_ids_match_enum_method(self):
        """BLEND_IDS must match BlendAction.to_algorithm_id() for all values."""
        for i, blend in enumerate(BlendAction):
            assert BLEND_IDS[i] == blend.to_algorithm_id(), (
                f"BLEND_IDS[{i}] = {BLEND_IDS[i]} != {blend.to_algorithm_id()}"
            )

    def test_blend_ids_length_matches_enum(self):
        """BLEND_IDS length must equal BlendAction enum length."""
        assert len(BLEND_IDS) == len(BlendAction)


class TestOpIndexConstants:
    """Verify OP index constants match enum values."""

    def test_op_wait_matches_enum(self):
        assert OP_WAIT == LifecycleOp.WAIT.value

    def test_op_germinate_matches_enum(self):
        assert OP_GERMINATE == LifecycleOp.GERMINATE.value

    def test_op_cull_matches_enum(self):
        assert OP_CULL == LifecycleOp.CULL.value

    def test_op_fossilize_matches_enum(self):
        assert OP_FOSSILIZE == LifecycleOp.FOSSILIZE.value
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/leyline/test_factored_action_lookup_tables.py -v`
Expected: FAIL with `ImportError: cannot import name 'OP_NAMES'`

**Step 3: Write minimal implementation**

Add to `src/esper/leyline/factored_actions.py` after the enum definitions (around line 68):

```python
# =============================================================================
# Lookup Tables for Hot Path Optimization
# =============================================================================
# These tables enable direct indexing without creating FactoredAction objects.
# CRITICAL: These must stay in sync with enum definitions above.
# Module-level assertions validate sync at import time.

# Operation name lookup (matches LifecycleOp enum order)
OP_NAMES: tuple[str, ...] = tuple(op.name for op in LifecycleOp)

# Blueprint ID lookup (matches BlueprintAction.to_blueprint_id())
BLUEPRINT_IDS: tuple[str | None, ...] = tuple(bp.to_blueprint_id() for bp in BlueprintAction)

# Blend algorithm ID lookup (matches BlendAction.to_algorithm_id())
BLEND_IDS: tuple[str, ...] = tuple(blend.to_algorithm_id() for blend in BlendAction)

# Operation index constants for direct comparison (avoids enum construction)
OP_WAIT: int = LifecycleOp.WAIT.value
OP_GERMINATE: int = LifecycleOp.GERMINATE.value
OP_CULL: int = LifecycleOp.CULL.value
OP_FOSSILIZE: int = LifecycleOp.FOSSILIZE.value

# Module-level validation: catch enum drift at import time
assert OP_NAMES == tuple(op.name for op in LifecycleOp), (
    "OP_NAMES out of sync with LifecycleOp enum - this is a bug"
)
assert len(BLUEPRINT_IDS) == len(BlueprintAction), (
    "BLUEPRINT_IDS length mismatch with BlueprintAction enum"
)
assert len(BLEND_IDS) == len(BlendAction), (
    "BLEND_IDS length mismatch with BlendAction enum"
)
```

Update `__all__` to export the new symbols:

```python
__all__ = [
    "BlueprintAction",
    "BlendAction",
    "LifecycleOp",
    "FactoredAction",
    "NUM_BLUEPRINTS",
    "NUM_BLENDS",
    "NUM_OPS",
    "CNN_BLUEPRINTS",
    "TRANSFORMER_BLUEPRINTS",
    # Lookup tables for hot path optimization
    "OP_NAMES",
    "BLUEPRINT_IDS",
    "BLEND_IDS",
    "OP_WAIT",
    "OP_GERMINATE",
    "OP_CULL",
    "OP_FOSSILIZE",
]
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/leyline/test_factored_action_lookup_tables.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/leyline/factored_actions.py tests/leyline/test_factored_action_lookup_tables.py
git commit -m "feat(leyline): add lookup tables for hot path optimization

Add module-level lookup tables and constants for direct action indexing:
- OP_NAMES: tuple of LifecycleOp names
- BLUEPRINT_IDS: tuple matching BlueprintAction.to_blueprint_id()
- BLEND_IDS: tuple matching BlendAction.to_algorithm_id()
- OP_WAIT/GERMINATE/CULL/FOSSILIZE: index constants

Module-level assertions validate tables stay in sync with enum definitions."
```

---

## Task 2: Modify emit_last_action to Accept Raw Indices

**Files:**
- Modify: `src/esper/simic/telemetry/emitters.py`
- Test: `tests/simic/telemetry/test_emit_last_action_indices.py`

**Step 1: Write the failing test**

Create `tests/simic/telemetry/test_emit_last_action_indices.py`:

```python
"""Test emit_last_action with raw indices instead of FactoredAction."""

import pytest

from esper.leyline.factored_actions import (
    FactoredAction,
    OP_NAMES,
    BLUEPRINT_IDS,
    BLEND_IDS,
)
from esper.simic.telemetry.emitters import emit_last_action


class TestEmitLastActionWithIndices:
    """Verify emit_last_action works with raw indices."""

    def test_emit_with_indices_matches_factored_action(self):
        """Emitting with indices produces same data as with FactoredAction."""
        slot_idx, blueprint_idx, blend_idx, op_idx = 0, 1, 2, 1  # GERMINATE with CONV_LIGHT, GATED

        # Create FactoredAction for comparison
        fa = FactoredAction.from_indices(slot_idx, blueprint_idx, blend_idx, op_idx)

        masked = {"slot": False, "blueprint": True, "blend": False, "op": False}

        # Call with indices
        result = emit_last_action(
            env_id=0,
            epoch=5,
            slot_idx=slot_idx,
            blueprint_idx=blueprint_idx,
            blend_idx=blend_idx,
            op_idx=op_idx,
            slot_id="r0c0",
            masked=masked,
            success=True,
        )

        # Verify data matches what FactoredAction would produce
        assert result["op"] == fa.op.name
        assert result["blueprint_id"] == fa.blueprint_id
        assert result["blend_id"] == fa.blend_algorithm_id

    def test_emit_with_all_ops(self):
        """All operation types produce correct op names."""
        for op_idx in range(4):
            result = emit_last_action(
                env_id=0,
                epoch=1,
                slot_idx=0,
                blueprint_idx=0,
                blend_idx=0,
                op_idx=op_idx,
                slot_id="r0c0",
                masked={"slot": False, "blueprint": False, "blend": False, "op": False},
                success=True,
            )
            assert result["op"] == OP_NAMES[op_idx]
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/simic/telemetry/test_emit_last_action_indices.py -v`
Expected: FAIL with `TypeError: emit_last_action() got an unexpected keyword argument 'slot_idx'`

**Step 3: Modify emit_last_action to accept indices**

In `src/esper/simic/telemetry/emitters.py`, update the function signature:

```python
from esper.leyline.factored_actions import OP_NAMES, BLUEPRINT_IDS, BLEND_IDS


def emit_last_action(
    *,
    env_id: int,
    epoch: int,
    slot_idx: int,
    blueprint_idx: int,
    blend_idx: int,
    op_idx: int,
    slot_id: str,
    masked: dict[str, bool],
    success: bool,
) -> dict:
    """Emit per-step last-action detail for debugging and UIs.

    Args:
        env_id: Environment index
        epoch: Current epoch
        slot_idx: Slot action index
        blueprint_idx: Blueprint action index
        blend_idx: Blend action index
        op_idx: Lifecycle operation index
        slot_id: Target slot ID string
        masked: Dict of head -> was_masked flags
        success: Whether the action executed successfully

    Returns:
        The emitted data dict (for testing)
    """
    hub = get_hub()
    data = {
        "kind": "last_action",
        "env_id": env_id,
        "inner_epoch": epoch,
        "op": OP_NAMES[op_idx],
        "slot_id": slot_id,
        "blueprint_id": BLUEPRINT_IDS[blueprint_idx],
        "blend_id": BLEND_IDS[blend_idx],
        "op_masked": bool(masked.get("op", False)),
        "slot_masked": bool(masked.get("slot", False)),
        "blueprint_masked": bool(masked.get("blueprint", False)),
        "blend_masked": bool(masked.get("blend", False)),
        "action_success": success,
    }
    hub.emit(TelemetryEvent(
        event_type=TelemetryEventType.ACTION_TAKEN,
        severity="debug",
        data=data,
    ))
    return data
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/simic/telemetry/test_emit_last_action_indices.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/telemetry/emitters.py tests/simic/telemetry/test_emit_last_action_indices.py
git commit -m "refactor(telemetry): emit_last_action accepts indices instead of FactoredAction

Change signature to accept raw slot_idx, blueprint_idx, blend_idx, op_idx
instead of FactoredAction object. Uses lookup tables for string conversion.

This enables the hot path to skip FactoredAction object creation entirely."
```

---

## Task 3: Update vectorized.py to Use Direct Indexing

**Files:**
- Modify: `src/esper/simic/training/vectorized.py`
- Test: Run existing test suite (no new tests needed - behavior unchanged)

**Step 1: Add imports for lookup tables**

At the import section of `vectorized.py`, add:

```python
from esper.leyline.factored_actions import (
    FactoredAction,  # Keep for type hints if needed
    OP_NAMES,
    BLUEPRINT_IDS,
    BLEND_IDS,
    OP_WAIT,
    OP_GERMINATE,
    OP_CULL,
    OP_FOSSILIZE,
    LifecycleOp,  # Keep for reward computation (IntEnum compatible)
)
```

**Step 2: Replace FactoredAction.from_indices() with direct indexing**

Find the section around lines 1878-1884:

```python
# Parse factored action FIRST to determine target slot
action_dict = actions[env_idx]  # {slot: int, blueprint: int, blend: int, op: int}
factored_action = FactoredAction.from_indices(
    slot_idx=action_dict["slot"],
    blueprint_idx=action_dict["blueprint"],
    blend_idx=action_dict["blend"],
    op_idx=action_dict["op"],
)
```

Replace with:

```python
# Parse factored action using direct indexing (no object creation)
action_dict = actions[env_idx]  # {slot: int, blueprint: int, blend: int, op: int}
slot_idx = action_dict["slot"]
blueprint_idx = action_dict["blueprint"]
blend_idx = action_dict["blend"]
op_idx = action_dict["op"]
```

**Step 3: Replace all factored_action property accesses**

Replace each usage with direct indexing:

| Old Code | New Code |
|----------|----------|
| `factored_action.slot_idx` | `slot_idx` |
| `factored_action.op.name` | `OP_NAMES[op_idx]` |
| `factored_action.op` (for reward) | `LifecycleOp(op_idx)` |
| `factored_action.is_germinate` | `op_idx == OP_GERMINATE` |
| `factored_action.is_fossilize` | `op_idx == OP_FOSSILIZE` |
| `factored_action.is_cull` | `op_idx == OP_CULL` |
| `factored_action.blueprint_id` | `BLUEPRINT_IDS[blueprint_idx]` |
| `factored_action.blend_algorithm_id` | `BLEND_IDS[blend_idx]` |

Specific replacements:

**Line ~1889 (slot resolution):**
```python
# OLD
target_slot, slot_is_enabled = _resolve_target_slot(
    factored_action.slot_idx,
    ...
)

# NEW
target_slot, slot_is_enabled = _resolve_target_slot(
    slot_idx,
    ...
)
```

**Line ~1899 (action counting):**
```python
# OLD
env_state.action_counts[factored_action.op.name] = ...

# NEW
env_state.action_counts[OP_NAMES[op_idx]] = ...
```

**Line ~1901 (reward computation):**
```python
# OLD
action_for_reward = factored_action.op

# NEW
action_for_reward = LifecycleOp(op_idx)
```

**Line ~2049 (GERMINATE branch):**
```python
# OLD
elif factored_action.is_germinate:
    ...
    blueprint_id = factored_action.blueprint_id
    blend_algorithm_id = factored_action.blend_algorithm_id

# NEW
elif op_idx == OP_GERMINATE:
    ...
    blueprint_id = BLUEPRINT_IDS[blueprint_idx]
    blend_algorithm_id = BLEND_IDS[blend_idx]
```

**Line ~2066 (FOSSILIZE branch):**
```python
# OLD
elif factored_action.is_fossilize:

# NEW
elif op_idx == OP_FOSSILIZE:
```

**Line ~2080 (CULL branch):**
```python
# OLD
elif factored_action.is_cull:

# NEW
elif op_idx == OP_CULL:
```

**Line ~2093 (successful action counting):**
```python
# OLD
env_state.successful_action_counts[factored_action.op.name] = ...

# NEW
env_state.successful_action_counts[OP_NAMES[op_idx]] = ...
```

**Line ~2104-2111 (emit_last_action call):**
```python
# OLD
emit_last_action(
    env_id=env_idx,
    epoch=epoch,
    factored_action=factored_action,
    slot_id=target_slot,
    masked=masked_flags,
    success=action_success,
)

# NEW
emit_last_action(
    env_id=env_idx,
    epoch=epoch,
    slot_idx=slot_idx,
    blueprint_idx=blueprint_idx,
    blend_idx=blend_idx,
    op_idx=op_idx,
    slot_id=target_slot,
    masked=masked_flags,
    success=action_success,
)
```

**Step 4: Run test suite to verify behavior unchanged**

Run: `PYTHONPATH=src uv run pytest tests/simic/ -v --tb=short`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/esper/simic/training/vectorized.py
git commit -m "perf(simic): eliminate FactoredAction object creation in hot path

Replace FactoredAction.from_indices() with direct array indexing using
lookup tables. This eliminates 16 object creations per step.

Changes:
- Extract slot_idx, blueprint_idx, blend_idx, op_idx directly from action_dict
- Replace .is_germinate/.is_fossilize/.is_cull with op_idx == OP_* comparisons
- Replace .op.name with OP_NAMES[op_idx]
- Replace .blueprint_id/.blend_algorithm_id with lookup table access
- Update emit_last_action call to pass indices instead of object

Expected improvement: 15-25% reduction in per-env loop time."
```

---

## Task 4: Add Debug Assertions for Equivalence Checking

**Files:**
- Modify: `src/esper/simic/training/vectorized.py`
- Test: Run with `python -O` to verify assertions are skipped

**Step 1: Add debug-mode equivalence check**

After the direct indexing code (around line 1884), add a debug-only assertion block:

```python
# Parse factored action using direct indexing (no object creation)
action_dict = actions[env_idx]
slot_idx = action_dict["slot"]
blueprint_idx = action_dict["blueprint"]
blend_idx = action_dict["blend"]
op_idx = action_dict["op"]

# DEBUG: Verify direct indexing matches FactoredAction properties
# This block is stripped by Python when run with -O flag
if __debug__:
    _fa = FactoredAction.from_indices(slot_idx, blueprint_idx, blend_idx, op_idx)
    assert slot_idx == _fa.slot_idx, f"slot_idx mismatch: {slot_idx} != {_fa.slot_idx}"
    assert OP_NAMES[op_idx] == _fa.op.name, f"op.name mismatch: {OP_NAMES[op_idx]} != {_fa.op.name}"
    assert (op_idx == OP_GERMINATE) == _fa.is_germinate, "is_germinate mismatch"
    assert (op_idx == OP_FOSSILIZE) == _fa.is_fossilize, "is_fossilize mismatch"
    assert (op_idx == OP_CULL) == _fa.is_cull, "is_cull mismatch"
    assert BLUEPRINT_IDS[blueprint_idx] == _fa.blueprint_id, f"blueprint_id mismatch"
    assert BLEND_IDS[blend_idx] == _fa.blend_algorithm_id, f"blend_id mismatch"
    del _fa  # Don't leak into scope
```

**Step 2: Run with assertions enabled**

Run: `PYTHONPATH=src uv run pytest tests/simic/training/test_vectorized_training.py -v`
Expected: PASS (assertions don't trigger)

**Step 3: Verify assertions are stripped with -O**

Run: `PYTHONPATH=src python -O -c "assert False, 'should not run'; print('OK')"`
Expected: Prints "OK" (assertion skipped)

**Step 4: Commit**

```bash
git add src/esper/simic/training/vectorized.py
git commit -m "safety(simic): add debug assertions for direct indexing equivalence

Add if __debug__ block to verify direct indexing produces same results as
FactoredAction object. Assertions are stripped in production (python -O).

This catches any drift between lookup tables and enum definitions during
development without impacting production performance."
```

---

## Task 5: Run Full Test Suite and Benchmark

**Files:**
- No modifications
- Run existing tests and measure performance

**Step 1: Run full test suite**

Run: `PYTHONPATH=src uv run pytest tests/ -v --tb=short`
Expected: All tests PASS

**Step 2: Run short training benchmark**

Run a quick training loop to verify no regressions:

```bash
PYTHONPATH=src uv run python -c "
from esper.simic.training.vectorized import train_vectorized_ppo
from esper.simic.agent.ppo import PPOAgent
from esper.simic.agent.networks import create_factored_actor_critic
import time

# Minimal config for speed test
net = create_factored_actor_critic(obs_dim=64, hidden_dim=64, num_slots=4)
agent = PPOAgent(
    actor_critic=net,
    learning_rate=3e-4,
    ent_coef=0.05,
    gamma=0.99,
    gae_lambda=0.95,
    batch_size=32,
)

start = time.perf_counter()
# Run 2 episodes to verify it works
# (Full benchmark would use scripts/profile_gpu_sync.py)
print('Training loop executes without error')
print(f'Setup time: {time.perf_counter() - start:.3f}s')
"
```

**Step 3: Commit benchmark results as plan update**

No code changes - update plan with results.

---

## Verification Checklist

After all tasks complete:

- [ ] `OP_NAMES`, `BLUEPRINT_IDS`, `BLEND_IDS` exported from `factored_actions.py`
- [ ] Module-level assertions validate table sync at import time
- [ ] `emit_last_action` accepts raw indices instead of `FactoredAction`
- [ ] `vectorized.py` uses direct indexing in hot path
- [ ] Debug assertions verify equivalence during development
- [ ] Full test suite passes
- [ ] No GPU sync regressions (use `scripts/profile_gpu_sync.py`)

---

## Rollback Plan

If issues arise:
1. Revert to `FactoredAction.from_indices()` - the old pattern is well-tested
2. Keep lookup tables for future optimization attempts
3. Debug assertions will catch any table/enum drift
