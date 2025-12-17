# Kasmina Stability Fixes + Coordinate System Migration — Full Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make Kasmina seed lifecycle robust under PyTorch 2.9 and Python 3.13, eliminate known correctness traps, and migrate slot addressing from `early/mid/late` to canonical 2D coordinates (`"r{row}c{col}"`).

**Architecture:** Clean break migration (no backwards compatibility). Fix checkpoint serialization first (M4), then gated blending semantics (M2), then coordinate system (M1), then RL action space (M1.5). Test suite migration runs throughout.

**Tech Stack:** Python 3.13, PyTorch 2.9, pytest, Hypothesis

---

## Pre-Work Status (Completed)

| Work Package | Status | Commit | Key Findings |
|--------------|--------|--------|--------------|
| **WP-A**: Checkpoint Audit | ✅ DONE | `05e5bdd` | PPO compatible, MorphogeneticModel has 10-17 issues per stage |
| **WP-B**: Dynamic Slots Spike | ✅ DONE | `e66c0e8` | GREEN LIGHT for M1.5, network already supports N slots |
| **WP-C**: Slot ID Module | ✅ DONE | `8f605d2` | `leyline/slot_id.py` implemented with format/parse/validate |
| **WP-D**: Gated Blending Characterization | ✅ DONE | `e7bf96f` | G3 gate NEVER passes with gated blending (state.alpha=0.5) |

### Critical Findings from Pre-Work

1. **Checkpoint Issues (WP-A):**
   - `SeedState` dataclass needs `to_dict()`/`from_dict()`
   - `SeedStage` enum → convert to int
   - `datetime` → convert to ISO 8601 string
   - `deque` → convert to list
   - `alpha_schedule` nn.Module persists after BLENDING (should be discarded)

2. **Dynamic Slots (WP-B):**
   - Network layer ALREADY supports N slots
   - Blockers are integration-only: `action_masks.py`, `factored_actions.py`
   - Checkpoints NOT portable across `num_slots` changes (expected, document it)

3. **Gated Blending Bug (WP-D):**
   - `get_alpha()` always returns 0.5, lifecycle uses this for `state.alpha`
   - G3 gate checks `state.alpha >= 0.95`, so seeds are permanently stuck
   - GatedBlend params ARE trained (PyTorch auto-registers submodules)

---

## Milestone 4: Checkpoint Compatibility (PyTorch 2.9)

**Priority:** CRITICAL — blocks all other work
**Affected files:** `kasmina/slot.py`, `leyline/telemetry.py`, `leyline/stages.py`

### Task 4.1: Add SeedState.to_dict() and from_dict()

**Files:**
- Modify: `src/esper/kasmina/slot.py`
- Test: `tests/kasmina/test_seed_state_serialization.py`

**Step 1: Write the failing test**

```python
# tests/kasmina/test_seed_state_serialization.py
"""Test SeedState serialization for PyTorch 2.9 weights_only=True."""

import pytest
from datetime import datetime, timezone
from collections import deque

from esper.kasmina.slot import SeedState, SeedMetrics
from esper.leyline.stages import SeedStage


class TestSeedStateToDict:
    """Test SeedState.to_dict() produces only primitives."""

    def test_to_dict_returns_primitives_only(self):
        """to_dict() output contains no custom types."""
        state = SeedState(
            seed_id="test-seed",
            blueprint_id="norm",
            slot_id="r0c0",
            stage=SeedStage.TRAINING,
        )

        result = state.to_dict()

        # Must be a plain dict
        assert isinstance(result, dict)
        # Stage must be int, not Enum
        assert isinstance(result["stage"], int)
        assert result["stage"] == SeedStage.TRAINING.value
        # Datetime must be string
        assert isinstance(result["stage_entered_at"], str)
        # Stage history must be list, not deque
        assert isinstance(result["stage_history"], list)

    def test_to_dict_from_dict_roundtrip(self):
        """to_dict() -> from_dict() preserves all state."""
        original = SeedState(
            seed_id="test-seed",
            blueprint_id="norm",
            slot_id="r0c0",
            stage=SeedStage.BLENDING,
            previous_stage=SeedStage.TRAINING,
        )
        original.alpha = 0.75
        original.stage_history.append((SeedStage.GERMINATED, datetime.now(timezone.utc)))

        data = original.to_dict()
        restored = SeedState.from_dict(data)

        assert restored.seed_id == original.seed_id
        assert restored.blueprint_id == original.blueprint_id
        assert restored.slot_id == original.slot_id
        assert restored.stage == original.stage
        assert restored.previous_stage == original.previous_stage
        assert restored.alpha == original.alpha
        assert len(restored.stage_history) == len(original.stage_history)
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/kasmina/test_seed_state_serialization.py -v`
Expected: FAIL with `AttributeError: 'SeedState' object has no attribute 'to_dict'`

**Step 3: Implement to_dict() and from_dict()**

Add to `src/esper/kasmina/slot.py` in the `SeedState` class:

```python
def to_dict(self) -> dict:
    """Convert to primitive dict for PyTorch 2.9 weights_only=True serialization."""
    return {
        "seed_id": self.seed_id,
        "blueprint_id": self.blueprint_id,
        "slot_id": self.slot_id,
        "stage": self.stage.value,  # Enum -> int
        "previous_stage": self.previous_stage.value if self.previous_stage else None,
        "stage_entered_at": self.stage_entered_at.isoformat(),  # datetime -> str
        "alpha": self.alpha,
        "stage_history": [
            (stage.value, ts.isoformat()) for stage, ts in self.stage_history
        ],  # deque of (Enum, datetime) -> list of (int, str)
        "blend_algorithm_id": self.blend_algorithm_id,
        "metrics": self.metrics.to_dict() if self.metrics else None,
        "telemetry": self.telemetry.to_dict() if self.telemetry else None,
    }

@classmethod
def from_dict(cls, data: dict) -> "SeedState":
    """Reconstruct from primitive dict."""
    from datetime import datetime
    from collections import deque

    state = cls(
        seed_id=data["seed_id"],
        blueprint_id=data["blueprint_id"],
        slot_id=data["slot_id"],
        stage=SeedStage(data["stage"]),
        previous_stage=SeedStage(data["previous_stage"]) if data.get("previous_stage") else None,
        blend_algorithm_id=data.get("blend_algorithm_id"),
    )
    state.stage_entered_at = datetime.fromisoformat(data["stage_entered_at"])
    state.alpha = data.get("alpha", 0.0)
    state.stage_history = deque(
        (SeedStage(stage), datetime.fromisoformat(ts))
        for stage, ts in data.get("stage_history", [])
    )
    if data.get("metrics"):
        state.metrics = SeedMetrics.from_dict(data["metrics"])
    if data.get("telemetry"):
        state.telemetry = SeedTelemetry.from_dict(data["telemetry"])
    return state
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/kasmina/test_seed_state_serialization.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/kasmina/slot.py tests/kasmina/test_seed_state_serialization.py
git commit -m "feat(kasmina): add SeedState.to_dict/from_dict for PyTorch 2.9 compat"
```

---

### Task 4.2: Add SeedMetrics.to_dict() and from_dict()

**Files:**
- Modify: `src/esper/kasmina/slot.py`
- Test: `tests/kasmina/test_seed_state_serialization.py`

**Step 1: Write the failing test**

Add to test file:

```python
class TestSeedMetricsToDict:
    """Test SeedMetrics serialization."""

    def test_metrics_to_dict_roundtrip(self):
        """to_dict() -> from_dict() preserves metrics."""
        metrics = SeedMetrics()
        metrics.epochs_in_current_stage = 5
        metrics.blending_steps_completed = 100

        data = metrics.to_dict()
        restored = SeedMetrics.from_dict(data)

        assert restored.epochs_in_current_stage == 5
        assert restored.blending_steps_completed == 100
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/kasmina/test_seed_state_serialization.py::TestSeedMetricsToDict -v`
Expected: FAIL

**Step 3: Implement**

Add to `SeedMetrics` class in `src/esper/kasmina/slot.py`:

```python
def to_dict(self) -> dict:
    """Convert to primitive dict."""
    return {
        "epochs_in_current_stage": self.epochs_in_current_stage,
        "blending_steps_completed": self.blending_steps_completed,
        "total_epochs": self.total_epochs,
        "accuracy_at_blending_start": self.accuracy_at_blending_start,
        "loss_at_blending_start": self.loss_at_blending_start,
    }

@classmethod
def from_dict(cls, data: dict) -> "SeedMetrics":
    """Reconstruct from primitive dict."""
    metrics = cls()
    metrics.epochs_in_current_stage = data.get("epochs_in_current_stage", 0)
    metrics.blending_steps_completed = data.get("blending_steps_completed", 0)
    metrics.total_epochs = data.get("total_epochs", 0)
    metrics.accuracy_at_blending_start = data.get("accuracy_at_blending_start")
    metrics.loss_at_blending_start = data.get("loss_at_blending_start")
    return metrics
```

**Step 4: Run test**

Run: `PYTHONPATH=src uv run pytest tests/kasmina/test_seed_state_serialization.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/kasmina/slot.py tests/kasmina/test_seed_state_serialization.py
git commit -m "feat(kasmina): add SeedMetrics.to_dict/from_dict"
```

---

### Task 4.3: Add SeedTelemetry.to_dict() and from_dict()

**Files:**
- Modify: `src/esper/leyline/telemetry.py`
- Test: `tests/leyline/test_telemetry_serialization.py`

**Step 1: Write the failing test**

```python
# tests/leyline/test_telemetry_serialization.py
"""Test SeedTelemetry serialization."""

import pytest
from datetime import datetime, timezone

from esper.leyline.telemetry import SeedTelemetry


class TestSeedTelemetryToDict:
    """Test SeedTelemetry.to_dict() produces only primitives."""

    def test_to_dict_converts_datetime(self):
        """captured_at datetime is converted to ISO string."""
        telemetry = SeedTelemetry(captured_at=datetime.now(timezone.utc))

        data = telemetry.to_dict()

        assert isinstance(data["captured_at"], str)

    def test_roundtrip(self):
        """to_dict() -> from_dict() preserves data."""
        original = SeedTelemetry(captured_at=datetime.now(timezone.utc))

        data = original.to_dict()
        restored = SeedTelemetry.from_dict(data)

        # Allow small time delta due to serialization precision
        assert abs((restored.captured_at - original.captured_at).total_seconds()) < 1
```

**Step 2: Run test**

Run: `PYTHONPATH=src uv run pytest tests/leyline/test_telemetry_serialization.py -v`
Expected: FAIL

**Step 3: Implement**

Add to `SeedTelemetry` in `src/esper/leyline/telemetry.py`:

```python
def to_dict(self) -> dict:
    """Convert to primitive dict for serialization."""
    return {
        "captured_at": self.captured_at.isoformat() if self.captured_at else None,
        # Add other fields as needed
    }

@classmethod
def from_dict(cls, data: dict) -> "SeedTelemetry":
    """Reconstruct from primitive dict."""
    from datetime import datetime

    return cls(
        captured_at=datetime.fromisoformat(data["captured_at"]) if data.get("captured_at") else None,
    )
```

**Step 4: Run test**

Run: `PYTHONPATH=src uv run pytest tests/leyline/test_telemetry_serialization.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/leyline/telemetry.py tests/leyline/test_telemetry_serialization.py
git commit -m "feat(leyline): add SeedTelemetry.to_dict/from_dict"
```

---

### Task 4.4: Update SeedSlot.get_extra_state() to use primitives

**Files:**
- Modify: `src/esper/kasmina/slot.py`
- Test: `tests/kasmina/test_seed_slot_checkpoint.py`

**Step 1: Write the failing test**

```python
# tests/kasmina/test_seed_slot_checkpoint.py
"""Test SeedSlot checkpoint compatibility with PyTorch 2.9."""

import pytest
import torch
import tempfile
from pathlib import Path

from esper.kasmina.slot import SeedSlot
from esper.simic.features import TaskConfig
from esper.leyline.stages import SeedStage


class TestSeedSlotCheckpoint:
    """Test SeedSlot save/load with weights_only=True."""

    @pytest.fixture
    def slot(self):
        """Create a SeedSlot in BLENDING stage."""
        slot = SeedSlot(
            slot_id="r0c0",
            channels=64,
            device="cpu",
            task_config=TaskConfig(
                topology="cnn",
                blending_steps=10,
                num_classes=10,
                input_channels=3,
                input_height=32,
                input_width=32,
            ),
        )
        slot.germinate(
            blueprint_id="norm",
            seed_id="test-seed",
            blend_algorithm_id="linear",
        )
        slot.state.transition(SeedStage.TRAINING)
        slot.state.transition(SeedStage.BLENDING)
        slot.start_blending(total_steps=10)
        return slot

    def test_extra_state_contains_only_primitives(self, slot):
        """get_extra_state() returns only primitive types."""
        extra = slot.get_extra_state()

        # Recursively check no custom types
        def check_primitives(obj, path=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    check_primitives(v, f"{path}.{k}")
            elif isinstance(obj, (list, tuple)):
                for i, v in enumerate(obj):
                    check_primitives(v, f"{path}[{i}]")
            elif obj is None:
                pass
            elif isinstance(obj, (str, int, float, bool)):
                pass
            else:
                pytest.fail(f"Non-primitive at {path}: {type(obj).__name__}")

        check_primitives(extra)

    def test_checkpoint_roundtrip_weights_only(self, slot):
        """Save and load with weights_only=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"

            # Save
            torch.save(slot.state_dict(), path)

            # Load with weights_only=True (PyTorch 2.9 default)
            loaded = torch.load(path, weights_only=True)

            # Should not raise
            assert "seed_slots.r0c0._extra_state" in str(loaded) or loaded is not None
```

**Step 2: Run test**

Run: `PYTHONPATH=src uv run pytest tests/kasmina/test_seed_slot_checkpoint.py -v`
Expected: FAIL (extra_state contains nn.Module)

**Step 3: Update get_extra_state()**

Modify `SeedSlot.get_extra_state()` in `src/esper/kasmina/slot.py`:

```python
def get_extra_state(self) -> dict:
    """Persist SeedState for PyTorch 2.9+ weights_only=True compatibility.

    Returns only primitive types (dict, list, str, int, float, bool, None).
    The alpha_schedule nn.Module weights are saved via state_dict(), not here.
    """
    state_dict = {
        "isolate_gradients": self.isolate_gradients,
    }

    if self.state is not None:
        state_dict["seed_state"] = self.state.to_dict()

    # Alpha schedule: save config only, not the nn.Module
    # The nn.Module weights are saved in state_dict() automatically
    if self.alpha_schedule is not None:
        state_dict["alpha_schedule_config"] = {
            "algorithm_id": getattr(self.alpha_schedule, "algorithm_id", None),
            "total_steps": getattr(self.alpha_schedule, "total_steps", None),
            "current_step": getattr(self.alpha_schedule, "_step", 0),
        }
    else:
        state_dict["alpha_schedule_config"] = None

    return state_dict
```

**Step 4: Update set_extra_state()**

```python
def set_extra_state(self, state: dict) -> None:
    """Restore SeedState from primitive dict."""
    self.isolate_gradients = state.get("isolate_gradients", False)

    if state.get("seed_state"):
        self.state = SeedState.from_dict(state["seed_state"])

    # Alpha schedule reconstruction handled separately if needed
    # The nn.Module weights are restored via load_state_dict()
    if state.get("alpha_schedule_config"):
        config = state["alpha_schedule_config"]
        if config.get("algorithm_id") and self.state and self.state.stage == SeedStage.BLENDING:
            self.start_blending(total_steps=config.get("total_steps", 10))
            # Restore step count
            if hasattr(self.alpha_schedule, "_step"):
                self.alpha_schedule._step = config.get("current_step", 0)
```

**Step 5: Run test**

Run: `PYTHONPATH=src uv run pytest tests/kasmina/test_seed_slot_checkpoint.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/esper/kasmina/slot.py tests/kasmina/test_seed_slot_checkpoint.py
git commit -m "feat(kasmina): update SeedSlot extra_state for PyTorch 2.9 compat"
```

---

### Task 4.5: Discard alpha_schedule after BLENDING completes

**Files:**
- Modify: `src/esper/kasmina/slot.py`
- Test: `tests/kasmina/test_gated_blending_characterization.py` (update)

**Step 1: Write the failing test**

Add to characterization test file or create new:

```python
class TestAlphaScheduleCleanup:
    """Test alpha_schedule is discarded after BLENDING."""

    def test_alpha_schedule_cleared_on_probationary_transition(self):
        """alpha_schedule should be None after BLENDING -> PROBATIONARY."""
        slot = SeedSlot(
            slot_id="r0c0",
            channels=64,
            device="cpu",
            task_config=TaskConfig(
                topology="cnn",
                blending_steps=3,
                num_classes=10,
                input_channels=3,
                input_height=32,
                input_width=32,
            ),
        )
        slot.germinate(
            blueprint_id="norm",
            seed_id="test-seed",
            blend_algorithm_id="linear",
        )
        slot.state.transition(SeedStage.TRAINING)
        slot.state.transition(SeedStage.BLENDING)
        slot.start_blending(total_steps=3)

        # Verify schedule exists during BLENDING
        assert slot.alpha_schedule is not None

        # Force transition to PROBATIONARY
        slot.state.alpha = 1.0
        slot.state.transition(SeedStage.PROBATIONARY)
        slot._on_blending_complete()  # Cleanup hook

        # Schedule should be cleared
        assert slot.alpha_schedule is None
        assert slot.state.alpha == 1.0
```

**Step 2: Run test**

Run: `PYTHONPATH=src uv run pytest tests/kasmina/test_seed_slot_checkpoint.py::TestAlphaScheduleCleanup -v`
Expected: FAIL

**Step 3: Implement cleanup**

Add to `SeedSlot` in `src/esper/kasmina/slot.py`:

```python
def _on_blending_complete(self) -> None:
    """Clean up after BLENDING stage completes.

    Discards alpha_schedule (no longer needed after full integration).
    Sets state.alpha = 1.0 (permanently fully blended).
    """
    self.alpha_schedule = None
    if self.state:
        self.state.alpha = 1.0
```

And update the transition logic to call this when entering PROBATIONARY from BLENDING.

**Step 4: Run test**

Run: `PYTHONPATH=src uv run pytest tests/kasmina/test_seed_slot_checkpoint.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/kasmina/slot.py tests/kasmina/test_seed_slot_checkpoint.py
git commit -m "fix(kasmina): discard alpha_schedule after BLENDING completes"
```

---

### Task 4.6: Verify checkpoint compatibility with audit tool

**Files:**
- None (verification task)

**Step 1: Run audit on BLENDING checkpoint**

```bash
PYTHONPATH=src python scripts/checkpoint_audit.py --generate-blending
```

Expected: `SUCCESS - checkpoint is already compatible!`

**Step 2: Run audit on PROBATIONARY checkpoint**

```bash
PYTHONPATH=src python scripts/checkpoint_audit.py --generate-probationary
```

Expected: `SUCCESS - checkpoint is already compatible!`

**Step 3: Commit verification**

```bash
git commit --allow-empty -m "verify: M4 checkpoint compatibility confirmed via audit"
```

---

## Milestone 2: Lifecycle and Blending Correctness

**Priority:** HIGH — fixes critical gated blending bug
**Affected files:** `kasmina/slot.py`, `kasmina/blending.py`

### Task 2.1: Fix GatedBlend.get_alpha() to track progress

**Files:**
- Modify: `src/esper/kasmina/blending.py`
- Test: `tests/kasmina/test_gated_blending_characterization.py` (update)

**Step 1: Write the test for new behavior**

```python
class TestGatedBlendFixed:
    """Test fixed GatedBlend behavior (post-M2)."""

    def test_get_alpha_tracks_step_progress(self):
        """get_alpha() should return step-based progress, not constant 0.5."""
        gate = GatedBlend(channels=64, topology="cnn", total_steps=10)

        # At step 0, alpha should be low
        gate.step(0)
        assert gate.get_alpha(0) < 0.2

        # At step 5, alpha should be ~0.5
        gate.step(5)
        alpha_mid = gate.get_alpha(5)
        assert 0.4 < alpha_mid < 0.6

        # At step 10, alpha should be ~1.0
        gate.step(10)
        assert gate.get_alpha(10) >= 0.95
```

**Step 2: Run test**

Expected: FAIL (currently returns 0.5 always)

**Step 3: Implement fix**

Update `GatedBlend.get_alpha()` in `src/esper/kasmina/blending.py`:

```python
def get_alpha(self, step: int | None = None) -> float:
    """Return blending progress for lifecycle tracking.

    Unlike schedule-based blends, gated blending uses learned gates
    during forward(). For lifecycle/G3 gate compatibility, we report
    step-based progress: step / total_steps.

    This ensures G3 gate can pass naturally when blending completes.
    """
    if self.total_steps is None or self.total_steps == 0:
        return 0.5  # Fallback if not configured

    current = step if step is not None else self._step
    return min(1.0, current / self.total_steps)
```

**Step 4: Run test**

Expected: PASS

**Step 5: Update characterization tests**

Update `tests/kasmina/test_gated_blending_characterization.py` to reflect new expected behavior (remove CURRENT_BEHAVIOR markers for fixed tests).

**Step 6: Commit**

```bash
git add src/esper/kasmina/blending.py tests/kasmina/test_gated_blending_characterization.py
git commit -m "fix(kasmina): GatedBlend.get_alpha() tracks step progress for G3 gate"
```

---

### Task 2.2: Unify advance_stage() with step_epoch() transition handling

**Files:**
- Modify: `src/esper/kasmina/slot.py`
- Test: `tests/kasmina/test_step_epoch_lifecycle.py`

**Step 1: Write the test**

```python
class TestUnifiedTransitions:
    """Test advance_stage() and step_epoch() behave identically."""

    def test_advance_stage_initializes_blending(self):
        """advance_stage() to BLENDING should initialize schedule."""
        slot = create_slot_in_training()

        # Use advance_stage directly
        slot.advance_stage(SeedStage.BLENDING)

        assert slot.state.stage == SeedStage.BLENDING
        assert slot._blending_started is True
        assert slot.alpha_schedule is not None
```

**Step 2: Implement _on_enter_stage() hook**

```python
def _on_enter_stage(self, new_stage: SeedStage, old_stage: SeedStage) -> None:
    """Handle stage entry logic uniformly."""
    if new_stage == SeedStage.BLENDING and old_stage == SeedStage.TRAINING:
        self._blending_started = True
        self.start_blending(total_steps=self.task_config.blending_steps)
        if self.state:
            self.state.metrics.accuracy_at_blending_start = self._last_accuracy

    elif new_stage == SeedStage.PROBATIONARY and old_stage == SeedStage.BLENDING:
        self._on_blending_complete()
```

**Step 3: Run tests**

**Step 4: Commit**

```bash
git commit -m "refactor(kasmina): unify stage transition logic via _on_enter_stage()"
```

---

## Milestone 1: Slot Identity Migration (Coordinate System)

**Priority:** MEDIUM — WP-C already completed slot_id module
**Remaining work:** Update consumers to use canonical IDs

### Task 1.1: Update _SLOT_ID_TO_INDEX in action_masks.py

**Files:**
- Modify: `src/esper/simic/action_masks.py`
- Test: `tests/simic/test_action_masks.py`

**Step 1: Write test for canonical IDs**

```python
def test_slot_id_to_index_canonical():
    """_SLOT_ID_TO_INDEX accepts canonical slot IDs."""
    from esper.simic.action_masks import slot_id_to_index

    # Canonical IDs work
    assert slot_id_to_index("r0c0") == 0
    assert slot_id_to_index("r0c1") == 1
    assert slot_id_to_index("r0c2") == 2

    # Legacy names raise error
    with pytest.raises(ValueError, match="no longer supported"):
        slot_id_to_index("early")
```

**Step 2: Update implementation**

```python
from esper.leyline.slot_id import parse_slot_id, SlotIdError

def slot_id_to_index(slot_id: str, slot_ids: tuple[str, ...] = ("r0c0", "r0c1", "r0c2")) -> int:
    """Convert canonical slot ID to action index.

    Args:
        slot_id: Canonical slot ID (e.g., "r0c0")
        slot_ids: Ordered tuple of valid slot IDs

    Returns:
        Index in slot_ids tuple

    Raises:
        ValueError: If slot_id not in slot_ids or uses legacy format
    """
    # Validate format (will raise for legacy names)
    try:
        parse_slot_id(slot_id)
    except SlotIdError as e:
        raise ValueError(str(e)) from e

    try:
        return slot_ids.index(slot_id)
    except ValueError:
        raise ValueError(f"Unknown slot_id: {slot_id}. Valid: {slot_ids}")
```

**Step 3: Commit**

```bash
git commit -m "refactor(simic): update action_masks to use canonical slot IDs"
```

---

### Task 1.2: Update CLI --slots argument

**Files:**
- Modify: `src/esper/scripts/train.py`

**Step 1: Update argument parser**

```python
parser.add_argument(
    "--slots",
    nargs="+",
    default=["r0c0", "r0c1", "r0c2"],
    help="Canonical slot IDs to use (e.g., r0c0 r0c1 r0c2)",
)
```

**Step 2: Add validation**

```python
from esper.leyline.slot_id import validate_slot_id, SlotIdError

def validate_slots(slot_ids: list[str]) -> list[str]:
    """Validate CLI slot arguments."""
    for slot_id in slot_ids:
        if not validate_slot_id(slot_id):
            raise ValueError(
                f"Invalid slot '{slot_id}'. Use canonical format: r0c0, r0c1, r0c2, ..."
            )
    return slot_ids
```

**Step 3: Commit**

```bash
git commit -m "feat(cli): update --slots to require canonical IDs"
```

---

## Milestone 1.5: RL Action Space Migration

**Priority:** MEDIUM — spike confirmed GREEN LIGHT
**Blockers identified:** action_masks.py, factored_actions.py

### Task 1.5.1: Create SlotConfig dataclass

**Files:**
- Create: `src/esper/leyline/slot_config.py`
- Test: `tests/leyline/test_slot_config.py`

**Step 1: Write test**

```python
# tests/leyline/test_slot_config.py
from esper.leyline.slot_config import SlotConfig


def test_slot_config_default():
    """Default SlotConfig has 3 slots."""
    config = SlotConfig.default()
    assert config.num_slots == 3
    assert config.slot_ids == ("r0c0", "r0c1", "r0c2")


def test_slot_config_custom():
    """SlotConfig accepts custom slot IDs."""
    config = SlotConfig(slot_ids=("r0c0", "r0c1"))
    assert config.num_slots == 2


def test_slot_id_for_index():
    """slot_id_for_index() returns correct ID."""
    config = SlotConfig.default()
    assert config.slot_id_for_index(0) == "r0c0"
    assert config.slot_id_for_index(1) == "r0c1"
    assert config.slot_id_for_index(2) == "r0c2"
```

**Step 2: Implement**

```python
# src/esper/leyline/slot_config.py
"""Slot configuration for dynamic action spaces."""

from __future__ import annotations
from dataclasses import dataclass

from esper.leyline.slot_id import format_slot_id, validate_slot_id


@dataclass(frozen=True)
class SlotConfig:
    """Configuration for slot action space.

    Replaces the fixed SlotAction enum with dynamic slot configuration.
    """
    slot_ids: tuple[str, ...]

    @property
    def num_slots(self) -> int:
        """Number of slots in this configuration."""
        return len(self.slot_ids)

    def slot_id_for_index(self, idx: int) -> str:
        """Get slot ID for action index."""
        return self.slot_ids[idx]

    def index_for_slot_id(self, slot_id: str) -> int:
        """Get action index for slot ID."""
        return self.slot_ids.index(slot_id)

    @classmethod
    def default(cls) -> SlotConfig:
        """Default 3-slot configuration (legacy compatible)."""
        return cls(slot_ids=("r0c0", "r0c1", "r0c2"))

    @classmethod
    def for_grid(cls, rows: int, cols: int) -> SlotConfig:
        """Create config for a full grid."""
        slot_ids = tuple(
            format_slot_id(r, c) for r in range(rows) for c in range(cols)
        )
        return cls(slot_ids=slot_ids)
```

**Step 3: Commit**

```bash
git commit -m "feat(leyline): add SlotConfig for dynamic action spaces"
```

---

### Task 1.5.2: Update action_masks.py to use SlotConfig

**Files:**
- Modify: `src/esper/simic/action_masks.py`

**Step 1: Update compute_action_masks()**

```python
def compute_action_masks(
    model_state: MorphogeneticModelState,
    slot_config: SlotConfig | None = None,
    device: torch.device | None = None,
) -> ActionMasks:
    """Compute action masks based on model state.

    Args:
        model_state: Current state of MorphogeneticModel
        slot_config: Slot configuration (defaults to SlotConfig.default())
        device: Target device for tensors
    """
    if slot_config is None:
        slot_config = SlotConfig.default()

    num_slots = slot_config.num_slots
    slot_mask = torch.zeros(num_slots, dtype=torch.bool, device=device)
    # ... rest of implementation
```

**Step 2: Update all callers**

Thread `slot_config` through PPOAgent, buffer, etc.

**Step 3: Commit**

```bash
git commit -m "refactor(simic): update action_masks to use SlotConfig"
```

---

### Task 1.5.3: Deprecate SlotAction enum

**Files:**
- Modify: `src/esper/leyline/factored_actions.py`

**Step 1: Add deprecation warning**

```python
import warnings

class SlotAction(IntEnum):
    """DEPRECATED: Use SlotConfig instead."""
    EARLY = 0
    MID = 1
    LATE = 2

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "SlotAction enum is deprecated. Use SlotConfig for dynamic slot counts.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)

# Keep NUM_SLOTS as legacy alias
NUM_SLOTS = 3  # DEPRECATED: Use slot_config.num_slots
```

**Step 2: Commit**

```bash
git commit -m "deprecate(leyline): mark SlotAction enum as deprecated"
```

---

## Milestone 5: Test Suite Migration

**Priority:** ONGOING — runs throughout

### Task 5.1: Update test fixtures

**Files:**
- Modify: `tests/conftest.py`
- Modify: `tests/strategies.py`

Replace all `"early"/"mid"/"late"` with canonical IDs.

### Task 5.2: Update integration tests

Update files listed in master plan section M5.2-M5.3.

---

## Milestone 6: Documentation

### Task 6.1: Update README CLI examples

### Task 6.2: Update docstrings

---

## Acceptance Checklist

- [ ] M4: Checkpoints load with `weights_only=True`
- [ ] M4: `alpha_schedule` discarded after BLENDING
- [ ] M2: `GatedBlend.get_alpha()` tracks progress
- [ ] M2: G3 gate passes naturally with gated blending
- [ ] M1: All slot IDs use canonical format
- [ ] M1.5: PPO works with N≠3 slots
- [ ] M5: All tests pass
- [ ] M6: Documentation updated

---

## Outputs

1. **Checkpoint compatibility** — `weights_only=True` works for all stages
2. **Fixed gated blending** — G3 gate passes naturally
3. **Canonical slot IDs** — `r0c0` format everywhere
4. **Dynamic action space** — PPO supports N slots
5. **Clean codebase** — no legacy names, no backwards compat

---

## Estimated Task Count

| Milestone | Tasks | Complexity |
|-----------|-------|------------|
| M4 (Checkpoints) | 6 | Medium |
| M2 (Blending) | 2 | Medium |
| M1 (Coordinates) | 2 | Low |
| M1.5 (RL) | 3 | High |
| M5 (Tests) | ~10 | Medium |
| M6 (Docs) | 2 | Low |
| **Total** | ~25 | — |
