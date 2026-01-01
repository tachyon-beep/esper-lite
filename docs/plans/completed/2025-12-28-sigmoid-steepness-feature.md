# Sigmoid Steepness Feature Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable Tamiyo to control sigmoid curve steepness (transition sharpness) via the existing alpha_curve action head.

**Architecture:** Expand AlphaCurveAction enum to include steepness variants (SIGMOID_GENTLE, SIGMOID, SIGMOID_SHARP), add alpha_steepness field to AlphaController, and wire the steepness value through retarget() calls.

**Tech Stack:** Python dataclasses, PyTorch (for RL integration), Hypothesis (property-based testing)

---

## Pre-Implementation: leyline/factored_actions.py (Already Complete)

The AlphaCurveAction enum has already been expanded:
- `SIGMOID_GENTLE` (steepness=6): Gradual S-curve
- `SIGMOID` (steepness=12): Standard S-curve (default)
- `SIGMOID_SHARP` (steepness=24): Steep S-curve

Helper methods `to_curve()` and `to_steepness()` are already implemented.

---

### Task 1: Add alpha_steepness to AlphaController

**Files:**
- Modify: `src/esper/kasmina/alpha_controller.py:21-41` (_curve_progress function)
- Modify: `src/esper/kasmina/alpha_controller.py:44-54` (AlphaController dataclass)
- Modify: `src/esper/kasmina/alpha_controller.py:64-100` (retarget method)
- Modify: `src/esper/kasmina/alpha_controller.py:119-120` (step method)
- Modify: `src/esper/kasmina/alpha_controller.py:133-160` (serialization)
- Test: `tests/kasmina/test_alpha_steepness.py`

**Step 1: Write the failing test for steepness parameterization**

Create `tests/kasmina/test_alpha_steepness.py`:

```python
"""Tests for sigmoid steepness parameterization in AlphaController."""

import math
import pytest

from esper.kasmina.alpha_controller import AlphaController, _curve_progress
from esper.leyline.alpha import AlphaCurve


class TestCurveProgressSteepness:
    """Test _curve_progress with different steepness values."""

    def test_sigmoid_default_steepness_is_12(self):
        """Default steepness=12 should produce the original behavior."""
        result = _curve_progress(0.5, AlphaCurve.SIGMOID, steepness=12.0)
        assert abs(result - 0.5) < 1e-6

    def test_sigmoid_gentle_is_less_steep(self):
        """Gentle steepness=6 should have less curvature at t=0.25."""
        gentle = _curve_progress(0.25, AlphaCurve.SIGMOID, steepness=6.0)
        standard = _curve_progress(0.25, AlphaCurve.SIGMOID, steepness=12.0)
        # Gentle curve should be closer to linear (0.25) at this point
        assert gentle > standard

    def test_sigmoid_sharp_is_more_steep(self):
        """Sharp steepness=24 should have more curvature at t=0.25."""
        sharp = _curve_progress(0.25, AlphaCurve.SIGMOID, steepness=24.0)
        standard = _curve_progress(0.25, AlphaCurve.SIGMOID, steepness=12.0)
        # Sharp curve should be closer to 0 at t=0.25
        assert sharp < standard

    def test_linear_ignores_steepness(self):
        """LINEAR curve should ignore steepness parameter."""
        result = _curve_progress(0.5, AlphaCurve.LINEAR, steepness=100.0)
        assert result == 0.5

    def test_cosine_ignores_steepness(self):
        """COSINE curve should ignore steepness parameter."""
        result = _curve_progress(0.5, AlphaCurve.COSINE, steepness=100.0)
        expected = 0.5 * (1.0 - math.cos(math.pi * 0.5))
        assert abs(result - expected) < 1e-6


class TestAlphaControllerSteepness:
    """Test AlphaController with steepness field."""

    def test_default_steepness_is_12(self):
        """AlphaController should default to steepness=12."""
        controller = AlphaController()
        assert controller.alpha_steepness == 12.0

    def test_retarget_accepts_steepness(self):
        """retarget() should accept and store steepness."""
        controller = AlphaController()
        controller.retarget(
            alpha_target=1.0,
            alpha_steps_total=5,
            alpha_curve=AlphaCurve.SIGMOID,
            alpha_steepness=6.0,
        )
        assert controller.alpha_steepness == 6.0

    def test_steepness_affects_transition(self):
        """Different steepness should produce different alpha progression."""
        # Gentle curve
        gentle = AlphaController()
        gentle.retarget(alpha_target=1.0, alpha_steps_total=4, alpha_curve=AlphaCurve.SIGMOID, alpha_steepness=6.0)
        gentle.step()
        gentle.step()
        alpha_gentle = gentle.alpha

        # Sharp curve
        sharp = AlphaController()
        sharp.retarget(alpha_target=1.0, alpha_steps_total=4, alpha_curve=AlphaCurve.SIGMOID, alpha_steepness=24.0)
        sharp.step()
        sharp.step()
        alpha_sharp = sharp.alpha

        # At midpoint, gentle should be closer to 0.5, sharp further
        assert alpha_gentle > alpha_sharp


class TestAlphaControllerSteepnessSerialization:
    """Test checkpoint round-trip with steepness."""

    def test_to_dict_includes_steepness(self):
        """to_dict() should include alpha_steepness."""
        controller = AlphaController(alpha_steepness=6.0)
        data = controller.to_dict()
        assert "alpha_steepness" in data
        assert data["alpha_steepness"] == 6.0

    def test_from_dict_restores_steepness(self):
        """from_dict() should restore alpha_steepness."""
        original = AlphaController(alpha_steepness=24.0)
        data = original.to_dict()
        restored = AlphaController.from_dict(data)
        assert restored.alpha_steepness == 24.0

    def test_from_dict_defaults_steepness_for_old_checkpoints(self):
        """from_dict() should default steepness=12 for old checkpoints."""
        old_data = {
            "alpha": 0.5,
            "alpha_start": 0.0,
            "alpha_target": 1.0,
            "alpha_mode": 1,
            "alpha_curve": 3,
            "alpha_steps_total": 5,
            "alpha_steps_done": 2,
            # No alpha_steepness - old checkpoint
        }
        restored = AlphaController.from_dict(old_data)
        assert restored.alpha_steepness == 12.0  # Default
```

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src uv run pytest tests/kasmina/test_alpha_steepness.py -v`
Expected: FAIL with "unexpected keyword argument 'steepness'"

**Step 3: Update _curve_progress to accept steepness parameter**

In `src/esper/kasmina/alpha_controller.py`, change:

```python
def _curve_progress(t: float, curve: AlphaCurve, steepness: float = 12.0) -> float:
    """Apply easing curve to linear progress t.

    Args:
        t: Linear progress in [0, 1].
        curve: Which easing curve to apply.
        steepness: Sigmoid steepness (only affects SIGMOID curve).
            Higher values = sharper transition. Default 12.0.

    Returns:
        Eased progress in [0, 1].
    """
    t = max(0.0, min(1.0, t))
    match curve:
        case AlphaCurve.LINEAR:
            return t
        case AlphaCurve.COSINE:
            # Smooth start/end: 0 -> 1 with zero slope at endpoints.
            return 0.5 * (1.0 - math.cos(math.pi * t))
        case AlphaCurve.SIGMOID:
            # Logistic curve normalized to [0, 1] at t in [0, 1].
            raw = 1.0 / (1.0 + math.exp(-steepness * (t - 0.5)))
            raw0 = 1.0 / (1.0 + math.exp(-steepness * (0.0 - 0.5)))
            raw1 = 1.0 / (1.0 + math.exp(-steepness * (1.0 - 0.5)))
            if raw1 == raw0:
                # Guard against division by zero if steepness -> 0
                return t
            return (raw - raw0) / (raw1 - raw0)
        case _:
            raise ValueError(f"Unknown AlphaCurve: {curve!r}")
```

**Step 4: Add alpha_steepness field to AlphaController dataclass**

```python
@dataclass(slots=True)
class AlphaController:
    """Schedule alpha from start -> target over N controller ticks."""

    alpha: float = 0.0
    alpha_start: float = 0.0
    alpha_target: float = 0.0
    alpha_mode: AlphaMode = AlphaMode.HOLD
    alpha_curve: AlphaCurve = AlphaCurve.LINEAR
    alpha_steepness: float = 12.0  # Sigmoid steepness (default matches original)
    alpha_steps_total: int = 0
    alpha_steps_done: int = 0

    def __post_init__(self) -> None:
        self.alpha = _clamp01(self.alpha)
        self.alpha_start = _clamp01(self.alpha_start)
        self.alpha_target = _clamp01(self.alpha_target)
        self.alpha_steepness = max(0.1, float(self.alpha_steepness))  # Prevent div-by-zero
        self.alpha_steps_total = max(0, int(self.alpha_steps_total))
        self.alpha_steps_done = max(0, int(self.alpha_steps_done))
        self.alpha_steps_done = min(self.alpha_steps_done, self.alpha_steps_total)
```

**Step 5: Update retarget() to accept steepness**

```python
def retarget(
    self,
    *,
    alpha_target: float,
    alpha_steps_total: int,
    alpha_curve: AlphaCurve | None = None,
    alpha_steepness: float | None = None,
) -> None:
    """Set a new target and schedule from the current alpha.

    Args:
        alpha_target: Target alpha value in [0, 1].
        alpha_steps_total: Number of controller ticks to reach target.
        alpha_curve: Easing curve (None to keep current).
        alpha_steepness: Sigmoid steepness (None to keep current).

    Contract: retargeting is only allowed from HOLD to prevent alpha dithering
    during a transition.
    """
    if self.alpha_mode != AlphaMode.HOLD:
        raise ValueError("AlphaController.retarget() is only allowed from HOLD")

    target = _clamp01(alpha_target)
    steps_total = max(0, int(alpha_steps_total))

    self.alpha_start = self.alpha
    self.alpha_target = target
    if alpha_curve is not None:
        self.alpha_curve = alpha_curve
    if alpha_steepness is not None:
        self.alpha_steepness = max(0.1, float(alpha_steepness))

    self.alpha_steps_total = steps_total
    self.alpha_steps_done = 0

    if target > self.alpha:
        self.alpha_mode = AlphaMode.UP
    elif target < self.alpha:
        self.alpha_mode = AlphaMode.DOWN
    else:
        self.alpha_mode = AlphaMode.HOLD
        self.alpha = target

    if steps_total == 0:
        self.alpha = target
        self.alpha_mode = AlphaMode.HOLD
```

**Step 6: Update step() to pass steepness to _curve_progress**

```python
t = self.alpha_steps_done / max(self.alpha_steps_total, 1)
progress = _curve_progress(t, self.alpha_curve, self.alpha_steepness)
```

**Step 7: Update serialization**

```python
def to_dict(self) -> dict[str, int | float]:
    """Primitive serialization for checkpointing."""
    return {
        "alpha": self.alpha,
        "alpha_start": self.alpha_start,
        "alpha_target": self.alpha_target,
        "alpha_mode": int(self.alpha_mode),
        "alpha_curve": int(self.alpha_curve),
        "alpha_steepness": self.alpha_steepness,
        "alpha_steps_total": self.alpha_steps_total,
        "alpha_steps_done": self.alpha_steps_done,
    }

@classmethod
def from_dict(cls, data: dict[str, Any]) -> "AlphaController":
    """Deserialize from checkpoint dict.

    Raises:
        KeyError: If required fields are missing (corrupt checkpoint).
    """
    return cls(
        alpha=float(data["alpha"]),
        alpha_start=float(data["alpha_start"]),
        alpha_target=float(data["alpha_target"]),
        alpha_mode=AlphaMode(int(data["alpha_mode"])),
        alpha_curve=AlphaCurve(int(data["alpha_curve"])),
        alpha_steepness=float(data.get("alpha_steepness", 12.0)),  # Default for old checkpoints
        alpha_steps_total=int(data["alpha_steps_total"]),
        alpha_steps_done=int(data["alpha_steps_done"]),
    )
```

**Step 8: Run tests to verify they pass**

Run: `PYTHONPATH=src uv run pytest tests/kasmina/test_alpha_steepness.py -v`
Expected: All tests PASS

**Step 9: Run existing alpha controller tests**

Run: `PYTHONPATH=src uv run pytest tests/kasmina/properties/test_alpha_controller_properties.py -v`
Expected: All tests PASS (backward compatible)

**Step 10: Commit**

```bash
git add src/esper/kasmina/alpha_controller.py tests/kasmina/test_alpha_steepness.py
git commit -m "feat(kasmina): add alpha_steepness to AlphaController

- Add alpha_steepness field with default 12.0 (backward compatible)
- Update _curve_progress() to accept steepness parameter
- Update retarget() to accept optional alpha_steepness
- Update serialization with backward-compatible deserialization
- Add comprehensive tests for steepness parameterization

Resolves B2-DRL-07 by making the division-by-zero guard legitimate."
```

---

### Task 2: Wire steepness through SeedSlot

**Files:**
- Modify: `src/esper/kasmina/slot.py` (3 retarget call sites)
- Test: `tests/kasmina/test_alpha_steepness.py` (extend)

**Step 1: Find all retarget() call sites in slot.py**

There are 3 call sites:
- Line ~1657: `schedule_prune()`
- Line ~1760: `set_alpha_target()`
- Line ~2112: `start_blending()`

**Step 2: Update schedule_prune() to accept steepness**

```python
def schedule_prune(
    self,
    steps: int,
    *,
    curve: AlphaCurve = AlphaCurve.LINEAR,
    steepness: float = 12.0,
    reason: str | None = None,
    initiator: str = "policy",
) -> None:
    # ...
    self.state.alpha_controller.retarget(
        alpha_target=0.0,
        alpha_steps_total=steps,
        alpha_curve=curve,
        alpha_steepness=steepness,
    )
```

**Step 3: Update set_alpha_target() to accept steepness**

```python
def set_alpha_target(
    self,
    target: float,
    steps: int,
    *,
    curve: AlphaCurve = AlphaCurve.LINEAR,
    steepness: float = 12.0,
    alpha_algorithm: AlphaAlgorithm | None = None,
) -> None:
    # ...
    controller.retarget(
        alpha_target=alpha_target,
        alpha_steps_total=steps,
        alpha_curve=curve,
        alpha_steepness=steepness,
    )
```

**Step 4: Update start_blending() to accept steepness**

```python
def start_blending(
    self,
    alpha_target: float | None = None,
    *,
    curve: AlphaCurve = AlphaCurve.COSINE,
    steepness: float = 12.0,
) -> None:
    # ...
    self.state.alpha_controller.retarget(
        alpha_target=alpha_target,
        alpha_steps_total=total_steps,
        alpha_curve=curve,
        alpha_steepness=steepness,
    )
```

**Step 5: Run existing slot tests**

Run: `PYTHONPATH=src uv run pytest tests/kasmina/test_seed_slot.py -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/esper/kasmina/slot.py
git commit -m "feat(kasmina): wire alpha_steepness through SeedSlot

- Add steepness parameter to schedule_prune(), set_alpha_target(), start_blending()
- All default to 12.0 for backward compatibility"
```

---

### Task 3: Wire steepness through vectorized training

**Files:**
- Modify: `src/esper/simic/training/vectorized.py` (2 call sites)

**Step 1: Find both action dispatch sites**

There are 2 sites that need steepness extraction:
- Line ~2794: PRUNE operation - calls `schedule_prune()`
- Line ~2824: SET_ALPHA_TARGET operation - calls `set_alpha_target()`

**Step 2: Update PRUNE action dispatch (~line 2794)**

```python
curve_action = AlphaCurveAction(action_dict["alpha_curve"])
curve = curve_action.to_curve()
steepness = curve_action.to_steepness()
# ... pass steepness to schedule_prune()
```

**Step 3: Update SET_ALPHA_TARGET action dispatch (~line 2824)**

```python
curve_action = AlphaCurveAction(action_dict["alpha_curve"])
curve = curve_action.to_curve()
steepness = curve_action.to_steepness()
# ... pass steepness to set_alpha_target()
```

**Step 4: Run vectorized training tests**

Run: `PYTHONPATH=src uv run pytest tests/simic/training/ -v -k "not slow"`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/esper/simic/training/vectorized.py
git commit -m "feat(simic): extract alpha_steepness from AlphaCurveAction in vectorized training

- Wire steepness through PRUNE action dispatch
- Wire steepness through SET_ALPHA_TARGET action dispatch"
```

---

### Task 4: Close B2-DRL-07 ticket

**Files:**
- Modify: `docs/bugs/batch2/B2-DRL-07.md`
- Move to: `docs/bugs/fixed/B2-DRL-07.md`

**Step 1: Update ticket resolution**

Add Resolution section documenting that steepness is now configurable, making the guard legitimate.

**Step 2: Move ticket to fixed/**

```bash
git mv docs/bugs/batch2/B2-DRL-07.md docs/bugs/fixed/
git commit -m "docs: close B2-DRL-07 - sigmoid steepness now configurable"
```

---

## Summary

| Task | Files | Tests |
|------|-------|-------|
| 1. AlphaController | alpha_controller.py | test_alpha_steepness.py |
| 2. SeedSlot wiring | slot.py | existing slot tests |
| 3. Vectorized training | vectorized.py | existing simic tests |
| 4. Close ticket | B2-DRL-07.md | - |

**Total estimated time:** 30-45 minutes

**Backward compatibility:** All changes default to steepness=12.0, preserving existing behavior.
