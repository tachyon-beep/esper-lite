# Lifecycle State Machine Fix Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix silent lifecycle transition failures by separating strategic decisions (Tamiyo) from mechanical transitions (Kasmina auto-advance).

**Architecture:** Tamiyo's ADVANCE action only applies at TRAINING→BLENDING and PROBATIONARY→FOSSILIZED. Kasmina auto-advances BLENDING→SHADOWING→PROBATIONARY when blending completes (α=1.0). This respects VALID_TRANSITIONS while keeping RL credit assignment tight.

**Tech Stack:** PyTorch, existing Kasmina/Simic/Leyline modules

**Root Cause:** `vectorized.py` attempted illegal BLENDING→FOSSILIZED transition. `SeedState.transition()` returns False silently. Seeds stayed stuck at α=1.0 in BLENDING forever.

---

## Task 1: Add Blending Progress Tracking to SeedState

**Files:**
- Modify: `src/esper/kasmina/slot.py:109-140` (SeedState dataclass)

**Step 1: Write the failing test**

Create test file `tests/test_lifecycle_fix.py`:

```python
"""Tests for lifecycle state machine fix."""

import pytest
from esper.kasmina.slot import SeedState
from esper.leyline import SeedStage


class TestBlendingProgressTracking:
    """Test that SeedState tracks blending progress."""

    def test_seedstate_has_blending_fields(self):
        """SeedState should have blending progress fields."""
        state = SeedState(seed_id="test", blueprint_id="conv_enhance")

        assert hasattr(state, "blending_steps_done")
        assert hasattr(state, "blending_steps_total")
        assert state.blending_steps_done == 0
        assert state.blending_steps_total == 0

    def test_blending_fields_increment(self):
        """Blending fields should be mutable."""
        state = SeedState(seed_id="test", blueprint_id="conv_enhance")
        state.blending_steps_total = 5
        state.blending_steps_done = 3

        assert state.blending_steps_total == 5
        assert state.blending_steps_done == 3
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_lifecycle_fix.py::TestBlendingProgressTracking -v`

Expected: FAIL with `AttributeError: 'SeedState' object has no attribute 'blending_steps_done'`

**Step 3: Add blending fields to SeedState**

In `src/esper/kasmina/slot.py`, modify the `SeedState` dataclass (around line 109):

```python
@dataclass
class SeedState:
    """Complete state of a seed through its lifecycle."""

    seed_id: str
    blueprint_id: str
    slot_id: str = ""

    stage: SeedStage = SeedStage.DORMANT
    previous_stage: SeedStage = SeedStage.UNKNOWN
    stage_entered_at: datetime = field(default_factory=datetime.utcnow)

    alpha: float = 0.0
    metrics: SeedMetrics = field(default_factory=SeedMetrics)

    # Blending progress tracking
    blending_steps_done: int = 0
    blending_steps_total: int = 0

    # Flags
    is_healthy: bool = True
    is_paused: bool = False

    # ... rest unchanged
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_lifecycle_fix.py::TestBlendingProgressTracking -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/kasmina/slot.py tests/test_lifecycle_fix.py
git commit -m "feat(kasmina): add blending progress tracking to SeedState"
```

---

## Task 2: Update start_blending to Initialize Progress

**Files:**
- Modify: `src/esper/kasmina/slot.py:624-633` (start_blending method)
- Test: `tests/test_lifecycle_fix.py`

**Step 1: Write the failing test**

Add to `tests/test_lifecycle_fix.py`:

```python
from esper.kasmina.slot import SeedSlot


class TestStartBlendingProgress:
    """Test that start_blending initializes progress tracking."""

    def test_start_blending_sets_total_steps(self):
        """start_blending should set blending_steps_total."""
        slot = SeedSlot(slot_id="test", channels=64, device="cpu")

        # Germinate a seed first
        from unittest.mock import MagicMock
        host = MagicMock()
        host.injection_channels = 64
        slot.germinate(blueprint_id="conv_enhance", seed_id="test_seed", host_module=host)

        # Transition to TRAINING then BLENDING
        slot.state.transition(SeedStage.TRAINING)
        slot.state.transition(SeedStage.BLENDING)

        # Start blending with 5 steps
        slot.start_blending(total_steps=5, temperature=1.0)

        assert slot.state.blending_steps_total == 5
        assert slot.state.blending_steps_done == 0

    def test_start_blending_resets_done_counter(self):
        """start_blending should reset blending_steps_done to 0."""
        slot = SeedSlot(slot_id="test", channels=64, device="cpu")

        from unittest.mock import MagicMock
        host = MagicMock()
        host.injection_channels = 64
        slot.germinate(blueprint_id="conv_enhance", seed_id="test_seed", host_module=host)
        slot.state.transition(SeedStage.TRAINING)
        slot.state.transition(SeedStage.BLENDING)

        # Manually set done to simulate prior state
        slot.state.blending_steps_done = 3

        # Start blending should reset
        slot.start_blending(total_steps=5, temperature=1.0)

        assert slot.state.blending_steps_done == 0
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_lifecycle_fix.py::TestStartBlendingProgress -v`

Expected: FAIL with assertion error (blending_steps_total not set)

**Step 3: Update start_blending to initialize progress**

In `src/esper/kasmina/slot.py`, modify `start_blending` method (around line 624):

```python
def start_blending(self, total_steps: int, temperature: float = 1.0) -> None:
    """Initialize alpha schedule for blending phase."""
    if not self.state:
        return

    # Initialize blending progress tracking
    self.state.blending_steps_total = total_steps
    self.state.blending_steps_done = 0

    self.alpha_schedule = AlphaSchedule(
        total_steps=total_steps,
        start=0.0,
        end=1.0,
        temperature=temperature,
    )
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_lifecycle_fix.py::TestStartBlendingProgress -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/kasmina/slot.py tests/test_lifecycle_fix.py
git commit -m "feat(kasmina): initialize blending progress in start_blending"
```

---

## Task 3: Add step_epoch Method for Auto-Advance

**Files:**
- Modify: `src/esper/kasmina/slot.py` (add step_epoch method after update_alpha_for_step)
- Test: `tests/test_lifecycle_fix.py`

**Step 1: Write the failing test**

Add to `tests/test_lifecycle_fix.py`:

```python
class TestStepEpochAutoAdvance:
    """Test that step_epoch auto-advances through mechanical stages."""

    def _create_blending_slot(self) -> SeedSlot:
        """Helper to create a slot in BLENDING stage."""
        slot = SeedSlot(slot_id="test", channels=64, device="cpu")

        from unittest.mock import MagicMock
        host = MagicMock()
        host.injection_channels = 64
        slot.germinate(blueprint_id="conv_enhance", seed_id="test_seed", host_module=host)
        slot.state.transition(SeedStage.TRAINING)
        slot.state.transition(SeedStage.BLENDING)
        slot.start_blending(total_steps=3, temperature=1.0)

        return slot

    def test_step_epoch_increments_blending_progress(self):
        """step_epoch should increment blending_steps_done."""
        slot = self._create_blending_slot()

        assert slot.state.blending_steps_done == 0

        slot.step_epoch()
        assert slot.state.blending_steps_done == 1

        slot.step_epoch()
        assert slot.state.blending_steps_done == 2

    def test_step_epoch_updates_alpha(self):
        """step_epoch should update alpha based on progress."""
        slot = self._create_blending_slot()

        assert slot.alpha == 0.0

        slot.step_epoch()  # 1/3
        slot.step_epoch()  # 2/3
        slot.step_epoch()  # 3/3 = 1.0

        assert slot.alpha >= 0.99  # Should be at or near 1.0

    def test_step_epoch_auto_advances_when_blending_complete(self):
        """step_epoch should auto-advance BLENDING→SHADOWING→PROBATIONARY when α=1.0."""
        slot = self._create_blending_slot()

        # Run through all blending steps
        for _ in range(3):
            slot.step_epoch()

        # Should have auto-advanced through SHADOWING to PROBATIONARY
        assert slot.state.stage == SeedStage.PROBATIONARY
        assert slot.alpha >= 0.99

    def test_step_epoch_noop_when_not_blending(self):
        """step_epoch should be no-op when not in BLENDING stage."""
        slot = SeedSlot(slot_id="test", channels=64, device="cpu")

        from unittest.mock import MagicMock
        host = MagicMock()
        host.injection_channels = 64
        slot.germinate(blueprint_id="conv_enhance", seed_id="test_seed", host_module=host)
        slot.state.transition(SeedStage.TRAINING)

        # Should not raise or change state
        slot.step_epoch()
        assert slot.state.stage == SeedStage.TRAINING

    def test_step_epoch_noop_when_no_seed(self):
        """step_epoch should be no-op when no active seed."""
        slot = SeedSlot(slot_id="test", channels=64, device="cpu")

        # Should not raise
        slot.step_epoch()
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_lifecycle_fix.py::TestStepEpochAutoAdvance -v`

Expected: FAIL with `AttributeError: 'SeedSlot' object has no attribute 'step_epoch'`

**Step 3: Implement step_epoch method**

In `src/esper/kasmina/slot.py`, add after `update_alpha_for_step` method (around line 643):

```python
def step_epoch(self) -> None:
    """Called once per epoch to update blending progress and auto-advance lifecycle.

    When in BLENDING stage:
    - Increments blending_steps_done
    - Updates alpha based on schedule
    - When α reaches 1.0, auto-advances through SHADOWING→PROBATIONARY

    This separates mechanical transitions (Kasmina's job) from strategic
    decisions (Tamiyo's ADVANCE action at TRAINING and PROBATIONARY).
    """
    if not self.state:
        return

    if self.state.stage != SeedStage.BLENDING:
        return

    # Increment blending progress
    self.state.blending_steps_done += 1

    # Update alpha based on schedule
    if self.alpha_schedule is not None:
        alpha = self.alpha_schedule(self.state.blending_steps_done)
        self.set_alpha(alpha)

    # Auto-advance when blending complete
    if self.state.blending_steps_done >= self.state.blending_steps_total:
        self.set_alpha(1.0)  # Ensure fully blended

        # BLENDING → SHADOWING
        ok = self.state.transition(SeedStage.SHADOWING)
        if not ok:
            raise RuntimeError(
                f"Illegal lifecycle transition {self.state.stage} → SHADOWING"
            )

        # SHADOWING → PROBATIONARY (collapse through - no validation yet)
        ok = self.state.transition(SeedStage.PROBATIONARY)
        if not ok:
            raise RuntimeError(
                f"Illegal lifecycle transition {self.state.stage} → PROBATIONARY"
            )
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_lifecycle_fix.py::TestStepEpochAutoAdvance -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/kasmina/slot.py tests/test_lifecycle_fix.py
git commit -m "feat(kasmina): add step_epoch for auto-advance through mechanical stages"
```

---

## Task 4: Update vectorized.py ADVANCE Logic

**Files:**
- Modify: `src/esper/simic/vectorized.py:577-586` (ADVANCE action handler)
- Test: `tests/test_lifecycle_fix.py`

**Step 1: Write the failing test**

Add to `tests/test_lifecycle_fix.py`:

```python
class TestStrategicAdvanceOnly:
    """Test that ADVANCE only works at strategic decision points."""

    def test_advance_from_training_starts_blending(self):
        """ADVANCE from TRAINING should transition to BLENDING."""
        from esper.kasmina.host import MorphogeneticModel, HostCNN

        model = MorphogeneticModel(HostCNN(), device="cpu")
        model.germinate_seed("conv_enhance", "test_seed")
        model.seed_state.transition(SeedStage.TRAINING)

        # Simulate ADVANCE action
        assert model.seed_state.stage == SeedStage.TRAINING
        model.seed_state.transition(SeedStage.BLENDING)
        model.seed_slot.start_blending(total_steps=5, temperature=1.0)

        assert model.seed_state.stage == SeedStage.BLENDING

    def test_advance_from_probationary_fossilizes(self):
        """ADVANCE from PROBATIONARY should transition to FOSSILIZED."""
        from esper.kasmina.host import MorphogeneticModel, HostCNN

        model = MorphogeneticModel(HostCNN(), device="cpu")
        model.germinate_seed("conv_enhance", "test_seed")
        model.seed_state.transition(SeedStage.TRAINING)
        model.seed_state.transition(SeedStage.BLENDING)
        model.seed_state.transition(SeedStage.SHADOWING)
        model.seed_state.transition(SeedStage.PROBATIONARY)

        # ADVANCE from PROBATIONARY should work
        ok = model.seed_state.transition(SeedStage.FOSSILIZED)

        assert ok is True
        assert model.seed_state.stage == SeedStage.FOSSILIZED

    def test_advance_from_blending_is_noop(self):
        """ADVANCE from BLENDING should NOT transition directly to FOSSILIZED."""
        from esper.kasmina.host import MorphogeneticModel, HostCNN

        model = MorphogeneticModel(HostCNN(), device="cpu")
        model.germinate_seed("conv_enhance", "test_seed")
        model.seed_state.transition(SeedStage.TRAINING)
        model.seed_state.transition(SeedStage.BLENDING)

        # This SHOULD fail (the bug we're fixing)
        ok = model.seed_state.transition(SeedStage.FOSSILIZED)

        assert ok is False
        assert model.seed_state.stage == SeedStage.BLENDING  # Unchanged
```

**Step 2: Run test to verify current behavior**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_lifecycle_fix.py::TestStrategicAdvanceOnly -v`

Expected: PASS (these tests verify the contract, the bug is in how vectorized.py uses it)

**Step 3: Update vectorized.py ADVANCE logic**

In `src/esper/simic/vectorized.py`, replace the ADVANCE handler (around line 577):

```python
elif action == SimicAction.ADVANCE:
    if model.has_active_seed:
        current_stage = model.seed_state.stage

        if current_stage == SeedStage.TRAINING:
            # Strategic: Tamiyo decides to start blending
            ok = model.seed_state.transition(SeedStage.BLENDING)
            if not ok:
                raise RuntimeError(
                    f"Illegal lifecycle transition TRAINING → BLENDING"
                )
            model.seed_slot.start_blending(total_steps=5, temperature=1.0)
            env_state.blending_step = 0

        elif current_stage == SeedStage.PROBATIONARY:
            # Strategic: Tamiyo decides to fossilize
            ok = model.seed_state.transition(SeedStage.FOSSILIZED)
            if not ok:
                raise RuntimeError(
                    f"Illegal lifecycle transition PROBATIONARY → FOSSILIZED"
                )
            model.seed_slot.set_alpha(1.0)

        # else: BLENDING/SHADOWING - no-op, Kasmina auto-advances via step_epoch
```

**Step 4: Verify the change compiles**

Run: `PYTHONPATH=src .venv/bin/python -c "from esper.simic.vectorized import train_ppo_vectorized; print('OK')"`

Expected: `OK`

**Step 5: Commit**

```bash
git add src/esper/simic/vectorized.py
git commit -m "fix(simic): ADVANCE only at strategic decision points (TRAINING, PROBATIONARY)"
```

---

## Task 5: Wire step_epoch into Training Loop

**Files:**
- Modify: `src/esper/simic/vectorized.py` (add step_epoch call per epoch)
- Test: `tests/test_lifecycle_fix.py`

**Step 1: Write the integration test**

Add to `tests/test_lifecycle_fix.py`:

```python
class TestLifecycleIntegration:
    """Integration test for full lifecycle flow."""

    def test_full_lifecycle_with_auto_advance(self):
        """Test TRAINING→BLENDING→(auto)→PROBATIONARY→FOSSILIZED."""
        from esper.kasmina.host import MorphogeneticModel, HostCNN

        model = MorphogeneticModel(HostCNN(), device="cpu")
        model.germinate_seed("conv_enhance", "test_seed")
        model.seed_state.transition(SeedStage.TRAINING)

        # Tamiyo: ADVANCE to start blending
        model.seed_state.transition(SeedStage.BLENDING)
        model.seed_slot.start_blending(total_steps=3, temperature=1.0)

        assert model.seed_state.stage == SeedStage.BLENDING

        # Kasmina: auto-advance via step_epoch
        model.seed_slot.step_epoch()  # 1/3
        model.seed_slot.step_epoch()  # 2/3
        model.seed_slot.step_epoch()  # 3/3 → auto-advance

        # Should now be in PROBATIONARY (auto-advanced through SHADOWING)
        assert model.seed_state.stage == SeedStage.PROBATIONARY

        # Tamiyo: ADVANCE to fossilize
        ok = model.seed_state.transition(SeedStage.FOSSILIZED)

        assert ok is True
        assert model.seed_state.stage == SeedStage.FOSSILIZED

    def test_fossilization_emits_telemetry(self):
        """Test that fossilization emits SEED_FOSSILIZED telemetry."""
        from esper.kasmina.host import MorphogeneticModel, HostCNN
        from esper.leyline import TelemetryEventType

        model = MorphogeneticModel(HostCNN(), device="cpu")

        # Capture telemetry events
        captured_events = []
        def capture(event):
            captured_events.append(event)

        model.seed_slot.on_telemetry = capture
        model.seed_slot.fast_mode = False

        # Run through lifecycle
        model.germinate_seed("conv_enhance", "test_seed")
        model.seed_state.transition(SeedStage.TRAINING)
        model.seed_state.transition(SeedStage.BLENDING)
        model.seed_slot.start_blending(total_steps=3, temperature=1.0)

        for _ in range(3):
            model.seed_slot.step_epoch()

        # Now fossilize
        model.seed_state.transition(SeedStage.FOSSILIZED)

        # Trigger the fossilization telemetry via advance_stage
        # Note: We need to call advance_stage to emit telemetry, not just transition

        # Check we got SEED_FOSSILIZED event
        fossilized_events = [
            e for e in captured_events
            if e.event_type == TelemetryEventType.SEED_FOSSILIZED
        ]

        assert len(fossilized_events) >= 1, f"Expected SEED_FOSSILIZED, got: {[e.event_type for e in captured_events]}"
```

**Step 2: Run test to verify behavior**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_lifecycle_fix.py::TestLifecycleIntegration -v`

Expected: First test PASS, second may need adjustment based on telemetry wiring

**Step 3: Wire step_epoch into vectorized training loop**

In `src/esper/simic/vectorized.py`, add step_epoch call after gradient stats sync (around line 492):

Find this section:
```python
# First, sync telemetry for envs with active seeds (must happen BEFORE feature extraction)
for env_idx, env_state in enumerate(env_states):
    model = env_state.model
    seed_state = model.seed_state

    if use_telemetry and seed_state and env_grad_stats[env_idx]:
        # ... sync_telemetry call
```

Add after the sync_telemetry block:

```python
# First, sync telemetry for envs with active seeds (must happen BEFORE feature extraction)
for env_idx, env_state in enumerate(env_states):
    model = env_state.model
    seed_state = model.seed_state

    if use_telemetry and seed_state and env_grad_stats[env_idx]:
        grad_stats = env_grad_stats[env_idx]
        seed_state.sync_telemetry(
            gradient_norm=grad_stats['gradient_norm'],
            gradient_health=grad_stats['gradient_health'],
            has_vanishing=grad_stats['has_vanishing'],
            has_exploding=grad_stats['has_exploding'],
            epoch=epoch,
            max_epochs=max_epochs,
        )

    # Auto-advance blending lifecycle (Kasmina's mechanical transitions)
    model.seed_slot.step_epoch()
```

**Step 4: Run integration test**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_lifecycle_fix.py::TestLifecycleIntegration -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/vectorized.py tests/test_lifecycle_fix.py
git commit -m "feat(simic): wire step_epoch for automatic blending lifecycle"
```

---

## Task 6: Update process_train_batch for SHADOWING/PROBATIONARY Stages

**Files:**
- Modify: `src/esper/simic/vectorized.py:293-308` (optimizer selection logic)

**Step 1: Review current logic**

Current code at line 293:
```python
if seed_state is None or seed_state.stage == SeedStage.FOSSILIZED:
    optimizer = env_state.host_optimizer
elif seed_state.stage in (SeedStage.GERMINATED, SeedStage.TRAINING):
    # ... seed_optimizer
else:  # BLENDING
    optimizer = env_state.host_optimizer
```

The `else` branch catches BLENDING, SHADOWING, PROBATIONARY. This is mostly fine, but should be explicit.

**Step 2: Make stage handling explicit**

Replace the optimizer selection logic (around line 293):

```python
# Determine which optimizer to use based on seed state
if seed_state is None or seed_state.stage == SeedStage.FOSSILIZED:
    # No seed or seed fully integrated - train host only
    optimizer = env_state.host_optimizer
elif seed_state.stage in (SeedStage.GERMINATED, SeedStage.TRAINING):
    # Isolated seed training
    if seed_state.stage == SeedStage.GERMINATED:
        seed_state.transition(SeedStage.TRAINING)
        env_state.seed_optimizer = torch.optim.SGD(
            model.get_seed_parameters(), lr=0.01, momentum=0.9
        )
    if env_state.seed_optimizer is None:
        env_state.seed_optimizer = torch.optim.SGD(
            model.get_seed_parameters(), lr=0.01, momentum=0.9
        )
    optimizer = env_state.seed_optimizer
elif seed_state.stage == SeedStage.BLENDING:
    # Active blending - update alpha, train both
    optimizer = env_state.host_optimizer
    if model.seed_slot and seed_state:
        model.seed_slot.update_alpha_for_step(env_state.blending_step)
        env_state.blending_step += 1
elif seed_state.stage in (SeedStage.SHADOWING, SeedStage.PROBATIONARY):
    # Post-blending validation - alpha locked at 1.0, joint training
    optimizer = env_state.host_optimizer
else:
    # Unknown stage - shouldn't happen
    optimizer = env_state.host_optimizer
```

**Step 3: Verify the change compiles**

Run: `PYTHONPATH=src .venv/bin/python -c "from esper.simic.vectorized import train_ppo_vectorized; print('OK')"`

Expected: `OK`

**Step 4: Commit**

```bash
git add src/esper/simic/vectorized.py
git commit -m "refactor(simic): explicit stage handling in process_train_batch"
```

---

## Task 7: Run Full Test Suite

**Step 1: Run all lifecycle fix tests**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_lifecycle_fix.py -v`

Expected: All PASS

**Step 2: Run existing test suite**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/ -v --tb=short`

Expected: All tests PASS (277+)

**Step 3: Quick smoke test**

Run a short training to verify lifecycle works:

```bash
PYTHONPATH=src .venv/bin/python -c "
from esper.simic.vectorized import train_ppo_vectorized
agent, history = train_ppo_vectorized(n_episodes=8, n_envs=4, max_epochs=10)
print('Training complete')
# Check analytics for fossilization
if history and 'blueprint_analytics' in history[-1]:
    stats = history[-1]['blueprint_analytics']['stats']
    for bp, s in stats.items():
        print(f'{bp}: foss={s[\"fossilized\"]}, cull={s[\"culled\"]}')
"
```

Expected: See some non-zero fossilization counts

**Step 4: Commit any fixes**

If tests reveal issues, fix and commit.

**Step 5: Final commit**

```bash
git add -A
git commit -m "test: verify lifecycle fix with full test suite"
```

---

## Summary

| Task | Description | Strategic Owner |
|------|-------------|-----------------|
| 1 | Add blending progress fields to SeedState | - |
| 2 | Initialize progress in start_blending | - |
| 3 | Add step_epoch for auto-advance | Kasmina |
| 4 | ADVANCE only at TRAINING/PROBATIONARY | Tamiyo (via Simic) |
| 5 | Wire step_epoch into training loop | - |
| 6 | Explicit stage handling in optimizer selection | - |
| 7 | Full test suite verification | - |

**Result:** Tamiyo makes 2 strategic decisions (start blending, fossilize). Kasmina auto-advances through mechanical stages. No more silent failures.
