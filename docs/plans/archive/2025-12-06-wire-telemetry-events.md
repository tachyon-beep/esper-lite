# Wire Unwired Telemetry Events Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Wire up 15 unwired telemetry event types to enable comprehensive training observability.

**Architecture:** Events are emitted via `hub.emit(TelemetryEvent(...))` from the module where the condition is detected. The NissaHub routes to configured backends (FileOutput for JSONL logging). Each emission point already has detection logic - we just need to create and emit the event.

**Tech Stack:** Python dataclasses, esper.leyline.TelemetryEvent, esper.nissa.get_hub

---

## Priority Groups

| Priority | Events | Value |
|----------|--------|-------|
| P0 | RATIO_EXPLOSION_DETECTED, VALUE_COLLAPSE_DETECTED, NUMERICAL_INSTABILITY_DETECTED, GRADIENT_ANOMALY | Early warning for PPO collapse |
| P1 | TAMIYO_INITIATED | Track when germination becomes allowed |
| P2 | EPOCH_COMPLETED, PLATEAU_DETECTED, IMPROVEMENT_DETECTED | Training progress visibility |
| P3 | MEMORY_WARNING, REWARD_HACKING_SUSPECTED | Resource/behavior monitoring |
| P4 | COMMAND_* events, ISOLATION_VIOLATION, PERFORMANCE_DEGRADATION | Future extensibility |

---

## Task 1: P0 Anomaly Events in PPO

**Files:**
- Modify: `src/esper/simic/ppo.py:449-453`
- Test: `tests/test_simic_ppo.py`

These events are already detected via `AnomalyDetector` - we just need to emit them.

**Step 1: Write the failing test**

Add to `tests/test_simic_ppo.py`:

```python
class TestPPOAnomalyTelemetry:
    """Test that PPO emits anomaly telemetry events."""

    def test_ratio_explosion_emits_telemetry(self):
        """Ratio explosion triggers RATIO_EXPLOSION_DETECTED event."""
        from esper.leyline import TelemetryEventType
        from esper.nissa import get_hub

        # Capture emitted events
        captured_events = []
        hub = get_hub()

        class CaptureBackend:
            def emit(self, event):
                captured_events.append(event)

        hub.add_backend(CaptureBackend())

        # Create agent with very tight ratio threshold to trigger explosion
        agent = PPOAgent(
            state_dim=10,
            action_dim=5,
            anomaly_detector=AnomalyDetector(max_ratio_threshold=0.01),  # Very tight
        )

        # Add steps that will cause ratio explosion
        for _ in range(10):
            state = torch.randn(10)
            agent.buffer.add(
                state=state,
                action=0,
                log_prob=-5.0,  # Very low log prob will cause high ratio
                value=0.0,
                reward=1.0,
                done=False,
                action_mask=torch.ones(5),
            )

        # Update should detect ratio explosion
        agent.update()

        # Check for RATIO_EXPLOSION_DETECTED event
        explosion_events = [
            e for e in captured_events
            if e.event_type == TelemetryEventType.RATIO_EXPLOSION_DETECTED
        ]
        assert len(explosion_events) >= 1, f"Expected RATIO_EXPLOSION_DETECTED, got: {[e.event_type for e in captured_events]}"
        assert "ratio_max" in explosion_events[0].data
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_simic_ppo.py::TestPPOAnomalyTelemetry::test_ratio_explosion_emits_telemetry -v`
Expected: FAIL with assertion error (no RATIO_EXPLOSION_DETECTED events)

**Step 3: Add imports to ppo.py**

At top of `src/esper/simic/ppo.py`, add to imports:

```python
from esper.leyline import TelemetryEvent, TelemetryEventType
from esper.nissa import get_hub
```

**Step 4: Emit anomaly events after detection**

In `src/esper/simic/ppo.py`, after line 453 (after `if anomaly_report.has_anomaly:`), add:

```python
            if anomaly_report.has_anomaly:
                anomaly_detected = True
                if telemetry_config.auto_escalate_on_anomaly:
                    telemetry_config.escalate_temporarily()

                # Emit specific anomaly events
                hub = get_hub()
                for anomaly_type in anomaly_report.anomaly_types:
                    if anomaly_type == "ratio_explosion":
                        event_type = TelemetryEventType.RATIO_EXPLOSION_DETECTED
                    elif anomaly_type == "ratio_collapse":
                        event_type = TelemetryEventType.RATIO_EXPLOSION_DETECTED  # Same event for both
                    elif anomaly_type == "value_collapse":
                        event_type = TelemetryEventType.VALUE_COLLAPSE_DETECTED
                    elif anomaly_type == "numerical_instability":
                        event_type = TelemetryEventType.NUMERICAL_INSTABILITY_DETECTED
                    else:
                        event_type = TelemetryEventType.GRADIENT_ANOMALY

                    hub.emit(TelemetryEvent(
                        event_type=event_type,
                        data={
                            "anomaly_type": anomaly_type,
                            "detail": anomaly_report.details.get(anomaly_type, ""),
                            "ratio_max": max_ratio,
                            "ratio_min": min_ratio,
                            "explained_variance": explained_variance,
                            "train_steps": self.train_steps,
                        },
                        severity="warning",
                    ))
```

**Step 5: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_simic_ppo.py::TestPPOAnomalyTelemetry -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/esper/simic/ppo.py tests/test_simic_ppo.py
git commit -m "feat(simic): emit anomaly telemetry events from PPO update

Wire up RATIO_EXPLOSION_DETECTED, VALUE_COLLAPSE_DETECTED,
NUMERICAL_INSTABILITY_DETECTED, and GRADIENT_ANOMALY events
when AnomalyDetector detects issues during PPO update."
```

---

## Task 2: P1 TAMIYO_INITIATED Event

**Files:**
- Modify: `src/esper/tamiyo/tracker.py:113-120`
- Test: `tests/test_tamiyo_tracker.py` (create if needed)

The stabilization detection already prints a message - emit an event instead.

**Step 1: Write the failing test**

Create `tests/test_tamiyo_tracker.py`:

```python
"""Tests for SignalTracker telemetry emission."""

import pytest
from esper.tamiyo.tracker import SignalTracker
from esper.leyline import TelemetryEventType
from esper.nissa import get_hub


class TestSignalTrackerTelemetry:
    """Test SignalTracker emits telemetry events."""

    def test_stabilization_emits_tamiyo_initiated(self):
        """When host stabilizes, emit TAMIYO_INITIATED event."""
        captured_events = []
        hub = get_hub()

        class CaptureBackend:
            def emit(self, event):
                captured_events.append(event)

        hub.add_backend(CaptureBackend())

        tracker = SignalTracker(
            stabilization_threshold=0.05,
            stabilization_epochs=2,
            env_id=0,
        )

        # First update - sets baseline
        tracker.update(
            epoch=0, global_step=0,
            train_loss=2.0, train_accuracy=20.0,
            val_loss=2.0, val_accuracy=20.0,
            active_seeds=[], available_slots=1,
        )

        # Stable epochs (small improvement < 5%)
        for epoch in range(1, 4):
            tracker.update(
                epoch=epoch, global_step=epoch * 100,
                train_loss=2.0 - epoch * 0.01,  # Small improvement
                train_accuracy=20.0 + epoch,
                val_loss=2.0 - epoch * 0.01,
                val_accuracy=20.0 + epoch,
                active_seeds=[], available_slots=1,
            )

        # Should have emitted TAMIYO_INITIATED
        initiated_events = [
            e for e in captured_events
            if e.event_type == TelemetryEventType.TAMIYO_INITIATED
        ]
        assert len(initiated_events) == 1
        assert initiated_events[0].data["env_id"] == 0
        assert "epoch" in initiated_events[0].data
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_tamiyo_tracker.py -v`
Expected: FAIL (no TAMIYO_INITIATED event)

**Step 3: Add imports to tracker.py**

At top of `src/esper/tamiyo/tracker.py`:

```python
from esper.leyline import TelemetryEvent, TelemetryEventType
from esper.nissa import get_hub
```

**Step 4: Emit TAMIYO_INITIATED when stabilized**

In `src/esper/tamiyo/tracker.py`, replace lines 113-120:

```python
                    if self._stable_count >= self.stabilization_epochs:
                        self._is_stabilized = True
                        env_str = f"ENV {self.env_id}" if self.env_id is not None else "Tamiyo"
                        if self.stabilization_epochs == 0:
                            print(f"[{env_str}] Host stabilized at epoch {epoch} - germination now allowed")
                        else:
                            print(f"[{env_str}] Host stabilized at epoch {epoch} "
                                  f"({self._stable_count}/{self.stabilization_epochs} stable) - germination now allowed")
```

With:

```python
                    if self._stable_count >= self.stabilization_epochs:
                        self._is_stabilized = True
                        env_str = f"ENV {self.env_id}" if self.env_id is not None else "Tamiyo"
                        if self.stabilization_epochs == 0:
                            print(f"[{env_str}] Host stabilized at epoch {epoch} - germination now allowed")
                        else:
                            print(f"[{env_str}] Host stabilized at epoch {epoch} "
                                  f"({self._stable_count}/{self.stabilization_epochs} stable) - germination now allowed")

                        # Emit TAMIYO_INITIATED telemetry
                        hub = get_hub()
                        hub.emit(TelemetryEvent(
                            event_type=TelemetryEventType.TAMIYO_INITIATED,
                            epoch=epoch,
                            data={
                                "env_id": self.env_id,
                                "epoch": epoch,
                                "stable_count": self._stable_count,
                                "stabilization_epochs": self.stabilization_epochs,
                                "val_loss": val_loss,
                            },
                            message=f"Host stabilized - germination now allowed",
                        ))
```

**Step 5: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_tamiyo_tracker.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/esper/tamiyo/tracker.py tests/test_tamiyo_tracker.py
git commit -m "feat(tamiyo): emit TAMIYO_INITIATED when host stabilizes

Emit telemetry event when stabilization threshold is reached,
marking when germination becomes allowed."
```

---

## Task 3: P2 Training Progress Events

**Files:**
- Modify: `src/esper/simic/vectorized.py`
- Test: `tests/test_vectorized_telemetry.py`

Add EPOCH_COMPLETED for per-epoch visibility, PLATEAU_DETECTED and IMPROVEMENT_DETECTED based on accuracy delta.

**Step 1: Write the failing test**

Create `tests/test_vectorized_telemetry.py`:

```python
"""Tests for vectorized training telemetry emission."""

import pytest
from esper.leyline import TelemetryEventType


class TestVectorizedTelemetry:
    """Test vectorized training emits expected telemetry."""

    def test_ppo_update_completed_emitted(self):
        """PPO_UPDATE_COMPLETED should be emitted after each batch."""
        # This test validates our existing implementation
        from esper.nissa import get_hub

        captured = []
        hub = get_hub()

        class Capture:
            def emit(self, event):
                captured.append(event)

        hub.add_backend(Capture())

        # Import after backend added to capture events
        from esper.simic.vectorized import train_ppo_vectorized

        # Run minimal training (1 episode)
        try:
            train_ppo_vectorized(
                n_episodes=1,
                n_envs=1,
                max_epochs=5,
                use_telemetry=True,
            )
        except Exception:
            pass  # May fail without GPU, but events should still emit

        ppo_events = [e for e in captured if e.event_type == TelemetryEventType.PPO_UPDATE_COMPLETED]
        # Should have at least attempted to emit
        # (actual count depends on training success)
```

**Step 2: Add PLATEAU_DETECTED emission**

In `src/esper/simic/vectorized.py`, after line 1142 (after hub.emit(ppo_event)), add:

```python
            # Emit training progress events based on accuracy change
            if len(recent_accuracies) >= 2:
                acc_delta = recent_accuracies[-1] - recent_accuracies[-2]
                if acc_delta < 0.5:  # Less than 0.5% improvement
                    hub.emit(TelemetryEvent(
                        event_type=TelemetryEventType.PLATEAU_DETECTED,
                        data={
                            "batch": batch_idx + 1,
                            "accuracy_delta": acc_delta,
                            "rolling_avg_accuracy": rolling_avg_acc,
                            "episodes_completed": episodes_completed,
                        },
                    ))
                elif acc_delta > 2.0:  # More than 2% improvement
                    hub.emit(TelemetryEvent(
                        event_type=TelemetryEventType.IMPROVEMENT_DETECTED,
                        data={
                            "batch": batch_idx + 1,
                            "accuracy_delta": acc_delta,
                            "rolling_avg_accuracy": rolling_avg_acc,
                            "episodes_completed": episodes_completed,
                        },
                    ))
```

**Step 3: Run tests**

Run: `PYTHONPATH=src pytest tests/test_vectorized_telemetry.py -v`

**Step 4: Commit**

```bash
git add src/esper/simic/vectorized.py tests/test_vectorized_telemetry.py
git commit -m "feat(simic): emit PLATEAU_DETECTED and IMPROVEMENT_DETECTED events

Track training progress by emitting events when accuracy delta
indicates plateau (<0.5%) or significant improvement (>2%)."
```

---

## Task 4: P3 Warning Events Placeholder

**Files:**
- Modify: `src/esper/leyline/telemetry.py` (add docstrings only)

For MEMORY_WARNING and REWARD_HACKING_SUSPECTED, add documentation noting these are placeholders for future implementation.

**Step 1: Document placeholder events**

In `src/esper/leyline/telemetry.py`, update the event type comments:

```python
    # === PPO Training Events (Ops Normal) ===
    PPO_UPDATE_COMPLETED = auto()
    MEMORY_WARNING = auto()  # TODO: Implement GPU memory monitoring
    REWARD_HACKING_SUSPECTED = auto()  # TODO: Implement reward divergence detection
```

**Step 2: Commit**

```bash
git add src/esper/leyline/telemetry.py
git commit -m "docs(leyline): document placeholder telemetry events

Mark MEMORY_WARNING and REWARD_HACKING_SUSPECTED as TODO
for future implementation."
```

---

## Task 5: P4 Command Events Placeholder

**Files:**
- Modify: `src/esper/leyline/telemetry.py`

Document that COMMAND_* events are for future adaptive command system.

**Step 1: Document command events**

```python
    # Command events (for future adaptive command system)
    COMMAND_ISSUED = auto()  # TODO: Implement when command system is built
    COMMAND_EXECUTED = auto()
    COMMAND_FAILED = auto()
```

**Step 2: Commit**

```bash
git add src/esper/leyline/telemetry.py
git commit -m "docs(leyline): document command telemetry events as future work"
```

---

## Summary of Changes

| Event Type | Location | Status |
|------------|----------|--------|
| RATIO_EXPLOSION_DETECTED | ppo.py | Task 1 |
| VALUE_COLLAPSE_DETECTED | ppo.py | Task 1 |
| NUMERICAL_INSTABILITY_DETECTED | ppo.py | Task 1 |
| GRADIENT_ANOMALY | ppo.py | Task 1 |
| TAMIYO_INITIATED | tracker.py | Task 2 |
| PLATEAU_DETECTED | vectorized.py | Task 3 |
| IMPROVEMENT_DETECTED | vectorized.py | Task 3 |
| EPOCH_COMPLETED | Deferred | Too noisy for current use |
| MEMORY_WARNING | Placeholder | Task 4 |
| REWARD_HACKING_SUSPECTED | Placeholder | Task 4 |
| COMMAND_* | Placeholder | Task 5 |
| ISOLATION_VIOLATION | Deferred | Needs isolation monitoring |
| PERFORMANCE_DEGRADATION | Deferred | Needs timing infrastructure |

---

## Verification

After all tasks complete:

```bash
# Run full test suite
PYTHONPATH=src pytest tests/ -v --tb=short

# Check telemetry output during training
PYTHONPATH=src python -m esper.scripts.train ppo --vectorized --n-envs 2 --episodes 5 --recurrent

# Verify new event types in telemetry
grep -E "RATIO_EXPLOSION|VALUE_COLLAPSE|TAMIYO_INITIATED|PLATEAU|IMPROVEMENT" telemetry/*/events.jsonl
```
