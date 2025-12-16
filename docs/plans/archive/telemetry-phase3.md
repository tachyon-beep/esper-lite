# Telemetry Overwatch Phase 3 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Wire remaining telemetry events for health monitoring, reward integrity, and counterfactual coverage.

**Architecture:** Emit events from existing monitoring points (karn/health, tolaria/governor, simic/rewards, karn/counterfactual) with configurable thresholds. All new emissions gated behind flags to avoid throughput regressions.

**Tech Stack:** Python 3.11+, PyTorch, pytest, existing telemetry infrastructure.

---

## Scope

### In Scope
- **MEMORY_WARNING** — Emit when GPU memory exceeds threshold
- **PERFORMANCE_DEGRADATION** — Emit when rolling accuracy drops significantly
- **REWARD_HACKING_SUSPECTED** — Emit when attribution/improvement ratio is anomalous
- **Shapley telemetry** — Emit when Shapley values are computed

### Permanently Deferred
- **COMMAND_ISSUED / COMMAND_EXECUTED / COMMAND_FAILED** — No external controller exists. These event types remain as placeholders for future orchestration integration but will not be wired in Phase 3.

---

## Task 1: Wire MEMORY_WARNING from HealthMonitor

**Files:**
- Modify: `src/esper/karn/health.py`
- Modify: `src/esper/simic/vectorized.py` (optional hook)
- Test: `tests/karn/test_health_telemetry.py` (create)

**Step 1: Write failing test**

```python
# tests/karn/test_health_telemetry.py
from unittest.mock import Mock, patch

def test_memory_warning_emitted_when_threshold_exceeded():
    from esper.karn.health import HealthMonitor
    from esper.leyline import TelemetryEventType

    events = []
    def capture(event):
        events.append(event)

    with patch("esper.karn.health.get_hub") as mock_hub:
        mock_hub.return_value.emit = capture

        monitor = HealthMonitor(memory_warning_threshold=0.8)
        # Simulate 90% GPU utilization
        monitor._check_memory_and_warn(gpu_utilization=0.9, gpu_allocated_gb=10.0, gpu_total_gb=11.1)

    assert len(events) == 1
    assert events[0].event_type == TelemetryEventType.MEMORY_WARNING
    assert events[0].data["gpu_utilization"] == 0.9


def test_no_memory_warning_when_below_threshold():
    from esper.karn.health import HealthMonitor

    events = []
    with patch("esper.karn.health.get_hub") as mock_hub:
        mock_hub.return_value.emit = lambda e: events.append(e)

        monitor = HealthMonitor(memory_warning_threshold=0.8)
        monitor._check_memory_and_warn(gpu_utilization=0.7, gpu_allocated_gb=7.0, gpu_total_gb=10.0)

    assert len(events) == 0
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/test_health_telemetry.py -v`
Expected: FAIL with AttributeError (method doesn't exist)

**Step 3: Implement**

```python
# In src/esper/karn/health.py, add to HealthMonitor class:

from esper.nissa import get_hub
from esper.leyline import TelemetryEvent, TelemetryEventType

class HealthMonitor:
    def __init__(
        self,
        memory_warning_threshold: float = 0.85,  # Warn at 85% GPU utilization
        memory_warning_cooldown: float = 60.0,   # Don't spam warnings
    ):
        self.memory_warning_threshold = memory_warning_threshold
        self.memory_warning_cooldown = memory_warning_cooldown
        self._last_memory_warning: float = 0.0

    def _check_memory_and_warn(
        self,
        gpu_utilization: float,
        gpu_allocated_gb: float,
        gpu_total_gb: float,
    ) -> bool:
        """Check memory and emit MEMORY_WARNING if threshold exceeded.

        Returns True if warning was emitted.
        """
        import time

        if gpu_utilization < self.memory_warning_threshold:
            return False

        now = time.time()
        if now - self._last_memory_warning < self.memory_warning_cooldown:
            return False  # Cooldown active

        self._last_memory_warning = now
        hub = get_hub()
        if hub is not None:
            hub.emit(TelemetryEvent(
                event_type=TelemetryEventType.MEMORY_WARNING,
                severity="warning",
                data={
                    "gpu_utilization": gpu_utilization,
                    "gpu_allocated_gb": gpu_allocated_gb,
                    "gpu_total_gb": gpu_total_gb,
                    "threshold": self.memory_warning_threshold,
                },
            ))
        return True
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/test_health_telemetry.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/health.py tests/karn/test_health_telemetry.py
git commit -m "feat(telemetry): wire MEMORY_WARNING from HealthMonitor"
```

---

## Task 2: Wire PERFORMANCE_DEGRADATION from training loop

**Files:**
- Modify: `src/esper/simic/vectorized.py`
- Test: `tests/simic/test_vectorized.py` (extend)

**Step 1: Write failing test**

```python
# tests/simic/test_vectorized.py (add to existing file)
def test_performance_degradation_emitted_on_accuracy_drop():
    from esper.simic.vectorized import _check_performance_degradation
    from esper.leyline import TelemetryEventType
    from unittest.mock import Mock

    hub = Mock()

    # Accuracy dropped from 0.8 to 0.6 (25% drop)
    emitted = _check_performance_degradation(
        hub=hub,
        current_acc=0.6,
        rolling_avg_acc=0.8,
        degradation_threshold=0.1,  # 10% drop triggers
        env_id=0,
    )

    assert emitted is True
    assert hub.emit.called
    event = hub.emit.call_args[0][0]
    assert event.event_type == TelemetryEventType.PERFORMANCE_DEGRADATION
    assert event.data["current_acc"] == 0.6
    assert event.data["rolling_avg_acc"] == 0.8


def test_no_degradation_event_when_stable():
    from esper.simic.vectorized import _check_performance_degradation
    from unittest.mock import Mock

    hub = Mock()

    emitted = _check_performance_degradation(
        hub=hub,
        current_acc=0.78,
        rolling_avg_acc=0.8,
        degradation_threshold=0.1,
        env_id=0,
    )

    assert emitted is False
    assert not hub.emit.called
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/simic/test_vectorized.py::test_performance_degradation_emitted_on_accuracy_drop -v`
Expected: FAIL (function doesn't exist)

**Step 3: Implement**

```python
# In src/esper/simic/vectorized.py, add helper function:

def _check_performance_degradation(
    hub,
    *,
    current_acc: float,
    rolling_avg_acc: float,
    degradation_threshold: float = 0.1,
    env_id: int = 0,
) -> bool:
    """Emit PERFORMANCE_DEGRADATION if accuracy dropped significantly.

    Returns True if event was emitted.
    """
    if rolling_avg_acc <= 0:
        return False

    drop = (rolling_avg_acc - current_acc) / rolling_avg_acc

    if drop < degradation_threshold:
        return False

    hub.emit(TelemetryEvent(
        event_type=TelemetryEventType.PERFORMANCE_DEGRADATION,
        severity="warning",
        data={
            "current_acc": current_acc,
            "rolling_avg_acc": rolling_avg_acc,
            "drop_percent": drop * 100,
            "threshold_percent": degradation_threshold * 100,
            "env_id": env_id,
        },
    ))
    return True
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/simic/test_vectorized.py::test_performance_degradation_emitted_on_accuracy_drop tests/simic/test_vectorized.py::test_no_degradation_event_when_stable -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/vectorized.py tests/simic/test_vectorized.py
git commit -m "feat(telemetry): wire PERFORMANCE_DEGRADATION from training loop"
```

---

## Task 3: Wire REWARD_HACKING_SUSPECTED from rewards

**Files:**
- Modify: `src/esper/simic/rewards.py`
- Test: `tests/simic/test_rewards.py` (extend)

**Step 1: Write failing test**

```python
# tests/simic/test_rewards.py (add to existing file)
def test_reward_hacking_suspected_emitted_on_anomalous_ratio():
    from esper.simic.rewards import _check_reward_hacking
    from esper.leyline import TelemetryEventType
    from unittest.mock import Mock

    hub = Mock()

    # Seed claims 500% of total improvement (impossible without hacking)
    emitted = _check_reward_hacking(
        hub=hub,
        seed_contribution=5.0,
        total_improvement=1.0,
        hacking_ratio_threshold=3.0,  # 300% is suspicious
        slot_id="r0c0",
        seed_id="seed_001",
    )

    assert emitted is True
    event = hub.emit.call_args[0][0]
    assert event.event_type == TelemetryEventType.REWARD_HACKING_SUSPECTED
    assert event.data["ratio"] == 5.0
    assert event.data["slot_id"] == "r0c0"


def test_no_hacking_event_for_normal_ratios():
    from esper.simic.rewards import _check_reward_hacking
    from unittest.mock import Mock

    hub = Mock()

    emitted = _check_reward_hacking(
        hub=hub,
        seed_contribution=0.8,
        total_improvement=1.0,
        hacking_ratio_threshold=3.0,
        slot_id="r0c0",
        seed_id="seed_001",
    )

    assert emitted is False
    assert not hub.emit.called
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/simic/test_rewards.py::test_reward_hacking_suspected_emitted_on_anomalous_ratio -v`
Expected: FAIL (function doesn't exist)

**Step 3: Implement**

```python
# In src/esper/simic/rewards.py, add helper:

from esper.nissa import get_hub
from esper.leyline import TelemetryEvent, TelemetryEventType

def _check_reward_hacking(
    hub,
    *,
    seed_contribution: float,
    total_improvement: float,
    hacking_ratio_threshold: float = 3.0,
    slot_id: str,
    seed_id: str,
) -> bool:
    """Emit REWARD_HACKING_SUSPECTED if attribution ratio is anomalous.

    A seed claiming more than 300% of total improvement is suspicious
    and may indicate reward hacking or measurement error.

    Returns True if event was emitted.
    """
    if total_improvement <= 0 or seed_contribution <= 0:
        return False

    ratio = seed_contribution / total_improvement

    if ratio < hacking_ratio_threshold:
        return False

    hub.emit(TelemetryEvent(
        event_type=TelemetryEventType.REWARD_HACKING_SUSPECTED,
        severity="warning",
        data={
            "ratio": ratio,
            "seed_contribution": seed_contribution,
            "total_improvement": total_improvement,
            "threshold": hacking_ratio_threshold,
            "slot_id": slot_id,
            "seed_id": seed_id,
        },
    ))
    return True
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/simic/test_rewards.py::test_reward_hacking_suspected_emitted_on_anomalous_ratio tests/simic/test_rewards.py::test_no_hacking_event_for_normal_ratios -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/rewards.py tests/simic/test_rewards.py
git commit -m "feat(telemetry): wire REWARD_HACKING_SUSPECTED from rewards"
```

---

## Task 4: Wire Shapley telemetry from counterfactual engine

**Files:**
- Modify: `src/esper/karn/counterfactual.py`
- Modify: `src/esper/karn/counterfactual_helper.py`
- Test: `tests/karn/test_counterfactual_telemetry.py` (create)

**Step 1: Write failing test**

```python
# tests/karn/test_counterfactual_telemetry.py
from unittest.mock import Mock, patch

def test_shapley_computed_event_emitted():
    from esper.karn.counterfactual import CounterfactualEngine
    from esper.leyline import TelemetryEventType

    events = []

    with patch("esper.karn.counterfactual.get_hub") as mock_hub:
        mock_hub.return_value.emit = lambda e: events.append(e)

        engine = CounterfactualEngine(emit_telemetry=True)

        # Simulate Shapley computation with mock matrix
        import numpy as np
        matrix = np.array([
            [0.70, 0.75, 0.78],  # baseline, +slot0, +slot0+slot1
            [0.70, 0.72, 0.78],  # baseline, +slot1, +slot0+slot1
        ])
        values = engine.compute_shapley_values(matrix)

    shapley_events = [e for e in events if e.event_type == TelemetryEventType.ANALYTICS_SNAPSHOT
                      and e.data.get("kind") == "shapley_computed"]
    assert len(shapley_events) == 1
    assert "shapley_values" in shapley_events[0].data
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/test_counterfactual_telemetry.py -v`
Expected: FAIL (emit_telemetry parameter doesn't exist or event not emitted)

**Step 3: Implement**

```python
# In src/esper/karn/counterfactual.py, modify CounterfactualEngine:

class CounterfactualEngine:
    def __init__(self, emit_telemetry: bool = False):
        self.emit_telemetry = emit_telemetry

    def compute_shapley_values(self, contribution_matrix) -> dict[str, float]:
        """Compute Shapley values from contribution matrix.

        Emits ANALYTICS_SNAPSHOT with kind='shapley_computed' if emit_telemetry=True.
        """
        # ... existing computation logic ...

        # After computing values:
        if self.emit_telemetry:
            from esper.nissa import get_hub
            from esper.leyline import TelemetryEvent, TelemetryEventType

            hub = get_hub()
            if hub is not None:
                hub.emit(TelemetryEvent(
                    event_type=TelemetryEventType.ANALYTICS_SNAPSHOT,
                    data={
                        "kind": "shapley_computed",
                        "shapley_values": values,  # dict of slot_id -> value
                        "num_slots": len(values),
                    },
                ))

        return values
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/test_counterfactual_telemetry.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/counterfactual.py tests/karn/test_counterfactual_telemetry.py
git commit -m "feat(telemetry): wire Shapley telemetry from counterfactual engine"
```

---

## Task 5: Integration - Hook telemetry into training loop (optional)

**Files:**
- Modify: `src/esper/simic/vectorized.py`

This task wires the new emissions into actual training. It's optional for initial delivery since the helper functions are independently testable.

**Implementation notes:**
- Call `_check_performance_degradation()` at end of each epoch
- Call `HealthMonitor._check_memory_and_warn()` periodically (e.g., every 10 batches)
- `_check_reward_hacking()` is called from within `compute_bounded_reward()` when appropriate

---

## Final Verification

Run full test suite:
```bash
PYTHONPATH=src uv run pytest -m "not slow" -q
```

Smoke test with telemetry enabled:
```bash
PYTHONPATH=src uv run python -m esper.scripts.train heuristic --task cifar10 --episodes 1 --telemetry-level debug
```

---

## Execution Handoff

Plan complete and saved to `docs/plans/telemetry-phase3.md`. Two execution options:

1. **Subagent-Driven (this session)** — use superpowers:subagent-driven-development to execute tasks sequentially with code review between tasks.

2. **Parallel Session** — new session using superpowers:executing-plans to run the plan in batches with checkpoints.
