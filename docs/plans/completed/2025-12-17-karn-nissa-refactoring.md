# Karn & Nissa Architecture Refactoring Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix architectural antipatterns in the Karn and Nissa telemetry subsystems to improve maintainability, testability, and correctness.

**Architecture:** Remove the reverse dependency (Karn→Nissa) via callback injection, centralize scattered thresholds into a constants module, fix fragile enum comparisons, and extract shared serialization code. The TUI decomposition is deferred to a follow-up plan due to scope.

**Tech Stack:** Python 3.11+, dataclasses, Protocols, pytest

---

## Task Overview

| Task | Description | Priority |
|------|-------------|----------|
| 1 | Create `karn/constants.py` with centralized thresholds | P1 |
| 2 | Migrate threshold consumers to use constants | P1 |
| 3 | Fix fragile enum comparisons in health.py | P2 |
| 4 | Create `karn/serialization.py` for shared event serialization | P3 |
| 5 | Deduplicate serialization in websocket_output.py and integrated_dashboard.py | P3 |
| 6 | Remove Karn→Nissa dependency in health.py | P0 |
| 7 | Remove Karn→Nissa dependency in counterfactual.py | P0 |

---

## Task 1: Create `karn/constants.py` with Centralized Thresholds

**Files:**
- Create: `src/esper/karn/constants.py`
- Test: `tests/karn/test_constants.py`

**Step 1: Write the failing test**

```python
# tests/karn/test_constants.py
"""Tests for Karn constants module."""

import pytest
from esper.karn.constants import (
    AnomalyThresholds,
    PolicyThresholds,
    HealthThresholds,
    TUIThresholds,
)


class TestAnomalyThresholds:
    """Test anomaly detection thresholds."""

    def test_loss_spike_threshold_is_positive(self) -> None:
        assert AnomalyThresholds.LOSS_SPIKE_MULTIPLIER > 0

    def test_accuracy_drop_threshold_is_positive(self) -> None:
        assert AnomalyThresholds.ACCURACY_DROP_POINTS > 0

    def test_gradient_explosion_multiplier_is_large(self) -> None:
        # Should be at least 10x to count as "explosion"
        assert AnomalyThresholds.GRADIENT_EXPLOSION_MULTIPLIER >= 10.0


class TestPolicyThresholds:
    """Test PPO policy anomaly thresholds."""

    def test_value_std_threshold_is_small(self) -> None:
        # Value collapse threshold should be small (near zero)
        assert 0 < PolicyThresholds.VALUE_STD_COLLAPSE < 0.1

    def test_entropy_threshold_is_reasonable(self) -> None:
        # Entropy collapse should trigger before reaching 0
        assert 0 < PolicyThresholds.ENTROPY_COLLAPSE < 0.5

    def test_kl_threshold_is_positive(self) -> None:
        assert PolicyThresholds.KL_SPIKE > 0


class TestHealthThresholds:
    """Test system health thresholds."""

    def test_gpu_warning_is_high(self) -> None:
        # GPU warning should be above 80% utilization
        assert HealthThresholds.GPU_UTILIZATION_WARNING > 0.8

    def test_grad_norm_warning_less_than_error(self) -> None:
        assert HealthThresholds.GRAD_NORM_WARNING < HealthThresholds.GRAD_NORM_ERROR


class TestTUIThresholds:
    """Test TUI display thresholds."""

    def test_entropy_warning_less_than_max(self) -> None:
        assert TUIThresholds.ENTROPY_WARNING < TUIThresholds.ENTROPY_MAX

    def test_entropy_critical_less_than_warning(self) -> None:
        assert TUIThresholds.ENTROPY_CRITICAL < TUIThresholds.ENTROPY_WARNING
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/test_constants.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'esper.karn.constants'"

**Step 3: Write minimal implementation**

```python
# src/esper/karn/constants.py
"""Karn Constants - Centralized thresholds for telemetry subsystem.

All magic numbers for anomaly detection, health monitoring, and TUI
display are defined here. This enables:
- Global tuning without hunting through multiple files
- Clear documentation of threshold rationale
- Type-safe access via class attributes

Usage:
    from esper.karn.constants import AnomalyThresholds, HealthThresholds

    if loss_ratio > AnomalyThresholds.LOSS_SPIKE_MULTIPLIER:
        trigger_dense_trace()
"""

from __future__ import annotations


class AnomalyThresholds:
    """Thresholds for dense trace triggering (Tier 3 capture).

    These control when the system captures detailed diagnostics.
    """

    # Loss spike: trigger if loss > N× rolling EMA
    LOSS_SPIKE_MULTIPLIER: float = 2.0

    # Accuracy drop: trigger if accuracy drops by N percentage points
    ACCURACY_DROP_POINTS: float = 5.0

    # Gradient explosion: trigger if grad norm > N× rolling EMA
    GRADIENT_EXPLOSION_MULTIPLIER: float = 100.0

    # Dense trace window: capture N epochs after trigger
    TRACE_WINDOW_EPOCHS: int = 3


class PolicyThresholds:
    """Thresholds for PPO policy anomaly detection.

    These detect pathological policy behavior during RL training.
    """

    # Value collapse: critic outputs have std below this → collapse
    VALUE_STD_COLLAPSE: float = 0.01

    # Entropy collapse: policy entropy below this → deterministic
    ENTROPY_COLLAPSE: float = 0.1

    # KL spike: policy change above this → large update
    KL_SPIKE: float = 0.1

    # Rolling window for anomaly detection
    WINDOW_SIZE: int = 10


class HealthThresholds:
    """Thresholds for system health monitoring.

    These trigger warnings and errors for resource/gradient issues.
    """

    # GPU memory utilization (0-1)
    GPU_UTILIZATION_WARNING: float = 0.9
    MEMORY_WARNING_THRESHOLD: float = 0.85
    MEMORY_WARNING_COOLDOWN_SECONDS: float = 60.0

    # Gradient norm thresholds
    GRAD_NORM_WARNING: float = 50.0
    GRAD_NORM_ERROR: float = 100.0

    # Gradient explosion indicator (likely Inf)
    GRAD_NORM_EXPLOSION: float = 1e10


class TUIThresholds:
    """Thresholds for TUI color-coded health display.

    These control green/yellow/red status indicators.
    """

    # Entropy (healthy starts near ln(4) ≈ 1.39 for 4 actions)
    ENTROPY_MAX: float = 1.39  # ln(4) for 4 actions
    ENTROPY_WARNING: float = 0.5
    ENTROPY_CRITICAL: float = 0.3

    # Clip fraction (target 0.1-0.2)
    CLIP_WARNING: float = 0.25
    CLIP_CRITICAL: float = 0.3

    # Explained variance (value learning quality)
    EXPLAINED_VAR_WARNING: float = 0.7
    EXPLAINED_VAR_CRITICAL: float = 0.5

    # Gradient norm
    GRAD_NORM_WARNING: float = 5.0
    GRAD_NORM_CRITICAL: float = 10.0

    # KL divergence (policy change magnitude)
    KL_WARNING: float = 0.05

    # Action distribution (WAIT dominance is suspicious)
    WAIT_DOMINANCE_WARNING: float = 0.7  # > 70% WAIT


class VitalSignsThresholds:
    """Thresholds for vital signs monitoring.

    These detect training failure patterns.
    """

    # Loss spike relative to recent average
    LOSS_SPIKE_MULTIPLIER: float = 2.0

    # Epochs without improvement before stagnation warning
    STAGNATION_EPOCHS: int = 20
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/test_constants.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/constants.py tests/karn/test_constants.py
git commit -m "$(cat <<'EOF'
feat(karn): add centralized constants module for thresholds

Extracts all magic numbers from triggers.py, health.py, and tui.py
into a single constants.py module. This enables global tuning and
clear documentation of threshold rationale.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Migrate Threshold Consumers to Use Constants

**Files:**
- Modify: `src/esper/karn/triggers.py`
- Modify: `src/esper/karn/health.py`
- Modify: `src/esper/karn/tui.py`
- Modify: `src/esper/karn/store.py` (DenseTraceTrigger defaults)

**Step 1: Update triggers.py to use constants**

```python
# In src/esper/karn/triggers.py, add import at top:
from esper.karn.constants import AnomalyThresholds, PolicyThresholds

# Replace PolicyAnomalyDetector defaults (around line 232-235):
# OLD:
#     value_std_threshold: float = 0.01
#     entropy_threshold: float = 0.1
#     kl_threshold: float = 0.1
# NEW:
    value_std_threshold: float = PolicyThresholds.VALUE_STD_COLLAPSE
    entropy_threshold: float = PolicyThresholds.ENTROPY_COLLAPSE
    kl_threshold: float = PolicyThresholds.KL_SPIKE
    window_size: int = PolicyThresholds.WINDOW_SIZE

# Replace AnomalyDetector._trace_window_epochs (around line 103):
# OLD:
#     _trace_window_epochs: int = 3
# NEW:
    _trace_window_epochs: int = AnomalyThresholds.TRACE_WINDOW_EPOCHS
```

**Step 2: Update health.py to use constants**

```python
# In src/esper/karn/health.py, add import at top:
from esper.karn.constants import HealthThresholds

# Replace HealthMonitor.__init__ defaults (around line 137-141):
# OLD:
#         gpu_warning_threshold: float = 0.9,
#         grad_norm_warning: float = 50.0,
#         grad_norm_error: float = 100.0,
#         memory_warning_threshold: float = 0.85,
#         memory_warning_cooldown: float = 60.0,
# NEW:
        gpu_warning_threshold: float = HealthThresholds.GPU_UTILIZATION_WARNING,
        grad_norm_warning: float = HealthThresholds.GRAD_NORM_WARNING,
        grad_norm_error: float = HealthThresholds.GRAD_NORM_ERROR,
        memory_warning_threshold: float = HealthThresholds.MEMORY_WARNING_THRESHOLD,
        memory_warning_cooldown: float = HealthThresholds.MEMORY_WARNING_COOLDOWN_SECONDS,

# Replace GradientHealth.is_healthy threshold (around line 95):
# OLD:
#         return not self.has_nan and not self.has_inf and self.mean_norm < 100.0
# NEW:
        return not self.has_nan and not self.has_inf and self.mean_norm < HealthThresholds.GRAD_NORM_ERROR

# Replace _check_gradients explosion threshold (around line 271):
# OLD:
#         if grad_health.max_norm > 1e10:
# NEW:
        if grad_health.max_norm > HealthThresholds.GRAD_NORM_EXPLOSION:

# Replace VitalSignsMonitor.__init__ defaults (around line 365-367):
# OLD:
#         loss_spike_threshold: float = 2.0,
#         stagnation_epochs: int = 20,
# NEW:
from esper.karn.constants import VitalSignsThresholds
# ...
        loss_spike_threshold: float = VitalSignsThresholds.LOSS_SPIKE_MULTIPLIER,
        stagnation_epochs: int = VitalSignsThresholds.STAGNATION_EPOCHS,
```

**Step 3: Update tui.py to use constants**

```python
# In src/esper/karn/tui.py, replace ThresholdConfig class (around line 45-70):
from esper.karn.constants import TUIThresholds

class ThresholdConfig:
    """Thresholds for red flag detection (delegates to constants)."""

    # Entropy thresholds
    entropy_critical: float = TUIThresholds.ENTROPY_CRITICAL
    entropy_warning: float = TUIThresholds.ENTROPY_WARNING
    entropy_max: float = TUIThresholds.ENTROPY_MAX

    # Clip fraction thresholds
    clip_critical: float = TUIThresholds.CLIP_CRITICAL
    clip_warning: float = TUIThresholds.CLIP_WARNING

    # Explained variance
    explained_var_critical: float = TUIThresholds.EXPLAINED_VAR_CRITICAL
    explained_var_warning: float = TUIThresholds.EXPLAINED_VAR_WARNING

    # Gradient norm
    grad_norm_critical: float = TUIThresholds.GRAD_NORM_CRITICAL
    grad_norm_warning: float = TUIThresholds.GRAD_NORM_WARNING

    # KL divergence
    kl_warning: float = TUIThresholds.KL_WARNING

    # Action distribution
    wait_warning: float = TUIThresholds.WAIT_DOMINANCE_WARNING
```

**Step 4: Update store.py DenseTraceTrigger defaults**

```python
# In src/esper/karn/store.py, update DenseTraceTrigger (find the dataclass):
from esper.karn.constants import AnomalyThresholds

@dataclass
class DenseTraceTrigger:
    """Configuration for dense trace triggering."""

    loss_spike_threshold: float = AnomalyThresholds.LOSS_SPIKE_MULTIPLIER
    accuracy_drop_threshold: float = AnomalyThresholds.ACCURACY_DROP_POINTS
    gradient_explosion: float = AnomalyThresholds.GRADIENT_EXPLOSION_MULTIPLIER
    # ... rest unchanged
```

**Step 5: Run tests to verify nothing broke**

Run: `PYTHONPATH=src uv run pytest tests/karn/ -v`
Expected: All existing tests PASS

**Step 6: Commit**

```bash
git add src/esper/karn/triggers.py src/esper/karn/health.py src/esper/karn/tui.py src/esper/karn/store.py
git commit -m "$(cat <<'EOF'
refactor(karn): migrate threshold consumers to use constants module

All threshold values now reference karn/constants.py instead of
hardcoded magic numbers. This enables global tuning and removes
scattered threshold definitions.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Fix Fragile Enum Comparisons in health.py

**Files:**
- Modify: `src/esper/karn/health.py:421-427`
- Test: `tests/karn/test_health.py` (add test)

**Step 1: Write the failing test**

```python
# Add to tests/karn/test_health.py (create if doesn't exist):
"""Tests for Karn health module."""

import pytest
from esper.karn.health import VitalSignsMonitor, VitalSigns
from esper.karn.store import TelemetryStore, EpochSnapshot, SlotSnapshot, HostSnapshot
from esper.leyline import SeedStage


class TestVitalSignsEnumComparisons:
    """Test that vital signs correctly identify seed stages."""

    def test_counts_germinated_as_active(self) -> None:
        """GERMINATED seeds should count toward total_seeds."""
        store = TelemetryStore()
        monitor = VitalSignsMonitor(store=store)

        # Create epoch with GERMINATED seed
        store.start_episode_minimal()
        snapshot = store.start_epoch(1)
        snapshot.slots["r0c0"] = SlotSnapshot(
            slot_id="r0c0",
            stage=SeedStage.GERMINATED,
        )
        snapshot.host.val_accuracy = 0.5
        store.commit_epoch()

        vitals = monitor.check_vitals()
        # GERMINATED should count as started (total_seeds)
        # but active_seeds counts TRAINING/BLENDING/PROBATIONARY/FOSSILIZED
        assert vitals.active_seeds == 0  # GERMINATED not yet active

    def test_counts_training_as_active(self) -> None:
        """TRAINING seeds should count as active."""
        store = TelemetryStore()
        monitor = VitalSignsMonitor(store=store)

        store.start_episode_minimal()
        snapshot = store.start_epoch(1)
        snapshot.slots["r0c0"] = SlotSnapshot(
            slot_id="r0c0",
            stage=SeedStage.TRAINING,
        )
        snapshot.host.val_accuracy = 0.5
        store.commit_epoch()

        vitals = monitor.check_vitals()
        assert vitals.active_seeds == 1

    def test_counts_culled_correctly(self) -> None:
        """CULLED seeds should be counted in failure rate."""
        store = TelemetryStore()
        monitor = VitalSignsMonitor(store=store)

        store.start_episode_minimal()
        snapshot = store.start_epoch(1)
        # One germinated (counts as total), one culled
        snapshot.slots["r0c0"] = SlotSnapshot(
            slot_id="r0c0",
            stage=SeedStage.GERMINATED,
        )
        snapshot.slots["r0c1"] = SlotSnapshot(
            slot_id="r0c1",
            stage=SeedStage.CULLED,
        )
        snapshot.host.val_accuracy = 0.5
        store.commit_epoch()

        vitals = monitor.check_vitals()
        # Failure rate should be 1 culled / 2 total = 0.5
        assert vitals.seed_failure_rate == 0.5
```

**Step 2: Run test to verify behavior (may pass or fail depending on current bugs)**

Run: `PYTHONPATH=src uv run pytest tests/karn/test_health.py::TestVitalSignsEnumComparisons -v`

**Step 3: Fix the fragile enum comparisons**

```python
# In src/esper/karn/health.py, replace lines 417-430 in check_vitals():
# Import at top:
from esper.leyline.stages import is_active_stage, is_failure_stage

# OLD (fragile):
#         for slot in latest.slots.values():
#             if slot.stage.value >= 2:  # GERMINATED or beyond
#                 total_seeds += 1
#             if slot.stage.value == 7:  # CULLED
#                 culled_seeds += 1
#             if slot.stage.value in (2, 3, 4, 5):  # Active states
#                 active += 1

# NEW (robust):
        for slot in latest.slots.values():
            # Count seeds that have been germinated (past DORMANT)
            if slot.stage != SeedStage.DORMANT and slot.stage != SeedStage.UNKNOWN:
                total_seeds += 1
            # Count culled seeds for failure rate
            if slot.stage == SeedStage.CULLED:
                culled_seeds += 1
            # Count active seeds (contributing to forward pass)
            if is_active_stage(slot.stage):
                active += 1
```

**Step 4: Run test to verify fix**

Run: `PYTHONPATH=src uv run pytest tests/karn/test_health.py::TestVitalSignsEnumComparisons -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/health.py tests/karn/test_health.py
git commit -m "$(cat <<'EOF'
fix(karn): replace fragile enum value comparisons with explicit checks

Uses is_active_stage() from leyline and explicit SeedStage comparisons
instead of magic integer values. This prevents breakage if enum values
are reordered.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Create `karn/serialization.py` for Shared Event Serialization

**Files:**
- Create: `src/esper/karn/serialization.py`
- Test: `tests/karn/test_serialization.py`

**Step 1: Write the failing test**

```python
# tests/karn/test_serialization.py
"""Tests for Karn event serialization."""

import json
from datetime import datetime
from enum import Enum

import pytest
from esper.karn.serialization import serialize_event
from esper.karn.contracts import TelemetryEventLike


class MockEventType(Enum):
    TEST_EVENT = "test"


class MockEvent:
    """Mock event implementing TelemetryEventLike protocol."""

    def __init__(
        self,
        event_type: MockEventType | str = MockEventType.TEST_EVENT,
        timestamp: datetime | None = None,
    ):
        self._event_type = event_type
        self._timestamp = timestamp or datetime(2025, 1, 1, 12, 0, 0)

    @property
    def event_type(self) -> MockEventType | str:
        return self._event_type

    @property
    def timestamp(self) -> datetime:
        return self._timestamp

    @property
    def data(self) -> dict | None:
        return {"test": "value"}

    @property
    def epoch(self) -> int | None:
        return 5

    @property
    def seed_id(self) -> str | None:
        return "seed_123"

    @property
    def slot_id(self) -> str | None:
        return "r0c0"

    @property
    def severity(self) -> str | None:
        return "info"

    @property
    def message(self) -> str | None:
        return "test message"


class TestSerializeEvent:
    """Test event serialization."""

    def test_serializes_enum_event_type(self) -> None:
        """Enum event_type should be converted to string name."""
        event = MockEvent(event_type=MockEventType.TEST_EVENT)
        result = serialize_event(event)
        data = json.loads(result)
        assert data["event_type"] == "TEST_EVENT"

    def test_serializes_string_event_type(self) -> None:
        """String event_type should pass through."""
        event = MockEvent(event_type="STRING_TYPE")
        result = serialize_event(event)
        data = json.loads(result)
        assert data["event_type"] == "STRING_TYPE"

    def test_serializes_datetime(self) -> None:
        """Datetime should be ISO formatted."""
        ts = datetime(2025, 6, 15, 10, 30, 0)
        event = MockEvent(timestamp=ts)
        result = serialize_event(event)
        data = json.loads(result)
        assert data["timestamp"] == "2025-06-15T10:30:00"

    def test_includes_all_fields(self) -> None:
        """All TelemetryEventLike fields should be present."""
        event = MockEvent()
        result = serialize_event(event)
        data = json.loads(result)

        assert "event_type" in data
        assert "timestamp" in data
        assert "data" in data
        assert "epoch" in data
        assert "seed_id" in data
        assert "slot_id" in data
        assert "severity" in data
        assert "message" in data

    def test_returns_valid_json(self) -> None:
        """Result should be valid JSON string."""
        event = MockEvent()
        result = serialize_event(event)
        # Should not raise
        parsed = json.loads(result)
        assert isinstance(parsed, dict)
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/test_serialization.py -v`
Expected: FAIL with "ModuleNotFoundError" or "ImportError"

**Step 3: Write minimal implementation**

```python
# src/esper/karn/serialization.py
"""Karn Serialization - Shared event serialization for output backends.

Provides a single source of truth for converting TelemetryEvent-like
objects to JSON strings. Used by WebSocketOutput and DashboardServer.

Usage:
    from esper.karn.serialization import serialize_event

    json_str = serialize_event(event)
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from esper.karn.contracts import TelemetryEventLike


def serialize_event(event: "TelemetryEventLike") -> str:
    """Serialize a TelemetryEventLike object to JSON string.

    Handles:
    - Enum event_type → string name
    - datetime timestamp → ISO format string
    - All standard TelemetryEventLike protocol fields

    Args:
        event: Any object implementing TelemetryEventLike protocol

    Returns:
        JSON string representation of the event
    """
    # Extract event_type (handle both enum and string)
    # hasattr AUTHORIZED by John on 2025-12-17 15:00:00 UTC
    # Justification: Serialization - handle both enum and string event_type values
    event_type = event.event_type
    if hasattr(event_type, "name"):
        event_type = event_type.name

    # Extract timestamp (handle datetime objects)
    # hasattr AUTHORIZED by John on 2025-12-17 15:00:00 UTC
    # Justification: Serialization - safely handle datetime objects
    timestamp = event.timestamp
    if hasattr(timestamp, "isoformat"):
        timestamp = timestamp.isoformat()

    data = {
        "event_type": event_type,
        "timestamp": timestamp,
        "data": event.data,
        "epoch": event.epoch,
        "seed_id": event.seed_id,
        "slot_id": event.slot_id,
        "severity": event.severity,
        "message": event.message,
    }

    return json.dumps(data, default=str)
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/test_serialization.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/serialization.py tests/karn/test_serialization.py
git commit -m "$(cat <<'EOF'
feat(karn): add shared serialization module for event JSON conversion

Extracts event serialization logic into a single module that can be
shared by WebSocketOutput and DashboardServer, eliminating duplication.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Deduplicate Serialization in Output Backends

**Files:**
- Modify: `src/esper/karn/websocket_output.py`
- Modify: `src/esper/karn/integrated_dashboard.py`

**Step 1: Update websocket_output.py**

```python
# In src/esper/karn/websocket_output.py:

# Remove the local _serialize_event function (lines 37-56)
# Replace with import:
from esper.karn.serialization import serialize_event as _serialize_event
```

**Step 2: Update integrated_dashboard.py**

```python
# In src/esper/karn/integrated_dashboard.py:

# Remove the local _serialize_event function (lines 41-72)
# Replace with import:
from esper.karn.serialization import serialize_event as _serialize_event
```

**Step 3: Run tests to verify nothing broke**

Run: `PYTHONPATH=src uv run pytest tests/karn/ tests/integration/ -v -k "websocket or dashboard or telemetry"`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add src/esper/karn/websocket_output.py src/esper/karn/integrated_dashboard.py
git commit -m "$(cat <<'EOF'
refactor(karn): deduplicate event serialization in output backends

Both WebSocketOutput and DashboardServer now use the shared
serialize_event() from karn/serialization.py instead of duplicating
the logic locally.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Remove Karn→Nissa Dependency in health.py

**Files:**
- Modify: `src/esper/karn/health.py`
- Test: `tests/karn/test_health.py`

**Step 1: Write the failing test**

```python
# Add to tests/karn/test_health.py:

class TestHealthMonitorCallback:
    """Test HealthMonitor with emit callback injection."""

    def test_emits_memory_warning_via_callback(self) -> None:
        """Memory warning should be emitted via injected callback."""
        emitted_events: list = []

        def capture_emit(event):
            emitted_events.append(event)

        monitor = HealthMonitor(
            emit_callback=capture_emit,
            memory_warning_threshold=0.5,  # Low threshold to trigger easily
        )

        # Trigger memory warning check with high utilization
        warned = monitor._check_memory_and_warn(
            gpu_utilization=0.95,
            gpu_allocated_gb=10.0,
            gpu_total_gb=12.0,
        )

        assert warned is True
        assert len(emitted_events) == 1
        assert emitted_events[0].event_type.name == "MEMORY_WARNING"

    def test_no_emit_without_callback(self) -> None:
        """Without callback, no emission should occur (no crash)."""
        monitor = HealthMonitor(
            emit_callback=None,
            memory_warning_threshold=0.5,
        )

        # Should not crash even without callback
        warned = monitor._check_memory_and_warn(
            gpu_utilization=0.95,
            gpu_allocated_gb=10.0,
            gpu_total_gb=12.0,
        )

        # Still returns True (warning condition met) but no emission
        assert warned is True
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/test_health.py::TestHealthMonitorCallback -v`
Expected: FAIL (emit_callback parameter doesn't exist yet)

**Step 3: Modify health.py to use callback injection**

```python
# In src/esper/karn/health.py:

# REMOVE this import (line 26):
# from esper.nissa import get_hub

# ADD at top (after other imports):
from typing import Callable

# MODIFY HealthMonitor.__init__ (around line 134):
    def __init__(
        self,
        store: "TelemetryStore | None" = None,
        emit_callback: Callable[["TelemetryEvent"], None] | None = None,  # NEW
        gpu_warning_threshold: float = HealthThresholds.GPU_UTILIZATION_WARNING,
        grad_norm_warning: float = HealthThresholds.GRAD_NORM_WARNING,
        grad_norm_error: float = HealthThresholds.GRAD_NORM_ERROR,
        memory_warning_threshold: float = HealthThresholds.MEMORY_WARNING_THRESHOLD,
        memory_warning_cooldown: float = HealthThresholds.MEMORY_WARNING_COOLDOWN_SECONDS,
    ):
        self.store = store
        self._emit_callback = emit_callback  # NEW
        self.gpu_warning_threshold = gpu_warning_threshold
        # ... rest unchanged

# MODIFY _check_memory_and_warn method (around line 179):
        self._last_memory_warning = now
        # OLD:
        # hub = get_hub()
        # if hub is not None:
        #     hub.emit(TelemetryEvent(...))
        # NEW:
        if self._emit_callback is not None:
            self._emit_callback(TelemetryEvent(
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

Run: `PYTHONPATH=src uv run pytest tests/karn/test_health.py::TestHealthMonitorCallback -v`
Expected: PASS

**Step 5: Run all health tests**

Run: `PYTHONPATH=src uv run pytest tests/karn/test_health.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/esper/karn/health.py tests/karn/test_health.py
git commit -m "$(cat <<'EOF'
refactor(karn): remove Nissa dependency from health.py via callback injection

HealthMonitor now accepts an optional emit_callback parameter instead
of importing get_hub() from Nissa. This removes the architectural
inversion where Karn (consumer) depended on Nissa (producer).

Callers who need telemetry emission can pass:
  emit_callback=get_hub().emit

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Remove Karn→Nissa Dependency in counterfactual.py

**Files:**
- Modify: `src/esper/karn/counterfactual.py`
- Test: `tests/karn/test_counterfactual.py`

**Step 1: Write the failing test**

```python
# Add to tests/karn/test_counterfactual.py (create if doesn't exist):
"""Tests for Karn counterfactual engine."""

import pytest
from esper.karn.counterfactual import CounterfactualEngine, CounterfactualConfig


class TestCounterfactualEngineCallback:
    """Test CounterfactualEngine with emit callback injection."""

    def test_emits_shapley_via_callback(self) -> None:
        """Shapley computation should emit via injected callback."""
        emitted_events: list = []

        def capture_emit(event):
            emitted_events.append(event)

        engine = CounterfactualEngine(
            emit_callback=capture_emit,
        )

        # Create a simple matrix for Shapley computation
        slot_ids = ["r0c0"]

        def eval_fn(alphas):
            # Simple: accuracy = alpha value
            return (0.5, alphas.get("r0c0", 0.0))

        matrix = engine.compute_matrix(slot_ids, eval_fn)
        matrix.epoch = 10
        shapley = engine.compute_shapley_values(matrix)

        # Should have emitted ANALYTICS_SNAPSHOT
        assert len(emitted_events) == 1
        assert emitted_events[0].event_type.name == "ANALYTICS_SNAPSHOT"
        assert emitted_events[0].data["kind"] == "shapley_computed"

    def test_no_emit_without_callback(self) -> None:
        """Without callback, Shapley still works but no emission."""
        engine = CounterfactualEngine(
            emit_callback=None,
        )

        slot_ids = ["r0c0"]

        def eval_fn(alphas):
            return (0.5, alphas.get("r0c0", 0.0))

        matrix = engine.compute_matrix(slot_ids, eval_fn)
        shapley = engine.compute_shapley_values(matrix)

        # Should work without crash
        assert "r0c0" in shapley
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/test_counterfactual.py::TestCounterfactualEngineCallback -v`
Expected: FAIL (emit_callback parameter doesn't exist yet)

**Step 3: Modify counterfactual.py to use callback injection**

```python
# In src/esper/karn/counterfactual.py:

# REMOVE this import (line 24):
# from esper.nissa import get_hub

# ADD at top (after other imports):
from typing import Callable

# MODIFY CounterfactualEngine.__init__ (around line 183):
    def __init__(
        self,
        config: CounterfactualConfig | None = None,
        emit_callback: Callable[["TelemetryEvent"], None] | None = None,  # CHANGED from emit_telemetry: bool
    ):
        self.config = config or CounterfactualConfig()
        self._emit_callback = emit_callback  # CHANGED

# MODIFY compute_shapley_values (around line 379-400):
        # OLD:
        # if self.emit_telemetry:
        #     hub = get_hub()
        #     if hub is not None:
        #         hub.emit(TelemetryEvent(...))

        # NEW:
        if self._emit_callback is not None:
            shapley_dict = {
                slot_id: {
                    "mean": estimate.mean,
                    "std": estimate.std,
                    "n_samples": estimate.n_samples,
                }
                for slot_id, estimate in result.items()
            }
            self._emit_callback(TelemetryEvent(
                event_type=TelemetryEventType.ANALYTICS_SNAPSHOT,
                data={
                    "kind": "shapley_computed",
                    "shapley_values": shapley_dict,
                    "num_slots": len(result),
                    "epoch": matrix.epoch,
                }
            ))

        return result
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/test_counterfactual.py::TestCounterfactualEngineCallback -v`
Expected: PASS

**Step 5: Update any callers that used emit_telemetry=True**

Search for usages:
```bash
grep -r "emit_telemetry" src/
```

If found, update to use `emit_callback=get_hub().emit` instead.

**Step 6: Run all counterfactual tests**

Run: `PYTHONPATH=src uv run pytest tests/karn/test_counterfactual.py -v`
Expected: PASS

**Step 7: Commit**

```bash
git add src/esper/karn/counterfactual.py tests/karn/test_counterfactual.py
git commit -m "$(cat <<'EOF'
refactor(karn): remove Nissa dependency from counterfactual.py via callback

CounterfactualEngine now accepts an optional emit_callback parameter
instead of emit_telemetry bool + get_hub() import. This removes the
architectural inversion where Karn depended on Nissa.

Callers who need telemetry emission can pass:
  emit_callback=get_hub().emit

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Final Verification

**Run full test suite:**

```bash
PYTHONPATH=src uv run pytest tests/karn/ tests/nissa/ tests/integration/ -v
```

**Verify no Nissa imports in Karn (except contracts):**

```bash
grep -r "from esper.nissa" src/esper/karn/
# Should return empty or only TYPE_CHECKING imports
```

---

## Summary

After completing all tasks:

1. ✅ Centralized thresholds in `karn/constants.py`
2. ✅ All consumers use constants instead of magic numbers
3. ✅ Fixed fragile enum comparisons using `is_active_stage()`
4. ✅ Shared serialization in `karn/serialization.py`
5. ✅ Deduplicated serialization in output backends
6. ✅ Removed Karn→Nissa dependency in health.py
7. ✅ Removed Karn→Nissa dependency in counterfactual.py

**Deferred to follow-up plan:**
- TUI decomposition (1200 lines → display/state/backend modules)
- ConsoleOutput._emit_summary() refactoring (visitor pattern)
