# Overwatch Stage 0: Foundation + Replay Infrastructure

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Establish the data layer for Overwatch TUI: snapshot schema, serialization, and JSONL replay infrastructure.

**Architecture:** Dataclass-based schema (`TuiSnapshot`) with nested components for each UI region. Uses existing leyline types (`SeedStage`, `format_slot_id`) for consistency. JSONL format for replay files (one snapshot per line).

**Tech Stack:** Python 3.11, dataclasses, json, pytest, existing leyline contracts.

---

## Prerequisites

- Branch: `feat/overwatch-textual-ui` (already created)
- Read stage plan: `docs/plans/2025-12-18-overwatch-stage-plan.md`

---

## Task 1: Create Overwatch Package Structure

**Files:**
- Create: `src/esper/karn/overwatch/__init__.py`
- Create: `tests/karn/overwatch/__init__.py`

**Step 1: Create package directories and init files**

```python
# src/esper/karn/overwatch/__init__.py
"""Overwatch - Textual TUI for Esper training monitoring.

Provides real-time visibility into training environments, seed lifecycle,
and Tamiyo decision-making.
"""

__all__: list[str] = []
```

```python
# tests/karn/overwatch/__init__.py
"""Tests for Overwatch TUI."""
```

**Step 2: Verify package imports**

Run: `PYTHONPATH=src python -c "from esper.karn import overwatch; print('OK')"`

Expected: `OK`

**Step 3: Commit**

```bash
git add src/esper/karn/overwatch/__init__.py tests/karn/overwatch/__init__.py
git commit -m "chore(overwatch): create package structure"
```

---

## Task 2: Define SlotChipState Schema

**Files:**
- Create: `src/esper/karn/overwatch/schema.py`
- Create: `tests/karn/overwatch/test_schema.py`

**Step 1: Write failing test for SlotChipState**

```python
# tests/karn/overwatch/test_schema.py
"""Tests for Overwatch TUI snapshot schema."""

from __future__ import annotations

import json
import pytest


class TestSlotChipState:
    """Tests for SlotChipState dataclass."""

    def test_slot_chip_state_creation(self) -> None:
        """SlotChipState can be created with required fields."""
        from esper.karn.overwatch.schema import SlotChipState

        chip = SlotChipState(
            slot_id="r0c1",
            stage="TRAINING",
            blueprint_id="conv_light",
            alpha=0.45,
        )

        assert chip.slot_id == "r0c1"
        assert chip.stage == "TRAINING"
        assert chip.blueprint_id == "conv_light"
        assert chip.alpha == 0.45
        # Defaults
        assert chip.epochs_in_stage == 0
        assert chip.epochs_total == 0
        assert chip.gate_last is None
        assert chip.gate_passed is None

    def test_slot_chip_state_to_dict(self) -> None:
        """SlotChipState serializes to dict."""
        from esper.karn.overwatch.schema import SlotChipState

        chip = SlotChipState(
            slot_id="r0c1",
            stage="BLENDING",
            blueprint_id="mlp_narrow",
            alpha=0.78,
            epochs_in_stage=5,
            epochs_total=10,
            gate_last="G2",
            gate_passed=True,
        )

        d = chip.to_dict()

        assert d["slot_id"] == "r0c1"
        assert d["stage"] == "BLENDING"
        assert d["alpha"] == 0.78
        assert d["gate_last"] == "G2"
        assert d["gate_passed"] is True

    def test_slot_chip_state_from_dict(self) -> None:
        """SlotChipState deserializes from dict."""
        from esper.karn.overwatch.schema import SlotChipState

        d = {
            "slot_id": "r1c0",
            "stage": "FOSSILIZED",
            "blueprint_id": "conv_light",
            "alpha": 1.0,
            "epochs_in_stage": 0,
            "epochs_total": 42,
            "gate_last": "G3",
            "gate_passed": True,
        }

        chip = SlotChipState.from_dict(d)

        assert chip.slot_id == "r1c0"
        assert chip.stage == "FOSSILIZED"
        assert chip.alpha == 1.0

    def test_slot_chip_state_json_roundtrip(self) -> None:
        """SlotChipState survives JSON serialization."""
        from esper.karn.overwatch.schema import SlotChipState

        chip = SlotChipState(
            slot_id="r0c0",
            stage="GERMINATED",
            blueprint_id="test",
            alpha=0.1,
        )

        json_str = json.dumps(chip.to_dict())
        restored = SlotChipState.from_dict(json.loads(json_str))

        assert restored.slot_id == chip.slot_id
        assert restored.stage == chip.stage
        assert restored.alpha == chip.alpha
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_schema.py::TestSlotChipState -v`

Expected: FAIL with `ModuleNotFoundError` or `ImportError`

**Step 3: Implement SlotChipState**

```python
# src/esper/karn/overwatch/schema.py
"""Overwatch TUI Snapshot Schema.

Defines the data structures that flow from telemetry aggregator to UI renderer.
All schemas are JSON-serializable for replay support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SlotChipState:
    """State of a single slot for UI rendering.

    Represents one slot chip in the Flight Board, showing seed lifecycle
    stage, blending progress, and gate status.
    """

    # Identity
    slot_id: str  # Canonical format: "r0c1" (row 0, column 1)
    stage: str  # SeedStage name: DORMANT, GERMINATED, TRAINING, etc.
    blueprint_id: str  # Blueprint used for this seed (empty if dormant)
    alpha: float  # Blend weight 0.0-1.0

    # Progress
    epochs_in_stage: int = 0
    epochs_total: int = 0  # Total epochs since germination

    # Gate status
    gate_last: str | None = None  # Last gate evaluated: G0, G1, G2, G3
    gate_passed: bool | None = None  # Did the gate pass?

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "slot_id": self.slot_id,
            "stage": self.stage,
            "blueprint_id": self.blueprint_id,
            "alpha": self.alpha,
            "epochs_in_stage": self.epochs_in_stage,
            "epochs_total": self.epochs_total,
            "gate_last": self.gate_last,
            "gate_passed": self.gate_passed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SlotChipState:
        """Reconstruct from dict."""
        return cls(
            slot_id=data["slot_id"],
            stage=data["stage"],
            blueprint_id=data["blueprint_id"],
            alpha=data["alpha"],
            epochs_in_stage=data.get("epochs_in_stage", 0),
            epochs_total=data.get("epochs_total", 0),
            gate_last=data.get("gate_last"),
            gate_passed=data.get("gate_passed"),
        )
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_schema.py::TestSlotChipState -v`

Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add src/esper/karn/overwatch/schema.py tests/karn/overwatch/test_schema.py
git commit -m "feat(overwatch): add SlotChipState schema"
```

---

## Task 3: Define EnvSummary Schema

**Files:**
- Modify: `src/esper/karn/overwatch/schema.py`
- Modify: `tests/karn/overwatch/test_schema.py`

**Step 1: Write failing test for EnvSummary**

```python
# tests/karn/overwatch/test_schema.py (append to file)

class TestEnvSummary:
    """Tests for EnvSummary dataclass."""

    def test_env_summary_creation(self) -> None:
        """EnvSummary can be created with required fields."""
        from esper.karn.overwatch.schema import EnvSummary, SlotChipState

        env = EnvSummary(
            env_id=0,
            device_id=0,
            status="OK",
        )

        assert env.env_id == 0
        assert env.device_id == 0
        assert env.status == "OK"
        assert env.slots == {}
        assert env.anomaly_score == 0.0
        assert env.anomaly_reasons == []

    def test_env_summary_with_slots(self) -> None:
        """EnvSummary contains slot states."""
        from esper.karn.overwatch.schema import EnvSummary, SlotChipState

        chip = SlotChipState(
            slot_id="r0c1",
            stage="TRAINING",
            blueprint_id="conv_light",
            alpha=0.3,
        )

        env = EnvSummary(
            env_id=3,
            device_id=1,
            status="WARN",
            slots={"r0c1": chip},
            anomaly_score=0.65,
            anomaly_reasons=["High gradient ratio (3.2x)"],
        )

        assert "r0c1" in env.slots
        assert env.slots["r0c1"].stage == "TRAINING"
        assert env.anomaly_score == 0.65

    def test_env_summary_to_dict(self) -> None:
        """EnvSummary serializes to dict with nested slots."""
        from esper.karn.overwatch.schema import EnvSummary, SlotChipState

        chip = SlotChipState(
            slot_id="r0c1",
            stage="BLENDING",
            blueprint_id="mlp",
            alpha=0.7,
        )

        env = EnvSummary(
            env_id=2,
            device_id=0,
            status="OK",
            throughput_fps=98.5,
            slots={"r0c1": chip},
        )

        d = env.to_dict()

        assert d["env_id"] == 2
        assert d["throughput_fps"] == 98.5
        assert "r0c1" in d["slots"]
        assert d["slots"]["r0c1"]["stage"] == "BLENDING"

    def test_env_summary_from_dict(self) -> None:
        """EnvSummary deserializes from dict."""
        from esper.karn.overwatch.schema import EnvSummary

        d = {
            "env_id": 1,
            "device_id": 0,
            "status": "CRIT",
            "throughput_fps": 45.0,
            "reward_last": -0.5,
            "slots": {
                "r0c0": {
                    "slot_id": "r0c0",
                    "stage": "CULLED",
                    "blueprint_id": "bad_seed",
                    "alpha": 0.0,
                }
            },
            "anomaly_score": 0.85,
            "anomaly_reasons": ["Throughput 55% below baseline", "Negative reward"],
        }

        env = EnvSummary.from_dict(d)

        assert env.env_id == 1
        assert env.status == "CRIT"
        assert env.anomaly_score == 0.85
        assert len(env.anomaly_reasons) == 2
        assert env.slots["r0c0"].stage == "CULLED"

    def test_env_summary_json_roundtrip(self) -> None:
        """EnvSummary survives JSON serialization."""
        from esper.karn.overwatch.schema import EnvSummary, SlotChipState

        env = EnvSummary(
            env_id=0,
            device_id=0,
            status="OK",
            slots={
                "r0c1": SlotChipState("r0c1", "TRAINING", "conv", 0.5)
            },
        )

        json_str = json.dumps(env.to_dict())
        restored = EnvSummary.from_dict(json.loads(json_str))

        assert restored.env_id == env.env_id
        assert restored.slots["r0c1"].alpha == 0.5
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_schema.py::TestEnvSummary -v`

Expected: FAIL with `ImportError: cannot import name 'EnvSummary'`

**Step 3: Implement EnvSummary**

```python
# src/esper/karn/overwatch/schema.py (append after SlotChipState)

@dataclass
class EnvSummary:
    """Summary of a single training environment for Flight Board.

    Contains all information needed to render one row in the Flight Board:
    environment identity, status, throughput, slots, and anomaly info.
    """

    # Identity
    env_id: int
    device_id: int  # GPU device index
    status: str  # OK, INFO, WARN, CRIT

    # Throughput
    throughput_fps: float = 0.0
    step_time_ms: float = 0.0

    # Metrics
    reward_last: float = 0.0
    task_metric: float = 0.0  # Task-specific metric (e.g., accuracy)
    task_metric_delta: float = 0.0

    # Slots (keyed by slot_id like "r0c1")
    slots: dict[str, SlotChipState] = field(default_factory=dict)

    # Anomaly detection
    anomaly_score: float = 0.0  # 0.0-1.0, higher = more anomalous
    anomaly_reasons: list[str] = field(default_factory=list)

    # Staleness
    last_update_ts: float = 0.0  # Unix timestamp of last telemetry

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "env_id": self.env_id,
            "device_id": self.device_id,
            "status": self.status,
            "throughput_fps": self.throughput_fps,
            "step_time_ms": self.step_time_ms,
            "reward_last": self.reward_last,
            "task_metric": self.task_metric,
            "task_metric_delta": self.task_metric_delta,
            "slots": {k: v.to_dict() for k, v in self.slots.items()},
            "anomaly_score": self.anomaly_score,
            "anomaly_reasons": self.anomaly_reasons,
            "last_update_ts": self.last_update_ts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnvSummary:
        """Reconstruct from dict."""
        slots = {
            k: SlotChipState.from_dict(v)
            for k, v in data.get("slots", {}).items()
        }
        return cls(
            env_id=data["env_id"],
            device_id=data["device_id"],
            status=data["status"],
            throughput_fps=data.get("throughput_fps", 0.0),
            step_time_ms=data.get("step_time_ms", 0.0),
            reward_last=data.get("reward_last", 0.0),
            task_metric=data.get("task_metric", 0.0),
            task_metric_delta=data.get("task_metric_delta", 0.0),
            slots=slots,
            anomaly_score=data.get("anomaly_score", 0.0),
            anomaly_reasons=data.get("anomaly_reasons", []),
            last_update_ts=data.get("last_update_ts", 0.0),
        )
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_schema.py::TestEnvSummary -v`

Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/esper/karn/overwatch/schema.py tests/karn/overwatch/test_schema.py
git commit -m "feat(overwatch): add EnvSummary schema"
```

---

## Task 4: Define TamiyoState Schema

**Files:**
- Modify: `src/esper/karn/overwatch/schema.py`
- Modify: `tests/karn/overwatch/test_schema.py`

**Step 1: Write failing test for TamiyoState**

```python
# tests/karn/overwatch/test_schema.py (append to file)

class TestTamiyoState:
    """Tests for TamiyoState dataclass."""

    def test_tamiyo_state_creation(self) -> None:
        """TamiyoState can be created with defaults."""
        from esper.karn.overwatch.schema import TamiyoState

        state = TamiyoState()

        assert state.action_counts == {}
        assert state.recent_actions == []
        assert state.confidence_mean == 0.0
        assert state.exploration_pct == 0.0

    def test_tamiyo_state_with_data(self) -> None:
        """TamiyoState stores action distribution and PPO vitals."""
        from esper.karn.overwatch.schema import TamiyoState

        state = TamiyoState(
            action_counts={"GERMINATE": 34, "BLEND": 28, "CULL": 12, "WAIT": 26},
            recent_actions=["G", "B", "B", "W", "G"],
            confidence_mean=0.73,
            confidence_min=0.42,
            confidence_max=0.94,
            exploration_pct=31.0,
            kl_divergence=0.019,
            entropy=1.24,
            clip_fraction=0.048,
            explained_variance=0.42,
            learning_rate=3e-4,
        )

        assert state.action_counts["GERMINATE"] == 34
        assert len(state.recent_actions) == 5
        assert state.kl_divergence == 0.019

    def test_tamiyo_state_to_dict(self) -> None:
        """TamiyoState serializes to dict."""
        from esper.karn.overwatch.schema import TamiyoState

        state = TamiyoState(
            action_counts={"GERMINATE": 10},
            recent_actions=["G", "G"],
            confidence_mean=0.8,
            kl_divergence=0.02,
        )

        d = state.to_dict()

        assert d["action_counts"] == {"GERMINATE": 10}
        assert d["recent_actions"] == ["G", "G"]
        assert d["kl_divergence"] == 0.02

    def test_tamiyo_state_from_dict(self) -> None:
        """TamiyoState deserializes from dict."""
        from esper.karn.overwatch.schema import TamiyoState

        d = {
            "action_counts": {"BLEND": 5, "CULL": 3},
            "recent_actions": ["B", "C"],
            "confidence_mean": 0.65,
            "confidence_min": 0.4,
            "confidence_max": 0.9,
            "exploration_pct": 25.0,
            "kl_divergence": 0.015,
            "entropy": 1.5,
            "clip_fraction": 0.03,
            "explained_variance": 0.5,
            "learning_rate": 0.0003,
        }

        state = TamiyoState.from_dict(d)

        assert state.action_counts["BLEND"] == 5
        assert state.entropy == 1.5

    def test_tamiyo_state_json_roundtrip(self) -> None:
        """TamiyoState survives JSON serialization."""
        from esper.karn.overwatch.schema import TamiyoState

        state = TamiyoState(
            action_counts={"WAIT": 100},
            kl_divergence=0.01,
        )

        json_str = json.dumps(state.to_dict())
        restored = TamiyoState.from_dict(json.loads(json_str))

        assert restored.action_counts == state.action_counts
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_schema.py::TestTamiyoState -v`

Expected: FAIL with `ImportError`

**Step 3: Implement TamiyoState**

```python
# src/esper/karn/overwatch/schema.py (append after EnvSummary)

@dataclass
class TamiyoState:
    """Tamiyo agent state for the Tamiyo Strip and Detail Panel.

    Contains PPO vitals, action distribution, and decision confidence
    for rendering agent brain telemetry.
    """

    # Action distribution (counts over recent window)
    action_counts: dict[str, int] = field(default_factory=dict)
    recent_actions: list[str] = field(default_factory=list)  # Last N actions as codes

    # Confidence (from policy logprobs)
    confidence_mean: float = 0.0
    confidence_min: float = 0.0
    confidence_max: float = 0.0
    confidence_history: list[float] = field(default_factory=list)  # For sparkline

    # Exploration
    exploration_pct: float = 0.0  # Entropy as % of max entropy

    # PPO vitals
    kl_divergence: float = 0.0
    entropy: float = 0.0
    clip_fraction: float = 0.0
    explained_variance: float = 0.0
    grad_norm: float = 0.0
    learning_rate: float = 0.0

    # Trends (for arrows: positive = rising, negative = falling)
    kl_trend: float = 0.0
    entropy_trend: float = 0.0
    ev_trend: float = 0.0

    # Status flags
    entropy_collapsed: bool = False
    ev_warning: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "action_counts": self.action_counts,
            "recent_actions": self.recent_actions,
            "confidence_mean": self.confidence_mean,
            "confidence_min": self.confidence_min,
            "confidence_max": self.confidence_max,
            "confidence_history": self.confidence_history,
            "exploration_pct": self.exploration_pct,
            "kl_divergence": self.kl_divergence,
            "entropy": self.entropy,
            "clip_fraction": self.clip_fraction,
            "explained_variance": self.explained_variance,
            "grad_norm": self.grad_norm,
            "learning_rate": self.learning_rate,
            "kl_trend": self.kl_trend,
            "entropy_trend": self.entropy_trend,
            "ev_trend": self.ev_trend,
            "entropy_collapsed": self.entropy_collapsed,
            "ev_warning": self.ev_warning,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TamiyoState:
        """Reconstruct from dict."""
        return cls(
            action_counts=data.get("action_counts", {}),
            recent_actions=data.get("recent_actions", []),
            confidence_mean=data.get("confidence_mean", 0.0),
            confidence_min=data.get("confidence_min", 0.0),
            confidence_max=data.get("confidence_max", 0.0),
            confidence_history=data.get("confidence_history", []),
            exploration_pct=data.get("exploration_pct", 0.0),
            kl_divergence=data.get("kl_divergence", 0.0),
            entropy=data.get("entropy", 0.0),
            clip_fraction=data.get("clip_fraction", 0.0),
            explained_variance=data.get("explained_variance", 0.0),
            grad_norm=data.get("grad_norm", 0.0),
            learning_rate=data.get("learning_rate", 0.0),
            kl_trend=data.get("kl_trend", 0.0),
            entropy_trend=data.get("entropy_trend", 0.0),
            ev_trend=data.get("ev_trend", 0.0),
            entropy_collapsed=data.get("entropy_collapsed", False),
            ev_warning=data.get("ev_warning", False),
        )
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_schema.py::TestTamiyoState -v`

Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/esper/karn/overwatch/schema.py tests/karn/overwatch/test_schema.py
git commit -m "feat(overwatch): add TamiyoState schema"
```

---

## Task 5: Define ConnectionStatus and SystemVitals

**Files:**
- Modify: `src/esper/karn/overwatch/schema.py`
- Modify: `tests/karn/overwatch/test_schema.py`

**Step 1: Write failing tests**

```python
# tests/karn/overwatch/test_schema.py (append to file)

class TestConnectionStatus:
    """Tests for ConnectionStatus dataclass."""

    def test_connection_status_live(self) -> None:
        """ConnectionStatus reports Live when fresh."""
        from esper.karn.overwatch.schema import ConnectionStatus

        status = ConnectionStatus(
            connected=True,
            last_event_ts=1000.0,
            staleness_s=0.5,
        )

        assert status.connected is True
        assert "Live" in status.display_text

    def test_connection_status_stale(self) -> None:
        """ConnectionStatus reports Stale when old."""
        from esper.karn.overwatch.schema import ConnectionStatus

        status = ConnectionStatus(
            connected=True,
            last_event_ts=1000.0,
            staleness_s=8.0,
        )

        assert "Stale" in status.display_text
        assert "8" in status.display_text

    def test_connection_status_disconnected(self) -> None:
        """ConnectionStatus reports Disconnected when not connected."""
        from esper.karn.overwatch.schema import ConnectionStatus

        status = ConnectionStatus(
            connected=False,
            last_event_ts=0.0,
            staleness_s=30.0,
        )

        assert "Disconnected" in status.display_text

    def test_connection_status_json_roundtrip(self) -> None:
        """ConnectionStatus survives JSON serialization."""
        from esper.karn.overwatch.schema import ConnectionStatus

        status = ConnectionStatus(True, 1234.5, 2.0)

        json_str = json.dumps(status.to_dict())
        restored = ConnectionStatus.from_dict(json.loads(json_str))

        assert restored.connected == status.connected
        assert restored.staleness_s == status.staleness_s


class TestDeviceVitals:
    """Tests for DeviceVitals dataclass."""

    def test_device_vitals_creation(self) -> None:
        """DeviceVitals stores GPU metrics."""
        from esper.karn.overwatch.schema import DeviceVitals

        vitals = DeviceVitals(
            device_id=0,
            name="GPU 0",
            utilization_pct=94.0,
            memory_used_gb=11.2,
            memory_total_gb=12.0,
            temperature_c=72,
        )

        assert vitals.device_id == 0
        assert vitals.utilization_pct == 94.0
        assert vitals.memory_pct == pytest.approx(93.33, rel=0.01)

    def test_device_vitals_json_roundtrip(self) -> None:
        """DeviceVitals survives JSON serialization."""
        from esper.karn.overwatch.schema import DeviceVitals

        vitals = DeviceVitals(0, "GPU 0", 90.0, 10.0, 12.0, 70)

        json_str = json.dumps(vitals.to_dict())
        restored = DeviceVitals.from_dict(json.loads(json_str))

        assert restored.device_id == vitals.device_id
        assert restored.memory_used_gb == vitals.memory_used_gb
```

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_schema.py::TestConnectionStatus tests/karn/overwatch/test_schema.py::TestDeviceVitals -v`

Expected: FAIL with `ImportError`

**Step 3: Implement ConnectionStatus and DeviceVitals**

```python
# src/esper/karn/overwatch/schema.py (append after TamiyoState)

@dataclass
class ConnectionStatus:
    """Connection status for header display.

    Tracks whether telemetry is flowing and how stale it is.
    """

    connected: bool
    last_event_ts: float  # Unix timestamp
    staleness_s: float  # Seconds since last event

    @property
    def display_text(self) -> str:
        """Human-readable status text."""
        if not self.connected:
            return "Disconnected"
        if self.staleness_s < 2.0:
            return "Live"
        if self.staleness_s < 5.0:
            return f"Live ({self.staleness_s:.0f}s)"
        return f"Stale ({self.staleness_s:.0f}s)"

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "connected": self.connected,
            "last_event_ts": self.last_event_ts,
            "staleness_s": self.staleness_s,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConnectionStatus:
        """Reconstruct from dict."""
        return cls(
            connected=data["connected"],
            last_event_ts=data["last_event_ts"],
            staleness_s=data["staleness_s"],
        )


@dataclass
class DeviceVitals:
    """GPU device vitals for header resource display."""

    device_id: int
    name: str
    utilization_pct: float
    memory_used_gb: float
    memory_total_gb: float
    temperature_c: int = 0

    @property
    def memory_pct(self) -> float:
        """Memory usage as percentage."""
        if self.memory_total_gb == 0:
            return 0.0
        return (self.memory_used_gb / self.memory_total_gb) * 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "device_id": self.device_id,
            "name": self.name,
            "utilization_pct": self.utilization_pct,
            "memory_used_gb": self.memory_used_gb,
            "memory_total_gb": self.memory_total_gb,
            "temperature_c": self.temperature_c,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeviceVitals:
        """Reconstruct from dict."""
        return cls(
            device_id=data["device_id"],
            name=data["name"],
            utilization_pct=data["utilization_pct"],
            memory_used_gb=data["memory_used_gb"],
            memory_total_gb=data["memory_total_gb"],
            temperature_c=data.get("temperature_c", 0),
        )
```

**Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_schema.py::TestConnectionStatus tests/karn/overwatch/test_schema.py::TestDeviceVitals -v`

Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add src/esper/karn/overwatch/schema.py tests/karn/overwatch/test_schema.py
git commit -m "feat(overwatch): add ConnectionStatus and DeviceVitals schemas"
```

---

## Task 6: Define TuiSnapshot (Root Schema)

**Files:**
- Modify: `src/esper/karn/overwatch/schema.py`
- Modify: `tests/karn/overwatch/test_schema.py`

**Step 1: Write failing test for TuiSnapshot**

```python
# tests/karn/overwatch/test_schema.py (append to file)

class TestTuiSnapshot:
    """Tests for TuiSnapshot dataclass (root schema)."""

    def test_tui_snapshot_creation(self) -> None:
        """TuiSnapshot can be created with minimal fields."""
        from esper.karn.overwatch.schema import (
            TuiSnapshot,
            ConnectionStatus,
            TamiyoState,
        )

        snap = TuiSnapshot(
            schema_version=1,
            captured_at="2025-12-18T12:00:00Z",
            connection=ConnectionStatus(True, 1000.0, 0.5),
            tamiyo=TamiyoState(),
        )

        assert snap.schema_version == 1
        assert snap.connection.connected is True
        assert snap.flight_board == []

    def test_tui_snapshot_with_envs(self) -> None:
        """TuiSnapshot contains flight board envs."""
        from esper.karn.overwatch.schema import (
            TuiSnapshot,
            ConnectionStatus,
            TamiyoState,
            EnvSummary,
            SlotChipState,
        )

        envs = [
            EnvSummary(
                env_id=0,
                device_id=0,
                status="OK",
                slots={"r0c1": SlotChipState("r0c1", "TRAINING", "conv", 0.5)},
            ),
            EnvSummary(
                env_id=1,
                device_id=1,
                status="WARN",
                anomaly_score=0.6,
            ),
        ]

        snap = TuiSnapshot(
            schema_version=1,
            captured_at="2025-12-18T12:00:00Z",
            connection=ConnectionStatus(True, 1000.0, 0.5),
            tamiyo=TamiyoState(),
            flight_board=envs,
        )

        assert len(snap.flight_board) == 2
        assert snap.flight_board[0].slots["r0c1"].stage == "TRAINING"

    def test_tui_snapshot_to_dict(self) -> None:
        """TuiSnapshot serializes completely."""
        from esper.karn.overwatch.schema import (
            TuiSnapshot,
            ConnectionStatus,
            TamiyoState,
            EnvSummary,
            DeviceVitals,
        )

        snap = TuiSnapshot(
            schema_version=1,
            captured_at="2025-12-18T12:00:00Z",
            connection=ConnectionStatus(True, 1000.0, 0.5),
            tamiyo=TamiyoState(kl_divergence=0.02),
            run_id="run-123",
            task_name="cifar10",
            episode=47,
            batch=1203,
            best_metric=82.1,
            runtime_s=8040.0,
            devices=[DeviceVitals(0, "GPU 0", 94.0, 11.2, 12.0, 72)],
            flight_board=[EnvSummary(0, 0, "OK")],
            envs_ok=4,
            envs_warn=0,
            envs_crit=0,
        )

        d = snap.to_dict()

        assert d["schema_version"] == 1
        assert d["tamiyo"]["kl_divergence"] == 0.02
        assert len(d["devices"]) == 1
        assert d["devices"][0]["utilization_pct"] == 94.0

    def test_tui_snapshot_from_dict(self) -> None:
        """TuiSnapshot deserializes completely."""
        from esper.karn.overwatch.schema import TuiSnapshot

        d = {
            "schema_version": 1,
            "captured_at": "2025-12-18T12:00:00Z",
            "connection": {"connected": True, "last_event_ts": 1000.0, "staleness_s": 1.0},
            "tamiyo": {"kl_divergence": 0.015, "action_counts": {"BLEND": 5}},
            "run_id": "test-run",
            "task_name": "mnist",
            "episode": 10,
            "batch": 500,
            "best_metric": 95.0,
            "runtime_s": 600.0,
            "devices": [
                {"device_id": 0, "name": "GPU 0", "utilization_pct": 80.0,
                 "memory_used_gb": 8.0, "memory_total_gb": 12.0}
            ],
            "flight_board": [
                {"env_id": 0, "device_id": 0, "status": "OK", "slots": {}}
            ],
            "event_feed": [],
            "envs_ok": 1,
            "envs_warn": 0,
            "envs_crit": 0,
        }

        snap = TuiSnapshot.from_dict(d)

        assert snap.schema_version == 1
        assert snap.tamiyo.kl_divergence == 0.015
        assert snap.tamiyo.action_counts["BLEND"] == 5
        assert len(snap.devices) == 1
        assert snap.devices[0].name == "GPU 0"

    def test_tui_snapshot_json_roundtrip(self) -> None:
        """TuiSnapshot survives full JSON serialization cycle."""
        from esper.karn.overwatch.schema import (
            TuiSnapshot,
            ConnectionStatus,
            TamiyoState,
            EnvSummary,
            SlotChipState,
            DeviceVitals,
        )

        original = TuiSnapshot(
            schema_version=1,
            captured_at="2025-12-18T12:00:00Z",
            connection=ConnectionStatus(True, 1000.0, 0.5),
            tamiyo=TamiyoState(
                action_counts={"GERMINATE": 10, "BLEND": 20},
                kl_divergence=0.019,
            ),
            devices=[DeviceVitals(0, "GPU 0", 90.0, 10.0, 12.0, 68)],
            flight_board=[
                EnvSummary(
                    env_id=0,
                    device_id=0,
                    status="OK",
                    slots={"r0c1": SlotChipState("r0c1", "BLENDING", "conv", 0.7)},
                )
            ],
        )

        json_str = json.dumps(original.to_dict())
        restored = TuiSnapshot.from_dict(json.loads(json_str))

        assert restored.schema_version == original.schema_version
        assert restored.tamiyo.kl_divergence == original.tamiyo.kl_divergence
        assert restored.flight_board[0].slots["r0c1"].alpha == 0.7
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_schema.py::TestTuiSnapshot -v`

Expected: FAIL with `ImportError`

**Step 3: Implement TuiSnapshot**

```python
# src/esper/karn/overwatch/schema.py (append after DeviceVitals)

@dataclass
class FeedEvent:
    """Single event for the Event Feed panel."""

    timestamp: str  # ISO format or display format like "12:00:03"
    event_type: str  # GATE, STAGE, PPO, GERM, CULL, etc.
    env_id: int | None  # None for global events like PPO updates
    message: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "env_id": self.env_id,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeedEvent:
        """Reconstruct from dict."""
        return cls(
            timestamp=data["timestamp"],
            event_type=data["event_type"],
            env_id=data.get("env_id"),
            message=data["message"],
        )


@dataclass
class TuiSnapshot:
    """Complete snapshot for Overwatch TUI rendering.

    This is the root schema that contains all data needed to render
    the entire TUI at a point in time. Designed for JSON serialization
    to support replay functionality.
    """

    # Schema version for future migrations
    schema_version: int

    # Timing
    captured_at: str  # ISO format timestamp

    # Connection status
    connection: ConnectionStatus

    # Tamiyo agent state
    tamiyo: TamiyoState

    # Run identity
    run_id: str = ""
    task_name: str = ""
    episode: int = 0
    batch: int = 0
    best_metric: float = 0.0
    runtime_s: float = 0.0

    # Device vitals
    devices: list[DeviceVitals] = field(default_factory=list)

    # Flight board (list of envs, UI will sort by anomaly)
    flight_board: list[EnvSummary] = field(default_factory=list)

    # Event feed
    event_feed: list[FeedEvent] = field(default_factory=list)

    # Aggregate health counts
    envs_ok: int = 0
    envs_warn: int = 0
    envs_crit: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "schema_version": self.schema_version,
            "captured_at": self.captured_at,
            "connection": self.connection.to_dict(),
            "tamiyo": self.tamiyo.to_dict(),
            "run_id": self.run_id,
            "task_name": self.task_name,
            "episode": self.episode,
            "batch": self.batch,
            "best_metric": self.best_metric,
            "runtime_s": self.runtime_s,
            "devices": [d.to_dict() for d in self.devices],
            "flight_board": [e.to_dict() for e in self.flight_board],
            "event_feed": [e.to_dict() for e in self.event_feed],
            "envs_ok": self.envs_ok,
            "envs_warn": self.envs_warn,
            "envs_crit": self.envs_crit,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TuiSnapshot:
        """Reconstruct from dict."""
        return cls(
            schema_version=data["schema_version"],
            captured_at=data["captured_at"],
            connection=ConnectionStatus.from_dict(data["connection"]),
            tamiyo=TamiyoState.from_dict(data.get("tamiyo", {})),
            run_id=data.get("run_id", ""),
            task_name=data.get("task_name", ""),
            episode=data.get("episode", 0),
            batch=data.get("batch", 0),
            best_metric=data.get("best_metric", 0.0),
            runtime_s=data.get("runtime_s", 0.0),
            devices=[DeviceVitals.from_dict(d) for d in data.get("devices", [])],
            flight_board=[EnvSummary.from_dict(e) for e in data.get("flight_board", [])],
            event_feed=[FeedEvent.from_dict(e) for e in data.get("event_feed", [])],
            envs_ok=data.get("envs_ok", 0),
            envs_warn=data.get("envs_warn", 0),
            envs_crit=data.get("envs_crit", 0),
        )
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_schema.py::TestTuiSnapshot -v`

Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/esper/karn/overwatch/schema.py tests/karn/overwatch/test_schema.py
git commit -m "feat(overwatch): add TuiSnapshot root schema with FeedEvent"
```

---

## Task 7: Update Package Exports

**Files:**
- Modify: `src/esper/karn/overwatch/__init__.py`

**Step 1: Write failing test for imports**

```python
# tests/karn/overwatch/test_schema.py (append to file)

class TestPackageExports:
    """Tests that public API is exported correctly."""

    def test_all_schemas_importable_from_package(self) -> None:
        """All schema classes are importable from overwatch package."""
        from esper.karn.overwatch import (
            TuiSnapshot,
            EnvSummary,
            SlotChipState,
            TamiyoState,
            ConnectionStatus,
            DeviceVitals,
            FeedEvent,
        )

        # Just verify imports work
        assert TuiSnapshot is not None
        assert EnvSummary is not None
        assert SlotChipState is not None
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_schema.py::TestPackageExports -v`

Expected: FAIL with `ImportError`

**Step 3: Update __init__.py exports**

```python
# src/esper/karn/overwatch/__init__.py
"""Overwatch - Textual TUI for Esper training monitoring.

Provides real-time visibility into training environments, seed lifecycle,
and Tamiyo decision-making.
"""

from esper.karn.overwatch.schema import (
    TuiSnapshot,
    EnvSummary,
    SlotChipState,
    TamiyoState,
    ConnectionStatus,
    DeviceVitals,
    FeedEvent,
)

__all__ = [
    "TuiSnapshot",
    "EnvSummary",
    "SlotChipState",
    "TamiyoState",
    "ConnectionStatus",
    "DeviceVitals",
    "FeedEvent",
]
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_schema.py::TestPackageExports -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/overwatch/__init__.py tests/karn/overwatch/test_schema.py
git commit -m "feat(overwatch): export all schema classes from package"
```

---

## Task 8: Implement SnapshotWriter

**Files:**
- Create: `src/esper/karn/overwatch/replay.py`
- Create: `tests/karn/overwatch/test_replay.py`

**Step 1: Write failing test for SnapshotWriter**

```python
# tests/karn/overwatch/test_replay.py
"""Tests for Overwatch replay infrastructure."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


class TestSnapshotWriter:
    """Tests for SnapshotWriter class."""

    def test_writer_creates_file(self, tmp_path: Path) -> None:
        """SnapshotWriter creates JSONL file."""
        from esper.karn.overwatch.replay import SnapshotWriter
        from esper.karn.overwatch.schema import (
            TuiSnapshot,
            ConnectionStatus,
            TamiyoState,
        )

        path = tmp_path / "test.jsonl"
        writer = SnapshotWriter(path)

        snap = TuiSnapshot(
            schema_version=1,
            captured_at="2025-12-18T12:00:00Z",
            connection=ConnectionStatus(True, 1000.0, 0.5),
            tamiyo=TamiyoState(),
        )

        writer.write(snap)
        writer.close()

        assert path.exists()
        assert path.stat().st_size > 0

    def test_writer_writes_one_line_per_snapshot(self, tmp_path: Path) -> None:
        """SnapshotWriter writes JSONL format (one JSON per line)."""
        from esper.karn.overwatch.replay import SnapshotWriter
        from esper.karn.overwatch.schema import (
            TuiSnapshot,
            ConnectionStatus,
            TamiyoState,
        )

        path = tmp_path / "test.jsonl"
        writer = SnapshotWriter(path)

        for i in range(3):
            snap = TuiSnapshot(
                schema_version=1,
                captured_at=f"2025-12-18T12:00:0{i}Z",
                connection=ConnectionStatus(True, 1000.0 + i, 0.5),
                tamiyo=TamiyoState(),
            )
            writer.write(snap)

        writer.close()

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 3

        # Each line is valid JSON
        for line in lines:
            data = json.loads(line)
            assert "schema_version" in data

    def test_writer_context_manager(self, tmp_path: Path) -> None:
        """SnapshotWriter works as context manager."""
        from esper.karn.overwatch.replay import SnapshotWriter
        from esper.karn.overwatch.schema import (
            TuiSnapshot,
            ConnectionStatus,
            TamiyoState,
        )

        path = tmp_path / "test.jsonl"

        with SnapshotWriter(path) as writer:
            snap = TuiSnapshot(
                schema_version=1,
                captured_at="2025-12-18T12:00:00Z",
                connection=ConnectionStatus(True, 1000.0, 0.5),
                tamiyo=TamiyoState(),
            )
            writer.write(snap)

        # File should be closed and flushed
        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1

    def test_writer_flushes_on_each_write(self, tmp_path: Path) -> None:
        """SnapshotWriter flushes after each write for crash safety."""
        from esper.karn.overwatch.replay import SnapshotWriter
        from esper.karn.overwatch.schema import (
            TuiSnapshot,
            ConnectionStatus,
            TamiyoState,
        )

        path = tmp_path / "test.jsonl"
        writer = SnapshotWriter(path)

        snap = TuiSnapshot(
            schema_version=1,
            captured_at="2025-12-18T12:00:00Z",
            connection=ConnectionStatus(True, 1000.0, 0.5),
            tamiyo=TamiyoState(),
        )
        writer.write(snap)

        # Without closing, file should still have content (due to flush)
        assert path.stat().st_size > 0

        writer.close()
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_replay.py::TestSnapshotWriter -v`

Expected: FAIL with `ModuleNotFoundError`

**Step 3: Implement SnapshotWriter**

```python
# src/esper/karn/overwatch/replay.py
"""Overwatch Replay Infrastructure.

Provides snapshot persistence for replay functionality:
- SnapshotWriter: Writes TuiSnapshot to JSONL files
- SnapshotReader: Reads TuiSnapshot from JSONL files with filtering
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterator

if TYPE_CHECKING:
    from esper.karn.overwatch.schema import TuiSnapshot


class SnapshotWriter:
    """Writes TuiSnapshot objects to JSONL file.

    Each snapshot is written as a single JSON line, enabling:
    - Streaming writes during training
    - Crash-safe persistence (flush after each write)
    - Easy append for long-running sessions

    Usage:
        with SnapshotWriter(path) as writer:
            writer.write(snapshot)
    """

    def __init__(self, path: Path | str) -> None:
        """Initialize writer.

        Args:
            path: Path to JSONL file (will be created/overwritten)
        """
        self._path = Path(path)
        self._file = open(self._path, "w", encoding="utf-8")

    def write(self, snapshot: TuiSnapshot) -> None:
        """Write a snapshot as a single JSON line.

        Args:
            snapshot: TuiSnapshot to serialize
        """
        json_str = json.dumps(snapshot.to_dict(), separators=(",", ":"))
        self._file.write(json_str + "\n")
        self._file.flush()  # Crash safety

    def close(self) -> None:
        """Close the file."""
        self._file.close()

    def __enter__(self) -> SnapshotWriter:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_replay.py::TestSnapshotWriter -v`

Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add src/esper/karn/overwatch/replay.py tests/karn/overwatch/test_replay.py
git commit -m "feat(overwatch): add SnapshotWriter for JSONL persistence"
```

---

## Task 9: Implement SnapshotReader

**Files:**
- Modify: `src/esper/karn/overwatch/replay.py`
- Modify: `tests/karn/overwatch/test_replay.py`

**Step 1: Write failing test for SnapshotReader**

```python
# tests/karn/overwatch/test_replay.py (append to file)

class TestSnapshotReader:
    """Tests for SnapshotReader class."""

    def test_reader_yields_snapshots(self, tmp_path: Path) -> None:
        """SnapshotReader yields TuiSnapshot objects."""
        from esper.karn.overwatch.replay import SnapshotWriter, SnapshotReader
        from esper.karn.overwatch.schema import (
            TuiSnapshot,
            ConnectionStatus,
            TamiyoState,
        )

        # Write test data
        path = tmp_path / "test.jsonl"
        with SnapshotWriter(path) as writer:
            for i in range(5):
                snap = TuiSnapshot(
                    schema_version=1,
                    captured_at=f"2025-12-18T12:00:0{i}Z",
                    connection=ConnectionStatus(True, 1000.0 + i, 0.5),
                    tamiyo=TamiyoState(),
                    episode=i,
                )
                writer.write(snap)

        # Read back
        reader = SnapshotReader(path)
        snapshots = list(reader)

        assert len(snapshots) == 5
        assert snapshots[0].episode == 0
        assert snapshots[4].episode == 4
        assert all(isinstance(s, TuiSnapshot) for s in snapshots)

    def test_reader_with_filter(self, tmp_path: Path) -> None:
        """SnapshotReader filters snapshots."""
        from esper.karn.overwatch.replay import SnapshotWriter, SnapshotReader
        from esper.karn.overwatch.schema import (
            TuiSnapshot,
            ConnectionStatus,
            TamiyoState,
        )

        # Write test data
        path = tmp_path / "test.jsonl"
        with SnapshotWriter(path) as writer:
            for i in range(10):
                snap = TuiSnapshot(
                    schema_version=1,
                    captured_at=f"2025-12-18T12:00:{i:02d}Z",
                    connection=ConnectionStatus(True, 1000.0 + i, 0.5),
                    tamiyo=TamiyoState(),
                    episode=i,
                )
                writer.write(snap)

        # Read with filter
        reader = SnapshotReader(path, filter_fn=lambda s: s.episode >= 5)
        snapshots = list(reader)

        assert len(snapshots) == 5
        assert snapshots[0].episode == 5
        assert snapshots[4].episode == 9

    def test_reader_handles_empty_file(self, tmp_path: Path) -> None:
        """SnapshotReader handles empty file gracefully."""
        from esper.karn.overwatch.replay import SnapshotReader

        path = tmp_path / "empty.jsonl"
        path.touch()

        reader = SnapshotReader(path)
        snapshots = list(reader)

        assert snapshots == []

    def test_reader_preserves_nested_data(self, tmp_path: Path) -> None:
        """SnapshotReader preserves nested slot and env data."""
        from esper.karn.overwatch.replay import SnapshotWriter, SnapshotReader
        from esper.karn.overwatch.schema import (
            TuiSnapshot,
            ConnectionStatus,
            TamiyoState,
            EnvSummary,
            SlotChipState,
        )

        path = tmp_path / "test.jsonl"
        with SnapshotWriter(path) as writer:
            snap = TuiSnapshot(
                schema_version=1,
                captured_at="2025-12-18T12:00:00Z",
                connection=ConnectionStatus(True, 1000.0, 0.5),
                tamiyo=TamiyoState(
                    action_counts={"BLEND": 10, "CULL": 5},
                    kl_divergence=0.019,
                ),
                flight_board=[
                    EnvSummary(
                        env_id=0,
                        device_id=0,
                        status="OK",
                        slots={
                            "r0c1": SlotChipState(
                                slot_id="r0c1",
                                stage="BLENDING",
                                blueprint_id="conv_light",
                                alpha=0.7,
                                gate_last="G2",
                                gate_passed=True,
                            )
                        },
                    )
                ],
            )
            writer.write(snap)

        reader = SnapshotReader(path)
        restored = list(reader)[0]

        assert restored.tamiyo.action_counts["BLEND"] == 10
        assert restored.tamiyo.kl_divergence == 0.019
        assert restored.flight_board[0].slots["r0c1"].alpha == 0.7
        assert restored.flight_board[0].slots["r0c1"].gate_passed is True

    def test_reader_is_iterable(self, tmp_path: Path) -> None:
        """SnapshotReader can be iterated multiple times."""
        from esper.karn.overwatch.replay import SnapshotWriter, SnapshotReader
        from esper.karn.overwatch.schema import (
            TuiSnapshot,
            ConnectionStatus,
            TamiyoState,
        )

        path = tmp_path / "test.jsonl"
        with SnapshotWriter(path) as writer:
            for i in range(3):
                snap = TuiSnapshot(
                    schema_version=1,
                    captured_at=f"2025-12-18T12:00:0{i}Z",
                    connection=ConnectionStatus(True, 1000.0, 0.5),
                    tamiyo=TamiyoState(),
                )
                writer.write(snap)

        reader = SnapshotReader(path)

        # First iteration
        first = list(reader)
        assert len(first) == 3

        # Second iteration
        second = list(reader)
        assert len(second) == 3
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_replay.py::TestSnapshotReader -v`

Expected: FAIL with `ImportError`

**Step 3: Implement SnapshotReader**

```python
# src/esper/karn/overwatch/replay.py (append after SnapshotWriter)

class SnapshotReader:
    """Reads TuiSnapshot objects from JSONL file.

    Supports filtering for selective replay and is re-iterable.

    Usage:
        reader = SnapshotReader(path, filter_fn=lambda s: s.episode > 10)
        for snapshot in reader:
            process(snapshot)
    """

    def __init__(
        self,
        path: Path | str,
        filter_fn: Callable[[TuiSnapshot], bool] | None = None,
    ) -> None:
        """Initialize reader.

        Args:
            path: Path to JSONL file
            filter_fn: Optional predicate to filter snapshots
        """
        self._path = Path(path)
        self._filter_fn = filter_fn

    def __iter__(self) -> Iterator[TuiSnapshot]:
        """Iterate over snapshots in file."""
        from esper.karn.overwatch.schema import TuiSnapshot

        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                data = json.loads(line)
                snapshot = TuiSnapshot.from_dict(data)

                if self._filter_fn is None or self._filter_fn(snapshot):
                    yield snapshot

    def __len__(self) -> int:
        """Count snapshots (may be slow for large files)."""
        return sum(1 for _ in self)
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_replay.py::TestSnapshotReader -v`

Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/esper/karn/overwatch/replay.py tests/karn/overwatch/test_replay.py
git commit -m "feat(overwatch): add SnapshotReader with filtering support"
```

---

## Task 10: Update Package Exports for Replay

**Files:**
- Modify: `src/esper/karn/overwatch/__init__.py`
- Modify: `tests/karn/overwatch/test_replay.py`

**Step 1: Write failing test for replay imports**

```python
# tests/karn/overwatch/test_replay.py (append to file)

class TestReplayExports:
    """Tests that replay classes are exported correctly."""

    def test_replay_classes_importable_from_package(self) -> None:
        """Replay classes are importable from overwatch package."""
        from esper.karn.overwatch import SnapshotWriter, SnapshotReader

        assert SnapshotWriter is not None
        assert SnapshotReader is not None
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_replay.py::TestReplayExports -v`

Expected: FAIL with `ImportError`

**Step 3: Update __init__.py exports**

```python
# src/esper/karn/overwatch/__init__.py
"""Overwatch - Textual TUI for Esper training monitoring.

Provides real-time visibility into training environments, seed lifecycle,
and Tamiyo decision-making.
"""

from esper.karn.overwatch.schema import (
    TuiSnapshot,
    EnvSummary,
    SlotChipState,
    TamiyoState,
    ConnectionStatus,
    DeviceVitals,
    FeedEvent,
)

from esper.karn.overwatch.replay import (
    SnapshotWriter,
    SnapshotReader,
)

__all__ = [
    # Schema
    "TuiSnapshot",
    "EnvSummary",
    "SlotChipState",
    "TamiyoState",
    "ConnectionStatus",
    "DeviceVitals",
    "FeedEvent",
    # Replay
    "SnapshotWriter",
    "SnapshotReader",
]
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_replay.py::TestReplayExports -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/overwatch/__init__.py tests/karn/overwatch/test_replay.py
git commit -m "feat(overwatch): export SnapshotWriter and SnapshotReader"
```

---

## Task 11: Create Test Fixtures

**Files:**
- Create: `tests/karn/overwatch/fixtures/healthy_run.jsonl`
- Create: `tests/karn/overwatch/fixtures/anomaly_detected.jsonl`
- Create: `tests/karn/overwatch/fixtures/tamiyo_active.jsonl`
- Create: `tests/karn/overwatch/fixtures/__init__.py`
- Create: `tests/karn/overwatch/conftest.py`

**Step 1: Create fixture generation script**

```python
# tests/karn/overwatch/conftest.py
"""Pytest fixtures for Overwatch tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from esper.karn.overwatch import (
    TuiSnapshot,
    EnvSummary,
    SlotChipState,
    TamiyoState,
    ConnectionStatus,
    DeviceVitals,
    FeedEvent,
)


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def healthy_snapshot() -> TuiSnapshot:
    """A snapshot with all envs healthy."""
    return TuiSnapshot(
        schema_version=1,
        captured_at="2025-12-18T12:00:00Z",
        connection=ConnectionStatus(True, 1000.0, 0.5),
        tamiyo=TamiyoState(
            action_counts={"GERMINATE": 10, "BLEND": 20, "CULL": 5, "WAIT": 65},
            recent_actions=["W", "W", "B", "G", "W"],
            kl_divergence=0.019,
            entropy=1.24,
            clip_fraction=0.048,
            explained_variance=0.42,
            confidence_mean=0.73,
        ),
        run_id="healthy-run-001",
        task_name="cifar10",
        episode=47,
        batch=1203,
        best_metric=82.1,
        runtime_s=8040.0,
        devices=[
            DeviceVitals(0, "GPU 0", 94.0, 11.2, 12.0, 72),
            DeviceVitals(1, "GPU 1", 91.0, 10.8, 12.0, 68),
        ],
        flight_board=[
            EnvSummary(
                env_id=0,
                device_id=0,
                status="OK",
                throughput_fps=98.5,
                slots={"r0c1": SlotChipState("r0c1", "FOSSILIZED", "conv_light", 1.0)},
                anomaly_score=0.05,
            ),
            EnvSummary(
                env_id=1,
                device_id=0,
                status="OK",
                throughput_fps=101.2,
                slots={"r0c1": SlotChipState("r0c1", "TRAINING", "mlp_narrow", 0.3)},
                anomaly_score=0.08,
            ),
            EnvSummary(
                env_id=2,
                device_id=1,
                status="OK",
                throughput_fps=99.0,
                slots={"r0c1": SlotChipState("r0c1", "BLENDING", "conv_light", 0.7, gate_last="G2", gate_passed=True)},
                anomaly_score=0.1,
            ),
            EnvSummary(
                env_id=3,
                device_id=1,
                status="OK",
                throughput_fps=97.8,
                slots={"r0c1": SlotChipState("r0c1", "GERMINATED", "conv_light", 0.1)},
                anomaly_score=0.03,
            ),
        ],
        envs_ok=4,
        envs_warn=0,
        envs_crit=0,
    )


@pytest.fixture
def anomaly_snapshot() -> TuiSnapshot:
    """A snapshot with anomalies detected."""
    return TuiSnapshot(
        schema_version=1,
        captured_at="2025-12-18T12:05:00Z",
        connection=ConnectionStatus(True, 1000.0, 0.5),
        tamiyo=TamiyoState(
            kl_divergence=0.08,  # High KL
            entropy=0.15,  # Low entropy (collapsed)
            entropy_collapsed=True,
            explained_variance=0.25,  # Low EV
            ev_warning=True,
        ),
        run_id="anomaly-run-001",
        task_name="cifar10",
        episode=100,
        flight_board=[
            EnvSummary(
                env_id=0,
                device_id=0,
                status="OK",
                anomaly_score=0.1,
            ),
            EnvSummary(
                env_id=1,
                device_id=0,
                status="WARN",
                throughput_fps=45.0,  # Low throughput
                anomaly_score=0.65,
                anomaly_reasons=["Throughput 55% below baseline"],
            ),
            EnvSummary(
                env_id=2,
                device_id=1,
                status="CRIT",
                reward_last=-2.5,
                anomaly_score=0.85,
                anomaly_reasons=[
                    "Unusual negative reward (-2.5)",
                    "High gradient ratio (15.2x)",
                    "Memory pressure (97%)",
                ],
                slots={"r0c1": SlotChipState("r0c1", "CULLED", "bad_blueprint", 0.0)},
            ),
        ],
        envs_ok=1,
        envs_warn=1,
        envs_crit=1,
    )


@pytest.fixture
def fixtures_dir() -> Path:
    """Path to fixtures directory."""
    return FIXTURES_DIR
```

**Step 2: Create fixture JSONL files**

```python
# tests/karn/overwatch/fixtures/__init__.py
"""Test fixtures for Overwatch."""
```

Run this Python script to generate fixtures:

```bash
PYTHONPATH=src uv run python -c "
from pathlib import Path
from esper.karn.overwatch import (
    TuiSnapshot, EnvSummary, SlotChipState, TamiyoState,
    ConnectionStatus, DeviceVitals, FeedEvent, SnapshotWriter
)

fixtures_dir = Path('tests/karn/overwatch/fixtures')
fixtures_dir.mkdir(parents=True, exist_ok=True)

# healthy_run.jsonl - 10 snapshots, all healthy
with SnapshotWriter(fixtures_dir / 'healthy_run.jsonl') as w:
    for i in range(10):
        snap = TuiSnapshot(
            schema_version=1,
            captured_at=f'2025-12-18T12:00:{i:02d}Z',
            connection=ConnectionStatus(True, 1000.0 + i, 0.5),
            tamiyo=TamiyoState(
                action_counts={'GERMINATE': 10+i, 'BLEND': 20, 'WAIT': 70-i},
                kl_divergence=0.019,
                entropy=1.24,
            ),
            episode=i,
            flight_board=[
                EnvSummary(env_id=0, device_id=0, status='OK', anomaly_score=0.05,
                    slots={'r0c1': SlotChipState('r0c1', 'TRAINING', 'conv', 0.3 + i*0.05)}),
                EnvSummary(env_id=1, device_id=1, status='OK', anomaly_score=0.08),
            ],
            envs_ok=2,
        )
        w.write(snap)

# anomaly_detected.jsonl - 5 snapshots with issues
with SnapshotWriter(fixtures_dir / 'anomaly_detected.jsonl') as w:
    for i in range(5):
        snap = TuiSnapshot(
            schema_version=1,
            captured_at=f'2025-12-18T13:00:{i:02d}Z',
            connection=ConnectionStatus(True, 2000.0 + i, 0.5),
            tamiyo=TamiyoState(
                entropy=0.1,
                entropy_collapsed=True,
                explained_variance=0.2,
                ev_warning=True,
            ),
            episode=50 + i,
            flight_board=[
                EnvSummary(env_id=0, device_id=0, status='WARN', anomaly_score=0.65,
                    anomaly_reasons=['High gradient ratio']),
                EnvSummary(env_id=1, device_id=1, status='CRIT', anomaly_score=0.85,
                    anomaly_reasons=['Entropy collapsed', 'Throughput drop']),
            ],
            envs_ok=0, envs_warn=1, envs_crit=1,
        )
        w.write(snap)

# tamiyo_active.jsonl - 20 snapshots with varied Tamiyo activity
with SnapshotWriter(fixtures_dir / 'tamiyo_active.jsonl') as w:
    actions = ['G', 'B', 'W', 'C', 'W', 'B', 'G', 'W', 'W', 'B']
    for i in range(20):
        snap = TuiSnapshot(
            schema_version=1,
            captured_at=f'2025-12-18T14:{i:02d}:00Z',
            connection=ConnectionStatus(True, 3000.0 + i, 0.5),
            tamiyo=TamiyoState(
                action_counts={'GERMINATE': 15+i, 'BLEND': 30+i, 'CULL': 5, 'WAIT': 50-i},
                recent_actions=actions[i%5:i%5+5],
                confidence_mean=0.6 + (i % 10) * 0.03,
                kl_divergence=0.015 + i * 0.001,
                entropy=1.5 - i * 0.02,
            ),
            episode=i,
            flight_board=[
                EnvSummary(env_id=j, device_id=j%2, status='OK', anomaly_score=0.1)
                for j in range(4)
            ],
            envs_ok=4,
        )
        w.write(snap)

print('Fixtures created successfully!')
"
```

**Step 3: Verify fixtures exist**

Run: `ls -la tests/karn/overwatch/fixtures/`

Expected:
```
healthy_run.jsonl
anomaly_detected.jsonl
tamiyo_active.jsonl
__init__.py
```

**Step 4: Commit fixtures**

```bash
git add tests/karn/overwatch/fixtures/ tests/karn/overwatch/conftest.py
git commit -m "test(overwatch): add test fixtures and conftest"
```

---

## Task 12: Run Full Test Suite and Final Commit

**Step 1: Run all Overwatch tests**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/ -v`

Expected: All tests PASS (should be ~30+ tests)

**Step 2: Run linting**

Run: `uv run ruff check src/esper/karn/overwatch/`

Expected: No errors

**Step 3: Run type checking**

Run: `uv run mypy src/esper/karn/overwatch/ --ignore-missing-imports`

Expected: No errors (or only minor issues)

**Step 4: Final summary commit**

```bash
git add -A
git status
```

If any uncommitted changes:
```bash
git commit -m "chore(overwatch): Stage 0 complete - foundation and replay infrastructure"
```

---

## Verification Checklist

- [ ] `TuiSnapshot` round-trips through JSON without data loss
- [ ] `SnapshotWriter` creates valid JSONL files
- [ ] `SnapshotReader` yields `TuiSnapshot` objects
- [ ] Filter function works on reader
- [ ] Test fixtures created (3 JSONL files)
- [ ] All schema classes exported from package
- [ ] All tests pass (~30+ tests)
- [ ] Linting passes
- [ ] Can be merged to main independently

---

## Files Created

```
src/esper/karn/overwatch/
├── __init__.py          # Package exports
├── schema.py            # TuiSnapshot, EnvSummary, SlotChipState, etc.
└── replay.py            # SnapshotWriter, SnapshotReader

tests/karn/overwatch/
├── __init__.py
├── conftest.py          # Pytest fixtures
├── test_schema.py       # Schema tests
├── test_replay.py       # Replay tests
└── fixtures/
    ├── __init__.py
    ├── healthy_run.jsonl
    ├── anomaly_detected.jsonl
    └── tamiyo_active.jsonl
```

---

## Next Stage

After Stage 0 is merged, proceed to **Stage 1: Minimal Textual App Shell** which will:
- Create `OverwatchApp` with `compose()` method
- Add basic layout with placeholder widgets
- Implement `q` (quit) and `?` (help) bindings
- Load snapshot from JSONL file via CLI
