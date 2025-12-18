# Overwatch Stage 6: Live Telemetry Integration

**Status:** Ready for Implementation
**Prerequisites:** Stage 5 Complete (Event Feed + Replay)
**Estimated Tasks:** 7

## Overview

This stage connects the Overwatch TUI to live training telemetry, enabling real-time monitoring of training runs. The key components are:

1. **TelemetryAggregator** - Transforms streaming `TelemetryEvent` objects into `TuiSnapshot` state
2. **OverwatchBackend** - `OutputBackend` implementation that receives events and updates aggregator
3. **Live Mode Wiring** - `set_interval` polling in `OverwatchApp` to refresh from aggregator
4. **CLI Integration** - `--overwatch` flag launches TUI alongside training

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Training Process (Simic)                        │
│                                                                         │
│  train_ppo_vectorized() ──emit()──▶ NissaHub (get_hub())               │
│                                           │                             │
│                                           ├──▶ ConsoleOutput            │
│                                           ├──▶ FileOutput               │
│                                           ├──▶ KarnCollector            │
│                                           └──▶ OverwatchBackend ◀──NEW  │
│                                                      │                  │
│                                                      ▼                  │
│                                          TelemetryAggregator ◀──NEW    │
│                                                      │                  │
│                                          builds TuiSnapshot             │
└──────────────────────────────────────────────────────┼──────────────────┘
                                                       │
                                                       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           OverwatchApp (Textual)                        │
│                                                                         │
│     set_interval(250ms) ────▶ poll aggregator.get_snapshot()           │
│                                        │                                │
│                                        ▼                                │
│                              _update_all_widgets()                      │
│                                        │                                │
│              ┌─────────────────────────┼─────────────────────────┐     │
│              ▼                         ▼                         ▼     │
│        RunHeader                 FlightBoard                EventFeed  │
│        TamiyoStrip               DetailPanel                           │
└─────────────────────────────────────────────────────────────────────────┘
```

## Event → Snapshot Mapping

| TelemetryEvent Type | TuiSnapshot Updates |
|---------------------|---------------------|
| `TRAINING_STARTED` | `run_id`, `task_name`, `connection.connected=True` |
| `BATCH_COMPLETED` | `batch`, `episode`, `best_metric`, `runtime_s` |
| `PPO_UPDATE_COMPLETED` | `TamiyoState.*` (kl, entropy, clip_fraction, etc.) |
| `EPOCH_COMPLETED` | Per-env `EnvSummary.task_metric` |
| `SEED_GERMINATED` | `SlotChipState` creation, `FeedEvent(GERM)` |
| `SEED_STAGE_CHANGED` | `SlotChipState.stage`, `FeedEvent(STAGE)` |
| `SEED_GATE_EVALUATED` | `SlotChipState.gate_*`, `FeedEvent(GATE)` |
| `SEED_FOSSILIZED` | Stage update, `FeedEvent(STAGE)` |
| `SEED_CULLED` | Stage update, `FeedEvent(CULL)` |
| `REWARD_COMPUTED` | Per-env `EnvSummary` throughput/reward |
| `GOVERNOR_PANIC` | `FeedEvent(CRIT)` |
| `GOVERNOR_ROLLBACK` | `FeedEvent(CRIT)` |

---

## Task 1: TelemetryAggregator Core

**File:** `src/esper/karn/overwatch/aggregator.py`

Creates a stateful aggregator that maintains `TuiSnapshot` from streaming events.

**Thread-safety:** Uses `threading.Lock` to protect state during concurrent access from training thread (emit) and UI thread (get_snapshot).

### Test File: `tests/karn/overwatch/test_aggregator.py`

```python
"""Tests for TelemetryAggregator."""

import pytest
from datetime import datetime, timezone

from esper.leyline import TelemetryEvent, TelemetryEventType
from esper.karn.overwatch.aggregator import TelemetryAggregator
from esper.karn.overwatch.schema import TuiSnapshot


class TestAggregatorBasics:
    """Test basic aggregator lifecycle."""

    def test_initial_snapshot_is_disconnected(self):
        """Fresh aggregator shows disconnected state."""
        agg = TelemetryAggregator()
        snapshot = agg.get_snapshot()

        assert snapshot.connection.connected is False
        assert snapshot.run_id == ""
        assert len(snapshot.flight_board) == 0

    def test_training_started_connects(self):
        """TRAINING_STARTED event marks connection as live."""
        agg = TelemetryAggregator()

        event = TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data={
                "run_id": "run-abc123",
                "task": "cifar10",
                "max_epochs": 75,
                "num_envs": 4,
            },
        )
        agg.process_event(event)

        snapshot = agg.get_snapshot()
        assert snapshot.connection.connected is True
        assert snapshot.run_id == "run-abc123"
        assert snapshot.task_name == "cifar10"

    def test_staleness_tracking(self):
        """Staleness increases when no events received."""
        agg = TelemetryAggregator()

        # Simulate event at specific time
        event = TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            timestamp=datetime.now(timezone.utc),
            data={"run_id": "test"},
        )
        agg.process_event(event)

        snapshot = agg.get_snapshot()
        # Staleness should be very small immediately after event
        assert snapshot.connection.staleness_s < 1.0


class TestBatchAndEpisodeTracking:
    """Test batch/episode progress updates."""

    def test_batch_completed_updates_progress(self):
        """BATCH_COMPLETED updates batch and episode counters."""
        agg = TelemetryAggregator()
        agg.process_event(TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data={"run_id": "test", "num_envs": 2},
        ))

        agg.process_event(TelemetryEvent(
            event_type=TelemetryEventType.BATCH_COMPLETED,
            data={
                "batch_idx": 5,
                "episodes_completed": 10,
                "total_episodes": 100,
                "avg_accuracy": 65.5,
                "rolling_accuracy": 64.0,
            },
        ))

        snapshot = agg.get_snapshot()
        assert snapshot.batch == 5
        assert snapshot.episode == 10
        # best_metric stored as 0-1 range (65.5% -> 0.655)
        assert snapshot.best_metric == pytest.approx(0.655, rel=0.01)


class TestPPOVitals:
    """Test PPO update telemetry → TamiyoState."""

    def test_ppo_update_populates_tamiyo(self):
        """PPO_UPDATE_COMPLETED fills TamiyoState fields."""
        agg = TelemetryAggregator()
        agg.process_event(TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data={"run_id": "test"},
        ))

        agg.process_event(TelemetryEvent(
            event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
            data={
                "kl_divergence": 0.015,
                "entropy": 1.2,
                "clip_fraction": 0.08,
                "explained_variance": 0.85,
                "policy_loss": -0.02,
                "value_loss": 0.5,
                "grad_norm": 1.5,
                "learning_rate": 3e-4,
            },
        ))

        snapshot = agg.get_snapshot()
        assert snapshot.tamiyo.kl_divergence == pytest.approx(0.015)
        assert snapshot.tamiyo.entropy == pytest.approx(1.2)
        assert snapshot.tamiyo.clip_fraction == pytest.approx(0.08)
        assert snapshot.tamiyo.explained_variance == pytest.approx(0.85)


class TestEpochCompleted:
    """Test EPOCH_COMPLETED event handling."""

    def test_epoch_completed_updates_env_metric(self):
        """EPOCH_COMPLETED updates per-env task_metric."""
        agg = TelemetryAggregator(num_envs=2)
        agg.process_event(TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data={"run_id": "test", "num_envs": 2},
        ))

        agg.process_event(TelemetryEvent(
            event_type=TelemetryEventType.EPOCH_COMPLETED,
            data={
                "env_id": 0,
                "val_accuracy": 72.5,
                "val_loss": 0.8,
            },
        ))

        snapshot = agg.get_snapshot()
        env0 = next((e for e in snapshot.flight_board if e.env_id == 0), None)
        assert env0 is not None
        assert env0.task_metric == pytest.approx(72.5)


class TestSeedLifecycle:
    """Test seed events → SlotChipState + FeedEvent."""

    def test_seed_germinated_creates_slot(self):
        """SEED_GERMINATED creates SlotChipState entry."""
        agg = TelemetryAggregator(num_envs=2)
        agg.process_event(TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data={"run_id": "test", "num_envs": 2},
        ))

        agg.process_event(TelemetryEvent(
            event_type=TelemetryEventType.SEED_GERMINATED,
            slot_id="r0c0",
            data={
                "env_id": 0,
                "seed_id": "seed-001",
                "blueprint_id": "conv3x3",
                "params": 1500,
            },
        ))

        snapshot = agg.get_snapshot()
        # Find env 0's summary
        env0 = next((e for e in snapshot.flight_board if e.env_id == 0), None)
        assert env0 is not None
        assert "r0c0" in env0.slots
        assert env0.slots["r0c0"].stage == "GERMINATED"
        assert env0.slots["r0c0"].blueprint_id == "conv3x3"

        # Should also add feed event
        assert any(e.event_type == "GERM" for e in snapshot.event_feed)

    def test_seed_stage_changed(self):
        """SEED_STAGE_CHANGED updates slot stage."""
        agg = TelemetryAggregator(num_envs=1)
        agg.process_event(TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data={"run_id": "test", "num_envs": 1},
        ))

        # Germinate first
        agg.process_event(TelemetryEvent(
            event_type=TelemetryEventType.SEED_GERMINATED,
            slot_id="r0c0",
            data={"env_id": 0, "blueprint_id": "conv3x3"},
        ))

        # Transition to TRAINING
        agg.process_event(TelemetryEvent(
            event_type=TelemetryEventType.SEED_STAGE_CHANGED,
            slot_id="r0c0",
            data={"env_id": 0, "from": "GERMINATED", "to": "TRAINING"},
        ))

        snapshot = agg.get_snapshot()
        env0 = snapshot.flight_board[0]
        assert env0.slots["r0c0"].stage == "TRAINING"

    def test_seed_culled_adds_feed_event(self):
        """SEED_CULLED adds CULL event to feed."""
        agg = TelemetryAggregator(num_envs=1)
        agg.process_event(TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data={"run_id": "test", "num_envs": 1},
        ))
        agg.process_event(TelemetryEvent(
            event_type=TelemetryEventType.SEED_GERMINATED,
            slot_id="r0c0",
            data={"env_id": 0, "blueprint_id": "conv3x3"},
        ))
        agg.process_event(TelemetryEvent(
            event_type=TelemetryEventType.SEED_CULLED,
            slot_id="r0c0",
            data={"env_id": 0, "reason": "degradation"},
        ))

        snapshot = agg.get_snapshot()
        cull_events = [e for e in snapshot.event_feed if e.event_type == "CULL"]
        assert len(cull_events) == 1
        assert "degradation" in cull_events[0].message


class TestGateEvaluation:
    """Test gate evaluation events."""

    def test_gate_passed(self):
        """SEED_GATE_EVALUATED with passed=True."""
        agg = TelemetryAggregator(num_envs=1)
        agg.process_event(TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data={"run_id": "test", "num_envs": 1},
        ))
        agg.process_event(TelemetryEvent(
            event_type=TelemetryEventType.SEED_GERMINATED,
            slot_id="r0c0",
            data={"env_id": 0, "blueprint_id": "conv3x3"},
        ))
        agg.process_event(TelemetryEvent(
            event_type=TelemetryEventType.SEED_GATE_EVALUATED,
            slot_id="r0c0",
            data={"env_id": 0, "gate": "G1", "passed": True},
        ))

        snapshot = agg.get_snapshot()
        slot = snapshot.flight_board[0].slots["r0c0"]
        assert slot.gate_last == "G1"
        assert slot.gate_passed is True

        # Feed event for gate
        gate_events = [e for e in snapshot.event_feed if e.event_type == "GATE"]
        assert len(gate_events) == 1


class TestEventFeedManagement:
    """Test event feed size limits."""

    def test_feed_limited_to_max_events(self):
        """Event feed doesn't grow unbounded."""
        agg = TelemetryAggregator(num_envs=1, max_feed_events=5)
        agg.process_event(TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data={"run_id": "test", "num_envs": 1},
        ))

        # Add many events
        for i in range(10):
            agg.process_event(TelemetryEvent(
                event_type=TelemetryEventType.SEED_GERMINATED,
                slot_id="r0c0",
                data={"env_id": 0, "blueprint_id": f"bp{i}"},
            ))

        snapshot = agg.get_snapshot()
        assert len(snapshot.event_feed) <= 5


class TestThreadSafety:
    """Test thread-safety of aggregator."""

    def test_concurrent_access_no_crash(self):
        """Concurrent process_event and get_snapshot don't crash."""
        import threading

        agg = TelemetryAggregator(num_envs=2)
        agg.process_event(TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data={"run_id": "test", "num_envs": 2},
        ))

        errors = []

        def emit_events():
            try:
                for i in range(100):
                    agg.process_event(TelemetryEvent(
                        event_type=TelemetryEventType.BATCH_COMPLETED,
                        data={"batch_idx": i, "avg_accuracy": 50.0 + i * 0.1},
                    ))
            except Exception as e:
                errors.append(e)

        def read_snapshots():
            try:
                for _ in range(100):
                    snapshot = agg.get_snapshot()
                    _ = snapshot.batch  # Access field
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=emit_events),
            threading.Thread(target=read_snapshots),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"
```

### Implementation: `src/esper/karn/overwatch/aggregator.py`

```python
"""Telemetry Aggregator - Transforms event stream into TuiSnapshot.

Maintains stateful accumulation of telemetry events to build
real-time TuiSnapshot objects for the Overwatch TUI.

Thread-safe: Uses threading.Lock to protect state during concurrent
access from training thread (emit) and UI thread (get_snapshot).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from esper.karn.overwatch.schema import (
    TuiSnapshot,
    ConnectionStatus,
    TamiyoState,
    EnvSummary,
    SlotChipState,
    FeedEvent,
)

if TYPE_CHECKING:
    from esper.leyline import TelemetryEvent


@dataclass
class TelemetryAggregator:
    """Aggregates telemetry events into TuiSnapshot state.

    Thread-safe: process_event() and get_snapshot() can be called
    from different threads safely due to internal locking.

    Usage:
        agg = TelemetryAggregator(num_envs=4)

        # From backend thread
        agg.process_event(event)

        # From UI thread
        snapshot = agg.get_snapshot()
    """

    num_envs: int = 4
    max_feed_events: int = 100

    # Internal state (protected by _lock)
    _run_id: str = ""
    _task_name: str = ""
    _connected: bool = False
    _last_event_ts: float = 0.0

    # Progress tracking
    _episode: int = 0
    _batch: int = 0
    _best_metric: float = 0.0  # Stored as 0-1 range
    _runtime_s: float = 0.0
    _start_time: float = field(default_factory=time.time)

    # Tamiyo state
    _tamiyo: TamiyoState = field(default_factory=TamiyoState)

    # Per-env state: env_id -> EnvSummary
    _envs: dict[int, EnvSummary] = field(default_factory=dict)

    # Event feed (most recent last)
    _feed: list[FeedEvent] = field(default_factory=list)

    # Thread safety
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self):
        """Initialize per-env state."""
        self._envs = {}
        self._feed = []
        self._tamiyo = TamiyoState()
        self._start_time = time.time()
        self._lock = threading.Lock()

    def process_event(self, event: "TelemetryEvent") -> None:
        """Process a telemetry event and update internal state.

        Args:
            event: The telemetry event to process.
        """
        with self._lock:
            self._process_event_unlocked(event)

    def _process_event_unlocked(self, event: "TelemetryEvent") -> None:
        """Process event without locking (caller must hold lock)."""
        # Update last event timestamp
        if event.timestamp:
            self._last_event_ts = event.timestamp.timestamp()
        else:
            self._last_event_ts = time.time()

        # Get event type name
        # hasattr AUTHORIZED by operator on 2025-12-18 12:00:00 UTC
        # Justification: Serialization - handle both enum and string event_type values
        event_type = (
            event.event_type.name
            if hasattr(event.event_type, "name")
            else str(event.event_type)
        )

        # Route to handler
        handler = getattr(self, f"_handle_{event_type.lower()}", None)
        if handler:
            handler(event)

    def get_snapshot(self) -> TuiSnapshot:
        """Get current TuiSnapshot.

        Returns:
            Complete snapshot of current aggregator state.
        """
        with self._lock:
            return self._get_snapshot_unlocked()

    def _get_snapshot_unlocked(self) -> TuiSnapshot:
        """Get snapshot without locking (caller must hold lock)."""
        now = time.time()
        staleness = now - self._last_event_ts if self._last_event_ts else float("inf")

        return TuiSnapshot(
            schema_version=1,
            captured_at=datetime.now(timezone.utc).isoformat(),
            connection=ConnectionStatus(
                connected=self._connected,
                last_event_ts=self._last_event_ts,
                staleness_s=staleness,
            ),
            tamiyo=self._tamiyo,
            run_id=self._run_id,
            task_name=self._task_name,
            episode=self._episode,
            batch=self._batch,
            best_metric=self._best_metric,
            runtime_s=now - self._start_time if self._connected else 0.0,
            flight_board=list(self._envs.values()),
            event_feed=list(self._feed),
            envs_ok=sum(1 for e in self._envs.values() if e.status == "OK"),
            envs_warn=sum(1 for e in self._envs.values() if e.status == "WARN"),
            envs_crit=sum(1 for e in self._envs.values() if e.status == "CRIT"),
        )

    # =========================================================================
    # Event Handlers
    # =========================================================================

    def _handle_training_started(self, event: "TelemetryEvent") -> None:
        """Handle TRAINING_STARTED event."""
        data = event.data or {}
        self._run_id = data.get("run_id", data.get("episode_id", ""))
        self._task_name = data.get("task", "")
        self._connected = True
        self._start_time = time.time()

        # Initialize env summaries
        num_envs = data.get("num_envs", self.num_envs)
        for env_id in range(num_envs):
            if env_id not in self._envs:
                self._envs[env_id] = EnvSummary(
                    env_id=env_id,
                    device_id=0,  # Will be updated by device events
                    status="OK",
                )

    def _handle_batch_completed(self, event: "TelemetryEvent") -> None:
        """Handle BATCH_COMPLETED event."""
        data = event.data or {}
        self._batch = data.get("batch_idx", self._batch)
        self._episode = data.get("episodes_completed", self._episode)

        # avg_accuracy comes as 0-100, store as 0-1 for consistency with run_header
        avg_acc = data.get("avg_accuracy", 0.0)
        avg_acc_normalized = avg_acc / 100.0 if avg_acc > 1.0 else avg_acc
        if avg_acc_normalized > self._best_metric:
            self._best_metric = avg_acc_normalized

        # Update per-env accuracies if provided
        env_accs = data.get("env_accuracies", [])
        for i, acc in enumerate(env_accs):
            if i in self._envs:
                self._envs[i].task_metric = acc

    def _handle_ppo_update_completed(self, event: "TelemetryEvent") -> None:
        """Handle PPO_UPDATE_COMPLETED event."""
        data = event.data or {}

        if data.get("skipped"):
            return

        self._tamiyo.kl_divergence = data.get("kl_divergence", 0.0)
        self._tamiyo.entropy = data.get("entropy", 0.0)
        self._tamiyo.clip_fraction = data.get("clip_fraction", 0.0)
        self._tamiyo.explained_variance = data.get("explained_variance", 0.0)
        self._tamiyo.grad_norm = data.get("grad_norm", 0.0)
        self._tamiyo.learning_rate = data.get("learning_rate", 0.0)

        # Add PPO feed event for significant updates
        self._add_feed_event(
            event_type="PPO",
            env_id=None,
            message=f"KL={self._tamiyo.kl_divergence:.4f} H={self._tamiyo.entropy:.2f}",
            timestamp=event.timestamp,
        )

    def _handle_epoch_completed(self, event: "TelemetryEvent") -> None:
        """Handle EPOCH_COMPLETED event."""
        data = event.data or {}
        env_id = data.get("env_id")

        if env_id is None:
            return

        self._ensure_env(env_id)
        self._envs[env_id].task_metric = data.get("val_accuracy", 0.0)

    def _handle_seed_germinated(self, event: "TelemetryEvent") -> None:
        """Handle SEED_GERMINATED event."""
        data = event.data or {}
        env_id = data.get("env_id")
        slot_id = event.slot_id or data.get("slot_id", "unknown")

        if env_id is None:
            return

        self._ensure_env(env_id)
        self._envs[env_id].slots[slot_id] = SlotChipState(
            slot_id=slot_id,
            stage="GERMINATED",
            blueprint_id=data.get("blueprint_id", ""),
            alpha=0.0,
        )

        self._add_feed_event(
            event_type="GERM",
            env_id=env_id,
            message=f"{slot_id} germinated ({data.get('blueprint_id', '?')})",
            timestamp=event.timestamp,
        )

    def _handle_seed_stage_changed(self, event: "TelemetryEvent") -> None:
        """Handle SEED_STAGE_CHANGED event."""
        data = event.data or {}
        env_id = data.get("env_id")
        slot_id = event.slot_id or data.get("slot_id")

        if env_id is None or slot_id is None:
            return

        self._ensure_env(env_id)
        if slot_id in self._envs[env_id].slots:
            self._envs[env_id].slots[slot_id].stage = data.get("to", "UNKNOWN")
            self._envs[env_id].slots[slot_id].epochs_in_stage = 0

        self._add_feed_event(
            event_type="STAGE",
            env_id=env_id,
            message=f"{slot_id}: {data.get('from', '?')} -> {data.get('to', '?')}",
            timestamp=event.timestamp,
        )

    def _handle_seed_gate_evaluated(self, event: "TelemetryEvent") -> None:
        """Handle SEED_GATE_EVALUATED event."""
        data = event.data or {}
        env_id = data.get("env_id")
        slot_id = event.slot_id or data.get("slot_id")

        if env_id is None or slot_id is None:
            return

        self._ensure_env(env_id)
        if slot_id in self._envs[env_id].slots:
            slot = self._envs[env_id].slots[slot_id]
            slot.gate_last = data.get("gate")
            slot.gate_passed = data.get("passed")

        status = "PASS" if data.get("passed") else "FAIL"
        self._add_feed_event(
            event_type="GATE",
            env_id=env_id,
            message=f"{slot_id} {data.get('gate', '?')}: {status}",
            timestamp=event.timestamp,
        )

    def _handle_seed_fossilized(self, event: "TelemetryEvent") -> None:
        """Handle SEED_FOSSILIZED event."""
        data = event.data or {}
        env_id = data.get("env_id")
        slot_id = event.slot_id or data.get("slot_id")

        if env_id is None or slot_id is None:
            return

        self._ensure_env(env_id)
        if slot_id in self._envs[env_id].slots:
            self._envs[env_id].slots[slot_id].stage = "FOSSILIZED"

        self._add_feed_event(
            event_type="STAGE",
            env_id=env_id,
            message=f"{slot_id} fossilized",
            timestamp=event.timestamp,
        )

    def _handle_seed_culled(self, event: "TelemetryEvent") -> None:
        """Handle SEED_CULLED event."""
        data = event.data or {}
        env_id = data.get("env_id")
        slot_id = event.slot_id or data.get("slot_id")

        if env_id is None or slot_id is None:
            return

        self._ensure_env(env_id)
        if slot_id in self._envs[env_id].slots:
            self._envs[env_id].slots[slot_id].stage = "CULLED"

        reason = data.get("reason", "")
        self._add_feed_event(
            event_type="CULL",
            env_id=env_id,
            message=f"{slot_id} culled" + (f" ({reason})" if reason else ""),
            timestamp=event.timestamp,
        )

    def _handle_reward_computed(self, event: "TelemetryEvent") -> None:
        """Handle REWARD_COMPUTED event."""
        data = event.data or {}
        env_id = data.get("env_id")

        if env_id is None:
            return

        self._ensure_env(env_id)
        self._envs[env_id].reward_last = data.get("total_reward", 0.0)
        self._envs[env_id].task_metric = data.get("val_acc", 0.0)

    def _handle_governor_panic(self, event: "TelemetryEvent") -> None:
        """Handle GOVERNOR_PANIC event."""
        data = event.data or {}
        self._add_feed_event(
            event_type="CRIT",
            env_id=None,
            message=f"PANIC #{data.get('consecutive_panics', '?')}: loss={data.get('current_loss', '?')}",
            timestamp=event.timestamp,
        )

    def _handle_governor_rollback(self, event: "TelemetryEvent") -> None:
        """Handle GOVERNOR_ROLLBACK event."""
        data = event.data or {}
        self._add_feed_event(
            event_type="CRIT",
            env_id=None,
            message=f"ROLLBACK: {data.get('reason', 'unknown')}",
            timestamp=event.timestamp,
        )

    # =========================================================================
    # Helpers
    # =========================================================================

    def _ensure_env(self, env_id: int) -> None:
        """Ensure EnvSummary exists for env_id."""
        if env_id not in self._envs:
            self._envs[env_id] = EnvSummary(
                env_id=env_id,
                device_id=0,
                status="OK",
            )

    def _add_feed_event(
        self,
        event_type: str,
        env_id: int | None,
        message: str,
        timestamp: "datetime | None" = None,
    ) -> None:
        """Add event to feed, maintaining max size."""
        ts = timestamp or datetime.now(timezone.utc)
        ts_str = ts.strftime("%H:%M:%S") if hasattr(ts, "strftime") else str(ts)[:8]

        self._feed.append(FeedEvent(
            timestamp=ts_str,
            event_type=event_type,
            env_id=env_id,
            message=message,
        ))

        # Trim to max size
        if len(self._feed) > self.max_feed_events:
            self._feed = self._feed[-self.max_feed_events:]
```

### Verification

```bash
PYTHONPATH=src uv run pytest tests/karn/overwatch/test_aggregator.py -v
```

---

## Task 2: OverwatchBackend Implementation

**File:** `src/esper/karn/overwatch/backend.py`

OutputBackend that receives events and updates TelemetryAggregator.

### Test File: `tests/karn/overwatch/test_backend.py`

```python
"""Tests for OverwatchBackend."""

import pytest
from esper.leyline import TelemetryEvent, TelemetryEventType
from esper.karn.overwatch.backend import OverwatchBackend


class TestOverwatchBackend:
    """Test OverwatchBackend OutputBackend implementation."""

    def test_backend_protocol_compliance(self):
        """Backend implements OutputBackend protocol."""
        backend = OverwatchBackend()

        # Should have required methods
        assert hasattr(backend, "start")
        assert hasattr(backend, "emit")
        assert hasattr(backend, "close")

    def test_emit_updates_aggregator(self):
        """emit() passes events to aggregator."""
        backend = OverwatchBackend()
        backend.start()

        event = TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data={"run_id": "test-run", "task": "cifar10"},
        )
        backend.emit(event)

        snapshot = backend.get_snapshot()
        assert snapshot.run_id == "test-run"
        assert snapshot.connection.connected is True

    def test_get_snapshot_thread_safe(self):
        """get_snapshot() returns copy safe for cross-thread access."""
        backend = OverwatchBackend()
        backend.start()

        backend.emit(TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data={"run_id": "test"},
        ))

        snapshot1 = backend.get_snapshot()
        snapshot2 = backend.get_snapshot()

        # Should be separate objects
        assert snapshot1 is not snapshot2

    def test_close_is_idempotent(self):
        """close() can be called multiple times safely."""
        backend = OverwatchBackend()
        backend.start()
        backend.close()
        backend.close()  # Should not raise
```

### Implementation: `src/esper/karn/overwatch/backend.py`

```python
"""Overwatch Backend - OutputBackend for live telemetry.

Implements Nissa's OutputBackend protocol to receive telemetry
events and update the TelemetryAggregator for TUI consumption.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from esper.karn.overwatch.aggregator import TelemetryAggregator

if TYPE_CHECKING:
    from esper.leyline import TelemetryEvent
    from esper.karn.overwatch.schema import TuiSnapshot


class OverwatchBackend:
    """OutputBackend that feeds telemetry to Overwatch TUI.

    Thread-safe: emit() can be called from training thread while
    get_snapshot() is called from UI thread (aggregator handles locking).

    Usage:
        from esper.nissa import get_hub
        from esper.karn.overwatch.backend import OverwatchBackend

        backend = OverwatchBackend()
        get_hub().add_backend(backend)

        # In UI thread
        snapshot = backend.get_snapshot()
    """

    def __init__(self, num_envs: int = 4, max_feed_events: int = 100):
        """Initialize the backend.

        Args:
            num_envs: Expected number of training environments.
            max_feed_events: Maximum events to keep in feed.
        """
        self._aggregator = TelemetryAggregator(
            num_envs=num_envs,
            max_feed_events=max_feed_events,
        )
        self._started = False

    def start(self) -> None:
        """Start the backend (required by OutputBackend protocol)."""
        self._started = True

    def emit(self, event: "TelemetryEvent") -> None:
        """Emit telemetry event to aggregator.

        Args:
            event: The telemetry event to process.
        """
        if not self._started:
            return
        self._aggregator.process_event(event)

    def close(self) -> None:
        """Close the backend (required by OutputBackend protocol)."""
        self._started = False

    def get_snapshot(self) -> "TuiSnapshot":
        """Get current TuiSnapshot for UI rendering.

        Returns:
            Snapshot of current aggregator state.
        """
        return self._aggregator.get_snapshot()
```

### Verification

```bash
PYTHONPATH=src uv run pytest tests/karn/overwatch/test_backend.py -v
```

---

## Task 3: Live Mode App Wiring

**File:** `src/esper/karn/overwatch/app.py` (modify)

Add live mode support with `set_interval` polling.

### Step 0: Understand the delta from existing app.py

The current `app.py` has:
- `__init__` with `replay_path` parameter
- `on_mount` that calls `_init_replay()` or hides replay bar
- Replay-related methods and timers

We need to add:
- `backend` parameter for live mode
- `poll_interval_ms` parameter
- `_live_timer` attribute
- `_init_live()` method
- `_live_poll()` method
- Update `on_mount` to handle live mode

### Test File: `tests/karn/overwatch/test_live_mode.py`

```python
"""Tests for Overwatch live mode."""

import pytest

from esper.karn.overwatch.app import OverwatchApp
from esper.karn.overwatch.backend import OverwatchBackend
from esper.karn.overwatch.schema import (
    TuiSnapshot,
    ConnectionStatus,
    TamiyoState,
)


class TestLiveModeInitialization:
    """Test live mode app initialization."""

    def test_app_accepts_backend_parameter(self):
        """App can be initialized with OverwatchBackend."""
        backend = OverwatchBackend()
        app = OverwatchApp(backend=backend)
        assert app._backend is backend

    def test_app_defaults_to_no_backend(self):
        """App without backend starts in replay/demo mode."""
        app = OverwatchApp()
        assert app._backend is None


class TestLiveModePolling:
    """Test live mode polling behavior."""

    @pytest.mark.asyncio
    async def test_live_mode_starts_polling(self):
        """Live mode starts interval timer on mount."""
        backend = OverwatchBackend()
        backend.start()

        # Emit a training started event
        from esper.leyline import TelemetryEvent, TelemetryEventType
        backend.emit(TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data={"run_id": "test-run", "task": "cifar10"},
        ))

        app = OverwatchApp(backend=backend)

        async with app.run_test() as pilot:
            # App should have started polling
            assert app._live_timer is not None

            # Should have updated from backend
            assert app._snapshot is not None
            assert app._snapshot.run_id == "test-run"


class TestLiveModeUpdates:
    """Test live mode widget updates."""

    @pytest.mark.asyncio
    async def test_widgets_update_from_backend(self):
        """Widgets receive updates from backend snapshots."""
        backend = OverwatchBackend()
        backend.start()

        # Simulate some telemetry
        from esper.leyline import TelemetryEvent, TelemetryEventType
        backend.emit(TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data={"run_id": "live-test", "task": "cifar10", "num_envs": 2},
        ))

        app = OverwatchApp(backend=backend)

        async with app.run_test() as pilot:
            # Wait for poll
            await pilot.pause()

            # Verify snapshot was loaded
            assert app._snapshot is not None
            assert app._snapshot.run_id == "live-test"
```

### Implementation Changes to `app.py`

Add to imports at top:
```python
if TYPE_CHECKING:
    from esper.karn.overwatch.backend import OverwatchBackend
```

Modify `OverwatchApp.__init__`:
```python
def __init__(
    self,
    replay_path: Path | str | None = None,
    backend: "OverwatchBackend | None" = None,
    poll_interval_ms: int = 250,
    **kwargs,
) -> None:
    """Initialize the Overwatch app.

    Args:
        replay_path: Optional path to JSONL replay file
        backend: Optional OverwatchBackend for live mode
        poll_interval_ms: Polling interval for live updates (default: 250ms)
        **kwargs: Additional args passed to App
    """
    super().__init__(**kwargs)
    self._replay_path = Path(replay_path) if replay_path else None
    self._backend = backend
    self._poll_interval_ms = poll_interval_ms
    self._snapshot: TuiSnapshot | None = None
    self._help_visible = False
    self._replay_controller = None
    self._playback_timer = None
    self._live_timer = None
```

Modify `on_mount`:
```python
def on_mount(self) -> None:
    """Called when app is mounted."""
    # Initialize replay controller if replay file provided
    if self._replay_path:
        self._init_replay()
    elif self._backend:
        # Live mode
        self._init_live()
    else:
        # Demo/standalone mode - hide replay bar
        self.query_one(ReplayStatusBar).set_visible(False)

    # Set focus to flight board for navigation
    self.query_one(FlightBoard).focus()
```

Add new methods:
```python
def _init_live(self) -> None:
    """Initialize live telemetry mode."""
    # Hide replay status bar in live mode
    self.query_one(ReplayStatusBar).set_visible(False)

    # Start polling timer
    interval = self._poll_interval_ms / 1000.0
    self._live_timer = self.set_interval(interval, self._live_poll)

    # Initial poll
    self._live_poll()

def _live_poll(self) -> None:
    """Poll backend for latest snapshot."""
    if not self._backend:
        return

    snapshot = self._backend.get_snapshot()

    # Only update if snapshot changed (by captured_at timestamp)
    if snapshot.captured_at != getattr(self._snapshot, "captured_at", None):
        self._snapshot = snapshot
        self._update_all_widgets()
```

### Verification

```bash
PYTHONPATH=src uv run pytest tests/karn/overwatch/test_live_mode.py -v
```

---

## Task 4: Module Exports

**Files:**
- `src/esper/karn/overwatch/__init__.py`
- `src/esper/karn/__init__.py`

### Test: `tests/karn/overwatch/test_widgets.py` (add to existing)

```python
def test_aggregator_exported():
    """TelemetryAggregator is exported."""
    from esper.karn.overwatch import TelemetryAggregator
    assert TelemetryAggregator is not None

def test_backend_exported():
    """OverwatchBackend is exported."""
    from esper.karn.overwatch import OverwatchBackend
    assert OverwatchBackend is not None

def test_karn_exports_overwatch_backend():
    """Karn package exports OverwatchBackend."""
    from esper.karn import OverwatchBackend
    assert OverwatchBackend is not None
```

### Implementation

Update `src/esper/karn/overwatch/__init__.py`:
```python
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

from esper.karn.overwatch.aggregator import TelemetryAggregator
from esper.karn.overwatch.backend import OverwatchBackend

# Lazy import for OverwatchApp - Textual may not be installed
try:
    from esper.karn.overwatch.app import OverwatchApp
except ImportError:
    OverwatchApp = None  # type: ignore[misc, assignment]

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
    # Aggregator & Backend
    "TelemetryAggregator",
    "OverwatchBackend",
    # App (may be None if Textual not installed)
    "OverwatchApp",
]
```

Update `src/esper/karn/__init__.py` to add after existing imports:
```python
# Overwatch (live telemetry backend)
from esper.karn.overwatch.backend import OverwatchBackend
```

And add to `__all__`:
```python
    # Overwatch
    "OverwatchBackend",
```

### Verification

```bash
PYTHONPATH=src uv run pytest tests/karn/overwatch/test_widgets.py::test_aggregator_exported tests/karn/overwatch/test_widgets.py::test_backend_exported -v
```

---

## Task 5: CLI --overwatch Flag

**File:** `src/esper/scripts/train.py` (modify)

Add `--overwatch` flag that launches Overwatch TUI for monitoring.

**IMPORTANT:** This flag is **mutually exclusive** with the existing Rich TUI (`--no-tui` is implied). Overwatch takes control of the terminal while training runs in a background thread.

### Test: Manual CLI test

```bash
# Test that flag is recognized
uv run python -m esper.scripts.train ppo --help | grep overwatch
```

### Implementation Changes

Add to `telemetry_parent` ArgumentParser (around line 48):
```python
telemetry_parent.add_argument(
    "--overwatch",
    action="store_true",
    help="Launch Overwatch TUI for real-time monitoring (replaces Rich TUI)",
)
```

Modify main() to handle --overwatch. The key insight is that **Overwatch needs terminal control**, so we run training in a background thread instead:

After the telemetry hub setup section (around line 220), add:
```python
# Overwatch mode: run training in background, Overwatch controls terminal
if args.overwatch:
    from esper.karn import OverwatchBackend
    from esper.karn.overwatch import OverwatchApp
    import threading

    # Create backend and add to hub
    overwatch_backend = OverwatchBackend()
    hub.add_backend(overwatch_backend)

    # Disable Rich TUI (Overwatch replaces it)
    use_tui = False

    # Define training function to run in background
    def run_training():
        try:
            if args.algorithm == "heuristic":
                validated_slots = validate_slots(args.slots)
                from esper.simic.training import train_heuristic
                train_heuristic(
                    n_episodes=args.episodes,
                    max_epochs=args.max_epochs,
                    max_batches=args.max_batches if args.max_batches > 0 else None,
                    device=args.device,
                    task=args.task,
                    seed=args.seed,
                    slots=validated_slots,
                    telemetry_config=telemetry_config,
                    telemetry_lifecycle_only=args.telemetry_lifecycle_only,
                    min_fossilize_improvement=args.min_fossilize_improvement,
                )
            elif args.algorithm == "ppo":
                # ... PPO training code (same as existing)
                pass
        finally:
            # Signal training complete (Overwatch will detect via staleness)
            pass

    # Start training in background thread
    training_thread = threading.Thread(target=run_training, daemon=True)
    training_thread.start()

    # Run Overwatch TUI (blocks until user quits)
    app = OverwatchApp(backend=overwatch_backend)
    app.run()

    # Clean up
    hub.close()
    return  # Exit after Overwatch closes
```

The full implementation requires restructuring the existing if/else blocks. Here's the cleaner approach - wrap the training execution in a function and call it either directly or in a thread:

```python
def main():
    parser = build_parser()
    args = parser.parse_args()

    # ... existing telemetry config setup ...

    hub = get_hub()

    # Determine UI mode
    import sys
    is_tty = sys.stdout.isatty()
    use_overwatch = args.overwatch
    use_tui = not args.no_tui and is_tty and not use_overwatch

    # ... existing TUI/console backend setup (skip if use_overwatch) ...

    if not use_overwatch:
        if use_tui:
            from esper.karn import TUIOutput
            layout = None if args.tui_layout == "auto" else args.tui_layout
            tui_backend = TUIOutput(force_layout=layout)
            hub.add_backend(tui_backend)
        else:
            hub.add_backend(ConsoleOutput(min_severity=console_min_severity))

    # ... existing file/dir/dashboard backend setup ...

    # Setup Overwatch if requested
    overwatch_backend = None
    if use_overwatch:
        from esper.karn import OverwatchBackend
        overwatch_backend = OverwatchBackend()
        hub.add_backend(overwatch_backend)

    # Add Karn collector
    from esper.karn import get_collector
    karn_collector = get_collector()
    hub.add_backend(karn_collector)

    def run_training():
        """Execute the training algorithm."""
        if args.algorithm == "heuristic":
            validated_slots = validate_slots(args.slots)
            from esper.simic.training import train_heuristic
            train_heuristic(
                n_episodes=args.episodes,
                max_epochs=args.max_epochs,
                max_batches=args.max_batches if args.max_batches > 0 else None,
                device=args.device,
                task=args.task,
                seed=args.seed,
                slots=validated_slots,
                telemetry_config=telemetry_config,
                telemetry_lifecycle_only=args.telemetry_lifecycle_only,
                min_fossilize_improvement=args.min_fossilize_improvement,
            )
        elif args.algorithm == "ppo":
            # ... existing PPO setup and call ...
            pass

    try:
        if use_overwatch:
            # Run training in background, Overwatch in foreground
            import threading
            from esper.karn.overwatch import OverwatchApp

            training_thread = threading.Thread(target=run_training, daemon=True)
            training_thread.start()

            app = OverwatchApp(backend=overwatch_backend)
            app.run()
        else:
            # Normal mode: run training directly
            run_training()
    finally:
        # ... existing cleanup ...
        hub.close()
```

### Verification

```bash
# Should show --overwatch in help
uv run python -m esper.scripts.train ppo --help | grep -A2 overwatch
```

---

## Task 6: Integration Tests - Live Mode End-to-End

**File:** `tests/karn/overwatch/test_integration.py` (add to existing file)

```python
"""Integration tests for Overwatch live mode.

Added to existing test_integration.py from Stage 5.
"""

import pytest
from esper.leyline import TelemetryEvent, TelemetryEventType
from esper.karn.overwatch import OverwatchApp, OverwatchBackend


@pytest.mark.asyncio
async def test_live_mode_full_workflow():
    """Test complete live mode workflow."""
    backend = OverwatchBackend()
    backend.start()

    # Simulate training startup
    backend.emit(TelemetryEvent(
        event_type=TelemetryEventType.TRAINING_STARTED,
        data={
            "run_id": "integration-test",
            "task": "cifar10",
            "num_envs": 2,
            "max_epochs": 75,
        },
    ))

    # Simulate seed germination
    backend.emit(TelemetryEvent(
        event_type=TelemetryEventType.SEED_GERMINATED,
        slot_id="r0c0",
        data={
            "env_id": 0,
            "seed_id": "seed-001",
            "blueprint_id": "conv3x3",
        },
    ))

    # Simulate PPO update
    backend.emit(TelemetryEvent(
        event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
        data={
            "kl_divergence": 0.012,
            "entropy": 1.5,
            "clip_fraction": 0.05,
            "explained_variance": 0.8,
        },
    ))

    app = OverwatchApp(backend=backend)

    async with app.run_test() as pilot:
        await pilot.pause()

        # Verify app state
        assert app._snapshot is not None
        assert app._snapshot.run_id == "integration-test"
        assert app._snapshot.connection.connected is True

        # Verify tamiyo state
        assert app._snapshot.tamiyo.kl_divergence == pytest.approx(0.012)
        assert app._snapshot.tamiyo.entropy == pytest.approx(1.5)

        # Verify flight board has env
        assert len(app._snapshot.flight_board) >= 1
        env0 = app._snapshot.flight_board[0]
        assert "r0c0" in env0.slots
        assert env0.slots["r0c0"].stage == "GERMINATED"

        # Verify event feed has events
        assert len(app._snapshot.event_feed) >= 2  # GERM + PPO


@pytest.mark.asyncio
async def test_live_mode_staleness_detection():
    """Test that staleness is tracked correctly."""
    backend = OverwatchBackend()
    backend.start()

    backend.emit(TelemetryEvent(
        event_type=TelemetryEventType.TRAINING_STARTED,
        data={"run_id": "stale-test"},
    ))

    app = OverwatchApp(backend=backend)

    async with app.run_test() as pilot:
        await pilot.pause()

        # Staleness should be low immediately after event
        assert app._snapshot.connection.staleness_s < 2.0
        assert app._snapshot.connection.connected is True
```

### Verification

```bash
PYTHONPATH=src uv run pytest tests/karn/overwatch/test_integration.py::test_live_mode_full_workflow tests/karn/overwatch/test_integration.py::test_live_mode_staleness_detection -v
```

---

## Task 7: Full Test Suite Verification

Run complete test suite to ensure no regressions.

```bash
# Run all Overwatch tests
PYTHONPATH=src uv run pytest tests/karn/overwatch/ -v

# Run full test suite
PYTHONPATH=src uv run pytest

# Lint check
uv run ruff check src/esper/karn/overwatch/
```

---

## Summary

| Task | Description | New Files | Modified Files |
|------|-------------|-----------|----------------|
| 1 | TelemetryAggregator (with threading.Lock) | `aggregator.py`, `test_aggregator.py` | - |
| 2 | OverwatchBackend | `backend.py`, `test_backend.py` | - |
| 3 | Live Mode Wiring | `test_live_mode.py` | `app.py` |
| 4 | Module Exports | - | `overwatch/__init__.py`, `karn/__init__.py` |
| 5 | CLI --overwatch Flag | - | `train.py` |
| 6 | Integration Tests | - | `test_integration.py` (add to existing) |
| 7 | Full Verification | - | - |

## Notes on RunHeader

The existing `run_header.py` already implements connection status display with:
- `● Live` (green, staleness < 2s)
- `● Live (Xs)` (yellow, staleness 2-5s)
- `◐ Stale (Xs)` (warning, staleness > 5s)
- `○ Disconnected` (red)

No changes needed - Stage 6 provides the data, RunHeader already displays it.

## Next Stage

**Stage 7: Device Metrics & Health Indicators** will:
- Add GPU utilization monitoring (nvidia-smi/pynvml)
- Add per-env health scoring with anomaly detection
- Add DeviceVitals rendering in header
- Add status icons (OK/WARN/CRIT) to FlightBoard
