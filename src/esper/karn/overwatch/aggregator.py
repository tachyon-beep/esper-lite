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
        timestamp: datetime | None = None,
    ) -> None:
        """Add event to feed, maintaining max size."""
        ts = timestamp if timestamp is not None else datetime.now(timezone.utc)
        ts_str = ts.strftime("%H:%M:%S")

        self._feed.append(FeedEvent(
            timestamp=ts_str,
            event_type=event_type,
            env_id=env_id,
            message=message,
        ))

        # Trim to max size
        if len(self._feed) > self.max_feed_events:
            self._feed = self._feed[-self.max_feed_events:]
