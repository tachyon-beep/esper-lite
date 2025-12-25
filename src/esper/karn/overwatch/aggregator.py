"""Telemetry Aggregator - Transforms event stream into TuiSnapshot.

Maintains stateful accumulation of telemetry events to build
real-time TuiSnapshot objects for the Overwatch TUI.

Thread-safe: Uses threading.Lock to protect state during concurrent
access from training thread (emit) and UI thread (get_snapshot).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

_logger = logging.getLogger(__name__)

from esper.karn.overwatch.schema import (
    TuiSnapshot,
    ConnectionStatus,
    TamiyoState,
    EnvSummary,
    SlotChipState,
    FeedEvent,
)
from esper.leyline import (
    TrainingStartedPayload,
    BatchEpochCompletedPayload,
    PPOUpdatePayload,
    EpochCompletedPayload,
    SeedGerminatedPayload,
    SeedStageChangedPayload,
    SeedGateEvaluatedPayload,
    SeedFossilizedPayload,
    SeedPrunedPayload,
    RewardComputedPayload,
    AnalyticsSnapshotPayload,
)

if TYPE_CHECKING:
    from esper.leyline import TelemetryEvent


def _parse_cuda_device_index(device: str) -> int | None:
    """Parse a device string like 'cuda:0' into an integer index."""
    if not device:
        return None
    lower = device.lower()
    if not lower.startswith("cuda:"):
        return None
    suffix = lower.split("cuda:", 1)[1]
    try:
        return int(suffix)
    except ValueError:
        return None


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

    def __post_init__(self) -> None:
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
        if isinstance(event.data, TrainingStartedPayload):
            payload = event.data
            self._run_id = payload.episode_id
            self._task_name = payload.task
            self._connected = True
            self._start_time = time.time()

            # Initialize env summaries
            n_envs = payload.n_envs
            self.num_envs = n_envs
            env_devices = payload.env_devices
            for env_id in range(n_envs):
                if env_id not in self._envs:
                    device_id = 0
                    if env_id < len(env_devices):
                        parsed_device_id = _parse_cuda_device_index(str(env_devices[env_id]))
                        if parsed_device_id is not None:
                            device_id = parsed_device_id
                    self._envs[env_id] = EnvSummary(
                        env_id=env_id,
                        device_id=device_id,
                        status="OK",
                    )
        else:
            return

    def _handle_batch_epoch_completed(self, event: "TelemetryEvent") -> None:
        """Handle BATCH_EPOCH_COMPLETED event."""
        if isinstance(event.data, BatchEpochCompletedPayload):
            payload = event.data
            self._batch = payload.batch_idx
            self._episode = payload.episodes_completed

            # avg_accuracy comes as 0-100, store as 0-1 for consistency with run_header
            avg_acc = payload.avg_accuracy
            avg_acc_normalized = avg_acc / 100.0 if avg_acc > 1.0 else avg_acc
            if avg_acc_normalized > self._best_metric:
                self._best_metric = avg_acc_normalized

            # Update per-env accuracies if provided
            if payload.env_accuracies is not None:
                for i, acc in enumerate(payload.env_accuracies):
                    if i in self._envs:
                        self._envs[i].task_metric = acc
        else:
            return

    def _handle_ppo_update_completed(self, event: "TelemetryEvent") -> None:
        """Handle PPO_UPDATE_COMPLETED event."""
        if isinstance(event.data, PPOUpdatePayload):
            payload = event.data

            if payload.skipped:
                return

            self._tamiyo.kl_divergence = payload.kl_divergence
            self._tamiyo.entropy = payload.entropy
            self._tamiyo.clip_fraction = payload.clip_fraction
            # Use explicit None checks to distinguish "not computed" from "zero"
            self._tamiyo.explained_variance = (
                payload.explained_variance if payload.explained_variance is not None else 0.0
            )
            self._tamiyo.grad_norm = payload.grad_norm
            self._tamiyo.learning_rate = payload.lr if payload.lr is not None else 0.0
            self._tamiyo.entropy_collapsed = payload.entropy_collapsed

            # Add PPO feed event for significant updates
            self._add_feed_event(
                event_type="PPO",
                env_id=None,
                message=f"KL={self._tamiyo.kl_divergence:.4f} H={self._tamiyo.entropy:.2f}",
                timestamp=event.timestamp,
            )
        else:
            return

    def _handle_epoch_completed(self, event: "TelemetryEvent") -> None:
        """Handle EPOCH_COMPLETED event."""
        if isinstance(event.data, EpochCompletedPayload):
            payload = event.data
            env_id = payload.env_id

            self._ensure_env(env_id)
            env = self._envs[env_id]
            env.task_metric = payload.val_accuracy
            env.last_update_ts = self._last_event_ts

            # Per-slot telemetry (emitted in vectorized training) updates alpha/stage live.
            seeds = payload.seeds
            if isinstance(seeds, dict):
                for slot_id, info in seeds.items():
                    if not isinstance(slot_id, str) or not isinstance(info, dict):
                        continue
                    stage = str(info.get("stage", "UNKNOWN"))
                    blueprint_id = str(info.get("blueprint_id", ""))
                    alpha = float(info.get("alpha", 0.0))
                    epochs_in_stage = int(info.get("epochs_in_stage", 0))

                    if slot_id not in env.slots:
                        env.slots[slot_id] = SlotChipState(
                            slot_id=slot_id,
                            stage=stage,
                            blueprint_id=blueprint_id,
                            alpha=alpha,
                            epochs_in_stage=epochs_in_stage,
                        )
                    else:
                        chip = env.slots[slot_id]
                        chip.stage = stage
                        chip.blueprint_id = blueprint_id
                        chip.alpha = alpha
                        chip.epochs_in_stage = epochs_in_stage
        else:
            return

    def _handle_seed_germinated(self, event: "TelemetryEvent") -> None:
        """Handle SEED_GERMINATED event."""
        if isinstance(event.data, SeedGerminatedPayload):
            payload = event.data
            env_id = payload.env_id
            slot_id = event.slot_id or payload.slot_id

            self._ensure_env(env_id)
            self._envs[env_id].slots[slot_id] = SlotChipState(
                slot_id=slot_id,
                stage="GERMINATED",
                blueprint_id=payload.blueprint_id,
                alpha=payload.alpha,
            )

            self._add_feed_event(
                event_type="GERM",
                env_id=env_id,
                message=f"{slot_id} germinated ({payload.blueprint_id})",
                timestamp=event.timestamp,
            )
        else:
            return

    def _handle_seed_stage_changed(self, event: "TelemetryEvent") -> None:
        """Handle SEED_STAGE_CHANGED event."""
        if isinstance(event.data, SeedStageChangedPayload):
            payload = event.data
            env_id = payload.env_id
            slot_id = event.slot_id or payload.slot_id

            self._ensure_env(env_id)
            if slot_id in self._envs[env_id].slots:
                self._envs[env_id].slots[slot_id].stage = payload.to_stage
                self._envs[env_id].slots[slot_id].epochs_in_stage = payload.epochs_in_stage

            self._add_feed_event(
                event_type="STAGE",
                env_id=env_id,
                message=f"{slot_id}: {payload.from_stage} -> {payload.to_stage}",
                timestamp=event.timestamp,
            )
        else:
            return

    def _handle_seed_gate_evaluated(self, event: "TelemetryEvent") -> None:
        """Handle SEED_GATE_EVALUATED event."""
        if isinstance(event.data, SeedGateEvaluatedPayload):
            payload = event.data
            env_id = payload.env_id
            slot_id = event.slot_id or payload.slot_id

            self._ensure_env(env_id)
            if slot_id in self._envs[env_id].slots:
                slot = self._envs[env_id].slots[slot_id]
                slot.gate_last = payload.gate
                slot.gate_passed = payload.passed

            status = "PASS" if payload.passed else "FAIL"
            self._add_feed_event(
                event_type="GATE",
                env_id=env_id,
                message=f"{slot_id} {payload.gate}: {status}",
                timestamp=event.timestamp,
            )
        else:
            return

    def _handle_seed_fossilized(self, event: "TelemetryEvent") -> None:
        """Handle SEED_FOSSILIZED event."""
        if isinstance(event.data, SeedFossilizedPayload):
            payload = event.data
            env_id = payload.env_id
            slot_id = event.slot_id or payload.slot_id

            self._ensure_env(env_id)
            if slot_id in self._envs[env_id].slots:
                self._envs[env_id].slots[slot_id].stage = "FOSSILIZED"

            self._add_feed_event(
                event_type="STAGE",
                env_id=env_id,
                message=f"{slot_id} fossilized",
                timestamp=event.timestamp,
            )
        else:
            return

    def _handle_seed_pruned(self, event: "TelemetryEvent") -> None:
        """Handle SEED_PRUNED event."""
        if isinstance(event.data, SeedPrunedPayload):
            payload = event.data
            env_id = payload.env_id
            slot_id = event.slot_id or payload.slot_id

            self._ensure_env(env_id)
            if slot_id in self._envs[env_id].slots:
                self._envs[env_id].slots[slot_id].stage = "PRUNED"

            reason = payload.reason
            self._add_feed_event(
                event_type="PRUNE",
                env_id=env_id,
                message=f"{slot_id} pruned" + (f" ({reason})" if reason else ""),
                timestamp=event.timestamp,
            )
        else:
            return

    def _handle_reward_computed(self, event: "TelemetryEvent") -> None:
        """Handle REWARD_COMPUTED event."""
        if isinstance(event.data, RewardComputedPayload):
            payload = event.data
            env_id = payload.env_id

            self._ensure_env(env_id)
            self._envs[env_id].reward_last = payload.total_reward
            if payload.val_acc is not None:
                self._envs[env_id].task_metric = payload.val_acc
        else:
            return

    def _handle_governor_panic(self, event: "TelemetryEvent") -> None:
        """Handle GOVERNOR_PANIC event.

        Note: This event does not yet have a typed payload.
        TODO: Create GovernorPanicPayload and migrate this handler.
        """
        # No typed payload yet - this event is sent with raw dict data
        if event.data is None:
            data: dict[str, float | int] = {}
        elif isinstance(event.data, dict):
            data = event.data
        else:
            return

        self._add_feed_event(
            event_type="CRIT",
            env_id=None,
            message=f"PANIC #{data.get('consecutive_panics', '?')}: loss={data.get('current_loss', '?')}",
            timestamp=event.timestamp,
        )

    def _handle_governor_rollback(self, event: "TelemetryEvent") -> None:
        """Handle GOVERNOR_ROLLBACK event.

        Note: This event does not yet have a typed payload.
        TODO: Create GovernorRollbackPayload and migrate this handler.
        """
        # No typed payload yet - this event is sent with raw dict data
        if event.data is None:
            data: dict[str, str] = {}
        elif isinstance(event.data, dict):
            data = event.data
        else:
            return

        self._add_feed_event(
            event_type="CRIT",
            env_id=None,
            message=f"ROLLBACK: {data.get('reason', 'unknown')}",
            timestamp=event.timestamp,
        )

    def _handle_analytics_snapshot(self, event: "TelemetryEvent") -> None:
        """Handle ANALYTICS_SNAPSHOT events (UI wiring for high-frequency metrics)."""
        if not isinstance(event.data, AnalyticsSnapshotPayload):
            _logger.warning(
                "Expected AnalyticsSnapshotPayload for ANALYTICS_SNAPSHOT, got %s",
                type(event.data).__name__,
            )
            return

        payload = event.data
        kind = payload.kind

        if kind == "action_distribution":
            if payload.action_counts is not None:
                self._tamiyo.action_counts = {
                    str(action): int(count) for action, count in payload.action_counts.items()
                }
            return

        if kind == "last_action":
            op = payload.action_name
            if isinstance(op, str) and op:
                code = op[0].upper()
                if op.upper() == "WAIT":
                    code = "W"
                elif op.upper() == "GERMINATE":
                    code = "G"
                elif op.upper() == "PRUNE":
                    code = "P"
                elif op.upper() == "FOSSILIZE":
                    code = "F"
                elif op.upper() == "SET_ALPHA_TARGET":
                    code = "A"
                self._tamiyo.recent_actions.append(code)
                self._tamiyo.recent_actions = self._tamiyo.recent_actions[-20:]
                # Keep counts roughly in sync even before the batch-level summary arrives.
                self._tamiyo.action_counts[op] = self._tamiyo.action_counts.get(op, 0) + 1
            return

        if kind == "throughput":
            env_id = payload.env_id
            if env_id is not None:
                self._ensure_env(env_id)
                env = self._envs[env_id]
                if payload.fps is not None:
                    env.throughput_fps = payload.fps
                if payload.step_time_ms is not None:
                    env.step_time_ms = payload.step_time_ms
                # Update staleness timestamp so flight-board shows fresh data
                env.last_update_ts = self._last_event_ts
            return

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
