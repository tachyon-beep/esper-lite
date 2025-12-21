"""Karn Collector - Event collection, validation, and routing.

The collector is the central hub for Karn telemetry:
1. Receives events from emitters (Kasmina, Simic, Tamiyo, Tolaria)
2. Validates and timestamps events
3. Updates the TelemetryStore
4. Routes to output backends

Usage:
    from esper.karn import get_collector

    collector = get_collector()
    collector.emit(event)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol
from uuid import uuid4

from esper.karn.store import (
    TelemetryStore,
    EpisodeContext,
    EpochSnapshot,
    SlotSnapshot,
    PolicySnapshot,
    DenseTraceTrigger,
    SeedStage,
)
from esper.karn.triggers import AnomalyDetector, PolicyAnomalyDetector
from esper.karn.ingest import (
    coerce_bool_or_none,
    coerce_float,
    coerce_float_or_none,
    coerce_int,
    coerce_seed_stage,
    coerce_str,
    coerce_str_or_none,
)

if TYPE_CHECKING:
    from esper.leyline.telemetry import TelemetryEvent

_logger = logging.getLogger(__name__)

# Simic anomaly event types that should trigger dense trace capture
_ANOMALY_EVENT_TYPES = frozenset({
    "RATIO_EXPLOSION_DETECTED",
    "RATIO_COLLAPSE_DETECTED",
    "VALUE_COLLAPSE_DETECTED",
    "ENTROPY_COLLAPSE_DETECTED",
    "GRADIENT_EXPLOSION_DETECTED",
    "KL_DIVERGENCE_SPIKE",
    # Additional Simic anomaly types
    "NUMERICAL_INSTABILITY_DETECTED",
    "GRADIENT_ANOMALY",
})


class OutputBackend(Protocol):
    """Protocol for Karn output backends.

    Note:
        All methods are called by KarnCollector:
        - start() is called by add_backend() before the backend is registered
        - emit() is called for each telemetry event
        - close() is called during collector shutdown

        Backends can implement start() as a no-op if no initialization is needed.
        For convenience, subclass esper.nissa.output.OutputBackend which provides
        default no-op implementations for start() and close().
    """

    def start(self) -> None:
        """Start backend (called by add_backend(), can be no-op)."""
        ...

    def emit(self, event: "TelemetryEvent") -> None:
        """Emit event to this backend."""
        ...

    def close(self) -> None:
        """Close backend and release resources."""
        ...


@dataclass
class KarnConfig:
    """Configuration for Karn collector."""

    # Capture settings
    capture_dense_traces: bool = True
    dense_trigger: DenseTraceTrigger = field(default_factory=DenseTraceTrigger)

    # Fault tolerance
    on_emission_error: str = "warn"  # "ignore", "warn", "halt"


class KarnCollector:
    """Central telemetry collector for Karn.

    Receives events from all subsystems, maintains the TelemetryStore,
    and routes events to output backends.

    This is the Karn equivalent of Nissa's NissaHub, but with:
    - Stateful storage (TelemetryStore)
    - Typed event handling
    - Research-focused analytics integration

    Lifecycle Contract:
        - add_backend(): Starts backend immediately; raises RuntimeError if collector is closed
        - emit(): Routes to backends and updates store; silently drops (with warning) if closed
        - close(): Idempotent; closes all backends and marks collector as closed
        - reset(): Closes backends, clears store/detectors, and reopens collector for reuse
        - start(): No-op (backends are auto-started by add_backend())

        This contract is shared with NissaHub for consistency across telemetry systems.
    """

    def __init__(self, config: KarnConfig | None = None):
        self.config = config or KarnConfig()
        self.store = TelemetryStore()
        self._backends: list[OutputBackend] = []
        self._episode_active = False
        self._closed = False  # Idempotency flag for close()

        # Tier 3: Anomaly detection and dense trace capture
        self._anomaly_detector = AnomalyDetector(config=self.config.dense_trigger)
        self._policy_detector = PolicyAnomalyDetector()

    # =========================================================================
    # Backend Management (mirrors NissaHub interface)
    # =========================================================================

    def add_backend(self, backend: OutputBackend) -> None:
        """Add an output backend.

        The backend is started immediately on add. If start() fails,
        the backend is NOT added and the exception is re-raised
        (fail-fast to prevent half-initialized state and silent misconfiguration).

        Raises:
            RuntimeError: If the collector has been closed. Use reset() to reopen.
            Exception: If backend.start() fails, the original exception is
                logged and re-raised so callers can detect misconfiguration.
        """
        if self._closed:
            raise RuntimeError(
                "Cannot add backend to closed KarnCollector. "
                "Call reset() to reopen the collector before adding backends."
            )
        try:
            backend.start()
            self._backends.append(backend)
        except Exception:
            _logger.exception(f"Failed to start backend {backend}, not adding")
            raise

    def remove_backend(self, backend: OutputBackend) -> None:
        """Remove an output backend."""
        if backend in self._backends:
            self._backends.remove(backend)
            try:
                backend.close()
            except Exception as e:
                _logger.error(f"Error closing backend {backend}: {e}")

    def start(self) -> None:
        """No-op: backends are auto-started by add_backend().

        This method exists for API consistency with OutputBackend protocol
        but does nothing since add_backend() already starts each backend.
        Calling this method is harmless but unnecessary.

        Warning:
            Do NOT rely on this to start backends. Always use add_backend().
        """
        # No-op: backends are started in add_backend() to ensure fail-fast
        # behavior. Double-starting could cause issues with non-idempotent backends.
        pass

    def close(self) -> None:
        """Close all backends (idempotent).

        Safe to call multiple times - only closes backends once.
        """
        if self._closed:
            return
        self._closed = True
        for backend in self._backends:
            try:
                backend.close()
            except Exception as e:
                _logger.error(f"Error closing backend {backend}: {e}")

    def reset(self) -> None:
        """Reset collector state (clear store and backends).

        Warning:
            NOT THREAD-SAFE. Must be called when no other threads are emitting
            events. Intended for test cleanup or between training runs, not
            during active training. Calling reset() while emit() is running
            concurrently may cause dropped events or backend errors.
        """
        self.close()
        self.store = TelemetryStore()
        self._backends.clear()
        self._episode_active = False
        self._closed = False  # Allow collector to be reused after reset
        self._emit_after_close_warned = False
        self._anomaly_detector.reset()
        self._policy_detector.reset()

    # =========================================================================
    # Event Emission (primary interface)
    # =========================================================================

    _emit_after_close_warned = False

    def emit(self, event: "TelemetryEvent") -> None:
        """Emit a telemetry event.

        This is the main entry point for all telemetry. Events are:
        1. Validated
        2. Stored in TelemetryStore (if episode active)
        3. Routed to all output backends

        Note:
            Does nothing if the collector has been closed. This prevents
            sending events to backends that have already been shut down.
        """
        if self._closed:
            if not self._emit_after_close_warned:
                _logger.warning("emit() called on closed KarnCollector (event dropped)")
                self._emit_after_close_warned = True
            return

        try:
            # Update store based on event type
            self._update_store(event)

            # Route to backends
            for backend in self._backends:
                try:
                    backend.emit(event)
                except Exception as e:
                    self._handle_backend_error(backend, e)

        except Exception as e:
            self._handle_emission_error(event, e)

    def _update_store(self, event: "TelemetryEvent") -> None:
        """Update TelemetryStore based on event type."""
        # hasattr AUTHORIZED by John on 2025-12-14 03:30:00 UTC
        # Justification: Serialization - handle both enum and string event_type values
        event_type = (
            event.event_type.name
            if hasattr(event.event_type, "name")
            else str(event.event_type)
        )

        # Auto-start episode on TRAINING_STARTED (Nissa backend integration)
        if event_type == "TRAINING_STARTED":
            self._handle_training_started(event)
            return

        # Skip store updates if no episode active
        if not self._episode_active:
            return

        # Route to appropriate handler
        if event_type == "EPOCH_COMPLETED":
            self._handle_epoch_completed(event)
        elif event_type.startswith("SEED_"):
            self._handle_seed_event(event)
        elif event_type == "PPO_UPDATE_COMPLETED":
            self._handle_ppo_update(event)
        elif event_type == "REWARD_COMPUTED":
            self._handle_reward_computed(event)
        elif event_type == "COUNTERFACTUAL_COMPUTED":
            self._handle_counterfactual_computed(event)
        elif event_type in _ANOMALY_EVENT_TYPES:
            self._handle_anomaly_event(event, event_type)

    def _handle_training_started(self, event: "TelemetryEvent") -> None:
        """Handle TRAINING_STARTED event - auto-initialize episode AND first epoch."""
        data = event.data or {}
        episode_id = data.get("episode_id") or event.event_id
        self.start_episode(
            episode_id=episode_id,
            seed=coerce_int(data.get("seed", 42), field="seed", default=42, minimum=0),
            task_type=coerce_str(
                data.get("task", "classification"),
                field="task",
                default="classification",
            ),
            reward_mode=coerce_str(
                data.get("reward_mode", "shaped"),
                field="reward_mode",
                default="shaped",
            ),
            max_epochs=coerce_int(
                data.get("max_epochs", 75),
                field="max_epochs",
                default=75,
                minimum=1,
            ),
        )
        # Start at epoch 1 to match Simic's range(1, max_epochs + 1)
        self.store.start_epoch(1)
        _logger.debug(f"Auto-started episode and epoch 1 from TRAINING_STARTED: {episode_id}")

    def _handle_epoch_completed(self, event: "TelemetryEvent") -> None:
        """Handle EPOCH_COMPLETED event."""
        data = event.data or {}
        raw_epoch = event.epoch if event.epoch is not None else data.get("epoch", 0)
        epoch = coerce_int(raw_epoch, field="epoch", default=0, minimum=0)

        # Auto-start epoch if none exists
        if not self.store.current_epoch:
            self.store.start_epoch(epoch)

        # Update host snapshot
        # Keep EpochSnapshot.epoch aligned with the commit-barrier epoch identifier.
        # (HostSnapshot.epoch mirrors this value for convenience.)
        self.store.current_epoch.epoch = epoch
        self.store.current_epoch.host.epoch = epoch
        self.store.current_epoch.host.val_loss = coerce_float(
            data.get("val_loss"), field="val_loss", default=0.0
        )
        self.store.current_epoch.host.val_accuracy = coerce_float(
            data.get("val_accuracy"), field="val_accuracy", default=0.0
        )
        self.store.current_epoch.host.train_loss = coerce_float(
            data.get("train_loss"), field="train_loss", default=0.0
        )
        self.store.current_epoch.host.train_accuracy = coerce_float(
            data.get("train_accuracy"), field="train_accuracy", default=0.0
        )
        self.store.current_epoch.host.host_grad_norm = coerce_float(
            data.get("grad_norm"), field="grad_norm", default=0.0
        )

        # P0 Fix: Increment epochs_in_stage for all active slots
        for slot in self.store.current_epoch.slots.values():
            if slot.stage not in (SeedStage.DORMANT, SeedStage.CULLED, SeedStage.FOSSILIZED):
                slot.epochs_in_stage += 1

        # Tier 3: Check for anomalies before committing
        if self.config.capture_dense_traces:
            self._check_anomalies_and_capture(self.store.current_epoch)

        # Commit the epoch
        self.store.commit_epoch()

        # Start next epoch placeholder
        self.store.start_epoch(epoch + 1)

    def _check_anomalies_and_capture(self, snapshot: EpochSnapshot) -> None:
        """Check for anomalies and manage dense trace capture."""
        # Check for anomaly triggers
        trigger_reason = self._anomaly_detector.check_epoch(snapshot)

        # Start new trace if anomaly detected and not already capturing
        if trigger_reason and not self._anomaly_detector.is_capturing:
            self._anomaly_detector.start_trace(snapshot.epoch, trigger_reason)
            _logger.info(f"Dense trace triggered at epoch {snapshot.epoch}: {trigger_reason}")

        # Finalize trace if window ended
        completed_trace = self._anomaly_detector.finalize_trace(snapshot.epoch)
        if completed_trace:
            self.store.add_dense_trace(completed_trace)
            _logger.info(f"Dense trace completed: {completed_trace.trigger_reason}")

    def _handle_seed_event(self, event: "TelemetryEvent") -> None:
        """Handle seed lifecycle events with env-namespaced slots."""
        if not self.store.current_epoch:
            return

        data = event.data or {}

        if "env_id" not in data:
            return
        env_id = coerce_int(data.get("env_id"), field="env_id", default=-1, minimum=0)
        if env_id < 0:
            return

        # Get raw slot_id
        raw_slot_id = event.slot_id if event.slot_id is not None else data.get("slot_id", "unknown")
        slot_id = coerce_str(raw_slot_id, field="slot_id", default="unknown").strip() or "unknown"

        # Namespace slot key by env_id to prevent multi-env collisions
        slot_key = f"env{env_id}:{slot_id}"

        # Get or create slot snapshot with namespaced key
        if slot_key not in self.store.current_epoch.slots:
            self.store.current_epoch.slots[slot_key] = SlotSnapshot(slot_id=slot_key)

        slot = self.store.current_epoch.slots[slot_key]

        # Update based on event type
        # hasattr AUTHORIZED by John on 2025-12-14 03:30:00 UTC
        # Justification: Serialization - handle both enum and string event_type values
        event_type = (
            event.event_type.name
            if hasattr(event.event_type, "name")
            else str(event.event_type)
        )

        if event_type == "SEED_GERMINATED":
            slot.stage = SeedStage.GERMINATED
            slot.seed_id = coerce_str_or_none(data.get("seed_id"), field="seed_id")
            slot.blueprint_id = coerce_str_or_none(data.get("blueprint_id"), field="blueprint_id")
            slot.seed_params = coerce_int(data.get("params", 0), field="params", default=0, minimum=0)
        elif event_type == "SEED_STAGE_CHANGED":
            slot.stage = coerce_seed_stage(data.get("to"), field="to", default=SeedStage.DORMANT)
            slot.epochs_in_stage = 0
        elif event_type == "SEED_GATE_EVALUATED":
            slot.last_gate_attempted = coerce_str_or_none(data.get("gate"), field="gate")
            slot.last_gate_passed = coerce_bool_or_none(data.get("passed"), field="passed")
            slot.last_gate_reason = coerce_str_or_none(data.get("message"), field="message")
            if not slot.last_gate_reason:
                checks_failed = data.get("checks_failed")
                if isinstance(checks_failed, list) and checks_failed:
                    slot.last_gate_reason = ",".join(str(c) for c in checks_failed)
        elif event_type == "SEED_FOSSILIZED":
            slot.stage = SeedStage.FOSSILIZED
        elif event_type == "SEED_CULLED":
            slot.stage = SeedStage.CULLED

    def _handle_ppo_update(self, event: "TelemetryEvent") -> None:
        """Handle PPO_UPDATE_COMPLETED event."""
        if not self.store.current_epoch:
            return

        data = event.data or {}

        # Create policy snapshot if not exists
        if not self.store.current_epoch.policy:
            self.store.current_epoch.policy = PolicySnapshot()

        policy = self.store.current_epoch.policy
        policy.kl_divergence = coerce_float_or_none(data.get("kl_divergence"), field="kl_divergence")
        policy.explained_variance = coerce_float_or_none(
            data.get("explained_variance"), field="explained_variance"
        )
        policy.entropy = coerce_float_or_none(data.get("entropy"), field="entropy")

    def _handle_reward_computed(self, event: "TelemetryEvent") -> None:
        """Handle REWARD_COMPUTED event."""
        if not self.store.current_epoch:
            return

        # Create policy snapshot if not exists
        if not self.store.current_epoch.policy:
            self.store.current_epoch.policy = PolicySnapshot()

        data = event.data or {}
        policy = self.store.current_epoch.policy
        policy.reward_total = coerce_float(data.get("total_reward"), field="total_reward", default=0.0)
        policy.action_op = coerce_str(data.get("action_name"), field="action_name", default="")

    def _handle_counterfactual_computed(self, event: "TelemetryEvent") -> None:
        """Handle COUNTERFACTUAL_COMPUTED event."""
        if not self.store.current_epoch:
            return

        data = event.data or {}
        if "env_id" not in data:
            return
        env_id = coerce_int(data.get("env_id"), field="env_id", default=-1, minimum=0)
        if env_id < 0:
            return
        raw_slot_id = event.slot_id if event.slot_id is not None else data.get("slot_id")
        slot_id = coerce_str(raw_slot_id, field="slot_id", default="").strip()
        if not slot_id:
            return

        slot_key = f"env{env_id}:{slot_id}"
        if slot_key not in self.store.current_epoch.slots:
            self.store.current_epoch.slots[slot_key] = SlotSnapshot(slot_id=slot_key)
        slot = self.store.current_epoch.slots[slot_key]

        available = True
        if "available" in data:
            available_coerced = coerce_bool_or_none(data.get("available"), field="available")
            available = True if available_coerced is None else available_coerced

        if available is False:
            slot.counterfactual_contribution = None
            return

        slot.counterfactual_contribution = coerce_float_or_none(data.get("contribution"), field="contribution")
        slot.total_improvement = coerce_float_or_none(data.get("total_improvement"), field="total_improvement")
        slot.improvement_this_epoch = coerce_float(
            data.get("improvement_this_epoch"), field="improvement_this_epoch", default=0.0
        )

    def _handle_anomaly_event(self, event: "TelemetryEvent", event_type: str) -> None:
        """Handle Simic anomaly events for dense trace capture.

        P1-06 Fix: Use epoch (not episode) for trace windowing.
        """
        if not self.store.current_epoch:
            return

        # P1-06: Prefer epoch field over episode for trace windowing
        data = event.data or {}
        epoch = event.epoch or data.get("epoch", self.store.current_epoch.epoch)

        # Skip if anomaly detection disabled
        if not self.config.capture_dense_traces:
            return

        # Trigger dense trace if not already capturing
        if not self._anomaly_detector.is_capturing:
            self._anomaly_detector.start_trace(epoch, event_type)
            _logger.info(f"Dense trace triggered by {event_type} at epoch {epoch}")

    def _handle_backend_error(self, backend: OutputBackend, error: Exception) -> None:
        """Handle error from output backend."""
        if self.config.on_emission_error == "halt":
            raise error
        elif self.config.on_emission_error == "warn":
            _logger.warning(f"Backend {backend} error: {error}")
        # "ignore" does nothing

    def _handle_emission_error(self, event: "TelemetryEvent", error: Exception) -> None:
        """Handle error during event emission."""
        if self.config.on_emission_error == "halt":
            raise error
        elif self.config.on_emission_error == "warn":
            _logger.warning(f"Emission error for {event.event_type}: {error}")

    # =========================================================================
    # Episode Lifecycle
    # =========================================================================

    def start_episode(
        self,
        episode_id: str | None = None,
        seed: int = 42,
        task_type: str = "classification",
        reward_mode: str = "shaped",
        max_epochs: int = 75,
        **kwargs: Any,
    ) -> EpisodeContext:
        """Start a new training episode.

        Creates EpisodeContext and initializes the store.
        """
        context = EpisodeContext(
            episode_id=episode_id or str(uuid4()),
            base_seed=seed,
            torch_seed=seed,
            numpy_seed=seed,
            task_type=task_type,
            reward_mode=reward_mode,
            max_epochs=max_epochs,
            hyperparameters=tuple(kwargs.items()),
        )

        self.store.start_episode(context)
        self._episode_active = True

        # Reset Tier 3 detectors for new episode
        self._anomaly_detector.reset()
        self._policy_detector.reset()

        _logger.info(f"Started episode {context.episode_id}")
        return context

    def end_episode(self) -> None:
        """End the current episode."""
        self._episode_active = False
        _logger.info("Episode ended")

    def start_epoch(self, epoch: int) -> EpochSnapshot:
        """Start a new epoch within the current episode."""
        return self.store.start_epoch(epoch)

    def commit_epoch(self) -> None:
        """Commit the current epoch snapshot."""
        self.store.commit_epoch()

    # =========================================================================
    # Analytics Access
    # =========================================================================

    @property
    def latest_accuracy(self) -> float:
        """Get the most recent validation accuracy."""
        if self.store.latest_epoch:
            return self.store.latest_epoch.host.val_accuracy
        return 0.0

    @property
    def accuracy_trajectory(self) -> list[tuple[int, float]]:
        """Get (epoch, accuracy) pairs for all stored epochs."""
        return [
            (snap.epoch, snap.host.val_accuracy) for snap in self.store.epoch_snapshots
        ]

    def get_slot_contributions(self) -> dict[str, float]:
        """Get current counterfactual contributions per slot."""
        if not self.store.latest_epoch:
            return {}

        return {
            slot_id: slot.counterfactual_contribution or 0.0
            for slot_id, slot in self.store.latest_epoch.slots.items()
            if slot.counterfactual_contribution is not None
        }


# =============================================================================
# Global Collector Instance
# =============================================================================

_global_collector: KarnCollector | None = None


def get_collector() -> KarnCollector:
    """Get or create the global KarnCollector instance."""
    global _global_collector
    if _global_collector is None:
        _global_collector = KarnCollector()
    return _global_collector


def configure(config: KarnConfig) -> KarnCollector:
    """Configure and return the global collector."""
    global _global_collector
    _global_collector = KarnCollector(config)
    return _global_collector


def reset_collector() -> None:
    """Reset the global KarnCollector instance.

    Resets state and closes backends. Useful for test cleanup.
    """
    global _global_collector
    if _global_collector is not None:
        _global_collector.reset()


def emit(event: "TelemetryEvent") -> None:
    """Emit event to the global collector (convenience function)."""
    get_collector().emit(event)
