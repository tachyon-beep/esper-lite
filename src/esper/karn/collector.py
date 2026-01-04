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
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from esper.karn.store import (
    TelemetryStore,
    EpisodeContext,
    EpochSnapshot,
    SlotSnapshot,
    PolicySnapshot,
    DenseTraceTrigger,
)
from esper.karn.triggers import AnomalyDetector, PolicyAnomalyDetector
from esper.karn.ingest import (
    coerce_int,
    coerce_seed_stage,
)
from esper.nissa.output import OutputBackend
from esper.leyline.telemetry import (
    TrainingStartedPayload,
    EpochCompletedPayload,
    BatchEpochCompletedPayload,
    PPOUpdatePayload,
    SeedGerminatedPayload,
    SeedStageChangedPayload,
    SeedGateEvaluatedPayload,
    SeedFossilizedPayload,
    SeedPrunedPayload,
    AnomalyDetectedPayload,
    AnalyticsSnapshotPayload,
)
from esper.leyline import SeedStage

if TYPE_CHECKING:
    from esper.leyline.telemetry import TelemetryEvent

_logger = logging.getLogger(__name__)

# Simic anomaly event types that should trigger dense trace capture
_ANOMALY_EVENT_TYPES = frozenset({
    "RATIO_EXPLOSION_DETECTED",
    "RATIO_COLLAPSE_DETECTED",
    "VALUE_COLLAPSE_DETECTED",
    "NUMERICAL_INSTABILITY_DETECTED",
    "GRADIENT_ANOMALY",
    "GRADIENT_PATHOLOGY_DETECTED",
})


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

        # Multi-env epoch handling: buffer per-env metrics until all envs report
        # Key: (inner_epoch, env_id), Value: EpochCompletedPayload
        # This prevents committing N times per inner epoch (once per env).
        # We commit when len(buffer for epoch X) == n_envs.
        self._pending_epoch_metrics: dict[tuple[int, int], EpochCompletedPayload] = {}
        self._n_envs: int = 1  # Set by TRAINING_STARTED, default 1 for single-env
        self._saw_epoch_completed_since_batch: bool = False

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
        # Multi-env aggregation state
        self._pending_epoch_metrics.clear()  # Clear stale buffered epochs
        self._n_envs = 1  # Reset to default single-env
        self._saw_epoch_completed_since_batch = False

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
        event_type = event.event_type.name

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
        elif event_type == "BATCH_EPOCH_COMPLETED":
            self._handle_batch_epoch_completed(event)
        elif event_type.startswith("SEED_"):
            self._handle_seed_event(event)
        elif event_type == "PPO_UPDATE_COMPLETED":
            self._handle_ppo_update(event)
        elif event_type in _ANOMALY_EVENT_TYPES:
            self._handle_anomaly_event(event, event_type)
        elif event_type == "ANALYTICS_SNAPSHOT":
            self._handle_analytics_snapshot(event)

    def _handle_training_started(self, event: "TelemetryEvent") -> None:
        """Handle TRAINING_STARTED event - auto-initialize episode AND first epoch."""
        # Typed payload path
        if isinstance(event.data, TrainingStartedPayload):
            episode_id = event.data.episode_id or event.event_id
            # Capture n_envs for multi-env epoch commit logic
            self._n_envs = event.data.n_envs or 1
            self._pending_epoch_metrics.clear()  # Reset buffer for new training run
            self._saw_epoch_completed_since_batch = False
            self.start_episode(
                episode_id=episode_id,
                seed=event.data.seed,
                task_type=event.data.task,
                reward_mode=event.data.reward_mode or "shaped",
                max_epochs=event.data.max_epochs,
            )
        else:
            _logger.warning(f"Unknown data type in TRAINING_STARTED: {type(event.data)}")
            return
        # Start at epoch 1 to match Simic's range(1, max_epochs + 1)
        self.store.start_epoch(1)
        _logger.debug(f"Auto-started episode and epoch 1 from TRAINING_STARTED: {episode_id}")

    def _handle_epoch_completed(self, event: "TelemetryEvent") -> None:
        """Handle per-env EPOCH_COMPLETED event - buffer and commit when all envs report.

        EPOCH_COMPLETED is emitted once per environment per inner epoch. With n_envs
        environments, we receive n_envs EPOCH_COMPLETED events per inner epoch.

        This handler buffers per-env metrics keyed by (inner_epoch, env_id). When all
        n_envs have reported for a given inner_epoch, we aggregate and commit once.

        For single-env runs (n_envs=1), this commits immediately on each event.
        """
        if not isinstance(event.data, EpochCompletedPayload):
            _logger.warning(
                "Expected EpochCompletedPayload for EPOCH_COMPLETED, got %s",
                type(event.data).__name__,
            )
            return

        payload = event.data
        env_id = coerce_int(payload.env_id, field="env_id", default=0, minimum=0)
        raw_epoch = event.epoch if event.epoch is not None else payload.inner_epoch
        inner_epoch = coerce_int(raw_epoch, field="epoch", default=0, minimum=0)

        # Buffer per-env metrics keyed by (inner_epoch, env_id)
        self._pending_epoch_metrics[(inner_epoch, env_id)] = payload
        self._saw_epoch_completed_since_batch = True

        # Auto-start epoch if none exists (first event)
        if not self.store.current_epoch:
            self.store.start_epoch(inner_epoch)

        # Check if all envs have reported for this inner_epoch
        envs_for_this_epoch = [
            key for key in self._pending_epoch_metrics.keys()
            if key[0] == inner_epoch
        ]

        if len(envs_for_this_epoch) >= self._n_envs:
            # All envs have reported for this inner_epoch - aggregate and commit
            self._commit_epoch_from_buffer(inner_epoch)

    def _commit_epoch_from_buffer(self, inner_epoch: int) -> None:
        """Aggregate buffered per-env metrics and commit the epoch.

        Called when all n_envs have reported for a given inner_epoch.
        """
        current_epoch = self.store.current_epoch
        if current_epoch is None:
            return

        # Collect payloads for this inner_epoch
        payloads_for_epoch = [
            payload for (epoch, _), payload in self._pending_epoch_metrics.items()
            if epoch == inner_epoch
        ]

        if not payloads_for_epoch:
            return

        n_envs = len(payloads_for_epoch)

        # Required metrics - always present, sum over all envs
        total_val_loss = 0.0
        total_val_accuracy = 0.0

        # Optional metrics - track count of non-None values separately
        # None means "not computed", 0.0 means "computed as zero"
        total_train_loss = 0.0
        train_loss_count = 0
        total_train_accuracy = 0.0
        train_accuracy_count = 0
        total_host_grad_norm = 0.0
        host_grad_norm_count = 0

        for payload in payloads_for_epoch:
            total_val_loss += payload.val_loss
            total_val_accuracy += payload.val_accuracy
            # Only accumulate non-None values and track count
            if payload.train_loss is not None:
                total_train_loss += payload.train_loss
                train_loss_count += 1
            if payload.train_accuracy is not None:
                total_train_accuracy += payload.train_accuracy
                train_accuracy_count += 1
            if payload.host_grad_norm is not None:
                total_host_grad_norm += payload.host_grad_norm
                host_grad_norm_count += 1

        # Update host snapshot with aggregated (mean) metrics
        current_epoch.epoch = inner_epoch
        current_epoch.host.epoch = inner_epoch
        current_epoch.host.val_loss = total_val_loss / n_envs
        current_epoch.host.val_accuracy = total_val_accuracy / n_envs

        # Optional metrics: mean of present values, or None if all were None
        current_epoch.host.train_loss = (
            total_train_loss / train_loss_count if train_loss_count > 0 else None
        )
        current_epoch.host.train_accuracy = (
            total_train_accuracy / train_accuracy_count if train_accuracy_count > 0 else None
        )
        current_epoch.host.host_grad_norm = (
            total_host_grad_norm / host_grad_norm_count if host_grad_norm_count > 0 else None
        )

        # Tier 3: Check for anomalies before mutating slot stage timers.
        # Stage-transition triggers rely on epochs_in_stage == 0 (just transitioned),
        # which would be masked if we increment before checking.
        if self.config.capture_dense_traces:
            self._check_anomalies_and_capture(current_epoch)

        # Increment epochs_in_stage for all occupied slots ONCE per epoch.
        # Only DORMANT is excluded (truly empty slot); PRUNED/FOSSILIZED are
        # terminal but still track dwell time for analytics.
        for slot in current_epoch.slots.values():
            if slot.stage != SeedStage.DORMANT:
                slot.epochs_in_stage += 1

        # Commit the epoch
        self.store.commit_epoch()

        # Start next epoch placeholder
        self.store.start_epoch(inner_epoch + 1)

        # Clear buffer entries for this inner_epoch only
        keys_to_remove = [key for key in self._pending_epoch_metrics if key[0] == inner_epoch]
        for key in keys_to_remove:
            del self._pending_epoch_metrics[key]

    def _handle_batch_epoch_completed(self, event: "TelemetryEvent") -> None:
        """Handle BATCH_EPOCH_COMPLETED - flush any remaining buffered epochs.

        BATCH_EPOCH_COMPLETED is emitted once per episode (batch of N envs).
        It serves two purposes:
        1. Flush partial batches: The last batch may have fewer envs than n_envs,
           so the barrier in _handle_epoch_completed never triggers. This handler
           commits any remaining buffered epochs.
        2. Minimal telemetry fallback: If EPOCH_COMPLETED was gated (ops_normal off),
           create a single epoch snapshot from BATCH aggregate metrics.
        """
        if not isinstance(event.data, BatchEpochCompletedPayload):
            return

        payload = event.data
        did_flush = False

        # Case 1: Flush any remaining buffered epochs (partial batch)
        if self._pending_epoch_metrics:
            # Get unique inner_epochs that have buffered data
            buffered_epochs = sorted(set(key[0] for key in self._pending_epoch_metrics))
            for inner_epoch in buffered_epochs:
                self._commit_epoch_from_buffer(inner_epoch)
            did_flush = True

        # Case 2: Minimal telemetry fallback - no EPOCH_COMPLETED events received
        # Create a single epoch snapshot from BATCH aggregate metrics
        if not did_flush and not self._saw_epoch_completed_since_batch:
            if self.store.current_epoch is None:
                self._saw_epoch_completed_since_batch = False
                return

            current_epoch = self.store.current_epoch
            # batch_idx is emitted 1-indexed
            epoch_num = payload.batch_idx
            current_epoch.epoch = epoch_num
            current_epoch.host.epoch = epoch_num
            current_epoch.host.val_accuracy = payload.avg_accuracy
            # BATCH doesn't have val_loss - leave at default 0.0

            # Increment epochs_in_stage for all occupied slots (only DORMANT excluded)
            for slot in current_epoch.slots.values():
                if slot.stage != SeedStage.DORMANT:
                    slot.epochs_in_stage += 1

            # Commit and start next
            self.store.commit_epoch()
            self.store.start_epoch(epoch_num + 1)

        self._saw_epoch_completed_since_batch = False

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

        event_type = event.event_type.name

        # Extract env_id and slot_id
        env_id: int = -1
        slot_id: str = "unknown"

        # Typed payload path
        if isinstance(event.data, (SeedGerminatedPayload, SeedStageChangedPayload, SeedGateEvaluatedPayload, SeedFossilizedPayload, SeedPrunedPayload)):
            env_id = event.data.env_id
            # Event envelope slot_id is authoritative (set at emission time)
            slot_id = event.slot_id if event.slot_id else event.data.slot_id
        else:
            return

        if env_id < 0:
            return

        # Namespace slot key by env_id to prevent multi-env collisions
        slot_key = f"env{env_id}:{slot_id}"

        # Get or create slot snapshot with namespaced key
        if slot_key not in self.store.current_epoch.slots:
            self.store.current_epoch.slots[slot_key] = SlotSnapshot(slot_id=slot_key)

        slot = self.store.current_epoch.slots[slot_key]

        # Update based on event type
        if event_type == "SEED_GERMINATED":
            if isinstance(event.data, SeedGerminatedPayload):
                slot.stage = SeedStage.GERMINATED
                slot.seed_id = event.seed_id  # From TelemetryEvent, not payload
                slot.blueprint_id = event.data.blueprint_id
                slot.seed_params = event.data.params
        elif event_type == "SEED_STAGE_CHANGED":
            if isinstance(event.data, SeedStageChangedPayload):
                new_stage = coerce_seed_stage(event.data.to_stage, field="to", default=slot.stage)
                if new_stage != slot.stage:
                    slot.stage = new_stage
                    slot.epochs_in_stage = 0
        elif event_type == "SEED_GATE_EVALUATED":
            if isinstance(event.data, SeedGateEvaluatedPayload):
                slot.last_gate_attempted = event.data.gate
                slot.last_gate_passed = event.data.passed
                slot.last_gate_reason = event.data.message
                if not slot.last_gate_reason:
                    if event.data.checks_failed:
                        slot.last_gate_reason = ",".join(str(c) for c in event.data.checks_failed)
        elif event_type == "SEED_FOSSILIZED":
            slot.stage = SeedStage.FOSSILIZED
            slot.epochs_in_stage = 0  # Reset to trigger "just transitioned" detection
        elif event_type == "SEED_PRUNED":
            slot.stage = SeedStage.PRUNED
            slot.epochs_in_stage = 0

    def _handle_ppo_update(self, event: "TelemetryEvent") -> None:
        """Handle PPO_UPDATE_COMPLETED event."""
        if not self.store.current_epoch:
            return

        # Create policy snapshot if not exists
        if not self.store.current_epoch.policy:
            self.store.current_epoch.policy = PolicySnapshot()

        policy = self.store.current_epoch.policy

        # Typed payload path
        if isinstance(event.data, PPOUpdatePayload):
            policy.kl_divergence = event.data.kl_divergence
            policy.explained_variance = event.data.explained_variance
            policy.entropy = event.data.entropy
        else:
            _logger.warning(
                "Expected PPOUpdatePayload for PPO_UPDATE_COMPLETED, got %s",
                type(event.data).__name__,
            )
            return

    def _handle_analytics_snapshot(self, event: "TelemetryEvent") -> None:
        """Handle ANALYTICS_SNAPSHOT event for per-step policy data.

        Specifically handles kind="last_action" to populate PolicySnapshot with
        action/reward data. This replaces the removed REWARD_COMPUTED handler.
        """
        if not self.store.current_epoch:
            return

        # Only handle typed AnalyticsSnapshotPayload
        if not isinstance(event.data, AnalyticsSnapshotPayload):
            return

        payload = event.data

        # Only handle kind="last_action" for policy snapshot updates
        if payload.kind != "last_action":
            return

        # Create policy snapshot if not exists
        if not self.store.current_epoch.policy:
            self.store.current_epoch.policy = PolicySnapshot()

        policy = self.store.current_epoch.policy

        # Populate fields from payload (mirrors old REWARD_COMPUTED handling)
        if payload.total_reward is not None:
            policy.reward_total = payload.total_reward
        if payload.action_name:
            policy.action_op = payload.action_name
        if payload.value_estimate is not None:
            policy.value_estimate = payload.value_estimate

    def _handle_anomaly_event(self, event: "TelemetryEvent", event_type: str) -> None:
        """Handle Simic anomaly events for dense trace capture.

        P1-06 Fix: Use epoch (not episode) for trace windowing.
        """
        if not self.store.current_epoch:
            return

        # P1-06: Prefer epoch field over episode for trace windowing
        # Anomaly events use AnomalyDetectedPayload with episode field
        epoch = event.epoch
        if epoch is None and isinstance(event.data, AnomalyDetectedPayload):
            # AnomalyDetectedPayload uses 'episode' field, not 'epoch'
            epoch = event.data.episode
        elif epoch is None:
            epoch = self.store.current_epoch.epoch

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
            slot_id: slot.counterfactual_contribution  # Already filtered for not None
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
