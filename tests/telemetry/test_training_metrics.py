"""End-to-end tests for training metrics (TELE-001 to TELE-020).

This module contains TWO types of tests:

1. TRANSPORT TESTS (passing): Verify NissaHub correctly routes events to backends.
   These tests manually create payloads and verify hub→backend transport works.
   They do NOT verify that real training code emits these events.

2. WIRING TESTS (xfailing): Verify the FULL data flow from source to consumer.
   These tests should verify: emitter→hub→aggregator→snapshot
   They are marked xfail because the wiring is not yet tested end-to-end.

The TELE records covered:
- TELE-001: task_name (TrainingStartedPayload.task → SanctumSnapshot.task_name)
- TELE-010: current_episode (BatchEpochCompletedPayload.episodes_completed → snapshot.current_episode)
- TELE-011: current_epoch (EpochCompletedPayload.inner_epoch → snapshot.current_epoch)
- TELE-012: max_epochs (TrainingStartedPayload.max_epochs → snapshot.max_epochs)
- TELE-013: current_batch (BatchEpochCompletedPayload.batch_idx → snapshot.current_batch)
- TELE-014: max_batches (TrainingStartedPayload.max_batches → snapshot.max_batches)
- TELE-020: runtime_seconds (computed from _start_time in aggregator → snapshot.runtime_seconds)
- TELE-021: episode_return_history (BatchEpochCompletedPayload.avg_reward)
"""

import pytest

from esper.leyline import TelemetryEvent, TelemetryEventType
from esper.leyline.telemetry import (
    TrainingStartedPayload,
    BatchEpochCompletedPayload,
    EpochCompletedPayload,
)

from tests.telemetry.conftest import CaptureHubResult


# =============================================================================
# Helper: Create minimal valid payloads
# =============================================================================


def _minimal_training_started(**overrides) -> TrainingStartedPayload:
    """Create minimal valid TrainingStartedPayload with optional overrides."""
    defaults = dict(
        n_envs=4,
        max_epochs=150,
        max_batches=100,
        task="test_task",
        host_params=1_000_000,
        slot_ids=("r0c0",),
        seed=42,
        n_episodes=100,
        lr=3e-4,
        clip_ratio=0.2,
        entropy_coef=0.01,
        param_budget=10_000_000,
        policy_device="cpu",
        env_devices=("cpu",),
        reward_mode="shaped",
    )
    defaults.update(overrides)
    return TrainingStartedPayload(**defaults)


def _minimal_batch_epoch_completed(**overrides) -> BatchEpochCompletedPayload:
    """Create minimal valid BatchEpochCompletedPayload with optional overrides."""
    defaults = dict(
        episodes_completed=10,
        batch_idx=5,
        avg_accuracy=75.0,
        avg_reward=0.5,
        total_episodes=100,
        n_envs=4,
    )
    defaults.update(overrides)
    return BatchEpochCompletedPayload(**defaults)


def _minimal_epoch_completed(**overrides) -> EpochCompletedPayload:
    """Create minimal valid EpochCompletedPayload with optional overrides."""
    defaults = dict(
        env_id=0,
        val_accuracy=80.0,
        val_loss=0.5,
        inner_epoch=10,
    )
    defaults.update(overrides)
    return EpochCompletedPayload(**defaults)


# =============================================================================
# TELE-001: Task Name - TRANSPORT TESTS
# =============================================================================


class TestTELE001TaskNameTransport:
    """TELE-001: Verify hub→backend transport for task_name field.

    These tests verify NissaHub correctly passes TrainingStartedPayload
    to backends. They do NOT verify:
    - train_ppo_vectorized() emits TRAINING_STARTED with correct task
    - SanctumAggregator extracts task into _task_name
    - SanctumSnapshot.task_name reflects the emitted value
    """

    def test_task_name_in_training_started_payload(self, capture_hub: CaptureHubResult):
        """TELE-001: task field is present and passed through hub."""
        hub, backend = capture_hub

        payload = _minimal_training_started(task="cifar10")
        event = TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data=payload,
        )
        hub.emit(event)
        hub.flush()

        events = backend.find_events(TelemetryEventType.TRAINING_STARTED)
        assert len(events) == 1, "Expected exactly one TRAINING_STARTED event"
        assert events[0].data.task == "cifar10"

    def test_task_name_empty_string_allowed(self, capture_hub: CaptureHubResult):
        """TELE-001: empty string task name passes through hub."""
        hub, backend = capture_hub

        payload = _minimal_training_started(task="")
        event = TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data=payload,
        )
        hub.emit(event)
        hub.flush()

        events = backend.find_events(TelemetryEventType.TRAINING_STARTED)
        assert len(events) == 1
        assert events[0].data.task == ""

    def test_task_name_long_string_preserved(self, capture_hub: CaptureHubResult):
        """TELE-001: long task names pass through hub unchanged."""
        hub, backend = capture_hub
        long_task_name = "custom_experiment_with_very_long_descriptive_name_2024"

        payload = _minimal_training_started(task=long_task_name)
        event = TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data=payload,
        )
        hub.emit(event)
        hub.flush()

        events = backend.find_events(TelemetryEventType.TRAINING_STARTED)
        assert len(events) == 1
        assert events[0].data.task == long_task_name


# =============================================================================
# TELE-001: Task Name - WIRING TESTS
# =============================================================================


class TestTELE001TaskNameWiring:
    """TELE-001: Verify FULL wiring from emitter to snapshot.

    A valid wiring test must verify:
    1. Event is processed by SanctumAggregator
    2. Aggregator stores task in _task_name
    3. Snapshot.task_name contains the emitted value (not default "")
    """

    def test_task_name_flows_to_snapshot(self):
        """TELE-001 WIRING: task flows from event → aggregator → snapshot."""
        from esper.karn.sanctum.aggregator import SanctumAggregator

        aggregator = SanctumAggregator(num_envs=4)

        payload = _minimal_training_started(task="wiring_test_task")
        event = TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data=payload,
        )
        aggregator.process_event(event)

        snapshot = aggregator.get_snapshot()
        assert snapshot.task_name == "wiring_test_task"

    def test_task_name_empty_string_flows_to_snapshot(self):
        """TELE-001 WIRING: empty task name flows correctly."""
        from esper.karn.sanctum.aggregator import SanctumAggregator

        aggregator = SanctumAggregator(num_envs=4)

        payload = _minimal_training_started(task="")
        event = TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data=payload,
        )
        aggregator.process_event(event)

        snapshot = aggregator.get_snapshot()
        assert snapshot.task_name == ""

    def test_task_name_default_before_training_started(self):
        """TELE-001 WIRING: task_name is empty string before TRAINING_STARTED."""
        from esper.karn.sanctum.aggregator import SanctumAggregator

        aggregator = SanctumAggregator(num_envs=4)
        snapshot = aggregator.get_snapshot()

        # Default should be empty string, not None
        assert snapshot.task_name == ""


# =============================================================================
# TELE-010: Current Episode - TRANSPORT TESTS
# =============================================================================


class TestTELE010CurrentEpisodeTransport:
    """TELE-010: Verify hub→backend transport for episodes_completed field."""

    def test_episodes_completed_in_batch_payload(self, capture_hub: CaptureHubResult):
        """TELE-010: episodes_completed passes through hub."""
        hub, backend = capture_hub

        payload = _minimal_batch_epoch_completed(episodes_completed=47)
        event = TelemetryEvent(
            event_type=TelemetryEventType.BATCH_EPOCH_COMPLETED,
            data=payload,
        )
        hub.emit(event)
        hub.flush()

        events = backend.find_events(TelemetryEventType.BATCH_EPOCH_COMPLETED)
        assert len(events) == 1, "Expected exactly one BATCH_EPOCH_COMPLETED event"
        assert events[0].data.episodes_completed == 47

    def test_episodes_completed_zero_at_start(self, capture_hub: CaptureHubResult):
        """TELE-010: episodes_completed=0 passes through hub."""
        hub, backend = capture_hub

        payload = _minimal_batch_epoch_completed(episodes_completed=0, batch_idx=0)
        event = TelemetryEvent(
            event_type=TelemetryEventType.BATCH_EPOCH_COMPLETED,
            data=payload,
        )
        hub.emit(event)
        hub.flush()

        events = backend.find_events(TelemetryEventType.BATCH_EPOCH_COMPLETED)
        assert len(events) == 1
        assert events[0].data.episodes_completed == 0

    def test_episodes_completed_sequence(self, capture_hub: CaptureHubResult):
        """TELE-010: multiple episode counts pass through hub in order."""
        hub, backend = capture_hub
        n_envs = 4

        for batch_idx in range(3):
            episodes = (batch_idx + 1) * n_envs
            payload = _minimal_batch_epoch_completed(
                episodes_completed=episodes,
                batch_idx=batch_idx,
                n_envs=n_envs,
            )
            event = TelemetryEvent(
                event_type=TelemetryEventType.BATCH_EPOCH_COMPLETED,
                data=payload,
            )
            hub.emit(event)

        hub.flush()

        events = backend.find_events(TelemetryEventType.BATCH_EPOCH_COMPLETED)
        assert len(events) == 3
        assert events[0].data.episodes_completed == 4
        assert events[1].data.episodes_completed == 8
        assert events[2].data.episodes_completed == 12


# =============================================================================
# TELE-010: Current Episode - WIRING TESTS
# =============================================================================


class TestTELE010CurrentEpisodeWiring:
    """TELE-010: Verify FULL wiring from emitter to snapshot.

    Wiring path:
    - BatchEpochCompletedPayload.episodes_completed
    - → aggregator._handle_batch_epoch_completed() sets _current_episode
    - → snapshot.current_episode
    """

    def test_current_episode_flows_to_snapshot(self):
        """TELE-010 WIRING: episodes_completed → aggregator → snapshot.current_episode."""
        from esper.karn.sanctum.aggregator import SanctumAggregator

        aggregator = SanctumAggregator(num_envs=4)

        payload = _minimal_batch_epoch_completed(episodes_completed=42)
        event = TelemetryEvent(
            event_type=TelemetryEventType.BATCH_EPOCH_COMPLETED,
            data=payload,
        )
        aggregator.process_event(event)

        snapshot = aggregator.get_snapshot()
        assert snapshot.current_episode == 42

    def test_current_episode_updates_on_each_batch(self):
        """TELE-010 WIRING: current_episode updates with each batch."""
        from esper.karn.sanctum.aggregator import SanctumAggregator

        aggregator = SanctumAggregator(num_envs=4)

        # Simulate 3 batches with increasing episode counts
        for episodes in [4, 8, 12]:
            payload = _minimal_batch_epoch_completed(episodes_completed=episodes)
            event = TelemetryEvent(
                event_type=TelemetryEventType.BATCH_EPOCH_COMPLETED,
                data=payload,
            )
            aggregator.process_event(event)

        snapshot = aggregator.get_snapshot()
        assert snapshot.current_episode == 12  # Last value wins

    def test_current_episode_default_is_zero(self):
        """TELE-010 WIRING: current_episode defaults to 0 before any batch."""
        from esper.karn.sanctum.aggregator import SanctumAggregator

        aggregator = SanctumAggregator(num_envs=4)
        snapshot = aggregator.get_snapshot()
        assert snapshot.current_episode == 0

    def test_current_episode_respects_start_episode(self):
        """TELE-010 WIRING: TRAINING_STARTED can set initial episode for resumed training."""
        from esper.karn.sanctum.aggregator import SanctumAggregator

        aggregator = SanctumAggregator(num_envs=4)

        # Simulate resumed training starting at episode 100
        start_payload = _minimal_training_started(start_episode=100)
        start_event = TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data=start_payload,
        )
        aggregator.process_event(start_event)

        snapshot = aggregator.get_snapshot()
        assert snapshot.current_episode == 100  # Set by start_episode


# =============================================================================
# TELE-011: Current Epoch - TRANSPORT TESTS
# =============================================================================


class TestTELE011CurrentEpochTransport:
    """TELE-011: Verify hub→backend transport for inner_epoch field."""

    def test_inner_epoch_in_epoch_completed_payload(self, capture_hub: CaptureHubResult):
        """TELE-011: inner_epoch passes through hub."""
        hub, backend = capture_hub

        payload = _minimal_epoch_completed(inner_epoch=47)
        event = TelemetryEvent(
            event_type=TelemetryEventType.EPOCH_COMPLETED,
            data=payload,
        )
        hub.emit(event)
        hub.flush()

        events = backend.find_events(TelemetryEventType.EPOCH_COMPLETED)
        assert len(events) == 1, "Expected exactly one EPOCH_COMPLETED event"
        assert events[0].data.inner_epoch == 47

    def test_inner_epoch_zero_at_start(self, capture_hub: CaptureHubResult):
        """TELE-011: inner_epoch=0 passes through hub."""
        hub, backend = capture_hub

        payload = _minimal_epoch_completed(inner_epoch=0)
        event = TelemetryEvent(
            event_type=TelemetryEventType.EPOCH_COMPLETED,
            data=payload,
        )
        hub.emit(event)
        hub.flush()

        events = backend.find_events(TelemetryEventType.EPOCH_COMPLETED)
        assert len(events) == 1
        assert events[0].data.inner_epoch == 0

    def test_inner_epoch_sequence(self, capture_hub: CaptureHubResult):
        """TELE-011: epoch sequence passes through hub in order."""
        hub, backend = capture_hub

        for epoch in range(5):
            payload = _minimal_epoch_completed(inner_epoch=epoch)
            event = TelemetryEvent(
                event_type=TelemetryEventType.EPOCH_COMPLETED,
                data=payload,
            )
            hub.emit(event)

        hub.flush()

        events = backend.find_events(TelemetryEventType.EPOCH_COMPLETED)
        assert len(events) == 5
        for i, event in enumerate(events):
            assert event.data.inner_epoch == i


# =============================================================================
# TELE-011: Current Epoch - WIRING TESTS
# =============================================================================


class TestTELE011CurrentEpochWiring:
    """TELE-011: Verify FULL wiring from emitter to snapshot.

    Wiring path:
    - EpochCompletedPayload.inner_epoch
    - → aggregator._handle_epoch_completed() sets _current_epoch
    - → snapshot.current_epoch
    """

    def test_current_epoch_flows_to_snapshot(self):
        """TELE-011 WIRING: inner_epoch → aggregator → snapshot.current_epoch."""
        from esper.karn.sanctum.aggregator import SanctumAggregator

        aggregator = SanctumAggregator(num_envs=4)

        payload = _minimal_epoch_completed(inner_epoch=99)
        event = TelemetryEvent(
            event_type=TelemetryEventType.EPOCH_COMPLETED,
            data=payload,
        )
        aggregator.process_event(event)

        snapshot = aggregator.get_snapshot()
        assert snapshot.current_epoch == 99

    def test_current_epoch_updates_on_each_epoch(self):
        """TELE-011 WIRING: current_epoch updates with each epoch completion."""
        from esper.karn.sanctum.aggregator import SanctumAggregator

        aggregator = SanctumAggregator(num_envs=4)

        # Simulate epochs 0 through 4
        for epoch in range(5):
            payload = _minimal_epoch_completed(inner_epoch=epoch)
            event = TelemetryEvent(
                event_type=TelemetryEventType.EPOCH_COMPLETED,
                data=payload,
            )
            aggregator.process_event(event)

        snapshot = aggregator.get_snapshot()
        assert snapshot.current_epoch == 4  # Last epoch value

    def test_current_epoch_default_is_zero(self):
        """TELE-011 WIRING: current_epoch defaults to 0 before any epoch."""
        from esper.karn.sanctum.aggregator import SanctumAggregator

        aggregator = SanctumAggregator(num_envs=4)
        snapshot = aggregator.get_snapshot()
        assert snapshot.current_epoch == 0

    def test_current_epoch_resets_on_training_started(self):
        """TELE-011 WIRING: TRAINING_STARTED resets current_epoch to 0."""
        from esper.karn.sanctum.aggregator import SanctumAggregator

        aggregator = SanctumAggregator(num_envs=4)

        # First, simulate some epochs
        for epoch in range(10):
            payload = _minimal_epoch_completed(inner_epoch=epoch)
            event = TelemetryEvent(
                event_type=TelemetryEventType.EPOCH_COMPLETED,
                data=payload,
            )
            aggregator.process_event(event)

        # Verify we're at epoch 9
        assert aggregator.get_snapshot().current_epoch == 9

        # Now start a new training run
        start_payload = _minimal_training_started()
        start_event = TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data=start_payload,
        )
        aggregator.process_event(start_event)

        # current_epoch should be reset to 0
        snapshot = aggregator.get_snapshot()
        assert snapshot.current_epoch == 0


# =============================================================================
# TELE-012: Max Epochs - TRANSPORT TESTS
# =============================================================================


class TestTELE012MaxEpochsTransport:
    """TELE-012: Verify hub→backend transport for max_epochs field."""

    def test_max_epochs_in_training_started_payload(self, capture_hub: CaptureHubResult):
        """TELE-012: max_epochs passes through hub."""
        hub, backend = capture_hub

        payload = _minimal_training_started(max_epochs=150)
        event = TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data=payload,
        )
        hub.emit(event)
        hub.flush()

        events = backend.find_events(TelemetryEventType.TRAINING_STARTED)
        assert len(events) == 1, "Expected exactly one TRAINING_STARTED event"
        assert events[0].data.max_epochs == 150

    def test_max_epochs_zero_for_unbounded_training(self, capture_hub: CaptureHubResult):
        """TELE-012: max_epochs=0 (unbounded) passes through hub."""
        hub, backend = capture_hub

        payload = _minimal_training_started(max_epochs=0, max_batches=1000)
        event = TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data=payload,
        )
        hub.emit(event)
        hub.flush()

        events = backend.find_events(TelemetryEventType.TRAINING_STARTED)
        assert len(events) == 1
        assert events[0].data.max_epochs == 0


# =============================================================================
# TELE-012: Max Epochs - WIRING TESTS
# =============================================================================


class TestTELE012MaxEpochsWiring:
    """TELE-012: Verify FULL wiring from emitter to snapshot.

    Wiring path:
    - TrainingStartedPayload.max_epochs
    - → aggregator._handle_training_started() sets _max_epochs
    - → snapshot.max_epochs
    """

    def test_max_epochs_flows_to_snapshot(self):
        """TELE-012 WIRING: max_epochs → aggregator → snapshot.max_epochs."""
        from esper.karn.sanctum.aggregator import SanctumAggregator

        aggregator = SanctumAggregator(num_envs=4)

        payload = _minimal_training_started(max_epochs=200)
        event = TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data=payload,
        )
        aggregator.process_event(event)

        snapshot = aggregator.get_snapshot()
        assert snapshot.max_epochs == 200

    def test_max_epochs_zero_for_unbounded(self):
        """TELE-012 WIRING: max_epochs=0 (unbounded training) flows to snapshot."""
        from esper.karn.sanctum.aggregator import SanctumAggregator

        aggregator = SanctumAggregator(num_envs=4)

        payload = _minimal_training_started(max_epochs=0)
        event = TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data=payload,
        )
        aggregator.process_event(event)

        snapshot = aggregator.get_snapshot()
        assert snapshot.max_epochs == 0

    def test_max_epochs_default_matches_leyline(self):
        """TELE-012 WIRING: max_epochs defaults to DEFAULT_EPISODE_LENGTH."""
        from esper.karn.sanctum.aggregator import SanctumAggregator
        from esper.leyline import DEFAULT_EPISODE_LENGTH

        aggregator = SanctumAggregator(num_envs=4)
        snapshot = aggregator.get_snapshot()
        # Default matches leyline.DEFAULT_EPISODE_LENGTH for consistent UI
        assert snapshot.max_epochs == DEFAULT_EPISODE_LENGTH


# =============================================================================
# TELE-013: Current Batch - TRANSPORT TESTS
# =============================================================================


class TestTELE013CurrentBatchTransport:
    """TELE-013: Verify hub→backend transport for batch_idx field."""

    def test_batch_idx_in_batch_payload(self, capture_hub: CaptureHubResult):
        """TELE-013: batch_idx passes through hub."""
        hub, backend = capture_hub

        payload = _minimal_batch_epoch_completed(batch_idx=25)
        event = TelemetryEvent(
            event_type=TelemetryEventType.BATCH_EPOCH_COMPLETED,
            data=payload,
        )
        hub.emit(event)
        hub.flush()

        events = backend.find_events(TelemetryEventType.BATCH_EPOCH_COMPLETED)
        assert len(events) == 1, "Expected exactly one BATCH_EPOCH_COMPLETED event"
        assert events[0].data.batch_idx == 25

    def test_batch_idx_zero_at_start(self, capture_hub: CaptureHubResult):
        """TELE-013: batch_idx=0 passes through hub."""
        hub, backend = capture_hub

        payload = _minimal_batch_epoch_completed(batch_idx=0)
        event = TelemetryEvent(
            event_type=TelemetryEventType.BATCH_EPOCH_COMPLETED,
            data=payload,
        )
        hub.emit(event)
        hub.flush()

        events = backend.find_events(TelemetryEventType.BATCH_EPOCH_COMPLETED)
        assert len(events) == 1
        assert events[0].data.batch_idx == 0

    def test_batch_idx_warmup_phase_values(self, capture_hub: CaptureHubResult):
        """TELE-013: batch_idx < 50 (warmup phase) passes through hub."""
        hub, backend = capture_hub

        for batch_idx in [5, 25, 49]:
            payload = _minimal_batch_epoch_completed(batch_idx=batch_idx)
            event = TelemetryEvent(
                event_type=TelemetryEventType.BATCH_EPOCH_COMPLETED,
                data=payload,
            )
            hub.emit(event)

        hub.flush()

        events = backend.find_events(TelemetryEventType.BATCH_EPOCH_COMPLETED)
        assert len(events) == 3
        for event in events:
            assert event.data.batch_idx < 50

    def test_batch_idx_post_warmup_values(self, capture_hub: CaptureHubResult):
        """TELE-013: batch_idx >= 50 (post-warmup) passes through hub."""
        hub, backend = capture_hub

        for batch_idx in [50, 100]:
            payload = _minimal_batch_epoch_completed(batch_idx=batch_idx)
            event = TelemetryEvent(
                event_type=TelemetryEventType.BATCH_EPOCH_COMPLETED,
                data=payload,
            )
            hub.emit(event)

        hub.flush()

        events = backend.find_events(TelemetryEventType.BATCH_EPOCH_COMPLETED)
        assert len(events) == 2
        assert events[0].data.batch_idx == 50
        assert events[1].data.batch_idx == 100


# =============================================================================
# TELE-013: Current Batch - WIRING TESTS
# =============================================================================


class TestTELE013CurrentBatchWiring:
    """TELE-013: Verify FULL wiring from emitter to snapshot.

    Wiring path:
    - BatchEpochCompletedPayload.batch_idx
    - → aggregator._handle_batch_epoch_completed() sets _current_batch
    - → snapshot.current_batch (with fallback to _batches_completed)
    """

    def test_current_batch_flows_to_snapshot(self):
        """TELE-013 WIRING: batch_idx → aggregator → snapshot.current_batch."""
        from esper.karn.sanctum.aggregator import SanctumAggregator

        aggregator = SanctumAggregator(num_envs=4)

        payload = _minimal_batch_epoch_completed(batch_idx=77)
        event = TelemetryEvent(
            event_type=TelemetryEventType.BATCH_EPOCH_COMPLETED,
            data=payload,
        )
        aggregator.process_event(event)

        snapshot = aggregator.get_snapshot()
        assert snapshot.current_batch == 77

    def test_current_batch_updates_on_each_batch(self):
        """TELE-013 WIRING: current_batch updates with each batch completion."""
        from esper.karn.sanctum.aggregator import SanctumAggregator

        aggregator = SanctumAggregator(num_envs=4)

        # Simulate batches 0, 25, 50, 100
        for batch_idx in [0, 25, 50, 100]:
            payload = _minimal_batch_epoch_completed(batch_idx=batch_idx)
            event = TelemetryEvent(
                event_type=TelemetryEventType.BATCH_EPOCH_COMPLETED,
                data=payload,
            )
            aggregator.process_event(event)

        snapshot = aggregator.get_snapshot()
        assert snapshot.current_batch == 100  # Last batch index

    def test_current_batch_default_is_zero(self):
        """TELE-013 WIRING: current_batch defaults to 0 before any batch."""
        from esper.karn.sanctum.aggregator import SanctumAggregator

        aggregator = SanctumAggregator(num_envs=4)
        snapshot = aggregator.get_snapshot()
        assert snapshot.current_batch == 0

    def test_current_batch_fallback_to_batches_completed(self):
        """TELE-013 WIRING: if batch_idx=0, fallback to batches_completed count."""
        from esper.karn.sanctum.aggregator import SanctumAggregator

        aggregator = SanctumAggregator(num_envs=4)

        # Process 3 batches all with batch_idx=0 (simulating old emitter behavior)
        for _ in range(3):
            payload = _minimal_batch_epoch_completed(batch_idx=0)
            event = TelemetryEvent(
                event_type=TelemetryEventType.BATCH_EPOCH_COMPLETED,
                data=payload,
            )
            aggregator.process_event(event)

        snapshot = aggregator.get_snapshot()
        # Fallback: current_batch = _current_batch or _batches_completed
        # _current_batch is 0, so falls back to _batches_completed = 3
        assert snapshot.current_batch == 3


# =============================================================================
# TELE-014: Max Batches - TRANSPORT TESTS
# =============================================================================


class TestTELE014MaxBatchesTransport:
    """TELE-014: Verify hub→backend transport for max_batches field."""

    def test_max_batches_in_training_started_payload(self, capture_hub: CaptureHubResult):
        """TELE-014: max_batches passes through hub."""
        hub, backend = capture_hub

        payload = _minimal_training_started(max_batches=100)
        event = TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data=payload,
        )
        hub.emit(event)
        hub.flush()

        events = backend.find_events(TelemetryEventType.TRAINING_STARTED)
        assert len(events) == 1, "Expected exactly one TRAINING_STARTED event"
        assert events[0].data.max_batches == 100

    def test_max_batches_includes_start_episode_offset(self, capture_hub: CaptureHubResult):
        """TELE-014: max_batches with resume offset passes through hub."""
        hub, backend = capture_hub

        # max_batches = n_episodes + start_episode = 50 + 25 = 75
        payload = _minimal_training_started(
            max_batches=75,
            n_episodes=50,
            start_episode=25,
        )
        event = TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data=payload,
        )
        hub.emit(event)
        hub.flush()

        events = backend.find_events(TelemetryEventType.TRAINING_STARTED)
        assert len(events) == 1
        assert events[0].data.max_batches == 75
        assert events[0].data.start_episode == 25


# =============================================================================
# TELE-014: Max Batches - WIRING TESTS
# =============================================================================


class TestTELE014MaxBatchesWiring:
    """TELE-014: Verify FULL wiring from emitter to snapshot.

    Wiring path:
    - TrainingStartedPayload.max_batches
    - → aggregator._handle_training_started() sets _max_batches
    - → snapshot.max_batches
    """

    def test_max_batches_flows_to_snapshot(self):
        """TELE-014 WIRING: max_batches → aggregator → snapshot.max_batches."""
        from esper.karn.sanctum.aggregator import SanctumAggregator

        aggregator = SanctumAggregator(num_envs=4)

        payload = _minimal_training_started(max_batches=500)
        event = TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data=payload,
        )
        aggregator.process_event(event)

        snapshot = aggregator.get_snapshot()
        assert snapshot.max_batches == 500

    def test_max_batches_with_resume_offset(self):
        """TELE-014 WIRING: max_batches includes resume offset correctly."""
        from esper.karn.sanctum.aggregator import SanctumAggregator

        aggregator = SanctumAggregator(num_envs=4)

        # Simulating resumed training: n_episodes=50, start_episode=25
        # max_batches = 75 (computed by training code before emitting)
        payload = _minimal_training_started(
            max_batches=75,
            n_episodes=50,
            start_episode=25,
        )
        event = TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data=payload,
        )
        aggregator.process_event(event)

        snapshot = aggregator.get_snapshot()
        assert snapshot.max_batches == 75

    def test_max_batches_default_is_100(self):
        """TELE-014 WIRING: max_batches defaults to 100 before TRAINING_STARTED."""
        from esper.karn.sanctum.aggregator import SanctumAggregator

        aggregator = SanctumAggregator(num_envs=4)
        snapshot = aggregator.get_snapshot()
        # Default before any training starts
        assert snapshot.max_batches == 100


# =============================================================================
# TELE-020: Runtime Seconds - TRANSPORT TESTS
# =============================================================================


class TestTELE020RuntimeSecondsTransport:
    """TELE-020: Verify TelemetryEvent has timestamp for runtime computation.

    Note: runtime_seconds is COMPUTED by SanctumAggregator from _start_time,
    not transmitted via payload. These tests verify the foundation (timestamps
    on events) not the computed metric itself.
    """

    def test_telemetry_event_has_timestamp(self, capture_hub: CaptureHubResult):
        """TELE-020: TelemetryEvent includes timestamp for runtime computation."""
        hub, backend = capture_hub

        payload = _minimal_training_started(task="timestamp_test")
        event = TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data=payload,
        )
        hub.emit(event)
        hub.flush()

        events = backend.find_events(TelemetryEventType.TRAINING_STARTED)
        assert len(events) == 1

        # Verify timestamp is present (foundation for runtime calculation)
        assert events[0].timestamp is not None
        # Verify timestamp is timezone-aware (UTC)
        assert events[0].timestamp.tzinfo is not None

    def test_events_have_monotonic_timestamps(self, capture_hub: CaptureHubResult):
        """TELE-020: event timestamps increase monotonically."""
        hub, backend = capture_hub

        for i in range(3):
            payload = _minimal_batch_epoch_completed(batch_idx=i)
            event = TelemetryEvent(
                event_type=TelemetryEventType.BATCH_EPOCH_COMPLETED,
                data=payload,
            )
            hub.emit(event)

        hub.flush()

        events = backend.find_events(TelemetryEventType.BATCH_EPOCH_COMPLETED)
        assert len(events) == 3

        for i in range(len(events) - 1):
            assert events[i].timestamp <= events[i + 1].timestamp


# =============================================================================
# TELE-020: Runtime Seconds - WIRING TESTS
# =============================================================================


class TestTELE020RuntimeSecondsWiring:
    """TELE-020: Verify FULL wiring for computed runtime_seconds.

    runtime_seconds is computed in SanctumAggregator._get_snapshot_unlocked():
        runtime = now - self._start_time if self._connected else 0.0

    Wiring path:
    - TRAINING_STARTED sets _connected=True and _start_time=time.time()
    - get_snapshot() computes runtime = now - _start_time
    - snapshot.runtime_seconds contains the computed value
    """

    def test_runtime_seconds_zero_before_training_started(self):
        """TELE-020 WIRING: runtime_seconds is 0 before TRAINING_STARTED."""
        from esper.karn.sanctum.aggregator import SanctumAggregator

        aggregator = SanctumAggregator(num_envs=4)
        snapshot = aggregator.get_snapshot()

        # Before TRAINING_STARTED, _connected is False, so runtime is 0
        assert snapshot.runtime_seconds == 0.0

    def test_runtime_seconds_positive_after_training_started(self):
        """TELE-020 WIRING: runtime_seconds > 0 after TRAINING_STARTED."""
        import time
        from esper.karn.sanctum.aggregator import SanctumAggregator

        aggregator = SanctumAggregator(num_envs=4)

        payload = _minimal_training_started()
        event = TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data=payload,
        )
        aggregator.process_event(event)

        # Small sleep to ensure measurable elapsed time
        time.sleep(0.01)

        snapshot = aggregator.get_snapshot()
        # After TRAINING_STARTED, runtime should be > 0
        assert snapshot.runtime_seconds > 0.0

    def test_runtime_seconds_increases_over_time(self):
        """TELE-020 WIRING: runtime_seconds increases with elapsed time."""
        import time
        from esper.karn.sanctum.aggregator import SanctumAggregator

        aggregator = SanctumAggregator(num_envs=4)

        payload = _minimal_training_started()
        event = TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data=payload,
        )
        aggregator.process_event(event)

        # Take two snapshots with a delay between them
        snapshot1 = aggregator.get_snapshot()
        time.sleep(0.02)
        snapshot2 = aggregator.get_snapshot()

        # Second snapshot should have larger runtime
        assert snapshot2.runtime_seconds > snapshot1.runtime_seconds


# =============================================================================
# TELE-021: Episode Return History - TRANSPORT TESTS
# =============================================================================


class TestTELE021EpisodeReturnHistoryTransport:
    """TELE-021: Verify hub→backend transport for avg_reward field."""

    def test_avg_reward_in_batch_payload(self, capture_hub: CaptureHubResult):
        """TELE-021: avg_reward passes through hub."""
        hub, backend = capture_hub

        payload = _minimal_batch_epoch_completed(avg_reward=0.75)
        event = TelemetryEvent(
            event_type=TelemetryEventType.BATCH_EPOCH_COMPLETED,
            data=payload,
        )
        hub.emit(event)
        hub.flush()

        events = backend.find_events(TelemetryEventType.BATCH_EPOCH_COMPLETED)
        assert len(events) == 1, "Expected exactly one BATCH_EPOCH_COMPLETED event"
        assert events[0].data.avg_reward == pytest.approx(0.75)

    def test_avg_reward_negative_allowed(self, capture_hub: CaptureHubResult):
        """TELE-021: negative avg_reward passes through hub."""
        hub, backend = capture_hub

        payload = _minimal_batch_epoch_completed(avg_reward=-1.5)
        event = TelemetryEvent(
            event_type=TelemetryEventType.BATCH_EPOCH_COMPLETED,
            data=payload,
        )
        hub.emit(event)
        hub.flush()

        events = backend.find_events(TelemetryEventType.BATCH_EPOCH_COMPLETED)
        assert len(events) == 1
        assert events[0].data.avg_reward == pytest.approx(-1.5)

    def test_avg_reward_sequence(self, capture_hub: CaptureHubResult):
        """TELE-021: multiple avg_reward values pass through hub in order."""
        hub, backend = capture_hub

        rewards = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
        for i, reward in enumerate(rewards):
            payload = _minimal_batch_epoch_completed(batch_idx=i, avg_reward=reward)
            event = TelemetryEvent(
                event_type=TelemetryEventType.BATCH_EPOCH_COMPLETED,
                data=payload,
            )
            hub.emit(event)

        hub.flush()

        events = backend.find_events(TelemetryEventType.BATCH_EPOCH_COMPLETED)
        assert len(events) == 8

        captured_rewards = [e.data.avg_reward for e in events]
        assert captured_rewards == pytest.approx(rewards)

    def test_avg_reward_precision_preserved(self, capture_hub: CaptureHubResult):
        """TELE-021: avg_reward float precision is preserved."""
        hub, backend = capture_hub

        precise_reward = 0.123456789
        payload = _minimal_batch_epoch_completed(avg_reward=precise_reward)
        event = TelemetryEvent(
            event_type=TelemetryEventType.BATCH_EPOCH_COMPLETED,
            data=payload,
        )
        hub.emit(event)
        hub.flush()

        events = backend.find_events(TelemetryEventType.BATCH_EPOCH_COMPLETED)
        assert len(events) == 1
        assert events[0].data.avg_reward == pytest.approx(precise_reward, rel=1e-9)


# =============================================================================
# TELE-021: Episode Return History - WIRING TESTS
# =============================================================================


class TestTELE021EpisodeReturnHistoryWiring:
    """TELE-021: Verify FULL wiring for episode_return_history.

    Wiring path:
    - BatchEpochCompletedPayload.avg_reward
    - → aggregator._handle_batch_epoch_completed() appends to _tamiyo.episode_return_history
    - → snapshot.tamiyo.episode_return_history (copied from deque)
    """

    def test_episode_return_history_accumulates_in_snapshot(self):
        """TELE-021 WIRING: avg_reward → aggregator → tamiyo.episode_return_history."""
        from esper.karn.sanctum.aggregator import SanctumAggregator

        aggregator = SanctumAggregator(num_envs=4)

        rewards = [0.1, 0.2, 0.3]
        for i, reward in enumerate(rewards):
            payload = _minimal_batch_epoch_completed(batch_idx=i, avg_reward=reward)
            event = TelemetryEvent(
                event_type=TelemetryEventType.BATCH_EPOCH_COMPLETED,
                data=payload,
            )
            aggregator.process_event(event)

        snapshot = aggregator.get_snapshot()
        history = list(snapshot.tamiyo.episode_return_history)
        assert history == pytest.approx(rewards)

    def test_episode_return_history_empty_before_batches(self):
        """TELE-021 WIRING: history is empty before any batch events."""
        from esper.karn.sanctum.aggregator import SanctumAggregator

        aggregator = SanctumAggregator(num_envs=4)
        snapshot = aggregator.get_snapshot()

        # Before any batches, history should be empty
        assert len(snapshot.tamiyo.episode_return_history) == 0

    def test_episode_return_history_preserves_order(self):
        """TELE-021 WIRING: history preserves insertion order."""
        from esper.karn.sanctum.aggregator import SanctumAggregator

        aggregator = SanctumAggregator(num_envs=4)

        # Emit rewards in specific order with varied values
        rewards = [-0.5, 0.0, 0.3, -0.2, 0.8, 0.1]
        for i, reward in enumerate(rewards):
            payload = _minimal_batch_epoch_completed(batch_idx=i, avg_reward=reward)
            event = TelemetryEvent(
                event_type=TelemetryEventType.BATCH_EPOCH_COMPLETED,
                data=payload,
            )
            aggregator.process_event(event)

        snapshot = aggregator.get_snapshot()
        history = list(snapshot.tamiyo.episode_return_history)
        # Order must be preserved exactly
        assert history == pytest.approx(rewards)

    def test_episode_return_history_handles_negative_rewards(self):
        """TELE-021 WIRING: history correctly stores negative rewards."""
        from esper.karn.sanctum.aggregator import SanctumAggregator

        aggregator = SanctumAggregator(num_envs=4)

        # All negative rewards (common early in training)
        rewards = [-2.0, -1.5, -1.2, -0.8]
        for i, reward in enumerate(rewards):
            payload = _minimal_batch_epoch_completed(batch_idx=i, avg_reward=reward)
            event = TelemetryEvent(
                event_type=TelemetryEventType.BATCH_EPOCH_COMPLETED,
                data=payload,
            )
            aggregator.process_event(event)

        snapshot = aggregator.get_snapshot()
        history = list(snapshot.tamiyo.episode_return_history)
        assert history == pytest.approx(rewards)


# =============================================================================
# Cross-Payload Tests (Transport Only)
# =============================================================================


class TestCrossPayloadConsistencyTransport:
    """Tests for consistency across related payload types (transport only)."""

    def test_training_started_and_batch_events_share_n_envs(
        self, capture_hub: CaptureHubResult
    ):
        """Training config from TRAINING_STARTED matches BATCH_EPOCH_COMPLETED."""
        hub, backend = capture_hub

        # Emit training started with n_envs=4
        hub.emit(
            TelemetryEvent(
                event_type=TelemetryEventType.TRAINING_STARTED,
                data=_minimal_training_started(n_envs=4),
            )
        )

        # Emit batch completed with matching n_envs=4
        hub.emit(
            TelemetryEvent(
                event_type=TelemetryEventType.BATCH_EPOCH_COMPLETED,
                data=_minimal_batch_epoch_completed(n_envs=4),
            )
        )

        hub.flush()

        start_events = backend.find_events(TelemetryEventType.TRAINING_STARTED)
        batch_events = backend.find_events(TelemetryEventType.BATCH_EPOCH_COMPLETED)

        assert len(start_events) == 1
        assert len(batch_events) == 1
        assert start_events[0].data.n_envs == batch_events[0].data.n_envs

    def test_multiple_event_types_captured_separately(
        self, capture_hub: CaptureHubResult
    ):
        """Different event types are properly segregated in capture backend."""
        hub, backend = capture_hub

        hub.emit(
            TelemetryEvent(
                event_type=TelemetryEventType.TRAINING_STARTED,
                data=_minimal_training_started(),
            )
        )
        hub.emit(
            TelemetryEvent(
                event_type=TelemetryEventType.EPOCH_COMPLETED,
                data=_minimal_epoch_completed(),
            )
        )
        hub.emit(
            TelemetryEvent(
                event_type=TelemetryEventType.BATCH_EPOCH_COMPLETED,
                data=_minimal_batch_epoch_completed(),
            )
        )

        hub.flush()

        assert len(backend.find_events(TelemetryEventType.TRAINING_STARTED)) == 1
        assert len(backend.find_events(TelemetryEventType.EPOCH_COMPLETED)) == 1
        assert len(backend.find_events(TelemetryEventType.BATCH_EPOCH_COMPLETED)) == 1
        assert len(backend.events) == 3
