"""End-to-end tests for training metrics (TELE-001 to TELE-099).

Verifies training telemetry flows from source to nissa:
- TELE-001: task_name (TrainingStartedPayload.task)
- TELE-010: current_episode (BatchEpochCompletedPayload.episodes_completed)
- TELE-011: current_epoch (EpochCompletedPayload.inner_epoch)
- TELE-012: max_epochs (TrainingStartedPayload.max_epochs)
- TELE-013: current_batch (BatchEpochCompletedPayload.batch_idx)
- TELE-014: max_batches (TrainingStartedPayload.max_batches)
- TELE-020: runtime_seconds (computed from session start time)
- TELE-021: episode_return_history (BatchEpochCompletedPayload.avg_reward)

These tests verify the telemetry contract: when training events are emitted,
the correct payload fields reach the capture backend.

Note: NissaHub processes events asynchronously via background workers.
Tests must call hub.flush() after emitting events to ensure they are processed
before assertions.
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
# TELE-001: Task Name
# =============================================================================


class TestTELE001TaskName:
    """TELE-001: task_name is emitted in TRAINING_STARTED event."""

    def test_task_name_in_training_started_payload(self, capture_hub: CaptureHubResult):
        """TELE-001: task field is present and correct in TrainingStartedPayload."""
        hub, backend = capture_hub

        # Create and emit TRAINING_STARTED with known task name
        payload = TrainingStartedPayload(
            n_envs=4,
            max_epochs=150,
            max_batches=100,
            task="cifar10",  # The metric we're testing
            host_params=1_000_000,
            slot_ids=("r0c0", "r0c1"),
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

        event = TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data=payload,
        )
        hub.emit(event)
        hub.flush()  # Wait for async processing

        # Verify event captured
        events = backend.find_events(TelemetryEventType.TRAINING_STARTED)
        assert len(events) == 1, "Expected exactly one TRAINING_STARTED event"
        assert events[0].data.task == "cifar10"

    def test_task_name_empty_string_allowed(self, capture_hub: CaptureHubResult):
        """TELE-001: empty string task name is valid (pre-training default)."""
        hub, backend = capture_hub

        payload = TrainingStartedPayload(
            n_envs=4,
            max_epochs=150,
            max_batches=100,
            task="",  # Empty task name
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

        event = TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data=payload,
        )
        hub.emit(event)
        hub.flush()  # Wait for async processing

        events = backend.find_events(TelemetryEventType.TRAINING_STARTED)
        assert len(events) == 1
        assert events[0].data.task == ""

    def test_task_name_long_string_preserved(self, capture_hub: CaptureHubResult):
        """TELE-001: long task names are preserved in payload (UI may truncate)."""
        hub, backend = capture_hub
        long_task_name = "custom_experiment_with_very_long_descriptive_name_2024"

        payload = TrainingStartedPayload(
            n_envs=4,
            max_epochs=150,
            max_batches=100,
            task=long_task_name,
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

        event = TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data=payload,
        )
        hub.emit(event)
        hub.flush()  # Wait for async processing

        events = backend.find_events(TelemetryEventType.TRAINING_STARTED)
        assert len(events) == 1
        assert events[0].data.task == long_task_name


# =============================================================================
# TELE-010: Current Episode
# =============================================================================


class TestTELE010CurrentEpisode:
    """TELE-010: current_episode (episodes_completed) in BATCH_EPOCH_COMPLETED."""

    def test_episodes_completed_in_batch_payload(self, capture_hub: CaptureHubResult):
        """TELE-010: episodes_completed field is present in BatchEpochCompletedPayload."""
        hub, backend = capture_hub

        payload = BatchEpochCompletedPayload(
            episodes_completed=47,  # The metric we're testing
            batch_idx=10,
            avg_accuracy=85.5,
            avg_reward=0.75,
            total_episodes=100,
            n_envs=4,
        )

        event = TelemetryEvent(
            event_type=TelemetryEventType.BATCH_EPOCH_COMPLETED,
            data=payload,
        )
        hub.emit(event)
        hub.flush()  # Wait for async processing

        events = backend.find_events(TelemetryEventType.BATCH_EPOCH_COMPLETED)
        assert len(events) == 1, "Expected exactly one BATCH_EPOCH_COMPLETED event"
        assert events[0].data.episodes_completed == 47

    def test_episodes_completed_zero_at_start(self, capture_hub: CaptureHubResult):
        """TELE-010: episodes_completed can be 0 before first batch completes."""
        hub, backend = capture_hub

        payload = BatchEpochCompletedPayload(
            episodes_completed=0,  # First batch, zero episodes
            batch_idx=0,
            avg_accuracy=50.0,
            avg_reward=0.0,
            total_episodes=100,
            n_envs=4,
        )

        event = TelemetryEvent(
            event_type=TelemetryEventType.BATCH_EPOCH_COMPLETED,
            data=payload,
        )
        hub.emit(event)
        hub.flush()  # Wait for async processing

        events = backend.find_events(TelemetryEventType.BATCH_EPOCH_COMPLETED)
        assert len(events) == 1
        assert events[0].data.episodes_completed == 0

    def test_episodes_completed_increments_by_n_envs(self, capture_hub: CaptureHubResult):
        """TELE-010: episodes_completed increments by n_envs per batch (vectorized)."""
        hub, backend = capture_hub
        n_envs = 4

        # Simulate batch sequence: each batch processes n_envs episodes
        for batch_idx in range(3):
            episodes = (batch_idx + 1) * n_envs  # 4, 8, 12
            payload = BatchEpochCompletedPayload(
                episodes_completed=episodes,
                batch_idx=batch_idx,
                avg_accuracy=50.0 + batch_idx * 5,
                avg_reward=0.1 * batch_idx,
                total_episodes=100,
                n_envs=n_envs,
            )
            event = TelemetryEvent(
                event_type=TelemetryEventType.BATCH_EPOCH_COMPLETED,
                data=payload,
            )
            hub.emit(event)

        hub.flush()  # Wait for async processing

        events = backend.find_events(TelemetryEventType.BATCH_EPOCH_COMPLETED)
        assert len(events) == 3
        assert events[0].data.episodes_completed == 4
        assert events[1].data.episodes_completed == 8
        assert events[2].data.episodes_completed == 12


# =============================================================================
# TELE-011: Current Epoch
# =============================================================================


class TestTELE011CurrentEpoch:
    """TELE-011: current_epoch (inner_epoch) in EPOCH_COMPLETED event."""

    def test_inner_epoch_in_epoch_completed_payload(self, capture_hub: CaptureHubResult):
        """TELE-011: inner_epoch field is present in EpochCompletedPayload."""
        hub, backend = capture_hub

        payload = EpochCompletedPayload(
            env_id=0,
            val_accuracy=85.5,
            val_loss=0.45,
            inner_epoch=47,  # The metric we're testing
        )

        event = TelemetryEvent(
            event_type=TelemetryEventType.EPOCH_COMPLETED,
            data=payload,
        )
        hub.emit(event)
        hub.flush()  # Wait for async processing

        events = backend.find_events(TelemetryEventType.EPOCH_COMPLETED)
        assert len(events) == 1, "Expected exactly one EPOCH_COMPLETED event"
        assert events[0].data.inner_epoch == 47

    def test_inner_epoch_zero_at_start(self, capture_hub: CaptureHubResult):
        """TELE-011: inner_epoch starts at 0."""
        hub, backend = capture_hub

        payload = EpochCompletedPayload(
            env_id=0,
            val_accuracy=25.0,
            val_loss=2.5,
            inner_epoch=0,
        )

        event = TelemetryEvent(
            event_type=TelemetryEventType.EPOCH_COMPLETED,
            data=payload,
        )
        hub.emit(event)
        hub.flush()  # Wait for async processing

        events = backend.find_events(TelemetryEventType.EPOCH_COMPLETED)
        assert len(events) == 1
        assert events[0].data.inner_epoch == 0

    def test_inner_epoch_increments_with_validation(self, capture_hub: CaptureHubResult):
        """TELE-011: inner_epoch increments after each validation cycle."""
        hub, backend = capture_hub

        for epoch in range(5):
            payload = EpochCompletedPayload(
                env_id=0,
                val_accuracy=50.0 + epoch * 10,
                val_loss=2.0 - epoch * 0.3,
                inner_epoch=epoch,
            )
            event = TelemetryEvent(
                event_type=TelemetryEventType.EPOCH_COMPLETED,
                data=payload,
            )
            hub.emit(event)

        hub.flush()  # Wait for async processing

        events = backend.find_events(TelemetryEventType.EPOCH_COMPLETED)
        assert len(events) == 5
        for i, event in enumerate(events):
            assert event.data.inner_epoch == i


# =============================================================================
# TELE-012: Max Epochs
# =============================================================================


class TestTELE012MaxEpochs:
    """TELE-012: max_epochs in TRAINING_STARTED event."""

    def test_max_epochs_in_training_started_payload(self, capture_hub: CaptureHubResult):
        """TELE-012: max_epochs field is present in TrainingStartedPayload."""
        hub, backend = capture_hub

        payload = TrainingStartedPayload(
            n_envs=4,
            max_epochs=150,  # The metric we're testing
            max_batches=100,
            task="cifar10",
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

        event = TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data=payload,
        )
        hub.emit(event)
        hub.flush()  # Wait for async processing

        events = backend.find_events(TelemetryEventType.TRAINING_STARTED)
        assert len(events) == 1, "Expected exactly one TRAINING_STARTED event"
        assert events[0].data.max_epochs == 150

    def test_max_epochs_zero_for_unbounded_training(self, capture_hub: CaptureHubResult):
        """TELE-012: max_epochs=0 indicates unbounded (batch-limited) training."""
        hub, backend = capture_hub

        payload = TrainingStartedPayload(
            n_envs=4,
            max_epochs=0,  # Unbounded
            max_batches=1000,
            task="unbounded_run",
            host_params=1_000_000,
            slot_ids=("r0c0",),
            seed=42,
            n_episodes=1000,
            lr=3e-4,
            clip_ratio=0.2,
            entropy_coef=0.01,
            param_budget=10_000_000,
            policy_device="cpu",
            env_devices=("cpu",),
            reward_mode="shaped",
        )

        event = TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data=payload,
        )
        hub.emit(event)
        hub.flush()  # Wait for async processing

        events = backend.find_events(TelemetryEventType.TRAINING_STARTED)
        assert len(events) == 1
        assert events[0].data.max_epochs == 0


# =============================================================================
# TELE-013: Current Batch
# =============================================================================


class TestTELE013CurrentBatch:
    """TELE-013: current_batch (batch_idx) in BATCH_EPOCH_COMPLETED event."""

    def test_batch_idx_in_batch_payload(self, capture_hub: CaptureHubResult):
        """TELE-013: batch_idx field is present in BatchEpochCompletedPayload."""
        hub, backend = capture_hub

        payload = BatchEpochCompletedPayload(
            episodes_completed=100,
            batch_idx=25,  # The metric we're testing
            avg_accuracy=85.5,
            avg_reward=0.75,
            total_episodes=400,
            n_envs=4,
        )

        event = TelemetryEvent(
            event_type=TelemetryEventType.BATCH_EPOCH_COMPLETED,
            data=payload,
        )
        hub.emit(event)
        hub.flush()  # Wait for async processing

        events = backend.find_events(TelemetryEventType.BATCH_EPOCH_COMPLETED)
        assert len(events) == 1, "Expected exactly one BATCH_EPOCH_COMPLETED event"
        assert events[0].data.batch_idx == 25

    def test_batch_idx_zero_at_start(self, capture_hub: CaptureHubResult):
        """TELE-013: batch_idx starts at 0."""
        hub, backend = capture_hub

        payload = BatchEpochCompletedPayload(
            episodes_completed=4,
            batch_idx=0,  # First batch
            avg_accuracy=50.0,
            avg_reward=0.0,
            total_episodes=100,
            n_envs=4,
        )

        event = TelemetryEvent(
            event_type=TelemetryEventType.BATCH_EPOCH_COMPLETED,
            data=payload,
        )
        hub.emit(event)
        hub.flush()  # Wait for async processing

        events = backend.find_events(TelemetryEventType.BATCH_EPOCH_COMPLETED)
        assert len(events) == 1
        assert events[0].data.batch_idx == 0

    def test_batch_idx_warmup_phase_detection(self, capture_hub: CaptureHubResult):
        """TELE-013: batch_idx < 50 indicates warmup phase for UI display."""
        hub, backend = capture_hub

        # Emit batches in warmup phase (< 50)
        for batch_idx in [5, 25, 49]:
            payload = BatchEpochCompletedPayload(
                episodes_completed=batch_idx * 4,
                batch_idx=batch_idx,
                avg_accuracy=50.0,
                avg_reward=0.1,
                total_episodes=500,
                n_envs=4,
            )
            event = TelemetryEvent(
                event_type=TelemetryEventType.BATCH_EPOCH_COMPLETED,
                data=payload,
            )
            hub.emit(event)

        hub.flush()  # Wait for async processing

        events = backend.find_events(TelemetryEventType.BATCH_EPOCH_COMPLETED)
        assert len(events) == 3

        # All in warmup phase (< 50)
        for event in events:
            assert event.data.batch_idx < 50

    def test_batch_idx_post_warmup(self, capture_hub: CaptureHubResult):
        """TELE-013: batch_idx >= 50 indicates post-warmup training phase."""
        hub, backend = capture_hub

        # Emit batch at warmup boundary and after
        for batch_idx in [50, 100]:
            payload = BatchEpochCompletedPayload(
                episodes_completed=batch_idx * 4,
                batch_idx=batch_idx,
                avg_accuracy=75.0,
                avg_reward=0.5,
                total_episodes=500,
                n_envs=4,
            )
            event = TelemetryEvent(
                event_type=TelemetryEventType.BATCH_EPOCH_COMPLETED,
                data=payload,
            )
            hub.emit(event)

        hub.flush()  # Wait for async processing

        events = backend.find_events(TelemetryEventType.BATCH_EPOCH_COMPLETED)
        assert len(events) == 2
        assert events[0].data.batch_idx == 50
        assert events[1].data.batch_idx == 100


# =============================================================================
# TELE-014: Max Batches
# =============================================================================


class TestTELE014MaxBatches:
    """TELE-014: max_batches in TRAINING_STARTED event."""

    def test_max_batches_in_training_started_payload(self, capture_hub: CaptureHubResult):
        """TELE-014: max_batches field is present in TrainingStartedPayload."""
        hub, backend = capture_hub

        payload = TrainingStartedPayload(
            n_envs=4,
            max_epochs=150,
            max_batches=100,  # The metric we're testing
            task="cifar10",
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

        event = TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data=payload,
        )
        hub.emit(event)
        hub.flush()  # Wait for async processing

        events = backend.find_events(TelemetryEventType.TRAINING_STARTED)
        assert len(events) == 1, "Expected exactly one TRAINING_STARTED event"
        assert events[0].data.max_batches == 100

    def test_max_batches_includes_start_episode_offset(self, capture_hub: CaptureHubResult):
        """TELE-014: max_batches = n_episodes + start_episode for resume support."""
        hub, backend = capture_hub

        # Simulate resumed run: start_episode=25, n_episodes=50 -> max_batches=75
        # Note: The training loop computes this, we just verify the payload carries it
        payload = TrainingStartedPayload(
            n_envs=4,
            max_epochs=150,
            max_batches=75,  # 25 + 50 for resumed run
            task="resumed_run",
            host_params=1_000_000,
            slot_ids=("r0c0",),
            seed=42,
            n_episodes=50,
            lr=3e-4,
            clip_ratio=0.2,
            entropy_coef=0.01,
            param_budget=10_000_000,
            policy_device="cpu",
            env_devices=("cpu",),
            reward_mode="shaped",
            start_episode=25,  # Resume offset
        )

        event = TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data=payload,
        )
        hub.emit(event)
        hub.flush()  # Wait for async processing

        events = backend.find_events(TelemetryEventType.TRAINING_STARTED)
        assert len(events) == 1
        assert events[0].data.max_batches == 75
        assert events[0].data.start_episode == 25


# =============================================================================
# TELE-020: Runtime Seconds
# =============================================================================


class TestTELE020RuntimeSeconds:
    """TELE-020: runtime_seconds (elapsed time since training start).

    Note: runtime_seconds is computed by SanctumAggregator from _start_time,
    not emitted via telemetry payload. These tests verify the time field exists
    on TelemetryEvent timestamp which is the foundation for runtime calculation.
    """

    def test_telemetry_event_has_timestamp(self, capture_hub: CaptureHubResult):
        """TELE-020: TelemetryEvent includes timestamp for runtime computation."""
        hub, backend = capture_hub

        payload = TrainingStartedPayload(
            n_envs=4,
            max_epochs=150,
            max_batches=100,
            task="timestamp_test",
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

        event = TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data=payload,
        )
        hub.emit(event)
        hub.flush()  # Wait for async processing

        events = backend.find_events(TelemetryEventType.TRAINING_STARTED)
        assert len(events) == 1

        # Verify timestamp is present (foundation for runtime calculation)
        assert events[0].timestamp is not None
        # Verify timestamp is timezone-aware (UTC)
        assert events[0].timestamp.tzinfo is not None

    def test_events_have_monotonic_timestamps(self, capture_hub: CaptureHubResult):
        """TELE-020: event timestamps increase monotonically over time."""
        hub, backend = capture_hub

        # Emit multiple events
        for i in range(3):
            payload = BatchEpochCompletedPayload(
                episodes_completed=i * 4,
                batch_idx=i,
                avg_accuracy=50.0 + i * 5,
                avg_reward=0.1 * i,
                total_episodes=100,
                n_envs=4,
            )
            event = TelemetryEvent(
                event_type=TelemetryEventType.BATCH_EPOCH_COMPLETED,
                data=payload,
            )
            hub.emit(event)

        hub.flush()  # Wait for async processing

        events = backend.find_events(TelemetryEventType.BATCH_EPOCH_COMPLETED)
        assert len(events) == 3

        # Timestamps should be monotonically non-decreasing
        for i in range(len(events) - 1):
            assert events[i].timestamp <= events[i + 1].timestamp


# =============================================================================
# TELE-021: Episode Return History
# =============================================================================


class TestTELE021EpisodeReturnHistory:
    """TELE-021: episode_return_history (from BatchEpochCompletedPayload.avg_reward)."""

    def test_avg_reward_in_batch_payload(self, capture_hub: CaptureHubResult):
        """TELE-021: avg_reward field is present in BatchEpochCompletedPayload."""
        hub, backend = capture_hub

        payload = BatchEpochCompletedPayload(
            episodes_completed=40,
            batch_idx=10,
            avg_accuracy=85.5,
            avg_reward=0.75,  # The metric we're testing
            total_episodes=100,
            n_envs=4,
        )

        event = TelemetryEvent(
            event_type=TelemetryEventType.BATCH_EPOCH_COMPLETED,
            data=payload,
        )
        hub.emit(event)
        hub.flush()  # Wait for async processing

        events = backend.find_events(TelemetryEventType.BATCH_EPOCH_COMPLETED)
        assert len(events) == 1, "Expected exactly one BATCH_EPOCH_COMPLETED event"
        assert events[0].data.avg_reward == pytest.approx(0.75)

    def test_avg_reward_negative_allowed(self, capture_hub: CaptureHubResult):
        """TELE-021: avg_reward can be negative (early training, random policy)."""
        hub, backend = capture_hub

        payload = BatchEpochCompletedPayload(
            episodes_completed=4,
            batch_idx=0,
            avg_accuracy=25.0,
            avg_reward=-1.5,  # Negative reward (random policy baseline)
            total_episodes=100,
            n_envs=4,
        )

        event = TelemetryEvent(
            event_type=TelemetryEventType.BATCH_EPOCH_COMPLETED,
            data=payload,
        )
        hub.emit(event)
        hub.flush()  # Wait for async processing

        events = backend.find_events(TelemetryEventType.BATCH_EPOCH_COMPLETED)
        assert len(events) == 1
        assert events[0].data.avg_reward == pytest.approx(-1.5)

    def test_avg_reward_history_accumulates(self, capture_hub: CaptureHubResult):
        """TELE-021: multiple avg_reward values can be collected for history."""
        hub, backend = capture_hub

        # Simulate improving returns over training
        rewards = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
        for i, reward in enumerate(rewards):
            payload = BatchEpochCompletedPayload(
                episodes_completed=(i + 1) * 4,
                batch_idx=i,
                avg_accuracy=25.0 + i * 10,
                avg_reward=reward,
                total_episodes=100,
                n_envs=4,
            )
            event = TelemetryEvent(
                event_type=TelemetryEventType.BATCH_EPOCH_COMPLETED,
                data=payload,
            )
            hub.emit(event)

        hub.flush()  # Wait for async processing

        events = backend.find_events(TelemetryEventType.BATCH_EPOCH_COMPLETED)
        assert len(events) == 8

        # Verify all rewards are captured
        captured_rewards = [e.data.avg_reward for e in events]
        assert captured_rewards == pytest.approx(rewards)

    def test_avg_reward_precision_preserved(self, capture_hub: CaptureHubResult):
        """TELE-021: avg_reward maintains float precision."""
        hub, backend = capture_hub

        # Use precise float value
        precise_reward = 0.123456789

        payload = BatchEpochCompletedPayload(
            episodes_completed=4,
            batch_idx=0,
            avg_accuracy=50.0,
            avg_reward=precise_reward,
            total_episodes=100,
            n_envs=4,
        )

        event = TelemetryEvent(
            event_type=TelemetryEventType.BATCH_EPOCH_COMPLETED,
            data=payload,
        )
        hub.emit(event)
        hub.flush()  # Wait for async processing

        events = backend.find_events(TelemetryEventType.BATCH_EPOCH_COMPLETED)
        assert len(events) == 1
        # Float equality with tolerance
        assert events[0].data.avg_reward == pytest.approx(precise_reward, rel=1e-9)


# =============================================================================
# Cross-Payload Tests
# =============================================================================


class TestCrossPayloadConsistency:
    """Tests for consistency across related payload types."""

    def test_training_started_sets_context_for_batch_events(
        self, capture_hub: CaptureHubResult
    ):
        """Training config from TRAINING_STARTED should match BATCH_EPOCH_COMPLETED."""
        hub, backend = capture_hub

        # Emit training started
        start_payload = TrainingStartedPayload(
            n_envs=4,
            max_epochs=150,
            max_batches=100,
            task="consistency_test",
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
        hub.emit(
            TelemetryEvent(
                event_type=TelemetryEventType.TRAINING_STARTED,
                data=start_payload,
            )
        )

        # Emit batch completed
        batch_payload = BatchEpochCompletedPayload(
            episodes_completed=40,
            batch_idx=10,
            avg_accuracy=75.0,
            avg_reward=0.5,
            total_episodes=100,  # Should match max_batches
            n_envs=4,  # Should match n_envs
        )
        hub.emit(
            TelemetryEvent(
                event_type=TelemetryEventType.BATCH_EPOCH_COMPLETED,
                data=batch_payload,
            )
        )

        hub.flush()  # Wait for async processing

        # Verify consistency
        start_events = backend.find_events(TelemetryEventType.TRAINING_STARTED)
        batch_events = backend.find_events(TelemetryEventType.BATCH_EPOCH_COMPLETED)

        assert len(start_events) == 1
        assert len(batch_events) == 1

        # n_envs should match
        assert start_events[0].data.n_envs == batch_events[0].data.n_envs

    def test_multiple_event_types_captured_separately(
        self, capture_hub: CaptureHubResult
    ):
        """Different event types are properly segregated in capture backend."""
        hub, backend = capture_hub

        # Emit TRAINING_STARTED
        hub.emit(
            TelemetryEvent(
                event_type=TelemetryEventType.TRAINING_STARTED,
                data=TrainingStartedPayload(
                    n_envs=4,
                    max_epochs=150,
                    max_batches=100,
                    task="multi_test",
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
                ),
            )
        )

        # Emit EPOCH_COMPLETED
        hub.emit(
            TelemetryEvent(
                event_type=TelemetryEventType.EPOCH_COMPLETED,
                data=EpochCompletedPayload(
                    env_id=0,
                    val_accuracy=85.5,
                    val_loss=0.45,
                    inner_epoch=47,
                ),
            )
        )

        # Emit BATCH_EPOCH_COMPLETED
        hub.emit(
            TelemetryEvent(
                event_type=TelemetryEventType.BATCH_EPOCH_COMPLETED,
                data=BatchEpochCompletedPayload(
                    episodes_completed=40,
                    batch_idx=10,
                    avg_accuracy=85.5,
                    avg_reward=0.75,
                    total_episodes=100,
                    n_envs=4,
                ),
            )
        )

        hub.flush()  # Wait for async processing

        # Verify each type is captured separately
        assert len(backend.find_events(TelemetryEventType.TRAINING_STARTED)) == 1
        assert len(backend.find_events(TelemetryEventType.EPOCH_COMPLETED)) == 1
        assert len(backend.find_events(TelemetryEventType.BATCH_EPOCH_COMPLETED)) == 1

        # Total events
        assert len(backend.events) == 3
