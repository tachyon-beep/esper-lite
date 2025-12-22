"""Integration tests for telemetry event emission.

Tests new telemetry event types and console output formatting.
"""

import pytest
from esper.leyline import TelemetryEvent, TelemetryEventType
from esper.nissa.output import ConsoleOutput, NissaHub


class MockBackend:
    """Capture events for testing."""

    def __init__(self) -> None:
        self.events: list[TelemetryEvent] = []

    def start(self) -> None:
        """Start the backend (no-op for mock)."""
        pass

    def emit(self, event: TelemetryEvent) -> None:
        self.events.append(event)

    def close(self) -> None:
        pass


class TestGovernorEventTypes:
    """Verify Governor event types are defined."""

    def test_governor_rollback_exists(self) -> None:
        """GOVERNOR_ROLLBACK should be a valid event type."""
        event_type = TelemetryEventType.GOVERNOR_ROLLBACK
        assert event_type.name == "GOVERNOR_ROLLBACK"

    def test_governor_panic_exists(self) -> None:
        """GOVERNOR_PANIC should be a valid event type."""
        event_type = TelemetryEventType.GOVERNOR_PANIC
        assert event_type.name == "GOVERNOR_PANIC"

    def test_governor_snapshot_exists(self) -> None:
        """GOVERNOR_SNAPSHOT should be a valid event type."""
        event_type = TelemetryEventType.GOVERNOR_SNAPSHOT
        assert event_type.name == "GOVERNOR_SNAPSHOT"


class TestBatchEventTypes:
    """Verify batch progress event types are defined."""

    def test_batch_completed_exists(self) -> None:
        """BATCH_EPOCH_COMPLETED should be a valid event type."""
        event_type = TelemetryEventType.BATCH_EPOCH_COMPLETED
        assert event_type.name == "BATCH_EPOCH_COMPLETED"

    def test_checkpoint_saved_exists(self) -> None:
        """CHECKPOINT_SAVED should be a valid event type."""
        event_type = TelemetryEventType.CHECKPOINT_SAVED
        assert event_type.name == "CHECKPOINT_SAVED"

    def test_checkpoint_loaded_exists(self) -> None:
        """CHECKPOINT_LOADED should be a valid event type."""
        event_type = TelemetryEventType.CHECKPOINT_LOADED
        assert event_type.name == "CHECKPOINT_LOADED"

    def test_training_started_exists(self) -> None:
        """TRAINING_STARTED should be a valid event type."""
        event_type = TelemetryEventType.TRAINING_STARTED
        assert event_type.name == "TRAINING_STARTED"


class TestCounterfactualEventType:
    """Verify counterfactual event type is defined."""

    def test_counterfactual_computed_exists(self) -> None:
        """COUNTERFACTUAL_COMPUTED should be a valid event type."""
        event_type = TelemetryEventType.COUNTERFACTUAL_COMPUTED
        assert event_type.name == "COUNTERFACTUAL_COMPUTED"


class TestSeedGateEventType:
    """Verify seed gate event types are defined."""

    def test_seed_gate_evaluated_exists(self) -> None:
        """SEED_GATE_EVALUATED should be a valid event type."""
        event_type = TelemetryEventType.SEED_GATE_EVALUATED
        assert event_type.name == "SEED_GATE_EVALUATED"


class TestConsoleOutputFormatters:
    """Test ConsoleOutput formats new event types correctly."""

    def test_formats_governor_rollback(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Verify ConsoleOutput formats GOVERNOR_ROLLBACK correctly."""
        console = ConsoleOutput()
        event = TelemetryEvent(
            event_type=TelemetryEventType.GOVERNOR_ROLLBACK,
            data={
                "reason": "Structural Collapse",
                "loss_at_panic": 15.3,
                "loss_threshold": 5.2,
                "consecutive_panics": 2,
            },
        )
        console.emit(event)
        captured = capsys.readouterr()
        assert "GOVERNOR" in captured.out
        assert "ROLLBACK" in captured.out
        assert "Structural Collapse" in captured.out

    def test_formats_batch_completed(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Verify ConsoleOutput formats BATCH_EPOCH_COMPLETED correctly."""
        console = ConsoleOutput()
        event = TelemetryEvent(
            event_type=TelemetryEventType.BATCH_EPOCH_COMPLETED,
            data={
                "batch_idx": 3,
                "episodes_completed": 24,
                "total_episodes": 100,
                "avg_accuracy": 67.2,
                "rolling_accuracy": 65.1,
                "avg_reward": 2.3,
            },
        )
        console.emit(event)
        captured = capsys.readouterr()
        assert "BATCH 3" in captured.out
        assert "24/100" in captured.out
        assert "67.2%" in captured.out

    def test_formats_counterfactual_computed(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Verify ConsoleOutput formats COUNTERFACTUAL_COMPUTED correctly."""
        console = ConsoleOutput()
        event = TelemetryEvent(
            event_type=TelemetryEventType.COUNTERFACTUAL_COMPUTED,
            data={
                "slot": 0,
                "contribution": 2.3,
                "marginal_lift": 0.8,
            },
        )
        console.emit(event)
        captured = capsys.readouterr()
        assert "Counterfactual" in captured.out

    def test_formats_checkpoint_saved(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Verify ConsoleOutput formats CHECKPOINT_SAVED correctly."""
        console = ConsoleOutput()
        event = TelemetryEvent(
            event_type=TelemetryEventType.CHECKPOINT_SAVED,
            data={
                "path": "/tmp/checkpoint.pt",
                "episode": 50,
            },
        )
        console.emit(event)
        captured = capsys.readouterr()
        assert "CHECKPOINT" in captured.out
        assert "Saved" in captured.out

    def test_formats_checkpoint_loaded(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Verify ConsoleOutput formats CHECKPOINT_LOADED correctly."""
        console = ConsoleOutput()
        event = TelemetryEvent(
            event_type=TelemetryEventType.CHECKPOINT_LOADED,
            data={
                "path": "/tmp/checkpoint.pt",
                "episode": 50,
            },
        )
        console.emit(event)
        captured = capsys.readouterr()
        assert "CHECKPOINT" in captured.out
        assert "Loaded" in captured.out


class TestNissaHubRouting:
    """Test NissaHub routes events to backends."""

    def test_routes_events_to_single_backend(self) -> None:
        """NissaHub should route events to a single backend."""
        hub = NissaHub()
        mock = MockBackend()
        hub.add_backend(mock)

        event = TelemetryEvent(event_type=TelemetryEventType.GOVERNOR_ROLLBACK)
        hub.emit(event)

        assert len(mock.events) == 1
        assert mock.events[0].event_type == TelemetryEventType.GOVERNOR_ROLLBACK

    def test_routes_events_to_multiple_backends(self) -> None:
        """NissaHub should route events to all backends."""
        hub = NissaHub()
        mock1 = MockBackend()
        mock2 = MockBackend()
        hub.add_backend(mock1)
        hub.add_backend(mock2)

        event = TelemetryEvent(event_type=TelemetryEventType.GOVERNOR_ROLLBACK)
        hub.emit(event)

        assert len(mock1.events) == 1
        assert len(mock2.events) == 1
        assert mock1.events[0].event_type == TelemetryEventType.GOVERNOR_ROLLBACK
        assert mock2.events[0].event_type == TelemetryEventType.GOVERNOR_ROLLBACK

    def test_routes_multiple_events(self) -> None:
        """NissaHub should route multiple events correctly."""
        hub = NissaHub()
        mock = MockBackend()
        hub.add_backend(mock)

        hub.emit(TelemetryEvent(event_type=TelemetryEventType.BATCH_EPOCH_COMPLETED))
        hub.emit(TelemetryEvent(event_type=TelemetryEventType.GOVERNOR_PANIC))
        hub.emit(TelemetryEvent(event_type=TelemetryEventType.CHECKPOINT_SAVED))

        assert len(mock.events) == 3
        assert mock.events[0].event_type == TelemetryEventType.BATCH_EPOCH_COMPLETED
        assert mock.events[1].event_type == TelemetryEventType.GOVERNOR_PANIC
        assert mock.events[2].event_type == TelemetryEventType.CHECKPOINT_SAVED
