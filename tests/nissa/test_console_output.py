"""Tests for ConsoleOutput formatting and NissaHub routing."""

from __future__ import annotations

import pytest

from esper.leyline import TelemetryEvent, TelemetryEventType
from esper.leyline.telemetry import (
    BatchEpochCompletedPayload,
    GovernorRollbackPayload,
)
from esper.nissa.output import ConsoleOutput, NissaHub


class TestConsoleOutputFormatters:
    """ConsoleOutput formats key event types without crashing."""

    def test_formats_governor_rollback(self, capsys: pytest.CaptureFixture[str]) -> None:
        console = ConsoleOutput()
        event = TelemetryEvent(
            event_type=TelemetryEventType.GOVERNOR_ROLLBACK,
            data=GovernorRollbackPayload(
                env_id=0,
                device="cpu",
                reason="Structural Collapse",
                loss_at_panic=15.3,
                loss_threshold=5.2,
                consecutive_panics=2,
            ),
        )
        console.emit(event)
        captured = capsys.readouterr()
        assert "GOVERNOR" in captured.out
        assert "ROLLBACK" in captured.out
        assert "Structural Collapse" in captured.out

    def test_formats_batch_completed(self, capsys: pytest.CaptureFixture[str]) -> None:
        console = ConsoleOutput()
        event = TelemetryEvent(
            event_type=TelemetryEventType.BATCH_EPOCH_COMPLETED,
            data=BatchEpochCompletedPayload(
                batch_idx=3,
                episodes_completed=24,
                total_episodes=100,
                avg_accuracy=67.2,
                rolling_accuracy=65.1,
                avg_reward=2.3,
                n_envs=1,
            ),
        )
        console.emit(event)
        captured = capsys.readouterr()
        assert "BATCH 3" in captured.out
        assert "24/100" in captured.out
        assert "67.2%" in captured.out

    def test_formats_checkpoint_loaded(self, capsys: pytest.CaptureFixture[str]) -> None:
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


class _MockBackend:
    def __init__(self) -> None:
        self.events: list[TelemetryEvent] = []

    def start(self) -> None:
        pass

    def emit(self, event: TelemetryEvent) -> None:
        self.events.append(event)

    def close(self) -> None:
        pass


class TestNissaHubRouting:
    """NissaHub routes enqueued events to all backends."""

    def test_routes_events_to_single_backend(self) -> None:
        hub = NissaHub()
        backend = _MockBackend()
        hub.add_backend(backend)

        hub.emit(TelemetryEvent(event_type=TelemetryEventType.GOVERNOR_ROLLBACK))
        hub.close()

        assert len(backend.events) == 1
        assert backend.events[0].event_type == TelemetryEventType.GOVERNOR_ROLLBACK

    def test_routes_events_to_multiple_backends(self) -> None:
        hub = NissaHub()
        backend1 = _MockBackend()
        backend2 = _MockBackend()
        hub.add_backend(backend1)
        hub.add_backend(backend2)

        hub.emit(TelemetryEvent(event_type=TelemetryEventType.GOVERNOR_ROLLBACK))
        hub.close()

        assert len(backend1.events) == 1
        assert len(backend2.events) == 1
        assert backend1.events[0].event_type == TelemetryEventType.GOVERNOR_ROLLBACK
        assert backend2.events[0].event_type == TelemetryEventType.GOVERNOR_ROLLBACK

