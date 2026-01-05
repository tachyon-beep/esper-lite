"""Tests for ConsoleOutput formatting and NissaHub routing."""

from __future__ import annotations

import pytest

from esper.leyline import TelemetryEvent, TelemetryEventType
from esper.leyline.telemetry import (
    BatchEpochCompletedPayload,
    CheckpointLoadedPayload,
    GovernorRollbackPayload,
    PPOUpdatePayload,
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
        """CHECKPOINT_LOADED events print path and episode."""
        console = ConsoleOutput()
        event = TelemetryEvent(
            event_type=TelemetryEventType.CHECKPOINT_LOADED,
            data=CheckpointLoadedPayload(
                path="/tmp/checkpoint.pt",
                start_episode=50,
            ),
        )
        console.emit(event)
        captured = capsys.readouterr()
        assert "CHECKPOINT" in captured.out
        assert "Loaded" in captured.out
        assert "/tmp/checkpoint.pt" in captured.out
        assert "episode 50" in captured.out

    def test_formats_checkpoint_loaded_with_source(self, capsys: pytest.CaptureFixture[str]) -> None:
        """CHECKPOINT_LOADED with source shows source instead of path."""
        console = ConsoleOutput()
        event = TelemetryEvent(
            event_type=TelemetryEventType.CHECKPOINT_LOADED,
            data=CheckpointLoadedPayload(
                path="/tmp/checkpoint.pt",
                start_episode=50,
                source="best checkpoint",
                avg_accuracy=85.5,
            ),
        )
        console.emit(event)
        captured = capsys.readouterr()
        assert "CHECKPOINT" in captured.out
        assert "best checkpoint" in captured.out
        assert "85.5%" in captured.out

    def test_checkpoint_loaded_rejects_invalid_payload(self) -> None:
        """CHECKPOINT_LOADED raises TypeError for non-typed payload."""
        console = ConsoleOutput()
        event = TelemetryEvent(
            event_type=TelemetryEventType.CHECKPOINT_LOADED,
            data={"path": "/tmp/bad.pt", "start_episode": 1},  # dict instead of dataclass
        )
        with pytest.raises(TypeError, match="invalid payload type"):
            console.emit(event)

    def test_formats_ppo_update_completed(self, capsys: pytest.CaptureFixture[str]) -> None:
        """PPO_UPDATE_COMPLETED events print loss components."""
        console = ConsoleOutput()
        event = TelemetryEvent(
            event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
            data=PPOUpdatePayload(
                policy_loss=0.1234,
                value_loss=0.5678,
                entropy=0.456,
                grad_norm=1.0,
                kl_divergence=0.005,
                clip_fraction=0.15,
                nan_grad_count=0,
                entropy_coef=0.01,
                skipped=False,
            ),
        )
        console.emit(event)
        captured = capsys.readouterr()
        assert "PPO" in captured.out
        assert "policy=0.1234" in captured.out
        assert "value=0.5678" in captured.out
        assert "entropy=0.456" in captured.out
        assert "coef=0.0100" in captured.out

    def test_formats_ppo_update_skipped(self, capsys: pytest.CaptureFixture[str]) -> None:
        """PPO_UPDATE_COMPLETED with skipped=True shows skip message."""
        console = ConsoleOutput()
        event = TelemetryEvent(
            event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
            data=PPOUpdatePayload(
                policy_loss=0.0,
                value_loss=0.0,
                entropy=0.0,
                grad_norm=0.0,
                kl_divergence=0.0,
                clip_fraction=0.0,
                nan_grad_count=0,
                skipped=True,
            ),
        )
        console.emit(event)
        captured = capsys.readouterr()
        assert "PPO" in captured.out
        assert "skipped" in captured.out


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

