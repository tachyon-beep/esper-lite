"""Tests for Nissa global hub singleton reset behavior."""

from esper.leyline import TelemetryEvent, TelemetryEventType


class TestGlobalHubReset:
    def test_reset_hub_clears_backends_between_runs(self) -> None:
        from esper.nissa import get_hub, reset_hub

        reset_hub()
        hub = get_hub()

        captured: list[TelemetryEvent] = []

        class CaptureBackend:
            def start(self) -> None:
                pass

            def emit(self, event: TelemetryEvent) -> None:
                captured.append(event)

            def close(self) -> None:
                pass

        backend = CaptureBackend()
        hub.add_backend(backend)

        hub.emit(TelemetryEvent(event_type=TelemetryEventType.EPOCH_COMPLETED, epoch=1))
        hub.flush()  # Wait for async worker to process the event
        assert len(captured) == 1

        # Reset should clear all backends so subsequent emits don't duplicate.
        reset_hub()

        hub.emit(TelemetryEvent(event_type=TelemetryEventType.EPOCH_COMPLETED, epoch=2))
        hub.flush()  # Wait for async processing (should be no-op since reset clears backends)
        assert len(captured) == 1

        # Clean up (idempotent)
        reset_hub()
