"""Tests for Overwatch live mode."""

import pytest

from esper.karn.overwatch.app import OverwatchApp
from esper.karn.overwatch.backend import OverwatchBackend


class TestLiveModeInitialization:
    """Test live mode app initialization."""

    def test_app_accepts_backend_parameter(self):
        """App can be initialized with OverwatchBackend."""
        backend = OverwatchBackend()
        app = OverwatchApp(backend=backend)
        assert app._backend is backend

    def test_app_defaults_to_no_backend(self):
        """App without backend starts in replay/demo mode."""
        app = OverwatchApp()
        assert app._backend is None


class TestLiveModePolling:
    """Test live mode polling behavior."""

    @pytest.mark.asyncio
    async def test_live_mode_starts_polling(self):
        """Live mode starts interval timer on mount."""
        backend = OverwatchBackend()
        backend.start()

        # Emit a training started event
        from esper.leyline import TelemetryEvent, TelemetryEventType
        backend.emit(TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data={"run_id": "test-run", "task": "cifar10"},
        ))

        app = OverwatchApp(backend=backend)

        async with app.run_test():
            # App should have started polling
            assert app._live_timer is not None

            # Should have updated from backend
            assert app._snapshot is not None
            assert app._snapshot.run_id == "test-run"


class TestLiveModeUpdates:
    """Test live mode widget updates."""

    @pytest.mark.asyncio
    async def test_widgets_update_from_backend(self):
        """Widgets receive updates from backend snapshots."""
        backend = OverwatchBackend()
        backend.start()

        # Simulate some telemetry
        from esper.leyline import TelemetryEvent, TelemetryEventType
        backend.emit(TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data={"run_id": "live-test", "task": "cifar10", "num_envs": 2},
        ))

        app = OverwatchApp(backend=backend)

        async with app.run_test() as pilot:
            # Wait for poll
            await pilot.pause()

            # Verify snapshot was loaded
            assert app._snapshot is not None
            assert app._snapshot.run_id == "live-test"
