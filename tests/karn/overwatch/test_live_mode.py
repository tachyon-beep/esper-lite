"""Tests for Overwatch live mode."""

import pytest

from esper.leyline import TelemetryEvent, TelemetryEventType
from esper.leyline.telemetry import TrainingStartedPayload
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
        backend.emit(TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data=TrainingStartedPayload(
                n_envs=1,
                max_epochs=75,
                task="cifar10",
                host_params=1000,
                slot_ids=("r0c0",),
                seed=42,
                n_episodes=100,
                lr=3e-4,
                clip_ratio=0.2,
                entropy_coef=0.01,
                param_budget=100000,
                policy_device="cuda:0",
                env_devices=("cuda:0",),
                episode_id="test-run",
            ),
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
        backend.emit(TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data=TrainingStartedPayload(
                n_envs=2,
                max_epochs=75,
                task="cifar10",
                host_params=1000,
                slot_ids=("r0c0", "r0c1"),
                seed=42,
                n_episodes=100,
                lr=3e-4,
                clip_ratio=0.2,
                entropy_coef=0.01,
                param_budget=100000,
                policy_device="cuda:0",
                env_devices=("cuda:0", "cuda:1"),
                episode_id="live-test",
            ),
        ))

        app = OverwatchApp(backend=backend)

        async with app.run_test() as pilot:
            # Wait for poll
            await pilot.pause()

            # Verify snapshot was loaded
            assert app._snapshot is not None
            assert app._snapshot.run_id == "live-test"
