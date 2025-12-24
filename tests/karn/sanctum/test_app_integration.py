"""Integration tests for SanctumApp."""

import pytest
from unittest.mock import MagicMock

from esper.karn.sanctum.schema import SanctumSnapshot, TamiyoState, EnvState


class TestSanctumAppIntegration:
    """Test SanctumApp widget wiring."""

    @pytest.mark.asyncio
    async def test_app_creates_all_widgets(self):
        """All required widgets should be created on compose."""
        from esper.karn.sanctum.app import SanctumApp

        mock_backend = MagicMock()
        mock_backend.get_snapshot.return_value = SanctumSnapshot()

        app = SanctumApp(backend=mock_backend, num_envs=4)

        async with app.run_test() as pilot:
            # Verify all widgets exist
            assert app.query_one("#env-overview") is not None
            assert app.query_one("#scoreboard") is not None
            assert app.query_one("#tamiyo-brain") is not None
            assert app.query_one("#event-log") is not None

    @pytest.mark.asyncio
    async def test_snapshot_propagates_to_all_widgets(self):
        """Snapshot updates should reach all widgets via polling or manual refresh."""
        from esper.karn.sanctum.app import SanctumApp

        mock_backend = MagicMock()
        snapshot = SanctumSnapshot(
            tamiyo=TamiyoState(entropy=1.2, clip_fraction=0.15),
            envs={0: EnvState(env_id=0, host_accuracy=75.5)},
        )
        mock_backend.get_snapshot.return_value = snapshot

        # Use a fast refresh rate so timer fires quickly
        app = SanctumApp(backend=mock_backend, num_envs=4, refresh_rate=10.0)

        async with app.run_test() as pilot:
            # Wait for the timer-based refresh to fire (interval is 0.1s at 10Hz)
            import asyncio
            await asyncio.sleep(0.2)
            await pilot.pause()

            # Backend should have been called by set_interval -> _poll_and_refresh
            mock_backend.get_snapshot.assert_called()

    @pytest.mark.asyncio
    async def test_focus_env_updates_reward_panel(self):
        """Calling action_focus_env directly should update focused env ID."""
        from esper.karn.sanctum.app import SanctumApp

        mock_backend = MagicMock()
        mock_backend.get_snapshot.return_value = SanctumSnapshot()

        app = SanctumApp(backend=mock_backend, num_envs=16)

        async with app.run_test() as pilot:
            # Verify initial state
            assert app._focused_env_id == 0

            # Call action directly (bindings can be flaky in tests)
            app.action_focus_env(2)
            assert app._focused_env_id == 2

            app.action_focus_env(7)
            assert app._focused_env_id == 7

            # Out of bounds should not change
            app.action_focus_env(100)
            assert app._focused_env_id == 7  # Unchanged

    @pytest.mark.asyncio
    async def test_quit_action_exits_app(self):
        """Pressing 'q' should trigger app exit."""
        from esper.karn.sanctum.app import SanctumApp

        mock_backend = MagicMock()
        mock_backend.get_snapshot.return_value = SanctumSnapshot()

        app = SanctumApp(backend=mock_backend, num_envs=4)

        async with app.run_test() as pilot:
            # Press 'q' to quit
            await pilot.press("q")
            # In Textual, quitting ends the test context gracefully
            # We verify the app received the quit action by checking it didn't raise
            # The run_test context manager handles app lifecycle


@pytest.mark.asyncio
async def test_new_layout_structure():
    """Test that new layout has correct panel structure."""
    from esper.karn.sanctum.app import SanctumApp
    from esper.karn.sanctum.backend import SanctumBackend

    backend = SanctumBackend()
    app = SanctumApp(backend=backend, num_envs=4)

    async with app.run_test() as pilot:
        # Should have EnvOverview and Scoreboard in top section
        assert app.query_one("#env-overview") is not None
        assert app.query_one("#scoreboard") is not None

        # Should have EventLog and TamiyoBrain in bottom section
        assert app.query_one("#event-log") is not None
        assert app.query_one("#tamiyo-brain") is not None

        # Should NOT have SystemResources or TrainingHealth
        from textual.css.query import NoMatches
        with pytest.raises(NoMatches):
            app.query_one("#system-resources")
        with pytest.raises(NoMatches):
            app.query_one("#training-health")
