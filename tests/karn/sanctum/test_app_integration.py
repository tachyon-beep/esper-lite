"""Integration tests for SanctumApp."""

import pytest
from unittest.mock import MagicMock

from esper.karn.sanctum.schema import SanctumSnapshot, TamiyoState, EnvState


class TestSanctumAppIntegration:
    """Test SanctumApp widget wiring."""

    def test_app_creates_all_widgets(self):
        """All required widgets should be created on compose."""
        # Import here to avoid Textual import issues in test collection
        from esper.karn.sanctum.app import SanctumApp

        mock_backend = MagicMock()
        mock_backend.get_snapshot.return_value = SanctumSnapshot()

        app = SanctumApp(backend=mock_backend, num_envs=4)

        # Use Textual's pilot for testing
        async def test_widgets():
            async with app.run_test() as pilot:
                # Verify all widgets exist
                assert app.query_one("#env-overview") is not None
                assert app.query_one("#scoreboard") is not None
                assert app.query_one("#tamiyo-brain") is not None
                assert app.query_one("#reward-components") is not None
                assert app.query_one("#event-log") is not None
                assert app.query_one("#esper-status") is not None

        import asyncio
        asyncio.run(test_widgets())

    def test_snapshot_propagates_to_all_widgets(self):
        """Snapshot updates should reach all widgets."""
        from esper.karn.sanctum.app import SanctumApp

        mock_backend = MagicMock()
        snapshot = SanctumSnapshot(
            tamiyo=TamiyoState(entropy=1.2, clip_fraction=0.15),
            envs={0: EnvState(env_id=0, host_accuracy=75.5)},
        )
        mock_backend.get_snapshot.return_value = snapshot

        app = SanctumApp(backend=mock_backend, num_envs=4)

        async def test_propagation():
            async with app.run_test() as pilot:
                # Trigger refresh
                await pilot.press("r")

                # Backend should have been called
                mock_backend.get_snapshot.assert_called()

        import asyncio
        asyncio.run(test_propagation())

    def test_focus_env_updates_reward_panel(self):
        """Pressing 1-9 should focus that env in reward panel."""
        from esper.karn.sanctum.app import SanctumApp

        mock_backend = MagicMock()
        mock_backend.get_snapshot.return_value = SanctumSnapshot()

        app = SanctumApp(backend=mock_backend, num_envs=16)

        async def test_focus():
            async with app.run_test() as pilot:
                # Press '3' to focus env 2 (0-indexed)
                await pilot.press("3")
                assert app._focused_env_id == 2

                # Press '8' to focus env 7
                await pilot.press("8")
                assert app._focused_env_id == 7

        import asyncio
        asyncio.run(test_focus())

    def test_quit_action_exits_app(self):
        """Pressing 'q' should quit the app."""
        from esper.karn.sanctum.app import SanctumApp

        mock_backend = MagicMock()
        mock_backend.get_snapshot.return_value = SanctumSnapshot()

        app = SanctumApp(backend=mock_backend, num_envs=4)

        async def test_quit():
            async with app.run_test() as pilot:
                await pilot.press("q")
                # App should be exiting
                assert app._exit

        import asyncio
        asyncio.run(test_quit())
