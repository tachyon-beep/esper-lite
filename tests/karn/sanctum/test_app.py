"""Tests for Sanctum Textual app."""
import pytest
from unittest.mock import MagicMock

from esper.karn.sanctum.app import SanctumApp
from esper.karn.sanctum.schema import SanctumSnapshot


@pytest.mark.asyncio
async def test_app_launches():
    """App should launch without errors."""
    mock_backend = MagicMock()
    mock_backend.get_snapshot.return_value = SanctumSnapshot()

    app = SanctumApp(backend=mock_backend)
    async with app.run_test():
        assert app.title == "Sanctum - Developer Diagnostics"


@pytest.mark.asyncio
async def test_app_has_main_panels():
    """App should have all required panels."""
    mock_backend = MagicMock()
    mock_backend.get_snapshot.return_value = SanctumSnapshot()

    app = SanctumApp(backend=mock_backend)
    async with app.run_test():
        # Main panels from existing TUI layout
        assert app.query_one("#env-overview") is not None
        assert app.query_one("#scoreboard") is not None
        assert app.query_one("#tamiyo-container") is not None  # Container for dynamic widgets
        assert app.query_one("#event-log") is not None


@pytest.mark.asyncio
async def test_app_quit_binding():
    """Pressing q should trigger quit action."""
    mock_backend = MagicMock()
    mock_backend.get_snapshot.return_value = SanctumSnapshot()

    app = SanctumApp(backend=mock_backend)
    quit_called = False
    original_quit = app.action_quit

    def mock_quit():
        nonlocal quit_called
        quit_called = True
        return original_quit()

    app.action_quit = mock_quit

    async with app.run_test() as pilot:
        # Focus on app (not a child widget) to ensure q binding works
        app.set_focus(None)
        await pilot.pause()
        await pilot.press("q")
        await pilot.pause()
        # action_quit should have been called
        assert quit_called, "action_quit was not called when 'q' pressed"


@pytest.mark.asyncio
async def test_app_focus_navigation():
    """Tab should cycle through focusable panels."""
    mock_backend = MagicMock()
    mock_backend.get_snapshot.return_value = SanctumSnapshot()

    app = SanctumApp(backend=mock_backend)
    async with app.run_test() as pilot:
        await pilot.press("tab")
        # Should have moved focus
        assert app.focused is not None


@pytest.mark.asyncio
async def test_app_has_anomaly_strip():
    """App should have AnomalyStrip widget between RunHeader and main content."""
    from esper.karn.sanctum.widgets import AnomalyStrip

    mock_backend = MagicMock()
    mock_backend.get_snapshot.return_value = SanctumSnapshot()

    app = SanctumApp(backend=mock_backend, num_envs=4)
    async with app.run_test():
        # Query for anomaly strip
        strip = app.query_one("#anomaly-strip", AnomalyStrip)
        assert strip is not None


@pytest.mark.asyncio
async def test_app_shows_thread_death_modal():
    """App should show ThreadDeathModal when thread dies."""
    import threading

    mock_backend = MagicMock()
    mock_backend.get_snapshot.return_value = SanctumSnapshot()

    # Create a thread that immediately stops
    dead_thread = threading.Thread(target=lambda: None)
    dead_thread.start()
    dead_thread.join()  # Wait for it to die

    app = SanctumApp(backend=mock_backend, num_envs=4, training_thread=dead_thread)

    async with app.run_test() as pilot:
        # Trigger a refresh which should detect dead thread
        app._poll_and_refresh()
        await pilot.pause()

        # Check that modal was shown
        assert app._thread_death_shown is True


