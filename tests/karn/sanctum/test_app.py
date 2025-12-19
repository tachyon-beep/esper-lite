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
    async with app.run_test() as pilot:
        assert app.title == "Sanctum - Developer Diagnostics"


@pytest.mark.asyncio
async def test_app_has_main_panels():
    """App should have all required panels."""
    mock_backend = MagicMock()
    mock_backend.get_snapshot.return_value = SanctumSnapshot()

    app = SanctumApp(backend=mock_backend)
    async with app.run_test() as pilot:
        # Main panels from existing TUI layout
        assert app.query_one("#env-overview") is not None
        assert app.query_one("#scoreboard") is not None
        assert app.query_one("#tamiyo-brain") is not None
        assert app.query_one("#system-resources") is not None
        assert app.query_one("#training-health") is not None
        assert app.query_one("#event-log") is not None


@pytest.mark.asyncio
async def test_app_quit_binding():
    """Pressing q should quit the app."""
    mock_backend = MagicMock()
    mock_backend.get_snapshot.return_value = SanctumSnapshot()

    app = SanctumApp(backend=mock_backend)
    async with app.run_test() as pilot:
        await pilot.press("q")
        # App should have initiated exit
        assert not app.is_running


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
