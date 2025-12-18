"""Tests for Sanctum Textual app."""
import pytest

from esper.karn.sanctum.app import SanctumApp


@pytest.mark.asyncio
async def test_app_launches():
    """App should launch without errors."""
    app = SanctumApp()
    async with app.run_test() as pilot:
        assert app.title == "Esper Sanctum"


@pytest.mark.asyncio
async def test_app_has_main_panels():
    """App should have all required panels."""
    app = SanctumApp()
    async with app.run_test() as pilot:
        # Main panels from existing TUI layout
        assert app.query_one("#env-overview") is not None
        assert app.query_one("#scoreboard") is not None
        assert app.query_one("#tamiyo-brain") is not None
        assert app.query_one("#reward-components") is not None
        assert app.query_one("#esper-status") is not None
        assert app.query_one("#event-log") is not None


@pytest.mark.asyncio
async def test_app_quit_binding():
    """Pressing q should quit the app."""
    app = SanctumApp()
    async with app.run_test() as pilot:
        await pilot.press("q")
        # App should have initiated exit
        assert not app.is_running


@pytest.mark.asyncio
async def test_app_focus_navigation():
    """Tab should cycle through focusable panels."""
    app = SanctumApp()
    async with app.run_test() as pilot:
        await pilot.press("tab")
        # Should have moved focus
        assert app.focused is not None
