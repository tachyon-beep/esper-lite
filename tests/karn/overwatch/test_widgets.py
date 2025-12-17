"""Tests for Overwatch TUI widgets."""

from __future__ import annotations

import pytest


class TestHelpOverlay:
    """Tests for HelpOverlay widget."""

    def test_help_overlay_imports(self) -> None:
        """HelpOverlay can be imported."""
        from esper.karn.overwatch.widgets.help import HelpOverlay

        assert HelpOverlay is not None

    def test_help_overlay_has_content(self) -> None:
        """HelpOverlay contains help content."""
        from esper.karn.overwatch.widgets.help import HelpOverlay

        widget = HelpOverlay()
        # Widget should have compose method for rendering
        assert callable(getattr(widget, "compose", None))
