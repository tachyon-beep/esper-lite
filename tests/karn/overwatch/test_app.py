"""Tests for OverwatchApp."""

from __future__ import annotations


class TestOverwatchApp:
    """Tests for OverwatchApp class."""

    def test_app_imports(self) -> None:
        """OverwatchApp can be imported."""
        from esper.karn.overwatch.app import OverwatchApp

        assert OverwatchApp is not None

    def test_app_has_compose(self) -> None:
        """OverwatchApp has compose method."""
        from esper.karn.overwatch.app import OverwatchApp

        app = OverwatchApp()
        assert callable(getattr(app, "compose", None))

    def test_app_has_bindings(self) -> None:
        """OverwatchApp has keyboard bindings."""
        from esper.karn.overwatch.app import OverwatchApp

        app = OverwatchApp()
        binding_keys = [b.key for b in app.BINDINGS]
        assert "q" in binding_keys
        assert "question_mark" in binding_keys


class TestPackageExports:
    """Tests for package-level exports."""

    def test_app_importable_from_package(self) -> None:
        """OverwatchApp importable from overwatch package."""
        from esper.karn.overwatch import OverwatchApp

        assert OverwatchApp is not None

    def test_help_overlay_importable_from_widgets(self) -> None:
        """HelpOverlay importable from widgets package."""
        from esper.karn.overwatch.widgets import HelpOverlay

        assert HelpOverlay is not None
