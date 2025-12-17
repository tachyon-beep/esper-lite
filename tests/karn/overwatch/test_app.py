"""Tests for OverwatchApp."""

from __future__ import annotations

import pytest


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
