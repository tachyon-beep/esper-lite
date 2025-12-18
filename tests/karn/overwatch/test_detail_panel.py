"""Tests for DetailPanel container widget."""

from __future__ import annotations

from esper.karn.overwatch.schema import (
    EnvSummary,
    TamiyoState,
)


class TestDetailPanel:
    """Tests for DetailPanel container."""

    def test_detail_panel_imports(self) -> None:
        """DetailPanel can be imported."""
        from esper.karn.overwatch.widgets.detail_panel import DetailPanel

        assert DetailPanel is not None

    def test_detail_panel_starts_in_context_mode(self) -> None:
        """DetailPanel starts in context mode by default."""
        from esper.karn.overwatch.widgets.detail_panel import DetailPanel

        panel = DetailPanel()
        assert panel.mode == "context"

    def test_detail_panel_switches_to_tamiyo_mode(self) -> None:
        """DetailPanel can switch to tamiyo mode."""
        from esper.karn.overwatch.widgets.detail_panel import DetailPanel

        panel = DetailPanel()
        panel.set_mode("tamiyo")
        assert panel.mode == "tamiyo"

    def test_detail_panel_toggle_mode(self) -> None:
        """DetailPanel toggles between modes."""
        from esper.karn.overwatch.widgets.detail_panel import DetailPanel

        panel = DetailPanel()
        assert panel.mode == "context"

        panel.toggle_mode("tamiyo")
        assert panel.mode == "tamiyo"

        panel.toggle_mode("tamiyo")  # Toggle same mode hides (back to context)
        assert panel.mode == "context"

        panel.toggle_mode("context")
        assert panel.mode == "context"

    def test_detail_panel_updates_env(self) -> None:
        """DetailPanel forwards env updates to context panel."""
        from esper.karn.overwatch.widgets.detail_panel import DetailPanel

        panel = DetailPanel()
        env = EnvSummary(env_id=3, device_id=1, status="WARN")
        panel.update_env(env)

        assert panel._env == env

    def test_detail_panel_updates_tamiyo(self) -> None:
        """DetailPanel forwards tamiyo updates to tamiyo panel."""
        from esper.karn.overwatch.widgets.detail_panel import DetailPanel

        panel = DetailPanel()
        tamiyo = TamiyoState(entropy=1.5)
        panel.update_tamiyo(tamiyo)

        assert panel._tamiyo == tamiyo

    def test_detail_panel_mode_property(self) -> None:
        """DetailPanel exposes current mode."""
        from esper.karn.overwatch.widgets.detail_panel import DetailPanel

        panel = DetailPanel()
        assert panel.mode in ("context", "tamiyo")
