"""Tests for TamiyoDetailPanel widget."""

from __future__ import annotations

from esper.karn.overwatch.schema import TamiyoState


class TestTamiyoDetailPanel:
    """Tests for TamiyoDetailPanel widget."""

    def test_tamiyo_detail_imports(self) -> None:
        """TamiyoDetailPanel can be imported."""
        from esper.karn.overwatch.widgets.tamiyo_detail import TamiyoDetailPanel

        assert TamiyoDetailPanel is not None

    def test_tamiyo_detail_renders_action_distribution(self) -> None:
        """TamiyoDetailPanel shows action distribution bars."""
        from esper.karn.overwatch.widgets.tamiyo_detail import TamiyoDetailPanel

        tamiyo = TamiyoState(
            action_counts={"GERMINATE": 34, "BLEND": 28, "PRUNE": 12, "WAIT": 26},
        )
        panel = TamiyoDetailPanel()
        panel.update_tamiyo(tamiyo)

        content = panel.render_content()
        assert "GERM" in content or "Germinate" in content
        assert "BLEND" in content or "Blend" in content
        assert "34%" in content or "34" in content
        # Should have visual bars
        assert "█" in content or "▓" in content or "=" in content

    def test_tamiyo_detail_renders_recent_actions_grid(self) -> None:
        """TamiyoDetailPanel shows recent actions in grid format."""
        from esper.karn.overwatch.widgets.tamiyo_detail import TamiyoDetailPanel

        tamiyo = TamiyoState(
            recent_actions=["G", "B", "B", "W", "G", "P", "W", "W", "B", "G"],
        )
        panel = TamiyoDetailPanel()
        panel.update_tamiyo(tamiyo)

        content = panel.render_content()
        assert "Recent" in content
        # Should show the action codes
        assert "G" in content
        assert "B" in content
        assert "W" in content

    def test_tamiyo_detail_renders_confidence(self) -> None:
        """TamiyoDetailPanel shows confidence metrics."""
        from esper.karn.overwatch.widgets.tamiyo_detail import TamiyoDetailPanel

        tamiyo = TamiyoState(
            confidence_mean=0.73,
            confidence_min=0.45,
            confidence_max=0.92,
        )
        panel = TamiyoDetailPanel()
        panel.update_tamiyo(tamiyo)

        content = panel.render_content()
        assert "Confidence" in content
        assert "73" in content  # mean
        assert "45" in content or "92" in content  # min or max

    def test_tamiyo_detail_renders_exploration(self) -> None:
        """TamiyoDetailPanel shows exploration percentage."""
        from esper.karn.overwatch.widgets.tamiyo_detail import TamiyoDetailPanel

        tamiyo = TamiyoState(
            exploration_pct=0.65,
            entropy=1.5,
        )
        panel = TamiyoDetailPanel()
        panel.update_tamiyo(tamiyo)

        content = panel.render_content()
        assert "Exploration" in content or "Entropy" in content
        assert "65" in content or "1.5" in content

    def test_tamiyo_detail_renders_learning_signals(self) -> None:
        """TamiyoDetailPanel shows PPO learning signals with health status."""
        from esper.karn.overwatch.widgets.tamiyo_detail import TamiyoDetailPanel

        tamiyo = TamiyoState(
            kl_divergence=0.015,
            explained_variance=0.75,
            clip_fraction=0.08,
            grad_norm=0.5,
            learning_rate=3e-4,
        )
        panel = TamiyoDetailPanel()
        panel.update_tamiyo(tamiyo)

        content = panel.render_content()
        assert "KL" in content
        assert "0.015" in content or "015" in content
        assert "EV" in content or "Explained" in content
        assert "0.75" in content or "75" in content

    def test_tamiyo_detail_health_colors(self) -> None:
        """TamiyoDetailPanel applies health-based colors to signals."""
        from esper.karn.overwatch.widgets.tamiyo_detail import TamiyoDetailPanel

        # Critical entropy (collapsed)
        tamiyo = TamiyoState(
            entropy=0.1,
            entropy_collapsed=True,
        )
        panel = TamiyoDetailPanel()
        panel.update_tamiyo(tamiyo)

        content = panel.render_content()
        # Should have red color markup for critical entropy
        assert "red" in content or "CRIT" in content or "⚠" in content

    def test_tamiyo_detail_empty_state(self) -> None:
        """TamiyoDetailPanel shows empty state when no data."""
        from esper.karn.overwatch.widgets.tamiyo_detail import TamiyoDetailPanel

        panel = TamiyoDetailPanel()

        content = panel.render_content()
        assert "Waiting" in content or "warmup" in content or "No data" in content

    def test_tamiyo_detail_sparkline_placeholder(self) -> None:
        """TamiyoDetailPanel shows confidence history as sparkline."""
        from esper.karn.overwatch.widgets.tamiyo_detail import TamiyoDetailPanel

        tamiyo = TamiyoState(
            confidence_history=[0.5, 0.6, 0.7, 0.65, 0.8, 0.75, 0.9],
        )
        panel = TamiyoDetailPanel()
        panel.update_tamiyo(tamiyo)

        content = panel.render_content()
        # Should have some visual representation
        assert "▁" in content or "▂" in content or "▃" in content or "History" in content
