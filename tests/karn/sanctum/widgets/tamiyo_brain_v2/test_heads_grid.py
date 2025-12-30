"""Tests for HeadsPanel widget."""
from __future__ import annotations


def test_heads_panel_shows_gradient_flow_footer():
    """HeadsPanel should show gradient flow metrics as footer row."""
    from esper.karn.sanctum.schema import (
        GradientQualityMetrics,
        SanctumSnapshot,
        TamiyoState,
    )
    from esper.karn.sanctum.widgets.tamiyo_brain_v2.heads_grid import HeadsPanel

    snapshot = SanctumSnapshot()
    snapshot.tamiyo = TamiyoState()
    snapshot.tamiyo.gradient_quality = GradientQualityMetrics(
        gradient_cv=0.42,
        clip_fraction_positive=0.10,
        clip_fraction_negative=0.05,
    )
    # Set existing dead/exploding layers (already tracked)
    snapshot.tamiyo.dead_layers = 0
    snapshot.tamiyo.exploding_layers = 0

    panel = HeadsPanel()
    panel.update_snapshot(snapshot)
    content = panel.render()

    content_str = str(content)
    # Should show gradient CV with value and status
    assert "CV:0.42" in content_str
    assert "stable" in content_str
    # Should show directional clip fractions
    assert "Clip:" in content_str
