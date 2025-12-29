"""Tests for PPOHealthPanel - directional clip fraction display."""

from esper.karn.sanctum.schema import (
    GradientQualityMetrics,
    SanctumSnapshot,
    TamiyoState,
)
from esper.karn.sanctum.widgets.tamiyo_brain_v2.ppo_health import PPOHealthPanel


def test_ppo_health_shows_directional_clip():
    """PPO Health panel should show clip up/down breakdown."""
    snapshot = SanctumSnapshot()
    snapshot.tamiyo = TamiyoState(clip_fraction=0.15)
    snapshot.tamiyo.gradient_quality = GradientQualityMetrics(
        clip_fraction_positive=0.10,
        clip_fraction_negative=0.05,
    )
    snapshot.current_batch = 60

    panel = PPOHealthPanel()
    panel._snapshot = snapshot
    content = panel.render()

    content_str = str(content)
    # Should show directional breakdown with arrows
    assert "↑" in content_str or "↓" in content_str
