"""Tests for PPOLossesPanel - combined PPO gauges and loss sparklines."""

from collections import deque

from esper.karn.sanctum.schema import (
    GradientQualityMetrics,
    SanctumSnapshot,
    TamiyoState,
)
from esper.karn.sanctum.widgets.tamiyo_brain_v2.ppo_losses_panel import PPOLossesPanel


def test_ppo_losses_shows_gauges():
    """PPOLossesPanel should show the 3 main PPO gauge metrics."""
    snapshot = SanctumSnapshot()
    snapshot.tamiyo = TamiyoState(
        explained_variance=0.85,
        entropy=1.2,
        clip_fraction=0.15,
        kl_divergence=0.02,
    )
    snapshot.current_batch = 60

    panel = PPOLossesPanel()
    panel._snapshot = snapshot
    content = panel.render()

    content_str = str(content)
    # Should show 3 main gauge labels (Expl.Var, Entropy, Clip Frac)
    assert "Expl.Var" in content_str
    assert "Entropy" in content_str
    assert "Clip Frac" in content_str
    # Should have gauge bars
    assert "[" in content_str and "]" in content_str


def test_ppo_losses_shows_sparklines():
    """PPOLossesPanel should show loss sparklines and value/policy ratio."""
    snapshot = SanctumSnapshot()
    snapshot.tamiyo = TamiyoState(
        policy_loss=0.05,
        value_loss=0.12,
        current_episode_return=-5.5,
        episode_return_history=deque([1.0, 2.0, 3.0, 4.0, 5.0]),
        policy_loss_history=deque([0.1, 0.09, 0.08, 0.07, 0.05]),
        value_loss_history=deque([0.2, 0.18, 0.15, 0.13, 0.12]),
    )
    snapshot.current_batch = 60

    panel = PPOLossesPanel()
    panel._snapshot = snapshot
    content = panel.render()

    content_str = str(content)
    # Should show sparkline labels (P.Loss and V.Loss, not Ep.Return anymore)
    assert "P.Loss" in content_str
    assert "V.Loss" in content_str
    # Should show value/policy loss ratio
    assert "L_v/L_p" in content_str


def test_ppo_losses_shows_directional_clip():
    """PPOLossesPanel should show clip fraction breakdown with arrows."""
    snapshot = SanctumSnapshot()
    snapshot.tamiyo = TamiyoState(clip_fraction=0.15)
    snapshot.tamiyo.gradient_quality = GradientQualityMetrics(
        clip_fraction_positive=0.10,
        clip_fraction_negative=0.05,
    )
    snapshot.current_batch = 60

    panel = PPOLossesPanel()
    panel._snapshot = snapshot
    content = panel.render()

    content_str = str(content)
    # Should show directional breakdown with arrows
    assert "↑" in content_str and "↓" in content_str


def test_ppo_losses_has_separator():
    """PPOLossesPanel should have a separator line between gauge and sparkline sections."""
    snapshot = SanctumSnapshot()
    snapshot.tamiyo = TamiyoState(
        explained_variance=0.85,
        entropy=1.2,
        clip_fraction=0.15,
        kl_divergence=0.02,
        policy_loss=0.05,
        value_loss=0.12,
    )
    snapshot.current_batch = 60

    panel = PPOLossesPanel()
    panel._snapshot = snapshot
    content = panel.render()

    content_str = str(content)
    # Should have separator line (multiple dashes)
    assert "─" in content_str


def test_ppo_losses_warmup_title():
    """PPOLossesPanel should show warmup status in title during warmup."""
    snapshot = SanctumSnapshot()
    snapshot.tamiyo = TamiyoState()
    snapshot.current_batch = 25  # During warmup

    panel = PPOLossesPanel()
    panel.update_snapshot(snapshot)

    # Should show warmup in border title
    assert "WARMING UP" in panel.border_title
    assert "25/50" in panel.border_title


def test_ppo_losses_collapse_warning():
    """PPOLossesPanel should show collapse warning when risk is high."""
    snapshot = SanctumSnapshot()
    snapshot.tamiyo = TamiyoState(
        entropy=0.4,  # Low entropy
        entropy_velocity=-0.01,  # Declining
        collapse_risk_score=0.8,  # High risk
    )
    snapshot.current_batch = 60

    panel = PPOLossesPanel()
    panel.update_snapshot(snapshot)

    # Should show collapse warning in border title
    assert "COLLAPSE" in panel.border_title
