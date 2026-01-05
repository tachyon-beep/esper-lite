"""Tests for ActionContext widget."""

from datetime import datetime, timedelta, timezone

from esper.karn.sanctum.schema import DecisionSnapshot, SanctumSnapshot, TamiyoState
from esper.karn.sanctum.widgets.tamiyo.action_distribution import ActionContext


def test_action_context_critic_preference_flat_q_values_renders_bars():
    """Flat Q-values should still show visible bars (not empty due to zero range)."""
    snapshot = SanctumSnapshot(
        tamiyo=TamiyoState(
            q_germinate=0.0,
            q_advance=0.0,
            q_set_alpha=0.0,
            q_fossilize=0.0,
            q_prune=0.0,
            q_wait=0.0,
            q_variance=0.0,
            q_spread=0.0,
        )
    )

    panel = ActionContext()
    panel.update_snapshot(snapshot)

    lines = panel.render().plain.splitlines()
    start = lines.index("▶ Critic Preference")
    q_lines = lines[start + 1 : start + 7]
    assert any("█" in line for line in q_lines)


def test_action_context_sequence_renders_warnings_on_separate_line():
    """Sequence pattern warnings should not share the arrow chain line."""
    now = datetime.now(timezone.utc)
    slot_states = {"r0c0": "Empty"}  # Dormant slot exists -> enables STUCK detection
    decisions = [
        DecisionSnapshot(
            timestamp=now - timedelta(seconds=i),
            slot_states=slot_states,
            host_accuracy=0.0,
            chosen_action="WAIT",
            chosen_slot=None,
            confidence=1.0,
            expected_value=0.0,
            actual_reward=None,
            alternatives=[],
            action_success=True,
        )
        for i in range(12)
    ]

    snapshot = SanctumSnapshot(tamiyo=TamiyoState(recent_decisions=decisions))

    panel = ActionContext()
    panel.update_snapshot(snapshot)

    lines = panel.render().plain.splitlines()
    warning_line = next(line for line in lines if "⚠ STUCK" in line)
    assert "→" not in warning_line
