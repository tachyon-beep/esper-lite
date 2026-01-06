"""Tests for ActionContext widget."""

from datetime import datetime, timedelta, timezone

from esper.karn.sanctum.schema import DecisionSnapshot, SanctumSnapshot, TamiyoState
from esper.leyline import NUM_OPS
from esper.karn.sanctum.widgets.tamiyo.action_distribution import ActionContext


def test_action_context_critic_preference_flat_q_values_renders_bars():
    """Flat Q-values should still show visible bars (not empty due to zero range)."""
    snapshot = SanctumSnapshot(
        tamiyo=TamiyoState(
            op_q_values=tuple(0.0 for _ in range(NUM_OPS)),
            op_valid_mask=tuple(True for _ in range(NUM_OPS)),
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


def test_action_context_marks_masked_ops():
    """Masked ops should be visibly marked in critic preference."""
    op_q_values = (1.0, 0.5, 0.2, -0.1, 0.0, 0.3)
    op_valid_mask = (True, False, True, True, False, True)

    snapshot = SanctumSnapshot(
        tamiyo=TamiyoState(
            op_q_values=op_q_values,
            op_valid_mask=op_valid_mask,
            q_variance=0.12,
            q_spread=1.1,
        )
    )

    panel = ActionContext()
    panel.update_snapshot(snapshot)

    lines = panel.render().plain.splitlines()
    assert any("[M]" in line for line in lines)


def test_action_context_non_flat_q_values_show_rank_markers():
    """Non-flat Q-values should show BEST/WORST markers."""
    op_q_values = (0.1, 1.2, -0.3, 0.05, 0.4, 0.9)
    op_valid_mask = (True, True, True, True, True, True)

    snapshot = SanctumSnapshot(
        tamiyo=TamiyoState(
            op_q_values=op_q_values,
            op_valid_mask=op_valid_mask,
            q_variance=0.2,
            q_spread=1.5,
        )
    )

    panel = ActionContext()
    panel.update_snapshot(snapshot)

    lines = panel.render().plain.splitlines()
    assert any("← BEST" in line for line in lines)
    assert any("← WORST" in line for line in lines)


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
