"""Tests for DecisionDetailScreen modal.

Tests cover:
- DecisionDetailScreen: Drill-down view for a single Tamiyo decision
- Bug-hiding pattern fixes (decision_id without fallback)
"""
from __future__ import annotations

from datetime import datetime, timezone
from io import StringIO

from rich.console import Console
from rich.text import Text

from esper.karn.sanctum.schema import DecisionSnapshot
from esper.karn.sanctum.widgets.tamiyo_brain.decision_detail_screen import (
    DecisionDetailScreen,
)


def render_to_text(renderable) -> str:
    """Helper to render a Rich renderable to plain text."""
    console = Console(file=StringIO(), width=120, legacy_windows=False)
    console.print(renderable)
    return console.file.getvalue()


def make_decision_snapshot(
    *,
    decision_id: str = "test-decision-001",
    env_id: int = 0,
    epoch: int = 10,
    batch: int = 5,
    chosen_action: str = "GERMINATE",
    chosen_slot: str | None = "r0c0",
    confidence: float = 0.85,
    decision_entropy: float = 0.5,
    expected_value: float = 0.1,
    actual_reward: float | None = 0.15,
    td_advantage: float | None = 0.05,
) -> DecisionSnapshot:
    """Create a DecisionSnapshot for testing."""
    return DecisionSnapshot(
        decision_id=decision_id,
        timestamp=datetime.now(timezone.utc),
        env_id=env_id,
        epoch=epoch,
        batch=batch,
        host_accuracy=80.0,
        chosen_action=chosen_action,
        chosen_slot=chosen_slot,
        confidence=confidence,
        decision_entropy=decision_entropy,
        expected_value=expected_value,
        actual_reward=actual_reward,
        value_residual=0.0,
        td_advantage=td_advantage,
        chosen_blueprint="conv3x3",
        blueprint_confidence=0.9,
        blueprint_entropy=0.3,
        chosen_tempo="STANDARD",
        tempo_confidence=0.8,
        tempo_entropy=0.4,
        chosen_style="EAGER",
        style_confidence=0.75,
        style_entropy=0.5,
        chosen_curve="LINEAR",
        curve_confidence=0.7,
        curve_entropy=0.6,
        chosen_alpha_target="α=0.5",
        alpha_target_confidence=0.65,
        alpha_target_entropy=0.7,
        chosen_alpha_speed="MEDIUM",
        alpha_speed_confidence=0.6,
        alpha_speed_entropy=0.8,
        slot_states={"r0c0": "TRAINING", "r0c1": "DORMANT"},
        alternatives=[("WAIT", 0.10), ("PRUNE", 0.05)],
    )


class TestDecisionDetailScreenRendering:
    """Test DecisionDetailScreen renders all fields correctly."""

    def test_modal_creation(self):
        """Modal should be creatable with decision and group_id."""
        decision = make_decision_snapshot()
        modal = DecisionDetailScreen(decision=decision, group_id="env0_ep10")
        assert modal is not None
        assert modal._decision == decision
        assert modal._group_id == "env0_ep10"

    def test_renders_title(self):
        """Title should show group ID."""
        decision = make_decision_snapshot()
        modal = DecisionDetailScreen(decision=decision, group_id="test_group")
        title = modal._render_title()

        assert "TAMIYO DECISION DETAIL" in title.plain
        assert "test_group" in title.plain

    def test_renders_summary_section(self):
        """Detail should include summary with decision ID."""
        decision = make_decision_snapshot(decision_id="decision-abc-123")
        modal = DecisionDetailScreen(decision=decision, group_id="test")
        detail = modal._render_detail()
        detail_text = detail.plain

        assert "Summary" in detail_text
        assert "decision-abc-123" in detail_text
        assert "Env:" in detail_text
        assert "Epoch:" in detail_text

    def test_decision_id_displayed_directly(self):
        """Decision ID should be displayed directly without fallback."""
        # This tests the bug-hiding fix - decision_id is required
        decision = make_decision_snapshot(decision_id="my-unique-id")
        modal = DecisionDetailScreen(decision=decision, group_id="test")
        detail = modal._render_detail()
        detail_text = detail.plain

        assert "Decision ID: my-unique-id" in detail_text
        # Should NOT contain the old fallback
        assert "(missing)" not in detail_text

    def test_renders_action_section(self):
        """Detail should include action information."""
        decision = make_decision_snapshot(
            chosen_action="GERMINATE",
            chosen_slot="r1c0",
            confidence=0.92,
        )
        modal = DecisionDetailScreen(decision=decision, group_id="test")
        detail = modal._render_detail()
        detail_text = detail.plain

        assert "Action" in detail_text
        assert "Op:" in detail_text
        assert "GERMINATE" in detail_text
        assert "r1c0" in detail_text
        assert "92%" in detail_text  # confidence

    def test_renders_values_section(self):
        """Detail should include value function information."""
        decision = make_decision_snapshot(
            expected_value=0.25,
            actual_reward=0.30,
            td_advantage=0.05,
        )
        modal = DecisionDetailScreen(decision=decision, group_id="test")
        detail = modal._render_detail()
        detail_text = detail.plain

        assert "Values" in detail_text
        assert "V(s):" in detail_text
        assert "Reward:" in detail_text
        assert "TD(0):" in detail_text

    def test_renders_pending_reward(self):
        """Pending reward should show 'pending' text."""
        decision = make_decision_snapshot(actual_reward=None)
        modal = DecisionDetailScreen(decision=decision, group_id="test")
        detail = modal._render_detail()
        detail_text = detail.plain

        assert "Reward:     pending" in detail_text

    def test_renders_factored_heads(self):
        """Detail should include factored head information."""
        decision = make_decision_snapshot()
        modal = DecisionDetailScreen(decision=decision, group_id="test")
        detail = modal._render_detail()
        detail_text = detail.plain

        assert "Factored Heads" in detail_text
        assert "Blueprint:" in detail_text
        assert "Tempo:" in detail_text
        assert "Style:" in detail_text
        assert "Curve:" in detail_text

    def test_renders_slot_states(self):
        """Detail should include slot state information."""
        decision = make_decision_snapshot()
        decision.slot_states = {"r0c0": "TRAINING", "r0c1": "DORMANT"}
        modal = DecisionDetailScreen(decision=decision, group_id="test")
        detail = modal._render_detail()
        detail_text = detail.plain

        assert "Slot State" in detail_text
        assert "r0c0: TRAINING" in detail_text
        assert "r0c1: DORMANT" in detail_text

    def test_renders_alternatives(self):
        """Detail should include alternative actions."""
        decision = make_decision_snapshot()
        decision.alternatives = [("WAIT", 0.15), ("PRUNE", 0.08)]
        modal = DecisionDetailScreen(decision=decision, group_id="test")
        detail = modal._render_detail()
        detail_text = detail.plain

        assert "Alternatives" in detail_text
        assert "WAIT" in detail_text
        assert "15%" in detail_text


class TestDecisionDetailScreenEdgeCases:
    """Test edge cases and optional fields."""

    def test_handles_none_chosen_slot(self):
        """WAIT action has no slot - should show dash."""
        decision = make_decision_snapshot(chosen_action="WAIT", chosen_slot=None)
        modal = DecisionDetailScreen(decision=decision, group_id="test")
        detail = modal._render_detail()

        # Should not raise and should show placeholder
        assert isinstance(detail, Text)

    def test_handles_empty_alternatives(self):
        """Empty alternatives should show placeholder."""
        decision = make_decision_snapshot()
        decision.alternatives = []
        modal = DecisionDetailScreen(decision=decision, group_id="test")
        detail = modal._render_detail()
        detail_text = detail.plain

        assert "(none captured)" in detail_text

    def test_handles_empty_slot_states(self):
        """Empty slot states should show placeholder."""
        decision = make_decision_snapshot()
        decision.slot_states = {}
        modal = DecisionDetailScreen(decision=decision, group_id="test")
        detail = modal._render_detail()
        detail_text = detail.plain

        assert "(no slot state captured)" in detail_text
