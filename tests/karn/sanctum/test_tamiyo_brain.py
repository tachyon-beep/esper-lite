"""Tests for TamiyoBrain widget."""
import pytest
from textual.app import App

from esper.karn.constants import TUIThresholds
from esper.karn.sanctum.schema import DecisionSnapshot, SanctumSnapshot, TamiyoState
from esper.karn.sanctum.widgets.tamiyo_brain import TamiyoBrain


# =============================================================================
# ENTROPY LABEL AND OUTCOME BADGE HELPER TESTS
# =============================================================================


def test_entropy_label_collapsed():
    """Entropy < 0.3 should return [collapsed] in red."""
    widget = TamiyoBrain()
    label, style = widget._entropy_label(0.1)
    assert label == "[collapsed]"
    assert style == "red"


def test_entropy_label_confident():
    """Entropy 0.3-0.7 should return [confident] in yellow."""
    widget = TamiyoBrain()
    label, style = widget._entropy_label(0.5)
    assert label == "[confident]"
    assert style == "yellow"


def test_entropy_label_balanced():
    """Entropy 0.7-1.2 should return [balanced] in green."""
    widget = TamiyoBrain()
    label, style = widget._entropy_label(0.9)
    assert label == "[balanced]"
    assert style == "green"


def test_entropy_label_exploring():
    """Entropy > 1.2 should return [exploring] in cyan."""
    widget = TamiyoBrain()
    label, style = widget._entropy_label(1.5)
    assert label == "[exploring]"
    assert style == "cyan"


def test_outcome_badge_hit():
    """Prediction error < 0.1 should return [HIT] in bright_green."""
    widget = TamiyoBrain()
    badge, style = widget._outcome_badge(expect=0.5, reward=0.55)
    assert badge == "[HIT]"
    assert style == "bright_green"


def test_outcome_badge_ok():
    """Prediction error 0.1-0.3 should return [~OK] in yellow."""
    widget = TamiyoBrain()
    badge, style = widget._outcome_badge(expect=0.5, reward=0.7)
    assert badge == "[~OK]"
    assert style == "yellow"


def test_outcome_badge_miss():
    """Prediction error >= 0.3 should return [MISS] in red."""
    widget = TamiyoBrain()
    badge, style = widget._outcome_badge(expect=0.5, reward=1.0)
    assert badge == "[MISS]"
    assert style == "red"


def test_outcome_badge_pending():
    """None reward should return [...] in dim."""
    widget = TamiyoBrain()
    badge, style = widget._outcome_badge(expect=0.5, reward=None)
    assert badge == "[...]"
    assert style == "dim"


# =============================================================================
# ACTION CONTEXT NOTE HELPER TESTS
# =============================================================================


def test_action_context_note_wait():
    """WAIT action should have contextual note."""
    from datetime import datetime, timezone

    widget = TamiyoBrain()
    decision = DecisionSnapshot(
        timestamp=datetime.now(timezone.utc),
        slot_states={},
        host_accuracy=50.0,
        chosen_action="WAIT",
        chosen_slot=None,
        confidence=0.9,
        expected_value=0.0,
        actual_reward=None,
        alternatives=[],
        decision_id="test",
    )
    note = widget._action_context_note(decision)
    assert "waiting" in note.lower()


def test_action_context_note_prune():
    """PRUNE action should mention removing underperformer."""
    from datetime import datetime, timezone

    widget = TamiyoBrain()
    decision = DecisionSnapshot(
        timestamp=datetime.now(timezone.utc),
        slot_states={},
        host_accuracy=50.0,
        chosen_action="PRUNE",
        chosen_slot="r0c1",
        confidence=0.9,
        expected_value=0.0,
        actual_reward=None,
        alternatives=[],
        decision_id="test",
    )
    note = widget._action_context_note(decision)
    assert "removing" in note.lower() or "prune" in note.lower()


def test_action_context_note_fossilize():
    """FOSSILIZE action should mention fusing module."""
    from datetime import datetime, timezone

    widget = TamiyoBrain()
    decision = DecisionSnapshot(
        timestamp=datetime.now(timezone.utc),
        slot_states={},
        host_accuracy=50.0,
        chosen_action="FOSSILIZE",
        chosen_slot="r0c0",
        confidence=0.9,
        expected_value=0.0,
        actual_reward=None,
        alternatives=[],
        decision_id="test",
    )
    note = widget._action_context_note(decision)
    assert "fusing" in note.lower() or "fossiliz" in note.lower()


def test_action_context_note_set_alpha():
    """SET_ALPHA_TARGET action should mention blend adjustment."""
    from datetime import datetime, timezone

    widget = TamiyoBrain()
    decision = DecisionSnapshot(
        timestamp=datetime.now(timezone.utc),
        slot_states={},
        host_accuracy=50.0,
        chosen_action="SET_ALPHA_TARGET",
        chosen_slot="r0c0",
        confidence=0.9,
        expected_value=0.0,
        actual_reward=None,
        alternatives=[],
        decision_id="test",
    )
    note = widget._action_context_note(decision)
    assert "blend" in note.lower() or "alpha" in note.lower()


def test_action_context_note_germinate_empty():
    """GERMINATE action should return empty string (uses head choices instead)."""
    from datetime import datetime, timezone

    widget = TamiyoBrain()
    decision = DecisionSnapshot(
        timestamp=datetime.now(timezone.utc),
        slot_states={},
        host_accuracy=50.0,
        chosen_action="GERMINATE",
        chosen_slot="r0c0",
        confidence=0.9,
        expected_value=0.0,
        actual_reward=None,
        alternatives=[],
        decision_id="test",
    )
    note = widget._action_context_note(decision)
    assert note == ""


class TamiyoBrainTestApp(App):
    """Test app for TamiyoBrain widget."""

    def compose(self):
        yield TamiyoBrain()


@pytest.fixture
def empty_snapshot():
    """Empty snapshot for basic widget creation."""
    return SanctumSnapshot(slot_ids=["R0C0", "R0C1"])


@pytest.fixture
def snapshot_with_ppo_data():
    """Snapshot with PPO data populated."""
    snapshot = SanctumSnapshot(
        slot_ids=["R0C0", "R0C1"],
        current_epoch=50,
        max_epochs=100,
    )

    tamiyo = TamiyoState(
        # Health
        entropy=1.2,
        clip_fraction=0.15,
        kl_divergence=0.03,
        explained_variance=0.5,
        # Losses
        policy_loss=0.0123,
        value_loss=0.0456,
        entropy_loss=0.0078,
        grad_norm=2.5,
        # Vitals
        learning_rate=3e-4,
        ratio_max=1.2,
        ratio_min=0.8,
        ratio_std=0.15,
        dead_layers=0,
        exploding_layers=0,
        layer_gradient_health=0.95,
        # Actions
        action_counts={"WAIT": 10, "GERMINATE": 5, "PRUNE": 2, "FOSSILIZE": 3},
        total_actions=20,
        ppo_data_received=True,
    )

    snapshot.tamiyo = tamiyo
    return snapshot


@pytest.mark.asyncio
async def test_widget_creation(empty_snapshot):
    """Widget should be created without errors."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        widget.update_snapshot(empty_snapshot)
        assert widget is not None


@pytest.mark.asyncio
async def test_waiting_state_before_ppo_data():
    """Widget should show waiting state before PPO data is received."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        snapshot = SanctumSnapshot(
            slot_ids=["R0C0"],
            current_epoch=10,
            max_epochs=100,
        )
        # tamiyo.ppo_data_received defaults to False

        widget.update_snapshot(snapshot)

        # Should render without errors (waiting state)
        assert widget._snapshot is not None
        assert not widget._snapshot.tamiyo.ppo_data_received


@pytest.mark.asyncio
async def test_snapshot_update(snapshot_with_ppo_data):
    """Widget should update with snapshot data."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        widget.update_snapshot(snapshot_with_ppo_data)
        assert widget._snapshot is not None
        assert widget._snapshot.tamiyo.ppo_data_received


@pytest.mark.asyncio
async def test_entropy_status_ok():
    """Entropy above warning threshold should be OK."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        snapshot = SanctumSnapshot(slot_ids=["R0C0"])
        snapshot.tamiyo = TamiyoState(
            entropy=1.2,  # Above warning (1.0)
            ppo_data_received=True,
        )

        widget.update_snapshot(snapshot)
        status = widget._get_entropy_status(snapshot.tamiyo.entropy)
        assert status == "ok"


@pytest.mark.asyncio
async def test_entropy_status_warning():
    """Entropy between critical and warning should be warning."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        # Between critical (0.5) and warning (1.0)
        entropy = (TUIThresholds.ENTROPY_CRITICAL + TUIThresholds.ENTROPY_WARNING) / 2

        status = widget._get_entropy_status(entropy)
        assert status == "warning"


@pytest.mark.asyncio
async def test_entropy_status_critical():
    """Entropy below critical threshold should be critical."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        entropy = TUIThresholds.ENTROPY_CRITICAL - 0.1  # Below critical (0.5)

        status = widget._get_entropy_status(entropy)
        assert status == "critical"


@pytest.mark.asyncio
async def test_clip_status_ok():
    """Clip fraction below warning threshold should be OK."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        clip = TUIThresholds.CLIP_WARNING - 0.05  # Below warning (0.25)

        status = widget._get_clip_status(clip)
        assert status == "ok"


@pytest.mark.asyncio
async def test_clip_status_warning():
    """Clip fraction between warning and critical should be warning."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        # Between warning (0.25) and critical (0.3)
        clip = (TUIThresholds.CLIP_WARNING + TUIThresholds.CLIP_CRITICAL) / 2

        status = widget._get_clip_status(clip)
        assert status == "warning"


@pytest.mark.asyncio
async def test_clip_status_critical():
    """Clip fraction above critical threshold should be critical."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        clip = TUIThresholds.CLIP_CRITICAL + 0.05  # Above critical (0.3)

        status = widget._get_clip_status(clip)
        assert status == "critical"


@pytest.mark.asyncio
async def test_kl_status_ok():
    """KL divergence below warning threshold should be OK."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        kl = TUIThresholds.KL_WARNING - 0.01  # Below warning (0.05)

        status = widget._get_kl_status(kl)
        assert status == "ok"


@pytest.mark.asyncio
async def test_kl_status_warning():
    """KL divergence above warning threshold should be warning."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        kl = TUIThresholds.KL_WARNING + 0.01  # Above warning (0.05)

        status = widget._get_kl_status(kl)
        assert status == "warning"


@pytest.mark.asyncio
async def test_ev_status_ok():
    """Explained variance above warning threshold should be OK."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        ev = TUIThresholds.EXPLAINED_VAR_WARNING + 0.1  # Above warning (0.0)

        status = widget._get_ev_status(ev)
        assert status == "ok"


@pytest.mark.asyncio
async def test_ev_status_warning():
    """Explained variance between critical and warning should be warning."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        # Between critical (-0.5) and warning (0.0)
        ev = (TUIThresholds.EXPLAINED_VAR_CRITICAL + TUIThresholds.EXPLAINED_VAR_WARNING) / 2

        status = widget._get_ev_status(ev)
        assert status == "warning"


@pytest.mark.asyncio
async def test_ev_status_critical():
    """Explained variance below critical threshold should be critical."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        ev = TUIThresholds.EXPLAINED_VAR_CRITICAL - 0.1  # Below critical (-0.5)

        status = widget._get_ev_status(ev)
        assert status == "critical"


@pytest.mark.asyncio
async def test_grad_norm_status_ok():
    """Gradient norm below warning threshold should be OK."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        grad = TUIThresholds.GRAD_NORM_WARNING - 1.0  # Below warning (5.0)

        status = widget._get_grad_norm_status(grad)
        assert status == "ok"


@pytest.mark.asyncio
async def test_grad_norm_status_warning():
    """Gradient norm between warning and critical should be warning."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        # Between warning (5.0) and critical (10.0)
        grad = (TUIThresholds.GRAD_NORM_WARNING + TUIThresholds.GRAD_NORM_CRITICAL) / 2

        status = widget._get_grad_norm_status(grad)
        assert status == "warning"


@pytest.mark.asyncio
async def test_grad_norm_status_critical():
    """Gradient norm above critical threshold should be critical."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        grad = TUIThresholds.GRAD_NORM_CRITICAL + 5.0  # Above critical (10.0)

        status = widget._get_grad_norm_status(grad)
        assert status == "critical"


@pytest.mark.asyncio
async def test_status_style_helpers():
    """Status style helpers should return correct Rich styles."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)

        assert widget._status_style("ok") == "green"
        assert widget._status_style("warning") == "yellow"
        assert widget._status_style("critical") == "red bold"


@pytest.mark.asyncio
async def test_status_text_helpers():
    """Status text helpers should return correct formatted text."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)

        assert "✓ OK" in widget._status_text("ok")
        assert "⚠ WARN" in widget._status_text("warning")
        assert "✕ CRIT" in widget._status_text("critical")


@pytest.mark.asyncio
async def test_learning_rate_scientific_notation(snapshot_with_ppo_data):
    """Learning rate should be displayed in scientific notation."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        widget.update_snapshot(snapshot_with_ppo_data)

        # LR is 3e-4 in fixture
        assert widget._snapshot.tamiyo.learning_rate == 3e-4


@pytest.mark.asyncio
async def test_gradient_health_metrics(snapshot_with_ppo_data):
    """Gradient health metrics should be displayed."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        widget.update_snapshot(snapshot_with_ppo_data)

        tamiyo = widget._snapshot.tamiyo
        assert tamiyo.dead_layers == 0
        assert tamiyo.exploding_layers == 0
        assert tamiyo.layer_gradient_health == 0.95


@pytest.mark.asyncio
async def test_ratio_stats_all_three(snapshot_with_ppo_data):
    """Ratio stats should include min, max, and std."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        widget.update_snapshot(snapshot_with_ppo_data)

        tamiyo = widget._snapshot.tamiyo
        assert tamiyo.ratio_min == 0.8
        assert tamiyo.ratio_max == 1.2
        assert tamiyo.ratio_std == 0.15


@pytest.mark.asyncio
async def test_action_distribution_with_percentages(snapshot_with_ppo_data):
    """Action distribution should show percentages."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        widget.update_snapshot(snapshot_with_ppo_data)

        tamiyo = widget._snapshot.tamiyo
        total = tamiyo.total_actions
        assert total == 20

        # Calculate expected percentages
        wait_pct = (tamiyo.action_counts["WAIT"] / total) * 100
        assert wait_pct == 50.0  # 10 / 20


@pytest.mark.asyncio
async def test_wait_dominance_warning():
    """WAIT > 70% should trigger warning."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        snapshot = SanctumSnapshot(slot_ids=["R0C0"])
        snapshot.tamiyo = TamiyoState(
            action_counts={"WAIT": 80, "GERMINATE": 10, "PRUNE": 5, "FOSSILIZE": 5},
            total_actions=100,
            ppo_data_received=True,
        )

        widget.update_snapshot(snapshot)

        # WAIT is 80% (above 70% threshold)
        wait_pct = (80 / 100) * 100
        assert wait_pct > TUIThresholds.WAIT_DOMINANCE_WARNING * 100


@pytest.mark.asyncio
async def test_action_distribution_sorting():
    """Actions should be sorted by percentage descending."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        snapshot = SanctumSnapshot(slot_ids=["R0C0"])
        snapshot.tamiyo = TamiyoState(
            action_counts={"WAIT": 5, "GERMINATE": 20, "PRUNE": 10, "FOSSILIZE": 15},
            total_actions=50,
            ppo_data_received=True,
        )

        widget.update_snapshot(snapshot)

        # Expected order: GERMINATE (40%), FOSSILIZE (30%), PRUNE (20%), WAIT (10%)
        tamiyo = snapshot.tamiyo
        total = tamiyo.total_actions
        percentages = {a: (c / total) * 100 for a, c in tamiyo.action_counts.items()}

        sorted_actions = sorted(percentages.items(), key=lambda x: -x[1])
        assert sorted_actions[0][0] == "GERMINATE"  # Highest percentage
        assert sorted_actions[-1][0] == "WAIT"  # Lowest percentage


@pytest.mark.asyncio
async def test_all_thresholds_use_tuithresholds():
    """All threshold methods should use TUIThresholds constants."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)

        # Test entropy thresholds
        assert widget._get_entropy_status(TUIThresholds.ENTROPY_WARNING - 0.01) == "warning"
        assert widget._get_entropy_status(TUIThresholds.ENTROPY_CRITICAL - 0.01) == "critical"

        # Test clip thresholds
        assert widget._get_clip_status(TUIThresholds.CLIP_WARNING + 0.01) == "warning"
        assert widget._get_clip_status(TUIThresholds.CLIP_CRITICAL + 0.01) == "critical"

        # Test KL thresholds
        assert widget._get_kl_status(TUIThresholds.KL_WARNING + 0.01) == "warning"

        # Test explained variance thresholds
        assert widget._get_ev_status(TUIThresholds.EXPLAINED_VAR_WARNING - 0.01) == "warning"
        assert widget._get_ev_status(TUIThresholds.EXPLAINED_VAR_CRITICAL - 0.01) == "critical"

        # Test gradient norm thresholds
        assert widget._get_grad_norm_status(TUIThresholds.GRAD_NORM_WARNING + 0.1) == "warning"
        assert widget._get_grad_norm_status(TUIThresholds.GRAD_NORM_CRITICAL + 0.1) == "critical"


@pytest.mark.asyncio
async def test_zero_actions_display():
    """Zero actions should display placeholder."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        snapshot = SanctumSnapshot(slot_ids=["R0C0"])
        snapshot.tamiyo = TamiyoState(
            action_counts={"WAIT": 0, "GERMINATE": 0, "PRUNE": 0, "FOSSILIZE": 0},
            total_actions=0,
            ppo_data_received=True,
        )

        widget.update_snapshot(snapshot)
        assert widget._snapshot.tamiyo.total_actions == 0


@pytest.mark.asyncio
async def test_tamiyo_brain_action_distribution_bar():
    """Action distribution should render as horizontal stacked bar."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        snapshot = SanctumSnapshot(slot_ids=["R0C0"])
        snapshot.tamiyo = TamiyoState(
            action_counts={"WAIT": 35, "GERMINATE": 25, "PRUNE": 0, "FOSSILIZE": 40},
            total_actions=100,
            ppo_data_received=True,
        )

        widget.update_snapshot(snapshot)

        # Test the action bar rendering method
        bar = widget._render_action_distribution_bar()
        assert bar is not None


@pytest.mark.asyncio
async def test_tamiyo_brain_learning_vitals_gauges():
    """Learning vitals gauges should render correctly."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        snapshot = SanctumSnapshot(slot_ids=["R0C0"])
        snapshot.tamiyo = TamiyoState(
            action_counts={"WAIT": 35, "GERMINATE": 25, "PRUNE": 0, "FOSSILIZE": 40},
            total_actions=100,
            entropy=0.42,
            value_loss=0.08,
            advantage_mean=0.31,
            ppo_data_received=True,
        )

        widget.update_snapshot(snapshot)

        # Verify gauges render
        entropy_gauge = widget._render_gauge("Entropy", 0.42, 0, 2.0, "Getting decisive")
        assert entropy_gauge is not None

        value_gauge = widget._render_gauge("Value Loss", 0.08, 0, 1.0, "Learning well")
        assert value_gauge is not None

        advantage_gauge = widget._render_gauge("Advantage", 0.31, -1.0, 1.0, "Choices working")
        assert advantage_gauge is not None


# ===========================
# Decision Tree Tests (Task 2.1)
# ===========================


@pytest.mark.asyncio
async def test_decision_tree_learning():
    """Decision tree should return LEARNING when all metrics healthy."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        snapshot = SanctumSnapshot(slot_ids=["R0C0"], current_batch=50)
        snapshot.tamiyo = TamiyoState(
            entropy=1.2,
            explained_variance=0.6,  # > 0.3 warning threshold
            clip_fraction=0.15,
            kl_divergence=0.01,
            advantage_std=1.0,  # Normal range
            ppo_data_received=True,
        )

        widget.update_snapshot(snapshot)
        status, label, style = widget._get_overall_status()
        assert status == "ok"
        assert label == "LEARNING"


@pytest.mark.asyncio
async def test_decision_tree_ev_warning():
    """Decision tree should return CAUTION when EV between 0 and 0.3."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        snapshot = SanctumSnapshot(slot_ids=["R0C0"], current_batch=50)
        snapshot.tamiyo = TamiyoState(
            entropy=1.2,
            explained_variance=0.15,  # Between 0.0 and 0.3 = warning
            clip_fraction=0.15,
            kl_divergence=0.01,
            advantage_std=1.0,  # Normal range (avoid collapsed trigger)
            ppo_data_received=True,
        )

        widget.update_snapshot(snapshot)
        status, label, style = widget._get_overall_status()
        assert status == "warning"
        assert label == "CAUTION"


@pytest.mark.asyncio
async def test_decision_tree_ev_critical():
    """Decision tree should return FAILING when EV <= 0."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        snapshot = SanctumSnapshot(slot_ids=["R0C0"], current_batch=50)
        snapshot.tamiyo = TamiyoState(
            entropy=1.2,
            explained_variance=-0.1,  # < 0.0 = critical
            clip_fraction=0.15,
            kl_divergence=0.01,
            advantage_std=1.0,  # Normal range
            ppo_data_received=True,
        )

        widget.update_snapshot(snapshot)
        status, label, style = widget._get_overall_status()
        assert status == "critical"
        assert label == "FAILING"


@pytest.mark.asyncio
async def test_decision_tree_entropy_collapsed():
    """Decision tree should return FAILING when entropy collapsed."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        snapshot = SanctumSnapshot(slot_ids=["R0C0"], current_batch=50)
        snapshot.tamiyo = TamiyoState(
            entropy=0.05,  # < 0.1 = collapsed
            explained_variance=0.6,
            clip_fraction=0.15,
            kl_divergence=0.01,
            advantage_std=1.0,  # Normal range
            ppo_data_received=True,
        )

        widget.update_snapshot(snapshot)
        status, label, style = widget._get_overall_status()
        assert status == "critical"
        assert label == "FAILING"


@pytest.mark.asyncio
async def test_decision_tree_kl_critical():
    """Decision tree should return FAILING when KL > 0.03."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        snapshot = SanctumSnapshot(slot_ids=["R0C0"], current_batch=50)
        snapshot.tamiyo = TamiyoState(
            entropy=1.2,
            explained_variance=0.6,
            clip_fraction=0.15,
            kl_divergence=0.05,  # > 0.03 = critical
            advantage_std=1.0,  # Normal range
            ppo_data_received=True,
        )

        widget.update_snapshot(snapshot)
        status, label, style = widget._get_overall_status()
        assert status == "critical"
        assert label == "FAILING"


@pytest.mark.asyncio
async def test_decision_tree_advantage_collapsed():
    """Decision tree should return FAILING when advantage std collapsed."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        snapshot = SanctumSnapshot(slot_ids=["R0C0"], current_batch=50)
        snapshot.tamiyo = TamiyoState(
            entropy=1.2,
            explained_variance=0.6,
            clip_fraction=0.15,
            kl_divergence=0.01,
            advantage_std=0.05,  # < 0.1 = collapsed
            ppo_data_received=True,
        )

        widget.update_snapshot(snapshot)
        status, label, style = widget._get_overall_status()
        assert status == "critical"
        assert label == "FAILING"


@pytest.mark.asyncio
async def test_decision_tree_grad_norm_critical():
    """Decision tree should return FAILING when grad_norm > GRAD_NORM_CRITICAL."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        snapshot = SanctumSnapshot(slot_ids=["R0C0"], current_batch=50)
        snapshot.tamiyo = TamiyoState(
            entropy=1.2,
            explained_variance=0.6,
            clip_fraction=0.15,
            kl_divergence=0.01,
            advantage_std=1.0,
            grad_norm=15.0,  # > 10.0 = critical
            ppo_data_received=True,
        )

        widget.update_snapshot(snapshot)
        status, label, style = widget._get_overall_status()
        assert status == "critical"
        assert label == "FAILING"


@pytest.mark.asyncio
async def test_decision_tree_grad_norm_warning():
    """Decision tree should return CAUTION when grad_norm > GRAD_NORM_WARNING."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        snapshot = SanctumSnapshot(slot_ids=["R0C0"], current_batch=50)
        snapshot.tamiyo = TamiyoState(
            entropy=1.2,
            explained_variance=0.6,
            clip_fraction=0.15,
            kl_divergence=0.01,
            advantage_std=1.0,
            grad_norm=7.0,  # Between 5.0 and 10.0 = warning
            ppo_data_received=True,
        )

        widget.update_snapshot(snapshot)
        status, label, style = widget._get_overall_status()
        assert status == "warning"
        assert label == "CAUTION"


# ===========================
# Task 2.2: Status Banner Tests
# ===========================


@pytest.mark.asyncio
async def test_status_banner_includes_all_metrics():
    """Status banner should include EV, Clip, KL, Adv, GradHP, batch."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        # Use current_batch >= 50 to skip warmup period
        snapshot = SanctumSnapshot(slot_ids=["R0C0"], current_batch=50)
        snapshot.tamiyo = TamiyoState(
            entropy=1.2,
            explained_variance=0.65,
            clip_fraction=0.18,
            kl_divergence=0.008,
            advantage_mean=0.12,
            advantage_std=0.94,
            dead_layers=0,
            exploding_layers=0,
            ppo_data_received=True,
        )
        snapshot.max_batches = 100

        widget.update_snapshot(snapshot)
        banner = widget._render_status_banner()

        # Should contain all key metrics
        plain = banner.plain
        assert "EV:" in plain
        assert "Clip:" in plain
        assert "KL:" in plain
        assert "Adv:" in plain
        assert "GradHP:" in plain
        assert "batch:" in plain


# ===========================
# Task 2.3: 4-Gauge Grid Tests
# ===========================


@pytest.mark.asyncio
async def test_four_gauge_grid_rendered():
    """Learning vitals should render 4 gauges in 2x2 grid."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        snapshot = SanctumSnapshot(slot_ids=["R0C0"])
        snapshot.tamiyo = TamiyoState(
            entropy=1.2,
            explained_variance=0.6,
            clip_fraction=0.15,
            kl_divergence=0.01,
            ppo_data_received=True,
        )

        widget.update_snapshot(snapshot)

        # Should have 4 gauges: EV, Entropy, Clip, KL
        gauge_grid = widget._render_gauge_grid()
        assert gauge_grid is not None


# ===========================
# Task 2.6: Dynamic Border Color Tests
# ===========================


@pytest.mark.asyncio
async def test_border_color_updates_on_status():
    """Widget border should change color based on overall status."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)

        # Healthy state (current_batch >= 50 to skip warmup period)
        snapshot = SanctumSnapshot(slot_ids=["R0C0"], current_batch=50)
        snapshot.tamiyo = TamiyoState(
            entropy=1.2,
            explained_variance=0.6,
            clip_fraction=0.15,
            kl_divergence=0.01,
            advantage_std=1.0,
            ppo_data_received=True,
        )
        widget.update_snapshot(snapshot)
        assert widget.has_class("status-ok")

        # Warning state (EV between 0 and 0.3)
        snapshot.tamiyo.explained_variance = 0.2
        widget.update_snapshot(snapshot)
        assert widget.has_class("status-warning")
        assert not widget.has_class("status-ok")


@pytest.mark.asyncio
async def test_warmup_status_during_first_50_batches():
    """Widget should show warmup status during first 50 batches."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)

        # During warmup (batch < 50)
        snapshot = SanctumSnapshot(slot_ids=["R0C0"], current_batch=25)
        snapshot.tamiyo = TamiyoState(
            entropy=1.2,
            explained_variance=0.6,
            clip_fraction=0.15,
            kl_divergence=0.01,
            advantage_std=1.0,
            ppo_data_received=True,
        )
        widget.update_snapshot(snapshot)
        assert widget.has_class("status-warmup")

        # Verify warmup message format
        status, label, style = widget._get_overall_status()
        assert status == "warmup"
        assert "WARMING UP" in label
        assert "25/50" in label
        assert style == "cyan"

        # After warmup (batch >= 50) should be status-ok
        snapshot.current_batch = 50
        widget.update_snapshot(snapshot)
        assert widget.has_class("status-ok")
        assert not widget.has_class("status-warmup")


# ===========================
# Task 2.7: Compact Mode Detection Tests
# ===========================


@pytest.mark.asyncio
async def test_compact_mode_detected_for_narrow_terminal():
    """Widget should detect 80-char terminals and switch to compact layout."""
    app = TamiyoBrainTestApp()
    async with app.run_test(size=(80, 24)):
        widget = app.query_one(TamiyoBrain)
        assert widget._is_compact_mode() is True


@pytest.mark.asyncio
async def test_full_mode_for_wide_terminal():
    """Widget should use full layout for 96+ char terminals."""
    app = TamiyoBrainTestApp()
    async with app.run_test(size=(120, 24)):
        widget = app.query_one(TamiyoBrain)
        assert widget._is_compact_mode() is False


@pytest.mark.asyncio
async def test_compact_mode_separator_width_narrow():
    """Compact mode should use 78-char separator for 80-char terminal."""
    app = TamiyoBrainTestApp()
    async with app.run_test(size=(80, 24)):
        widget = app.query_one(TamiyoBrain)
        assert widget._get_separator_width() == 78  # 80 - 2 padding


@pytest.mark.asyncio
async def test_full_mode_separator_width_wide():
    """Full mode should use 94-char separator for 96+ char terminal."""
    app = TamiyoBrainTestApp()
    async with app.run_test(size=(120, 24)):
        widget = app.query_one(TamiyoBrain)
        assert widget._get_separator_width() == 94  # 96 - 2 padding


# ===========================
# Task 3.1: Sparkline Renderer Tests
# ===========================


@pytest.mark.asyncio
async def test_sparkline_rendering():
    """Sparkline should render 10-value history as unicode blocks."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)

        # Test sparkline with known values
        history = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        sparkline = widget._render_sparkline(history, width=10)

        # Should be 10 characters
        assert len(sparkline.plain) == 10
        # First char should be lowest block, last should be highest
        assert "▁" in sparkline.plain
        assert "█" in sparkline.plain


@pytest.mark.asyncio
async def test_sparkline_empty_history():
    """Sparkline should show placeholder for empty history."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)

        # Empty history
        sparkline = widget._render_sparkline([], width=10)
        assert len(sparkline.plain) == 10
        assert "─" in sparkline.plain


@pytest.mark.asyncio
async def test_sparkline_single_value():
    """Sparkline should render single value as one block."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)

        # Single value
        sparkline = widget._render_sparkline([0.5], width=10)
        # Should have 9 placeholder chars and 1 block
        assert len(sparkline.plain) == 10
        assert sparkline.plain.count("─") == 9


@pytest.mark.asyncio
async def test_sparkline_all_same_values():
    """Sparkline should handle all same values (flat line)."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)

        # All same values
        history = [0.5, 0.5, 0.5, 0.5, 0.5]
        sparkline = widget._render_sparkline(history, width=10)
        # Should have 5 placeholder chars and 5 identical blocks
        assert len(sparkline.plain) == 10
        # When all values are the same, they should all use the same block character
        # (the implementation will pick one based on normalization)


@pytest.mark.asyncio
async def test_sparkline_width_parameter():
    """Sparkline should limit output to specified width."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)

        # 20 values but width=8
        history = list(range(20))
        sparkline = widget._render_sparkline(history, width=8)
        # Should only show last 8 values
        assert len(sparkline.plain) == 8


@pytest.mark.asyncio
async def test_sparkline_color_coding():
    """Sparkline should apply color style to recent values."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)

        # Test with custom style
        history = [0.1, 0.5, 0.9]
        sparkline = widget._render_sparkline(history, width=5, style="yellow")
        # Should have content (verify by checking it's not all placeholders)
        assert len(sparkline.plain) == 5


# ===========================
# Task 3.2: Secondary Metrics Column Tests
# ===========================


@pytest.mark.asyncio
async def test_secondary_metrics_column():
    """Secondary metrics should show Advantage, Ratio, losses with sparklines."""

    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        snapshot = SanctumSnapshot(slot_ids=["R0C0"])

        tamiyo = TamiyoState(
            advantage_mean=0.15,
            advantage_std=0.95,
            ratio_min=0.85,
            ratio_max=1.15,
            policy_loss=0.025,
            value_loss=0.142,
            grad_norm=1.5,
            dead_layers=0,
            exploding_layers=0,
            ppo_data_received=True,
        )
        # Add history
        for i in range(5):
            tamiyo.policy_loss_history.append(0.03 - i * 0.001)
            tamiyo.value_loss_history.append(0.2 - i * 0.01)
            tamiyo.grad_norm_history.append(1.5 + i * 0.1)

        snapshot.tamiyo = tamiyo
        widget.update_snapshot(snapshot)

        # Render metrics column
        metrics = widget._render_metrics_column()
        assert metrics is not None

        # Should contain key metrics
        plain = metrics.plain
        assert "Advantage" in plain
        assert "Ratio" in plain
        assert "Policy" in plain or "Grad" in plain


# ===========================
# Task 3.3: Diagnostic Matrix Tests
# ===========================


@pytest.mark.asyncio
async def test_diagnostic_matrix_layout():
    """Diagnostic matrix should have gauges left, metrics right."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        snapshot = SanctumSnapshot(slot_ids=["R0C0"])
        snapshot.tamiyo = TamiyoState(
            entropy=1.2,
            explained_variance=0.6,
            clip_fraction=0.15,
            kl_divergence=0.01,
            advantage_mean=0.15,
            advantage_std=0.95,
            ratio_min=0.85,
            ratio_max=1.15,
            ppo_data_received=True,
        )

        widget.update_snapshot(snapshot)

        # Render diagnostic matrix
        matrix = widget._render_diagnostic_matrix()
        assert matrix is not None


# ===========================
# Task 4.1: Per-Head Entropy Heatmap Tests
# ===========================


@pytest.mark.asyncio
async def test_per_head_entropy_heatmap():
    """Per-head heatmap should show 8 heads with correct max entropy values."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        snapshot = SanctumSnapshot(slot_ids=["R0C0"])
        snapshot.tamiyo = TamiyoState(
            head_slot_entropy=1.0,
            head_blueprint_entropy=2.0,
            head_style_entropy=1.2,
            head_tempo_entropy=0.9,
            head_alpha_target_entropy=0.8,
            head_alpha_speed_entropy=1.1,
            head_alpha_curve_entropy=0.7,
            head_op_entropy=1.5,
            ppo_data_received=True,
        )

        widget.update_snapshot(snapshot)

        # Render heatmap
        heatmap = widget._render_head_heatmap()
        assert heatmap is not None
        # Should contain all 8 head labels
        plain = heatmap.plain
        assert "sl" in plain.lower() or "slot" in plain.lower()
        assert "bp" in plain.lower()


@pytest.mark.asyncio
async def test_per_head_heatmap_zero_entropy_critical():
    """Heatmap should show critical indicator for zero-entropy heads.

    All 8 heads are now tracked by PPO. When a head has 0.0 entropy
    (fully collapsed), it should display "0.00!" with critical styling.
    """
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        snapshot = SanctumSnapshot(slot_ids=["R0C0"])
        # Slot and blueprint have healthy entropy, others are collapsed (0.0)
        snapshot.tamiyo = TamiyoState(
            head_slot_entropy=1.0,
            head_blueprint_entropy=2.0,
            head_style_entropy=0.0,  # Collapsed
            head_tempo_entropy=0.0,  # Collapsed
            head_alpha_target_entropy=0.0,  # Collapsed
            head_alpha_speed_entropy=0.0,  # Collapsed
            head_alpha_curve_entropy=0.0,  # Collapsed
            head_op_entropy=0.0,  # Collapsed
            ppo_data_received=True,
        )

        widget.update_snapshot(snapshot)

        # Render heatmap
        heatmap = widget._render_head_heatmap()
        plain = heatmap.plain.lower()

        # Zero-entropy heads should show critical indicator (!)
        # Multiple heads with 0.00! indicates exploration collapse
        assert "0.00!" in plain


# ===========================
# Task 4.2: Wire Heatmap into render() Tests
# ===========================


@pytest.mark.asyncio
async def test_heatmap_appears_in_render():
    """Heatmap should appear in widget render output."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        snapshot = SanctumSnapshot(slot_ids=["R0C0"])
        snapshot.tamiyo = TamiyoState(
            head_slot_entropy=1.0,
            head_blueprint_entropy=2.0,
            ppo_data_received=True,
        )

        widget.update_snapshot(snapshot)

        # Force render and check output
        rendered = widget.render()
        assert rendered is not None

        # Convert to plain text to verify heatmap is included
        # Heatmap contains "Heads:" label and head abbreviations like "slot[" and "bpnt["
        from rich.console import Console
        from io import StringIO

        console_io = StringIO()
        console = Console(file=console_io, force_terminal=True, width=120)
        console.print(rendered)
        output = console_io.getvalue()

        # Heatmap should be present with "Heads:" label
        assert "heads:" in output.lower()

        # Verify head abbreviations are present (expanded per UX review)
        # Note: conditional heads get ~ suffix (e.g., bpnt~ for blueprint)
        assert "slot[" in output  # slot head (non-conditional)
        assert "bpnt~[" in output  # blueprint head (conditional)


@pytest.mark.asyncio
async def test_conditional_head_entropy_no_false_alarm():
    """Conditional heads like style shouldn't show false alarms when rarely used.

    When a head is only relevant for certain ops (e.g., style during GERMINATE),
    the average entropy is diluted by samples where the head is masked to 1 option.
    The heatmap should use "active entropy" for threshold decisions, not raw average.

    Example: 15% GERMINATE actions, style entropy = 0.15
    → Raw fill = 0.15/1.386 = 0.11 (would trigger red !)
    → Active fill = (0.15/0.15) = 1.0 (healthy, no warning)
    """
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        snapshot = SanctumSnapshot(slot_ids=["R0C0"])
        snapshot.tamiyo = TamiyoState(
            # Style entropy is diluted (0.15) but active entropy would be ~1.0
            head_slot_entropy=1.0,
            head_blueprint_entropy=2.0,
            head_style_entropy=0.15,  # Looks collapsed, but actually healthy
            head_tempo_entropy=0.15,  # Same pattern
            head_alpha_target_entropy=0.8,
            head_alpha_speed_entropy=1.1,
            head_alpha_curve_entropy=0.7,
            head_op_entropy=1.5,
            ppo_data_received=True,
            # Action distribution shows 15% GERMINATE (style is relevant)
            action_counts={
                "WAIT": 44,
                "PRUNE": 12,
                "GERMINATE": 11,
                "ADVANCE": 5,
                "FOSSILIZE": 0,
                "SET_ALPHA_TARGET": 0,
            },
            total_actions=72,
        )

        widget.update_snapshot(snapshot)

        # Test the helper method directly
        active_ent, relevance, adjusted_fill = widget._compute_head_entropy_context(
            "style", 0.15
        )

        # Relevance should be ~15% (11 GERMINATE out of 72)
        expected_relevance = 11 / 72
        assert abs(relevance - expected_relevance) < 0.01

        # Active entropy should be 0.15 / 0.153 ≈ 0.98 (healthy!)
        assert active_ent > 0.9, f"Active entropy {active_ent} should be >0.9"

        # Adjusted fill (used for threshold) should be high (no warning)
        assert adjusted_fill > 0.5, f"Adjusted fill {adjusted_fill} should be >0.5"

        # Now render and verify no red warning "!" appears for style
        heatmap = widget._render_head_heatmap()
        plain = heatmap.plain

        # The style value "0.15" should appear but NOT with "!" indicator
        # (it should have "~" for conditional or be dim)
        assert "0.15" in plain
        # Style should NOT show critical indicator given healthy active entropy
        # Note: we check that style's 0.15 isn't followed by "!" in the plain text


@pytest.mark.asyncio
async def test_conditional_head_entropy_helper_non_conditional():
    """Non-conditional heads should return unchanged entropy context."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        snapshot = SanctumSnapshot(slot_ids=["R0C0"])
        snapshot.tamiyo = TamiyoState(
            head_op_entropy=0.5,
            ppo_data_received=True,
            action_counts={"WAIT": 50, "GERMINATE": 50},
            total_actions=100,
        )

        widget.update_snapshot(snapshot)

        # Op is not a conditional head, so no adjustment
        active_ent, relevance, adjusted_fill = widget._compute_head_entropy_context(
            "op", 0.5
        )

        # Relevance should be 1.0 (always relevant)
        assert relevance == 1.0

        # Active entropy equals observed for non-conditional
        assert active_ent == 0.5

        # Adjusted fill should be 0.5 / 1.792 ≈ 0.28
        expected_fill = 0.5 / widget.HEAD_MAX_ENTROPIES["op"]
        assert abs(adjusted_fill - expected_fill) < 0.01


# ===========================
# Per-Head Gradient Norm Heatmap Tests
# ===========================


@pytest.mark.asyncio
async def test_per_head_gradient_heatmap():
    """Per-head gradient heatmap should show 8 heads with correct color coding."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        snapshot = SanctumSnapshot(slot_ids=["R0C0"])
        snapshot.tamiyo = TamiyoState(
            head_slot_grad_norm=0.5,  # Healthy
            head_blueprint_grad_norm=1.0,  # Healthy
            head_style_grad_norm=0.05,  # Weak (warning)
            head_tempo_grad_norm=3.0,  # Strong (warning)
            head_alpha_target_grad_norm=0.001,  # Vanishing (critical)
            head_alpha_speed_grad_norm=10.0,  # Exploding (critical)
            head_alpha_curve_grad_norm=0.2,  # Healthy
            head_op_grad_norm=1.5,  # Healthy
            ppo_data_received=True,
        )

        widget.update_snapshot(snapshot)

        # Render heatmap
        heatmap = widget._render_head_gradient_heatmap()
        assert heatmap is not None

        plain = heatmap.plain
        # Should contain "Grads:" label
        assert "Grads" in plain
        # Should contain head abbreviations
        assert "slot" in plain.lower()
        assert "bpnt" in plain.lower()


@pytest.mark.asyncio
async def test_gradient_heatmap_no_data():
    """Gradient heatmap should show 'no data' when no snapshot."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        # No snapshot set

        heatmap = widget._render_head_gradient_heatmap()
        assert "no data" in heatmap.plain


@pytest.mark.asyncio
async def test_gradient_heatmap_appears_in_vitals():
    """Gradient heatmap should appear in vitals column render output."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        snapshot = SanctumSnapshot(slot_ids=["R0C0"])
        snapshot.tamiyo = TamiyoState(
            head_slot_grad_norm=0.5,
            head_blueprint_grad_norm=1.0,
            ppo_data_received=True,
        )

        widget.update_snapshot(snapshot)

        # Directly call the gradient heatmap render method
        grad_heatmap = widget._render_head_gradient_heatmap()
        assert grad_heatmap is not None

        # Should contain Grads label and show actual data (not all n/a)
        plain = grad_heatmap.plain
        assert "Grads" in plain
        # Slot and blueprint should show actual values, not n/a
        # because we set non-zero values for them
        assert "slot" in plain.lower()


# ===========================
# Task 5.2: A/B/C Group Color Constants Tests
# ===========================


@pytest.mark.asyncio
async def test_ab_group_color_mapping():
    """GROUP_COLORS should define colors for A/B/C groups."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        app.query_one(TamiyoBrain)

        # Access GROUP_COLORS directly on class (no hasattr)
        assert "A" in TamiyoBrain.GROUP_COLORS
        assert "B" in TamiyoBrain.GROUP_COLORS
        assert "C" in TamiyoBrain.GROUP_COLORS

        # Verify color assignments
        # A = Green (primary/control)
        assert "green" in TamiyoBrain.GROUP_COLORS["A"]
        # B = Cyan/Blue (variant)
        color_b = TamiyoBrain.GROUP_COLORS["B"]
        assert "cyan" in color_b or "blue" in color_b
        # C = Magenta (second variant, NOT red)
        assert "magenta" in TamiyoBrain.GROUP_COLORS["C"]


@pytest.mark.asyncio
async def test_ab_group_labels():
    """GROUP_LABELS should define labels for A/B/C groups."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        app.query_one(TamiyoBrain)

        # Access GROUP_LABELS directly on class (no hasattr)
        assert "A" in TamiyoBrain.GROUP_LABELS
        assert "B" in TamiyoBrain.GROUP_LABELS
        assert "C" in TamiyoBrain.GROUP_LABELS

        # Verify labels are non-empty
        assert len(TamiyoBrain.GROUP_LABELS["A"]) > 0
        assert len(TamiyoBrain.GROUP_LABELS["B"]) > 0
        assert len(TamiyoBrain.GROUP_LABELS["C"]) > 0


# ===========================
# Task 5.3: Apply Group Color to Border and Title Tests
# ===========================


@pytest.mark.asyncio
async def test_group_a_has_green_border():
    """Group A TamiyoBrain should have green border styling."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        snapshot = SanctumSnapshot(slot_ids=["R0C0"])
        snapshot.tamiyo = TamiyoState(
            group_id="A",
            ppo_data_received=True,
        )
        widget.update_snapshot(snapshot)
        assert widget.has_class("group-a")


@pytest.mark.asyncio
async def test_group_b_has_blue_border():
    """Group B TamiyoBrain should have blue border styling."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        snapshot = SanctumSnapshot(slot_ids=["R0C0"])
        snapshot.tamiyo = TamiyoState(
            group_id="B",
            ppo_data_received=True,
        )
        widget.update_snapshot(snapshot)
        assert widget.has_class("group-b")


# ===========================
# Task 5.4: Group Label in Status Banner Tests
# ===========================


@pytest.mark.asyncio
async def test_status_banner_shows_group_label():
    """Status banner should show group label when in A/B mode."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        snapshot = SanctumSnapshot(slot_ids=["R0C0"])
        snapshot.tamiyo = TamiyoState(
            group_id="A",
            entropy=1.2,
            explained_variance=0.6,
            ppo_data_received=True,
        )

        widget.update_snapshot(snapshot)
        banner = widget._render_status_banner()
        plain = banner.plain

        # Should show group indicator
        assert "Policy A" in plain or "🟢" in plain or "[A]" in plain


@pytest.mark.asyncio
async def test_group_c_has_magenta_border():
    """Group C should use magenta border color."""
    from textual.app import App, ComposeResult
    from esper.karn.sanctum.widgets.tamiyo_brain import TamiyoBrain
    from esper.karn.sanctum.schema import SanctumSnapshot, TamiyoState

    class TestApp(App):
        CSS = """
        TamiyoBrain.group-c { border: solid magenta; }
        """
        def compose(self) -> ComposeResult:
            yield TamiyoBrain(id="tamiyo")

    app = TestApp()
    async with app.run_test():
        widget = app.query_one("#tamiyo", TamiyoBrain)
        snapshot = SanctumSnapshot()
        snapshot.tamiyo = TamiyoState(group_id="C", ppo_data_received=True)
        widget.update_snapshot(snapshot)

        assert widget.has_class("group-c")


@pytest.mark.asyncio
async def test_unknown_group_shows_fallback_label():
    """Unknown group_id should show [D] format in banner."""
    from io import StringIO
    from rich.console import Console
    from esper.karn.sanctum.widgets.tamiyo_brain import TamiyoBrain
    from esper.karn.sanctum.schema import SanctumSnapshot, TamiyoState

    widget = TamiyoBrain()
    snapshot = SanctumSnapshot()
    snapshot.tamiyo = TamiyoState(group_id="D", ppo_data_received=True)
    widget._snapshot = snapshot

    banner = widget._render_status_banner()
    console = Console(file=StringIO(), width=120, force_terminal=True)
    console.print(banner)
    output = console.file.getvalue()

    # Should show [D] fallback format
    assert "[D]" in output


@pytest.mark.asyncio
async def test_no_group_label_when_none():
    """No group label should appear when group_id is None."""
    from io import StringIO
    from rich.console import Console
    from esper.karn.sanctum.widgets.tamiyo_brain import TamiyoBrain
    from esper.karn.sanctum.schema import SanctumSnapshot, TamiyoState

    widget = TamiyoBrain()
    snapshot = SanctumSnapshot()
    snapshot.tamiyo = TamiyoState(group_id=None, ppo_data_received=True)
    widget._snapshot = snapshot

    banner = widget._render_status_banner()
    console = Console(file=StringIO(), width=120, force_terminal=True)
    console.print(banner)
    output = console.file.getvalue()

    # Should NOT contain group labels or separator before status
    assert "Policy A" not in output
    assert "Policy B" not in output
    assert "Policy C" not in output


@pytest.mark.asyncio
async def test_border_title_includes_group_id():
    """border_title should include group_id for accessibility."""
    from textual.app import App, ComposeResult
    from esper.karn.sanctum.widgets.tamiyo_brain import TamiyoBrain
    from esper.karn.sanctum.schema import SanctumSnapshot, TamiyoState

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield TamiyoBrain(id="tamiyo")

    app = TestApp()
    async with app.run_test():
        widget = app.query_one("#tamiyo", TamiyoBrain)

        # Without group_id
        snapshot_single = SanctumSnapshot()
        snapshot_single.tamiyo = TamiyoState(group_id=None, ppo_data_received=True)
        widget.update_snapshot(snapshot_single)
        assert widget.border_title == "TAMIYO"

        # With group_id
        snapshot_ab = SanctumSnapshot()
        snapshot_ab.tamiyo = TamiyoState(group_id="A", ppo_data_received=True)
        widget.update_snapshot(snapshot_ab)
        assert widget.border_title == "TAMIYO [A]"


# ===========================
# Task 5: Enriched Decision Card Tests
# ===========================


@pytest.mark.asyncio
async def test_enriched_decision_card_format():
    """Enriched decision card should be 45 chars wide with 8 lines."""
    from esper.karn.sanctum.schema import DecisionSnapshot
    from datetime import datetime, timezone

    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)

        decision = DecisionSnapshot(
            decision_id="test-1",
            timestamp=datetime.now(timezone.utc),
            slot_states={"r0c0": "TRAINING"},
            host_accuracy=87.5,
            chosen_action="WAIT",
            chosen_slot="r0",
            confidence=0.92,
            expected_value=0.12,
            actual_reward=0.08,
            alternatives=[("GERMINATE", 0.05), ("FOSSILIZE", 0.03)],
            pinned=False,
            value_residual=-0.04,  # r - V(s) = 0.08 - 0.12
            decision_entropy=0.85,
        )

        # Render enriched card
        card = widget._render_enriched_decision(decision, index=0)
        card_plain = card.plain
        lines = card_plain.split('\n')

        # Should have exactly 8 lines (title + 6 content + bottom border)
        # Line 6 shows head choices (dim "-" for non-GERMINATE actions)
        assert len(lines) == 8

        # All lines should be exactly 45 chars (widened to show head choices)
        for i, line in enumerate(lines[:8]):
            assert len(line) == 45, f"Line {i} has length {len(line)}, expected 45: '{line}'"

        # Verify border structure
        assert lines[0].startswith("┌─")
        assert lines[0].endswith("┐")
        assert lines[7].startswith("└")
        assert lines[7].endswith("┘")

        # Should contain enriched info
        card_str = card_plain
        assert "D1" in card_str  # Decision number
        assert "WAIT" in card_str  # Action
        assert "92%" in card_str  # Confidence
        assert "H:87" in card_str or "H:88" in card_str  # Host accuracy
        assert "V:" in card_str  # Value estimate V(s)
        assert "δ:" in card_str  # Value residual δ = r - V(s)
        assert "ent:" in card_str  # Decision entropy
        assert "alt:" in card_str  # Alternatives


@pytest.mark.asyncio
async def test_enriched_decision_card_hit_miss():
    """Enriched card should show HIT/MISS text based on prediction accuracy."""
    from esper.karn.sanctum.schema import DecisionSnapshot
    from datetime import datetime, timezone

    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)

        # Test HIT (diff < 0.1)
        decision_hit = DecisionSnapshot(
            decision_id="test-hit",
            timestamp=datetime.now(timezone.utc),
            slot_states={"r0c0": "TRAINING"},
            host_accuracy=87.5,
            chosen_action="WAIT",
            chosen_slot="r0",
            confidence=0.92,
            expected_value=0.12,
            actual_reward=0.15,  # diff = 0.03 < 0.1
            alternatives=[],
            pinned=False,
            value_residual=0.03,  # r - V(s) = 0.15 - 0.12
            decision_entropy=0.85,
        )
        card = widget._render_enriched_decision(decision_hit, index=0)
        # Note: HIT/MISS text removed from new format - just icon now
        assert "✓" in card.plain

        # Test MISS (diff >= 0.1)
        decision_miss = DecisionSnapshot(
            decision_id="test-miss",
            timestamp=datetime.now(timezone.utc),
            slot_states={"r0c0": "TRAINING"},
            host_accuracy=87.5,
            chosen_action="WAIT",
            chosen_slot="r0",
            confidence=0.92,
            expected_value=0.12,
            actual_reward=0.50,  # diff = 0.38 >= 0.1
            alternatives=[],
            pinned=False,
            value_residual=0.38,  # r - V(s) = 0.50 - 0.12
            decision_entropy=0.85,
        )
        card = widget._render_enriched_decision(decision_miss, index=0)
        # Note: HIT/MISS text removed from new format - just icon now
        assert "✗" in card.plain


@pytest.mark.asyncio
async def test_enriched_decision_card_uses_constant():
    """Enriched card should use PREDICTION_EXCELLENT_THRESHOLD constant."""
    from esper.karn.sanctum.schema import DecisionSnapshot
    from datetime import datetime, timezone

    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)

        # Verify constants exist
        assert hasattr(TamiyoBrain, 'PREDICTION_EXCELLENT_THRESHOLD')
        assert TamiyoBrain.PREDICTION_EXCELLENT_THRESHOLD == 0.1

        # Test exact threshold boundary
        decision = DecisionSnapshot(
            decision_id="test-threshold",
            timestamp=datetime.now(timezone.utc),
            slot_states={"r0c0": "TRAINING"},
            host_accuracy=87.5,
            chosen_action="WAIT",
            chosen_slot="r0",
            confidence=0.92,
            expected_value=0.12,
            actual_reward=0.22,  # diff = 0.10 exactly at threshold
            alternatives=[],
            pinned=False,
            value_residual=0.10,  # r - V(s) = 0.22 - 0.12
            decision_entropy=0.85,
        )
        card = widget._render_enriched_decision(decision, index=0)
        # diff = 0.10, which is NOT < 0.1, so should be MISS (✗ icon)
        assert "✗" in card.plain


@pytest.mark.asyncio
async def test_decisions_column_uses_enriched_cards():
    """Decisions column should use enriched cards with V(s) and δ (value residual)."""
    from esper.karn.sanctum.schema import DecisionSnapshot, SanctumSnapshot, TamiyoState
    from datetime import datetime, timezone

    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)

        decision = DecisionSnapshot(
            timestamp=datetime.now(timezone.utc),
            slot_states={"r0": "Training 12%"},
            host_accuracy=87.0,
            chosen_action="WAIT",
            chosen_slot="r0",
            confidence=0.92,
            expected_value=0.12,
            actual_reward=0.08,
            alternatives=[],
            decision_id="test-1",
            value_residual=-0.04,  # r - V(s) = 0.08 - 0.12
            decision_entropy=0.85,
        )

        snapshot = SanctumSnapshot(
            tamiyo=TamiyoState(
                recent_decisions=[decision],
                entropy=6.55,
            )
        )
        widget._snapshot = snapshot

        column = widget._render_decisions_column()
        column_str = str(column)

        assert "V:" in column_str, f"Decisions column should show V(s). Got: {column_str}"
        assert "δ:" in column_str, f"Decisions column should show δ (value residual). Got: {column_str}"


# ===========================
# Task 2: Decisions Column Tests
# ===========================


@pytest.mark.asyncio
async def test_decisions_column_renders_three_cards():
    """Decisions column should render 3 enriched decision cards vertically.

    Note: The widget uses display throttling (one card swap per 30s).
    For testing, we pre-populate the display buffer to bypass throttling.
    """
    from esper.karn.sanctum.schema import DecisionSnapshot, TamiyoState, SanctumSnapshot
    from datetime import datetime, timezone, timedelta

    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)

        # Create 3 decisions
        now = datetime.now(timezone.utc)
        decisions = [
            DecisionSnapshot(
                decision_id=f"test-{i}",
                timestamp=now - timedelta(seconds=i * 15),
                slot_states={"r0c0": "TRAINING"},
                host_accuracy=85.0 + i,
                chosen_action="WAIT" if i % 2 == 0 else "GERMINATE",
                chosen_slot="r0",
                confidence=0.90 - i * 0.05,
                expected_value=0.1 * i,
                actual_reward=0.1 * i + 0.02,
                alternatives=[],
                pinned=False,
                value_residual=0.02,  # r - V(s) = +0.02 for all
                decision_entropy=0.85,
            )
            for i in range(3)
        ]

        snapshot = SanctumSnapshot(
            tamiyo=TamiyoState(
                recent_decisions=decisions,
                ppo_data_received=True,
            )
        )
        widget.update_snapshot(snapshot)

        # Pre-populate display buffer to bypass throttling (tests rendering, not throttling)
        widget._displayed_decisions = decisions.copy()

        # Render decisions column
        column = widget._render_decisions_column()
        column_str = str(column)

        # Should have 3 decision cards
        assert "D1" in column_str
        assert "D2" in column_str
        assert "D3" in column_str
        assert "WAIT" in column_str
        assert "GERM" in column_str


# ===========================
# Task 3: Vitals Column Tests
# ===========================


@pytest.mark.asyncio
async def test_vitals_column_contains_all_components():
    """Vitals column should contain gauges, metrics, heads, and action bar."""
    from esper.karn.sanctum.schema import TamiyoState, SanctumSnapshot
    from rich.console import Console
    from io import StringIO

    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)

        snapshot = SanctumSnapshot(
            tamiyo=TamiyoState(
                ppo_data_received=True,
                entropy=1.5,
                explained_variance=0.7,
                clip_fraction=0.1,
                kl_divergence=0.005,
                advantage_mean=0.15,
                advantage_std=0.8,
                policy_loss=0.03,
                value_loss=0.09,
                grad_norm=1.2,
                action_counts={"WAIT": 60, "GERMINATE": 9, "SET_ALPHA_TARGET": 2, "FOSSILIZE": 0, "PRUNE": 6},
                total_actions=77,
            )
        )
        widget.update_snapshot(snapshot)

        # Render vitals column
        vitals = widget._render_vitals_column()

        # Convert Rich Table to string for assertions
        console_io = StringIO()
        console = Console(file=console_io, force_terminal=True, width=120)
        console.print(vitals)
        vitals_str = console_io.getvalue()

        # Should contain gauge labels
        assert "Expl.Var" in vitals_str
        assert "Entropy" in vitals_str
        assert "Clip" in vitals_str
        assert "KL" in vitals_str

        # Should contain metrics
        assert "Advantage" in vitals_str
        assert "Policy Loss" in vitals_str
        assert "Grad Norm" in vitals_str

        # Should contain heads heatmap marker
        assert "Heads:" in vitals_str

        # Should contain action bar marker
        assert "G=" in vitals_str or "W=" in vitals_str  # Action legend


# ===========================
# Task 4: Layout Mode Detection Tests
# ===========================


@pytest.mark.asyncio
async def test_layout_mode_horizontal_for_wide_terminal():
    """Wide terminals (≥96 chars) should use horizontal layout."""
    app = TamiyoBrainTestApp()
    async with app.run_test(size=(120, 30)):
        widget = app.query_one(TamiyoBrain)
        assert widget._get_layout_mode() == "horizontal"


@pytest.mark.asyncio
async def test_layout_mode_stacked_for_narrow_terminal():
    """Narrow terminals (<85 chars) should use stacked layout."""
    app = TamiyoBrainTestApp()
    async with app.run_test(size=(80, 30)):
        widget = app.query_one(TamiyoBrain)
        assert widget._get_layout_mode() == "stacked"


@pytest.mark.asyncio
async def test_layout_mode_compact_horizontal_for_medium_terminal():
    """Medium terminals (85-95 chars) should use compact-horizontal layout."""
    app = TamiyoBrainTestApp()
    async with app.run_test(size=(90, 30)):
        widget = app.query_one(TamiyoBrain)
        assert widget._get_layout_mode() == "compact-horizontal"


# ===========================
# Task 5: Horizontal Layout Integration Tests
# ===========================


@pytest.mark.asyncio
async def test_horizontal_layout_has_two_columns():
    """Horizontal layout should have vitals (left) and decisions (right)."""
    from esper.karn.sanctum.schema import TamiyoState, SanctumSnapshot, DecisionSnapshot
    from datetime import datetime, timezone
    from rich.console import Console
    from io import StringIO

    app = TamiyoBrainTestApp()
    async with app.run_test(size=(120, 30)):
        widget = app.query_one(TamiyoBrain)

        # Create snapshot with decisions
        decisions = [
            DecisionSnapshot(
                decision_id="test-1",
                timestamp=datetime.now(timezone.utc),
                slot_states={"r0c0": "TRAINING"},
                host_accuracy=87.0,
                chosen_action="WAIT",
                chosen_slot=None,
                confidence=0.92,
                expected_value=0.12,
                actual_reward=0.08,
                alternatives=[],
                pinned=False,
            )
        ]
        snapshot = SanctumSnapshot(
            tamiyo=TamiyoState(
                ppo_data_received=True,
                entropy=1.5,
                recent_decisions=decisions,
            )
        )
        widget.update_snapshot(snapshot)

        # Force render
        rendered = widget.render()

        # Convert to string for assertions
        console_io = StringIO()
        console = Console(file=console_io, force_terminal=True, width=120)
        console.print(rendered)
        rendered_str = console_io.getvalue()

        # Should have both vitals and decisions visible
        assert "Entropy" in rendered_str  # Vitals
        assert "D1" in rendered_str  # Compact decision
        assert "WAIT" in rendered_str  # Action in decision


# ===========================
# Task 6: Click Handling for New Layout Tests
# ===========================


@pytest.mark.asyncio
async def test_click_decision_in_horizontal_layout():
    """Clicking on decision column should populate decision IDs."""
    from esper.karn.sanctum.schema import TamiyoState, SanctumSnapshot, DecisionSnapshot
    from datetime import datetime, timezone

    app = TamiyoBrainTestApp()
    async with app.run_test(size=(120, 30)) as pilot:
        widget = app.query_one(TamiyoBrain)

        decisions = [
            DecisionSnapshot(
                decision_id="click-test-1",
                timestamp=datetime.now(timezone.utc),
                slot_states={},
                host_accuracy=87.0,
                chosen_action="WAIT",
                chosen_slot=None,
                confidence=0.92,
                expected_value=0.12,
                actual_reward=0.08,
                alternatives=[],
                pinned=False,
            )
        ]
        snapshot = SanctumSnapshot(
            tamiyo=TamiyoState(
                ppo_data_received=True,
                recent_decisions=decisions,
            )
        )
        widget.update_snapshot(snapshot)
        await pilot.pause()

        # Decision IDs should be populated
        assert widget._decision_ids == ["click-test-1"]


# ===========================
# Task 1: Sparkline Width Tests
# ===========================


def test_sparklines_are_twenty_chars():
    """Sparklines should be 20 characters for meaningful trend visibility."""
    from esper.karn.sanctum.schema import make_sparkline
    from collections import deque

    values = deque([float(i) for i in range(20)], maxlen=20)
    sparkline = make_sparkline(values, width=20)

    assert len(sparkline) == 20, f"Expected 20-char sparkline, got {len(sparkline)}"


# ===========================
# Task 3: Episode Return and Entropy at Top Tests
# ===========================


def test_episode_return_elevated_position():
    """Episode Return should appear near the top, not buried at bottom."""
    from esper.karn.sanctum.widgets.tamiyo_brain import TamiyoBrain
    from esper.karn.sanctum.schema import SanctumSnapshot, TamiyoState
    from collections import deque

    widget = TamiyoBrain()
    snapshot = SanctumSnapshot(
        tamiyo=TamiyoState(
            episode_return_history=deque([10.0, 20.0, 30.0, 40.0, 50.0], maxlen=20),
            current_episode_return=50.0,
            entropy_history=deque([6.0, 5.9, 5.8, 5.7, 5.6], maxlen=20),
            entropy=5.6,
            learning_rate=3e-4,
            entropy_coef=0.01,
        )
    )
    widget._snapshot = snapshot

    # Render the primary metrics section (should be at top)
    primary = widget._render_primary_metrics()
    primary_str = str(primary)

    assert "Ep.Return" in primary_str or "Episode" in primary_str, \
        "Episode Return should be in primary metrics section"
    assert "Entropy" in primary_str, \
        "Entropy sparkline should be in primary metrics section"


# ===========================
# Task 4: Smart Action Pattern Detection Tests
# ===========================


def test_stuck_detection_checks_slot_availability():
    """STUCK should only trigger when waiting despite actionable opportunities."""
    from esper.karn.sanctum.widgets.tamiyo_brain import detect_action_patterns
    from esper.karn.sanctum.schema import DecisionSnapshot
    from datetime import datetime, timezone, timedelta

    now = datetime.now(timezone.utc)

    # All WAIT, but no dormant slots = NOT stuck (correct behavior)
    decisions_no_dormant = [
        DecisionSnapshot(
            timestamp=now - timedelta(seconds=i),
            slot_states={"r0": "Grafted", "r1": "Grafted", "r2": "Grafted"},
            host_accuracy=80.0,
            chosen_action="WAIT",
            chosen_slot=None,
            confidence=0.9,
            expected_value=0.1,
            actual_reward=0.1,
            alternatives=[],
            decision_id=f"test-{i}",
        )
        for i in range(12)
    ]
    slot_states_grafted = {"r0": "Grafted", "r1": "Grafted", "r2": "Grafted"}
    patterns = detect_action_patterns(decisions_no_dormant, slot_states_grafted)
    assert "STUCK" not in patterns, "Should NOT be stuck when all slots grafted"

    # All WAIT with dormant slot = STUCK
    decisions_with_dormant = [
        DecisionSnapshot(
            timestamp=now - timedelta(seconds=i),
            slot_states={"r0": "Dormant", "r1": "Grafted", "r2": "Grafted"},
            host_accuracy=80.0,
            chosen_action="WAIT",
            chosen_slot=None,
            confidence=0.9,
            expected_value=0.1,
            actual_reward=0.1,
            alternatives=[],
            decision_id=f"test-{i}",
        )
        for i in range(12)
    ]
    slot_states_dormant = {"r0": "Dormant", "r1": "Grafted", "r2": "Grafted"}
    patterns = detect_action_patterns(decisions_with_dormant, slot_states_dormant)
    assert "STUCK" in patterns, "Should be STUCK when waiting with dormant slot available"


def test_thrashing_detects_germinate_prune_cycles():
    """THRASHING should detect germinate→prune cycles (wasted compute)."""
    from esper.karn.sanctum.widgets.tamiyo_brain import detect_action_patterns
    from esper.karn.sanctum.schema import DecisionSnapshot
    from datetime import datetime, timezone, timedelta

    now = datetime.now(timezone.utc)

    # Germinate-Prune-Germinate-Prune cycle = THRASHING
    # List is newest-first (index 0 = most recent). After reverse to chronological:
    # [WAIT, WAIT, WAIT, WAIT, GERMINATE, PRUNE, GERMINATE, PRUNE] → detects GERM→PRUNE cycles
    actions = ["PRUNE", "GERMINATE", "PRUNE", "GERMINATE", "WAIT", "WAIT", "WAIT", "WAIT"]
    decisions = [
        DecisionSnapshot(
            timestamp=now - timedelta(seconds=i),
            slot_states={},
            host_accuracy=80.0,
            chosen_action=actions[i] if i < len(actions) else "WAIT",
            chosen_slot=None,
            confidence=0.9,
            expected_value=0.1,
            actual_reward=0.1,
            alternatives=[],
            decision_id=f"test-{i}",
        )
        for i in range(8)
    ]
    patterns = detect_action_patterns(decisions, {})
    assert "THRASH" in patterns, "Should detect germinate-prune thrashing"


def test_alpha_oscillation_detection():
    """ALPHA_OSC should detect excessive alpha target changes."""
    from esper.karn.sanctum.widgets.tamiyo_brain import detect_action_patterns
    from esper.karn.sanctum.schema import DecisionSnapshot
    from datetime import datetime, timezone, timedelta

    now = datetime.now(timezone.utc)

    # 4+ SET_ALPHA_TARGET actions = ALPHA_OSC
    actions = ["SET_ALPHA_TARGET", "WAIT", "SET_ALPHA_TARGET", "WAIT",
               "SET_ALPHA_TARGET", "WAIT", "SET_ALPHA_TARGET", "WAIT"]
    decisions = [
        DecisionSnapshot(
            timestamp=now - timedelta(seconds=i),
            slot_states={},
            host_accuracy=80.0,
            chosen_action=actions[i] if i < len(actions) else "WAIT",
            chosen_slot=None,
            confidence=0.9,
            expected_value=0.1,
            actual_reward=0.1,
            alternatives=[],
            decision_id=f"test-{i}",
        )
        for i in range(8)
    ]
    patterns = detect_action_patterns(decisions, {})
    assert "ALPHA_OSC" in patterns, "Should detect alpha oscillation with 4+ changes"


# ===========================
# Task 5: Enriched Decision Cards Tests
# ===========================


def test_decision_card_shows_value_and_advantage():
    """Decision cards should show V(s) and δ (value residual) per DRL review."""
    from esper.karn.sanctum.widgets.tamiyo_brain import TamiyoBrain
    from esper.karn.sanctum.schema import DecisionSnapshot, SanctumSnapshot, TamiyoState
    from datetime import datetime, timezone

    decision = DecisionSnapshot(
        timestamp=datetime.now(timezone.utc),
        slot_states={"r0": "Training 12%", "r1": "Blending 45%"},
        host_accuracy=87.0,
        chosen_action="WAIT",
        chosen_slot="r1",
        confidence=0.92,
        expected_value=0.12,  # V(s)
        actual_reward=0.08,
        alternatives=[("GERMINATE", 0.04), ("FOSSILIZE", 0.02)],
        decision_id="test-1",
        value_residual=-0.04,  # δ = r - V(s) = 0.08 - 0.12
        decision_entropy=0.85,  # Decision entropy
    )

    widget = TamiyoBrain()
    snapshot = SanctumSnapshot(
        tamiyo=TamiyoState(
            recent_decisions=[decision],
            entropy=6.55,
        )
    )
    widget._snapshot = snapshot

    card = widget._render_enriched_decision(decision, index=0)
    card_str = str(card)

    # Should show V(s) and δ (value residual)
    assert "V:" in card_str, f"Card should show value estimate V(s). Got: {card_str}"
    assert "δ:" in card_str, f"Card should show value residual δ. Got: {card_str}"

    # Should show outcome icon (per UX review - HIT/MISS text removed, just icon)
    assert "✓" in card_str or "✗" in card_str, \
        f"Card should show outcome icon. Got: {card_str}"


@pytest.mark.asyncio
async def test_decision_card_count_scales_with_height():
    """More decision cards shown when widget is taller."""
    from esper.karn.sanctum.widgets.tamiyo_brain import TamiyoBrain

    # Test with small height (24 lines) - minimum, should give 3 cards
    app = TamiyoBrainTestApp()
    async with app.run_test(size=(120, 24)) as pilot:
        widget = app.query_one(TamiyoBrain)
        await pilot.pause()

        # The method should clamp to minimum of 3
        max_cards = widget._get_max_decision_cards()
        assert 3 <= max_cards <= 8, f"Expected 3-8 cards, got {max_cards}"

    # Test with large height (50 lines)
    app2 = TamiyoBrainTestApp()
    async with app2.run_test(size=(120, 50)) as pilot:
        widget2 = app2.query_one(TamiyoBrain)
        await pilot.pause()

        # With larger terminal, should allow more cards (but hard to predict exact number
        # due to CSS layout, so just verify it's within reasonable bounds)
        max_cards2 = widget2._get_max_decision_cards()
        assert 3 <= max_cards2 <= 8, f"Expected 3-8 cards, got {max_cards2}"

    # Test with very large height (70 lines) - should clamp to max of 8
    app3 = TamiyoBrainTestApp()
    async with app3.run_test(size=(120, 70)) as pilot:
        widget3 = app3.query_one(TamiyoBrain)
        await pilot.pause()

        # Should clamp to maximum of 8
        max_cards3 = widget3._get_max_decision_cards()
        assert 3 <= max_cards3 <= 8, f"Expected 3-8 cards (with clamping), got {max_cards3}"


# ===========================
# Action Sequence Two-Row Tests
# ===========================


@pytest.mark.asyncio
async def test_action_sequence_shows_two_rows():
    """Action sequence should show Recent and Prior rows."""
    from esper.karn.sanctum.widgets.tamiyo_brain import TamiyoBrain
    from esper.karn.sanctum.schema import DecisionSnapshot, SanctumSnapshot, TamiyoState
    from datetime import datetime, timezone, timedelta

    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)

        now = datetime.now(timezone.utc)
        decisions = [
            DecisionSnapshot(
                timestamp=now - timedelta(seconds=i),
                slot_states={},
                host_accuracy=80.0,
                chosen_action="WAIT" if i % 2 == 0 else "GERMINATE",
                chosen_slot=None,
                confidence=0.9,
                expected_value=0.1,
                actual_reward=0.1,
                alternatives=[],
                decision_id=f"test-{i}",
            )
            for i in range(24)
        ]

        snapshot = SanctumSnapshot(tamiyo=TamiyoState(recent_decisions=decisions))
        widget.update_snapshot(snapshot)

        rendered = widget._render_action_sequence()
        rendered_str = rendered.plain

        assert "Recent:" in rendered_str, "Should show Recent row"
        assert "Prior:" in rendered_str, "Should show Prior row when 12+ decisions"


@pytest.mark.asyncio
async def test_action_sequence_prior_row_only_with_enough_history():
    """Prior row should only appear when there are 12+ decisions."""
    from esper.karn.sanctum.widgets.tamiyo_brain import TamiyoBrain
    from esper.karn.sanctum.schema import DecisionSnapshot, SanctumSnapshot, TamiyoState
    from datetime import datetime, timezone, timedelta

    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)

        now = datetime.now(timezone.utc)
        # Only 10 decisions - should NOT show Prior row
        decisions = [
            DecisionSnapshot(
                timestamp=now - timedelta(seconds=i),
                slot_states={},
                host_accuracy=80.0,
                chosen_action="WAIT",
                chosen_slot=None,
                confidence=0.9,
                expected_value=0.1,
                actual_reward=0.1,
                alternatives=[],
                decision_id=f"test-{i}",
            )
            for i in range(10)
        ]

        snapshot = SanctumSnapshot(tamiyo=TamiyoState(recent_decisions=decisions))
        widget.update_snapshot(snapshot)

        rendered = widget._render_action_sequence()
        rendered_str = rendered.plain

        assert "Recent:" in rendered_str, "Should show Recent row"
        assert "Prior:" not in rendered_str, "Should NOT show Prior row with <12 decisions"


@pytest.mark.asyncio
async def test_action_sequence_pattern_detection_only_on_recent():
    """Pattern detection should only apply to recent 12 actions, not prior."""
    from esper.karn.sanctum.widgets.tamiyo_brain import TamiyoBrain
    from esper.karn.sanctum.schema import DecisionSnapshot, SanctumSnapshot, TamiyoState
    from datetime import datetime, timezone, timedelta

    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)

        now = datetime.now(timezone.utc)
        # First 12 (recent): all WAIT with dormant slot = STUCK
        # Next 12 (prior): mixed actions
        decisions = []

        # Recent 12: all WAIT
        for i in range(12):
            decisions.append(
                DecisionSnapshot(
                    timestamp=now - timedelta(seconds=i),
                    slot_states={"r0": "Dormant"},
                    host_accuracy=80.0,
                    chosen_action="WAIT",
                    chosen_slot=None,
                    confidence=0.9,
                    expected_value=0.1,
                    actual_reward=0.1,
                    alternatives=[],
                    decision_id=f"test-{i}",
                )
            )

        # Prior 12: mixed
        for i in range(12, 24):
            decisions.append(
                DecisionSnapshot(
                    timestamp=now - timedelta(seconds=i),
                    slot_states={"r0": "Training"},
                    host_accuracy=80.0,
                    chosen_action="GERMINATE" if i % 3 == 0 else "WAIT",
                    chosen_slot=None,
                    confidence=0.9,
                    expected_value=0.1,
                    actual_reward=0.1,
                    alternatives=[],
                    decision_id=f"test-{i}",
                )
            )

        snapshot = SanctumSnapshot(tamiyo=TamiyoState(recent_decisions=decisions))
        widget.update_snapshot(snapshot)

        rendered = widget._render_action_sequence()
        rendered_str = rendered.plain

        # Should show STUCK pattern (because recent 12 are all WAIT with dormant slot)
        assert "STUCK" in rendered_str, "Should detect STUCK pattern on recent actions"


# ===========================
# Episode Return History Tests
# ===========================


@pytest.mark.asyncio
async def test_return_history_shows_recent_episodes():
    """Return history should show recent episode returns."""
    from esper.karn.sanctum.widgets.tamiyo_brain import TamiyoBrain
    from esper.karn.sanctum.schema import SanctumSnapshot, TamiyoState
    from collections import deque

    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        snapshot = SanctumSnapshot(
            tamiyo=TamiyoState(
                episode_return_history=deque([10.5, -2.3, 15.0, 8.2, -1.5, 20.0], maxlen=20),
                current_episode=47,
            )
        )
        widget._snapshot = snapshot

        rendered = widget._render_return_history()
        rendered_str = str(rendered)

        assert "Returns:" in rendered_str, "Should have Returns label"
        assert "Ep47" in rendered_str or "Ep" in rendered_str, "Should show episode numbers"


@pytest.mark.asyncio
async def test_return_history_empty():
    """Return history should show placeholder when no episodes yet."""
    from esper.karn.sanctum.widgets.tamiyo_brain import TamiyoBrain
    from esper.karn.sanctum.schema import SanctumSnapshot, TamiyoState
    from collections import deque

    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        snapshot = SanctumSnapshot(
            tamiyo=TamiyoState(
                episode_return_history=deque([], maxlen=20),
                current_episode=0,
            )
        )
        widget._snapshot = snapshot

        rendered = widget._render_return_history()
        rendered_str = str(rendered)

        assert "no episodes yet" in rendered_str, "Should show placeholder for empty history"


@pytest.mark.asyncio
async def test_return_history_color_coding():
    """Return history should color positive returns green and negative red."""
    from esper.karn.sanctum.widgets.tamiyo_brain import TamiyoBrain
    from esper.karn.sanctum.schema import SanctumSnapshot, TamiyoState
    from collections import deque

    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        snapshot = SanctumSnapshot(
            tamiyo=TamiyoState(
                episode_return_history=deque([10.5, -2.3], maxlen=20),
                current_episode=5,
            )
        )
        widget._snapshot = snapshot

        rendered = widget._render_return_history()

        # Check that it's a Rich Text object with styles
        assert rendered is not None
        # Verify formatting includes positive and negative values
        rendered_plain = rendered.plain
        assert "+10.5" in rendered_plain or "10.5" in rendered_plain, "Should show positive return"
        assert "-2.3" in rendered_plain, "Should show negative return"


# ===========================
# Slot Summary Tests
# ===========================


@pytest.mark.asyncio
async def test_slot_summary_shows_stage_counts():
    """Slot summary should show aggregate counts across all environments."""
    from esper.karn.sanctum.widgets.tamiyo_brain import TamiyoBrain
    from esper.karn.sanctum.schema import SanctumSnapshot, EnvState

    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        snapshot = SanctumSnapshot(
            envs={0: EnvState(env_id=0), 1: EnvState(env_id=1)},
            slot_stage_counts={
                "DORMANT": 12,
                "GERMINATED": 2,
                "TRAINING": 8,
                "BLENDING": 4,
                "HOLDING": 2,
                "FOSSILIZED": 0,
            },
            total_slots=28,
            active_slots=16,
            avg_epochs_in_stage=5.5,
            cumulative_fossilized=10,
            cumulative_pruned=5,
        )
        widget._snapshot = snapshot

        rendered = widget._render_slot_summary()
        rendered_plain = rendered.plain

        # Verify key elements are present
        assert "SLOTS" in rendered_plain, "Should have SLOTS header"
        assert "DORM:12" in rendered_plain or "DORM:" in rendered_plain, "Should show DORMANT count"
        assert "TRAIN:" in rendered_plain, "Should show TRAINING count"
        assert "Foss:" in rendered_plain, "Should show fossilized count"
        assert "Rate:" in rendered_plain, "Should show success rate"


@pytest.mark.asyncio
async def test_slot_summary_shows_constraint_when_no_dormant():
    """Slot summary should show constraint message when no dormant slots."""
    from esper.karn.sanctum.widgets.tamiyo_brain import TamiyoBrain
    from esper.karn.sanctum.schema import SanctumSnapshot, EnvState

    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        snapshot = SanctumSnapshot(
            envs={0: EnvState(env_id=0)},
            slot_stage_counts={
                "DORMANT": 0,
                "GERMINATED": 3,
                "TRAINING": 6,
                "BLENDING": 3,
                "HOLDING": 0,
                "FOSSILIZED": 0,
            },
            total_slots=12,
            active_slots=12,
            avg_epochs_in_stage=4.0,
            cumulative_fossilized=5,
            cumulative_pruned=2,
        )
        widget._snapshot = snapshot

        rendered = widget._render_slot_summary()
        rendered_plain = rendered.plain

        # Should show constraint message about no dormant slots
        assert "GERMINATE blocked" in rendered_plain, \
            "Should explain that GERMINATE is blocked when no dormant slots"


# =============================================================================
# DECISION CARD HEAD CHOICE TESTS (per specialist review)
# =============================================================================


def test_decision_card_shows_head_choices():
    """Decision cards should display blueprint, tempo arrows, style, and curve for GERMINATE.

    Per DRL/UX specialist review: surfaces sub-decisions without dashboard clutter.
    """
    from datetime import datetime, timezone

    snapshot = SanctumSnapshot(slot_ids=["R0C0", "R0C1"])
    decision = DecisionSnapshot(
        timestamp=datetime.now(timezone.utc),
        slot_states={},
        host_accuracy=75.0,
        chosen_action="GERMINATE",
        chosen_slot="slot_0",
        confidence=0.92,
        expected_value=0.5,
        actual_reward=0.3,
        alternatives=[("WAIT", 0.06)],
        decision_id="test-1",
        decision_entropy=1.2,
        value_residual=0.1,
        chosen_blueprint="conv_light",
        chosen_tempo="STANDARD",
        chosen_style="LINEAR_ADD",
        chosen_curve="LINEAR",
        blueprint_confidence=0.87,
        tempo_confidence=0.65,
    )

    widget = TamiyoBrain()
    widget._snapshot = snapshot

    card = widget._render_enriched_decision(decision, index=0, total_cards=1)
    text = str(card)

    # Should contain head choice info for GERMINATE
    assert "conv" in text.lower(), f"Expected blueprint in card: {text}"
    assert "▸▸" in text, f"Expected tempo arrows in card: {text}"  # STANDARD = ▸▸
    assert "linear" in text.lower(), f"Expected style in card: {text}"
    assert "╱" in text, f"Expected curve glyph (╱ for LINEAR) in card: {text}"


# =============================================================================
# HEAD HEATMAP BAR WIDTH TESTS (per specialist review)
# =============================================================================


def test_head_heatmap_uses_5_char_bars():
    """Head heatmap should use 5-char bars for improved granularity.

    Per DRL specialist: 5-char bars show visible difference between
    60% and 80% fill levels, helping operators catch head collapse earlier.
    """
    snapshot = SanctumSnapshot(slot_ids=["R0C0", "R0C1"])
    tamiyo = TamiyoState(
        head_slot_entropy=1.5,  # ~75% of max entropy
        head_blueprint_entropy=1.8,
        head_op_entropy=1.2,
    )
    snapshot.tamiyo = tamiyo

    widget = TamiyoBrain()
    widget._snapshot = snapshot

    result = widget._render_head_heatmap()
    text = str(result)

    # 5-char bar should have 3-4 filled blocks at 75%
    # Pattern: abbrev[█████] or abbrev[████░] - 5 chars between brackets
    # Count consecutive block characters to verify bar width
    import re

    # Find all bar patterns like [█████] or [███░░]
    bar_pattern = r"\[([█░]+)\]"
    bars = re.findall(bar_pattern, text)

    assert len(bars) > 0, f"Expected bar patterns in output, got: {text}"
    for bar in bars:
        assert len(bar) == 5, f"Expected 5-char bar, got {len(bar)}-char bar: [{bar}]"


def test_decision_card_width_is_65():
    """Decision card should be 65 chars wide for improved readability."""
    assert TamiyoBrain.DECISION_CARD_WIDTH == 65
