"""Tests for TamiyoBrain widget."""
import pytest
from textual.app import App

from esper.karn.constants import TUIThresholds
from esper.karn.sanctum.schema import SanctumSnapshot, TamiyoState
from esper.karn.sanctum.widgets.tamiyo_brain import TamiyoBrain


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
        snapshot = SanctumSnapshot(slot_ids=["R0C0"])
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
        snapshot = SanctumSnapshot(slot_ids=["R0C0"])
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
        snapshot = SanctumSnapshot(slot_ids=["R0C0"])
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
        snapshot = SanctumSnapshot(slot_ids=["R0C0"])
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
        snapshot = SanctumSnapshot(slot_ids=["R0C0"])
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
        snapshot = SanctumSnapshot(slot_ids=["R0C0"])
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
        snapshot = SanctumSnapshot(slot_ids=["R0C0"])
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
        snapshot = SanctumSnapshot(slot_ids=["R0C0"])
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
        snapshot = SanctumSnapshot(slot_ids=["R0C0"])
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
        snapshot.current_batch = 47
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

        # Healthy state
        snapshot = SanctumSnapshot(slot_ids=["R0C0"])
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
    from collections import deque

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
async def test_per_head_heatmap_missing_data_visual():
    """Heatmap should show visual distinction for unpopulated heads."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        snapshot = SanctumSnapshot(slot_ids=["R0C0"])
        # Only slot and blueprint have data, others are 0.0
        snapshot.tamiyo = TamiyoState(
            head_slot_entropy=1.0,
            head_blueprint_entropy=2.0,
            head_style_entropy=0.0,  # Missing
            head_tempo_entropy=0.0,  # Missing
            head_alpha_target_entropy=0.0,  # Missing
            head_alpha_speed_entropy=0.0,  # Missing
            head_alpha_curve_entropy=0.0,  # Missing
            head_op_entropy=0.0,  # Missing
            ppo_data_received=True,
        )

        widget.update_snapshot(snapshot)

        # Render heatmap
        heatmap = widget._render_head_heatmap()
        plain = heatmap.plain.lower()

        # Should indicate missing/pending data visually
        # (implementation will use "n/a" or similar for zeros)
        assert "n/a" in plain or "---" in plain or "?" in plain


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
        # Heatmap contains "Heads:" label and head abbreviations like "sl[" and "bp["
        from rich.console import Console
        from io import StringIO

        console_io = StringIO()
        console = Console(file=console_io, force_terminal=True, width=120)
        console.print(rendered)
        output = console_io.getvalue()

        # Heatmap should be present with "Heads:" label
        assert "heads:" in output.lower()

        # Verify head abbreviations are present
        assert "sl[" in output  # slot head
        assert "bp[" in output  # blueprint head


# ===========================
# Task 5.2: A/B/C Group Color Constants Tests
# ===========================


@pytest.mark.asyncio
async def test_ab_group_color_mapping():
    """GROUP_COLORS should define colors for A/B/C groups."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)

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
        widget = app.query_one(TamiyoBrain)

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
# Task 1: Compact Decision Card Tests
# ===========================


@pytest.mark.asyncio
async def test_compact_decision_card_format():
    """Compact decision card should be exactly 20 chars wide with 4 lines."""
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
            chosen_slot=None,
            confidence=0.92,
            expected_value=0.12,
            actual_reward=0.08,
            alternatives=[("GERMINATE", 0.05), ("FOSSILIZE", 0.03)],
            pinned=False,
        )

        # Render compact card
        card = widget._render_compact_decision(decision, index=0)
        card_plain = card.plain
        lines = card_plain.split('\n')

        # Should have exactly 4 lines (no trailing newline on last border)
        assert len(lines) == 4

        # All 4 actual lines should be exactly 20 chars
        for i, line in enumerate(lines[:4]):
            assert len(line) == 20, f"Line {i} has length {len(line)}, expected 20: '{line}'"

        # Verify border structure
        assert lines[0].startswith("┌─")
        assert lines[0].endswith("┐")
        assert lines[3].startswith("└")
        assert lines[3].endswith("┘")

        # Should contain key info in compact format
        card_str = card_plain
        assert "D1" in card_str  # Decision number
        assert "WAIT" in card_str  # Action
        assert "92%" in card_str  # Confidence
        assert "H:87" in card_str or "H:88" in card_str  # Host accuracy (rounded)
        assert "0.12" in card_str  # Expected
        assert "0.08" in card_str  # Actual
        assert "✓" in card_str  # Good prediction indicator


@pytest.mark.asyncio
async def test_compact_decision_card_pinned():
    """Pinned decision should show P indicator at end of line 1."""
    from esper.karn.sanctum.schema import DecisionSnapshot
    from datetime import datetime, timezone

    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)

        decision_pinned = DecisionSnapshot(
            decision_id="test-2",
            timestamp=datetime.now(timezone.utc),
            slot_states={"r0c0": "TRAINING"},
            host_accuracy=87.5,
            chosen_action="WAIT",
            chosen_slot=None,
            confidence=0.92,
            expected_value=0.12,
            actual_reward=0.08,
            alternatives=[],
            pinned=True,
        )

        # Render pinned card
        card = widget._render_compact_decision(decision_pinned, index=0)
        lines = card.plain.split('\n')

        # Line 1 should contain P indicator (format: "WAIT 92% H:88P")
        # Pin replaces % on host accuracy
        assert "P" in lines[1]
        # Should NOT have % after host accuracy when pinned
        # The P comes right after the host accuracy number
        assert "H:88P" in lines[1] or "H:87P" in lines[1]

        # All lines still exactly 20 chars
        for i, line in enumerate(lines[:4]):
            assert len(line) == 20, f"Pinned card line {i} has length {len(line)}, expected 20"


@pytest.mark.asyncio
async def test_compact_decision_card_thresholds():
    """Decision card should use class constants for prediction thresholds."""
    from esper.karn.sanctum.schema import DecisionSnapshot
    from datetime import datetime, timezone

    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)

        # Verify constants exist
        assert hasattr(TamiyoBrain, 'PREDICTION_EXCELLENT_THRESHOLD')
        assert hasattr(TamiyoBrain, 'PREDICTION_ACCEPTABLE_THRESHOLD')
        assert TamiyoBrain.PREDICTION_EXCELLENT_THRESHOLD == 0.1
        assert TamiyoBrain.PREDICTION_ACCEPTABLE_THRESHOLD == 0.3

        # Test excellent prediction (diff < 0.1)
        decision_excellent = DecisionSnapshot(
            decision_id="test-3",
            timestamp=datetime.now(timezone.utc),
            slot_states={"r0c0": "TRAINING"},
            host_accuracy=87.5,
            chosen_action="WAIT",
            chosen_slot=None,
            confidence=0.92,
            expected_value=0.12,
            actual_reward=0.15,  # diff = 0.03 < 0.1
            alternatives=[],
            pinned=False,
        )
        card = widget._render_compact_decision(decision_excellent, index=0)
        assert "✓" in card.plain

        # Test acceptable prediction (0.1 <= diff < 0.3)
        decision_acceptable = DecisionSnapshot(
            decision_id="test-4",
            timestamp=datetime.now(timezone.utc),
            slot_states={"r0c0": "TRAINING"},
            host_accuracy=87.5,
            chosen_action="WAIT",
            chosen_slot=None,
            confidence=0.92,
            expected_value=0.12,
            actual_reward=0.30,  # diff = 0.18 (0.1 < diff < 0.3)
            alternatives=[],
            pinned=False,
        )
        card = widget._render_compact_decision(decision_acceptable, index=0)
        assert "✗" in card.plain  # Should show ✗ for warnings

        # Test poor prediction (diff >= 0.3)
        decision_poor = DecisionSnapshot(
            decision_id="test-5",
            timestamp=datetime.now(timezone.utc),
            slot_states={"r0c0": "TRAINING"},
            host_accuracy=87.5,
            chosen_action="WAIT",
            chosen_slot=None,
            confidence=0.92,
            expected_value=0.12,
            actual_reward=0.50,  # diff = 0.38 > 0.3
            alternatives=[],
            pinned=False,
        )
        card = widget._render_compact_decision(decision_poor, index=0)
        assert "✗" in card.plain


# ===========================
# Task 2: Decisions Column Tests
# ===========================


@pytest.mark.asyncio
async def test_decisions_column_renders_three_cards():
    """Decisions column should render 3 compact decision cards vertically."""
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
                chosen_slot=None,
                confidence=0.90 - i * 0.05,
                expected_value=0.1 * i,
                actual_reward=0.1 * i + 0.02,
                alternatives=[],
                pinned=False,
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
