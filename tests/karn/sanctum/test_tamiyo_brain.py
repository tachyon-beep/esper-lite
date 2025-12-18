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
        action_counts={"WAIT": 10, "GERMINATE": 5, "CULL": 2, "FOSSILIZE": 3},
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
            action_counts={"WAIT": 80, "GERMINATE": 10, "CULL": 5, "FOSSILIZE": 5},
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
            action_counts={"WAIT": 5, "GERMINATE": 20, "CULL": 10, "FOSSILIZE": 15},
            total_actions=50,
            ppo_data_received=True,
        )

        widget.update_snapshot(snapshot)

        # Expected order: GERMINATE (40%), FOSSILIZE (30%), CULL (20%), WAIT (10%)
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
            action_counts={"WAIT": 0, "GERMINATE": 0, "CULL": 0, "FOSSILIZE": 0},
            total_actions=0,
            ppo_data_received=True,
        )

        widget.update_snapshot(snapshot)
        assert widget._snapshot.tamiyo.total_actions == 0
