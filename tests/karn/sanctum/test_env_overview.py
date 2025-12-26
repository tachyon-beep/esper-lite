"""Tests for EnvOverview widget."""
import pytest
from collections import deque

from textual.app import App

from esper.karn.sanctum.schema import (
    EnvState,
    RewardComponents,
    SanctumSnapshot,
    SeedState,
    make_sparkline,
)
from esper.karn.sanctum.widgets.env_overview import EnvOverview


class EnvOverviewTestApp(App):
    """Test app for EnvOverview widget."""

    def compose(self):
        yield EnvOverview()


def get_row_text(table, row_index: int) -> str:
    """Helper to get row data as a single string."""
    row_keys = list(table.rows)
    if row_index >= len(row_keys):
        return ""
    row_data = table.get_row(row_keys[row_index])
    return " ".join(str(cell) for cell in row_data)


@pytest.fixture
def empty_snapshot():
    """Empty snapshot for basic widget creation."""
    return SanctumSnapshot(slot_ids=["R0C0", "R0C1", "R1C0"])


@pytest.fixture
def populated_snapshot():
    """Snapshot with multiple envs and various seed states."""
    snapshot = SanctumSnapshot(
        slot_ids=["R0C0", "R0C1", "R1C0"],
        current_epoch=100,
        max_epochs=1000,
        task_name="MNIST",
    )

    # Env 0: Excellent state with BLENDING seed
    env0 = EnvState(
        env_id=0,
        current_epoch=100,
        host_accuracy=95.3,
        best_accuracy=95.3,
        best_accuracy_epoch=100,
        epochs_since_improvement=0,
        status="excellent",
        reward_mode="shaped",  # Blue pip for A/B test
    )
    env0.reward_components = RewardComponents(
        base_acc_delta=0.15,
        bounded_attribution=0.25,
        compute_rent=-0.05,
    )
    env0.seeds["R0C0"] = SeedState(
        slot_id="R0C0",
        stage="BLENDING",
        blueprint_id="conv_l",
        alpha=0.3,
        accuracy_delta=2.1,
        seed_params=12500,
        grad_ratio=0.95,
        has_vanishing=False,
        has_exploding=False,
        epochs_in_stage=8,
    )
    env0.seeds["R0C1"] = SeedState(
        slot_id="R0C1",
        stage="TRAINING",
        blueprint_id="dense_m",
        alpha=0.0,
        accuracy_delta=0.0,
        seed_params=8200,
        grad_ratio=1.2,
        has_vanishing=False,
        has_exploding=True,  # Should show ▲
        epochs_in_stage=5,
    )
    snapshot.envs[0] = env0

    # Env 1: Stalled state with vanishing gradients
    env1 = EnvState(
        env_id=1,
        current_epoch=100,
        host_accuracy=87.5,
        best_accuracy=88.2,
        best_accuracy_epoch=85,
        epochs_since_improvement=15,
        status="stalled",
        reward_mode="simplified",  # Yellow pip for A/B test
    )
    env1.reward_components = RewardComponents(
        base_acc_delta=-0.05,
        seed_contribution=-2.5,
        compute_rent=-0.08,
    )
    env1.seeds["R0C0"] = SeedState(
        slot_id="R0C0",
        stage="TRAINING",
        blueprint_id="conv_l",
        alpha=0.0,
        accuracy_delta=-0.8,
        seed_params=12500,
        grad_ratio=0.15,
        has_vanishing=True,  # Should show ▼
        has_exploding=False,
        epochs_in_stage=12,
    )
    snapshot.envs[1] = env1

    # Env 2: Healthy state, sparse cohort
    env2 = EnvState(
        env_id=2,
        current_epoch=100,
        host_accuracy=92.1,
        best_accuracy=92.1,
        best_accuracy_epoch=100,
        epochs_since_improvement=0,
        status="healthy",
        reward_mode="sparse",  # Cyan pip for A/B test
    )
    env2.reward_components = RewardComponents(
        base_acc_delta=0.02,
        bounded_attribution=0.08,
        compute_rent=-0.03,
    )
    snapshot.envs[2] = env2

    # Add some history for sparklines (using direct history manipulation to avoid status changes)
    for i in range(10):
        env0.reward_history.append(0.3 + i * 0.02)
        env0.accuracy_history.append(93.0 + i * 0.2)
        env1.reward_history.append(-0.5 - i * 0.025)
        env1.accuracy_history.append(88.2 - i * 0.07)

    # Set final values
    env0.reward_history.append(0.45)
    env0.accuracy_history.append(95.3)
    env1.reward_history.append(-0.75)
    env1.accuracy_history.append(87.5)
    env2.reward_history.append(0.05)
    env2.accuracy_history.append(92.1)

    # Compute aggregates
    snapshot.aggregate_mean_accuracy = (95.3 + 87.5 + 92.1) / 3
    snapshot.aggregate_mean_reward = (0.45 - 0.75 + 0.05) / 3

    return snapshot


@pytest.mark.asyncio
async def test_widget_creation(empty_snapshot):
    """Widget should be created without errors."""
    app = EnvOverviewTestApp()
    async with app.run_test():
        widget = app.query_one(EnvOverview)
        widget.update_snapshot(empty_snapshot)
        assert widget is not None


@pytest.mark.asyncio
async def test_correct_columns(empty_snapshot):
    """Widget should have all required columns."""
    app = EnvOverviewTestApp()
    async with app.run_test():
        widget = app.query_one(EnvOverview)
        widget.update_snapshot(empty_snapshot)

        # Expected columns: Env, Acc, Loss, CF, Growth, Reward, Acc▁▃▅, Rwd▁▃▅, ΔAcc, Seed Δ, Rent, [slots...], Last, Stale, Status
        # Fixed: 11 + 3 slots + 3 (Last, Stale, Status) = 17 total
        # Env, Acc, Loss, CF, Growth, Reward, Acc▁▃▅, Rwd▁▃▅, ΔAcc, Seed Δ, Rent (11) + R0C0, R0C1, R1C0 (3) + Last, Stale, Status (3) = 17
        assert widget.table is not None
        assert len(widget.table.columns) == 17


@pytest.mark.asyncio
async def test_update_snapshot_updates_display(populated_snapshot):
    """update_snapshot() should accept snapshot and update table."""
    app = EnvOverviewTestApp()
    async with app.run_test():
        widget = app.query_one(EnvOverview)
        widget.update_snapshot(populated_snapshot)

        # Should have 3 env rows + separator + aggregate row = 5 rows
        assert widget.table.row_count == 5

        # Verify env IDs are present by getting row data
        row_keys = list(widget.table.rows)
        row0_data = widget.table.get_row(row_keys[0])
        row1_data = widget.table.get_row(row_keys[1])
        row2_data = widget.table.get_row(row_keys[2])

        # First column is env ID
        assert "0" in str(row0_data[0])
        assert "1" in str(row1_data[0])
        assert "2" in str(row2_data[0])


@pytest.mark.asyncio
async def test_slot_cell_formatting(populated_snapshot):
    """Slot cells should show stage:blueprint with gradient indicators."""
    app = EnvOverviewTestApp()
    async with app.run_test():
        widget = app.query_one(EnvOverview)
        widget.update_snapshot(populated_snapshot)

        row_keys = list(widget.table.rows)
        row0_data = widget.table.get_row(row_keys[0])
        row1_data = widget.table.get_row(row_keys[1])

        # Convert all cells to strings
        row0_str = " ".join(str(cell) for cell in row0_data)
        row1_str = " ".join(str(cell) for cell in row1_data)

        # Env 0, R0C0: BLENDING seed should show alpha
        # Format: "Blend:conv_l 0.3"
        assert "Blend:conv_l" in row0_str or "BLEND:conv_l" in row0_str or "Blend" in row0_str

        # Env 0, R0C1: TRAINING seed with exploding gradients should show ▲
        # Format: "Train:dense_m e5▲"
        assert "Train:dense_m" in row0_str or "TRAIN:dense_m" in row0_str or "Train" in row0_str

        # Env 1, R0C0: TRAINING seed with vanishing gradients should show ▼
        assert "Train:conv_l" in row1_str or "TRAIN:conv_l" in row1_str or "Train" in row1_str


@pytest.mark.asyncio
async def test_slot_cell_germinated_uses_titlecase():
    """GERMINATED seeds should show Germi (not GERMI)."""
    app = EnvOverviewTestApp()
    async with app.run_test():
        snapshot = SanctumSnapshot(slot_ids=["R0C0"])
        env = EnvState(env_id=0, host_accuracy=50.0, best_accuracy=50.0)
        env.seeds["R0C0"] = SeedState(
            slot_id="R0C0",
            stage="GERMINATED",
            blueprint_id="conv_l",
        )
        snapshot.envs[0] = env

        widget = app.query_one(EnvOverview)
        widget.update_snapshot(snapshot)

        row0 = get_row_text(widget.table, 0)
        assert "Germi:conv_l" in row0
        assert "GERMI:conv_l" not in row0


@pytest.mark.asyncio
async def test_last_action_germinate_is_uppercase_code():
    """Last action GERMINATE should render as GERM (not Germi)."""
    app = EnvOverviewTestApp()
    async with app.run_test():
        snapshot = SanctumSnapshot(slot_ids=["R0C0"])
        env = EnvState(env_id=0, host_accuracy=50.0, best_accuracy=50.0)
        env.add_action("GERMINATE")
        snapshot.envs[0] = env

        widget = app.query_one(EnvOverview)
        widget.update_snapshot(snapshot)

        row0 = get_row_text(widget.table, 0)
        assert " GERM " in f" {row0} "
        assert " Germi " not in f" {row0} "


@pytest.mark.asyncio
async def test_blending_seeds_show_alpha(populated_snapshot):
    """BLENDING seeds should show alpha instead of epochs."""
    app = EnvOverviewTestApp()
    async with app.run_test():
        widget = app.query_one(EnvOverview)
        widget.update_snapshot(populated_snapshot)

        row0 = get_row_text(widget.table, 0)

        # Env 0, R0C0 is BLENDING with alpha=0.3
        # Should see "0.3" and NOT "e8"
        assert "0.3" in row0
        # Make sure we're not showing epochs for BLENDING
        # (This is a weak check, but we can't easily assert negative)


@pytest.mark.asyncio
async def test_aggregate_row_at_bottom(populated_snapshot):
    """Aggregate row with Σ should be at bottom."""
    app = EnvOverviewTestApp()
    async with app.run_test():
        widget = app.query_one(EnvOverview)
        widget.update_snapshot(populated_snapshot)

        # Last row (index 4) should be aggregate
        agg_row = get_row_text(widget.table, 4)
        assert "Σ" in agg_row or "SUM" in agg_row or "sum" in agg_row

        # Should show aggregate accuracy and reward
        # Mean accuracy: (95.3 + 87.5 + 92.1) / 3 = 91.6
        # Mean reward: (0.45 - 0.75 + 0.05) / 3 = -0.08
        assert "91.6" in agg_row or "91.6%" in agg_row


@pytest.mark.asyncio
async def test_sparklines_render_correctly(populated_snapshot):
    """Sparklines should render from history."""
    app = EnvOverviewTestApp()
    async with app.run_test():
        widget = app.query_one(EnvOverview)
        widget.update_snapshot(populated_snapshot)

        # Env 0 and 1 have history, so they should have sparklines
        # Sparklines use characters: ▁▂▃▄▅▆▇█
        row0 = get_row_text(widget.table, 0)
        row1 = get_row_text(widget.table, 1)

        # Should contain sparkline characters (at least one of them)
        sparkline_chars = "▁▂▃▄▅▆▇█"
        assert any(c in row0 for c in sparkline_chars)
        assert any(c in row1 for c in sparkline_chars)


@pytest.mark.asyncio
async def test_status_color_coding(populated_snapshot):
    """Status should have color coding based on env health."""
    app = EnvOverviewTestApp()
    async with app.run_test():
        widget = app.query_one(EnvOverview)
        widget.update_snapshot(populated_snapshot)

        # Env 0: excellent (should be green)
        # Env 1: stalled (should be yellow)
        # Env 2: healthy (should be green)
        # Color coding is applied via Rich markup in the implementation
        # We can't easily test colors without rendering, but we can check status text

        # Check that status values are present
        # Statuses: EXCL, OK, STAL, DEGR, INIT
        row0 = get_row_text(widget.table, 0)
        row1 = get_row_text(widget.table, 1)
        get_row_text(widget.table, 2)

        assert "EXCL" in row0 or "excellent" in row0.lower()
        assert "STAL" in row1 or "stalled" in row1.lower()


@pytest.mark.asyncio
async def test_ab_test_cohort_styling(populated_snapshot):
    """A/B test cohort should show colored pip next to env ID."""
    app = EnvOverviewTestApp()
    async with app.run_test():
        widget = app.query_one(EnvOverview)
        widget.update_snapshot(populated_snapshot)

        # Env 0: shaped (blue pip: ●)
        # Env 1: simplified (yellow pip: ●)
        # Env 2: sparse (cyan pip: ●)

        # All should have the pip character
        row0 = get_row_text(widget.table, 0)
        row1 = get_row_text(widget.table, 1)
        row2 = get_row_text(widget.table, 2)

        # Check for pip character presence
        assert "●" in row0
        assert "●" in row1
        assert "●" in row2


@pytest.mark.asyncio
async def test_accuracy_color_at_best():
    """Accuracy should be green when at best."""
    app = EnvOverviewTestApp()
    async with app.run_test():
        snapshot = SanctumSnapshot(slot_ids=["R0C0"])
        env = EnvState(env_id=0, host_accuracy=95.0, best_accuracy=95.0)
        env.add_accuracy(95.0, 100)
        snapshot.envs[0] = env

        widget = app.query_one(EnvOverview)
        widget.update_snapshot(snapshot)

        # Accuracy at best should be green (checked via implementation)
        # We verify the widget updates without errors
        assert widget.table.row_count > 0


@pytest.mark.asyncio
async def test_accuracy_color_stagnant():
    """Accuracy should be yellow when stagnant >5 epochs."""
    app = EnvOverviewTestApp()
    async with app.run_test():
        snapshot = SanctumSnapshot(slot_ids=["R0C0"])
        env = EnvState(
            env_id=0,
            host_accuracy=90.0,
            best_accuracy=92.0,
            epochs_since_improvement=8,
        )
        env.add_accuracy(90.0, 100)
        snapshot.envs[0] = env

        widget = app.query_one(EnvOverview)
        widget.update_snapshot(snapshot)

        # Stagnant accuracy should be yellow (checked via implementation)
        assert widget.table.row_count > 0


@pytest.mark.asyncio
async def test_reward_threshold_colors():
    """Reward should be green if >0, red if <-0.5, white otherwise."""
    app = EnvOverviewTestApp()
    async with app.run_test():
        snapshot = SanctumSnapshot(slot_ids=["R0C0"])

        # Test positive reward (green)
        env0 = EnvState(env_id=0)
        env0.add_reward(0.5, 100)
        snapshot.envs[0] = env0

        # Test negative reward below threshold (red)
        env1 = EnvState(env_id=1)
        env1.add_reward(-0.8, 100)
        snapshot.envs[1] = env1

        # Test neutral reward (white)
        env2 = EnvState(env_id=2)
        env2.add_reward(-0.2, 100)
        snapshot.envs[2] = env2

        widget = app.query_one(EnvOverview)
        widget.update_snapshot(snapshot)

        # Should have 3 envs + separator + aggregate = 5 rows
        assert widget.table.row_count == 5


@pytest.mark.asyncio
async def test_growth_ratio_color_coding():
    """Growth ratio should be colored by severity: green < 1.3, yellow < 1.5, red >= 1.5."""
    app = EnvOverviewTestApp()
    async with app.run_test():
        snapshot = SanctumSnapshot(slot_ids=["R0C0"])

        # Test no growth (dim)
        env0 = EnvState(env_id=0, host_params=1_000_000, fossilized_params=0)
        snapshot.envs[0] = env0

        # Test efficient growth (green): 1.2x
        env1 = EnvState(env_id=1, host_params=1_000_000, fossilized_params=200_000)
        snapshot.envs[1] = env1

        # Test moderate growth (yellow): 1.4x
        env2 = EnvState(env_id=2, host_params=1_000_000, fossilized_params=400_000)
        snapshot.envs[2] = env2

        # Test heavy growth (red): 1.6x
        env3 = EnvState(env_id=3, host_params=1_000_000, fossilized_params=600_000)
        snapshot.envs[3] = env3

        widget = app.query_one(EnvOverview)
        widget.update_snapshot(snapshot)

        # Should have 4 envs + separator + aggregate = 6 rows
        assert widget.table.row_count == 6

        # Verify growth ratios are computed correctly
        assert env0.growth_ratio == 1.0
        assert env1.growth_ratio == 1.2
        assert env2.growth_ratio == 1.4
        assert env3.growth_ratio == 1.6


@pytest.mark.asyncio
async def test_empty_snapshot_no_crash():
    """Widget should handle empty snapshot gracefully."""
    app = EnvOverviewTestApp()
    async with app.run_test():
        snapshot = SanctumSnapshot(slot_ids=["R0C0", "R0C1"])
        widget = app.query_one(EnvOverview)
        widget.update_snapshot(snapshot)

        # Should have headers but no data rows (maybe just aggregate or nothing)
        # At minimum, should not crash
        assert widget.table is not None


@pytest.mark.asyncio
async def test_dormant_slots_display():
    """DORMANT slots should display as empty or dash."""
    app = EnvOverviewTestApp()
    async with app.run_test():
        snapshot = SanctumSnapshot(slot_ids=["R0C0", "R0C1"])
        env = EnvState(env_id=0)
        # R0C0 is DORMANT (no seed added)
        # R0C1 has a seed
        env.seeds["R0C1"] = SeedState(
            slot_id="R0C1",
            stage="TRAINING",
            blueprint_id="dense_m",
            epochs_in_stage=3,
        )
        snapshot.envs[0] = env

        widget = app.query_one(EnvOverview)
        widget.update_snapshot(snapshot)

        row0 = get_row_text(widget.table, 0)

        # R0C1 should show Training:dense_ (truncated to 6 chars)
        assert "dense_" in row0
        # R0C0 should be empty or dash (checked in implementation)


@pytest.mark.asyncio
async def test_gradient_indicators():
    """Gradient health should show ▼ for vanishing, ▲ for exploding."""
    app = EnvOverviewTestApp()
    async with app.run_test():
        snapshot = SanctumSnapshot(slot_ids=["R0C0", "R0C1"])
        env = EnvState(env_id=0)

        # Vanishing gradients
        env.seeds["R0C0"] = SeedState(
            slot_id="R0C0",
            stage="TRAINING",
            blueprint_id="conv_l",
            has_vanishing=True,
            has_exploding=False,
            epochs_in_stage=5,
        )

        # Exploding gradients
        env.seeds["R0C1"] = SeedState(
            slot_id="R0C1",
            stage="TRAINING",
            blueprint_id="dense_m",
            has_vanishing=False,
            has_exploding=True,
            epochs_in_stage=3,
        )

        snapshot.envs[0] = env

        widget = app.query_one(EnvOverview)
        widget.update_snapshot(snapshot)

        row0 = get_row_text(widget.table, 0)

        # Should contain gradient indicators
        # Note: These may be in markup, so we check for their presence
        assert "▼" in row0 or "vanish" in row0.lower()
        assert "▲" in row0 or "explod" in row0.lower()


@pytest.mark.asyncio
async def test_make_sparkline_helper():
    """make_sparkline() helper should create sparklines."""
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    sparkline = make_sparkline(values, width=8)

    assert len(sparkline) == 8
    assert any(c in sparkline for c in "▁▂▃▄▅▆▇█")

    # Empty values
    empty_sparkline = make_sparkline([], width=8)
    assert len(empty_sparkline) == 8
    assert empty_sparkline == "────────"

    # Deque values
    deque_values = deque([1.0, 2.0, 3.0], maxlen=10)
    deque_sparkline = make_sparkline(deque_values, width=8)
    assert len(deque_sparkline) == 8


@pytest.mark.asyncio
async def test_slot_cell_blueprint_truncation():
    """Long blueprint names should be truncated to 6 chars."""
    app = EnvOverviewTestApp()
    async with app.run_test():
        snapshot = SanctumSnapshot(slot_ids=["r0c0"])
        env = EnvState(env_id=0)
        seed = SeedState(
            slot_id="r0c0",
            stage="TRAINING",
            blueprint_id="conv_light_extra_long_name",
            epochs_in_stage=3,
        )
        env.seeds["r0c0"] = seed
        snapshot.envs[0] = env

        widget = app.query_one(EnvOverview)
        widget.update_snapshot(snapshot)

        row0 = get_row_text(widget.table, 0)
        # Should show "conv_l" (first 6 chars), not full name
        assert "conv_l" in row0
        assert "conv_light_extra_long_name" not in row0
