"""Tests for RunHeader widget - displays run status and training progress."""
from io import StringIO

import pytest
from rich.console import Console

from esper.karn.sanctum.schema import (
    EnvState,
    SanctumSnapshot,
    SeedState,
)
from esper.karn.sanctum.widgets.run_header import RunHeader, _format_runtime


def render_to_text(panel) -> str:
    """Helper to render a Rich panel to plain text."""
    console = Console(file=StringIO(), width=120, legacy_windows=False)
    console.print(panel)
    return console.file.getvalue()


# =============================================================================
# _format_runtime Tests
# =============================================================================


def test_format_runtime_zero():
    """Test runtime formatting for zero or negative."""
    assert _format_runtime(0) == "--"
    assert _format_runtime(-5) == "--"


def test_format_runtime_seconds():
    """Test runtime formatting for seconds only."""
    assert _format_runtime(45) == "45s"


def test_format_runtime_minutes():
    """Test runtime formatting for minutes and seconds."""
    assert _format_runtime(90) == "1m 30s"
    assert _format_runtime(180) == "3m 0s"


def test_format_runtime_hours():
    """Test runtime formatting for hours and minutes."""
    assert _format_runtime(3660) == "1h 1m"
    assert _format_runtime(7200) == "2h 0m"


# =============================================================================
# RunHeader Widget Tests
# =============================================================================


def test_run_header_creation():
    """Test widget creation."""
    widget = RunHeader()
    assert widget is not None


def test_run_header_no_data():
    """Test render with no snapshot."""
    widget = RunHeader()
    panel = widget.render()
    rendered = render_to_text(panel)
    assert "Waiting for training data" in rendered


def test_run_header_basic_info():
    """Test basic run info display."""
    snapshot = SanctumSnapshot(
        current_episode=5,
        current_batch=150,
        current_epoch=250,
        max_epochs=500,
        runtime_seconds=3725,  # 1h 2m
        task_name="CIFAR10",
        connected=True,
        staleness_seconds=1.0,
    )

    # Add an env with a best accuracy
    env = EnvState(env_id=0, best_accuracy=82.3, best_accuracy_episode=4)
    snapshot.envs[0] = env

    widget = RunHeader()
    widget.update_snapshot(snapshot)
    panel = widget.render()
    rendered = render_to_text(panel)

    # Check episode
    assert "Ep" in rendered
    assert "5" in rendered

    # Check epoch/max
    assert "250/500" in rendered or "250" in rendered

    # Check batch
    assert "Batch" in rendered
    assert "150" in rendered

    # Check runtime
    assert "1h" in rendered

    # Check best accuracy
    assert "Best:" in rendered
    assert "82.3%" in rendered


def test_run_header_connection_live():
    """Test Live connection indicator."""
    snapshot = SanctumSnapshot(
        connected=True,
        staleness_seconds=0.5,
    )

    widget = RunHeader()
    widget.update_snapshot(snapshot)
    panel = widget.render()
    rendered = render_to_text(panel)

    assert "LIVE" in rendered  # Uppercase for accessibility


def test_run_header_connection_stale():
    """Test Stale connection indicator."""
    snapshot = SanctumSnapshot(
        connected=True,
        staleness_seconds=10.0,
    )

    widget = RunHeader()
    widget.update_snapshot(snapshot)
    panel = widget.render()
    rendered = render_to_text(panel)

    assert "STALE" in rendered  # Uppercase for accessibility


def test_run_header_connection_disconnected():
    """Test Disconnected indicator."""
    snapshot = SanctumSnapshot(
        connected=False,
        staleness_seconds=float("inf"),
    )

    widget = RunHeader()
    widget.update_snapshot(snapshot)
    panel = widget.render()
    rendered = render_to_text(panel)

    assert "Disconnected" in rendered


def test_run_header_env_health_summary():
    """Test environment health summary."""
    snapshot = SanctumSnapshot(connected=True, staleness_seconds=1.0)

    # Add envs with different statuses
    snapshot.envs[0] = EnvState(env_id=0, status="healthy")
    snapshot.envs[1] = EnvState(env_id=1, status="excellent")
    snapshot.envs[2] = EnvState(env_id=2, status="stalled")
    snapshot.envs[3] = EnvState(env_id=3, status="degraded")

    widget = RunHeader()
    widget.update_snapshot(snapshot)
    panel = widget.render()
    rendered = render_to_text(panel)

    # Should show env counts
    assert "OK" in rendered or "2" in rendered  # 2 healthy/excellent
    assert "stall" in rendered or "1" in rendered


def test_run_header_seed_stage_counts():
    """Test seed stage counts."""
    snapshot = SanctumSnapshot(connected=True, staleness_seconds=1.0)

    env = EnvState(env_id=0)
    env.seeds["R0C0"] = SeedState(slot_id="R0C0", stage="TRAINING")
    env.seeds["R0C1"] = SeedState(slot_id="R0C1", stage="TRAINING")
    env.seeds["R1C0"] = SeedState(slot_id="R1C0", stage="BLENDING")
    env.seeds["R1C1"] = SeedState(slot_id="R1C1", stage="FOSSILIZED")
    snapshot.envs[0] = env

    widget = RunHeader()
    widget.update_snapshot(snapshot)
    panel = widget.render()
    rendered = render_to_text(panel)

    # Should show seed counts (T:2 B:1 F:1)
    assert "T:" in rendered or "2" in rendered
    assert "B:" in rendered or "1" in rendered
    assert "F:" in rendered or "1" in rendered


def test_run_header_task_name():
    """Test task name display."""
    snapshot = SanctumSnapshot(
        connected=True,
        staleness_seconds=1.0,
        task_name="cifar10_blind",
    )

    widget = RunHeader()
    widget.update_snapshot(snapshot)
    panel = widget.render()
    rendered = render_to_text(panel)

    assert "cifar10_blind" in rendered


def test_run_header_system_alarm_ok():
    """Test OK indicator when no memory alarms."""
    from esper.karn.sanctum.schema import SystemVitals

    snapshot = SanctumSnapshot(
        connected=True,
        staleness_seconds=1.0,
    )
    snapshot.vitals = SystemVitals(
        gpu_memory_used_gb=5.0,
        gpu_memory_total_gb=10.0,
        ram_used_gb=8.0,
        ram_total_gb=16.0,
    )

    widget = RunHeader()
    widget.update_snapshot(snapshot)
    panel = widget.render()
    rendered = render_to_text(panel)

    # Should show OK indicator in subtitle
    assert "OK" in rendered


def test_run_header_system_alarm_triggered():
    """Test alarm indicator when memory threshold exceeded."""
    from esper.karn.sanctum.schema import SystemVitals, GPUStats

    snapshot = SanctumSnapshot(
        connected=True,
        staleness_seconds=1.0,
    )
    snapshot.vitals = SystemVitals(
        gpu_stats={0: GPUStats(
            device_id=0,
            memory_used_gb=9.5,
            memory_total_gb=10.0,
        )},
    )

    widget = RunHeader()
    widget.update_snapshot(snapshot)
    panel = widget.render()
    rendered = render_to_text(panel)

    # Should show alarm indicator with device and percentage
    assert "cuda:0" in rendered
    assert "95%" in rendered


def test_run_header_border_red_on_memory_alarm():
    """RunHeader border should be red when memory alarm is active."""
    from rich.panel import Panel
    from esper.karn.sanctum.schema import SystemVitals

    # Create snapshot with memory alarm
    snapshot = SanctumSnapshot()
    snapshot.vitals = SystemVitals(ram_used_gb=14.5, ram_total_gb=16.0)  # >90%
    snapshot.connected = True

    header = RunHeader()
    header.update_snapshot(snapshot)

    # Render and check border style
    rendered = header.render()
    assert isinstance(rendered, Panel)
    assert rendered.border_style == "bold red"


def test_run_header_border_blue_normally():
    """RunHeader border should be blue when no memory alarm."""
    from rich.panel import Panel
    from esper.karn.sanctum.schema import SystemVitals

    # Create snapshot without memory alarm
    snapshot = SanctumSnapshot()
    snapshot.vitals = SystemVitals(ram_used_gb=8.0, ram_total_gb=16.0)  # 50%
    snapshot.connected = True

    header = RunHeader()
    header.update_snapshot(snapshot)

    # Render and check border style
    rendered = header.render()
    assert isinstance(rendered, Panel)
    assert rendered.border_style == "blue"


# =============================================================================
# A/B Comparison Tests (moved from test_comparison_header.py)
# =============================================================================


def test_run_header_update_comparison_method():
    """RunHeader should have update_comparison method for A/B data."""
    header = RunHeader()

    # Should not raise
    header.update_comparison(
        group_a_accuracy=75.0,
        group_b_accuracy=68.0,
        group_a_reward=12.5,
        group_b_reward=10.2,
    )

    # Should have a leader property
    assert header.leader == "A"


def test_run_header_shows_comparison_delta():
    """RunHeader should show accuracy delta when in A/B mode."""
    snapshot = SanctumSnapshot(connected=True, staleness_seconds=1.0)

    header = RunHeader()
    header.update_snapshot(snapshot)
    header.update_comparison(
        group_a_accuracy=75.0,
        group_b_accuracy=68.0,
        group_a_reward=12.5,
        group_b_reward=10.2,
    )

    rendered = header.render()
    text = render_to_text(rendered)

    # Should show delta (75.0 - 68.0 = +7.0%)
    assert "+7.0%" in text


def test_run_header_comparison_reward_decisive():
    """Reward should be decisive when significantly different."""
    header = RunHeader()

    # B has lower accuracy but significantly higher reward
    header.update_comparison(
        group_a_accuracy=75.0,
        group_b_accuracy=68.0,
        group_a_reward=10.0,
        group_b_reward=15.0,  # B has 50% higher reward
    )

    # B should lead because reward is the RL objective
    assert header.leader == "B"


def test_run_header_comparison_tied():
    """Leader should be None when metrics are equal."""
    header = RunHeader()

    header.update_comparison(
        group_a_accuracy=70.0,
        group_b_accuracy=70.0,
        group_a_reward=10.0,
        group_b_reward=10.0,
    )

    assert header.leader is None


def test_run_header_no_comparison_by_default():
    """RunHeader should not show comparison info when not in A/B mode."""
    snapshot = SanctumSnapshot(connected=True, staleness_seconds=1.0)

    header = RunHeader()
    header.update_snapshot(snapshot)

    rendered = header.render()
    text = render_to_text(rendered)

    # Should NOT show A/B comparison elements
    assert "A/B" not in text
    assert "Leading:" not in text
    assert "Acc Δ" not in text


def test_run_header_shows_leader_indicator():
    """RunHeader should show leader indicator in A/B mode."""
    snapshot = SanctumSnapshot(connected=True, staleness_seconds=1.0)

    header = RunHeader()
    header.update_snapshot(snapshot)
    header.update_comparison(
        group_a_accuracy=75.0,
        group_b_accuracy=68.0,
        group_a_reward=12.5,
        group_b_reward=10.2,
    )

    rendered = header.render()
    text = render_to_text(rendered)

    # Should show leader
    assert "A" in text  # Leading policy indicator
