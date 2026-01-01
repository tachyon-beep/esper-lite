"""Tests for RunHeader widget - displays run status and training progress."""
from io import StringIO

from rich.console import Console

from esper.karn.sanctum.formatting import format_runtime
from esper.karn.sanctum.schema import (
    EnvState,
    SanctumSnapshot,
    SeedState,
)
from esper.karn.sanctum.widgets.run_header import RunHeader


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
    assert format_runtime(0) == "--"
    assert format_runtime(-5) == "--"


def test_format_runtime_seconds():
    """Test runtime formatting for seconds only."""
    assert format_runtime(45) == "45s"


def test_format_runtime_minutes():
    """Test runtime formatting for minutes and seconds."""
    assert format_runtime(90) == "1m 30s"
    assert format_runtime(180) == "3m 0s"


def test_format_runtime_hours():
    """Test runtime formatting for hours and minutes."""
    assert format_runtime(3660) == "1h 1m"
    assert format_runtime(7200) == "2h 0m"


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

    # Check batch (format is now "B:150" instead of "Batch 150")
    assert "B:150" in rendered or "150" in rendered

    # Check runtime
    assert "1h" in rendered

    # Note: Best accuracy is no longer shown in RunHeader (shown in env detail)


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

    assert "STAL" in rendered  # Abbreviated to 4 chars for fixed-width layout


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

    assert "DISC" in rendered  # Abbreviated to 4 chars for fixed-width layout


def test_run_header_env_health_summary():
    """RunHeader no longer shows env health summary (moved to other panels)."""
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

    # RunHeader now focuses on connection, episode/epoch progress, and system alarms
    # Env health is shown in EnvOverview panel instead
    assert "LIVE" in rendered or "STAL" in rendered


def test_run_header_seed_stage_counts():
    """RunHeader no longer shows seed stage counts (moved to other panels)."""
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

    # RunHeader now focuses on connection, episode/epoch progress, and system alarms
    # Seed stage counts are shown in EnvOverview panel instead
    assert "LIVE" in rendered or "STAL" in rendered


def test_run_header_task_name():
    """Test task name display."""
    snapshot = SanctumSnapshot(
        connected=True,
        staleness_seconds=1.0,
        task_name="cifar_impaired",
    )

    widget = RunHeader()
    widget.update_snapshot(snapshot)
    panel = widget.render()
    rendered = render_to_text(panel)

    assert "cifar_impaired" in rendered


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

    # Should show system indicator (now shows "✓ System" instead of "OK")
    assert "System" in rendered


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
    """RunHeader should show alarm indicator when memory alarm is active."""
    from rich.text import Text
    from esper.karn.sanctum.schema import SystemVitals

    # Create snapshot with memory alarm
    snapshot = SanctumSnapshot()
    snapshot.vitals = SystemVitals(ram_used_gb=14.5, ram_total_gb=16.0)  # >90%
    snapshot.connected = True

    header = RunHeader()
    header.update_snapshot(snapshot)

    # Render and check for alarm indicator
    rendered = header.render()
    assert isinstance(rendered, Text)
    # Should show alarm indicator (⚠ RAM 90%)
    rendered_text = rendered.plain
    assert "RAM" in rendered_text or "⚠" in rendered_text


def test_run_header_border_blue_normally():
    """RunHeader should show system OK when no memory alarm."""
    from rich.text import Text
    from esper.karn.sanctum.schema import SystemVitals

    # Create snapshot without memory alarm
    snapshot = SanctumSnapshot()
    snapshot.vitals = SystemVitals(ram_used_gb=8.0, ram_total_gb=16.0)  # 50%
    snapshot.connected = True

    header = RunHeader()
    header.update_snapshot(snapshot)

    # Render and check for system OK indicator
    rendered = header.render()
    assert isinstance(rendered, Text)
    # Should show "✓ System" when no alarms
    assert "System" in rendered.plain


# =============================================================================
# A/B Comparison Tests - REMOVED
# =============================================================================
# NOTE: A/B comparison functionality has been removed from RunHeader.
# A/B comparison is now handled at the app level with multiple policy tabs.
# These tests have been removed as the functionality no longer exists.
