"""Tests for Sanctum widgets - SystemResources, TrainingHealth, EventLog.

Tests cover:
- SystemResources: CPU/RAM/GPU monitoring with visual progress bars
- TrainingHealth: Entropy, gradients, action distribution, seed stage counts
- EventLog: Event log display with color-coded entries
"""
from io import StringIO

import pytest
from rich.console import Console

from esper.karn.sanctum.schema import (
    EnvState,
    GPUStats,
    SanctumSnapshot,
    SeedState,
    SystemVitals,
    TamiyoState,
)
from esper.karn.sanctum.widgets.system_resources import SystemResources
from esper.karn.sanctum.widgets.training_health import TrainingHealth


def render_to_text(renderable) -> str:
    """Helper to render a Rich renderable to plain text."""
    console = Console(file=StringIO(), width=120, legacy_windows=False)
    console.print(renderable)
    return console.file.getvalue()


# ============================================================================
# SystemResources Tests
# ============================================================================


def test_system_resources_creation():
    """Test widget creation."""
    widget = SystemResources()
    assert widget is not None


def test_system_resources_no_data():
    """Test render with no snapshot shows waiting message."""
    widget = SystemResources()
    result = widget.render()
    rendered = render_to_text(result)
    assert "Waiting for data" in rendered


def test_system_resources_cpu_bar():
    """Test CPU usage bar is displayed."""
    vitals = SystemVitals(cpu_percent=65.5)
    snapshot = SanctumSnapshot(vitals=vitals)
    widget = SystemResources()
    widget.update_snapshot(snapshot)
    result = widget.render()

    rendered = render_to_text(result)
    assert "CPU" in rendered
    assert "66%" in rendered  # Rounded to nearest integer


def test_system_resources_cpu_zero_not_shown():
    """Test CPU not shown when 0%."""
    vitals = SystemVitals(cpu_percent=0.0)
    snapshot = SanctumSnapshot(vitals=vitals)
    widget = SystemResources()
    widget.update_snapshot(snapshot)
    result = widget.render()

    rendered = render_to_text(result)
    # CPU line should not appear when 0
    assert "CPU" not in rendered


def test_system_resources_ram_bar():
    """Test RAM usage bar with GB values."""
    vitals = SystemVitals(ram_used_gb=24.5, ram_total_gb=32.0)
    snapshot = SanctumSnapshot(vitals=vitals)
    widget = SystemResources()
    widget.update_snapshot(snapshot)
    result = widget.render()

    rendered = render_to_text(result)
    assert "RAM" in rendered
    assert "24/32G" in rendered  # GB values shown
    assert "77%" in rendered  # ~76.6% rounded


def test_system_resources_ram_zero_total():
    """Test RAM not shown when total is 0."""
    vitals = SystemVitals(ram_used_gb=0.0, ram_total_gb=0.0)
    snapshot = SanctumSnapshot(vitals=vitals)
    widget = SystemResources()
    widget.update_snapshot(snapshot)
    result = widget.render()

    rendered = render_to_text(result)
    # RAM line should not appear
    assert "RAM" not in rendered


def test_system_resources_single_gpu():
    """Test single GPU display with memory bar."""
    gpu = GPUStats(device_id=0, memory_used_gb=8.5, memory_total_gb=16.0, utilization=75)
    vitals = SystemVitals(gpu_stats={"cuda:0": gpu})
    snapshot = SanctumSnapshot(vitals=vitals)
    widget = SystemResources()
    widget.update_snapshot(snapshot)
    result = widget.render()

    rendered = render_to_text(result)
    assert "GPU" in rendered
    assert "8.5/16G" in rendered  # Memory GB values
    assert "53%" in rendered  # ~53.1% memory usage


def test_system_resources_multi_gpu():
    """Test multi-GPU display with device labels."""
    gpu0 = GPUStats(device_id=0, memory_used_gb=4.5, memory_total_gb=16.0)
    gpu1 = GPUStats(device_id=1, memory_used_gb=8.0, memory_total_gb=16.0)
    vitals = SystemVitals(gpu_stats={"cuda:0": gpu0, "cuda:1": gpu1})
    snapshot = SanctumSnapshot(vitals=vitals)
    widget = SystemResources()
    widget.update_snapshot(snapshot)
    result = widget.render()

    rendered = render_to_text(result)
    # Should show GPU0 and GPU1 labels for multi-GPU
    assert "GPU0" in rendered or "GPU1" in rendered
    assert "4.5/16G" in rendered  # GPU0 memory
    assert "8.0/16G" in rendered  # GPU1 memory


def test_system_resources_gpu_utilization():
    """Test GPU utilization bar displayed when available."""
    gpu = GPUStats(device_id=0, memory_used_gb=8.0, memory_total_gb=16.0, utilization=88)
    vitals = SystemVitals(gpu_stats={"cuda:0": gpu})
    snapshot = SanctumSnapshot(vitals=vitals)
    widget = SystemResources()
    widget.update_snapshot(snapshot)
    result = widget.render()

    rendered = render_to_text(result)
    assert "util" in rendered
    assert "88%" in rendered


def test_system_resources_gpu_utilization_zero_not_shown():
    """Test GPU utilization not shown when 0."""
    gpu = GPUStats(device_id=0, memory_used_gb=8.0, memory_total_gb=16.0, utilization=0)
    vitals = SystemVitals(gpu_stats={"cuda:0": gpu})
    snapshot = SanctumSnapshot(vitals=vitals)
    widget = SystemResources()
    widget.update_snapshot(snapshot)
    result = widget.render()

    rendered = render_to_text(result)
    # Should NOT show utilization line when 0
    assert "util" not in rendered


def test_system_resources_fallback_single_gpu():
    """Test fallback to legacy single-GPU fields."""
    vitals = SystemVitals(
        gpu_memory_used_gb=6.0,
        gpu_memory_total_gb=16.0,
    )
    snapshot = SanctumSnapshot(vitals=vitals)
    widget = SystemResources()
    widget.update_snapshot(snapshot)
    result = widget.render()

    rendered = render_to_text(result)
    assert "GPU" in rendered
    assert "6.0/16G" in rendered


def test_system_resources_no_cuda():
    """Test no CUDA message when no GPU available."""
    vitals = SystemVitals(gpu_stats={}, gpu_memory_total_gb=0.0)
    snapshot = SanctumSnapshot(vitals=vitals)
    widget = SystemResources()
    widget.update_snapshot(snapshot)
    result = widget.render()

    rendered = render_to_text(result)
    assert "no CUDA" in rendered


def test_system_resources_throughput():
    """Test throughput display."""
    vitals = SystemVitals(epochs_per_second=1.5)
    snapshot = SanctumSnapshot(vitals=vitals)
    widget = SystemResources()
    widget.update_snapshot(snapshot)
    result = widget.render()

    rendered = render_to_text(result)
    assert "Throughput" in rendered
    assert "1.5" in rendered
    assert "ep/s" in rendered


def test_system_resources_throughput_zero():
    """Test throughput shows dash when 0."""
    vitals = SystemVitals(epochs_per_second=0.0)
    snapshot = SanctumSnapshot(vitals=vitals)
    widget = SystemResources()
    widget.update_snapshot(snapshot)
    result = widget.render()

    rendered = render_to_text(result)
    assert "Throughput" in rendered
    assert "--" in rendered


# ============================================================================
# TrainingHealth Tests
# ============================================================================


def test_training_health_creation():
    """Test widget creation."""
    widget = TrainingHealth()
    assert widget is not None


def test_training_health_no_data():
    """Test render with no snapshot shows waiting message."""
    widget = TrainingHealth()
    result = widget.render()
    rendered = render_to_text(result)
    assert "Waiting for data" in rendered


def test_training_health_entropy_ok():
    """Test entropy shows green when healthy."""
    tamiyo = TamiyoState(entropy=0.5, entropy_collapsed=False)
    snapshot = SanctumSnapshot(tamiyo=tamiyo)
    widget = TrainingHealth()
    widget.update_snapshot(snapshot)
    result = widget.render()

    rendered = render_to_text(result)
    assert "Entropy" in rendered
    assert "0.500" in rendered


def test_training_health_entropy_warning():
    """Test entropy shows LOW when below warning threshold."""
    tamiyo = TamiyoState(entropy=0.05, entropy_collapsed=False)
    snapshot = SanctumSnapshot(tamiyo=tamiyo)
    widget = TrainingHealth()
    widget.update_snapshot(snapshot)
    result = widget.render()

    rendered = render_to_text(result)
    assert "Entropy" in rendered
    assert "LOW" in rendered


def test_training_health_entropy_collapsed():
    """Test entropy shows COLLAPSED when flag is true."""
    tamiyo = TamiyoState(entropy=0.001, entropy_collapsed=True)
    snapshot = SanctumSnapshot(tamiyo=tamiyo)
    widget = TrainingHealth()
    widget.update_snapshot(snapshot)
    result = widget.render()

    rendered = render_to_text(result)
    assert "Entropy" in rendered
    assert "COLLAPSED" in rendered


def test_training_health_gradients_ok():
    """Test gradients shows OK when healthy."""
    tamiyo = TamiyoState(dead_layers=0, exploding_layers=0, layer_gradient_health=0.8)
    snapshot = SanctumSnapshot(tamiyo=tamiyo)
    widget = TrainingHealth()
    widget.update_snapshot(snapshot)
    result = widget.render()

    rendered = render_to_text(result)
    assert "Gradients" in rendered
    assert "OK" in rendered


def test_training_health_gradients_dead():
    """Test gradients shows dead layer count."""
    tamiyo = TamiyoState(dead_layers=3, exploding_layers=0, layer_gradient_health=0.7)
    snapshot = SanctumSnapshot(tamiyo=tamiyo)
    widget = TrainingHealth()
    widget.update_snapshot(snapshot)
    result = widget.render()

    rendered = render_to_text(result)
    assert "Gradients" in rendered
    assert "3 dead" in rendered


def test_training_health_gradients_exploding():
    """Test gradients shows exploding layer count (takes priority over dead)."""
    tamiyo = TamiyoState(dead_layers=2, exploding_layers=1, layer_gradient_health=0.3)
    snapshot = SanctumSnapshot(tamiyo=tamiyo)
    widget = TrainingHealth()
    widget.update_snapshot(snapshot)
    result = widget.render()

    rendered = render_to_text(result)
    assert "Gradients" in rendered
    assert "1 exploding" in rendered
    assert "dead" not in rendered  # Exploding takes priority


def test_training_health_gradients_unhealthy():
    """Test gradients shows UNHEALTHY when health < 0.5."""
    tamiyo = TamiyoState(dead_layers=0, exploding_layers=0, layer_gradient_health=0.3)
    snapshot = SanctumSnapshot(tamiyo=tamiyo)
    widget = TrainingHealth()
    widget.update_snapshot(snapshot)
    result = widget.render()

    rendered = render_to_text(result)
    assert "Gradients" in rendered
    assert "UNHEALTHY" in rendered


def test_training_health_action_distribution():
    """Test action distribution shows top actions."""
    tamiyo = TamiyoState(
        action_counts={"WAIT": 100, "GERMINATE": 50, "FOSSILIZE": 30, "CULL": 20},
        total_actions=200,
    )
    snapshot = SanctumSnapshot(tamiyo=tamiyo)
    widget = TrainingHealth()
    widget.update_snapshot(snapshot)
    result = widget.render()

    rendered = render_to_text(result)
    assert "Actions" in rendered
    assert "WAIT" in rendered
    assert "50%" in rendered  # WAIT is 50%
    assert "GERM" in rendered  # GERMINATE abbreviated
    assert "25%" in rendered  # GERMINATE is 25%


def test_training_health_action_distribution_no_actions():
    """Test action distribution shows dash when no actions."""
    tamiyo = TamiyoState(action_counts={}, total_actions=0)
    snapshot = SanctumSnapshot(tamiyo=tamiyo)
    widget = TrainingHealth()
    widget.update_snapshot(snapshot)
    result = widget.render()

    rendered = render_to_text(result)
    assert "Actions" in rendered
    assert "--" in rendered


def test_training_health_seed_stages():
    """Test seed stage counts across envs."""
    env1 = EnvState(env_id=1)
    env1.seeds = {
        "R0C0": SeedState(slot_id="R0C0", stage="TRAINING"),
        "R0C1": SeedState(slot_id="R0C1", stage="BLENDING"),
    }
    env2 = EnvState(env_id=2)
    env2.seeds = {
        "R0C0": SeedState(slot_id="R0C0", stage="TRAINING"),
        "R0C1": SeedState(slot_id="R0C1", stage="FOSSILIZED"),
    }

    snapshot = SanctumSnapshot(envs={1: env1, 2: env2})
    widget = TrainingHealth()
    widget.update_snapshot(snapshot)
    result = widget.render()

    rendered = render_to_text(result)
    assert "Seeds" in rendered
    assert "T:2" in rendered  # 2 TRAINING
    assert "B:1" in rendered  # 1 BLENDING
    assert "F:1" in rendered  # 1 FOSSILIZED


def test_training_health_seed_stages_empty():
    """Test seed stages shows dash when no seeds."""
    snapshot = SanctumSnapshot(envs={})
    widget = TrainingHealth()
    widget.update_snapshot(snapshot)
    result = widget.render()

    rendered = render_to_text(result)
    assert "Seeds" in rendered
    assert "--" in rendered


def test_training_health_seed_stages_culled():
    """Test seed stages shows culled count."""
    env = EnvState(env_id=1)
    env.seeds = {
        "R0C0": SeedState(slot_id="R0C0", stage="CULLED"),
        "R0C1": SeedState(slot_id="R0C1", stage="CULLED"),
    }

    snapshot = SanctumSnapshot(envs={1: env})
    widget = TrainingHealth()
    widget.update_snapshot(snapshot)
    result = widget.render()

    rendered = render_to_text(result)
    assert "Seeds" in rendered
    assert "X:2" in rendered  # 2 CULLED


def test_training_health_kl_divergence():
    """Test KL divergence shown when non-zero."""
    tamiyo = TamiyoState(kl_divergence=0.0123)
    snapshot = SanctumSnapshot(tamiyo=tamiyo)
    widget = TrainingHealth()
    widget.update_snapshot(snapshot)
    result = widget.render()

    rendered = render_to_text(result)
    assert "KL" in rendered
    assert "0.0123" in rendered


def test_training_health_kl_divergence_zero_not_shown():
    """Test KL divergence not shown when 0."""
    tamiyo = TamiyoState(kl_divergence=0.0)
    snapshot = SanctumSnapshot(tamiyo=tamiyo)
    widget = TrainingHealth()
    widget.update_snapshot(snapshot)
    result = widget.render()

    rendered = render_to_text(result)
    assert "KL" not in rendered


def test_training_health_kl_divergence_warning():
    """Test KL divergence shows warning when high."""
    tamiyo = TamiyoState(kl_divergence=0.05)  # Above warning threshold
    snapshot = SanctumSnapshot(tamiyo=tamiyo)
    widget = TrainingHealth()
    widget.update_snapshot(snapshot)
    result = widget.render()

    rendered = render_to_text(result)
    assert "KL" in rendered
    assert "0.0500" in rendered


# ============================================================================
# EventLog Tests
# ============================================================================


def test_event_log_creation():
    """Test widget creation."""
    from esper.karn.sanctum.widgets.event_log import EventLog

    widget = EventLog()
    assert widget is not None
    assert widget._max_events == 20
    assert widget.border_title == "EVENT LOG"


def test_event_log_no_events():
    """Test render with no events shows waiting message."""
    from esper.karn.sanctum.widgets.event_log import EventLog
    from rich.text import Text

    widget = EventLog()
    widget._snapshot = SanctumSnapshot(event_log=[])
    result = widget.render()

    # Should return "Waiting for events..." text
    assert isinstance(result, Text)
    assert "Waiting for events" in result.plain


def test_event_log_with_events():
    """Test render with events shows formatted log."""
    from esper.karn.sanctum.widgets.event_log import EventLog
    from esper.karn.sanctum.schema import EventLogEntry
    from rich.console import Group

    widget = EventLog(max_events=10)
    widget._snapshot = SanctumSnapshot(
        event_log=[
            EventLogEntry(
                timestamp="10:15:30",
                event_type="TRAINING_STARTED",
                env_id=None,
                message="Training started"
            ),
            EventLogEntry(
                timestamp="10:15:31",
                event_type="REWARD_COMPUTED",
                env_id=0,
                message="WAIT r=+0.500"
            ),
            EventLogEntry(
                timestamp="10:15:32",
                event_type="SEED_GERMINATED",
                env_id=1,
                message="A1 germinated (dense_m)"
            ),
        ]
    )
    result = widget.render()

    # Should return Group of Text lines
    assert isinstance(result, Group)

    # Render to text and check content
    rendered = render_to_text(result)
    assert "10:15:30" in rendered
    assert "10:15:31" in rendered
    assert "10:15:32" in rendered
    assert "WAIT r=+0.500" in rendered
    assert "A1 germinated" in rendered


def test_event_log_max_events_limit():
    """Test that only max_events are shown."""
    from esper.karn.sanctum.widgets.event_log import EventLog
    from esper.karn.sanctum.schema import EventLogEntry

    events = [
        EventLogEntry(
            timestamp=f"10:00:{i:02d}",
            event_type="EPOCH_COMPLETED",
            env_id=0,
            message=f"Event {i}"
        )
        for i in range(30)  # 30 events
    ]

    widget = EventLog(max_events=10)  # Only show 10
    widget._snapshot = SanctumSnapshot(event_log=events)
    result = widget.render()

    rendered = render_to_text(result)
    # Should only have last 10 events (20-29)
    assert "Event 20" in rendered
    assert "Event 29" in rendered
    assert "Event 19" not in rendered  # Should be cut off


def test_event_log_color_mapping():
    """Test event types get correct colors."""
    from esper.karn.sanctum.widgets.event_log import _EVENT_COLORS

    # Verify color mapping exists for expected event types
    assert "TRAINING_STARTED" in _EVENT_COLORS
    assert "EPOCH_COMPLETED" in _EVENT_COLORS
    assert "PPO_UPDATE_COMPLETED" in _EVENT_COLORS
    assert "REWARD_COMPUTED" in _EVENT_COLORS
    assert "SEED_GERMINATED" in _EVENT_COLORS
    assert "SEED_STAGE_CHANGED" in _EVENT_COLORS
    assert "SEED_FOSSILIZED" in _EVENT_COLORS
    assert "SEED_CULLED" in _EVENT_COLORS
    assert "BATCH_COMPLETED" in _EVENT_COLORS
