"""Tests for remaining Sanctum widgets - RewardComponents and EsperStatus.

Tests cover:
- RewardComponents: reward breakdown display with conditional visibility and styling
- EsperStatus: system vitals with seed counts, throughput, GPU/RAM/CPU, color thresholds

Reference: src/esper/karn/tui.py lines 1565-1748
"""
from datetime import datetime, timedelta
from io import StringIO

import pytest
from rich.console import Console

from esper.karn.sanctum.schema import (
    EnvState,
    GPUStats,
    RewardComponents as RewardComponentsSchema,
    SanctumSnapshot,
    SeedState,
    SystemVitals,
)
from esper.karn.sanctum.widgets.esper_status import EsperStatus
from esper.karn.sanctum.widgets.reward_components import RewardComponents


def render_to_text(panel) -> str:
    """Helper to render a Rich panel to plain text."""
    console = Console(file=StringIO(), width=120, legacy_windows=False)
    console.print(panel)
    return console.file.getvalue()


# ============================================================================
# RewardComponents Tests
# ============================================================================


def test_reward_components_creation():
    """Test widget creation."""
    widget = RewardComponents()
    assert widget is not None


def test_reward_components_no_data():
    """Test render with no snapshot."""
    widget = RewardComponents()
    panel = widget.render()
    assert "No data" in panel.renderable


def test_reward_components_no_env():
    """Test render with no focused env."""
    snapshot = SanctumSnapshot(focused_env_id=999)  # Non-existent env
    widget = RewardComponents()
    widget.update_snapshot(snapshot)
    panel = widget.render()
    assert "No env selected" in panel.renderable


def test_reward_components_header_display():
    """Test header shows env_id, last_action, val_acc."""
    env = EnvState(env_id=3)
    env.action_history.append("GERMINATE_CONV_LIGHT")

    snapshot = SanctumSnapshot(
        focused_env_id=3,
        envs={3: env},
        rewards=RewardComponentsSchema(val_acc=85.5),
    )

    widget = RewardComponents()
    widget.update_snapshot(snapshot)
    panel = widget.render()

    rendered = render_to_text(panel)
    assert "GERMINATE_CONV_LIGHT" in rendered
    assert "85.5%" in rendered


def test_reward_components_base_delta_positive():
    """Test base_acc_delta displays when positive."""
    env = EnvState(env_id=1)
    env.reward_history.append(2.5)

    snapshot = SanctumSnapshot(
        focused_env_id=1,
        envs={1: env},
        rewards=RewardComponentsSchema(base_acc_delta=2.5),
    )
    widget = RewardComponents()
    widget.update_snapshot(snapshot)
    panel = widget.render()

    rendered = render_to_text(panel)
    assert "+2.50" in rendered


def test_reward_components_base_delta_negative():
    """Test base_acc_delta displays when negative."""
    env = EnvState(env_id=1)
    env.reward_history.append(-1.5)

    snapshot = SanctumSnapshot(
        focused_env_id=1,
        envs={1: env},
        rewards=RewardComponentsSchema(base_acc_delta=-1.5),
    )
    widget = RewardComponents()
    widget.update_snapshot(snapshot)
    panel = widget.render()

    rendered = render_to_text(panel)
    assert "-1.50" in rendered


def test_reward_components_attribution_conditional():
    """Test attribution only shows if non-zero."""
    # Zero attribution - should NOT show
    env1 = EnvState(env_id=1)
    env1.reward_history.append(0.0)

    snapshot1 = SanctumSnapshot(
        focused_env_id=1,
        envs={1: env1},
        rewards=RewardComponentsSchema(bounded_attribution=0.0),
    )
    widget = RewardComponents()
    widget.update_snapshot(snapshot1)
    panel1 = widget.render()
    rendered1 = render_to_text(panel1)
    assert "Attr:" not in rendered1

    # Non-zero attribution - SHOULD show
    env2 = EnvState(env_id=2)
    env2.reward_history.append(1.2)

    snapshot2 = SanctumSnapshot(
        focused_env_id=2,
        envs={2: env2},
        rewards=RewardComponentsSchema(bounded_attribution=1.2),
    )
    widget.update_snapshot(snapshot2)
    panel2 = widget.render()
    rendered2 = render_to_text(panel2)
    assert "Attr:" in rendered2
    assert "+1.20" in rendered2


def test_reward_components_attribution_negative():
    """Test attribution displays when negative."""
    env = EnvState(env_id=1)
    env.reward_history.append(-0.8)

    snapshot = SanctumSnapshot(
        focused_env_id=1,
        envs={1: env},
        rewards=RewardComponentsSchema(bounded_attribution=-0.8),
    )
    widget = RewardComponents()
    widget.update_snapshot(snapshot)
    panel = widget.render()

    rendered = render_to_text(panel)
    assert "Attr:" in rendered
    assert "-0.80" in rendered


def test_reward_components_compute_rent():
    """Test compute rent always displays (usually negative, red)."""
    env = EnvState(env_id=1)
    env.reward_history.append(-0.5)

    snapshot = SanctumSnapshot(
        focused_env_id=1,
        envs={1: env},
        rewards=RewardComponentsSchema(compute_rent=-0.5),
    )
    widget = RewardComponents()
    widget.update_snapshot(snapshot)
    panel = widget.render()

    rendered = render_to_text(panel)
    assert "Rent:" in rendered
    assert "-0.50" in rendered


def test_reward_components_ratio_penalty_conditional():
    """Test ratio penalty only shows if non-zero."""
    # Zero penalty - should NOT show
    env1 = EnvState(env_id=1)
    env1.reward_history.append(0.0)

    snapshot1 = SanctumSnapshot(
        focused_env_id=1,
        envs={1: env1},
        rewards=RewardComponentsSchema(ratio_penalty=0.0),
    )
    widget = RewardComponents()
    widget.update_snapshot(snapshot1)
    panel1 = widget.render()
    rendered1 = render_to_text(panel1)
    assert "Penalty:" not in rendered1

    # Non-zero penalty - SHOULD show (red if negative)
    env2 = EnvState(env_id=2)
    env2.reward_history.append(-2.0)

    snapshot2 = SanctumSnapshot(
        focused_env_id=2,
        envs={2: env2},
        rewards=RewardComponentsSchema(ratio_penalty=-2.0),
    )
    widget.update_snapshot(snapshot2)
    panel2 = widget.render()
    rendered2 = render_to_text(panel2)
    assert "Penalty:" in rendered2
    assert "-2.00" in rendered2


def test_reward_components_stage_bonus():
    """Test stage bonus shows if non-zero (blue styling)."""
    env = EnvState(env_id=1)
    env.reward_history.append(1.0)

    snapshot = SanctumSnapshot(
        focused_env_id=1,
        envs={1: env},
        rewards=RewardComponentsSchema(stage_bonus=1.0),
    )
    widget = RewardComponents()
    widget.update_snapshot(snapshot)
    panel = widget.render()

    rendered = render_to_text(panel)
    assert "Stage:" in rendered
    assert "+1.00" in rendered


def test_reward_components_fossil_bonus():
    """Test fossilize bonus shows if non-zero (blue styling)."""
    env = EnvState(env_id=1)
    env.reward_history.append(10.0)

    snapshot = SanctumSnapshot(
        focused_env_id=1,
        envs={1: env},
        rewards=RewardComponentsSchema(fossilize_terminal_bonus=10.0),
    )
    widget = RewardComponents()
    widget.update_snapshot(snapshot)
    panel = widget.render()

    rendered = render_to_text(panel)
    assert "Fossil:" in rendered
    assert "+10.00" in rendered


def test_reward_components_blending_warning():
    """Test blending warning shows if negative (yellow styling)."""
    env = EnvState(env_id=1)
    env.reward_history.append(-0.5)

    snapshot = SanctumSnapshot(
        focused_env_id=1,
        envs={1: env},
        rewards=RewardComponentsSchema(blending_warning=-0.5),
    )
    widget = RewardComponents()
    widget.update_snapshot(snapshot)
    panel = widget.render()

    rendered = render_to_text(panel)
    assert "Blend Warn:" in rendered
    assert "-0.50" in rendered


def test_reward_components_probation_warning():
    """Test probation warning shows if negative (yellow styling)."""
    env = EnvState(env_id=1)
    env.reward_history.append(-0.3)

    snapshot = SanctumSnapshot(
        focused_env_id=1,
        envs={1: env},
        rewards=RewardComponentsSchema(probation_warning=-0.3),
    )
    widget = RewardComponents()
    widget.update_snapshot(snapshot)
    panel = widget.render()

    rendered = render_to_text(panel)
    assert "Prob Warn:" in rendered
    assert "-0.30" in rendered


def test_reward_components_total_positive():
    """Test total displays bold green when positive."""
    env = EnvState(env_id=1)
    env.reward_history.append(3.5)

    snapshot = SanctumSnapshot(
        focused_env_id=1,
        envs={1: env},
        rewards=RewardComponentsSchema(total=3.5),
    )
    widget = RewardComponents()
    widget.update_snapshot(snapshot)
    panel = widget.render()

    rendered = render_to_text(panel)
    assert "Total:" in rendered
    assert "+3.50" in rendered
    # Check for bold green styling


def test_reward_components_total_negative():
    """Test total displays bold red when negative."""
    env = EnvState(env_id=1)
    env.reward_history.append(-2.5)

    snapshot = SanctumSnapshot(
        focused_env_id=1,
        envs={1: env},
        rewards=RewardComponentsSchema(total=-2.5),
    )
    widget = RewardComponents()
    widget.update_snapshot(snapshot)
    panel = widget.render()

    rendered = render_to_text(panel)
    assert "Total:" in rendered
    assert "-2.50" in rendered
    # Check for bold red styling


# ============================================================================
# EsperStatus Tests
# ============================================================================


def test_esper_status_creation():
    """Test widget creation."""
    widget = EsperStatus()
    assert widget is not None


def test_esper_status_no_data():
    """Test render with no snapshot."""
    widget = EsperStatus()
    panel = widget.render()
    assert "No data" in panel.renderable


def test_esper_status_seed_stage_counts():
    """Test seed stage counts aggregate across all envs."""
    # Create 2 envs with various seed stages
    env1 = EnvState(env_id=1)
    env1.seeds = {
        "R0C0": SeedState(slot_id="R0C0", stage="TRAINING"),
        "R0C1": SeedState(slot_id="R0C1", stage="BLENDING"),
    }

    env2 = EnvState(env_id=2)
    env2.seeds = {
        "R0C0": SeedState(slot_id="R0C0", stage="TRAINING"),
        "R0C1": SeedState(slot_id="R0C1", stage="FOSSILIZED"),
        "R1C0": SeedState(slot_id="R1C0", stage="DORMANT"),  # Should NOT count
    }

    snapshot = SanctumSnapshot(envs={1: env1, 2: env2})
    widget = EsperStatus()
    widget.update_snapshot(snapshot)
    panel = widget.render()

    rendered = render_to_text(panel)
    # Should show Train: 2, Blend: 1, Foss: 1
    assert "Train:" in rendered or "TRAINING" in rendered
    assert "Blend:" in rendered or "BLENDING" in rendered
    assert "Foss:" in rendered or "FOSSILIZED" in rendered


def test_esper_status_host_params_millions():
    """Test host params formatting for millions (M)."""
    vitals = SystemVitals(host_params=2_500_000)
    snapshot = SanctumSnapshot(vitals=vitals)
    widget = EsperStatus()
    widget.update_snapshot(snapshot)
    panel = widget.render()

    rendered = render_to_text(panel)
    assert "Host Params:" in rendered
    assert "2.5M" in rendered


def test_esper_status_host_params_thousands():
    """Test host params formatting for thousands (K)."""
    vitals = SystemVitals(host_params=50_000)
    snapshot = SanctumSnapshot(vitals=vitals)
    widget = EsperStatus()
    widget.update_snapshot(snapshot)
    panel = widget.render()

    rendered = render_to_text(panel)
    assert "Host Params:" in rendered
    assert "50K" in rendered


def test_esper_status_host_params_small():
    """Test host params formatting for small counts (raw number)."""
    vitals = SystemVitals(host_params=500)
    snapshot = SanctumSnapshot(vitals=vitals)
    widget = EsperStatus()
    widget.update_snapshot(snapshot)
    panel = widget.render()

    rendered = render_to_text(panel)
    assert "Host Params:" in rendered
    assert "500" in rendered


def test_esper_status_throughput():
    """Test throughput display (epochs/sec, batches/hr)."""
    vitals = SystemVitals(epochs_per_second=1.5, batches_per_hour=360)
    snapshot = SanctumSnapshot(vitals=vitals)
    widget = EsperStatus()
    widget.update_snapshot(snapshot)
    panel = widget.render()

    rendered = render_to_text(panel)
    assert "Epochs/sec:" in rendered
    assert "1.50" in rendered
    assert "Batches/hr:" in rendered
    assert "360" in rendered


def test_esper_status_runtime_formatting():
    """Test runtime formatting (Xh Ym Zs)."""
    start_time = datetime.now() - timedelta(hours=2, minutes=30, seconds=45)
    snapshot = SanctumSnapshot(start_time=start_time)
    widget = EsperStatus()
    widget.update_snapshot(snapshot)
    panel = widget.render()

    rendered = render_to_text(panel)
    assert "Runtime:" in rendered
    assert "2h" in rendered
    assert "30m" in rendered
    # Seconds may vary slightly due to test execution time


def test_esper_status_runtime_no_start():
    """Test runtime shows dash when no start_time."""
    snapshot = SanctumSnapshot(start_time=None)
    widget = EsperStatus()
    widget.update_snapshot(snapshot)
    panel = widget.render()

    rendered = render_to_text(panel)
    assert "Runtime:" in rendered
    assert "-" in rendered


def test_esper_status_multi_gpu_display():
    """Test multi-GPU display with device labels."""
    gpu0 = GPUStats(device_id=0, memory_used_gb=4.5, memory_total_gb=16.0, utilization=75)
    gpu1 = GPUStats(device_id=1, memory_used_gb=8.0, memory_total_gb=16.0, utilization=60)
    vitals = SystemVitals(gpu_stats={0: gpu0, 1: gpu1})
    snapshot = SanctumSnapshot(vitals=vitals)
    widget = EsperStatus()
    widget.update_snapshot(snapshot)
    panel = widget.render()

    rendered = render_to_text(panel)
    # Should show GPU0 and GPU1 labels
    assert "GPU0:" in rendered or "GPU1:" in rendered
    assert "4.5" in rendered  # GPU0 used
    assert "8.0" in rendered  # GPU1 used


def test_esper_status_gpu_memory_green():
    """Test GPU memory shows green when < 75%."""
    gpu = GPUStats(device_id=0, memory_used_gb=6.0, memory_total_gb=16.0)  # 37.5%
    vitals = SystemVitals(gpu_stats={0: gpu})
    snapshot = SanctumSnapshot(vitals=vitals)
    widget = EsperStatus()
    widget.update_snapshot(snapshot)
    panel = widget.render()

    rendered = render_to_text(panel)
    assert "6.0/16.0GB" in rendered


def test_esper_status_gpu_memory_yellow():
    """Test GPU memory shows yellow when 75-90%."""
    gpu = GPUStats(device_id=0, memory_used_gb=13.0, memory_total_gb=16.0)  # 81.25%
    vitals = SystemVitals(gpu_stats={0: gpu})
    snapshot = SanctumSnapshot(vitals=vitals)
    widget = EsperStatus()
    widget.update_snapshot(snapshot)
    panel = widget.render()

    rendered = render_to_text(panel)
    assert "13.0/16.0GB" in rendered


def test_esper_status_gpu_memory_red():
    """Test GPU memory shows red when > 90%."""
    gpu = GPUStats(device_id=0, memory_used_gb=15.0, memory_total_gb=16.0)  # 93.75%
    vitals = SystemVitals(gpu_stats={0: gpu})
    snapshot = SanctumSnapshot(vitals=vitals)
    widget = EsperStatus()
    widget.update_snapshot(snapshot)
    panel = widget.render()

    rendered = render_to_text(panel)
    assert "15.0/16.0GB" in rendered


def test_esper_status_gpu_utilization_green():
    """Test GPU utilization shows green when < 80%."""
    gpu = GPUStats(device_id=0, memory_used_gb=8.0, memory_total_gb=16.0, utilization=65)
    vitals = SystemVitals(gpu_stats={0: gpu})
    snapshot = SanctumSnapshot(vitals=vitals)
    widget = EsperStatus()
    widget.update_snapshot(snapshot)
    panel = widget.render()

    rendered = render_to_text(panel)
    assert "65%" in rendered


def test_esper_status_gpu_utilization_yellow():
    """Test GPU utilization shows yellow when 80-95%."""
    gpu = GPUStats(device_id=0, memory_used_gb=8.0, memory_total_gb=16.0, utilization=88)
    vitals = SystemVitals(gpu_stats={0: gpu})
    snapshot = SanctumSnapshot(vitals=vitals)
    widget = EsperStatus()
    widget.update_snapshot(snapshot)
    panel = widget.render()

    rendered = render_to_text(panel)
    assert "88%" in rendered


def test_esper_status_gpu_utilization_red():
    """Test GPU utilization shows red when > 95%."""
    gpu = GPUStats(device_id=0, memory_used_gb=8.0, memory_total_gb=16.0, utilization=98)
    vitals = SystemVitals(gpu_stats={0: gpu})
    snapshot = SanctumSnapshot(vitals=vitals)
    widget = EsperStatus()
    widget.update_snapshot(snapshot)
    panel = widget.render()

    rendered = render_to_text(panel)
    assert "98%" in rendered


def test_esper_status_ram_display():
    """Test RAM usage display with color thresholds."""
    vitals = SystemVitals(ram_used_gb=24.5, ram_total_gb=32.0)  # 76.5% - yellow
    snapshot = SanctumSnapshot(vitals=vitals)
    widget = EsperStatus()
    widget.update_snapshot(snapshot)
    panel = widget.render()

    rendered = render_to_text(panel)
    assert "RAM:" in rendered
    assert "24.5/32GB" in rendered


def test_esper_status_cpu_display():
    """Test CPU percentage display (FIX: was never shown in old TUI)."""
    vitals = SystemVitals(cpu_percent=45.2)
    snapshot = SanctumSnapshot(vitals=vitals)
    widget = EsperStatus()
    widget.update_snapshot(snapshot)
    panel = widget.render()

    rendered = render_to_text(panel)
    assert "CPU:" in rendered
    assert "45.2%" in rendered


def test_esper_status_cpu_zero_not_shown():
    """Test CPU percentage not shown if zero."""
    vitals = SystemVitals(cpu_percent=0.0)
    snapshot = SanctumSnapshot(vitals=vitals)
    widget = EsperStatus()
    widget.update_snapshot(snapshot)
    panel = widget.render()

    rendered = render_to_text(panel)
    # Should not show CPU line when 0
    assert "CPU:" not in rendered


def test_esper_status_fallback_single_gpu():
    """Test fallback to legacy single-GPU fields when gpu_stats is empty."""
    vitals = SystemVitals(
        gpu_memory_used_gb=6.0,
        gpu_memory_total_gb=16.0,
        gpu_utilization=70,
    )
    snapshot = SanctumSnapshot(vitals=vitals)
    widget = EsperStatus()
    widget.update_snapshot(snapshot)
    panel = widget.render()

    rendered = render_to_text(panel)
    assert "GPU:" in rendered
    assert "6.0/16.0GB" in rendered
    assert "70%" in rendered
