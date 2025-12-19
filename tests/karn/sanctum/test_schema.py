"""Tests for Sanctum schema - must match all existing TUI state."""
import pytest
from collections import deque

from esper.karn.sanctum.schema import (
    SanctumSnapshot,
    EnvState,
    SeedState,
    TamiyoState,
    SystemVitals,
    RewardComponents,
    GPUStats,
)


class TestSeedState:
    """SeedState must capture all per-slot data from existing TUI."""

    def test_defaults(self):
        seed = SeedState(slot_id="r0c0")
        assert seed.slot_id == "r0c0"
        assert seed.stage == "DORMANT"
        assert seed.alpha == 0.0

    def test_gradient_health_flags(self):
        """Must track vanishing/exploding gradients for ▼▲ indicators."""
        seed = SeedState(
            slot_id="r0c0",
            has_vanishing=True,
            has_exploding=False,
        )
        assert seed.has_vanishing is True
        assert seed.has_exploding is False

    def test_epochs_in_stage(self):
        """Must track epochs for 'e5' display in slot cell."""
        seed = SeedState(slot_id="r0c0", epochs_in_stage=5)
        assert seed.epochs_in_stage == 5


class TestEnvState:
    """EnvState must capture all per-environment data."""

    def test_defaults(self):
        env = EnvState(env_id=0)
        assert env.env_id == 0
        assert env.host_accuracy == 0.0
        assert env.seeds == {}

    def test_history_tracking(self):
        """Must support sparkline generation."""
        env = EnvState(env_id=0)
        env.accuracy_history.append(75.0)
        env.accuracy_history.append(76.5)
        env.reward_history.append(0.5)
        assert len(env.accuracy_history) == 2
        assert len(env.reward_history) == 1

    def test_action_tracking(self):
        """Must track per-env action distribution."""
        env = EnvState(env_id=0)
        env.action_counts["WAIT"] = 10
        env.action_counts["GERMINATE"] = 5
        assert env.action_counts["WAIT"] == 10

    def test_best_seeds_snapshot(self):
        """Must preserve seeds at best accuracy."""
        env = EnvState(env_id=0)
        env.best_seeds["r0c0"] = SeedState(slot_id="r0c0", stage="FOSSILIZED")
        assert "r0c0" in env.best_seeds

    def test_fossilized_params_tracking(self):
        """Must track fossilized_params for scoreboard display."""
        env = EnvState(env_id=0, fossilized_params=128000)
        assert env.fossilized_params == 128000


class TestTamiyoState:
    """TamiyoState must capture all policy agent metrics."""

    def test_defaults(self):
        tamiyo = TamiyoState()
        assert tamiyo.entropy == 0.0
        assert tamiyo.clip_fraction == 0.0

    def test_learning_rate(self):
        """Must track LR for Vitals display."""
        tamiyo = TamiyoState(learning_rate=3e-4)
        assert tamiyo.learning_rate == 3e-4

    def test_gradient_health_metrics(self):
        """Must track dead/exploding layers and GradHP."""
        tamiyo = TamiyoState(
            dead_layers=2,
            exploding_layers=0,
            layer_gradient_health=0.85,
        )
        assert tamiyo.dead_layers == 2
        assert tamiyo.layer_gradient_health == 0.85

    def test_ratio_stats(self):
        """Must track all ratio statistics including std."""
        tamiyo = TamiyoState(
            ratio_mean=1.02,
            ratio_min=0.8,
            ratio_max=1.5,
            ratio_std=0.15,
        )
        assert tamiyo.ratio_std == 0.15

    def test_total_actions_tracking(self):
        """Must track total_actions for action percentage calculation."""
        tamiyo = TamiyoState(total_actions=100)
        assert tamiyo.total_actions == 100

    def test_advantage_range(self):
        """Must track advantage min/max for value function analysis."""
        tamiyo = TamiyoState(
            advantage_min=-1.5,
            advantage_max=2.3,
        )
        assert tamiyo.advantage_min == -1.5
        assert tamiyo.advantage_max == 2.3


class TestSystemVitals:
    """SystemVitals must capture all system metrics."""

    def test_cpu_percent(self):
        """CPU was collected but never displayed - must fix."""
        vitals = SystemVitals(cpu_percent=67.0)
        assert vitals.cpu_percent == 67.0

    def test_multi_gpu(self):
        """Must support multiple GPUs."""
        vitals = SystemVitals()
        vitals.gpu_stats[0] = GPUStats(device_id=0, memory_used_gb=12.0)
        vitals.gpu_stats[1] = GPUStats(device_id=1, memory_used_gb=8.0)
        assert len(vitals.gpu_stats) == 2

    def test_throughput(self):
        """Must track epochs/sec and batches/hr."""
        vitals = SystemVitals(
            epochs_per_second=2.5,
            batches_per_hour=150.0,
        )
        assert vitals.epochs_per_second == 2.5


class TestRewardComponents:
    """RewardComponents must match Esper reward breakdown."""

    def test_esper_components(self):
        """Must have ALL Esper-specific reward components."""
        rewards = RewardComponents(
            base_acc_delta=0.5,
            bounded_attribution=0.3,
            seed_contribution=0.2,
            compute_rent=-0.1,
            ratio_penalty=-0.05,
            stage_bonus=0.2,
            fossilize_terminal_bonus=1.0,
            blending_warning=-0.1,
            probation_warning=0.0,
            val_acc=75.5,
        )
        assert rewards.base_acc_delta == 0.5
        assert rewards.compute_rent == -0.1
        assert rewards.val_acc == 75.5


class TestSanctumSnapshot:
    """SanctumSnapshot aggregates all state."""

    def test_creation(self):
        snapshot = SanctumSnapshot(
            envs={0: EnvState(env_id=0)},
            tamiyo=TamiyoState(),
            vitals=SystemVitals(),
        )
        assert len(snapshot.envs) == 1

    def test_staleness_detection(self):
        """Must detect stale data (>5s since update)."""
        snapshot = SanctumSnapshot()
        assert snapshot.is_stale is True  # No update yet

    def test_slot_config(self):
        """Must track dynamic slot configuration."""
        snapshot = SanctumSnapshot(slot_ids=["r0c0", "r0c1", "r1c0", "r1c1"])
        assert len(snapshot.slot_ids) == 4


def test_system_vitals_memory_alarm_threshold():
    """Memory above 90% should trigger alarm state."""
    vitals = SystemVitals(
        gpu_memory_used_gb=9.5,
        gpu_memory_total_gb=10.0,  # 95% usage
        ram_used_gb=30.0,
        ram_total_gb=32.0,  # 93.75% usage
    )
    assert vitals.has_memory_alarm is True
    assert vitals.memory_alarm_devices == ["cuda:0"]


def test_system_vitals_no_alarm_below_threshold():
    """Memory below 90% should not trigger alarm."""
    vitals = SystemVitals(
        gpu_memory_used_gb=7.0,
        gpu_memory_total_gb=10.0,  # 70% usage
        ram_used_gb=20.0,
        ram_total_gb=32.0,  # 62.5% usage
    )
    assert vitals.has_memory_alarm is False
    assert vitals.memory_alarm_devices == []
