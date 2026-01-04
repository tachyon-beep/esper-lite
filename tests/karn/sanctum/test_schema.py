"""Tests for Sanctum schema - must match all existing TUI state."""
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

    def test_growth_ratio_property(self):
        """Growth ratio shows mutation overhead: (host + fossilized) / host."""
        # Standard case: 1M host + 200K fossilized = 1.2x growth
        env = EnvState(env_id=0, host_params=1_000_000, fossilized_params=200_000)
        assert env.growth_ratio == 1.2

        # No fossilized seeds = 1.0x (no growth)
        env_no_seeds = EnvState(env_id=0, host_params=1_000_000, fossilized_params=0)
        assert env_no_seeds.growth_ratio == 1.0

        # Edge case: no host params = 1.0 (avoid division by zero)
        env_no_host = EnvState(env_id=0, host_params=0, fossilized_params=0)
        assert env_no_host.growth_ratio == 1.0

    def test_gaming_rate_property(self):
        """Gaming rate shows fraction of steps with anti-gaming penalties."""
        # Standard case: 5 gaming triggers out of 100 steps = 5%
        env = EnvState(env_id=0, gaming_trigger_count=5, total_reward_steps=100)
        assert abs(env.gaming_rate - 0.05) < 1e-9

        # No gaming triggers = 0%
        env_clean = EnvState(env_id=0, gaming_trigger_count=0, total_reward_steps=50)
        assert env_clean.gaming_rate == 0.0

        # Edge case: no reward steps = 0 (avoid division by zero)
        env_no_steps = EnvState(env_id=0, gaming_trigger_count=0, total_reward_steps=0)
        assert env_no_steps.gaming_rate == 0.0

        # High gaming rate
        env_problematic = EnvState(env_id=0, gaming_trigger_count=20, total_reward_steps=100)
        assert abs(env_problematic.gaming_rate - 0.20) < 1e-9

    def test_gaming_fields_default_to_zero(self):
        """Gaming tracking fields should default to zero."""
        env = EnvState(env_id=0)
        assert env.gaming_trigger_count == 0
        assert env.total_reward_steps == 0
        assert env.gaming_rate == 0.0


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

    def test_history_deques(self):
        """TamiyoState should have deque fields for sparkline history."""
        tamiyo = TamiyoState()

        # Should have history deques with maxlen=10
        assert isinstance(tamiyo.policy_loss_history, deque)
        assert isinstance(tamiyo.value_loss_history, deque)
        assert isinstance(tamiyo.grad_norm_history, deque)
        assert isinstance(tamiyo.entropy_history, deque)
        assert isinstance(tamiyo.explained_variance_history, deque)
        assert isinstance(tamiyo.kl_divergence_history, deque)
        assert isinstance(tamiyo.clip_fraction_history, deque)

        # Should have maxlen of 10
        assert tamiyo.policy_loss_history.maxlen == 10
        assert tamiyo.value_loss_history.maxlen == 10
        assert tamiyo.grad_norm_history.maxlen == 10
        assert tamiyo.entropy_history.maxlen == 10
        assert tamiyo.explained_variance_history.maxlen == 10
        assert tamiyo.kl_divergence_history.maxlen == 10
        assert tamiyo.clip_fraction_history.maxlen == 10

    def test_per_head_entropy_fields(self):
        """TamiyoState should have per-head entropy for all 8 action heads."""
        tamiyo = TamiyoState()

        # Should have all 8 head entropy fields
        assert hasattr(tamiyo, 'head_slot_entropy')
        assert hasattr(tamiyo, 'head_blueprint_entropy')
        assert hasattr(tamiyo, 'head_style_entropy')
        assert hasattr(tamiyo, 'head_tempo_entropy')
        assert hasattr(tamiyo, 'head_alpha_target_entropy')
        assert hasattr(tamiyo, 'head_alpha_speed_entropy')
        assert hasattr(tamiyo, 'head_alpha_curve_entropy')
        assert hasattr(tamiyo, 'head_op_entropy')

    def test_tamiyo_state_has_group_id(self):
        """TamiyoState should have group_id for A/B testing identification."""
        from esper.karn.sanctum.schema import TamiyoState

        # Default is None (not in A/B mode)
        state = TamiyoState()
        assert state.group_id is None

        # Can be set to policy group identifier
        state_a = TamiyoState(group_id="A")
        assert state_a.group_id == "A"

        state_b = TamiyoState(group_id="B")
        assert state_b.group_id == "B"


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
            holding_warning=0.0,
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
    assert vitals.memory_alarm_devices == ["RAM", "cuda:0"]


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


def test_system_vitals_ram_only_alarm():
    """RAM alarm should be included in memory_alarm_devices."""
    vitals = SystemVitals(
        gpu_memory_used_gb=5.0,
        gpu_memory_total_gb=10.0,  # 50% - no alarm
        ram_used_gb=30.0,
        ram_total_gb=32.0,  # 93.75% - alarm
    )
    assert vitals.has_memory_alarm is True
    assert "RAM" in vitals.memory_alarm_devices
    assert "cuda:0" not in vitals.memory_alarm_devices


def test_system_vitals_multi_gpu_alarm():
    """Multi-GPU alarm detection via gpu_stats dict."""
    vitals = SystemVitals(
        gpu_stats={
            0: GPUStats(device_id=0, memory_used_gb=5.0, memory_total_gb=10.0),  # 50%
            1: GPUStats(device_id=1, memory_used_gb=9.5, memory_total_gb=10.0),  # 95%
        },
        ram_used_gb=10.0,
        ram_total_gb=32.0,  # 31% - no alarm
    )
    assert vitals.has_memory_alarm is True
    assert vitals.memory_alarm_devices == ["cuda:1"]


def test_decision_snapshot_creation():
    """Test DecisionSnapshot dataclass creation."""
    from datetime import datetime, timezone
    from esper.karn.sanctum.schema import DecisionSnapshot

    decision = DecisionSnapshot(
        timestamp=datetime.now(timezone.utc),
        slot_states={"r0c0": "Training 12%", "r0c1": "Empty"},
        host_accuracy=67.0,
        chosen_action="GERMINATE",
        chosen_slot="r0c1",
        confidence=0.73,
        expected_value=0.42,
        actual_reward=0.38,
        alternatives=[("WAIT", 0.15), ("SET_ALPHA_TARGET r0c0", 0.12)],
    )

    assert decision.chosen_action == "GERMINATE"
    assert decision.confidence == 0.73
    assert len(decision.alternatives) == 2


def test_env_state_status_hysteresis():
    """Status changes require 3 consecutive epochs of the condition."""
    env = EnvState(env_id=0)

    # Initial state
    assert env.status == "initializing"
    assert env.stall_counter == 0

    # Simulate epochs 0-11 (12 calls) without improvement
    for i in range(12):
        env.add_accuracy(50.0, epoch=i)  # Same accuracy = no improvement

    # After epoch 11, epochs_since_improvement=11 (>10), so stall_counter=1
    # Status should still be initializing until counter reaches 3
    assert env.stall_counter == 1

    # Continue for 2 more epochs
    env.add_accuracy(50.0, epoch=12)
    env.add_accuracy(50.0, epoch=13)

    # NOW status should be stalled (3 consecutive epochs over threshold)
    assert env.status == "stalled"

    # Improvement resets counter
    env.add_accuracy(60.0, epoch=14)  # Better!
    assert env.stall_counter == 0
    assert env.status == "healthy"  # Improved but not > 80%


def test_env_state_degraded_hysteresis():
    """Degraded status requires 3 consecutive drops to trigger."""
    env = EnvState(env_id=0)

    # Start with good accuracy (must be >80.0, not ==80.0)
    env.add_accuracy(85.0, epoch=0)
    assert env.status == "excellent"
    assert env.degraded_counter == 0

    # First drop >1%
    env.add_accuracy(83.5, epoch=1)  # -1.5% drop
    assert env.degraded_counter == 1
    assert env.status == "excellent"  # Not degraded yet

    # Second drop >1%
    env.add_accuracy(82.0, epoch=2)  # -1.5% drop
    assert env.degraded_counter == 2
    assert env.status == "excellent"  # Still not degraded

    # Third drop >1% - should trigger degraded status
    env.add_accuracy(80.5, epoch=3)  # -1.5% drop
    assert env.degraded_counter == 3
    assert env.status == "degraded"  # NOW it's degraded

    # Improvement resets counter but status stays degraded (didn't beat best)
    env.add_accuracy(81.0, epoch=4)  # +0.5% improvement over previous
    assert env.degraded_counter == 0  # Counter resets on improvement
    assert env.status == "degraded"  # Still degraded (didn't beat best of 85.0)

    # New best accuracy - now status should change
    env.add_accuracy(86.0, epoch=5)  # New best!
    assert env.degraded_counter == 0
    assert env.status == "excellent"  # Now excellent (new best >80%)


def test_degraded_counter_persists_on_stable():
    """Degraded counter should NOT reset when accuracy is stable (no improvement)."""
    env = EnvState(env_id=0)

    # Start with good accuracy (must be >80.0)
    env.add_accuracy(85.0, epoch=0)
    assert env.degraded_counter == 0

    # First drop >1%
    env.add_accuracy(83.5, epoch=1)  # -1.5% drop
    assert env.degraded_counter == 1

    # Second drop >1%
    env.add_accuracy(82.0, epoch=2)  # -1.5% drop
    assert env.degraded_counter == 2

    # STABLE accuracy (no change) - counter should PERSIST
    env.add_accuracy(82.0, epoch=3)  # 0% change
    assert env.degraded_counter == 2  # Counter should NOT reset!
    assert env.status == "excellent"  # Not degraded yet

    # Another drop >1% - should NOW trigger degraded
    env.add_accuracy(80.5, epoch=4)  # -1.5% drop
    assert env.degraded_counter == 3
    assert env.status == "degraded"

    # Verify that only IMPROVEMENT resets counter
    env.add_accuracy(80.5, epoch=5)  # Stable again
    assert env.degraded_counter == 3  # Counter still persists

    env.add_accuracy(81.0, epoch=6)  # +0.5% improvement
    assert env.degraded_counter == 0  # NOW it resets


# ========================================================================
# Task 2: Trend Detection Tests
# ========================================================================

def test_detect_trend_improving_loss():
    """Decreasing loss should be detected as 'improving'."""
    from esper.karn.sanctum.schema import detect_trend

    # Clear downward trend in loss
    values = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3]
    trend = detect_trend(values, metric_name="policy_loss", metric_type="loss")
    assert trend == "improving", f"Expected 'improving', got '{trend}'"


def test_detect_trend_warning_loss():
    """Increasing loss should be detected as 'warning'."""
    from esper.karn.sanctum.schema import detect_trend

    values = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    trend = detect_trend(values, metric_name="policy_loss", metric_type="loss")
    assert trend == "warning", f"Expected 'warning', got '{trend}'"


def test_detect_trend_stable():
    """Flat values should be detected as 'stable'."""
    from esper.karn.sanctum.schema import detect_trend

    values = [0.5, 0.51, 0.49, 0.5, 0.51, 0.5, 0.49, 0.5, 0.51, 0.5, 0.49, 0.5, 0.51, 0.5, 0.49]
    trend = detect_trend(values, metric_name="policy_loss", metric_type="loss")
    assert trend == "stable", f"Expected 'stable', got '{trend}'"


def test_detect_trend_volatile():
    """High recent variance vs historical should be 'volatile'."""
    from esper.karn.sanctum.schema import detect_trend

    # Stable early, then wild swings
    values = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6]
    trend = detect_trend(values, metric_name="policy_loss", metric_type="loss")
    assert trend == "volatile", f"Expected 'volatile', got '{trend}'"


def test_detect_trend_accuracy_improving():
    """Increasing accuracy (metric_type='accuracy') should be 'improving'."""
    from esper.karn.sanctum.schema import detect_trend

    # Clear upward trend in accuracy
    values = [50.0, 52.0, 54.0, 56.0, 58.0, 60.0, 62.0, 64.0, 66.0, 68.0, 70.0, 72.0, 74.0, 76.0, 78.0]
    trend = detect_trend(values, metric_name="episode_return", metric_type="accuracy")
    assert trend == "improving", f"Expected 'improving', got '{trend}'"


def test_detect_trend_accuracy_warning():
    """Decreasing accuracy (metric_type='accuracy') should be 'warning'."""
    from esper.karn.sanctum.schema import detect_trend

    # Clear downward trend in accuracy
    values = [78.0, 76.0, 74.0, 72.0, 70.0, 68.0, 66.0, 64.0, 62.0, 60.0, 58.0, 56.0, 54.0, 52.0, 50.0]
    trend = detect_trend(values, metric_name="episode_return", metric_type="accuracy")
    assert trend == "warning", f"Expected 'warning', got '{trend}'"


def test_detect_trend_insufficient_data():
    """Less than 5 values should return 'stable'."""
    from esper.karn.sanctum.schema import detect_trend

    values = [0.5, 0.6, 0.7]
    trend = detect_trend(values, metric_name="policy_loss", metric_type="loss")
    assert trend == "stable", f"Expected 'stable' for insufficient data, got '{trend}'"


def test_trend_to_indicator():
    """Convert trend labels to display indicators and styles."""
    from esper.karn.sanctum.schema import trend_to_indicator

    # Test all trend types
    indicator, style = trend_to_indicator("improving")
    assert indicator == "^", f"Expected '^' for improving, got '{indicator}'"
    assert style == "green", f"Expected 'green' for improving, got '{style}'"

    indicator, style = trend_to_indicator("stable")
    assert indicator == "-", f"Expected '-' for stable, got '{indicator}'"
    assert style == "dim", f"Expected 'dim' for stable, got '{style}'"

    indicator, style = trend_to_indicator("volatile")
    assert indicator == "~", f"Expected '~' for volatile, got '{indicator}'"
    assert style == "yellow", f"Expected 'yellow' for volatile, got '{style}'"

    indicator, style = trend_to_indicator("warning")
    assert indicator == "v", f"Expected 'v' for warning, got '{indicator}'"
    assert style == "red", f"Expected 'red' for warning, got '{style}'"

    # Unknown trend should return stable
    indicator, style = trend_to_indicator("unknown")
    assert indicator == "-", f"Expected '-' for unknown, got '{indicator}'"
    assert style == "dim", f"Expected 'dim' for unknown, got '{style}'"


def test_detect_trend_minimum_window():
    """Minimum valid window size is 3."""
    from esper.karn.sanctum.schema import detect_trend

    # 6 values = window_size 3
    values = [1.0, 0.9, 0.8, 0.5, 0.4, 0.3]
    trend = detect_trend(values, metric_name="policy_loss", metric_type="loss")
    assert trend == "improving", f"Expected 'improving' with minimum window, got '{trend}'"


def test_detect_trend_uses_deque():
    """Function should accept deque as input."""
    from esper.karn.sanctum.schema import detect_trend
    from collections import deque

    values = deque([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1], maxlen=20)
    trend = detect_trend(values, metric_name="policy_loss", metric_type="loss")
    assert trend == "improving"


# =============================================================================
# DECISION SNAPSHOT HEAD CHOICE FIELDS (per DRL specialist review)
# =============================================================================


def test_decision_snapshot_has_head_choice_fields():
    """DecisionSnapshot should include blueprint, style, tempo head choices.

    Per DRL specialist: understanding which heads drive decisions helps
    diagnose credit assignment issues.
    """
    from esper.karn.sanctum.schema import DecisionSnapshot
    from datetime import datetime, timezone

    decision = DecisionSnapshot(
        timestamp=datetime.now(timezone.utc),
        slot_states={},
        host_accuracy=75.0,
        chosen_action="GERMINATE",
        chosen_slot="slot_0",
        confidence=0.92,
        expected_value=0.5,
        actual_reward=None,
        alternatives=[],
        decision_id="test-1",
        # New fields (per DRL specialist review)
        chosen_blueprint="conv_light",
        chosen_tempo="STANDARD",
        chosen_style="LINEAR_ADD",
        blueprint_confidence=0.87,
        tempo_confidence=0.65,
        op_confidence=0.92,
    )

    assert decision.chosen_blueprint == "conv_light"
    assert decision.chosen_tempo == "STANDARD"
    assert decision.chosen_style == "LINEAR_ADD"
    assert decision.blueprint_confidence == 0.87
    assert decision.tempo_confidence == 0.65
    assert decision.op_confidence == 0.92


# =============================================================================
# INFRASTRUCTURE METRICS (per PyTorch expert review)
# =============================================================================


def test_infrastructure_metrics_dataclass():
    """InfrastructureMetrics should contain CUDA memory and compile status."""
    from esper.karn.sanctum.schema import InfrastructureMetrics

    metrics = InfrastructureMetrics()

    # Memory fields
    assert metrics.cuda_memory_allocated_gb == 0.0
    assert metrics.cuda_memory_reserved_gb == 0.0
    assert metrics.cuda_memory_peak_gb == 0.0
    assert metrics.cuda_memory_fragmentation == 0.0

    # Compile status (static session metadata - no runtime health detection)
    assert metrics.compile_enabled is False
    assert metrics.compile_backend == ""
    assert metrics.compile_mode == ""


def test_infrastructure_metrics_memory_usage_percent():
    """memory_usage_percent property should compute (allocated/reserved) * 100."""
    from esper.karn.sanctum.schema import InfrastructureMetrics

    # Standard case: 4.2 / 8.0 = 52.5%
    metrics = InfrastructureMetrics(
        cuda_memory_allocated_gb=4.2,
        cuda_memory_reserved_gb=8.0,
    )
    assert abs(metrics.memory_usage_percent - 52.5) < 0.1

    # Edge case: no reserved memory = 0%
    metrics_empty = InfrastructureMetrics(
        cuda_memory_allocated_gb=0.0,
        cuda_memory_reserved_gb=0.0,
    )
    assert metrics_empty.memory_usage_percent == 0.0


# =============================================================================
# GRADIENT QUALITY METRICS (per DRL expert review)
# =============================================================================


def test_gradient_quality_metrics_dataclass():
    """GradientQualityMetrics should contain gradient CV and directional clip."""
    from esper.karn.sanctum.schema import GradientQualityMetrics

    metrics = GradientQualityMetrics()

    # Gradient coefficient of variation (NOT SNR - per DRL review)
    # Low CV (<0.5) = high signal quality, High CV (>2.0) = noisy
    assert metrics.gradient_cv == 0.0

    # Directional clip fraction (per DRL expert)
    # clip+ = probability increases capped (r > 1+ε)
    # clip- = probability decreases capped (r < 1-ε)
    assert metrics.clip_fraction_positive == 0.0
    assert metrics.clip_fraction_negative == 0.0


def test_tamiyo_state_has_nested_metrics():
    """TamiyoState should have infrastructure and gradient_quality nested fields."""
    from esper.karn.sanctum.schema import (
        TamiyoState,
        InfrastructureMetrics,
        GradientQualityMetrics,
    )

    state = TamiyoState()

    # Nested dataclasses should be present with defaults
    assert isinstance(state.infrastructure, InfrastructureMetrics)
    assert isinstance(state.gradient_quality, GradientQualityMetrics)

    # Access nested fields
    assert state.infrastructure.cuda_memory_allocated_gb == 0.0
    assert state.gradient_quality.gradient_cv == 0.0


# =============================================================================
# DECISION SNAPSHOT PER-HEAD ENTROPY FIELDS (Task 4)
# =============================================================================


def test_tamiyo_state_has_nan_inf_latch_fields():
    """TamiyoState should have per-head NaN/Inf latch dicts pre-populated."""
    from esper.leyline import HEAD_NAMES
    from esper.karn.sanctum.schema import TamiyoState

    state = TamiyoState()

    # Should have latch dicts pre-populated with all heads set to False
    assert hasattr(state, "head_nan_latch")
    assert hasattr(state, "head_inf_latch")

    # All HEAD_NAMES keys should exist (no .get() needed in display code)
    for head in HEAD_NAMES:
        assert head in state.head_nan_latch
        assert head in state.head_inf_latch
        assert state.head_nan_latch[head] is False
        assert state.head_inf_latch[head] is False


def test_decision_snapshot_has_entropy_fields():
    """DecisionSnapshot should have per-head entropy fields."""
    from datetime import datetime, timezone
    from esper.karn.sanctum.schema import DecisionSnapshot

    decision = DecisionSnapshot(
        timestamp=datetime.now(timezone.utc),
        slot_states={},
        host_accuracy=0.9,
        chosen_action="GERMINATE",
        chosen_slot="r0c0",
        confidence=0.85,
        expected_value=0.5,
        actual_reward=0.6,
        alternatives=[],
        decision_id="abc123",
        decision_entropy=0.4,
        env_id=0,
        value_residual=0.1,
        # Entropy fields
        op_entropy=0.3,
        slot_entropy=0.8,
        blueprint_entropy=0.5,
        style_entropy=0.6,
        tempo_entropy=0.4,
        alpha_target_entropy=0.55,
        alpha_speed_entropy=0.45,
        curve_entropy=0.35,
    )

    assert decision.op_entropy == 0.3
    assert decision.slot_entropy == 0.8
    assert decision.blueprint_entropy == 0.5
    assert decision.style_entropy == 0.6
    assert decision.tempo_entropy == 0.4
    assert decision.alpha_target_entropy == 0.55
    assert decision.alpha_speed_entropy == 0.45
    assert decision.curve_entropy == 0.35
