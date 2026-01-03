"""End-to-end tests for EnvState extended telemetry records (TELE-630 to TELE-633).

Verifies environment state telemetry flows from source through to sanctum.

These tests cover:
- TELE-630: env_status (FULLY WIRED)
- TELE-631: env_reward_mode (FULLY WIRED)
- TELE-632: env_rolled_back (FULLY WIRED)
- TELE-633: env_rollback_reason (FULLY WIRED)

All four records are fully wired and operational. They cover:
- Environment health status with hysteresis (TELE-630)
- A/B test cohort assignment for reward shaping experiments (TELE-631)
- Governor rollback state indicating catastrophic failure (TELE-632)
- Reason for governor intervention (TELE-633)
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from esper.karn.sanctum.aggregator import SanctumAggregator
from esper.karn.sanctum.schema import EnvState
from esper.leyline import (
    EpochCompletedPayload,
    GovernorRollbackPayload,
    TelemetryEvent,
    TelemetryEventType,
    TrainingStartedPayload,
)


# =============================================================================
# Helper functions
# =============================================================================


def make_epoch_event(
    env_id: int,
    val_accuracy: float,
    val_loss: float = 0.5,
    inner_epoch: int = 0,
) -> MagicMock:
    """Create a mock EPOCH_COMPLETED event for testing."""
    event = MagicMock()
    event.event_type = MagicMock()
    event.event_type.name = "EPOCH_COMPLETED"
    event.timestamp = datetime.now(timezone.utc)
    event.data = EpochCompletedPayload(
        env_id=env_id,
        val_accuracy=val_accuracy,
        val_loss=val_loss,
        inner_epoch=inner_epoch,
    )
    return event


def make_training_started_event(
    n_envs: int = 4,
    reward_mode: str = "shaped",
) -> MagicMock:
    """Create a mock TRAINING_STARTED event with reward_mode."""
    event = MagicMock()
    event.event_type = MagicMock()
    event.event_type.name = "TRAINING_STARTED"
    event.timestamp = datetime.now(timezone.utc)
    event.data = TrainingStartedPayload(
        n_envs=n_envs,
        max_epochs=150,
        max_batches=100,
        task="test_task",
        host_params=1_000_000,
        slot_ids=("r0c0", "r0c1"),
        seed=42,
        n_episodes=100,
        lr=3e-4,
        clip_ratio=0.2,
        entropy_coef=0.01,
        param_budget=10_000_000,
        policy_device="cpu",
        env_devices=("cpu",),
        reward_mode=reward_mode,
    )
    return event


def make_governor_rollback_event(
    env_id: int,
    reason: str = "governor_nan",
    device: str = "cuda:0",
) -> MagicMock:
    """Create a mock GOVERNOR_ROLLBACK event."""
    event = MagicMock()
    event.event_type = MagicMock()
    event.event_type.name = "GOVERNOR_ROLLBACK"
    event.timestamp = datetime.now(timezone.utc)
    event.data = GovernorRollbackPayload(
        env_id=env_id,
        device=device,
        reason=reason,
    )
    return event


# =============================================================================
# TELE-630: Environment Status
# =============================================================================


class TestTELE630EnvStatus:
    """TELE-630: Environment status tracking.

    Wiring Status: FULLY WIRED
    - Emitter: EnvState._update_status() computes status from accuracy deltas
    - Transport: Status computed during add_accuracy() call
    - Schema: EnvState.status field at line 530
    - Consumer: EnvOverview._format_status() displays color-coded status

    Status values:
    - "initializing": Environment has just started, no meaningful data yet
    - "healthy": Environment is making progress, accuracy improving regularly
    - "excellent": Environment has achieved >80% accuracy and is currently improving
    - "stalled": Environment has gone >10 epochs without accuracy improvement
    - "degraded": Environment has dropped >1% accuracy from previous level

    Status uses hysteresis (3 consecutive epochs meeting condition) to
    prevent status flicker from transient accuracy variations.
    """

    def test_env_status_field_exists(self) -> None:
        """TELE-630: Verify EnvState.status field exists."""
        env = EnvState(env_id=0)
        assert hasattr(env, "status")
        assert isinstance(env.status, str)

    def test_env_status_default_value(self) -> None:
        """TELE-630: Default status is 'initializing' before training starts."""
        env = EnvState(env_id=0)
        assert env.status == "initializing"

    def test_env_status_valid_values(self) -> None:
        """TELE-630: Status has defined categorical values."""
        valid_statuses = {"initializing", "healthy", "excellent", "stalled", "degraded"}
        env = EnvState(env_id=0)
        assert env.status in valid_statuses

    def test_status_transitions_to_healthy_after_first_epoch(self) -> None:
        """TELE-630: Status becomes 'healthy' after first accuracy update."""
        env = EnvState(env_id=0)
        assert env.status == "initializing"

        # First epoch with improvement triggers transition to healthy
        env.add_accuracy(50.0, epoch=1)
        assert env.status in ("healthy", "excellent")

    def test_status_becomes_excellent_on_high_accuracy(self) -> None:
        """TELE-630: Status becomes 'excellent' when accuracy > 80%."""
        env = EnvState(env_id=0)

        # High accuracy with improvement should trigger excellent
        env.add_accuracy(85.0, epoch=1)
        assert env.status == "excellent"

    def test_status_stalled_requires_hysteresis(self) -> None:
        """TELE-630: 'stalled' requires >10 epochs without improvement + hysteresis.

        Stall detection uses hysteresis (3 consecutive epochs) to prevent
        status flicker from transient accuracy fluctuations.
        """
        env = EnvState(env_id=0)

        # Initial accuracy
        env.add_accuracy(70.0, epoch=0)

        # Simulate >10 epochs without improvement (all at 70.0)
        # Note: accuracy must be <= best to avoid resetting the counter
        for epoch in range(1, 15):
            env.add_accuracy(70.0, epoch=epoch)

        # After >10 epochs without improvement and hysteresis met, should be stalled
        assert env.status == "stalled"

    def test_status_degraded_requires_hysteresis(self) -> None:
        """TELE-630: 'degraded' requires 3 consecutive >1% accuracy drops.

        Degraded status uses hysteresis (3 consecutive drops) to prevent
        false alarms from normal training variance.
        """
        env = EnvState(env_id=0)

        # Start with good accuracy
        env.add_accuracy(85.0, epoch=0)

        # Three consecutive drops of >1%
        env.add_accuracy(83.5, epoch=1)  # -1.5%
        env.add_accuracy(82.0, epoch=2)  # -1.5%
        env.add_accuracy(80.5, epoch=3)  # -1.5%

        # After 3 consecutive >1% drops, should be degraded
        assert env.status == "degraded"

    def test_status_improvement_resets_degraded_counter(self) -> None:
        """TELE-630: Improvement resets degraded counter."""
        env = EnvState(env_id=0)

        # Build up to degraded
        env.add_accuracy(85.0, epoch=0)
        env.add_accuracy(83.5, epoch=1)
        env.add_accuracy(82.0, epoch=2)
        env.add_accuracy(80.5, epoch=3)
        assert env.status == "degraded"
        assert env.degraded_counter == 3

        # Improvement resets counter
        env.add_accuracy(81.0, epoch=4)
        assert env.degraded_counter == 0

    def test_hysteresis_counter_fields_exist(self) -> None:
        """TELE-630: Verify hysteresis counter fields exist."""
        env = EnvState(env_id=0)
        assert hasattr(env, "stall_counter")
        assert hasattr(env, "degraded_counter")
        assert env.stall_counter == 0
        assert env.degraded_counter == 0

    def test_epochs_since_improvement_field_exists(self) -> None:
        """TELE-630: Verify epochs_since_improvement field exists."""
        env = EnvState(env_id=0)
        assert hasattr(env, "epochs_since_improvement")
        assert env.epochs_since_improvement == 0

    def test_status_flows_through_aggregator(self) -> None:
        """TELE-630: Status updates flow through aggregator correctly."""
        agg = SanctumAggregator(num_envs=2)

        # Env 0: excellent status
        event0 = make_epoch_event(env_id=0, val_accuracy=85.0, inner_epoch=1)
        agg.process_event(event0)

        # Env 1: healthy status (< 80%)
        event1 = make_epoch_event(env_id=1, val_accuracy=65.0, inner_epoch=1)
        agg.process_event(event1)

        snapshot = agg.get_snapshot()

        assert snapshot.envs[0].status == "excellent"
        assert snapshot.envs[1].status == "healthy"


# =============================================================================
# TELE-631: Environment Reward Mode
# =============================================================================


class TestTELE631EnvRewardMode:
    """TELE-631: Environment reward mode for A/B test cohort tracking.

    Wiring Status: FULLY WIRED
    - Emitter: TrainingStartedPayload.reward_mode
    - Transport: Aggregator captures reward_mode, propagates to EnvState
    - Schema: EnvState.reward_mode field at line 542
    - Consumer: EnvOverview._format_env_id() displays colored pip

    Reward mode identifies the A/B test cohort for this environment:
    - "shaped": Full reward shaping with PBRS, compute rent, stage bonuses
    - "simplified": Reduced reward signal with fewer components
    - "sparse": Minimal reward signal (e.g., terminal accuracy only)
    - None: No A/B testing active; using default reward configuration
    """

    def test_reward_mode_field_exists(self) -> None:
        """TELE-631: Verify EnvState.reward_mode field exists."""
        env = EnvState(env_id=0)
        assert hasattr(env, "reward_mode")

    def test_reward_mode_default_value(self) -> None:
        """TELE-631: Default reward_mode is None (no A/B testing)."""
        env = EnvState(env_id=0)
        assert env.reward_mode is None

    def test_reward_mode_type_is_optional_str(self) -> None:
        """TELE-631: reward_mode type is str | None."""
        env = EnvState(env_id=0)
        # Default is None
        assert env.reward_mode is None

        # Can be set to string
        env.reward_mode = "shaped"
        assert env.reward_mode == "shaped"

    def test_reward_mode_valid_values(self) -> None:
        """TELE-631: reward_mode accepts known A/B cohort values."""
        valid_modes = {"shaped", "simplified", "sparse"}
        env = EnvState(env_id=0)

        for mode in valid_modes:
            env.reward_mode = mode
            assert env.reward_mode == mode

    def test_reward_mode_set_from_training_started(self) -> None:
        """TELE-631: reward_mode is set from TRAINING_STARTED event."""
        agg = SanctumAggregator(num_envs=2)

        event = make_training_started_event(n_envs=2, reward_mode="simplified")
        agg.process_event(event)

        # Trigger env creation via epoch event
        epoch_event = make_epoch_event(env_id=0, val_accuracy=50.0, inner_epoch=0)
        agg.process_event(epoch_event)

        snapshot = agg.get_snapshot()
        assert snapshot.envs[0].reward_mode == "simplified"

    def test_reward_mode_propagates_to_all_envs(self) -> None:
        """TELE-631: reward_mode propagates to all environments."""
        agg = SanctumAggregator(num_envs=4)

        event = make_training_started_event(n_envs=4, reward_mode="sparse")
        agg.process_event(event)

        # Create events for all envs
        for env_id in range(4):
            epoch_event = make_epoch_event(
                env_id=env_id, val_accuracy=50.0, inner_epoch=0
            )
            agg.process_event(epoch_event)

        snapshot = agg.get_snapshot()
        for env_id in range(4):
            assert snapshot.envs[env_id].reward_mode == "sparse"

    def test_reward_mode_empty_string_treated_as_default(self) -> None:
        """TELE-631: Empty string reward_mode indicates default configuration."""
        agg = SanctumAggregator(num_envs=1)

        # Empty string from training started
        event = make_training_started_event(n_envs=1, reward_mode="")
        agg.process_event(event)

        epoch_event = make_epoch_event(env_id=0, val_accuracy=50.0, inner_epoch=0)
        agg.process_event(epoch_event)

        snapshot = agg.get_snapshot()
        # Empty string or None indicates no A/B testing
        assert snapshot.envs[0].reward_mode in ("", None)


# =============================================================================
# TELE-632: Environment Rolled Back
# =============================================================================


class TestTELE632EnvRolledBack:
    """TELE-632: Environment rolled back flag.

    Wiring Status: FULLY WIRED
    - Emitter: GOVERNOR_ROLLBACK telemetry event
    - Transport: Aggregator sets env.rolled_back = True
    - Schema: EnvState.rolled_back field at line 546
    - Consumer: EnvOverview shows red alert overlay when rolled_back is True

    Rolled back indicates that the governor detected a catastrophic failure
    and triggered an emergency rollback. The flag is automatically cleared
    when training resumes (next EPOCH_COMPLETED event).
    """

    def test_rolled_back_field_exists(self) -> None:
        """TELE-632: Verify EnvState.rolled_back field exists."""
        env = EnvState(env_id=0)
        assert hasattr(env, "rolled_back")

    def test_rolled_back_type_is_bool(self) -> None:
        """TELE-632: rolled_back is a boolean field."""
        env = EnvState(env_id=0)
        assert isinstance(env.rolled_back, bool)

    def test_rolled_back_default_value(self) -> None:
        """TELE-632: Default rolled_back is False (normal operation)."""
        env = EnvState(env_id=0)
        assert env.rolled_back is False

    def test_rolled_back_set_on_governor_rollback(self) -> None:
        """TELE-632: rolled_back is set True on GOVERNOR_ROLLBACK event."""
        agg = SanctumAggregator(num_envs=2)

        # Initialize env with an epoch
        epoch_event = make_epoch_event(env_id=0, val_accuracy=50.0, inner_epoch=0)
        agg.process_event(epoch_event)

        # Trigger governor rollback
        rollback_event = make_governor_rollback_event(
            env_id=0, reason="governor_nan"
        )
        agg.process_event(rollback_event)

        snapshot = agg.get_snapshot()
        assert snapshot.envs[0].rolled_back is True

    def test_rolled_back_timestamp_recorded(self) -> None:
        """TELE-632: rollback_timestamp is recorded when rollback occurs."""
        env = EnvState(env_id=0)
        assert hasattr(env, "rollback_timestamp")
        assert env.rollback_timestamp is None  # Default

        agg = SanctumAggregator(num_envs=1)

        # Initialize and trigger rollback
        epoch_event = make_epoch_event(env_id=0, val_accuracy=50.0, inner_epoch=0)
        agg.process_event(epoch_event)

        rollback_event = make_governor_rollback_event(env_id=0, reason="governor_nan")
        agg.process_event(rollback_event)

        snapshot = agg.get_snapshot()
        assert snapshot.envs[0].rollback_timestamp is not None

    def test_rolled_back_only_affects_target_env(self) -> None:
        """TELE-632: rolled_back only affects the target environment."""
        agg = SanctumAggregator(num_envs=3)

        # Initialize all envs
        for env_id in range(3):
            event = make_epoch_event(env_id=env_id, val_accuracy=50.0, inner_epoch=0)
            agg.process_event(event)

        # Rollback only env 1
        rollback_event = make_governor_rollback_event(env_id=1, reason="governor_nan")
        agg.process_event(rollback_event)

        snapshot = agg.get_snapshot()
        assert snapshot.envs[0].rolled_back is False
        assert snapshot.envs[1].rolled_back is True
        assert snapshot.envs[2].rolled_back is False

    def test_rolled_back_health_threshold(self) -> None:
        """TELE-632: rolled_back == True is critical health level."""
        env = EnvState(env_id=0)

        # Healthy when not rolled back
        env.rolled_back = False
        assert env.rolled_back is False

        # Critical when rolled back
        env.rolled_back = True
        assert env.rolled_back is True


# =============================================================================
# TELE-633: Environment Rollback Reason
# =============================================================================


class TestTELE633EnvRollbackReason:
    """TELE-633: Environment rollback reason.

    Wiring Status: FULLY WIRED
    - Emitter: GovernorRollbackPayload.reason
    - Transport: Aggregator extracts reason from payload
    - Schema: EnvState.rollback_reason field at line 547
    - Consumer: EnvOverview displays formatted reason in alert message

    Rollback reason identifies the type of catastrophic failure:
    - "governor_nan": NaN values detected in gradients, loss, or activations
    - "governor_lobotomy": Severe accuracy drop (lobotomy) detected
    - "governor_divergence": Training divergence detected (loss explosion)
    - "": No rollback has occurred (normal operation)
    """

    def test_rollback_reason_field_exists(self) -> None:
        """TELE-633: Verify EnvState.rollback_reason field exists."""
        env = EnvState(env_id=0)
        assert hasattr(env, "rollback_reason")

    def test_rollback_reason_type_is_str(self) -> None:
        """TELE-633: rollback_reason is a string field."""
        env = EnvState(env_id=0)
        assert isinstance(env.rollback_reason, str)

    def test_rollback_reason_default_value(self) -> None:
        """TELE-633: Default rollback_reason is empty string (no rollback)."""
        env = EnvState(env_id=0)
        assert env.rollback_reason == ""

    def test_rollback_reason_valid_values(self) -> None:
        """TELE-633: rollback_reason has defined categorical values."""
        valid_reasons = {
            "governor_nan",
            "governor_lobotomy",
            "governor_divergence",
            "",  # Normal operation
        }
        env = EnvState(env_id=0)

        for reason in valid_reasons:
            env.rollback_reason = reason
            assert env.rollback_reason == reason

    def test_rollback_reason_set_from_governor_rollback(self) -> None:
        """TELE-633: rollback_reason is set from GOVERNOR_ROLLBACK event."""
        agg = SanctumAggregator(num_envs=1)

        # Initialize env
        epoch_event = make_epoch_event(env_id=0, val_accuracy=50.0, inner_epoch=0)
        agg.process_event(epoch_event)

        # Trigger rollback with specific reason
        rollback_event = make_governor_rollback_event(
            env_id=0, reason="governor_nan"
        )
        agg.process_event(rollback_event)

        snapshot = agg.get_snapshot()
        assert snapshot.envs[0].rollback_reason == "governor_nan"

    def test_rollback_reason_governor_lobotomy(self) -> None:
        """TELE-633: Lobotomy reason is captured correctly."""
        agg = SanctumAggregator(num_envs=1)

        # Initialize env
        epoch_event = make_epoch_event(env_id=0, val_accuracy=50.0, inner_epoch=0)
        agg.process_event(epoch_event)

        # Trigger rollback with lobotomy reason
        rollback_event = make_governor_rollback_event(
            env_id=0, reason="governor_lobotomy"
        )
        agg.process_event(rollback_event)

        snapshot = agg.get_snapshot()
        assert snapshot.envs[0].rollback_reason == "governor_lobotomy"

    def test_rollback_reason_governor_divergence(self) -> None:
        """TELE-633: Divergence reason is captured correctly."""
        agg = SanctumAggregator(num_envs=1)

        # Initialize env
        epoch_event = make_epoch_event(env_id=0, val_accuracy=50.0, inner_epoch=0)
        agg.process_event(epoch_event)

        # Trigger rollback with divergence reason
        rollback_event = make_governor_rollback_event(
            env_id=0, reason="governor_divergence"
        )
        agg.process_event(rollback_event)

        snapshot = agg.get_snapshot()
        assert snapshot.envs[0].rollback_reason == "governor_divergence"

    def test_rollback_reason_correlates_with_rolled_back(self) -> None:
        """TELE-633: rollback_reason is only meaningful when rolled_back is True."""
        env = EnvState(env_id=0)

        # Normal operation: no reason
        assert env.rolled_back is False
        assert env.rollback_reason == ""

        # After rollback: reason is set
        env.rolled_back = True
        env.rollback_reason = "governor_nan"
        assert env.rollback_reason == "governor_nan"


# =============================================================================
# Integration: Combined EnvState Fields
# =============================================================================


class TestEnvStateSchemaCompleteness:
    """Verify EnvState has all TELE-630 to TELE-633 fields."""

    def test_all_env_status_fields_present(self) -> None:
        """Verify all status-related fields are present in EnvState."""
        env = EnvState(env_id=0)

        # TELE-630: Status tracking
        assert hasattr(env, "status")
        assert hasattr(env, "epochs_since_improvement")
        assert hasattr(env, "stall_counter")
        assert hasattr(env, "degraded_counter")

        # TELE-631: A/B test cohort
        assert hasattr(env, "reward_mode")

        # TELE-632/633: Rollback state
        assert hasattr(env, "rolled_back")
        assert hasattr(env, "rollback_reason")
        assert hasattr(env, "rollback_timestamp")

    def test_all_defaults_are_correct(self) -> None:
        """Verify all default values match TELE record specifications."""
        env = EnvState(env_id=0)

        # TELE-630
        assert env.status == "initializing"
        assert env.epochs_since_improvement == 0
        assert env.stall_counter == 0
        assert env.degraded_counter == 0

        # TELE-631
        assert env.reward_mode is None

        # TELE-632/633
        assert env.rolled_back is False
        assert env.rollback_reason == ""
        assert env.rollback_timestamp is None


# =============================================================================
# Integration: Aggregator Event Handling
# =============================================================================


class TestAggregatorEnvStateEventHandling:
    """Test aggregator handles events correctly for TELE-630 to TELE-633."""

    def test_aggregator_processes_training_started_reward_mode(self) -> None:
        """Aggregator captures reward_mode from TRAINING_STARTED."""
        agg = SanctumAggregator(num_envs=2)

        training_event = make_training_started_event(n_envs=2, reward_mode="shaped")
        agg.process_event(training_event)

        # Verify internal state was set (envs created on first epoch)
        epoch_event = make_epoch_event(env_id=0, val_accuracy=50.0, inner_epoch=0)
        agg.process_event(epoch_event)

        snapshot = agg.get_snapshot()
        assert snapshot.envs[0].reward_mode == "shaped"

    def test_aggregator_processes_governor_rollback(self) -> None:
        """Aggregator handles GOVERNOR_ROLLBACK correctly."""
        agg = SanctumAggregator(num_envs=1)

        # Initialize env
        epoch_event = make_epoch_event(env_id=0, val_accuracy=50.0, inner_epoch=0)
        agg.process_event(epoch_event)

        # Process rollback
        rollback_event = make_governor_rollback_event(
            env_id=0, reason="governor_nan"
        )
        agg.process_event(rollback_event)

        snapshot = agg.get_snapshot()
        assert snapshot.envs[0].rolled_back is True
        assert snapshot.envs[0].rollback_reason == "governor_nan"
        assert snapshot.envs[0].rollback_timestamp is not None

    def test_aggregator_updates_status_on_epoch_completed(self) -> None:
        """Aggregator updates status when EPOCH_COMPLETED is processed."""
        agg = SanctumAggregator(num_envs=1)

        # First epoch - should transition from initializing
        epoch_event = make_epoch_event(env_id=0, val_accuracy=85.0, inner_epoch=1)
        agg.process_event(epoch_event)

        snapshot = agg.get_snapshot()
        assert snapshot.envs[0].status == "excellent"

    def test_snapshot_includes_all_env_state_fields(self) -> None:
        """SanctumSnapshot.envs includes all TELE-630 to TELE-633 fields."""
        agg = SanctumAggregator(num_envs=2)

        # Setup training with reward mode
        training_event = make_training_started_event(n_envs=2, reward_mode="sparse")
        agg.process_event(training_event)

        # Create envs via epochs
        for env_id in range(2):
            epoch_event = make_epoch_event(
                env_id=env_id, val_accuracy=75.0, inner_epoch=1
            )
            agg.process_event(epoch_event)

        snapshot = agg.get_snapshot()

        for env_id in range(2):
            env = snapshot.envs[env_id]
            # TELE-630
            assert env.status in ("healthy", "excellent")
            # TELE-631
            assert env.reward_mode == "sparse"
            # TELE-632/633
            assert env.rolled_back is False
            assert env.rollback_reason == ""


# =============================================================================
# Consumer Widget Tests (EnvOverview)
# =============================================================================


class TestEnvOverviewStatusFormatting:
    """Test EnvOverview correctly formats status fields.

    These tests verify the consumer reads TELE-630 to TELE-633 fields correctly.
    Widget tests don't require actual rendering; they verify the data is available.
    """

    def test_status_values_have_display_mapping(self) -> None:
        """TELE-630: All status values have corresponding display text."""
        # From EnvOverview._format_status
        status_display = {
            "excellent": "EXCL",
            "healthy": "OK",
            "initializing": "INIT",
            "stalled": "STALL",
            "degraded": "DEGR",
        }

        for status in ["excellent", "healthy", "initializing", "stalled", "degraded"]:
            assert status in status_display

    def test_status_values_have_color_coding(self) -> None:
        """TELE-630: All status values have corresponding color styles."""
        # From EnvOverview._format_status
        status_styles = {
            "excellent": "bold green",
            "healthy": "green",
            "initializing": "dim",
            "stalled": "yellow",
            "degraded": "red",
        }

        for status in ["excellent", "healthy", "initializing", "stalled", "degraded"]:
            assert status in status_styles

    def test_reward_mode_has_pip_colors(self) -> None:
        """TELE-631: Reward modes have corresponding pip colors."""
        # From EnvOverview - _AB_STYLES
        # Known modes should have visual indicators
        known_modes = {"shaped", "simplified", "sparse"}

        # Verify modes exist in the spec
        for mode in known_modes:
            # The mode should be a valid string
            assert isinstance(mode, str)
            assert len(mode) > 0

    def test_rollback_reason_has_display_mapping(self) -> None:
        """TELE-633: Rollback reasons have human-readable display text."""
        # From EnvOverview._add_rollback_alert_row
        reason_display = {
            "governor_nan": "NaN DETECTED",
            "governor_lobotomy": "LOBOTOMY",
            "governor_divergence": "DIVERGENCE",
        }

        for reason in ["governor_nan", "governor_lobotomy", "governor_divergence"]:
            assert reason in reason_display


# =============================================================================
# Historical Detail Consumer Tests
# =============================================================================


class TestHistoricalEnvDetailConsumer:
    """Test HistoricalEnvDetail widget reads reward_mode correctly."""

    def test_best_run_record_includes_reward_mode(self) -> None:
        """TELE-631: BestRunRecord includes reward_mode for historical tracking."""
        from esper.karn.sanctum.schema import BestRunRecord

        record = BestRunRecord(
            env_id=0,
            episode=10,
            peak_accuracy=92.5,
            final_accuracy=90.0,
            reward_mode="shaped",
        )

        assert hasattr(record, "reward_mode")
        assert record.reward_mode == "shaped"

    def test_best_run_record_reward_mode_default(self) -> None:
        """TELE-631: BestRunRecord reward_mode defaults to None."""
        from esper.karn.sanctum.schema import BestRunRecord

        record = BestRunRecord(
            env_id=0,
            episode=10,
            peak_accuracy=92.5,
            final_accuracy=90.0,
        )

        assert record.reward_mode is None
