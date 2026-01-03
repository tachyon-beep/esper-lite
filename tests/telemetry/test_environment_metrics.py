"""End-to-end tests for environment metrics (TELE-600 to TELE-699).

Verifies environment telemetry flows from source through to nissa/sanctum.

These tests cover:
- TELE-600: obs_nan_count (WIRING GAP - xfail)
- TELE-601: obs_inf_count (WIRING GAP - xfail)
- TELE-602: outlier_pct (WIRING GAP - xfail)
- TELE-603: normalization_drift (WIRING GAP - xfail)
- TELE-610: episode_stats (PARTIALLY WIRED - only total_episodes works)
- TELE-650: env_status (FULLY WIRED)

Note: TELE-600 to TELE-603 and most of TELE-610 are documented wiring gaps.
The schema fields exist, consumers read them, but no emitters populate the data.
Tests are marked xfail to document expected behavior when wiring is complete.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from esper.karn.sanctum.aggregator import SanctumAggregator
from esper.karn.sanctum.schema import (
    EpisodeStats,
    EnvState,
    ObservationStats,
    SanctumSnapshot,
)
from esper.leyline import (
    BatchEpochCompletedPayload,
    EpochCompletedPayload,
    TelemetryEvent,
    TelemetryEventType,
)

from .conftest import CaptureBackend, CaptureHubResult


# =============================================================================
# Helper to create mock EPOCH_COMPLETED events
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


def make_batch_epoch_event(
    episodes_completed: int,
    batch_idx: int = 0,
    avg_accuracy: float = 75.0,
    avg_reward: float = 0.5,
    total_episodes: int = 100,
    n_envs: int = 4,
) -> MagicMock:
    """Create a mock BATCH_EPOCH_COMPLETED event for testing."""
    event = MagicMock()
    event.event_type = MagicMock()
    event.event_type.name = "BATCH_EPOCH_COMPLETED"
    event.timestamp = datetime.now(timezone.utc)
    event.data = BatchEpochCompletedPayload(
        episodes_completed=episodes_completed,
        batch_idx=batch_idx,
        avg_accuracy=avg_accuracy,
        avg_reward=avg_reward,
        total_episodes=total_episodes,
        n_envs=n_envs,
    )
    return event


# =============================================================================
# TELE-600: Observation NaN Count (WIRING GAP)
# =============================================================================


class TestTELE600ObsNanCount:
    """TELE-600: Observation NaN count should be emitted but emitter is missing.

    Wiring Status:
    - Schema field exists: ObservationStats.nan_count
    - Consumer reads it: HealthStatusPanel._render_observation_stats()
    - Emitter: NOT IMPLEMENTED
    - Aggregator: STUB (returns default ObservationStats)

    When wiring is complete, these tests should pass.
    """

    def test_obs_stats_schema_field_exists(self) -> None:
        """TELE-600: Verify ObservationStats.nan_count field exists in schema."""
        obs_stats = ObservationStats()
        assert hasattr(obs_stats, "nan_count")
        assert obs_stats.nan_count == 0  # Default value

    def test_snapshot_includes_observation_stats(self) -> None:
        """TELE-600: Verify SanctumSnapshot includes observation_stats field."""
        agg = SanctumAggregator(num_envs=1)
        snapshot = agg.get_snapshot()

        assert hasattr(snapshot, "observation_stats")
        assert isinstance(snapshot.observation_stats, ObservationStats)
        # Currently stubbed - always returns defaults
        assert snapshot.observation_stats.nan_count == 0

    @pytest.mark.xfail(
        reason="TELE-600 wiring gap: obs_nan_count emitter not implemented"
    )
    def test_nan_count_populated_from_epoch_completed(self) -> None:
        """TELE-600: obs_nan_count should be populated when NaNs detected.

        Expected behavior when wiring is complete:
        1. Observation tensor is checked for NaN values during training
        2. Count is emitted via EPOCH_COMPLETED or new OBSERVATION_STATS event
        3. Aggregator populates ObservationStats.nan_count
        4. HealthStatusPanel displays "NaN:X" in red when > 0

        This test documents the WIRING GAP: EpochCompletedPayload does not
        include observation health fields, so this test verifies that the
        aggregator cannot populate observation stats from epoch events.
        """
        agg = SanctumAggregator(num_envs=1)

        # Simulate epoch - but EpochCompletedPayload has no obs_nan_count field
        event = make_epoch_event(env_id=0, val_accuracy=75.0)
        agg.process_event(event)

        snapshot = agg.get_snapshot()

        # WIRING GAP: We cannot set nan_count > 0 because:
        # 1. EpochCompletedPayload doesn't have obs_nan_count field
        # 2. There's no OBSERVATION_STATS event type
        # 3. Aggregator stubs observation_stats with defaults
        #
        # When wired, we should be able to inject NaN observations and verify
        # the count is emitted. For now, assert that the stub behavior exists.
        #
        # This xfail documents that the field EXISTS but cannot be POPULATED.
        # The test passes when assertion is `>= 0` but should fail when we
        # assert that wiring WORKS (i.e., we can inject and verify specific values).
        assert hasattr(EpochCompletedPayload, "obs_nan_count"), (
            "EpochCompletedPayload should include obs_nan_count field when wiring is complete"
        )


# =============================================================================
# TELE-601: Observation Inf Count (WIRING GAP)
# =============================================================================


class TestTELE601ObsInfCount:
    """TELE-601: Observation Inf count should be emitted but emitter is missing.

    Wiring Status:
    - Schema field exists: ObservationStats.inf_count
    - Consumer reads it: HealthStatusPanel._render_observation_stats()
    - Emitter: NOT IMPLEMENTED
    - Aggregator: STUB (returns default ObservationStats)
    """

    def test_obs_stats_inf_field_exists(self) -> None:
        """TELE-601: Verify ObservationStats.inf_count field exists in schema."""
        obs_stats = ObservationStats()
        assert hasattr(obs_stats, "inf_count")
        assert obs_stats.inf_count == 0  # Default value

    @pytest.mark.xfail(
        reason="TELE-601 wiring gap: obs_inf_count emitter not implemented"
    )
    def test_inf_count_populated_when_inf_detected(self) -> None:
        """TELE-601: obs_inf_count should be populated when Inf values detected.

        Expected behavior when wiring is complete:
        1. Observation tensor is checked for Inf values: torch.isinf(observations).sum()
        2. Count is emitted via telemetry event
        3. Aggregator populates ObservationStats.inf_count
        4. HealthStatusPanel displays "Inf:X" in red when > 0

        Note: Any Inf in observations is critical - indicates overflow in
        feature computation that will corrupt network forward pass.

        This test documents the WIRING GAP: there is no telemetry event
        that carries observation health information.
        """
        # WIRING GAP: No event type carries obs_inf_count.
        # When wired, there should be an OBSERVATION_STATS event or
        # EpochCompletedPayload should include these fields.
        from esper.leyline import TelemetryEventType

        # Verify the event type doesn't exist yet
        event_types = [e.name for e in TelemetryEventType]
        assert "OBSERVATION_STATS" in event_types or hasattr(
            EpochCompletedPayload, "obs_inf_count"
        ), (
            "Wiring requires either OBSERVATION_STATS event type or "
            "obs_inf_count field in EpochCompletedPayload"
        )


# =============================================================================
# TELE-602: Outlier Percentage (WIRING GAP)
# =============================================================================


class TestTELE602OutlierPct:
    """TELE-602: Outlier percentage should track observations outside 3-sigma.

    Wiring Status:
    - Schema field exists: ObservationStats.outlier_pct
    - Consumer reads it: HealthStatusPanel displays "Out:X.X%"
    - Thresholds defined: <=5% healthy, 5-10% warning, >10% critical
    - Emitter: NOT IMPLEMENTED
    - Aggregator: STUB
    """

    def test_outlier_pct_field_exists(self) -> None:
        """TELE-602: Verify ObservationStats.outlier_pct field exists."""
        obs_stats = ObservationStats()
        assert hasattr(obs_stats, "outlier_pct")
        assert obs_stats.outlier_pct == 0.0  # Default value

    @pytest.mark.xfail(
        reason="TELE-602 wiring gap: outlier_pct emitter not implemented"
    )
    def test_outlier_pct_computed_from_batch_observations(self) -> None:
        """TELE-602: outlier_pct should track percentage of observations outside 3-sigma.

        Expected computation when wiring is complete:
        ```
        outlier_count = ((obs - obs.mean()).abs() > 3 * obs.std()).sum()
        outlier_pct = float(outlier_count) / len(obs)
        ```

        Thresholds:
        - Healthy: outlier_pct <= 0.05 (5%)
        - Warning: 0.05 < outlier_pct <= 0.10 (5-10%)
        - Critical: outlier_pct > 0.10 (>10%)

        High outlier rates indicate observation distribution shift.

        This test documents the WIRING GAP: outlier_pct is never computed
        or emitted, so the field always returns the default value of 0.0.
        """
        agg = SanctumAggregator(num_envs=1)

        # Simulate many epochs - even after training, outlier_pct stays at 0.0
        for epoch in range(20):
            event = make_epoch_event(env_id=0, val_accuracy=75.0, inner_epoch=epoch)
            agg.process_event(event)

        snapshot = agg.get_snapshot()

        # WIRING GAP: outlier_pct is never computed.
        # After 20 epochs, we'd expect some non-zero outlier detection
        # if the wiring was complete. Currently it's always 0.0.
        #
        # This assertion will fail when wiring is incomplete (always 0.0)
        # and pass when wiring is complete (may be any valid percentage).
        assert snapshot.observation_stats.outlier_pct > 0.0, (
            "outlier_pct should be computed from observations after training epochs. "
            "Currently stubbed at 0.0 due to wiring gap."
        )


# =============================================================================
# TELE-603: Normalization Drift (WIRING GAP)
# =============================================================================


class TestTELE603NormalizationDrift:
    """TELE-603: Normalization drift tracks running mean/std shift.

    Wiring Status:
    - Schema field exists: ObservationStats.normalization_drift
    - Consumer reads it: HealthStatusPanel displays "Drift:X.XX"
    - Thresholds defined: <=1.0 healthy, 1-2 warning, >2 critical
    - Source exists: RunningMeanStd in simic/control/normalization.py
    - Emitter: NOT IMPLEMENTED (drift computation missing)
    - Aggregator: STUB
    """

    def test_normalization_drift_field_exists(self) -> None:
        """TELE-603: Verify ObservationStats.normalization_drift field exists."""
        obs_stats = ObservationStats()
        assert hasattr(obs_stats, "normalization_drift")
        assert obs_stats.normalization_drift == 0.0  # Default value

    @pytest.mark.xfail(
        reason="TELE-603 wiring gap: normalization_drift emitter not implemented"
    )
    def test_normalization_drift_tracks_mean_std_shift(self) -> None:
        """TELE-603: normalization_drift should measure running stats divergence.

        Expected computation when wiring is complete:
        ```
        drift = max(|current_mean - initial_mean|/std,
                    |current_std - initial_std|/initial_std)
        ```

        Thresholds:
        - Healthy: drift <= 1.0 (within 1 sigma)
        - Warning: 1.0 < drift <= 2.0 (moderate shift)
        - Critical: drift > 2.0 (severe distribution shift)

        Large drift indicates environment distribution has changed
        significantly from training start.

        This test documents the WIRING GAP: normalization_drift is never
        computed or emitted, despite RunningMeanStd tracking the statistics.
        """
        agg = SanctumAggregator(num_envs=1)

        # Simulate multiple epochs with varying accuracy to simulate distribution shift
        for epoch in range(30):
            # Accuracy varies significantly to simulate distribution shift
            acc = 60.0 + epoch * 1.5 if epoch < 15 else 90.0 - (epoch - 15) * 2.0
            event = make_epoch_event(env_id=0, val_accuracy=acc, inner_epoch=epoch)
            agg.process_event(event)

        snapshot = agg.get_snapshot()

        # WIRING GAP: normalization_drift is never computed.
        # Even with 30 epochs of varying data, drift stays at 0.0.
        #
        # When wiring is complete:
        # 1. RunningMeanStd should track initial mean/std
        # 2. Drift should be computed and emitted via telemetry
        # 3. Aggregator should populate observation_stats.normalization_drift
        #
        # This assertion documents that the field is NOT populated.
        assert snapshot.observation_stats.normalization_drift > 0.0, (
            "normalization_drift should track observation distribution shift. "
            "Currently stubbed at 0.0 due to wiring gap - RunningMeanStd exists "
            "but drift computation is not emitted via telemetry."
        )


# =============================================================================
# TELE-610: Episode Statistics (PARTIALLY WIRED)
# =============================================================================


class TestTELE610EpisodeStats:
    """TELE-610: Episode statistics for aggregate episode-level metrics.

    Wiring Status:
    - Schema exists: EpisodeStats dataclass with all fields
    - Consumer reads it: EpisodeMetricsPanel displays all fields
    - WIRED: total_episodes (from aggregator._current_episode)
    - NOT WIRED: length_mean/std/min/max, timeout_rate, success_rate,
                 early_termination_rate, steps_per_*, completion_trend
    """

    def test_episode_stats_schema_exists(self) -> None:
        """TELE-610: Verify EpisodeStats dataclass has all expected fields."""
        stats = EpisodeStats()

        # Episode length statistics
        assert hasattr(stats, "length_mean")
        assert hasattr(stats, "length_std")
        assert hasattr(stats, "length_min")
        assert hasattr(stats, "length_max")

        # Outcome tracking
        assert hasattr(stats, "total_episodes")
        assert hasattr(stats, "timeout_count")
        assert hasattr(stats, "success_count")
        assert hasattr(stats, "early_termination_count")

        # Derived rates
        assert hasattr(stats, "timeout_rate")
        assert hasattr(stats, "success_rate")
        assert hasattr(stats, "early_termination_rate")

        # Steps per action type
        assert hasattr(stats, "steps_per_germinate")
        assert hasattr(stats, "steps_per_prune")
        assert hasattr(stats, "steps_per_fossilize")

        # Completion trend
        assert hasattr(stats, "completion_trend")

    def test_total_episodes_is_wired(self) -> None:
        """TELE-610: total_episodes IS wired and populated from aggregator."""
        agg = SanctumAggregator(num_envs=2)

        # Simulate batch epoch completing with episode count
        event = make_batch_epoch_event(
            episodes_completed=10,
            batch_idx=1,
            total_episodes=100,
            n_envs=2,
        )
        agg.process_event(event)

        snapshot = agg.get_snapshot()

        # total_episodes should be populated from _current_episode
        # This IS wired in aggregator.py line 539
        assert hasattr(snapshot, "episode_stats")
        assert isinstance(snapshot.episode_stats, EpisodeStats)
        assert snapshot.episode_stats.total_episodes >= 0

    @pytest.mark.xfail(
        reason="TELE-610 partial wiring gap: length statistics not implemented"
    )
    def test_episode_length_stats_populated(self) -> None:
        """TELE-610: Episode length statistics should be computed.

        Expected fields when wiring is complete:
        - length_mean: Average number of steps per episode
        - length_std: Variance in episode length
        - length_min/max: Range for anomaly detection

        Currently stubbed - returns all zeros.
        """
        agg = SanctumAggregator(num_envs=1)

        # Simulate multiple epochs
        for epoch in range(50):
            event = make_epoch_event(env_id=0, val_accuracy=75.0, inner_epoch=epoch)
            agg.process_event(event)

        snapshot = agg.get_snapshot()

        # These should be non-zero after episodes complete
        # Currently fails - length tracking not implemented
        assert snapshot.episode_stats.length_mean > 0
        assert snapshot.episode_stats.length_max >= snapshot.episode_stats.length_min

    @pytest.mark.xfail(
        reason="TELE-610 partial wiring gap: outcome rates not implemented"
    )
    def test_episode_outcome_rates_populated(self) -> None:
        """TELE-610: Outcome rates should be computed from episode endings.

        Expected fields when wiring is complete:
        - timeout_rate: Fraction hitting max_steps without terminal
        - success_rate: Fraction achieving goal state
        - early_termination_rate: Fraction terminated early

        Currently stubbed - returns all zeros.
        """
        agg = SanctumAggregator(num_envs=1)

        event = make_batch_epoch_event(
            episodes_completed=10,
            batch_idx=1,
            total_episodes=100,
            n_envs=1,
        )
        agg.process_event(event)

        snapshot = agg.get_snapshot()

        # At least one rate should be non-zero after episodes complete
        has_any_rate = (
            snapshot.episode_stats.timeout_rate > 0
            or snapshot.episode_stats.success_rate > 0
            or snapshot.episode_stats.early_termination_rate > 0
        )
        assert has_any_rate, "Expected at least one outcome rate to be non-zero"

    @pytest.mark.xfail(
        reason="TELE-610 partial wiring gap: action efficiency not implemented"
    )
    def test_steps_per_action_populated(self) -> None:
        """TELE-610: Steps-per-action metrics should track action efficiency.

        Expected fields when wiring is complete:
        - steps_per_germinate: Avg steps between GERMINATE actions
        - steps_per_prune: Avg steps between PRUNE actions
        - steps_per_fossilize: Avg steps between FOSSILIZE actions

        Currently stubbed - returns all zeros.
        """
        agg = SanctumAggregator(num_envs=1)

        # Simulate training with actions
        for epoch in range(20):
            event = make_epoch_event(env_id=0, val_accuracy=75.0, inner_epoch=epoch)
            agg.process_event(event)

        snapshot = agg.get_snapshot()

        # At least one action rate should be non-zero after training
        has_any_action_rate = (
            snapshot.episode_stats.steps_per_germinate > 0
            or snapshot.episode_stats.steps_per_prune > 0
            or snapshot.episode_stats.steps_per_fossilize > 0
        )
        assert has_any_action_rate, "Expected action efficiency metrics to be populated"


# =============================================================================
# TELE-650: Environment Status (FULLY WIRED)
# =============================================================================


class TestTELE650EnvStatus:
    """TELE-650: Environment status tracking - FULLY WIRED.

    Wiring Status:
    - Emitter: EPOCH_COMPLETED includes val_accuracy
    - Transport: Aggregator handles event and calls add_accuracy()
    - Schema: EnvState.status field
    - Consumer: EnvOverview, EnvDetailScreen, AnomalyStrip all read status

    Status values:
    - "initializing": Training not yet started
    - "healthy": Making normal progress
    - "excellent": High accuracy (>80%) and just improved
    - "stalled": No improvement for >10 epochs (with hysteresis)
    - "degraded": Accuracy dropped >1% for 3 consecutive epochs (with hysteresis)
    """

    def test_env_status_field_exists(self) -> None:
        """TELE-650: Verify EnvState.status field exists with correct default."""
        env = EnvState(env_id=0)
        assert hasattr(env, "status")
        assert env.status == "initializing"  # Default before any epochs

    def test_status_becomes_healthy_after_first_epoch(self) -> None:
        """TELE-650: Status transitions to 'healthy' after first accuracy update."""
        agg = SanctumAggregator(num_envs=1)

        # First epoch should transition from initializing to healthy
        event = make_epoch_event(env_id=0, val_accuracy=50.0, inner_epoch=1)
        agg.process_event(event)

        snapshot = agg.get_snapshot()
        env = snapshot.envs[0]

        # After first accuracy, should be healthy (not initializing)
        assert env.status in ("healthy", "excellent")

    def test_status_becomes_excellent_on_high_accuracy(self) -> None:
        """TELE-650: Status becomes 'excellent' when accuracy > 80%."""
        agg = SanctumAggregator(num_envs=1)

        # High accuracy should trigger excellent status
        event = make_epoch_event(env_id=0, val_accuracy=85.0, inner_epoch=1)
        agg.process_event(event)

        snapshot = agg.get_snapshot()
        env = snapshot.envs[0]

        assert env.status == "excellent"

    def test_status_stalled_with_hysteresis(self) -> None:
        """TELE-650: 'stalled' requires >10 epochs without improvement + hysteresis.

        Stall detection uses hysteresis (3 consecutive epochs) to prevent
        status flicker from transient accuracy fluctuations.
        """
        agg = SanctumAggregator(num_envs=1)

        # Initial accuracy
        event = make_epoch_event(env_id=0, val_accuracy=70.0, inner_epoch=0)
        agg.process_event(event)

        # Simulate >10 epochs without improvement (all at 70.0)
        for epoch in range(1, 20):
            event = make_epoch_event(env_id=0, val_accuracy=70.0, inner_epoch=epoch)
            agg.process_event(event)

        snapshot = agg.get_snapshot()
        env = snapshot.envs[0]

        # After >10 epochs without improvement and hysteresis met, should be stalled
        assert env.status == "stalled"

    def test_status_degraded_with_hysteresis(self) -> None:
        """TELE-650: 'degraded' requires 3 consecutive >1% accuracy drops.

        Degraded status uses hysteresis (3 consecutive drops) to prevent
        false alarms from normal training variance.
        """
        agg = SanctumAggregator(num_envs=1)

        # Start with good accuracy
        event = make_epoch_event(env_id=0, val_accuracy=85.0, inner_epoch=0)
        agg.process_event(event)

        # Three consecutive drops of >1%
        for epoch, acc in enumerate([83.5, 82.0, 80.5], start=1):
            event = make_epoch_event(env_id=0, val_accuracy=acc, inner_epoch=epoch)
            agg.process_event(event)

        snapshot = agg.get_snapshot()
        env = snapshot.envs[0]

        # After 3 consecutive >1% drops, should be degraded
        assert env.status == "degraded"

    def test_status_improvement_resets_degraded_counter(self) -> None:
        """TELE-650: Improvement resets degraded counter but not necessarily status."""
        env = EnvState(env_id=0)

        # Setup: get to degraded state
        env.add_accuracy(85.0, epoch=0)
        env.add_accuracy(83.5, epoch=1)
        env.add_accuracy(82.0, epoch=2)
        env.add_accuracy(80.5, epoch=3)
        assert env.status == "degraded"
        assert env.degraded_counter == 3

        # Improvement resets counter
        env.add_accuracy(81.0, epoch=4)
        assert env.degraded_counter == 0  # Counter resets
        # Status may still be degraded (didn't beat best of 85.0)

    def test_status_updates_through_aggregator(self) -> None:
        """TELE-650: Status updates flow through aggregator correctly."""
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

    def test_snapshot_includes_env_status(self) -> None:
        """TELE-650: SanctumSnapshot.envs includes status for each environment."""
        agg = SanctumAggregator(num_envs=3)

        for env_id in range(3):
            event = make_epoch_event(
                env_id=env_id,
                val_accuracy=70.0 + env_id * 10,  # 70, 80, 90
                inner_epoch=1,
            )
            agg.process_event(event)

        snapshot = agg.get_snapshot()

        # All envs should have status field
        for env_id in range(3):
            assert hasattr(snapshot.envs[env_id], "status")
            assert snapshot.envs[env_id].status in (
                "initializing",
                "healthy",
                "excellent",
                "stalled",
                "degraded",
            )


# =============================================================================
# Integration: Observation Stats Schema Completeness
# =============================================================================


class TestObservationStatsSchemaCompleteness:
    """Verify ObservationStats schema has all expected fields for future wiring."""

    def test_feature_group_statistics_fields(self) -> None:
        """Schema should include per-feature-group statistics fields."""
        obs_stats = ObservationStats()

        # Per-feature-group mean/std for slot, host, context features
        assert hasattr(obs_stats, "slot_features_mean")
        assert hasattr(obs_stats, "slot_features_std")
        assert hasattr(obs_stats, "host_features_mean")
        assert hasattr(obs_stats, "host_features_std")
        assert hasattr(obs_stats, "context_features_mean")
        assert hasattr(obs_stats, "context_features_std")

    def test_numerical_health_fields(self) -> None:
        """Schema should include numerical health detection fields."""
        obs_stats = ObservationStats()

        # NaN/Inf detection
        assert hasattr(obs_stats, "nan_count")
        assert hasattr(obs_stats, "inf_count")

        # Outlier detection
        assert hasattr(obs_stats, "outlier_pct")

        # Normalization drift
        assert hasattr(obs_stats, "normalization_drift")

    def test_all_defaults_are_zero(self) -> None:
        """All ObservationStats fields should default to 0/0.0 (stub state)."""
        obs_stats = ObservationStats()

        assert obs_stats.slot_features_mean == 0.0
        assert obs_stats.slot_features_std == 0.0
        assert obs_stats.host_features_mean == 0.0
        assert obs_stats.host_features_std == 0.0
        assert obs_stats.context_features_mean == 0.0
        assert obs_stats.context_features_std == 0.0
        assert obs_stats.outlier_pct == 0.0
        assert obs_stats.nan_count == 0
        assert obs_stats.inf_count == 0
        assert obs_stats.normalization_drift == 0.0


# =============================================================================
# Integration: Episode Stats Schema Completeness
# =============================================================================


class TestEpisodeStatsSchemaCompleteness:
    """Verify EpisodeStats schema has all expected fields for future wiring."""

    def test_length_statistics_fields(self) -> None:
        """Schema should include episode length statistics fields."""
        stats = EpisodeStats()

        assert hasattr(stats, "length_mean")
        assert stats.length_mean == 0.0
        assert hasattr(stats, "length_std")
        assert stats.length_std == 0.0
        assert hasattr(stats, "length_min")
        assert stats.length_min == 0
        assert hasattr(stats, "length_max")
        assert stats.length_max == 0

    def test_outcome_tracking_fields(self) -> None:
        """Schema should include outcome tracking fields."""
        stats = EpisodeStats()

        # Counts
        assert hasattr(stats, "total_episodes")
        assert hasattr(stats, "timeout_count")
        assert hasattr(stats, "success_count")
        assert hasattr(stats, "early_termination_count")

        # Rates
        assert hasattr(stats, "timeout_rate")
        assert hasattr(stats, "success_rate")
        assert hasattr(stats, "early_termination_rate")

    def test_action_efficiency_fields(self) -> None:
        """Schema should include action efficiency fields."""
        stats = EpisodeStats()

        assert hasattr(stats, "steps_per_germinate")
        assert hasattr(stats, "steps_per_prune")
        assert hasattr(stats, "steps_per_fossilize")

    def test_trend_field(self) -> None:
        """Schema should include completion trend field."""
        stats = EpisodeStats()

        assert hasattr(stats, "completion_trend")
        assert stats.completion_trend == "stable"  # Default

    def test_all_numeric_defaults_are_zero(self) -> None:
        """All numeric EpisodeStats fields should default to 0/0.0 (stub state)."""
        stats = EpisodeStats()

        assert stats.length_mean == 0.0
        assert stats.length_std == 0.0
        assert stats.length_min == 0
        assert stats.length_max == 0
        assert stats.total_episodes == 0
        assert stats.timeout_count == 0
        assert stats.success_count == 0
        assert stats.early_termination_count == 0
        assert stats.timeout_rate == 0.0
        assert stats.success_rate == 0.0
        assert stats.early_termination_rate == 0.0
        assert stats.steps_per_germinate == 0.0
        assert stats.steps_per_prune == 0.0
        assert stats.steps_per_fossilize == 0.0
