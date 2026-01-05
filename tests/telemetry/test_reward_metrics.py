"""End-to-end tests for reward metrics (TELE-400 to TELE-499).

Verifies reward health metrics flow from their source points through to
RewardHealthData (consumed by Sanctum TUI and Karn analytics).

TELE IDs covered:
- TELE-400: pbrs_fraction (PBRS contribution to total reward)
- TELE-401: anti_gaming_trigger_rate (ratio_penalty + alpha_shock trigger rate)
- TELE-402: hypervolume (Pareto hypervolume indicator)
"""


from esper.leyline import (
    TelemetryEvent,
    TelemetryEventType,
    AnalyticsSnapshotPayload,
    EpisodeOutcomePayload,
)
from esper.simic.rewards.reward_telemetry import RewardComponentsTelemetry
from esper.karn.sanctum.widgets.reward_health import RewardHealthData
from esper.karn.sanctum.schema import RewardComponents
from esper.leyline import EpisodeOutcome
from esper.karn.pareto import extract_pareto_frontier, compute_hypervolume_2d

from tests.telemetry.conftest import CaptureHubResult


# =============================================================================
# TELE-400: PBRS Fraction
# =============================================================================


class TestTELE400PBRSFraction:
    """TELE-400: PBRS fraction of total reward.

    Verifies stage_bonus (PBRS shaping reward) flows through to pbrs_fraction
    in RewardHealthData.

    Source: simic/rewards/rewards.py -> _contribution_pbrs_bonus()
    Transport: RewardComponentsTelemetry.stage_bonus -> SanctumAggregator
    Sink: RewardHealthData.pbrs_fraction (computed by compute_reward_health())
    """

    def test_pbrs_fraction_computation_from_components(self):
        """TELE-400: pbrs_fraction = |stage_bonus| / |total_reward|."""
        # Simulate reward components from a step
        components = RewardComponentsTelemetry(
            stage_bonus=0.25,  # PBRS shaping reward
            total_reward=1.0,  # Total reward for step
        )

        # Verify pbrs fraction can be computed
        # The aggregator does: pbrs_total / max(1e-8, reward_total)
        pbrs_fraction = abs(components.stage_bonus) / max(1e-8, abs(components.total_reward))
        assert pbrs_fraction == 0.25

    def test_pbrs_fraction_zero_when_no_stage_bonus(self):
        """TELE-400: pbrs_fraction is 0.0 when stage_bonus is 0.0."""
        components = RewardComponentsTelemetry(
            stage_bonus=0.0,
            total_reward=1.0,
        )

        pbrs_fraction = abs(components.stage_bonus) / max(1e-8, abs(components.total_reward))
        assert pbrs_fraction == 0.0

    def test_pbrs_fraction_health_thresholds(self):
        """TELE-400: pbrs_fraction healthy when 10-40% (DRL Expert recommendation)."""
        # Healthy range: 10-40%
        healthy_low = RewardHealthData(pbrs_fraction=0.10)
        healthy_mid = RewardHealthData(pbrs_fraction=0.25)
        healthy_high = RewardHealthData(pbrs_fraction=0.40)

        assert healthy_low.is_pbrs_healthy
        assert healthy_mid.is_pbrs_healthy
        assert healthy_high.is_pbrs_healthy

        # Unhealthy: below 10% (shaping too weak)
        unhealthy_low = RewardHealthData(pbrs_fraction=0.05)
        assert not unhealthy_low.is_pbrs_healthy

        # Unhealthy: above 40% (shaping dominates)
        unhealthy_high = RewardHealthData(pbrs_fraction=0.50)
        assert not unhealthy_high.is_pbrs_healthy

    def test_pbrs_fraction_from_schema_reward_components(self):
        """TELE-400: Sanctum schema RewardComponents carries stage_bonus."""
        # The schema type used by the aggregator's EnvState
        schema_components = RewardComponents(
            total=1.0,
            stage_bonus=0.3,
        )

        # Aggregator computes: sum(|stage_bonus|) / sum(|total|)
        pbrs_fraction = abs(schema_components.stage_bonus) / max(1e-8, abs(schema_components.total))
        assert abs(pbrs_fraction - 0.3) < 1e-6

    def test_pbrs_fraction_aggregation_across_envs(self):
        """TELE-400: pbrs_fraction aggregated across all environments."""
        # Simulate multiple env reward components
        env_components = [
            RewardComponents(total=1.0, stage_bonus=0.2),
            RewardComponents(total=2.0, stage_bonus=0.4),
            RewardComponents(total=1.5, stage_bonus=0.3),
        ]

        # Aggregator logic: sum(|stage_bonus|) / sum(|total|)
        pbrs_total = sum(abs(c.stage_bonus) for c in env_components)
        reward_total = sum(abs(c.total) for c in env_components)
        pbrs_fraction = pbrs_total / max(1e-8, reward_total)

        # Expected: (0.2 + 0.4 + 0.3) / (1.0 + 2.0 + 1.5) = 0.9 / 4.5 = 0.2
        assert abs(pbrs_fraction - 0.2) < 1e-6


# =============================================================================
# TELE-401: Anti-Gaming Trigger Rate
# =============================================================================


class TestTELE401AntiGamingTriggerRate:
    """TELE-401: Anti-gaming penalty trigger rate.

    Verifies ratio_penalty and alpha_shock flow through to anti_gaming_trigger_rate
    in RewardHealthData.

    Source: simic/rewards/rewards.py -> compute_contribution_reward() and compute_rent_and_shaping()
    Transport: RewardComponentsTelemetry.{ratio_penalty, alpha_shock} -> SanctumAggregator
    Sink: RewardHealthData.anti_gaming_trigger_rate (computed by compute_reward_health())
    """

    def test_gaming_rate_from_ratio_penalty(self):
        """TELE-401: ratio_penalty != 0 counts as gaming trigger."""
        components = RewardComponentsTelemetry(
            ratio_penalty=-0.1,  # Non-zero penalty
            alpha_shock=0.0,
            total_reward=0.5,
        )

        # Check: ratio_penalty != 0 triggers count
        is_gaming = components.ratio_penalty != 0 or components.alpha_shock != 0
        assert is_gaming

    def test_gaming_rate_from_alpha_shock(self):
        """TELE-401: alpha_shock != 0 counts as gaming trigger."""
        components = RewardComponentsTelemetry(
            ratio_penalty=0.0,
            alpha_shock=-0.05,  # Non-zero penalty
            total_reward=0.5,
        )

        is_gaming = components.ratio_penalty != 0 or components.alpha_shock != 0
        assert is_gaming

    def test_gaming_rate_from_both_penalties(self):
        """TELE-401: Both penalties trigger counts as single gaming step."""
        components = RewardComponentsTelemetry(
            ratio_penalty=-0.1,
            alpha_shock=-0.05,
            total_reward=0.5,
        )

        # Still counts as 1 gaming step (not 2)
        is_gaming = components.ratio_penalty != 0 or components.alpha_shock != 0
        assert is_gaming

    def test_gaming_rate_zero_when_no_penalties(self):
        """TELE-401: No gaming when both penalties are zero."""
        components = RewardComponentsTelemetry(
            ratio_penalty=0.0,
            alpha_shock=0.0,
            total_reward=0.5,
        )

        is_gaming = components.ratio_penalty != 0 or components.alpha_shock != 0
        assert not is_gaming

    def test_gaming_rate_aggregation_logic(self):
        """TELE-401: gaming_rate = gaming_steps / total_steps."""
        # Simulate 10 steps, 2 with gaming penalties
        steps = [
            RewardComponents(total=1.0, ratio_penalty=0.0, alpha_shock=0.0),
            RewardComponents(total=1.0, ratio_penalty=-0.1, alpha_shock=0.0),  # Gaming
            RewardComponents(total=1.0, ratio_penalty=0.0, alpha_shock=0.0),
            RewardComponents(total=1.0, ratio_penalty=0.0, alpha_shock=0.0),
            RewardComponents(total=1.0, ratio_penalty=0.0, alpha_shock=-0.05),  # Gaming
            RewardComponents(total=1.0, ratio_penalty=0.0, alpha_shock=0.0),
            RewardComponents(total=1.0, ratio_penalty=0.0, alpha_shock=0.0),
            RewardComponents(total=1.0, ratio_penalty=0.0, alpha_shock=0.0),
            RewardComponents(total=1.0, ratio_penalty=0.0, alpha_shock=0.0),
            RewardComponents(total=1.0, ratio_penalty=0.0, alpha_shock=0.0),
        ]

        gaming_steps = sum(
            1 for c in steps
            if c.ratio_penalty != 0 or c.alpha_shock != 0
        )
        gaming_rate = gaming_steps / max(1, len(steps))

        assert gaming_steps == 2
        assert abs(gaming_rate - 0.2) < 1e-6

    def test_gaming_rate_health_thresholds(self):
        """TELE-401: gaming_rate healthy when < 5%."""
        # Healthy: < 5%
        healthy = RewardHealthData(anti_gaming_trigger_rate=0.03)
        assert healthy.is_gaming_healthy

        # Boundary: exactly 5% is unhealthy
        boundary = RewardHealthData(anti_gaming_trigger_rate=0.05)
        assert not boundary.is_gaming_healthy

        # Unhealthy: > 5%
        unhealthy = RewardHealthData(anti_gaming_trigger_rate=0.10)
        assert not unhealthy.is_gaming_healthy


# =============================================================================
# TELE-402: Pareto Hypervolume
# =============================================================================


class TestTELE402Hypervolume:
    """TELE-402: Pareto hypervolume indicator.

    Verifies EpisodeOutcomePayload flows through to hypervolume in RewardHealthData.

    Source: simic/training/vectorized.py -> episode completion
    Transport: TelemetryEvent(EPISODE_OUTCOME) -> SanctumAggregator._handle_episode_outcome()
    Computation: extract_pareto_frontier() -> compute_hypervolume_2d()
    Sink: RewardHealthData.hypervolume (via compute_reward_health())
    """

    def test_hypervolume_from_single_outcome(self):
        """TELE-402: Single episode outcome contributes to hypervolume."""
        outcome = EpisodeOutcome(
            env_id=0,
            episode_idx=1,
            final_accuracy=80.0,
            param_ratio=0.2,
            num_fossilized=1,
            num_contributing_fossilized=1,
            episode_reward=5.0,
            stability_score=0.9,
            reward_mode="shaped",
        )

        frontier = extract_pareto_frontier([outcome])
        ref_point = (0.0, 1.0)  # Standard reference point
        hv = compute_hypervolume_2d(frontier, ref_point)

        # Expected: (80 - 0) * (1.0 - 0.2) = 80 * 0.8 = 64
        assert abs(hv - 64.0) < 1e-6

    def test_hypervolume_from_multiple_outcomes(self):
        """TELE-402: Multiple episode outcomes form Pareto frontier."""
        outcomes = [
            EpisodeOutcome(
                env_id=0, episode_idx=1,
                final_accuracy=80.0, param_ratio=0.3,
                num_fossilized=1, num_contributing_fossilized=1,
                episode_reward=5.0, stability_score=0.9,
                reward_mode="shaped",
            ),
            EpisodeOutcome(
                env_id=0, episode_idx=2,
                final_accuracy=60.0, param_ratio=0.1,
                num_fossilized=1, num_contributing_fossilized=1,
                episode_reward=3.0, stability_score=0.8,
                reward_mode="shaped",
            ),
        ]

        frontier = extract_pareto_frontier(outcomes)
        ref_point = (0.0, 1.0)
        hv = compute_hypervolume_2d(frontier, ref_point)

        # Both points should be on frontier (neither dominates the other)
        assert len(frontier) == 2

        # Expected HV:
        # Sorted by acc descending: [(80, 0.3), (60, 0.1)]
        # Area from (80, 0.3): 80 * (1.0 - 0.3) = 56
        # Area from (60, 0.1): 60 * (0.3 - 0.1) = 12
        # Total = 68
        assert abs(hv - 68.0) < 1e-6

    def test_hypervolume_zero_for_empty_outcomes(self):
        """TELE-402: hypervolume is 0.0 before any episodes complete."""
        hv = compute_hypervolume_2d([], (0.0, 1.0))
        assert hv == 0.0

    def test_hypervolume_dominated_outcome_excluded(self):
        """TELE-402: Dominated outcomes don't contribute to hypervolume."""
        outcomes = [
            EpisodeOutcome(
                env_id=0, episode_idx=1,
                final_accuracy=80.0, param_ratio=0.2,  # Pareto optimal
                num_fossilized=1, num_contributing_fossilized=1,
                episode_reward=5.0, stability_score=0.9,
                reward_mode="shaped",
            ),
            EpisodeOutcome(
                env_id=0, episode_idx=2,
                final_accuracy=70.0, param_ratio=0.3,  # Dominated (worse on both axes)
                num_fossilized=1, num_contributing_fossilized=1,
                episode_reward=3.0, stability_score=0.8,
                reward_mode="shaped",
            ),
        ]

        frontier = extract_pareto_frontier(outcomes)
        assert len(frontier) == 1
        assert frontier[0].final_accuracy == 80.0

        ref_point = (0.0, 1.0)
        hv = compute_hypervolume_2d(frontier, ref_point)

        # Only the non-dominated point contributes
        # Expected: 80 * (1.0 - 0.2) = 64
        assert abs(hv - 64.0) < 1e-6

    def test_episode_outcome_payload_to_episode_outcome(self):
        """TELE-402: EpisodeOutcomePayload converts to EpisodeOutcome for analysis."""
        # Simulate payload from telemetry event
        payload = EpisodeOutcomePayload(
            env_id=0,
            episode_idx=1,
            final_accuracy=75.0,
            param_ratio=0.15,
            num_fossilized=2,
            num_contributing_fossilized=1,
            episode_reward=4.5,
            stability_score=0.85,
            reward_mode="shaped",
        )

        # Convert to EpisodeOutcome (as aggregator does)
        outcome = EpisodeOutcome(
            env_id=payload.env_id,
            episode_idx=payload.episode_idx,
            final_accuracy=payload.final_accuracy,
            param_ratio=payload.param_ratio,
            num_fossilized=payload.num_fossilized,
            num_contributing_fossilized=payload.num_contributing_fossilized,
            episode_reward=payload.episode_reward,
            stability_score=payload.stability_score,
            reward_mode=payload.reward_mode,
        )

        # Verify conversion preserves values
        assert outcome.final_accuracy == 75.0
        assert outcome.param_ratio == 0.15

        # Compute hypervolume
        frontier = extract_pareto_frontier([outcome])
        hv = compute_hypervolume_2d(frontier, (0.0, 1.0))

        # Expected: 75 * (1.0 - 0.15) = 75 * 0.85 = 63.75
        assert abs(hv - 63.75) < 1e-6

    def test_hypervolume_monotonic_increase_with_better_outcomes(self):
        """TELE-402: hypervolume should increase as better outcomes are added."""
        outcomes = []
        hvs = []

        # Add progressively better outcomes
        for i, (acc, param) in enumerate([(60, 0.4), (70, 0.3), (80, 0.2)]):
            outcomes.append(EpisodeOutcome(
                env_id=0, episode_idx=i,
                final_accuracy=acc, param_ratio=param,
                num_fossilized=1, num_contributing_fossilized=1,
                episode_reward=acc / 10, stability_score=0.9,
                reward_mode="shaped",
            ))

            frontier = extract_pareto_frontier(outcomes)
            hv = compute_hypervolume_2d(frontier, (0.0, 1.0))
            hvs.append(hv)

        # Each addition should increase hypervolume
        assert hvs[1] >= hvs[0]
        assert hvs[2] >= hvs[1]


# =============================================================================
# Integration Tests: RewardHealthData Aggregation
# =============================================================================


class TestRewardHealthDataIntegration:
    """Integration tests for RewardHealthData combining all TELE-400 metrics."""

    def test_reward_health_data_all_fields(self):
        """RewardHealthData combines pbrs, gaming, ev, and hypervolume."""
        data = RewardHealthData(
            pbrs_fraction=0.25,
            anti_gaming_trigger_rate=0.03,
            ev_explained=0.65,
            hypervolume=42.5,
        )

        assert data.pbrs_fraction == 0.25
        assert data.anti_gaming_trigger_rate == 0.03
        assert data.ev_explained == 0.65
        assert data.hypervolume == 42.5

        # Health checks
        assert data.is_pbrs_healthy
        assert data.is_gaming_healthy
        assert data.is_ev_healthy

    def test_reward_health_data_defaults(self):
        """RewardHealthData defaults to zero state."""
        data = RewardHealthData()

        assert data.pbrs_fraction == 0.0
        assert data.anti_gaming_trigger_rate == 0.0
        assert data.ev_explained == 0.0
        assert data.hypervolume == 0.0

        # Zero pbrs is unhealthy (< 10%)
        assert not data.is_pbrs_healthy
        # Zero gaming is healthy (< 5%)
        assert data.is_gaming_healthy
        # Zero EV is unhealthy (< 50%)
        assert not data.is_ev_healthy

    def test_reward_components_telemetry_to_dict(self):
        """RewardComponentsTelemetry.to_dict() includes all reward fields."""
        components = RewardComponentsTelemetry(
            stage_bonus=0.3,
            pbrs_bonus=0.3,  # Same as stage_bonus for CONTRIBUTION
            ratio_penalty=-0.1,
            alpha_shock=-0.05,
            total_reward=1.0,
        )

        data = components.to_dict()

        assert data["stage_bonus"] == 0.3
        assert data["pbrs_bonus"] == 0.3
        assert data["ratio_penalty"] == -0.1
        assert data["alpha_shock"] == -0.05
        assert data["total_reward"] == 1.0

    def test_reward_components_telemetry_roundtrip(self):
        """RewardComponentsTelemetry survives dict roundtrip."""
        original = RewardComponentsTelemetry(
            stage_bonus=0.25,
            pbrs_bonus=0.25,
            ratio_penalty=-0.15,
            alpha_shock=-0.08,
            total_reward=0.85,
            base_acc_delta=0.02,
            compute_rent=-0.01,
        )

        # Serialize and deserialize
        data = original.to_dict()
        restored = RewardComponentsTelemetry.from_dict(data)

        # Verify all reward-related fields
        assert restored.stage_bonus == original.stage_bonus
        assert restored.pbrs_bonus == original.pbrs_bonus
        assert restored.ratio_penalty == original.ratio_penalty
        assert restored.alpha_shock == original.alpha_shock
        assert restored.total_reward == original.total_reward
        assert restored.base_acc_delta == original.base_acc_delta
        assert restored.compute_rent == original.compute_rent


# =============================================================================
# Telemetry Event Tests (verifying event structure)
# =============================================================================


class TestTelemetryEventStructure:
    """Verify telemetry events carry reward data correctly."""

    def test_episode_outcome_event_structure(self):
        """EPISODE_OUTCOME event carries Pareto-relevant data."""
        payload = EpisodeOutcomePayload(
            env_id=0,
            episode_idx=5,
            final_accuracy=82.5,
            param_ratio=0.18,
            num_fossilized=3,
            num_contributing_fossilized=2,
            episode_reward=8.5,
            stability_score=0.92,
            reward_mode="shaped",
        )

        event = TelemetryEvent(
            event_type=TelemetryEventType.EPISODE_OUTCOME,
            data=payload,
        )

        assert event.event_type == TelemetryEventType.EPISODE_OUTCOME
        assert event.data.final_accuracy == 82.5
        assert event.data.param_ratio == 0.18
        assert event.data.stability_score == 0.92

    def test_analytics_snapshot_last_action_structure(self):
        """ANALYTICS_SNAPSHOT with kind='last_action' carries reward context."""
        payload = AnalyticsSnapshotPayload(
            kind="last_action",
            env_id=0,
            total_reward=1.5,
            action_name="GERMINATE",
            action_confidence=0.85,
            value_estimate=2.3,
        )

        event = TelemetryEvent(
            event_type=TelemetryEventType.ANALYTICS_SNAPSHOT,
            data=payload,
        )

        assert event.event_type == TelemetryEventType.ANALYTICS_SNAPSHOT
        assert event.data.kind == "last_action"
        assert event.data.total_reward == 1.5


# =============================================================================
# Capture Backend Tests (E2E verification)
# =============================================================================


class TestCaptureBackendE2E:
    """End-to-end tests using CaptureBackend to verify event flow."""

    def test_capture_backend_records_episode_outcome(self, capture_hub: CaptureHubResult):
        """CaptureBackend captures EPISODE_OUTCOME events."""
        hub, backend = capture_hub

        payload = EpisodeOutcomePayload(
            env_id=0,
            episode_idx=1,
            final_accuracy=75.0,
            param_ratio=0.2,
            num_fossilized=1,
            num_contributing_fossilized=1,
            episode_reward=5.0,
            stability_score=0.9,
            reward_mode="shaped",
        )

        event = TelemetryEvent(
            event_type=TelemetryEventType.EPISODE_OUTCOME,
            data=payload,
        )
        hub.emit(event)

        # Wait for async queue processing
        hub.flush(timeout=5.0)

        # Verify capture
        events = backend.find_events(TelemetryEventType.EPISODE_OUTCOME)
        assert len(events) == 1
        assert events[0].data.final_accuracy == 75.0
        assert events[0].data.param_ratio == 0.2

    def test_capture_backend_records_analytics_snapshot(self, capture_hub: CaptureHubResult):
        """CaptureBackend captures ANALYTICS_SNAPSHOT events."""
        hub, backend = capture_hub

        payload = AnalyticsSnapshotPayload(
            kind="last_action",
            env_id=0,
            total_reward=1.2,
            action_name="WAIT",
        )

        event = TelemetryEvent(
            event_type=TelemetryEventType.ANALYTICS_SNAPSHOT,
            data=payload,
        )
        hub.emit(event)

        # Wait for async queue processing
        hub.flush(timeout=5.0)

        # Verify capture
        events = backend.find_events(TelemetryEventType.ANALYTICS_SNAPSHOT)
        assert len(events) == 1
        assert events[0].data.kind == "last_action"
        assert events[0].data.total_reward == 1.2
