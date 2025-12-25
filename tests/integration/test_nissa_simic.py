"""Integration tests for Nissa-Simic interaction.

Tests the core integration where:
- Simic training emits telemetry events
- NissaHub receives and routes events
- Reward components are correctly structured for telemetry
- Anomaly detection produces correct reports
"""

import pytest

from esper.nissa import NissaHub
from esper.leyline import LifecycleOp, SeedStage, TelemetryEvent, TelemetryEventType
from esper.leyline.telemetry import AnalyticsSnapshotPayload
from esper.simic.telemetry import AnomalyDetector
from esper.simic.rewards import RewardComponentsTelemetry
from esper.simic.rewards import (
    compute_contribution_reward,
    SeedInfo,
    ContributionRewardConfig,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def hub():
    """Fresh NissaHub instance."""
    return NissaHub()


@pytest.fixture
def anomaly_detector():
    """Anomaly detector for training metrics."""
    return AnomalyDetector()


@pytest.fixture
def seed_info():
    """SeedInfo for reward computation."""
    return SeedInfo(
        stage=SeedStage.TRAINING.value,
        improvement_since_stage_start=0.02,
        total_improvement=0.05,
        epochs_in_stage=3,
        seed_params=10000,
        previous_stage=SeedStage.GERMINATED.value,
        previous_epochs_in_stage=1,
        seed_age_epochs=4,
    )


# =============================================================================
# Reward Telemetry Tests
# =============================================================================


class TestRewardTelemetry:
    """Tests for Simic reward telemetry flowing to Nissa."""

    def test_compute_reward_returns_telemetry_components(self, seed_info):
        """compute_contribution_reward with return_components=True returns telemetry."""
        config = ContributionRewardConfig()

        reward, components = compute_contribution_reward(
            action=LifecycleOp.WAIT,
            seed_contribution=0.02,
            val_acc=75.0,
            seed_info=seed_info,
            epoch=5,
            max_epochs=25,
            total_params=10000,
            host_params=1000000,
            acc_at_germination=70.0,
            acc_delta=0.01,
            config=config,
            return_components=True,
            num_fossilized_seeds=0,
            num_contributing_fossilized=0,
        )

        assert isinstance(components, RewardComponentsTelemetry)
        assert components.total_reward == reward
        assert hasattr(components, "seed_contribution")
        assert hasattr(components, "pbrs_bonus")
        assert hasattr(components, "compute_rent")

    def test_telemetry_components_to_dict(self, seed_info):
        """RewardComponentsTelemetry.to_dict() produces serializable data."""
        config = ContributionRewardConfig()

        _, components = compute_contribution_reward(
            action=LifecycleOp.WAIT,
            seed_contribution=0.02,
            val_acc=75.0,
            seed_info=seed_info,
            epoch=5,
            max_epochs=25,
            total_params=10000,
            host_params=1000000,
            acc_at_germination=70.0,
            acc_delta=0.01,
            config=config,
            return_components=True,
            num_fossilized_seeds=0,
            num_contributing_fossilized=0,
        )

        data = components.to_dict()
        assert isinstance(data, dict)
        assert "total_reward" in data
        assert "seed_contribution" in data


# =============================================================================
# Anomaly Detection Tests
# =============================================================================


class TestAnomalyDetection:
    """Tests for anomaly detection producing correct reports."""

    def test_anomaly_detector_threshold_varies_by_episode(self, anomaly_detector):
        """Anomaly threshold should vary based on training progress."""
        early_threshold = anomaly_detector.get_ev_threshold(
            current_episode=100,
            total_episodes=1000,
        )
        late_threshold = anomaly_detector.get_ev_threshold(
            current_episode=900,
            total_episodes=1000,
        )

        # Thresholds should be reasonable floats
        assert isinstance(early_threshold, float)
        assert isinstance(late_threshold, float)
        # Late threshold is typically higher (more strict)
        assert late_threshold >= early_threshold

    def test_hub_can_emit_telemetry_event(self, hub):
        """NissaHub should accept telemetry events."""
        event = TelemetryEvent(
            event_type=TelemetryEventType.ANALYTICS_SNAPSHOT,
            data=AnalyticsSnapshotPayload(
                kind="last_action",
                env_id=0,
                total_reward=0.5,
                action_name="WAIT",
                value_estimate=0.3,
                action_confidence=0.8,
            ),
            epoch=5,
        )

        # Should not raise
        hub.emit(event)
