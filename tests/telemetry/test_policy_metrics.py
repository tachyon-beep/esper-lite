"""End-to-end tests for policy metrics (TELE-100 to TELE-199).

Verifies PPO training metrics flow from their source point through to nissa.
These tests ensure that policy-related telemetry is correctly emitted via
PPOUpdatePayload and reaches consumers through the NissaHub.

Coverage:
- TELE-100: ppo_data_received (gate flag for PPO metrics display)
- TELE-110: kl_divergence
- TELE-111: kl_divergence_history (derived in aggregator)
- TELE-120: entropy
- TELE-121: entropy_velocity (derived in aggregator)
- TELE-122: collapse_risk_score (derived in aggregator)
- TELE-123: entropy_clip_correlation (derived in aggregator)
- TELE-124: entropy_collapsed
- TELE-130: clip_fraction
- TELE-131: policy_loss
- TELE-140: advantage_std
- TELE-141: advantage_mean
- TELE-142: advantage_skewness
- TELE-143: advantage_kurtosis
- TELE-144: advantage_positive_ratio
- TELE-150: joint_ratio_max
- TELE-151: head_*_ratio_max (per-head ratio max)
- TELE-160: log_prob_min
- TELE-161: log_prob_max
- TELE-170: head_*_entropy (per-head entropy)
"""

import math

import pytest

from esper.leyline import PPOUpdatePayload, TelemetryEvent, TelemetryEventType
from esper.nissa.output import NissaHub

from .conftest import CaptureBackend, CaptureHubResult


def _make_base_ppo_payload(**overrides) -> PPOUpdatePayload:
    """Create a valid PPOUpdatePayload with all required fields.

    This factory ensures tests have complete payloads while allowing
    specific field overrides for targeted assertions.
    """
    defaults = {
        # Required core metrics
        "policy_loss": 0.5,
        "value_loss": 0.3,
        "entropy": 1.2,
        "grad_norm": 0.8,
        "kl_divergence": 0.015,
        "clip_fraction": 0.12,
        "nan_grad_count": 0,
        "pre_clip_grad_norm": 0.9,
        # Advantage stats
        "advantage_mean": 0.0,
        "advantage_std": 1.0,
        "advantage_skewness": 0.1,
        "advantage_kurtosis": 0.05,
        "advantage_positive_ratio": 0.52,
        # Ratio stats
        "ratio_mean": 1.0,
        "ratio_min": 0.85,
        "ratio_max": 1.15,
        "ratio_std": 0.08,
        # Log prob extremes
        "log_prob_min": -8.0,
        "log_prob_max": -0.3,
        # Boolean flags
        "entropy_collapsed": False,
        # Timing and context
        "update_time_ms": 15.0,
        "inner_epoch": 1,
        "batch": 0,
        "ppo_updates_count": 4,
        # Value stats
        "value_mean": 0.5,
        "value_std": 0.2,
        "value_min": 0.1,
        "value_max": 0.9,
        # Gradient quality
        "clip_fraction_positive": 0.06,
        "clip_fraction_negative": 0.04,
        "gradient_cv": 0.3,
        # Pre-norm stats
        "pre_norm_advantage_mean": 0.05,
        "pre_norm_advantage_std": 2.5,
        # Return stats
        "return_mean": 0.6,
        "return_std": 0.25,
    }
    defaults.update(overrides)
    return PPOUpdatePayload(**defaults)


def _emit_ppo_update(hub: NissaHub, payload: PPOUpdatePayload, epoch: int = 1) -> None:
    """Emit a PPO_UPDATE_COMPLETED event with the given payload.

    Flushes the hub after emission to ensure the event is processed
    before tests check the backend for captured events.
    """
    event = TelemetryEvent(
        event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
        epoch=epoch,
        data=payload,
    )
    hub.emit(event)
    # Flush to ensure background worker processes the event
    hub.flush(timeout=5.0)


# =============================================================================
# TELE-100: ppo_data_received
# =============================================================================


class TestTELE100PPODataReceived:
    """TELE-100: Boolean gate flag for PPO metrics display."""

    def test_ppo_update_event_is_emitted(self, capture_hub: CaptureHubResult):
        """TELE-100: PPO_UPDATE_COMPLETED event is emitted and captured."""
        hub, backend = capture_hub

        payload = _make_base_ppo_payload()
        _emit_ppo_update(hub, payload)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert len(events) == 1, "Expected exactly one PPO_UPDATE_COMPLETED event"

    def test_ppo_update_event_has_valid_payload(self, capture_hub: CaptureHubResult):
        """TELE-100: PPO_UPDATE_COMPLETED event contains valid payload data."""
        hub, backend = capture_hub

        payload = _make_base_ppo_payload(entropy=1.5, kl_divergence=0.02)
        _emit_ppo_update(hub, payload)

        event = backend.find_first(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert event is not None
        assert event.data.entropy == 1.5
        assert event.data.kl_divergence == 0.02


# =============================================================================
# TELE-110/111: KL Divergence
# =============================================================================


class TestTELE110KLDivergence:
    """TELE-110: KL divergence measures policy trust region stability."""

    def test_kl_divergence_in_payload(self, capture_hub: CaptureHubResult):
        """TELE-110: kl_divergence field is present and matches input."""
        hub, backend = capture_hub

        payload = _make_base_ppo_payload(kl_divergence=0.0123)
        _emit_ppo_update(hub, payload)

        event = backend.find_first(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert event is not None
        assert event.data.kl_divergence == pytest.approx(0.0123, abs=1e-6)

    def test_kl_divergence_zero_is_valid(self, capture_hub: CaptureHubResult):
        """TELE-110: Zero KL divergence indicates no policy change."""
        hub, backend = capture_hub

        payload = _make_base_ppo_payload(kl_divergence=0.0)
        _emit_ppo_update(hub, payload)

        event = backend.find_first(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert event is not None
        assert event.data.kl_divergence == 0.0

    def test_high_kl_divergence_emitted(self, capture_hub: CaptureHubResult):
        """TELE-110: High KL values (indicating policy drift) are captured."""
        hub, backend = capture_hub

        # Above critical threshold (0.03)
        payload = _make_base_ppo_payload(kl_divergence=0.05)
        _emit_ppo_update(hub, payload)

        event = backend.find_first(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert event is not None
        assert event.data.kl_divergence == pytest.approx(0.05, abs=1e-6)


class TestTELE111KLDivergenceHistory:
    """TELE-111: KL divergence history for trend analysis.

    Note: History is aggregated in SanctumAggregator, not in the payload.
    These tests verify the raw kl_divergence values that feed the history.
    """

    def test_multiple_kl_values_emitted_for_history(self, capture_hub: CaptureHubResult):
        """TELE-111: Multiple PPO updates provide values for history deque."""
        hub, backend = capture_hub

        kl_values = [0.01, 0.012, 0.015, 0.018, 0.02]
        for i, kl in enumerate(kl_values):
            payload = _make_base_ppo_payload(kl_divergence=kl)
            _emit_ppo_update(hub, payload, epoch=i + 1)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert len(events) == 5

        for event, expected_kl in zip(events, kl_values):
            assert event.data.kl_divergence == pytest.approx(expected_kl, abs=1e-6)


# =============================================================================
# TELE-120-124: Entropy Metrics
# =============================================================================


class TestTELE120Entropy:
    """TELE-120: Policy entropy measures exploration/exploitation balance."""

    def test_entropy_in_payload(self, capture_hub: CaptureHubResult):
        """TELE-120: entropy field is present and matches input."""
        hub, backend = capture_hub

        payload = _make_base_ppo_payload(entropy=1.234)
        _emit_ppo_update(hub, payload)

        event = backend.find_first(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert event is not None
        assert event.data.entropy == pytest.approx(1.234, abs=1e-6)

    def test_low_entropy_emitted(self, capture_hub: CaptureHubResult):
        """TELE-120: Low entropy values (near collapse) are captured."""
        hub, backend = capture_hub

        # Below critical threshold (0.1)
        payload = _make_base_ppo_payload(entropy=0.05)
        _emit_ppo_update(hub, payload)

        event = backend.find_first(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert event is not None
        assert event.data.entropy == pytest.approx(0.05, abs=1e-6)

    def test_high_entropy_emitted(self, capture_hub: CaptureHubResult):
        """TELE-120: High entropy values (good exploration) are captured."""
        hub, backend = capture_hub

        payload = _make_base_ppo_payload(entropy=2.5)
        _emit_ppo_update(hub, payload)

        event = backend.find_first(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert event is not None
        assert event.data.entropy == pytest.approx(2.5, abs=1e-6)


class TestTELE121EntropyVelocity:
    """TELE-121: Entropy velocity (rate of change) is derived from history.

    Note: Velocity is computed in the aggregator from entropy_history.
    These tests verify entropy values are correctly emitted to feed the computation.
    """

    def test_declining_entropy_pattern_emitted(self, capture_hub: CaptureHubResult):
        """TELE-121: Declining entropy values provide data for velocity computation."""
        hub, backend = capture_hub

        # Declining pattern -> negative velocity
        entropy_values = [1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9]
        for i, ent in enumerate(entropy_values):
            payload = _make_base_ppo_payload(entropy=ent)
            _emit_ppo_update(hub, payload, epoch=i + 1)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert len(events) == 7

        for event, expected_ent in zip(events, entropy_values):
            assert event.data.entropy == pytest.approx(expected_ent, abs=1e-6)

    def test_stable_entropy_pattern_emitted(self, capture_hub: CaptureHubResult):
        """TELE-121: Stable entropy values produce near-zero velocity."""
        hub, backend = capture_hub

        # Stable pattern -> velocity ~ 0
        entropy_values = [1.2, 1.21, 1.19, 1.2, 1.2]
        for i, ent in enumerate(entropy_values):
            payload = _make_base_ppo_payload(entropy=ent)
            _emit_ppo_update(hub, payload, epoch=i + 1)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert len(events) == 5


class TestTELE122CollapseRiskScore:
    """TELE-122: Collapse risk score is derived from entropy and velocity.

    Note: Risk score is computed in the aggregator.
    These tests verify the entropy data feeding the computation.
    """

    def test_near_collapse_entropy_emitted(self, capture_hub: CaptureHubResult):
        """TELE-122: Low entropy values that would trigger high collapse risk."""
        hub, backend = capture_hub

        # Near critical threshold
        payload = _make_base_ppo_payload(entropy=0.15)
        _emit_ppo_update(hub, payload)

        event = backend.find_first(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert event is not None
        assert event.data.entropy < 0.3  # Below warning threshold


class TestTELE123EntropyClipCorrelation:
    """TELE-123: Correlation between entropy and clip_fraction.

    Note: Correlation is computed in the aggregator from history.
    These tests verify both entropy and clip_fraction are emitted.
    """

    def test_entropy_and_clip_both_emitted(self, capture_hub: CaptureHubResult):
        """TELE-123: Both entropy and clip_fraction are present for correlation."""
        hub, backend = capture_hub

        payload = _make_base_ppo_payload(entropy=0.8, clip_fraction=0.25)
        _emit_ppo_update(hub, payload)

        event = backend.find_first(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert event is not None
        assert event.data.entropy == pytest.approx(0.8, abs=1e-6)
        assert event.data.clip_fraction == pytest.approx(0.25, abs=1e-6)

    def test_collapse_pattern_entropy_declining_clip_rising(
        self, capture_hub: CaptureHubResult
    ):
        """TELE-123: Pattern of declining entropy + rising clip (collapse risk)."""
        hub, backend = capture_hub

        patterns = [
            (1.5, 0.05),  # High entropy, low clip
            (1.3, 0.10),
            (1.0, 0.15),
            (0.7, 0.20),
            (0.4, 0.30),  # Low entropy, high clip - collapse pattern
        ]

        for i, (ent, clip) in enumerate(patterns):
            payload = _make_base_ppo_payload(entropy=ent, clip_fraction=clip)
            _emit_ppo_update(hub, payload, epoch=i + 1)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert len(events) == 5


class TestTELE124EntropyCollapsed:
    """TELE-124: Boolean flag indicating entropy has crossed critical threshold."""

    def test_entropy_collapsed_false(self, capture_hub: CaptureHubResult):
        """TELE-124: entropy_collapsed=False when entropy is healthy."""
        hub, backend = capture_hub

        payload = _make_base_ppo_payload(entropy=1.0, entropy_collapsed=False)
        _emit_ppo_update(hub, payload)

        event = backend.find_first(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert event is not None
        assert event.data.entropy_collapsed is False

    def test_entropy_collapsed_true(self, capture_hub: CaptureHubResult):
        """TELE-124: entropy_collapsed=True when entropy below critical threshold."""
        hub, backend = capture_hub

        # Below critical (0.1)
        payload = _make_base_ppo_payload(entropy=0.05, entropy_collapsed=True)
        _emit_ppo_update(hub, payload)

        event = backend.find_first(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert event is not None
        assert event.data.entropy_collapsed is True


# =============================================================================
# TELE-130/131: Clip Fraction and Policy Loss
# =============================================================================


class TestTELE130ClipFraction:
    """TELE-130: Clip fraction measures PPO clipping mechanism activation."""

    def test_clip_fraction_in_payload(self, capture_hub: CaptureHubResult):
        """TELE-130: clip_fraction field is present and matches input."""
        hub, backend = capture_hub

        payload = _make_base_ppo_payload(clip_fraction=0.18)
        _emit_ppo_update(hub, payload)

        event = backend.find_first(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert event is not None
        assert event.data.clip_fraction == pytest.approx(0.18, abs=1e-6)

    def test_clip_fraction_zero_valid(self, capture_hub: CaptureHubResult):
        """TELE-130: Zero clip fraction (no clipping) is valid."""
        hub, backend = capture_hub

        payload = _make_base_ppo_payload(clip_fraction=0.0)
        _emit_ppo_update(hub, payload)

        event = backend.find_first(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert event is not None
        assert event.data.clip_fraction == 0.0

    def test_high_clip_fraction_emitted(self, capture_hub: CaptureHubResult):
        """TELE-130: High clip fraction (above critical) is captured."""
        hub, backend = capture_hub

        # Above critical threshold (0.30)
        payload = _make_base_ppo_payload(clip_fraction=0.35)
        _emit_ppo_update(hub, payload)

        event = backend.find_first(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert event is not None
        assert event.data.clip_fraction == pytest.approx(0.35, abs=1e-6)

    def test_directional_clip_fractions_emitted(self, capture_hub: CaptureHubResult):
        """TELE-130: Positive and negative clip fractions are tracked separately."""
        hub, backend = capture_hub

        payload = _make_base_ppo_payload(
            clip_fraction=0.20,
            clip_fraction_positive=0.12,
            clip_fraction_negative=0.08,
        )
        _emit_ppo_update(hub, payload)

        event = backend.find_first(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert event is not None
        assert event.data.clip_fraction_positive == pytest.approx(0.12, abs=1e-6)
        assert event.data.clip_fraction_negative == pytest.approx(0.08, abs=1e-6)


class TestTELE131PolicyLoss:
    """TELE-131: Policy loss measures the PPO surrogate objective."""

    def test_policy_loss_in_payload(self, capture_hub: CaptureHubResult):
        """TELE-131: policy_loss field is present and matches input."""
        hub, backend = capture_hub

        payload = _make_base_ppo_payload(policy_loss=0.123)
        _emit_ppo_update(hub, payload)

        event = backend.find_first(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert event is not None
        assert event.data.policy_loss == pytest.approx(0.123, abs=1e-6)

    def test_policy_loss_negative_valid(self, capture_hub: CaptureHubResult):
        """TELE-131: Negative policy loss (valid for surrogate) is captured."""
        hub, backend = capture_hub

        payload = _make_base_ppo_payload(policy_loss=-0.05)
        _emit_ppo_update(hub, payload)

        event = backend.find_first(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert event is not None
        assert event.data.policy_loss == pytest.approx(-0.05, abs=1e-6)


# =============================================================================
# TELE-140-144: Advantage Statistics
# =============================================================================


class TestTELE140AdvantageStd:
    """TELE-140: Advantage standard deviation measures normalization health."""

    def test_advantage_std_in_payload(self, capture_hub: CaptureHubResult):
        """TELE-140: advantage_std field is present and matches input."""
        hub, backend = capture_hub

        payload = _make_base_ppo_payload(advantage_std=0.95)
        _emit_ppo_update(hub, payload)

        event = backend.find_first(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert event is not None
        assert event.data.advantage_std == pytest.approx(0.95, abs=1e-6)

    def test_advantage_std_healthy_range(self, capture_hub: CaptureHubResult):
        """TELE-140: Healthy advantage_std near 1.0 is captured."""
        hub, backend = capture_hub

        payload = _make_base_ppo_payload(advantage_std=1.05)
        _emit_ppo_update(hub, payload)

        event = backend.find_first(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert event is not None
        assert 0.5 < event.data.advantage_std < 2.0

    def test_advantage_std_collapsed(self, capture_hub: CaptureHubResult):
        """TELE-140: Very low advantage_std (collapsed) is captured."""
        hub, backend = capture_hub

        # Below collapsed threshold (0.1)
        payload = _make_base_ppo_payload(advantage_std=0.05)
        _emit_ppo_update(hub, payload)

        event = backend.find_first(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert event is not None
        assert event.data.advantage_std == pytest.approx(0.05, abs=1e-6)


class TestTELE141AdvantageMean:
    """TELE-141: Advantage mean measures normalization centering."""

    def test_advantage_mean_in_payload(self, capture_hub: CaptureHubResult):
        """TELE-141: advantage_mean field is present and matches input."""
        hub, backend = capture_hub

        payload = _make_base_ppo_payload(advantage_mean=0.025)
        _emit_ppo_update(hub, payload)

        event = backend.find_first(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert event is not None
        assert event.data.advantage_mean == pytest.approx(0.025, abs=1e-6)

    def test_advantage_mean_near_zero(self, capture_hub: CaptureHubResult):
        """TELE-141: Healthy advantage_mean near 0.0 (well-centered)."""
        hub, backend = capture_hub

        payload = _make_base_ppo_payload(advantage_mean=-0.001)
        _emit_ppo_update(hub, payload)

        event = backend.find_first(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert event is not None
        assert abs(event.data.advantage_mean) < 0.1

    def test_advantage_mean_biased(self, capture_hub: CaptureHubResult):
        """TELE-141: Biased advantage_mean (away from zero) is captured."""
        hub, backend = capture_hub

        payload = _make_base_ppo_payload(advantage_mean=0.35)
        _emit_ppo_update(hub, payload)

        event = backend.find_first(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert event is not None
        assert event.data.advantage_mean == pytest.approx(0.35, abs=1e-6)


class TestTELE142AdvantageSkewness:
    """TELE-142: Advantage skewness measures distribution asymmetry."""

    def test_advantage_skewness_in_payload(self, capture_hub: CaptureHubResult):
        """TELE-142: advantage_skewness field is present and matches input."""
        hub, backend = capture_hub

        payload = _make_base_ppo_payload(advantage_skewness=0.3)
        _emit_ppo_update(hub, payload)

        event = backend.find_first(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert event is not None
        assert event.data.advantage_skewness == pytest.approx(0.3, abs=1e-6)

    def test_advantage_skewness_symmetric(self, capture_hub: CaptureHubResult):
        """TELE-142: Symmetric distribution (skewness ~ 0) is captured."""
        hub, backend = capture_hub

        payload = _make_base_ppo_payload(advantage_skewness=0.0)
        _emit_ppo_update(hub, payload)

        event = backend.find_first(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert event is not None
        assert abs(event.data.advantage_skewness) < 0.5

    def test_advantage_skewness_right_skewed(self, capture_hub: CaptureHubResult):
        """TELE-142: Right-skewed distribution (positive skewness) is captured."""
        hub, backend = capture_hub

        payload = _make_base_ppo_payload(advantage_skewness=1.5)
        _emit_ppo_update(hub, payload)

        event = backend.find_first(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert event is not None
        assert event.data.advantage_skewness == pytest.approx(1.5, abs=1e-6)

    def test_advantage_skewness_left_skewed(self, capture_hub: CaptureHubResult):
        """TELE-142: Left-skewed distribution (negative skewness) is captured."""
        hub, backend = capture_hub

        payload = _make_base_ppo_payload(advantage_skewness=-0.8)
        _emit_ppo_update(hub, payload)

        event = backend.find_first(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert event is not None
        assert event.data.advantage_skewness == pytest.approx(-0.8, abs=1e-6)


class TestTELE143AdvantageKurtosis:
    """TELE-143: Advantage kurtosis measures distribution tail heaviness."""

    def test_advantage_kurtosis_in_payload(self, capture_hub: CaptureHubResult):
        """TELE-143: advantage_kurtosis field is present and matches input."""
        hub, backend = capture_hub

        payload = _make_base_ppo_payload(advantage_kurtosis=0.5)
        _emit_ppo_update(hub, payload)

        event = backend.find_first(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert event is not None
        assert event.data.advantage_kurtosis == pytest.approx(0.5, abs=1e-6)

    def test_advantage_kurtosis_normal(self, capture_hub: CaptureHubResult):
        """TELE-143: Normal distribution (kurtosis ~ 0) is captured."""
        hub, backend = capture_hub

        payload = _make_base_ppo_payload(advantage_kurtosis=0.0)
        _emit_ppo_update(hub, payload)

        event = backend.find_first(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert event is not None
        assert abs(event.data.advantage_kurtosis) < 3.0

    def test_advantage_kurtosis_heavy_tails(self, capture_hub: CaptureHubResult):
        """TELE-143: Heavy-tailed distribution (high kurtosis) is captured."""
        hub, backend = capture_hub

        payload = _make_base_ppo_payload(advantage_kurtosis=5.0)
        _emit_ppo_update(hub, payload)

        event = backend.find_first(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert event is not None
        assert event.data.advantage_kurtosis == pytest.approx(5.0, abs=1e-6)


class TestTELE144AdvantagePositiveRatio:
    """TELE-144: Fraction of positive advantages measures exploration balance."""

    def test_advantage_positive_ratio_in_payload(self, capture_hub: CaptureHubResult):
        """TELE-144: advantage_positive_ratio field is present and matches input."""
        hub, backend = capture_hub

        payload = _make_base_ppo_payload(advantage_positive_ratio=0.55)
        _emit_ppo_update(hub, payload)

        event = backend.find_first(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert event is not None
        assert event.data.advantage_positive_ratio == pytest.approx(0.55, abs=1e-6)

    def test_advantage_positive_ratio_balanced(self, capture_hub: CaptureHubResult):
        """TELE-144: Balanced ratio (near 0.5) indicates symmetric exploration."""
        hub, backend = capture_hub

        payload = _make_base_ppo_payload(advantage_positive_ratio=0.48)
        _emit_ppo_update(hub, payload)

        event = backend.find_first(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert event is not None
        assert 0.4 <= event.data.advantage_positive_ratio <= 0.6

    def test_advantage_positive_ratio_biased_high(self, capture_hub: CaptureHubResult):
        """TELE-144: High ratio (>0.8) indicates positive bias is captured."""
        hub, backend = capture_hub

        payload = _make_base_ppo_payload(advantage_positive_ratio=0.85)
        _emit_ppo_update(hub, payload)

        event = backend.find_first(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert event is not None
        assert event.data.advantage_positive_ratio == pytest.approx(0.85, abs=1e-6)

    def test_advantage_positive_ratio_biased_low(self, capture_hub: CaptureHubResult):
        """TELE-144: Low ratio (<0.2) indicates negative bias is captured."""
        hub, backend = capture_hub

        payload = _make_base_ppo_payload(advantage_positive_ratio=0.15)
        _emit_ppo_update(hub, payload)

        event = backend.find_first(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert event is not None
        assert event.data.advantage_positive_ratio == pytest.approx(0.15, abs=1e-6)


# =============================================================================
# TELE-150/151: Ratio Max Metrics
# =============================================================================


class TestTELE150JointRatioMax:
    """TELE-150: Joint ratio max detects multi-head policy instability."""

    def test_joint_ratio_max_in_payload(self, capture_hub: CaptureHubResult):
        """TELE-150: joint_ratio_max field is present and matches input."""
        hub, backend = capture_hub

        payload = _make_base_ppo_payload(joint_ratio_max=1.8)
        _emit_ppo_update(hub, payload)

        event = backend.find_first(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert event is not None
        assert event.data.joint_ratio_max == pytest.approx(1.8, abs=1e-6)

    def test_joint_ratio_max_default_one(self, capture_hub: CaptureHubResult):
        """TELE-150: Default joint_ratio_max of 1.0 indicates no policy change."""
        hub, backend = capture_hub

        # Use default (1.0)
        payload = _make_base_ppo_payload()
        _emit_ppo_update(hub, payload)

        event = backend.find_first(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert event is not None
        assert event.data.joint_ratio_max == pytest.approx(1.0, abs=1e-6)

    def test_joint_ratio_max_high_value(self, capture_hub: CaptureHubResult):
        """TELE-150: High joint ratio (>3.0 critical) is captured."""
        hub, backend = capture_hub

        payload = _make_base_ppo_payload(joint_ratio_max=4.5)
        _emit_ppo_update(hub, payload)

        event = backend.find_first(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert event is not None
        assert event.data.joint_ratio_max == pytest.approx(4.5, abs=1e-6)


class TestTELE151HeadRatioMax:
    """TELE-151: Per-head ratio max for granular policy stability tracking."""

    def test_head_ratio_max_all_heads_present(self, capture_hub: CaptureHubResult):
        """TELE-151: All 8 head ratio max fields are present in payload."""
        hub, backend = capture_hub

        payload = _make_base_ppo_payload(
            head_slot_ratio_max=1.1,
            head_blueprint_ratio_max=1.2,
            head_style_ratio_max=1.15,
            head_tempo_ratio_max=1.08,
            head_alpha_target_ratio_max=1.05,
            head_alpha_speed_ratio_max=1.12,
            head_alpha_curve_ratio_max=1.03,
            head_op_ratio_max=1.25,
        )
        _emit_ppo_update(hub, payload)

        event = backend.find_first(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert event is not None

        # Verify all head ratios
        assert event.data.head_slot_ratio_max == pytest.approx(1.1, abs=1e-6)
        assert event.data.head_blueprint_ratio_max == pytest.approx(1.2, abs=1e-6)
        assert event.data.head_style_ratio_max == pytest.approx(1.15, abs=1e-6)
        assert event.data.head_tempo_ratio_max == pytest.approx(1.08, abs=1e-6)
        assert event.data.head_alpha_target_ratio_max == pytest.approx(1.05, abs=1e-6)
        assert event.data.head_alpha_speed_ratio_max == pytest.approx(1.12, abs=1e-6)
        assert event.data.head_alpha_curve_ratio_max == pytest.approx(1.03, abs=1e-6)
        assert event.data.head_op_ratio_max == pytest.approx(1.25, abs=1e-6)

    def test_head_ratio_max_defaults(self, capture_hub: CaptureHubResult):
        """TELE-151: Default head ratio max values are 1.0."""
        hub, backend = capture_hub

        payload = _make_base_ppo_payload()
        _emit_ppo_update(hub, payload)

        event = backend.find_first(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert event is not None

        # All should default to 1.0
        assert event.data.head_slot_ratio_max == 1.0
        assert event.data.head_blueprint_ratio_max == 1.0
        assert event.data.head_op_ratio_max == 1.0


# =============================================================================
# TELE-160/161: Log Probability Extremes
# =============================================================================


class TestTELE160LogProbMin:
    """TELE-160: Minimum log probability for NaN risk detection."""

    def test_log_prob_min_in_payload(self, capture_hub: CaptureHubResult):
        """TELE-160: log_prob_min field is present and matches input."""
        hub, backend = capture_hub

        payload = _make_base_ppo_payload(log_prob_min=-12.5)
        _emit_ppo_update(hub, payload)

        event = backend.find_first(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert event is not None
        assert event.data.log_prob_min == pytest.approx(-12.5, abs=1e-6)

    def test_log_prob_min_healthy_range(self, capture_hub: CaptureHubResult):
        """TELE-160: Healthy log_prob_min (> -50) is captured."""
        hub, backend = capture_hub

        payload = _make_base_ppo_payload(log_prob_min=-8.0)
        _emit_ppo_update(hub, payload)

        event = backend.find_first(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert event is not None
        assert event.data.log_prob_min > -50

    def test_log_prob_min_warning_range(self, capture_hub: CaptureHubResult):
        """TELE-160: Warning log_prob_min (-100 < x <= -50) is captured."""
        hub, backend = capture_hub

        payload = _make_base_ppo_payload(log_prob_min=-75.0)
        _emit_ppo_update(hub, payload)

        event = backend.find_first(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert event is not None
        assert -100 < event.data.log_prob_min <= -50

    def test_log_prob_min_critical_range(self, capture_hub: CaptureHubResult):
        """TELE-160: Critical log_prob_min (<= -100) is captured."""
        hub, backend = capture_hub

        payload = _make_base_ppo_payload(log_prob_min=-120.0)
        _emit_ppo_update(hub, payload)

        event = backend.find_first(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert event is not None
        assert event.data.log_prob_min == pytest.approx(-120.0, abs=1e-6)


class TestTELE161LogProbMax:
    """TELE-161: Maximum log probability for range display."""

    def test_log_prob_max_in_payload(self, capture_hub: CaptureHubResult):
        """TELE-161: log_prob_max field is present and matches input."""
        hub, backend = capture_hub

        payload = _make_base_ppo_payload(log_prob_max=-0.5)
        _emit_ppo_update(hub, payload)

        event = backend.find_first(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert event is not None
        assert event.data.log_prob_max == pytest.approx(-0.5, abs=1e-6)

    def test_log_prob_max_near_zero(self, capture_hub: CaptureHubResult):
        """TELE-161: log_prob_max near 0 (high probability action) is valid."""
        hub, backend = capture_hub

        payload = _make_base_ppo_payload(log_prob_max=-0.01)
        _emit_ppo_update(hub, payload)

        event = backend.find_first(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert event is not None
        assert event.data.log_prob_max == pytest.approx(-0.01, abs=1e-6)

    def test_log_prob_range_emitted(self, capture_hub: CaptureHubResult):
        """TELE-160/161: Both log_prob_min and log_prob_max form a valid range."""
        hub, backend = capture_hub

        payload = _make_base_ppo_payload(log_prob_min=-15.0, log_prob_max=-0.3)
        _emit_ppo_update(hub, payload)

        event = backend.find_first(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert event is not None
        assert event.data.log_prob_min < event.data.log_prob_max
        assert event.data.log_prob_max <= 0  # Log probs are always <= 0


# =============================================================================
# TELE-170: Per-Head Entropies
# =============================================================================


class TestTELE170HeadEntropies:
    """TELE-170: Per-head entropy for granular collapse detection."""

    def test_all_head_entropies_present(self, capture_hub: CaptureHubResult):
        """TELE-170: All 8 head entropy fields can be populated."""
        hub, backend = capture_hub

        payload = _make_base_ppo_payload(
            head_slot_entropy=0.95,
            head_blueprint_entropy=2.1,
            head_style_entropy=1.2,
            head_tempo_entropy=0.85,
            head_alpha_target_entropy=0.9,
            head_alpha_speed_entropy=1.1,
            head_alpha_curve_entropy=1.3,
            head_op_entropy=1.5,
        )
        _emit_ppo_update(hub, payload)

        event = backend.find_first(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert event is not None

        # Verify all head entropies
        assert event.data.head_slot_entropy == pytest.approx(0.95, abs=1e-6)
        assert event.data.head_blueprint_entropy == pytest.approx(2.1, abs=1e-6)
        assert event.data.head_style_entropy == pytest.approx(1.2, abs=1e-6)
        assert event.data.head_tempo_entropy == pytest.approx(0.85, abs=1e-6)
        assert event.data.head_alpha_target_entropy == pytest.approx(0.9, abs=1e-6)
        assert event.data.head_alpha_speed_entropy == pytest.approx(1.1, abs=1e-6)
        assert event.data.head_alpha_curve_entropy == pytest.approx(1.3, abs=1e-6)
        assert event.data.head_op_entropy == pytest.approx(1.5, abs=1e-6)

    def test_head_entropies_optional(self, capture_hub: CaptureHubResult):
        """TELE-170: Head entropies are optional (None when not provided)."""
        hub, backend = capture_hub

        # Base payload doesn't include head entropies
        payload = _make_base_ppo_payload()
        _emit_ppo_update(hub, payload)

        event = backend.find_first(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert event is not None

        # All should be None by default
        assert event.data.head_slot_entropy is None
        assert event.data.head_blueprint_entropy is None
        assert event.data.head_op_entropy is None

    def test_single_head_collapsed_entropy(self, capture_hub: CaptureHubResult):
        """TELE-170: Individual head with collapsed entropy is captured."""
        hub, backend = capture_hub

        # One head collapsed, others healthy
        payload = _make_base_ppo_payload(
            head_slot_entropy=0.05,  # Collapsed (< 25% of max 1.10)
            head_blueprint_entropy=2.0,  # Healthy
            head_op_entropy=1.5,  # Healthy
        )
        _emit_ppo_update(hub, payload)

        event = backend.find_first(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert event is not None
        assert event.data.head_slot_entropy == pytest.approx(0.05, abs=1e-6)

    def test_head_entropies_at_maximum(self, capture_hub: CaptureHubResult):
        """TELE-170: Head entropies at maximum (uniform distribution) are valid."""
        hub, backend = capture_hub

        # Near-maximum entropies (uniform exploration)
        payload = _make_base_ppo_payload(
            head_slot_entropy=1.09,  # Max for 3-action head is log(3) = 1.10
            head_blueprint_entropy=2.5,  # Max for 13-action head is log(13) = 2.56
            head_op_entropy=1.75,  # Max for 6-action head is log(6) = 1.79
        )
        _emit_ppo_update(hub, payload)

        event = backend.find_first(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert event is not None
        assert event.data.head_slot_entropy == pytest.approx(1.09, abs=1e-6)
        assert event.data.head_blueprint_entropy == pytest.approx(2.5, abs=1e-6)
        assert event.data.head_op_entropy == pytest.approx(1.75, abs=1e-6)


# =============================================================================
# Integration Tests: Complete Payload Flow
# =============================================================================


class TestPolicyMetricsIntegration:
    """Integration tests verifying complete policy metric flows."""

    def test_full_ppo_payload_round_trip(self, capture_hub: CaptureHubResult):
        """All policy metrics survive the full emit -> capture path."""
        hub, backend = capture_hub

        # Create a payload with all fields set to distinct values
        payload = _make_base_ppo_payload(
            policy_loss=0.123,
            entropy=1.456,
            kl_divergence=0.0234,
            clip_fraction=0.178,
            entropy_collapsed=False,
            advantage_mean=0.045,
            advantage_std=0.987,
            advantage_skewness=0.234,
            advantage_kurtosis=0.567,
            advantage_positive_ratio=0.523,
            log_prob_min=-9.876,
            log_prob_max=-0.432,
            clip_fraction_positive=0.089,
            clip_fraction_negative=0.067,
        )
        _emit_ppo_update(hub, payload)

        event = backend.find_first(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert event is not None

        # Verify all values match
        assert event.data.policy_loss == pytest.approx(0.123, abs=1e-6)
        assert event.data.entropy == pytest.approx(1.456, abs=1e-6)
        assert event.data.kl_divergence == pytest.approx(0.0234, abs=1e-6)
        assert event.data.clip_fraction == pytest.approx(0.178, abs=1e-6)
        assert event.data.entropy_collapsed is False
        assert event.data.advantage_mean == pytest.approx(0.045, abs=1e-6)
        assert event.data.advantage_std == pytest.approx(0.987, abs=1e-6)
        assert event.data.advantage_skewness == pytest.approx(0.234, abs=1e-6)
        assert event.data.advantage_kurtosis == pytest.approx(0.567, abs=1e-6)
        assert event.data.advantage_positive_ratio == pytest.approx(0.523, abs=1e-6)
        assert event.data.log_prob_min == pytest.approx(-9.876, abs=1e-6)
        assert event.data.log_prob_max == pytest.approx(-0.432, abs=1e-6)
        assert event.data.clip_fraction_positive == pytest.approx(0.089, abs=1e-6)
        assert event.data.clip_fraction_negative == pytest.approx(0.067, abs=1e-6)

    def test_multiple_updates_captured_in_order(self, capture_hub: CaptureHubResult):
        """Multiple PPO updates are captured with correct ordering."""
        hub, backend = capture_hub

        # Emit 5 updates with different entropy values
        for i in range(5):
            payload = _make_base_ppo_payload(entropy=1.0 + i * 0.1)
            _emit_ppo_update(hub, payload, epoch=i + 1)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert len(events) == 5

        # Verify ordering
        for i, event in enumerate(events):
            expected_entropy = 1.0 + i * 0.1
            assert event.data.entropy == pytest.approx(expected_entropy, abs=1e-6)
            assert event.epoch == i + 1

    def test_nan_values_handled(self, capture_hub: CaptureHubResult):
        """NaN values in optional fields are handled correctly."""
        hub, backend = capture_hub

        # Create payload with NaN in explained_variance (optional field)
        payload = _make_base_ppo_payload(explained_variance=float("nan"))
        _emit_ppo_update(hub, payload)

        event = backend.find_first(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert event is not None
        assert math.isnan(event.data.explained_variance)
