"""End-to-end tests for value metrics (TELE-200 to TELE-299).

Verifies value function telemetry flows from PPOAgent through to nissa.
These metrics are critical for detecting value function health issues:
- explained_variance: How well the value function predicts returns
- value_mean/std/min/max: Distribution of value predictions
- initial_value_spread: Baseline for relative explosion detection

All metrics are emitted in PPO_UPDATE_COMPLETED events via PPOUpdatePayload.
"""


from esper.leyline import TelemetryEvent, TelemetryEventType
from esper.leyline.telemetry import PPOUpdatePayload
from esper.nissa.output import NissaHub

from tests.telemetry.conftest import CaptureHubResult


def _make_ppo_update_payload(**overrides) -> PPOUpdatePayload:
    """Create PPOUpdatePayload with required fields and overrides.

    This ensures tests fail loudly if required fields are missing.
    All optional value fields have sensible defaults for testing.
    """
    base = {
        # Required core PPO metrics
        "policy_loss": 0.5,
        "value_loss": 0.3,
        "entropy": 1.2,
        "grad_norm": 1.0,
        "kl_divergence": 0.01,
        "clip_fraction": 0.15,
        "nan_grad_count": 0,
        "pre_clip_grad_norm": 1.5,
        # Required advantage stats
        "advantage_mean": 0.0,
        "advantage_std": 1.0,
        "advantage_skewness": 0.0,
        "advantage_kurtosis": 0.0,
        "advantage_positive_ratio": 0.5,
        # Required ratio stats
        "ratio_mean": 1.0,
        "ratio_min": 0.8,
        "ratio_max": 1.2,
        "ratio_std": 0.1,
        # Required log prob extremes
        "log_prob_min": -5.0,
        "log_prob_max": -0.1,
        # Required value stats (the metrics we're testing)
        "value_mean": 0.0,
        "value_std": 1.0,
        "value_min": -2.0,
        "value_max": 2.0,
        # Required Q-value diagnostics
        "op_q_values": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "op_valid_mask": (True, True, True, True, True, True),
        "q_variance": 0.0,
        "q_spread": 0.0,
        # Required diagnostics
        "entropy_collapsed": False,
        "update_time_ms": 50.0,
        "inner_epoch": 1,
        "batch": 0,
        "ppo_updates_count": 4,
        "clip_fraction_positive": 0.1,
        "clip_fraction_negative": 0.05,
        "gradient_cv": 0.3,
        "pre_norm_advantage_mean": 0.0,
        "pre_norm_advantage_std": 1.0,
        "return_mean": 0.5,
        "return_std": 0.2,
    }
    base.update(overrides)
    return PPOUpdatePayload(**base)


def _emit_and_flush(hub: NissaHub, event: TelemetryEvent) -> None:
    """Emit event and flush to ensure it's processed before assertions."""
    hub.emit(event)
    hub.flush(timeout=1.0)


class TestTELE200ExplainedVariance:
    """TELE-200: Explained Variance metric.

    Explained Variance (EV) measures how well the value function predicts returns:
    EV = 1 - (Var(returns - values) / Var(returns))

    - EV = 1.0: Perfect prediction
    - EV = 0.5: Explains 50% of return variance
    - EV = 0.0: No better than random
    - EV < 0.0: Worse than random (pathological)
    """

    def test_explained_variance_in_ppo_update_payload(self, capture_hub: CaptureHubResult):
        """TELE-200: explained_variance field is present in PPO_UPDATE_COMPLETED."""
        hub, backend = capture_hub

        payload = _make_ppo_update_payload(explained_variance=0.75)

        event = TelemetryEvent(
            event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
            data=payload,
        )
        _emit_and_flush(hub, event)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert len(events) == 1, "Expected PPO_UPDATE_COMPLETED event"
        assert events[0].data.explained_variance == 0.75

    def test_explained_variance_healthy_value(self, capture_hub: CaptureHubResult):
        """TELE-200: Healthy EV is >0.5 (value function strong)."""
        hub, backend = capture_hub

        payload = _make_ppo_update_payload(explained_variance=0.82)

        event = TelemetryEvent(
            event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
            data=payload,
        )
        _emit_and_flush(hub, event)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        ev = events[0].data.explained_variance
        assert ev is not None
        assert ev > 0.5, "EV > 0.5 indicates healthy value function"

    def test_explained_variance_warning_value(self, capture_hub: CaptureHubResult):
        """TELE-200: Warning EV is 0.0 <= EV <= 0.3 (value function weak)."""
        hub, backend = capture_hub

        payload = _make_ppo_update_payload(explained_variance=0.25)

        event = TelemetryEvent(
            event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
            data=payload,
        )
        _emit_and_flush(hub, event)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        ev = events[0].data.explained_variance
        assert ev is not None
        assert 0.0 <= ev <= 0.3, "EV in 0.0-0.3 indicates warning threshold"

    def test_explained_variance_critical_negative(self, capture_hub: CaptureHubResult):
        """TELE-200: Critical EV is < 0.0 (value function worse than random)."""
        hub, backend = capture_hub

        # Negative EV indicates value function is hurting, not helping
        payload = _make_ppo_update_payload(explained_variance=-0.1)

        event = TelemetryEvent(
            event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
            data=payload,
        )
        _emit_and_flush(hub, event)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        ev = events[0].data.explained_variance
        assert ev is not None
        assert ev < 0.0, "Negative EV indicates critical value function failure"

    def test_explained_variance_none_early_training(self, capture_hub: CaptureHubResult):
        """TELE-200: EV can be None early in training (before first update)."""
        hub, backend = capture_hub

        payload = _make_ppo_update_payload(explained_variance=None)

        event = TelemetryEvent(
            event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
            data=payload,
        )
        _emit_and_flush(hub, event)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert events[0].data.explained_variance is None


class TestTELE210ValueMean:
    """TELE-210: Value Mean metric.

    The mean of all value function predictions from the critic network in a batch.
    V(s) = E[G_t | s_t = s], where G_t is the discounted cumulative future reward.
    """

    def test_value_mean_in_ppo_update_payload(self, capture_hub: CaptureHubResult):
        """TELE-210: value_mean field is present in PPO_UPDATE_COMPLETED."""
        hub, backend = capture_hub

        payload = _make_ppo_update_payload(value_mean=5.5)

        event = TelemetryEvent(
            event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
            data=payload,
        )
        _emit_and_flush(hub, event)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert len(events) == 1
        assert events[0].data.value_mean == 5.5

    def test_value_mean_positive(self, capture_hub: CaptureHubResult):
        """TELE-210: value_mean can be positive (expected returns are positive)."""
        hub, backend = capture_hub

        payload = _make_ppo_update_payload(value_mean=10.0)

        event = TelemetryEvent(
            event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
            data=payload,
        )
        _emit_and_flush(hub, event)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert events[0].data.value_mean > 0

    def test_value_mean_negative(self, capture_hub: CaptureHubResult):
        """TELE-210: value_mean can be negative (expected returns are negative)."""
        hub, backend = capture_hub

        payload = _make_ppo_update_payload(value_mean=-3.5)

        event = TelemetryEvent(
            event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
            data=payload,
        )
        _emit_and_flush(hub, event)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert events[0].data.value_mean < 0

    def test_value_mean_near_zero(self, capture_hub: CaptureHubResult):
        """TELE-210: value_mean near zero is valid (balanced rewards)."""
        hub, backend = capture_hub

        payload = _make_ppo_update_payload(value_mean=0.001)

        event = TelemetryEvent(
            event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
            data=payload,
        )
        _emit_and_flush(hub, event)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert abs(events[0].data.value_mean) < 0.01


class TestTELE211ValueStd:
    """TELE-211: Value Std metric.

    Standard deviation of value function outputs across the batch.
    Low std = value function predicting constant outputs (collapse risk).
    """

    def test_value_std_in_ppo_update_payload(self, capture_hub: CaptureHubResult):
        """TELE-211: value_std field is present in PPO_UPDATE_COMPLETED."""
        hub, backend = capture_hub

        payload = _make_ppo_update_payload(value_std=1.2)

        event = TelemetryEvent(
            event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
            data=payload,
        )
        _emit_and_flush(hub, event)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert len(events) == 1
        assert events[0].data.value_std == 1.2

    def test_value_std_healthy(self, capture_hub: CaptureHubResult):
        """TELE-211: Healthy value_std > 0.01 (diverse predictions)."""
        hub, backend = capture_hub

        payload = _make_ppo_update_payload(value_std=0.5)

        event = TelemetryEvent(
            event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
            data=payload,
        )
        _emit_and_flush(hub, event)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        v_std = events[0].data.value_std
        assert v_std > 0.01, "value_std > 0.01 indicates healthy diversity"

    def test_value_std_collapse_warning(self, capture_hub: CaptureHubResult):
        """TELE-211: value_std < 0.01 indicates potential collapse."""
        hub, backend = capture_hub

        # Very low std could indicate value collapse
        payload = _make_ppo_update_payload(
            value_std=0.005,
            value_min=0.0,
            value_max=0.05,  # Also small range
        )

        event = TelemetryEvent(
            event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
            data=payload,
        )
        _emit_and_flush(hub, event)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        v_std = events[0].data.value_std
        assert v_std < 0.01, "value_std < 0.01 indicates collapse risk"

    def test_value_std_nonnegative(self, capture_hub: CaptureHubResult):
        """TELE-211: value_std is always non-negative."""
        hub, backend = capture_hub

        payload = _make_ppo_update_payload(value_std=0.0)

        event = TelemetryEvent(
            event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
            data=payload,
        )
        _emit_and_flush(hub, event)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert events[0].data.value_std >= 0.0


class TestTELE212ValueMin:
    """TELE-212: Value Min metric.

    The minimum value prediction across all valid states in the batch.
    Used for value collapse and explosion detection.
    """

    def test_value_min_in_ppo_update_payload(self, capture_hub: CaptureHubResult):
        """TELE-212: value_min field is present in PPO_UPDATE_COMPLETED."""
        hub, backend = capture_hub

        payload = _make_ppo_update_payload(value_min=-3.0)

        event = TelemetryEvent(
            event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
            data=payload,
        )
        _emit_and_flush(hub, event)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert len(events) == 1
        assert events[0].data.value_min == -3.0

    def test_value_min_less_than_max(self, capture_hub: CaptureHubResult):
        """TELE-212: value_min <= value_max by definition."""
        hub, backend = capture_hub

        payload = _make_ppo_update_payload(
            value_min=-5.0,
            value_max=10.0,
        )

        event = TelemetryEvent(
            event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
            data=payload,
        )
        _emit_and_flush(hub, event)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        data = events[0].data
        assert data.value_min <= data.value_max

    def test_value_min_extreme_negative(self, capture_hub: CaptureHubResult):
        """TELE-212: Extreme negative value_min may indicate explosion."""
        hub, backend = capture_hub

        # Very negative min could indicate value explosion
        payload = _make_ppo_update_payload(
            value_min=-5000.0,
            value_max=100.0,
        )

        event = TelemetryEvent(
            event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
            data=payload,
        )
        _emit_and_flush(hub, event)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        v_min = events[0].data.value_min
        assert v_min < -1000, "Extreme negative value_min may indicate explosion"

    def test_value_min_forms_range_with_max(self, capture_hub: CaptureHubResult):
        """TELE-212: value_min and value_max together form the value range."""
        hub, backend = capture_hub

        payload = _make_ppo_update_payload(
            value_min=-2.0,
            value_max=8.0,
        )

        event = TelemetryEvent(
            event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
            data=payload,
        )
        _emit_and_flush(hub, event)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        data = events[0].data
        value_range = data.value_max - data.value_min
        assert value_range == 10.0, "Value range = max - min"


class TestTELE213ValueMax:
    """TELE-213: Value Max metric.

    The maximum value prediction among all states in the batch.
    Detects value function explosion when critic predicts unreasonably high values.
    """

    def test_value_max_in_ppo_update_payload(self, capture_hub: CaptureHubResult):
        """TELE-213: value_max field is present in PPO_UPDATE_COMPLETED."""
        hub, backend = capture_hub

        payload = _make_ppo_update_payload(value_max=15.0)

        event = TelemetryEvent(
            event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
            data=payload,
        )
        _emit_and_flush(hub, event)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert len(events) == 1
        assert events[0].data.value_max == 15.0

    def test_value_max_healthy(self, capture_hub: CaptureHubResult):
        """TELE-213: Healthy value_max is < 5000 (absolute threshold)."""
        hub, backend = capture_hub

        payload = _make_ppo_update_payload(value_max=50.0)

        event = TelemetryEvent(
            event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
            data=payload,
        )
        _emit_and_flush(hub, event)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        v_max = events[0].data.value_max
        assert abs(v_max) < 5000, "Healthy value_max < 5000"

    def test_value_max_warning(self, capture_hub: CaptureHubResult):
        """TELE-213: Warning when abs(value_max) >= 5000."""
        hub, backend = capture_hub

        payload = _make_ppo_update_payload(value_max=7500.0)

        event = TelemetryEvent(
            event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
            data=payload,
        )
        _emit_and_flush(hub, event)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        v_max = events[0].data.value_max
        assert abs(v_max) >= 5000, "Warning threshold at 5000"
        assert abs(v_max) < 10000, "Below critical threshold"

    def test_value_max_critical(self, capture_hub: CaptureHubResult):
        """TELE-213: Critical when abs(value_max) >= 10000."""
        hub, backend = capture_hub

        payload = _make_ppo_update_payload(value_max=15000.0)

        event = TelemetryEvent(
            event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
            data=payload,
        )
        _emit_and_flush(hub, event)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        v_max = events[0].data.value_max
        assert abs(v_max) >= 10000, "Critical threshold at 10000"


class TestTELE214InitialValueSpread:
    """TELE-214: Initial Value Spread metric.

    Captures the first stable estimate of value range (max - min) after warmup.
    Used to compute relative explosion ratio: current_range / initial_spread.

    Note: initial_value_spread is NOT a field in PPOUpdatePayload - it's
    computed by SanctumAggregator after warmup. We test that the components
    needed for its computation (value_min, value_max) are properly emitted.
    """

    def test_value_range_components_emitted(self, capture_hub: CaptureHubResult):
        """TELE-214: value_min and value_max are emitted for spread calculation."""
        hub, backend = capture_hub

        payload = _make_ppo_update_payload(
            value_min=-5.0,
            value_max=10.0,
        )

        event = TelemetryEvent(
            event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
            data=payload,
        )
        _emit_and_flush(hub, event)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        data = events[0].data

        # Both components needed for initial_value_spread calculation
        assert data.value_min == -5.0
        assert data.value_max == 10.0

        # Spread would be computed as: max - min = 15.0
        computed_spread = data.value_max - data.value_min
        assert computed_spread == 15.0

    def test_value_range_for_relative_threshold(self, capture_hub: CaptureHubResult):
        """TELE-214: Relative thresholds use ratio = current_range / initial_spread.

        Warning at ratio > 5, Critical at ratio > 10.
        """
        hub, backend = capture_hub

        # Simulate an initial spread of 10.0
        initial_spread = 10.0

        # Current values with range of 80.0 (8x initial - should be warning)
        payload = _make_ppo_update_payload(
            value_min=-30.0,
            value_max=50.0,
        )

        event = TelemetryEvent(
            event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
            data=payload,
        )
        _emit_and_flush(hub, event)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        data = events[0].data

        current_range = data.value_max - data.value_min
        ratio = current_range / initial_spread

        assert ratio == 8.0, "Ratio = 80 / 10 = 8x"
        assert 5.0 <= ratio <= 10.0, "8x is in warning range (5-10)"

    def test_value_range_collapse_detection(self, capture_hub: CaptureHubResult):
        """TELE-214: Small range + small std indicates value collapse."""
        hub, backend = capture_hub

        # Very small range and std indicates collapse
        payload = _make_ppo_update_payload(
            value_min=0.01,
            value_max=0.05,  # Range = 0.04 (< 0.1 threshold)
            value_std=0.005,  # Std < 0.01 threshold
        )

        event = TelemetryEvent(
            event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
            data=payload,
        )
        _emit_and_flush(hub, event)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        data = events[0].data

        value_range = data.value_max - data.value_min
        assert value_range < 0.1, "Range < 0.1 indicates potential collapse"
        assert data.value_std < 0.01, "Std < 0.01 confirms collapse"


class TestValueMetricsIntegration:
    """Integration tests for value metrics cluster.

    Value metrics (mean, std, min, max) are always emitted together
    to enable joint analysis of value function health.
    """

    def test_all_value_metrics_emitted_together(self, capture_hub: CaptureHubResult):
        """All value metrics are present in a single PPO_UPDATE_COMPLETED event."""
        hub, backend = capture_hub

        payload = _make_ppo_update_payload(
            explained_variance=0.65,
            value_mean=2.5,
            value_std=1.5,
            value_min=-3.0,
            value_max=8.0,
        )

        event = TelemetryEvent(
            event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
            data=payload,
        )
        _emit_and_flush(hub, event)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert len(events) == 1

        data = events[0].data

        # All value metrics should be present
        assert data.explained_variance == 0.65
        assert data.value_mean == 2.5
        assert data.value_std == 1.5
        assert data.value_min == -3.0
        assert data.value_max == 8.0

    def test_coefficient_of_variation_calculable(self, capture_hub: CaptureHubResult):
        """value_mean and value_std enable CoV calculation (std/|mean|)."""
        hub, backend = capture_hub

        payload = _make_ppo_update_payload(
            value_mean=5.0,
            value_std=1.5,
        )

        event = TelemetryEvent(
            event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
            data=payload,
        )
        _emit_and_flush(hub, event)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        data = events[0].data

        # CoV = std / |mean|
        cov = data.value_std / abs(data.value_mean)
        assert cov == 0.3, "CoV = 1.5 / 5.0 = 0.3"

        # CoV < 2.0 is healthy
        assert cov < 2.0, "Low CoV indicates stable value estimates"

    def test_high_cov_warning(self, capture_hub: CaptureHubResult):
        """High CoV (> 2.0) indicates value function uncertainty."""
        hub, backend = capture_hub

        payload = _make_ppo_update_payload(
            value_mean=1.0,
            value_std=2.5,  # CoV = 2.5 (warning level)
        )

        event = TelemetryEvent(
            event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
            data=payload,
        )
        _emit_and_flush(hub, event)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        data = events[0].data

        cov = data.value_std / abs(data.value_mean)
        assert cov > 2.0, "CoV > 2.0 indicates warning"
        assert cov < 3.0, "CoV < 3.0 is not yet critical"

    def test_extreme_cov_critical(self, capture_hub: CaptureHubResult):
        """Extreme CoV (> 3.0) indicates critical value function failure."""
        hub, backend = capture_hub

        payload = _make_ppo_update_payload(
            value_mean=1.0,
            value_std=4.0,  # CoV = 4.0 (critical level)
        )

        event = TelemetryEvent(
            event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
            data=payload,
        )
        _emit_and_flush(hub, event)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        data = events[0].data

        cov = data.value_std / abs(data.value_mean)
        assert cov > 3.0, "CoV > 3.0 indicates critical"

    def test_multiple_updates_accumulate(self, capture_hub: CaptureHubResult):
        """Multiple PPO updates are all captured independently."""
        hub, backend = capture_hub

        # Emit 3 updates with different value metrics
        for i in range(3):
            payload = _make_ppo_update_payload(
                explained_variance=0.3 * (i + 1),  # 0.3, 0.6, 0.9
                value_mean=float(i + 1),  # 1.0, 2.0, 3.0
                value_std=0.5 * (i + 1),  # 0.5, 1.0, 1.5
                value_min=-float(i + 1),  # -1, -2, -3
                value_max=float((i + 1) * 2),  # 2, 4, 6
            )

            event = TelemetryEvent(
                event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
                data=payload,
            )
            _emit_and_flush(hub, event)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert len(events) == 3

        # Verify each event has its own values
        for i, event in enumerate(events):
            expected_ev = 0.3 * (i + 1)
            assert abs(event.data.explained_variance - expected_ev) < 1e-9
            assert event.data.value_mean == float(i + 1)


class TestPPOUpdatePayloadFromDict:
    """Tests for PPOUpdatePayload.from_dict() deserialization.

    Ensures value metrics survive serialization/deserialization round-trip.
    """

    def test_value_metrics_round_trip(self):
        """Value metrics survive dict serialization round-trip."""
        import dataclasses

        original = _make_ppo_update_payload(
            explained_variance=0.75,
            value_mean=3.5,
            value_std=1.2,
            value_min=-2.0,
            value_max=9.0,
        )

        # Serialize to dict
        data = dataclasses.asdict(original)

        # Deserialize back
        restored = PPOUpdatePayload.from_dict(data)

        # Verify value metrics
        assert restored.explained_variance == 0.75
        assert restored.value_mean == 3.5
        assert restored.value_std == 1.2
        assert restored.value_min == -2.0
        assert restored.value_max == 9.0

    def test_explained_variance_none_round_trip(self):
        """None explained_variance survives round-trip."""
        import dataclasses

        original = _make_ppo_update_payload(explained_variance=None)

        data = dataclasses.asdict(original)
        restored = PPOUpdatePayload.from_dict(data)

        assert restored.explained_variance is None
