"""End-to-end telemetry tests for gradient metrics (TELE-300 to TELE-399).

Verifies gradient-related metrics flow from PPOUpdatePayload through
NissaHub to capture backends.

Test Coverage:
    - TELE-300: nan_grad_count (required field)
    - TELE-301: inf_grad_count (wiring gap - marked xfail)
    - TELE-302: head_nan_latch (per-head NaN detection dict)
    - TELE-303: head_inf_latch (per-head Inf detection dict)
    - TELE-310: grad_norm (post-clip gradient norm)
    - TELE-311: grad_norm_history (rolling window in aggregator)
    - TELE-320: head_grad_norms (per-head gradient norms)
    - TELE-321: head_grad_norm_prev (previous values for trend)
    - TELE-330: gradient_cv (coefficient of variation)
    - TELE-331: dead_layers (vanishing gradient count)
    - TELE-332: exploding_layers (exploding gradient count)
"""

import pytest

from esper.leyline import HEAD_NAMES, TelemetryEvent, TelemetryEventType
from esper.leyline.telemetry import PPOUpdatePayload

from .conftest import CaptureBackend, CaptureHubResult


# =============================================================================
# Helper: Create minimal PPOUpdatePayload with defaults
# =============================================================================


def make_ppo_payload(**overrides) -> PPOUpdatePayload:
    """Create PPOUpdatePayload with required fields and optional overrides.

    All required fields are populated with healthy defaults. Override specific
    fields via kwargs to test different scenarios.
    """
    defaults = {
        # Required core fields
        "policy_loss": 0.5,
        "value_loss": 0.3,
        "entropy": 1.5,
        "grad_norm": 0.8,
        "kl_divergence": 0.01,
        "clip_fraction": 0.15,
        "nan_grad_count": 0,
        # Required pre-clip norm
        "pre_clip_grad_norm": 0.9,
        # Required advantage stats
        "advantage_mean": 0.0,
        "advantage_std": 1.0,
        "advantage_skewness": 0.0,
        "advantage_kurtosis": 0.0,
        "advantage_positive_ratio": 0.5,
        # Required ratio stats
        "ratio_mean": 1.0,
        "ratio_min": 0.9,
        "ratio_max": 1.1,
        "ratio_std": 0.05,
        # Required log prob extremes
        "log_prob_min": -5.0,
        "log_prob_max": -0.5,
        # Required boolean
        "entropy_collapsed": False,
        # Required timing
        "update_time_ms": 10.0,
        # Required context
        "inner_epoch": 1,
        "batch": 0,
        "ppo_updates_count": 4,
        # Required value stats
        "value_mean": 0.5,
        "value_std": 0.1,
        "value_min": 0.0,
        "value_max": 1.0,
        # Required gradient quality
        "clip_fraction_positive": 0.1,
        "clip_fraction_negative": 0.05,
        "gradient_cv": 0.3,
        # Required pre-norm stats
        "pre_norm_advantage_mean": 0.0,
        "pre_norm_advantage_std": 1.0,
        # Required return stats
        "return_mean": 0.5,
        "return_std": 0.2,
    }
    defaults.update(overrides)
    return PPOUpdatePayload(**defaults)


def emit_ppo_event(hub, payload: PPOUpdatePayload) -> TelemetryEvent:
    """Emit a PPO_UPDATE_COMPLETED event through the hub and wait for delivery.

    NissaHub uses async workers, so we must flush to ensure events reach
    the capture backend before assertions.
    """
    event = TelemetryEvent(
        event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
        data=payload,
    )
    hub.emit(event)
    # Flush to ensure async workers deliver the event to backends
    hub.flush(timeout=5.0)
    return event


# =============================================================================
# TELE-300: nan_grad_count
# =============================================================================


class TestTELE300NanGradCount:
    """TELE-300: NaN gradient count flows from payload to backend."""

    def test_nan_grad_count_zero_emitted(self, capture_hub: CaptureHubResult):
        """TELE-300: nan_grad_count=0 is emitted during healthy training."""
        hub, backend = capture_hub

        payload = make_ppo_payload(nan_grad_count=0)
        emit_ppo_event(hub, payload)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert len(events) == 1, "Expected exactly one PPO_UPDATE_COMPLETED event"
        assert events[0].data.nan_grad_count == 0

    def test_nan_grad_count_nonzero_emitted(self, capture_hub: CaptureHubResult):
        """TELE-300: Non-zero nan_grad_count indicates numerical instability."""
        hub, backend = capture_hub

        payload = make_ppo_payload(nan_grad_count=15)
        emit_ppo_event(hub, payload)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert len(events) == 1
        assert events[0].data.nan_grad_count == 15

    def test_nan_grad_count_required_field(self):
        """TELE-300: nan_grad_count is a required field (no default)."""
        # This should raise TypeError because nan_grad_count has no default
        with pytest.raises(TypeError, match="nan_grad_count"):
            PPOUpdatePayload(
                policy_loss=0.5,
                value_loss=0.3,
                entropy=1.5,
                grad_norm=0.8,
                kl_divergence=0.01,
                clip_fraction=0.15,
                # nan_grad_count intentionally omitted
            )


# =============================================================================
# TELE-301: inf_grad_count (WIRING GAP)
# =============================================================================


class TestTELE301InfGradCount:
    """TELE-301: Inf gradient count has broken wiring in emitter pipeline."""

    def test_inf_grad_count_default_zero(self, capture_hub: CaptureHubResult):
        """TELE-301: inf_grad_count defaults to 0 when not specified."""
        hub, backend = capture_hub

        # Create payload without specifying inf_grad_count
        payload = make_ppo_payload()
        emit_ppo_event(hub, payload)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert len(events) == 1
        # Default is 0
        assert events[0].data.inf_grad_count == 0

    def test_inf_grad_count_explicit_value_emitted(
        self, capture_hub: CaptureHubResult
    ):
        """TELE-301: Explicit inf_grad_count value is preserved in payload."""
        hub, backend = capture_hub

        payload = make_ppo_payload(inf_grad_count=7)
        emit_ppo_event(hub, payload)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert len(events) == 1
        assert events[0].data.inf_grad_count == 7

    @pytest.mark.xfail(
        reason=(
            "TELE-301 wiring gap: aggregate_layer_gradient_health() does not sum "
            "inf_counts from LayerGradientStats, and emit_ppo_update_event() "
            "hardcodes inf_grad_count=0. See TELE-301 doc for fix details."
        ),
        strict=True,
    )
    def test_inf_grad_count_aggregated_from_layers(self):
        """TELE-301: inf_grad_count should be aggregated from layer stats.

        This test documents the wiring gap. When fixed, this test should pass
        and the xfail marker should be removed.
        """
        from esper.simic.telemetry import LayerGradientStats
        from esper.simic.telemetry.emitters import aggregate_layer_gradient_health

        # Create layer stats with inf counts
        stats = [
            LayerGradientStats(
                layer_name="layer1",
                param_count=100,
                grad_norm=1.0,
                grad_mean=0.1,
                grad_std=0.5,
                grad_min=-1.0,
                grad_max=1.0,
                zero_fraction=0.1,
                small_fraction=0.2,
                large_fraction=0.0,
                nan_count=0,
                inf_count=5,  # Has Inf gradients
            ),
            LayerGradientStats(
                layer_name="layer2",
                param_count=100,
                grad_norm=1.0,
                grad_mean=0.1,
                grad_std=0.5,
                grad_min=-1.0,
                grad_max=1.0,
                zero_fraction=0.1,
                small_fraction=0.2,
                large_fraction=0.0,
                nan_count=0,
                inf_count=10,  # Has Inf gradients
            ),
        ]

        result = aggregate_layer_gradient_health(stats)

        # This assertion will FAIL until the wiring gap is fixed
        # Currently, inf_grad_count is not computed or returned
        assert "inf_grad_count" in result
        assert result["inf_grad_count"] == 15


# =============================================================================
# TELE-302: head_nan_latch (per-head NaN detection)
# =============================================================================


class TestTELE302HeadNanLatch:
    """TELE-302: Per-head NaN detection dictionary flows through telemetry."""

    def test_head_nan_detected_all_false(self, capture_hub: CaptureHubResult):
        """TELE-302: All heads False indicates no NaN detected."""
        hub, backend = capture_hub

        head_nan = {head: False for head in HEAD_NAMES}
        payload = make_ppo_payload(head_nan_detected=head_nan)
        emit_ppo_event(hub, payload)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert len(events) == 1
        assert events[0].data.head_nan_detected is not None
        for head in HEAD_NAMES:
            assert events[0].data.head_nan_detected[head] is False

    def test_head_nan_detected_single_head_true(self, capture_hub: CaptureHubResult):
        """TELE-302: Single head with NaN is correctly identified."""
        hub, backend = capture_hub

        head_nan = {head: False for head in HEAD_NAMES}
        head_nan["blueprint"] = True  # NaN detected in blueprint head
        payload = make_ppo_payload(head_nan_detected=head_nan)
        emit_ppo_event(hub, payload)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert len(events) == 1
        assert events[0].data.head_nan_detected["blueprint"] is True
        assert events[0].data.head_nan_detected["slot"] is False

    def test_head_nan_detected_multiple_heads_true(
        self, capture_hub: CaptureHubResult
    ):
        """TELE-302: Multiple heads with NaN are correctly identified."""
        hub, backend = capture_hub

        head_nan = {head: False for head in HEAD_NAMES}
        head_nan["op"] = True
        head_nan["style"] = True
        head_nan["tempo"] = True
        payload = make_ppo_payload(head_nan_detected=head_nan)
        emit_ppo_event(hub, payload)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert len(events) == 1
        data = events[0].data.head_nan_detected
        assert data["op"] is True
        assert data["style"] is True
        assert data["tempo"] is True
        assert data["slot"] is False

    def test_head_nan_detected_none_when_not_emitted(
        self, capture_hub: CaptureHubResult
    ):
        """TELE-302: head_nan_detected is None when not provided."""
        hub, backend = capture_hub

        payload = make_ppo_payload(head_nan_detected=None)
        emit_ppo_event(hub, payload)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert len(events) == 1
        assert events[0].data.head_nan_detected is None


# =============================================================================
# TELE-303: head_inf_latch (per-head Inf detection)
# =============================================================================


class TestTELE303HeadInfLatch:
    """TELE-303: Per-head Inf detection dictionary flows through telemetry."""

    def test_head_inf_detected_all_false(self, capture_hub: CaptureHubResult):
        """TELE-303: All heads False indicates no Inf detected."""
        hub, backend = capture_hub

        head_inf = {head: False for head in HEAD_NAMES}
        payload = make_ppo_payload(head_inf_detected=head_inf)
        emit_ppo_event(hub, payload)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert len(events) == 1
        assert events[0].data.head_inf_detected is not None
        for head in HEAD_NAMES:
            assert events[0].data.head_inf_detected[head] is False

    def test_head_inf_detected_single_head_true(self, capture_hub: CaptureHubResult):
        """TELE-303: Single head with Inf is correctly identified."""
        hub, backend = capture_hub

        head_inf = {head: False for head in HEAD_NAMES}
        head_inf["alpha_target"] = True  # Inf detected in alpha_target head
        payload = make_ppo_payload(head_inf_detected=head_inf)
        emit_ppo_event(hub, payload)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert len(events) == 1
        assert events[0].data.head_inf_detected["alpha_target"] is True
        assert events[0].data.head_inf_detected["slot"] is False

    def test_head_nan_and_inf_both_detected_same_head(
        self, capture_hub: CaptureHubResult
    ):
        """TELE-302/303: Same head can have both NaN and Inf detected."""
        hub, backend = capture_hub

        head_nan = {head: False for head in HEAD_NAMES}
        head_inf = {head: False for head in HEAD_NAMES}
        # Both NaN and Inf in the same head
        head_nan["slot"] = True
        head_inf["slot"] = True
        payload = make_ppo_payload(
            head_nan_detected=head_nan, head_inf_detected=head_inf
        )
        emit_ppo_event(hub, payload)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert len(events) == 1
        assert events[0].data.head_nan_detected["slot"] is True
        assert events[0].data.head_inf_detected["slot"] is True


# =============================================================================
# TELE-310: grad_norm (post-clip gradient norm)
# =============================================================================


class TestTELE310GradNorm:
    """TELE-310: Post-clip gradient norm flows from payload to backend."""

    def test_grad_norm_healthy_value(self, capture_hub: CaptureHubResult):
        """TELE-310: Healthy grad_norm (< 5.0) is emitted correctly."""
        hub, backend = capture_hub

        payload = make_ppo_payload(grad_norm=0.85)
        emit_ppo_event(hub, payload)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert len(events) == 1
        assert events[0].data.grad_norm == pytest.approx(0.85)

    def test_grad_norm_warning_value(self, capture_hub: CaptureHubResult):
        """TELE-310: Warning-level grad_norm (5.0 < x <= 10.0) is emitted."""
        hub, backend = capture_hub

        payload = make_ppo_payload(grad_norm=7.5)
        emit_ppo_event(hub, payload)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert len(events) == 1
        assert events[0].data.grad_norm == pytest.approx(7.5)

    def test_grad_norm_critical_value(self, capture_hub: CaptureHubResult):
        """TELE-310: Critical grad_norm (> 10.0) indicates gradient explosion."""
        hub, backend = capture_hub

        payload = make_ppo_payload(grad_norm=15.3)
        emit_ppo_event(hub, payload)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert len(events) == 1
        assert events[0].data.grad_norm == pytest.approx(15.3)

    def test_grad_norm_required_field(self):
        """TELE-310: grad_norm is a required field in PPOUpdatePayload."""
        with pytest.raises(TypeError, match="grad_norm"):
            PPOUpdatePayload(
                policy_loss=0.5,
                value_loss=0.3,
                entropy=1.5,
                # grad_norm intentionally omitted
                kl_divergence=0.01,
                clip_fraction=0.15,
                nan_grad_count=0,
            )


# =============================================================================
# TELE-311: grad_norm_history (Note: aggregator-managed, not in payload)
# =============================================================================


class TestTELE311GradNormHistory:
    """TELE-311: Gradient norm history is managed by the aggregator.

    The grad_norm_history deque is maintained by SanctumAggregator, not
    directly in PPOUpdatePayload. These tests verify the payload carries
    the grad_norm values that feed into the history.
    """

    def test_multiple_grad_norms_emitted_sequentially(
        self, capture_hub: CaptureHubResult
    ):
        """TELE-311: Multiple PPO updates emit sequential grad_norm values."""
        hub, backend = capture_hub

        grad_norms = [0.5, 0.8, 1.2, 0.9, 0.7]
        for gn in grad_norms:
            payload = make_ppo_payload(grad_norm=gn)
            emit_ppo_event(hub, payload)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert len(events) == 5

        emitted_norms = [e.data.grad_norm for e in events]
        assert emitted_norms == pytest.approx(grad_norms)

    def test_pre_clip_grad_norm_emitted(self, capture_hub: CaptureHubResult):
        """TELE-311: Pre-clip gradient norm is emitted alongside post-clip."""
        hub, backend = capture_hub

        payload = make_ppo_payload(grad_norm=1.0, pre_clip_grad_norm=4.5)
        emit_ppo_event(hub, payload)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert len(events) == 1
        # Post-clip norm (after clipping to max_grad_norm)
        assert events[0].data.grad_norm == pytest.approx(1.0)
        # Pre-clip norm (raw gradient magnitude before clipping)
        assert events[0].data.pre_clip_grad_norm == pytest.approx(4.5)


# =============================================================================
# TELE-320: head_grad_norms (per-head gradient norms)
# =============================================================================


class TestTELE320HeadGradNorms:
    """TELE-320: Per-head gradient norms flow from payload to backend."""

    def test_head_grad_norms_all_present(self, capture_hub: CaptureHubResult):
        """TELE-320: All 8 head gradient norm fields are emitted."""
        hub, backend = capture_hub

        payload = make_ppo_payload(
            head_slot_grad_norm=0.5,
            head_blueprint_grad_norm=0.6,
            head_style_grad_norm=0.7,
            head_tempo_grad_norm=0.8,
            head_alpha_target_grad_norm=0.9,
            head_alpha_speed_grad_norm=1.0,
            head_alpha_curve_grad_norm=1.1,
            head_op_grad_norm=1.2,
        )
        emit_ppo_event(hub, payload)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert len(events) == 1
        data = events[0].data
        assert data.head_slot_grad_norm == pytest.approx(0.5)
        assert data.head_blueprint_grad_norm == pytest.approx(0.6)
        assert data.head_style_grad_norm == pytest.approx(0.7)
        assert data.head_tempo_grad_norm == pytest.approx(0.8)
        assert data.head_alpha_target_grad_norm == pytest.approx(0.9)
        assert data.head_alpha_speed_grad_norm == pytest.approx(1.0)
        assert data.head_alpha_curve_grad_norm == pytest.approx(1.1)
        assert data.head_op_grad_norm == pytest.approx(1.2)

    def test_head_grad_norms_none_by_default(self, capture_hub: CaptureHubResult):
        """TELE-320: Head gradient norms default to None when not provided."""
        hub, backend = capture_hub

        payload = make_ppo_payload()  # No head grad norms specified
        emit_ppo_event(hub, payload)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert len(events) == 1
        data = events[0].data
        # Optional fields default to None
        assert data.head_slot_grad_norm is None
        assert data.head_blueprint_grad_norm is None

    def test_head_grad_norms_vanishing_detection(self, capture_hub: CaptureHubResult):
        """TELE-320: Very low head grad norm indicates vanishing gradients."""
        hub, backend = capture_hub

        payload = make_ppo_payload(
            head_slot_grad_norm=0.001,  # Vanishing (< 0.01 threshold)
            head_blueprint_grad_norm=0.5,  # Healthy
        )
        emit_ppo_event(hub, payload)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert len(events) == 1
        data = events[0].data
        # Slot head has vanishing gradient
        assert data.head_slot_grad_norm < 0.01
        # Blueprint head is healthy
        assert 0.1 <= data.head_blueprint_grad_norm <= 2.0

    def test_head_grad_norms_exploding_detection(self, capture_hub: CaptureHubResult):
        """TELE-320: High head grad norm indicates exploding gradients."""
        hub, backend = capture_hub

        payload = make_ppo_payload(
            head_slot_grad_norm=0.5,  # Healthy
            head_style_grad_norm=8.0,  # Exploding (> 5.0 threshold)
        )
        emit_ppo_event(hub, payload)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert len(events) == 1
        data = events[0].data
        # Style head has exploding gradient
        assert data.head_style_grad_norm > 5.0


# =============================================================================
# TELE-321: head_grad_norm_prev (previous values - aggregator-managed)
# =============================================================================


class TestTELE321HeadGradNormPrev:
    """TELE-321: Previous head gradient norms are aggregator-managed.

    The _prev fields are computed in SanctumAggregator by saving current
    values before updating with new ones. These tests verify the current
    values are correctly emitted (which feed into _prev tracking).
    """

    def test_sequential_head_grad_norms_for_trend(
        self, capture_hub: CaptureHubResult
    ):
        """TELE-321: Sequential emissions enable trend detection."""
        hub, backend = capture_hub

        # First emission - baseline
        payload1 = make_ppo_payload(head_slot_grad_norm=0.5)
        emit_ppo_event(hub, payload1)

        # Second emission - increasing (should trigger "increasing" trend)
        payload2 = make_ppo_payload(head_slot_grad_norm=0.7)
        emit_ppo_event(hub, payload2)

        # Third emission - decreasing (should trigger "decreasing" trend)
        payload3 = make_ppo_payload(head_slot_grad_norm=0.3)
        emit_ppo_event(hub, payload3)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert len(events) == 3

        slot_norms = [e.data.head_slot_grad_norm for e in events]
        assert slot_norms == pytest.approx([0.5, 0.7, 0.3])

        # Verify trend direction
        assert slot_norms[1] > slot_norms[0]  # Increasing
        assert slot_norms[2] < slot_norms[1]  # Decreasing


# =============================================================================
# TELE-330: gradient_cv (coefficient of variation)
# =============================================================================


class TestTELE330GradientCV:
    """TELE-330: Gradient coefficient of variation flows from payload."""

    def test_gradient_cv_healthy_value(self, capture_hub: CaptureHubResult):
        """TELE-330: Low CV (< 0.5) indicates uniform gradient flow."""
        hub, backend = capture_hub

        payload = make_ppo_payload(gradient_cv=0.35)
        emit_ppo_event(hub, payload)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert len(events) == 1
        assert events[0].data.gradient_cv == pytest.approx(0.35)
        # Healthy threshold: < 0.5
        assert events[0].data.gradient_cv < 0.5

    def test_gradient_cv_warning_value(self, capture_hub: CaptureHubResult):
        """TELE-330: Moderate CV (0.5-2.0) indicates uneven gradient flow."""
        hub, backend = capture_hub

        payload = make_ppo_payload(gradient_cv=1.2)
        emit_ppo_event(hub, payload)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert len(events) == 1
        cv = events[0].data.gradient_cv
        assert 0.5 <= cv < 2.0

    def test_gradient_cv_critical_value(self, capture_hub: CaptureHubResult):
        """TELE-330: High CV (>= 2.0) indicates training instability."""
        hub, backend = capture_hub

        payload = make_ppo_payload(gradient_cv=3.5)
        emit_ppo_event(hub, payload)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert len(events) == 1
        assert events[0].data.gradient_cv >= 2.0

    def test_gradient_cv_default_zero(self, capture_hub: CaptureHubResult):
        """TELE-330: gradient_cv defaults to 0.0."""
        # Verify the dataclass default
        payload = PPOUpdatePayload(
            policy_loss=0.5,
            value_loss=0.3,
            entropy=1.5,
            grad_norm=0.8,
            kl_divergence=0.01,
            clip_fraction=0.15,
            nan_grad_count=0,
        )
        assert payload.gradient_cv == 0.0


# =============================================================================
# TELE-331: dead_layers (vanishing gradient count)
# =============================================================================


class TestTELE331DeadLayers:
    """TELE-331: Dead layers count flows from payload to backend."""

    def test_dead_layers_zero_healthy(self, capture_hub: CaptureHubResult):
        """TELE-331: Zero dead layers indicates healthy gradient flow."""
        hub, backend = capture_hub

        payload = make_ppo_payload(dead_layers=0)
        emit_ppo_event(hub, payload)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert len(events) == 1
        assert events[0].data.dead_layers == 0

    def test_dead_layers_warning_count(self, capture_hub: CaptureHubResult):
        """TELE-331: Small dead layer count indicates minor gradient issues."""
        hub, backend = capture_hub

        payload = make_ppo_payload(dead_layers=2)
        emit_ppo_event(hub, payload)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert len(events) == 1
        assert events[0].data.dead_layers == 2
        # Warning threshold: 1-2 dead layers
        assert 1 <= events[0].data.dead_layers <= 2

    def test_dead_layers_critical_count(self, capture_hub: CaptureHubResult):
        """TELE-331: High dead layer count indicates severe gradient blockage."""
        hub, backend = capture_hub

        payload = make_ppo_payload(dead_layers=5)
        emit_ppo_event(hub, payload)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert len(events) == 1
        # Critical threshold: >= 3 dead layers
        assert events[0].data.dead_layers >= 3

    def test_dead_layers_default_zero(self, capture_hub: CaptureHubResult):
        """TELE-331: dead_layers defaults to 0."""
        hub, backend = capture_hub

        payload = make_ppo_payload()  # Not specifying dead_layers
        emit_ppo_event(hub, payload)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert len(events) == 1
        assert events[0].data.dead_layers == 0


# =============================================================================
# TELE-332: exploding_layers (exploding gradient count)
# =============================================================================


class TestTELE332ExplodingLayers:
    """TELE-332: Exploding layers count flows from payload to backend."""

    def test_exploding_layers_zero_healthy(self, capture_hub: CaptureHubResult):
        """TELE-332: Zero exploding layers indicates stable gradients."""
        hub, backend = capture_hub

        payload = make_ppo_payload(exploding_layers=0)
        emit_ppo_event(hub, payload)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert len(events) == 1
        assert events[0].data.exploding_layers == 0

    def test_exploding_layers_warning_count(self, capture_hub: CaptureHubResult):
        """TELE-332: Small exploding layer count indicates minor instability."""
        hub, backend = capture_hub

        payload = make_ppo_payload(exploding_layers=1)
        emit_ppo_event(hub, payload)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert len(events) == 1
        assert events[0].data.exploding_layers == 1

    def test_exploding_layers_critical_count(self, capture_hub: CaptureHubResult):
        """TELE-332: High exploding layer count indicates severe instability."""
        hub, backend = capture_hub

        payload = make_ppo_payload(exploding_layers=4)
        emit_ppo_event(hub, payload)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert len(events) == 1
        # Critical threshold: >= 3 exploding layers
        assert events[0].data.exploding_layers >= 3

    def test_exploding_layers_default_zero(self, capture_hub: CaptureHubResult):
        """TELE-332: exploding_layers defaults to 0."""
        hub, backend = capture_hub

        payload = make_ppo_payload()  # Not specifying exploding_layers
        emit_ppo_event(hub, payload)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert len(events) == 1
        assert events[0].data.exploding_layers == 0

    def test_dead_and_exploding_layers_combined(self, capture_hub: CaptureHubResult):
        """TELE-331/332: Both dead and exploding layers can be non-zero."""
        hub, backend = capture_hub

        payload = make_ppo_payload(dead_layers=2, exploding_layers=1)
        emit_ppo_event(hub, payload)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert len(events) == 1
        assert events[0].data.dead_layers == 2
        assert events[0].data.exploding_layers == 1


# =============================================================================
# Integration: Full Gradient Health Scenario
# =============================================================================


class TestGradientMetricsIntegration:
    """Integration tests verifying multiple gradient metrics together."""

    def test_healthy_training_gradient_profile(self, capture_hub: CaptureHubResult):
        """All gradient metrics in healthy training scenario."""
        hub, backend = capture_hub

        head_nan = {head: False for head in HEAD_NAMES}
        head_inf = {head: False for head in HEAD_NAMES}

        payload = make_ppo_payload(
            # Global gradient health
            nan_grad_count=0,
            inf_grad_count=0,
            grad_norm=0.85,
            pre_clip_grad_norm=1.2,
            gradient_cv=0.25,
            dead_layers=0,
            exploding_layers=0,
            # Per-head detection
            head_nan_detected=head_nan,
            head_inf_detected=head_inf,
            # Per-head gradient norms (all healthy)
            head_slot_grad_norm=0.5,
            head_blueprint_grad_norm=0.6,
            head_style_grad_norm=0.5,
            head_tempo_grad_norm=0.4,
            head_alpha_target_grad_norm=0.7,
            head_alpha_speed_grad_norm=0.6,
            head_alpha_curve_grad_norm=0.5,
            head_op_grad_norm=0.8,
        )
        emit_ppo_event(hub, payload)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert len(events) == 1
        data = events[0].data

        # Verify healthy profile
        assert data.nan_grad_count == 0
        assert data.inf_grad_count == 0
        assert data.grad_norm < 5.0  # Healthy threshold
        assert data.gradient_cv < 0.5  # Low variance
        assert data.dead_layers == 0
        assert data.exploding_layers == 0
        assert all(not v for v in data.head_nan_detected.values())
        assert all(not v for v in data.head_inf_detected.values())

    def test_gradient_explosion_scenario(self, capture_hub: CaptureHubResult):
        """Gradient metrics during gradient explosion scenario."""
        hub, backend = capture_hub

        head_nan = {head: False for head in HEAD_NAMES}
        head_inf = {head: False for head in HEAD_NAMES}
        head_inf["blueprint"] = True  # Inf in blueprint head

        payload = make_ppo_payload(
            # Global gradient health - signs of trouble
            nan_grad_count=0,
            inf_grad_count=5,
            grad_norm=12.5,  # Critical (> 10.0)
            pre_clip_grad_norm=100.0,  # Severe explosion before clipping
            gradient_cv=2.5,  # High variance
            dead_layers=0,
            exploding_layers=3,  # Critical
            # Per-head detection
            head_nan_detected=head_nan,
            head_inf_detected=head_inf,
            # Per-head gradient norms (blueprint exploding)
            head_slot_grad_norm=0.5,
            head_blueprint_grad_norm=25.0,  # Exploding
        )
        emit_ppo_event(hub, payload)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert len(events) == 1
        data = events[0].data

        # Verify explosion indicators
        assert data.grad_norm > 10.0  # Critical
        assert data.pre_clip_grad_norm > data.grad_norm  # Clipping active
        assert data.gradient_cv >= 2.0  # High variance
        assert data.exploding_layers >= 3  # Critical
        assert data.head_inf_detected["blueprint"] is True
        assert data.head_blueprint_grad_norm > 5.0  # Exploding

    def test_vanishing_gradient_scenario(self, capture_hub: CaptureHubResult):
        """Gradient metrics during vanishing gradient scenario."""
        hub, backend = capture_hub

        head_nan = {head: False for head in HEAD_NAMES}
        head_inf = {head: False for head in HEAD_NAMES}

        payload = make_ppo_payload(
            # Global gradient health - signs of vanishing
            nan_grad_count=0,
            inf_grad_count=0,
            grad_norm=0.01,  # Very low
            pre_clip_grad_norm=0.01,  # No clipping needed
            gradient_cv=0.8,  # Moderate variance
            dead_layers=4,  # Critical - many dead layers
            exploding_layers=0,
            # Per-head detection
            head_nan_detected=head_nan,
            head_inf_detected=head_inf,
            # Per-head gradient norms (many vanishing)
            head_slot_grad_norm=0.001,
            head_blueprint_grad_norm=0.002,
            head_style_grad_norm=0.001,
            head_tempo_grad_norm=0.5,  # This one is healthy
        )
        emit_ppo_event(hub, payload)

        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert len(events) == 1
        data = events[0].data

        # Verify vanishing indicators
        assert data.grad_norm < 0.1  # Very low
        assert data.dead_layers >= 3  # Critical
        assert data.head_slot_grad_norm < 0.01  # Vanishing
        assert data.head_blueprint_grad_norm < 0.01  # Vanishing
        assert data.head_tempo_grad_norm >= 0.1  # One healthy head
