"""Tests for telemetry payload dataclasses."""


def test_analytics_snapshot_payload_accepts_reward_components_dataclass():
    """AnalyticsSnapshotPayload should accept RewardComponentsTelemetry."""
    from esper.leyline.telemetry import AnalyticsSnapshotPayload
    from esper.simic.rewards.reward_telemetry import RewardComponentsTelemetry

    rc = RewardComponentsTelemetry(
        bounded_attribution=0.5,
        compute_rent=-0.1,
        seed_stage=2,
        action_shaping=0.05,
        total_reward=0.45,
    )

    payload = AnalyticsSnapshotPayload(
        kind="last_action",
        reward_components=rc,
    )

    assert payload.reward_components is rc
    assert payload.reward_components.seed_stage == 2
    assert payload.reward_components.bounded_attribution == 0.5


def test_telemetry_event_serializes_nested_reward_components():
    """TelemetryEvent should serialize reward_components dataclass to JSON.

    The serialized JSON should include the shaped_reward_ratio property,
    which is only available via to_dict() (not asdict()).
    """
    import json
    from datetime import datetime, timezone

    from esper.karn.serialization import serialize_event
    from esper.leyline.telemetry import (
        AnalyticsSnapshotPayload,
        TelemetryEvent,
        TelemetryEventType,
    )
    from esper.simic.rewards.reward_telemetry import RewardComponentsTelemetry

    rc = RewardComponentsTelemetry(
        bounded_attribution=0.5,
        seed_stage=2,
        total_reward=0.45,
    )

    payload = AnalyticsSnapshotPayload(
        kind="last_action",
        env_id=0,
        reward_components=rc,
    )

    event = TelemetryEvent(
        event_type=TelemetryEventType.ANALYTICS_SNAPSHOT,
        timestamp=datetime.now(timezone.utc),
        data=payload,
        epoch=10,
    )

    json_str = serialize_event(event)
    parsed = json.loads(json_str)

    # Verify nested structure in JSON
    assert "reward_components" in parsed["data"]
    assert parsed["data"]["reward_components"]["seed_stage"] == 2
    assert parsed["data"]["reward_components"]["bounded_attribution"] == 0.5

    # Critical: shaped_reward_ratio should be included (from to_dict())
    assert "shaped_reward_ratio" in parsed["data"]["reward_components"]


# =============================================================================
# PPOUpdatePayload Extensions (per UX specialist enhancements plan)
# =============================================================================


def test_ppo_update_payload_has_gradient_quality_metrics():
    """PPOUpdatePayload should include directional clip and gradient CV fields."""
    from esper.leyline.telemetry import PPOUpdatePayload

    # Create with gradient quality fields
    payload = PPOUpdatePayload(
        policy_loss=0.1,
        value_loss=0.2,
        entropy=1.5,
        grad_norm=0.5,
        kl_divergence=0.01,
        clip_fraction=0.15,
        nan_grad_count=0,
        # New gradient quality fields
        clip_fraction_positive=0.10,
        clip_fraction_negative=0.05,
        gradient_cv=0.35,
    )

    assert payload.clip_fraction_positive == 0.10
    assert payload.clip_fraction_negative == 0.05
    assert payload.gradient_cv == 0.35


def test_ppo_update_payload_gradient_quality_defaults():
    """New gradient quality fields should default to 0.0."""
    from esper.leyline.telemetry import PPOUpdatePayload

    payload = PPOUpdatePayload(
        policy_loss=0.1,
        value_loss=0.2,
        entropy=1.5,
        grad_norm=0.5,
        kl_divergence=0.01,
        clip_fraction=0.15,
        nan_grad_count=0,
    )

    assert payload.clip_fraction_positive == 0.0
    assert payload.clip_fraction_negative == 0.0
    assert payload.gradient_cv == 0.0


def test_ppo_update_payload_has_infrastructure_metrics():
    """PPOUpdatePayload should include CUDA memory fields."""
    from esper.leyline.telemetry import PPOUpdatePayload

    payload = PPOUpdatePayload(
        policy_loss=0.1,
        value_loss=0.2,
        entropy=1.5,
        grad_norm=0.5,
        kl_divergence=0.01,
        clip_fraction=0.15,
        nan_grad_count=0,
        # New infrastructure fields
        cuda_memory_allocated_gb=4.2,
        cuda_memory_reserved_gb=8.0,
        cuda_memory_peak_gb=6.5,
        cuda_memory_fragmentation=0.25,
    )

    assert payload.cuda_memory_allocated_gb == 4.2
    assert payload.cuda_memory_reserved_gb == 8.0
    assert payload.cuda_memory_peak_gb == 6.5
    assert payload.cuda_memory_fragmentation == 0.25


def test_ppo_update_payload_from_dict_parses_new_fields():
    """PPOUpdatePayload.from_dict should parse the new fields."""
    from esper.leyline.telemetry import PPOUpdatePayload

    data = {
        # Core required fields
        "policy_loss": 0.1,
        "value_loss": 0.2,
        "entropy": 1.5,
        "grad_norm": 0.5,
        "kl_divergence": 0.01,
        "clip_fraction": 0.15,
        "nan_grad_count": 0,
        "pre_clip_grad_norm": 4.5,
        # Advantage stats (always emitted)
        "advantage_mean": 0.5,
        "advantage_std": 1.0,
        "advantage_skewness": 0.1,
        "advantage_kurtosis": 3.0,
        "advantage_positive_ratio": 0.55,
        # Ratio stats (always emitted)
        "ratio_mean": 1.0,
        "ratio_min": 0.8,
        "ratio_max": 1.2,
        "ratio_std": 0.1,
        # Log prob extremes (always emitted)
        "log_prob_min": -5.0,
        "log_prob_max": -0.5,
        # Always emitted
        "entropy_collapsed": False,
        "update_time_ms": 150.0,
        "inner_epoch": 0,
        "batch": 1,
        "ppo_updates_count": 3,
        # Value function statistics (always emitted)
        "value_mean": 5.0,
        "value_std": 2.0,
        "value_min": 0.0,
        "value_max": 10.0,
        # Gradient quality metrics (always emitted)
        "clip_fraction_positive": 0.12,
        "clip_fraction_negative": 0.08,
        "gradient_cv": 0.45,
        # Infrastructure metrics (optional with defaults)
        "cuda_memory_allocated_gb": 5.0,
        "cuda_memory_reserved_gb": 10.0,
        "cuda_memory_peak_gb": 7.5,
        "cuda_memory_fragmentation": 0.30,
        # Pre-normalization advantage stats (always emitted)
        "pre_norm_advantage_mean": 0.8,
        "pre_norm_advantage_std": 2.5,
        # Return statistics (always emitted)
        "return_mean": 10.0,
        "return_std": 3.0,
    }

    payload = PPOUpdatePayload.from_dict(data)

    assert payload.pre_clip_grad_norm == 4.5
    assert payload.ppo_updates_count == 3
    assert payload.clip_fraction_positive == 0.12
    assert payload.clip_fraction_negative == 0.08
    assert payload.gradient_cv == 0.45
    assert payload.cuda_memory_allocated_gb == 5.0
    assert payload.cuda_memory_reserved_gb == 10.0
    assert payload.cuda_memory_peak_gb == 7.5
    assert payload.cuda_memory_fragmentation == 0.30


# =============================================================================
# HeadTelemetry Dataclass Tests
# =============================================================================


def test_head_telemetry_dataclass():
    """HeadTelemetry should hold confidence and entropy for all 8 heads."""
    from esper.leyline.telemetry import HeadTelemetry

    head_telem = HeadTelemetry(
        op_confidence=0.85,
        slot_confidence=0.72,
        blueprint_confidence=0.91,
        style_confidence=0.65,
        tempo_confidence=0.88,
        alpha_target_confidence=0.77,
        alpha_speed_confidence=0.69,
        curve_confidence=0.82,
        op_entropy=0.3,
        slot_entropy=0.8,
        blueprint_entropy=0.5,
        style_entropy=0.6,
        tempo_entropy=0.4,
        alpha_target_entropy=0.55,
        alpha_speed_entropy=0.45,
        curve_entropy=0.35,
    )

    # Verify all confidence fields
    assert head_telem.op_confidence == 0.85
    assert head_telem.slot_confidence == 0.72
    assert head_telem.blueprint_confidence == 0.91
    assert head_telem.style_confidence == 0.65
    assert head_telem.tempo_confidence == 0.88
    assert head_telem.alpha_target_confidence == 0.77
    assert head_telem.alpha_speed_confidence == 0.69
    assert head_telem.curve_confidence == 0.82

    # Verify all entropy fields
    assert head_telem.op_entropy == 0.3
    assert head_telem.slot_entropy == 0.8
    assert head_telem.blueprint_entropy == 0.5
    assert head_telem.style_entropy == 0.6
    assert head_telem.tempo_entropy == 0.4
    assert head_telem.alpha_target_entropy == 0.55
    assert head_telem.alpha_speed_entropy == 0.45
    assert head_telem.curve_entropy == 0.35


def test_analytics_snapshot_payload_accepts_head_telemetry():
    """AnalyticsSnapshotPayload should accept HeadTelemetry."""
    from esper.leyline.telemetry import AnalyticsSnapshotPayload, HeadTelemetry

    head_telem = HeadTelemetry(
        op_confidence=0.85,
        slot_confidence=0.72,
        blueprint_confidence=0.91,
        style_confidence=0.65,
        tempo_confidence=0.88,
        alpha_target_confidence=0.77,
        alpha_speed_confidence=0.69,
        curve_confidence=0.82,
        op_entropy=0.3,
        slot_entropy=0.8,
        blueprint_entropy=0.5,
        style_entropy=0.6,
        tempo_entropy=0.4,
        alpha_target_entropy=0.55,
        alpha_speed_entropy=0.45,
        curve_entropy=0.35,
    )

    payload = AnalyticsSnapshotPayload(
        kind="last_action",
        head_telemetry=head_telem,
    )

    assert payload.head_telemetry is head_telem
    assert payload.head_telemetry.op_confidence == 0.85
    assert payload.head_telemetry.op_entropy == 0.3


def test_telemetry_event_serializes_nested_head_telemetry():
    """TelemetryEvent should serialize head_telemetry dataclass to JSON.

    Verifies the full round-trip: HeadTelemetry -> JSON -> HeadTelemetry
    """
    import json
    from datetime import datetime, timezone

    from esper.karn.serialization import serialize_event
    from esper.leyline.telemetry import (
        AnalyticsSnapshotPayload,
        HeadTelemetry,
        TelemetryEvent,
        TelemetryEventType,
    )

    head_telem = HeadTelemetry(
        op_confidence=0.85,
        slot_confidence=0.72,
        blueprint_confidence=0.91,
        style_confidence=0.65,
        tempo_confidence=0.88,
        alpha_target_confidence=0.77,
        alpha_speed_confidence=0.69,
        curve_confidence=0.82,
        op_entropy=0.3,
        slot_entropy=0.8,
        blueprint_entropy=0.5,
        style_entropy=0.6,
        tempo_entropy=0.4,
        alpha_target_entropy=0.55,
        alpha_speed_entropy=0.45,
        curve_entropy=0.35,
    )

    payload = AnalyticsSnapshotPayload(
        kind="last_action",
        env_id=0,
        head_telemetry=head_telem,
    )

    event = TelemetryEvent(
        event_type=TelemetryEventType.ANALYTICS_SNAPSHOT,
        timestamp=datetime.now(timezone.utc),
        data=payload,
        epoch=10,
    )

    # Serialize to JSON
    json_str = serialize_event(event)
    parsed = json.loads(json_str)

    # Verify nested structure in JSON
    assert "head_telemetry" in parsed["data"]
    assert parsed["data"]["head_telemetry"]["op_confidence"] == 0.85
    assert parsed["data"]["head_telemetry"]["slot_entropy"] == 0.8

    # Round-trip: deserialize back to HeadTelemetry via AnalyticsSnapshotPayload.from_dict
    restored_payload = AnalyticsSnapshotPayload.from_dict(parsed["data"])
    assert restored_payload.head_telemetry is not None
    assert restored_payload.head_telemetry.op_confidence == 0.85
    assert restored_payload.head_telemetry.slot_confidence == 0.72
    assert restored_payload.head_telemetry.blueprint_confidence == 0.91
    assert restored_payload.head_telemetry.style_confidence == 0.65
    assert restored_payload.head_telemetry.tempo_confidence == 0.88
    assert restored_payload.head_telemetry.alpha_target_confidence == 0.77
    assert restored_payload.head_telemetry.alpha_speed_confidence == 0.69
    assert restored_payload.head_telemetry.curve_confidence == 0.82
    assert restored_payload.head_telemetry.op_entropy == 0.3
    assert restored_payload.head_telemetry.slot_entropy == 0.8
    assert restored_payload.head_telemetry.blueprint_entropy == 0.5
    assert restored_payload.head_telemetry.style_entropy == 0.6
    assert restored_payload.head_telemetry.tempo_entropy == 0.4
    assert restored_payload.head_telemetry.alpha_target_entropy == 0.55
    assert restored_payload.head_telemetry.alpha_speed_entropy == 0.45
    assert restored_payload.head_telemetry.curve_entropy == 0.35


# =============================================================================
# Q-Value Telemetry Tests (Policy V2)
# =============================================================================


def test_ppo_update_payload_with_q_values():
    """PPOUpdatePayload accepts and serializes Q-values per op."""
    from esper.leyline.telemetry import PPOUpdatePayload

    payload = PPOUpdatePayload(
        policy_loss=0.5,
        value_loss=0.3,
        entropy=1.2,
        grad_norm=2.0,
        kl_divergence=0.01,
        clip_fraction=0.15,
        nan_grad_count=0,
        # Q-values per operation
        q_germinate=5.2,
        q_advance=3.1,
        q_fossilize=2.8,
        q_prune=-1.5,
        q_wait=0.5,
        q_set_alpha=4.0,
        q_variance=2.3,
        q_spread=6.7,
    )

    assert payload.q_germinate == 5.2
    assert payload.q_advance == 3.1
    assert payload.q_fossilize == 2.8
    assert payload.q_prune == -1.5
    assert payload.q_wait == 0.5
    assert payload.q_set_alpha == 4.0
    assert payload.q_variance == 2.3
    assert payload.q_spread == 6.7


# =============================================================================
# Per-Head NaN/Inf Flags (for indicator lights with latch behavior)
# =============================================================================


def test_ppo_update_payload_has_per_head_nan_inf_flags():
    """PPOUpdatePayload should have per-head NaN/Inf flag dicts."""
    from esper.leyline.telemetry import PPOUpdatePayload

    payload = PPOUpdatePayload(
        policy_loss=0.1,
        value_loss=0.2,
        entropy=1.0,
        grad_norm=0.5,
        kl_divergence=0.01,
        clip_fraction=0.1,
        nan_grad_count=0,
        pre_clip_grad_norm=0.5,
        head_nan_detected={"op": True, "slot": False},
        head_inf_detected={"op": False, "slot": True},
    )

    assert payload.head_nan_detected == {"op": True, "slot": False}
    assert payload.head_inf_detected == {"op": False, "slot": True}


def test_ppo_update_payload_per_head_nan_inf_defaults_to_none():
    """Per-head NaN/Inf flags should default to None."""
    from esper.leyline.telemetry import PPOUpdatePayload

    payload = PPOUpdatePayload(
        policy_loss=0.1,
        value_loss=0.2,
        entropy=1.0,
        grad_norm=0.5,
        kl_divergence=0.01,
        clip_fraction=0.1,
        nan_grad_count=0,
    )

    assert payload.head_nan_detected is None
    assert payload.head_inf_detected is None


def test_ppo_update_payload_from_dict_with_per_head_nan_inf_flags():
    """PPOUpdatePayload.from_dict parses per-head NaN/Inf flags."""
    from esper.leyline.telemetry import PPOUpdatePayload

    data = {
        # Core required fields
        "policy_loss": 0.1,
        "value_loss": 0.2,
        "entropy": 1.0,
        "grad_norm": 0.5,
        "kl_divergence": 0.01,
        "clip_fraction": 0.1,
        "nan_grad_count": 0,
        "pre_clip_grad_norm": 0.5,
        # Advantage stats (always emitted)
        "advantage_mean": 0.0,
        "advantage_std": 1.0,
        "advantage_skewness": 0.0,
        "advantage_kurtosis": 0.0,
        "advantage_positive_ratio": 0.5,
        # Ratio stats (always emitted)
        "ratio_mean": 1.0,
        "ratio_min": 1.0,
        "ratio_max": 1.0,
        "ratio_std": 0.0,
        # Log prob extremes (always emitted)
        "log_prob_min": -1.0,
        "log_prob_max": 0.0,
        # Always emitted
        "entropy_collapsed": False,
        "update_time_ms": 100.0,
        "inner_epoch": 0,
        "batch": 0,
        "ppo_updates_count": 1,
        # Value function statistics (always emitted)
        "value_mean": 0.0,
        "value_std": 1.0,
        "value_min": -1.0,
        "value_max": 1.0,
        # Gradient quality metrics (always emitted)
        "clip_fraction_positive": 0.0,
        "clip_fraction_negative": 0.0,
        "gradient_cv": 0.0,
        # Pre-normalization advantage stats (always emitted)
        "pre_norm_advantage_mean": 0.0,
        "pre_norm_advantage_std": 1.0,
        # Return statistics (always emitted)
        "return_mean": 0.0,
        "return_std": 1.0,
        # Per-head NaN/Inf detection - what this test focuses on
        "head_nan_detected": {"op": True, "slot": False, "blueprint": False},
        "head_inf_detected": {"op": False, "slot": True, "blueprint": False},
    }

    payload = PPOUpdatePayload.from_dict(data)

    assert payload.head_nan_detected == {"op": True, "slot": False, "blueprint": False}
    assert payload.head_inf_detected == {"op": False, "slot": True, "blueprint": False}


def test_ppo_update_payload_from_dict_with_q_values():
    """PPOUpdatePayload.from_dict parses Q-values."""
    from esper.leyline.telemetry import PPOUpdatePayload

    data = {
        # Core required fields
        "policy_loss": 0.5,
        "value_loss": 0.3,
        "entropy": 1.2,
        "grad_norm": 2.0,
        "kl_divergence": 0.01,
        "clip_fraction": 0.15,
        "nan_grad_count": 0,
        "pre_clip_grad_norm": 8.5,
        # Advantage stats (always emitted)
        "advantage_mean": 0.3,
        "advantage_std": 0.8,
        "advantage_skewness": -0.1,
        "advantage_kurtosis": 2.5,
        "advantage_positive_ratio": 0.48,
        # Ratio stats (always emitted)
        "ratio_mean": 1.02,
        "ratio_min": 0.85,
        "ratio_max": 1.15,
        "ratio_std": 0.08,
        # Log prob extremes (always emitted)
        "log_prob_min": -4.5,
        "log_prob_max": -0.8,
        # Always emitted
        "entropy_collapsed": False,
        "update_time_ms": 120.0,
        "inner_epoch": 1,
        "batch": 2,
        "ppo_updates_count": 2,
        # Value function statistics (always emitted)
        "value_mean": 4.5,
        "value_std": 1.8,
        "value_min": 0.5,
        "value_max": 9.0,
        # Gradient quality metrics (always emitted)
        "clip_fraction_positive": 0.10,
        "clip_fraction_negative": 0.05,
        "gradient_cv": 0.35,
        # Q-values (optional but what this test focuses on)
        "q_germinate": 5.2,
        "q_advance": 3.1,
        "q_fossilize": 2.8,
        "q_prune": -1.5,
        "q_wait": 0.5,
        "q_set_alpha": 4.0,
        "q_variance": 2.3,
        "q_spread": 6.7,
        # Pre-normalization advantage stats (always emitted)
        "pre_norm_advantage_mean": 0.6,
        "pre_norm_advantage_std": 2.0,
        # Return statistics (always emitted)
        "return_mean": 8.5,
        "return_std": 2.5,
    }

    payload = PPOUpdatePayload.from_dict(data)

    assert payload.pre_clip_grad_norm == 8.5
    assert payload.ppo_updates_count == 2
    assert payload.q_germinate == 5.2
    assert payload.q_variance == 2.3
