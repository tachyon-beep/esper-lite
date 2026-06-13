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


def test_epoch_completed_payload_accepts_null_observation_stats() -> None:
    """EpochCompletedPayload.from_dict should accept observation_stats=None."""
    from esper.leyline.telemetry import EpochCompletedPayload

    payload = EpochCompletedPayload.from_dict(
        {
            "env_id": 0,
            "val_accuracy": 0.75,
            "val_loss": 0.42,
            "inner_epoch": 3,
            "episode_idx": 0,  # REQUIRED: injected by emit_with_env_context
            "observation_stats": None,
        }
    )

    assert payload.observation_stats is None


def test_morphology_causal_log_payload_requires_joinable_identity() -> None:
    """Morphology causal logs must join proposal, verdict, mutation, watch, and terminal evidence."""
    from esper.leyline.telemetry import MorphologyCausalLogPayload

    payload = MorphologyCausalLogPayload.from_dict(
        {
            "phase": "watch",
            "env_id": 2,
            "slot_id": "r0c0",
            "operation": "GERMINATE",
            "action_id": "morph-b3-e4-env2-r0c0-op1",
            "proposal_id": "morph-b3-e4-env2-r0c0-op1-proposal",
            "verdict_id": "morph-b3-e4-env2-r0c0-op1-verdict",
            "mutation_id": "morph-b3-e4-env2-r0c0-op1-mutation",
            "observation_hash": "obs-abc123",
            "rng_stream": "simic.lifecycle.env2",
            "rng_seed": 123456789,
            "topology": "cnn",
            "blueprint_id": "conv_light",
            "governor_approved": True,
            "governor_reason": "approved",
            "governor_blocked_factor": None,
            "watch_window_evidence": 1.25,
            "linked_event_id": "rollback-1",
        }
    )

    assert payload.phase == "watch"
    assert payload.action_id == "morph-b3-e4-env2-r0c0-op1"
    assert payload.proposal_id.endswith("-proposal")
    assert payload.verdict_id.endswith("-verdict")
    assert payload.mutation_id.endswith("-mutation")
    assert payload.observation_hash == "obs-abc123"
    assert payload.rng_stream == "simic.lifecycle.env2"
    assert payload.rng_seed == 123456789
    assert payload.governor_approved is True
    assert payload.watch_window_evidence == 1.25
    assert payload.linked_event_id == "rollback-1"


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
        "entropy_loss": 0.0,  # REQUIRED: always emitted
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
        # Q-values (required)
        "op_q_values": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "op_valid_mask": [True, True, True, True, True, True],
        "q_variance": 0.0,
        "q_spread": 0.0,
        # Infrastructure metrics
        "cuda_memory_allocated_gb": 5.0,
        "cuda_memory_reserved_gb": 10.0,
        "cuda_memory_peak_gb": 7.5,
        "cuda_memory_fragmentation": 0.30,
        "dataloader_wait_ratio": 0.2,
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
    assert payload.dataloader_wait_ratio == 0.2


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


def _full_head_telemetry_dict() -> dict[str, float]:
    """A complete head_telemetry dict (all 16 fields present)."""
    return {
        "op_confidence": 0.85,
        "slot_confidence": 0.72,
        "blueprint_confidence": 0.91,
        "style_confidence": 0.65,
        "tempo_confidence": 0.88,
        "alpha_target_confidence": 0.77,
        "alpha_speed_confidence": 0.69,
        "curve_confidence": 0.82,
        "op_entropy": 0.3,
        "slot_entropy": 0.8,
        "blueprint_entropy": 0.5,
        "style_entropy": 0.6,
        "tempo_entropy": 0.4,
        "alpha_target_entropy": 0.55,
        "alpha_speed_entropy": 0.45,
        "curve_entropy": 0.35,
    }


def test_head_telemetry_from_dict_fails_fast_on_missing_field():
    """TPD-004: a partial head_telemetry dict must NOT fabricate 0.0; fail loud.

    Every field is produced together by the policy or not at all (None), so a
    missing key means corrupted/partial telemetry.
    """
    import pytest

    from esper.leyline.telemetry import HeadTelemetry

    partial = _full_head_telemetry_dict()
    del partial["curve_entropy"]

    with pytest.raises(KeyError):
        HeadTelemetry.from_dict(partial)


def test_head_telemetry_from_dict_empty_fails_fast():
    """TPD-004: an empty head_telemetry dict must raise, not return all-zeros."""
    import pytest

    from esper.leyline.telemetry import HeadTelemetry

    with pytest.raises(KeyError):
        HeadTelemetry.from_dict({})


def test_head_telemetry_from_dict_preserves_explicit_zero():
    """TPD-004: an explicit measured 0.0 round-trips as 0.0 (not dropped)."""
    from esper.leyline.telemetry import HeadTelemetry

    data = _full_head_telemetry_dict()
    data["op_confidence"] = 0.0
    data["op_entropy"] = 0.0

    head = HeadTelemetry.from_dict(data)
    assert head.op_confidence == 0.0
    assert head.op_entropy == 0.0
    # Round-trip stability
    assert HeadTelemetry.from_dict(head.to_dict()) == head


# =============================================================================
# BatchEpochCompletedPayload rolling_accuracy missingness (LN-004)
# =============================================================================


def _full_batch_dict() -> dict:
    """A complete batch_epoch dict WITHOUT rolling_accuracy."""
    return {
        "episodes_completed": 8,
        "batch_idx": 2,
        "avg_accuracy": 61.0,
        "avg_reward": 1.5,
        "total_episodes": 32,
        "n_envs": 4,
    }


def test_batch_payload_rolling_accuracy_absent_is_none():
    """LN-004: absent rolling_accuracy parses as None (not measured), not 0.0."""
    from esper.leyline.telemetry import BatchEpochCompletedPayload

    payload = BatchEpochCompletedPayload.from_dict(_full_batch_dict())
    assert payload.rolling_accuracy is None


def test_batch_payload_rolling_accuracy_explicit_zero_preserved():
    """LN-004: an explicit measured 0.0 stays 0.0, distinct from absent (None)."""
    from esper.leyline.telemetry import BatchEpochCompletedPayload

    data = _full_batch_dict()
    data["rolling_accuracy"] = 0.0
    payload = BatchEpochCompletedPayload.from_dict(data)
    assert payload.rolling_accuracy == 0.0
    assert payload.rolling_accuracy is not None


def test_batch_payload_rolling_accuracy_roundtrip_stable():
    """LN-004: None stays None and 0.0 stays 0.0 through serialize -> from_dict."""
    from esper.karn.serialization import _payload_to_dict
    from esper.leyline.telemetry import BatchEpochCompletedPayload

    # None case
    none_payload = BatchEpochCompletedPayload.from_dict(_full_batch_dict())
    none_round = BatchEpochCompletedPayload.from_dict(_payload_to_dict(none_payload))
    assert none_round.rolling_accuracy is None

    # Explicit zero case
    zero_payload = BatchEpochCompletedPayload(
        episodes_completed=8,
        batch_idx=2,
        avg_accuracy=61.0,
        avg_reward=1.5,
        total_episodes=32,
        n_envs=4,
        rolling_accuracy=0.0,
    )
    zero_round = BatchEpochCompletedPayload.from_dict(_payload_to_dict(zero_payload))
    assert zero_round.rolling_accuracy == 0.0
    assert zero_round.rolling_accuracy is not None


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
        # Q-values per operation (LifecycleOp order)
        op_q_values=(0.5, 5.2, 4.0, -1.5, 2.8, 3.1),
        op_valid_mask=(True, True, True, True, True, True),
        q_variance=2.3,
        q_spread=6.7,
    )

    assert payload.op_q_values == (0.5, 5.2, 4.0, -1.5, 2.8, 3.1)
    assert payload.op_valid_mask == (True, True, True, True, True, True)
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
        "entropy_loss": 0.0,  # REQUIRED: always emitted
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
        # Q-values (required)
        "op_q_values": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "op_valid_mask": [True, True, True, True, True, True],
        "q_variance": 0.0,
        "q_spread": 0.0,
        # Pre-normalization advantage stats (always emitted)
        "pre_norm_advantage_mean": 0.0,
        "pre_norm_advantage_std": 1.0,
        # Return statistics (always emitted)
        "return_mean": 0.0,
        "return_std": 1.0,
        # Infrastructure metrics (always emitted)
        "dataloader_wait_ratio": 0.15,
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
        "entropy_loss": 0.0,  # REQUIRED: always emitted
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
        "op_q_values": [0.5, 5.2, 4.0, -1.5, 2.8, 3.1],
        "op_valid_mask": [True, True, True, True, True, True],
        "q_variance": 2.3,
        "q_spread": 6.7,
        # Pre-normalization advantage stats (always emitted)
        "pre_norm_advantage_mean": 0.6,
        "pre_norm_advantage_std": 2.0,
        # Return statistics (always emitted)
        "return_mean": 8.5,
        "return_std": 2.5,
        # Infrastructure metrics (always emitted)
        "dataloader_wait_ratio": 0.12,
    }

    payload = PPOUpdatePayload.from_dict(data)

    assert payload.pre_clip_grad_norm == 8.5
    assert payload.ppo_updates_count == 2
    assert payload.op_q_values == (0.5, 5.2, 4.0, -1.5, 2.8, 3.1)
    assert payload.q_variance == 2.3


# =============================================================================
# LN-001: EpochCompletedPayload.episode_idx survives serialization
# =============================================================================


def test_epoch_completed_to_dict_includes_episode_idx() -> None:
    """to_dict() must emit episode_idx; from_dict() requires it (no silent drop)."""
    from esper.leyline.telemetry import EpochCompletedPayload

    payload = EpochCompletedPayload(
        env_id=2,
        val_accuracy=0.81,
        val_loss=0.33,
        inner_epoch=7,
        episode_idx=5,
    )

    serialized = payload.to_dict()
    assert "episode_idx" in serialized
    assert serialized["episode_idx"] == 5


def test_epoch_completed_to_dict_preserves_none_episode_idx() -> None:
    """A genuinely absent episode_idx (None) round-trips as None, not 0."""
    from esper.leyline.telemetry import EpochCompletedPayload

    payload = EpochCompletedPayload(
        env_id=0,
        val_accuracy=0.5,
        val_loss=1.0,
        inner_epoch=0,
        episode_idx=None,
        observation_stats=None,
    )

    serialized = payload.to_dict()
    assert serialized["episode_idx"] is None

    restored = EpochCompletedPayload.from_dict(serialized)
    assert restored.episode_idx is None


def test_epoch_completed_round_trip_preserves_episode_idx() -> None:
    """to_dict() -> from_dict() must preserve episode_idx exactly."""
    from esper.leyline.telemetry import EpochCompletedPayload

    original = EpochCompletedPayload(
        env_id=3,
        val_accuracy=0.9,
        val_loss=0.12,
        inner_epoch=4,
        episode_idx=11,
        observation_stats=None,
    )

    restored = EpochCompletedPayload.from_dict(original.to_dict())
    assert restored.episode_idx == 11
    assert restored == original


# =============================================================================
# LN-002: SeedPrunedPayload.blueprint_id optionality
# =============================================================================


def test_seed_pruned_payload_preserves_blueprint_id() -> None:
    """A normal prune carries its blueprint id through from_dict unchanged."""
    from esper.leyline.telemetry import SeedPrunedPayload

    payload = SeedPrunedPayload.from_dict(
        {
            "slot_id": "r0c0",
            "env_id": 0,
            "reason": "no_improvement",
            "episode_idx": 4,
            "blueprint_id": "attention",
        }
    )
    assert payload.blueprint_id == "attention"


def test_seed_pruned_payload_allows_absent_blueprint_id() -> None:
    """A slot culled before germination recorded a blueprint has blueprint_id=None.

    None is a legitimate, distinguishable "unknown blueprint" marker, not coerced
    to a fake id.
    """
    from esper.leyline.telemetry import SeedPrunedPayload

    payload = SeedPrunedPayload.from_dict(
        {
            "slot_id": "r0c0",
            "env_id": 0,
            "reason": "probation",
            "episode_idx": 1,
        }
    )
    assert payload.blueprint_id is None
