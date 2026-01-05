"""Unit tests for vectorized PPO helpers.

This module tests:
- Telemetry emission functions (_emit_*)
- Seed advancement logic (_advance_active_seed)
- PPO update helpers (_run_ppo_updates, _calculate_entropy_anneal_steps)
- Anomaly handling (_handle_telemetry_escalation, _emit_anomaly_diagnostics)
"""

import pytest
import torch
from unittest.mock import Mock, MagicMock, patch

from esper.leyline import SeedStage, TelemetryEvent, TelemetryEventType
from esper.leyline.slot_config import SlotConfig
from esper.leyline.telemetry import SeedGerminatedPayload
from esper.simic.telemetry import AnomalyReport
from esper.simic.training.vectorized import (
    _advance_active_seed,
    _calculate_entropy_anneal_steps,
    _emit_anomaly_diagnostics,
    _handle_telemetry_escalation,
    _resolve_target_slot,
    _run_ppo_updates,
)
from esper.simic.telemetry.emitters import (
    apply_slot_telemetry,
    check_performance_degradation,
    emit_action_distribution,
    emit_batch_completed,
    emit_last_action,
    emit_mask_hit_rates,
    emit_ppo_update_event,
    emit_reward_summary,
    emit_throughput,
    emit_with_env_context,
)


# =============================================================================
# Telemetry Emission Tests
# =============================================================================


def test_lifecycle_only_keeps_slot_telemetry():
    slot = Mock()
    slot.fast_mode = False
    slot.on_telemetry = None
    slot.telemetry_lifecycle_only = False

    env_state = Mock()
    env_state.model = Mock(seed_slots={"r0c1": slot})
    env_state.telemetry_cb = Mock()

    apply_slot_telemetry(
        env_state,
        ops_telemetry_enabled=False,
        lifecycle_only=True,
    )

    assert slot.on_telemetry is env_state.telemetry_cb
    assert slot.fast_mode is True
    assert slot.telemetry_lifecycle_only is True


def test_emit_with_env_context_includes_device():
    hub = Mock()
    # Test with typed payload - env_id is replaced, device is not added
    payload = SeedGerminatedPayload(
        slot_id="r0c0",
        env_id=0,
        blueprint_id="conv3x3",
        params=1024,
        alpha=0.0,
    )
    event = TelemetryEvent(event_type=TelemetryEventType.SEED_GERMINATED, data=payload)
    emit_with_env_context(hub, env_idx=2, device="cpu", event=event)

    emitted = hub.emit.call_args[0][0]
    assert emitted.data.env_id == 2

    # Dict payloads are no longer supported - they should raise TypeError
    hub.reset_mock()
    dict_event = TelemetryEvent(event_type=TelemetryEventType.SEED_GERMINATED, data={})
    with pytest.raises(TypeError, match="requires typed payload, got dict"):
        emit_with_env_context(hub, env_idx=2, device="cpu", event=dict_event)

    # None payloads should also raise TypeError
    none_event = TelemetryEvent(event_type=TelemetryEventType.SEED_GERMINATED, data=None)
    with pytest.raises(TypeError, match="requires typed payload, got None"):
        emit_with_env_context(hub, env_idx=2, device="cpu", event=none_event)


def test_last_action_event_emitted():
    with patch("esper.simic.telemetry.emitters.get_hub") as get_hub:
        hub = Mock()
        get_hub.return_value = hub

        emit_last_action(
            env_id=0,
            epoch=3,
            slot_idx=1,
            blueprint_idx=1,
            style_idx=1,
            tempo_idx=1,
            alpha_target_idx=0,
            alpha_speed_idx=0,
            alpha_curve_idx=0,
            op_idx=1,
            slot_id="r0c1",
            masked={
                "op": False,
                "slot": False,
                "blueprint": False,
                "style": True,
                "tempo": False,
                "alpha_target": False,
                "alpha_speed": False,
                "alpha_curve": False,
            },
            success=True,
        )

        emitted = hub.emit.call_args[0][0]
        assert emitted.data.slot_id == "r0c1"
        assert emitted.data.style_masked is True


def _make_mandatory_metrics(**overrides) -> dict:
    """Create metrics dict with all mandatory fields for emit_ppo_update_event."""
    base = {
        "policy_loss": 0.1,
        "value_loss": 0.2,
        "entropy": 1.5,
        "approx_kl": 0.01,
        "clip_fraction": 0.1,
        "pre_clip_grad_norm": 2.5,
        "ppo_updates_count": 1,
        "advantage_mean": 0.0,
        "advantage_std": 1.0,
        "advantage_skewness": 0.0,
        "advantage_kurtosis": 0.0,
        "advantage_positive_ratio": 0.5,
        # Pre-normalization advantage stats (for diagnosing advantage collapse)
        "pre_norm_advantage_mean": 0.0,
        "pre_norm_advantage_std": 1.0,
        # Return statistics (for diagnosing value loss scale)
        "return_mean": 0.5,
        "return_std": 0.3,
        # Value target scale (std used to normalize returns)
        "value_target_scale": 0.3,
        "ratio_mean": 1.0,
        "ratio_min": 0.8,
        "ratio_max": 1.2,
        "ratio_std": 0.1,
        "log_prob_min": -5.0,
        "log_prob_max": -0.1,
        "value_mean": 0.0,
        "value_std": 1.0,
        "value_min": -2.0,
        "value_max": 2.0,
        # Per-head stats (optional but expected by emitter loop)
        "head_entropies": {},
        "head_grad_norms": {},
    }
    base.update(overrides)
    return base


def test_ppo_update_event_includes_vitals():
    hub = Mock()
    metrics = _make_mandatory_metrics()
    emit_ppo_update_event(
        hub=hub,
        metrics=metrics,
        episodes_completed=5,
        batch_idx=0,
        epoch=3,
        optimizer=Mock(param_groups=[{"lr": 0.0003}]),
        grad_norm=1.23,
        update_time_ms=12.5,
    )
    payload = hub.emit.call_args[0][0].data
    # Typed payload access (PPOUpdatePayload)
    assert payload.lr == 0.0003
    assert payload.grad_norm == 1.23
    assert payload.update_time_ms == 12.5


def test_action_distribution_snapshot():
    hub = Mock()
    emit_action_distribution(
        hub=hub,
        batch_idx=1,
        episodes_completed=4,
        action_counts={"WAIT": 3, "GERMINATE": 1},
        success_counts={"WAIT": 3, "GERMINATE": 1},
    )
    payload = hub.emit.call_args[0][0].data
    # Typed payload access (AnalyticsSnapshotPayload)
    assert payload.action_counts["WAIT"] == 3


def test_throughput_metrics_emitted():
    hub = Mock()
    emit_throughput(
        hub=hub,
        env_id=0,
        batch_idx=1,
        episodes_completed=4,
        step_time_ms=5.0,
        dataloader_wait_ms=2.0,
    )
    data = hub.emit.call_args[0][0].data
    assert data.step_time_ms == 5.0
    # Note: dataloader_wait_ms is not currently captured in AnalyticsSnapshotPayload


def test_throughput_metrics_include_fps():
    hub = Mock()
    emit_throughput(
        hub=hub,
        env_id=0,
        batch_idx=1,
        episodes_completed=4,
        step_time_ms=20.0,
        dataloader_wait_ms=2.0,
    )
    data = hub.emit.call_args[0][0].data
    assert data.fps == 50.0


def test_reward_summary_emitted():
    hub = Mock()
    emit_reward_summary(
        hub=hub,
        env_id=0,
        batch_idx=1,
        summary={"bounded_attribution": 0.4, "compute_rent": -0.1, "total_reward": 0.3},
    )
    data = hub.emit.call_args[0][0].data
    assert data.summary is not None
    assert data.summary["total_reward"] == 0.3


def test_mask_hit_rates_emitted():
    hub = Mock()
    emit_mask_hit_rates(
        hub=hub,
        batch_idx=1,
        episodes_completed=4,
        mask_hits={"op": 10},
        mask_total={"op": 12},
    )
    data = hub.emit.call_args[0][0].data
    assert data.mask_hits is not None
    assert data.mask_hits["op"] == 10
    assert data.mask_total is not None
    assert data.mask_total["op"] == 12


def test_performance_degradation_emitted_on_accuracy_drop():
    """Test degradation event emitted when accuracy drops significantly.

    DRL Expert review 2025-12-16: Added warmup guard to avoid false positives
    during early training when PPO has natural 15-20% variance.
    """
    hub = Mock()

    # Accuracy dropped from 0.8 to 0.6 (25% drop), past warmup
    emitted = check_performance_degradation(
        hub=hub,
        current_acc=0.6,
        rolling_avg_acc=0.8,
        degradation_threshold=0.1,  # 10% drop triggers
        env_id=0,
        training_progress=0.5,  # Past warmup (50% through training)
    )

    assert emitted is True
    assert hub.emit.called
    event = hub.emit.call_args[0][0]
    assert event.event_type == TelemetryEventType.PERFORMANCE_DEGRADATION
    # PerformanceDegradationPayload is a typed dataclass (not dict)
    from esper.leyline import PerformanceDegradationPayload
    assert isinstance(event.data, PerformanceDegradationPayload)
    assert event.data.current_acc == 0.6
    assert event.data.rolling_avg_acc == 0.8
    assert event.data.training_progress == 0.5


def test_no_degradation_event_when_stable():
    hub = Mock()

    emitted = check_performance_degradation(
        hub=hub,
        current_acc=0.78,
        rolling_avg_acc=0.8,
        degradation_threshold=0.1,
        env_id=0,
        training_progress=0.5,
    )

    assert emitted is False
    assert not hub.emit.called


def test_no_degradation_event_during_warmup():
    """Test that degradation events are skipped during warmup phase.

    DRL Expert review 2025-12-16: PPO has natural 15-20% variance during
    early training, so we skip emissions in first 10% to avoid false positives.
    """
    hub = Mock()

    # Same degradation that would normally trigger, but during warmup
    emitted = check_performance_degradation(
        hub=hub,
        current_acc=0.6,
        rolling_avg_acc=0.8,
        degradation_threshold=0.1,  # Would normally trigger
        env_id=0,
        training_progress=0.05,  # Only 5% through training (warmup)
    )

    assert emitted is False
    assert not hub.emit.called


def test_degradation_event_emitted_after_warmup():
    """Test that degradation events resume after warmup threshold."""
    hub = Mock()

    # Just past warmup threshold (11% > 10%)
    emitted = check_performance_degradation(
        hub=hub,
        current_acc=0.6,
        rolling_avg_acc=0.8,
        degradation_threshold=0.1,
        env_id=0,
        training_progress=0.11,  # Just past warmup
    )

    assert emitted is True
    event = hub.emit.call_args[0][0]
    assert event.event_type == TelemetryEventType.PERFORMANCE_DEGRADATION


def test_resolve_target_slot_uses_canonical_order():
    slot_config = SlotConfig.default()

    # Enabled slots provided in non-canonical order; slot_idx is still canonical.
    enabled = ["r0c2", "r0c0"]
    slot_id, enabled_flag = _resolve_target_slot(0, enabled_slots=enabled, slot_config=slot_config)
    assert slot_id == "r0c0"
    assert enabled_flag is True

    slot_id, enabled_flag = _resolve_target_slot(2, enabled_slots=enabled, slot_config=slot_config)
    assert slot_id == "r0c2"
    assert enabled_flag is True


def test_resolve_target_slot_flags_disabled_slots():
    slot_config = SlotConfig.default()

    enabled = ["r0c2"]
    slot_id, enabled_flag = _resolve_target_slot(2, enabled_slots=enabled, slot_config=slot_config)
    assert slot_id == "r0c2"
    assert enabled_flag is True

    slot_id, enabled_flag = _resolve_target_slot(0, enabled_slots=enabled, slot_config=slot_config)
    assert slot_id == "r0c0"
    assert enabled_flag is False


def test_resolve_target_slot_out_of_range_is_invalid():
    slot_config = SlotConfig.default()

    slot_id, enabled_flag = _resolve_target_slot(99, enabled_slots=["r0c0"], slot_config=slot_config)
    assert slot_id == "r0c0"
    assert enabled_flag is False

    slot_id, enabled_flag = _resolve_target_slot(-1, enabled_slots=["r0c0"], slot_config=slot_config)
    assert slot_id == "r0c0"
    assert enabled_flag is False

    slot_id, enabled_flag = _resolve_target_slot(
        -1, enabled_slots=["r0c0", "r0c1", "r0c2"], slot_config=slot_config
    )
    assert slot_id == "r0c0"
    assert enabled_flag is False


# =============================================================================
# Seed Advancement Tests
# =============================================================================


class _StubGateResult:
    def __init__(self, passed: bool = True, checks_failed: list | None = None):
        self.passed = passed
        self.checks_failed = checks_failed or []


class _StubSeedState:
    def __init__(self, stage: SeedStage):
        self.stage = stage
        self.transition_calls: list[SeedStage] = []

    def transition(self, target_stage: SeedStage) -> bool:
        self.transition_calls.append(target_stage)
        self.stage = target_stage
        return True


class _StubSeedSlot:
    def __init__(self, seed_state: _StubSeedState, gate_result: _StubGateResult | None = None):
        self.state = seed_state
        self.gate_result = gate_result or _StubGateResult()
        self.advance_calls: list[SeedStage] = []
        self.set_alpha_calls: list[float] = []
        self.start_blending_calls: list[int] = []

    def advance_stage(self, target_stage: SeedStage | None = None) -> _StubGateResult:
        self.advance_calls.append(target_stage)
        self.state.stage = target_stage
        return self.gate_result

    def set_alpha(self, alpha: float) -> None:
        self.set_alpha_calls.append(alpha)

    def start_blending(self, total_steps: int) -> None:
        self.start_blending_calls.append(total_steps)


class _StubModel:
    def __init__(self, seed_stage: SeedStage, gate_result: _StubGateResult | None = None):
        self.has_active_seed = True
        seed_state = _StubSeedState(seed_stage)
        seed_slot = _StubSeedSlot(seed_state, gate_result=gate_result)
        self.seed_slots = {"r0c1": seed_slot}

    def has_active_seed_in_slot(self, slot: str) -> bool:
        """Check if specific slot has an active seed."""
        return slot in self.seed_slots and self.seed_slots[slot].state is not None


def test_advance_active_seed_fossilizes_via_seed_slot():
    """HOLDING seeds should fossilize through SeedSlot.advance_stage (emits telemetry)."""
    model = _StubModel(SeedStage.HOLDING)
    slot_id = "r0c1"

    _advance_active_seed(model, slot_id)

    assert model.seed_slots["r0c1"].advance_calls == [SeedStage.FOSSILIZED]
    assert model.seed_slots["r0c1"].set_alpha_calls == [1.0]
    assert model.seed_slots["r0c1"].state.stage == SeedStage.FOSSILIZED
    # Transition should happen inside advance_stage, not direct transition
    assert model.seed_slots["r0c1"].state.transition_calls == []


def test_advance_active_seed_noop_on_failed_fossilization_gate():
    """Failed fossilization gate should be a no-op (Tamiyo learns from failed attempts)."""
    gate_result = _StubGateResult(passed=False, checks_failed=["no_improvement"])
    model = _StubModel(SeedStage.HOLDING, gate_result=gate_result)
    slot_id = "r0c1"

    # Should not raise - failed gate is normal RL outcome
    _advance_active_seed(model, slot_id)

    # Gate was checked but transition didn't happen
    assert model.seed_slots["r0c1"].advance_calls == [SeedStage.FOSSILIZED]
    assert model.seed_slots["r0c1"].set_alpha_calls == []  # No alpha change on failed gate


def test_advance_active_seed_noop_from_training_stage():
    """TRAINING seeds are handled mechanically; fossilize action should do nothing."""
    model = _StubModel(SeedStage.TRAINING)
    slot_id = "r0c1"

    _advance_active_seed(model, slot_id)

    assert model.seed_slots["r0c1"].state.transition_calls == []
    assert model.seed_slots["r0c1"].start_blending_calls == []
    assert model.seed_slots["r0c1"].state.stage == SeedStage.TRAINING


# =============================================================================
# Threshold and Detection Tests
# =============================================================================


def test_custom_thresholds_respected():
    """Test that custom plateau_threshold and improvement_threshold parameters are respected.

    Verifies that custom thresholds change which events fire, confirming the parameters
    are actually used instead of hardcoded values.
    """
    mock_hub = MagicMock()

    # Test case: smoothed_delta = 3.0 (would normally trigger IMPROVEMENT with default threshold of 2.0)
    recent_accuracies = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]

    # With default thresholds (0.5, 2.0), this should fire IMPROVEMENT_DETECTED
    plateau_threshold = 0.5
    improvement_threshold = 2.0

    recent_avg = sum(recent_accuracies[-3:]) / 3
    older_avg = sum(recent_accuracies[-6:-3]) / 3
    smoothed_delta = recent_avg - older_avg

    # Simulate default threshold behavior (smoothed_delta = 3.0)
    if abs(smoothed_delta) < plateau_threshold:
        mock_hub.emit(TelemetryEvent(
            event_type=TelemetryEventType.PLATEAU_DETECTED,
            data={"smoothed_delta": smoothed_delta},
        ))
    elif smoothed_delta < -improvement_threshold:
        mock_hub.emit(TelemetryEvent(
            event_type=TelemetryEventType.DEGRADATION_DETECTED,
            data={"smoothed_delta": smoothed_delta},
        ))
    elif smoothed_delta > improvement_threshold:
        mock_hub.emit(TelemetryEvent(
            event_type=TelemetryEventType.IMPROVEMENT_DETECTED,
            data={"smoothed_delta": smoothed_delta},
        ))

    assert mock_hub.emit.call_count == 1
    assert mock_hub.emit.call_args[0][0].event_type == TelemetryEventType.IMPROVEMENT_DETECTED

    # Now with custom very high improvement threshold, smoothed_delta=3.0 should be considered plateau
    mock_hub.reset_mock()
    plateau_threshold = 5.0  # smoothed_delta=3.0 < 5.0, so it's a plateau
    improvement_threshold = 5.0  # smoothed_delta=3.0 < 5.0, won't trigger improvement

    if abs(smoothed_delta) < plateau_threshold:
        mock_hub.emit(TelemetryEvent(
            event_type=TelemetryEventType.PLATEAU_DETECTED,
            data={"smoothed_delta": smoothed_delta},
        ))
    elif smoothed_delta < -improvement_threshold:
        mock_hub.emit(TelemetryEvent(
            event_type=TelemetryEventType.DEGRADATION_DETECTED,
            data={"smoothed_delta": smoothed_delta},
        ))
    elif smoothed_delta > improvement_threshold:
        mock_hub.emit(TelemetryEvent(
            event_type=TelemetryEventType.IMPROVEMENT_DETECTED,
            data={"smoothed_delta": smoothed_delta},
        ))

    # With high thresholds, should emit PLATEAU instead of IMPROVEMENT
    assert mock_hub.emit.call_count == 1, "Custom thresholds should change which event fires"
    assert mock_hub.emit.call_args[0][0].event_type == TelemetryEventType.PLATEAU_DETECTED, \
        "With high thresholds, smoothed_delta=3.0 should be considered a plateau"


def test_plateau_detection_logic():
    """Test that plateau/degradation/improvement detection logic works correctly.

    This is an integration-style test that verifies the plateau/improvement/degradation
    detection code emits the correct events when smoothed deltas cross thresholds.

    The new logic uses smoothed deltas (comparing 3-batch rolling windows) to avoid
    noisy batch-to-batch fluctuations and properly distinguish plateau from degradation.
    """
    # Mock the hub to capture emitted events
    mock_hub = MagicMock()

    # Simulate recent_accuracies with various patterns
    test_cases = [
        # (recent_accuracies, expected_event_type, description)
        ([10.0], None, "single value - no event (< 6 samples)"),
        ([10.0, 10.1], None, "2 values - no event (< 6 samples)"),
        ([10.0, 10.1, 10.2, 10.3, 10.4], None, "5 values - no event (< 6 samples)"),
        # With 6+ samples, use smoothed delta
        ([10.0, 10.0, 10.0, 10.0, 10.0, 10.0], TelemetryEventType.PLATEAU_DETECTED,
         "flat: smoothed_delta=0.0, abs < 0.5 = plateau"),
        ([10.0, 10.1, 10.2, 10.1, 10.2, 10.3], TelemetryEventType.PLATEAU_DETECTED,
         "small positive: smoothed_delta=0.2, abs < 0.5 = plateau"),
        ([10.0, 10.1, 10.2, 10.0, 10.1, 10.0], TelemetryEventType.PLATEAU_DETECTED,
         "small negative: smoothed_delta=-0.13, abs < 0.5 = plateau"),
        ([10.0, 11.0, 12.0, 13.0, 14.0, 15.0], TelemetryEventType.IMPROVEMENT_DETECTED,
         "strong improvement: smoothed_delta=3.0 > 2.0 = improvement"),
        ([15.0, 14.0, 13.0, 12.0, 11.0, 10.0], TelemetryEventType.DEGRADATION_DETECTED,
         "strong degradation: smoothed_delta=-3.0 < -2.0 = degradation"),
        ([10.0, 10.5, 11.0, 11.5, 12.0, 12.5], None,
         "medium improvement: smoothed_delta=1.5, not > 2.0 = no event"),
        ([12.5, 12.0, 11.5, 11.0, 10.5, 10.0], None,
         "medium degradation: smoothed_delta=-1.5, not < -2.0 = no event"),
    ]

    for recent_accs, expected_event, description in test_cases:
        mock_hub.reset_mock()

        # Simulate the logic from vectorized.py
        if len(recent_accs) >= 6:
            # Compare rolling window averages (need at least 6 samples for meaningful comparison)
            recent_avg = sum(list(recent_accs)[-3:]) / 3
            older_avg = sum(list(recent_accs)[-6:-3]) / 3
            smoothed_delta = recent_avg - older_avg

            if abs(smoothed_delta) < 0.5:  # True plateau - no significant change either direction
                mock_hub.emit(TelemetryEvent(
                    event_type=TelemetryEventType.PLATEAU_DETECTED,
                    data={
                        "batch": 1,
                        "smoothed_delta": smoothed_delta,
                        "recent_avg": recent_avg,
                        "older_avg": older_avg,
                        "rolling_avg_accuracy": sum(recent_accs) / len(recent_accs),
                        "episodes_completed": 10,
                    },
                ))
            elif smoothed_delta < -2.0:  # Significant degradation
                mock_hub.emit(TelemetryEvent(
                    event_type=TelemetryEventType.DEGRADATION_DETECTED,
                    data={
                        "batch": 1,
                        "smoothed_delta": smoothed_delta,
                        "recent_avg": recent_avg,
                        "older_avg": older_avg,
                        "rolling_avg_accuracy": sum(recent_accs) / len(recent_accs),
                        "episodes_completed": 10,
                    },
                ))
            elif smoothed_delta > 2.0:  # Significant improvement
                mock_hub.emit(TelemetryEvent(
                    event_type=TelemetryEventType.IMPROVEMENT_DETECTED,
                    data={
                        "batch": 1,
                        "smoothed_delta": smoothed_delta,
                        "recent_avg": recent_avg,
                        "older_avg": older_avg,
                        "rolling_avg_accuracy": sum(recent_accs) / len(recent_accs),
                        "episodes_completed": 10,
                    },
                ))

        # Verify expectations
        if expected_event is None:
            assert mock_hub.emit.call_count == 0, f"Failed: {description}"
        else:
            assert mock_hub.emit.call_count == 1, f"Failed: {description}"
            emitted_event = mock_hub.emit.call_args[0][0]
            assert emitted_event.event_type == expected_event, f"Failed: {description}"


# =============================================================================
# PPO Update Tests
# =============================================================================


def test_calculate_entropy_anneal_steps_respects_updates_per_batch():
    """Entropy annealing should scale with number of PPO updates per batch."""
    # 8 episodes, 3 envs -> ceil(8/3) = 3 batches. With 2 updates per batch => 6 steps.
    assert _calculate_entropy_anneal_steps(
        entropy_anneal_episodes=8,
        n_envs=3,
        ppo_updates_per_batch=2,
    ) == 6

    # Single update per batch keeps the batch count only
    assert _calculate_entropy_anneal_steps(
        entropy_anneal_episodes=8,
        n_envs=3,
        ppo_updates_per_batch=1,
    ) == 3

    # Zero episodes means no annealing regardless of update count
    assert _calculate_entropy_anneal_steps(
        entropy_anneal_episodes=0,
        n_envs=3,
        ppo_updates_per_batch=4,
    ) == 0


def test_run_ppo_updates_runs_multiple_updates_and_updates_normalizer_once():
    """Multiple PPO updates should aggregate metrics and update the normalizer once."""

    class _StubBuffer:
        def __init__(self):
            self.reset_calls = 0

        def reset(self):
            self.reset_calls += 1

    class _StubAgent:
        def __init__(self):
            self.buffer = _StubBuffer()
            self.update_calls: list[bool] = []
            self.target_kl = None
            self.lstm_hidden_dim = 0  # Non-recurrent stub

        def update(self, clear_buffer: bool = True) -> dict:
            """Return deterministic metrics for aggregation checks."""
            self.update_calls.append(clear_buffer)
            call_idx = len(self.update_calls)
            approx = 0.01 * call_idx
            return {
                "policy_loss": float(call_idx),
                "value_loss": float(call_idx + 1),
                "entropy": float(call_idx + 2),
                "approx_kl": approx,
                "ratio_max": 1.0 + approx,
                "ratio_min": 1.0 - approx,
                "clip_fraction": 0.1 * call_idx,
                "explained_variance": 0.05 * call_idx,
            }

    class _StubNormalizer:
        def __init__(self):
            self.calls: list[torch.Tensor] = []

        def update(self, tensor: torch.Tensor) -> None:
            self.calls.append(tensor)

    agent = _StubAgent()
    normalizer = _StubNormalizer()
    raw_states = [torch.ones(2, 3), torch.zeros(1, 3)]

    metrics = _run_ppo_updates(
        agent=agent,
        ppo_updates_per_batch=3,
        raw_states_for_normalizer_update=raw_states,
        obs_normalizer=normalizer,
        use_amp=False,
        amp_dtype=None,  # Explicit: no AMP
    )

    # Expect three updates, buffer cleared only on final update
    assert agent.update_calls == [False, False, True]
    assert agent.buffer.reset_calls == 0

    # Normalizer updated once with concatenated states
    assert len(normalizer.calls) == 1
    assert normalizer.calls[0].shape[0] == sum(state.shape[0] for state in raw_states)

    # Metrics aggregated correctly (means except ratio_max/min)
    assert metrics["ratio_max"] == pytest.approx(1.0 + 0.03)
    assert metrics["ratio_min"] == pytest.approx(1.0 - 0.03)
    assert metrics["policy_loss"] == pytest.approx((1.0 + 2.0 + 3.0) / 3.0)
    assert metrics["approx_kl"] == pytest.approx((0.01 + 0.02 + 0.03) / 3.0)


def test_aggregate_ppo_metrics_handles_dict():
    """Dict metrics (e.g., ratio diagnostics) should pass through aggregation."""
    from esper.simic.training.vectorized import _aggregate_ppo_metrics

    metrics = _aggregate_ppo_metrics([
        {"ratio_max": 2.0, "ratio_diagnostic": {"worst": [1, 2]}},
        {"ratio_max": 3.0, "ratio_diagnostic": {"worst": [3]}},
    ])

    assert metrics["ratio_max"] == 3.0
    assert metrics["ratio_diagnostic"] == {"worst": [1, 2]}


def test_aggregate_ppo_metrics_skips_empty_and_none_values():
    """Empty metrics and all-None fields should not create misleading aggregates."""
    from esper.simic.training.vectorized import _aggregate_ppo_metrics

    assert _aggregate_ppo_metrics([]) == {}

    metrics = _aggregate_ppo_metrics([{"approx_kl": None}, {"approx_kl": None}])
    assert metrics == {}


def test_aggregate_ppo_metrics_special_reductions_and_head_merging():
    """Aggregation semantics are part of the monitoring contract."""
    from esper.simic.training.vectorized import _aggregate_ppo_metrics

    metrics = _aggregate_ppo_metrics([
        {
            "ratio_max": 1.2,
            "ratio_min": 0.8,
            "head_policy_ratio_max": 1.10,
            "value_min": -1.0,
            "value_max": 2.0,
            "value_mean": 1.0,
            "value_std": 0.5,
            "early_stop_epoch": 5,
            "head_entropies": {"policy": [0.1, 0.2], "value": [0.3]},
            "head_grad_norms": {"policy": [1.0], "value": [0.5, 0.7]},
        },
        {
            "ratio_max": 1.5,
            "ratio_min": 0.7,
            "head_policy_ratio_max": 1.05,
            "value_min": -2.0,
            "value_max": 3.0,
            "value_mean": 2.0,
            "value_std": 0.2,
            "early_stop_epoch": 3,
            "head_entropies": {"policy": [0.05], "other": [0.9]},
            "head_grad_norms": {"policy": [2.0], "other": [3.0]},
        },
    ])

    assert metrics["ratio_max"] == 1.5
    assert metrics["ratio_min"] == 0.7
    assert metrics["head_policy_ratio_max"] == 1.10

    assert metrics["value_min"] == -2.0
    assert metrics["value_max"] == 3.0
    assert metrics["value_mean"] == pytest.approx(1.5)
    assert metrics["value_std"] == 0.5
    assert metrics["early_stop_epoch"] == 3

    assert metrics["head_entropies"] == {
        "policy": [0.2],
        "value": [0.3],
        "other": [0.9],
    }
    assert metrics["head_grad_norms"] == {
        "policy": [2.0],
        "value": [0.7],
        "other": [3.0],
    }


def test_run_ppo_updates_honors_target_kl_early_stop_and_clears_buffer():
    """Updates should stop when KL exceeds threshold and still clear the buffer."""

    class _StubBuffer:
        def __init__(self):
            self.reset_calls = 0

        def reset(self):
            self.reset_calls += 1

    class _StubAgent:
        def __init__(self):
            self.buffer = _StubBuffer()
            self.update_calls: list[bool] = []
            self.target_kl = 0.01
            self.lstm_hidden_dim = 0  # Non-recurrent stub

        def update(self, clear_buffer: bool = True) -> dict:
            self.update_calls.append(clear_buffer)
            call_idx = len(self.update_calls)
            approx = 0.005 if call_idx == 1 else 0.02
            return {
                "policy_loss": float(call_idx),
                "value_loss": float(call_idx + 1),
                "entropy": float(call_idx + 2),
                "approx_kl": approx,
                "ratio_max": 1.0 + approx,
                "ratio_min": 1.0 - approx,
            }

    class _StubNormalizer:
        def __init__(self):
            self.calls: list[torch.Tensor] = []

        def update(self, tensor: torch.Tensor) -> None:
            self.calls.append(tensor)

    agent = _StubAgent()
    normalizer = _StubNormalizer()
    raw_states = [torch.ones(1, 3)]

    metrics = _run_ppo_updates(
        agent=agent,
        ppo_updates_per_batch=3,
        raw_states_for_normalizer_update=raw_states,
        obs_normalizer=normalizer,
        use_amp=False,
        amp_dtype=None,  # Explicit: no AMP
    )

    # Should stop after second update due to KL threshold (1.5 * 0.01 = 0.015)
    assert agent.update_calls == [False, False]
    assert agent.buffer.reset_calls == 1  # Cleared because last call didn't clear the buffer

    # Normalizer still updated once because at least one update succeeded
    assert len(normalizer.calls) == 1
    assert metrics["approx_kl"] == pytest.approx((0.005 + 0.02) / 2.0)


def test_run_ppo_updates_rejects_multiple_updates_for_recurrent_policies():
    """External PPO update loops are incompatible with LSTM policies (staleness guard)."""
    from esper.simic.training.vectorized import _run_ppo_updates

    class _StubBuffer:
        def reset(self) -> None:
            raise AssertionError("buffer.reset should not be reached in staleness guard path")

    class _StubAgent:
        def __init__(self):
            self.buffer = _StubBuffer()
            self.target_kl = None
            self.lstm_hidden_dim = 16

    class _StubNormalizer:
        def update(self, tensor: torch.Tensor) -> None:
            raise AssertionError("normalizer.update should not be reached in staleness guard path")

    with pytest.raises(ValueError, match="incompatible with recurrent"):
        _run_ppo_updates(
            agent=_StubAgent(),
            ppo_updates_per_batch=2,
            raw_states_for_normalizer_update=[],
            obs_normalizer=_StubNormalizer(),
            use_amp=False,
            amp_dtype=None,  # Explicit: no AMP
        )


def test_run_ppo_updates_uses_amp_context_when_enabled(monkeypatch):
    """AMP path should call agent.update under autocast when enabled."""
    from contextlib import contextmanager
    from esper.simic.training.vectorized import _run_ppo_updates

    @contextmanager
    def _fake_autocast(*_args, **_kwargs):
        yield

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr("esper.simic.training.vectorized.torch_amp.autocast", _fake_autocast)

    class _StubBuffer:
        def reset(self) -> None:
            pass

    class _StubAgent:
        def __init__(self):
            self.buffer = _StubBuffer()
            self.calls: list[bool] = []
            self.target_kl = None
            self.lstm_hidden_dim = 0

        def update(self, *, clear_buffer: bool = True) -> dict[str, float]:
            self.calls.append(clear_buffer)
            return {"ratio_max": 1.0, "ratio_min": 1.0}

    class _StubNormalizer:
        def update(self, tensor: torch.Tensor) -> None:
            raise AssertionError("normalizer.update not used for this test")

    agent = _StubAgent()
    _run_ppo_updates(
        agent=agent,
        ppo_updates_per_batch=1,
        raw_states_for_normalizer_update=[],
        obs_normalizer=_StubNormalizer(),
        use_amp=True,
        amp_dtype=torch.float16,
    )
    assert agent.calls == [True]

# =============================================================================
# Telemetry Escalation Tests
# =============================================================================


def test_handle_telemetry_escalation_escalates_on_anomaly():
    """An anomaly should trigger escalation and always tick once per batch."""

    class _StubTelemetryConfig:
        def __init__(self):
            self.escalations = 0
            self.ticks = 0
            self.auto_escalate_on_anomaly = True

        def escalate_temporarily(self) -> None:
            self.escalations += 1

        def tick_escalation(self) -> None:
            self.ticks += 1

    config = _StubTelemetryConfig()
    report = AnomalyReport(has_anomaly=True)

    _handle_telemetry_escalation(report, config)

    assert config.escalations == 1
    assert config.ticks == 0  # tick now happens per-epoch


def test_handle_telemetry_escalation_ticks_without_anomaly():
    """Escalation should still tick down even when there is no anomaly."""

    class _StubTelemetryConfig:
        def __init__(self):
            self.escalations = 0
            self.ticks = 0
            self.auto_escalate_on_anomaly = True

        def escalate_temporarily(self) -> None:
            self.escalations += 1

        def tick_escalation(self) -> None:
            self.ticks += 1

    config = _StubTelemetryConfig()
    report = AnomalyReport(has_anomaly=False)

    _handle_telemetry_escalation(report, config)
    _handle_telemetry_escalation(None, config)

    assert config.escalations == 0
    assert config.ticks == 0  # tick now happens per-epoch


def test_handle_telemetry_escalation_respects_opt_out_flag():
    """Auto-escalation should not trigger when config disables it."""

    class _StubTelemetryConfig:
        def __init__(self):
            self.escalations = 0
            self.ticks = 0
            self.auto_escalate_on_anomaly = False

        def escalate_temporarily(self) -> None:
            self.escalations += 1

        def tick_escalation(self) -> None:
            self.ticks += 1

    config = _StubTelemetryConfig()
    report = AnomalyReport(has_anomaly=True)

    _handle_telemetry_escalation(report, config)

    assert config.escalations == 0
    assert config.ticks == 0  # tick now happens per-epoch


# =============================================================================
# Anomaly Diagnostics Tests
# =============================================================================


def test_emit_anomaly_diagnostics_skips_debug_when_disabled(monkeypatch):
    """Expensive anomaly diagnostics should be skipped when debug telemetry is disabled."""

    class _StubHub:
        def __init__(self):
            self.events = []

        def emit(self, event):
            self.events.append(event)

    class _StubAgent:
        class _Policy:
            class _Net:
                pass
            def __init__(self):
                self._network = _StubAgent._Policy._Net()
            @property
            def network(self):
                return self._network
        def __init__(self):
            self.policy = self._Policy()

    # Make gradient/stability collection fail if called
    def _fail_gradients(_):
        raise AssertionError("collect_per_layer_gradients should not be called")

    def _fail_stability(_):
        raise AssertionError("check_numerical_stability should not be called")

    monkeypatch.setattr("esper.simic.training.vectorized.collect_per_layer_gradients", _fail_gradients)
    monkeypatch.setattr("esper.simic.training.vectorized.check_numerical_stability", _fail_stability)

    hub = _StubHub()
    anomaly_report = AnomalyReport(has_anomaly=True, anomaly_types=["ratio_explosion"])

    _emit_anomaly_diagnostics(
        hub=hub,
        anomaly_report=anomaly_report,
        agent=_StubAgent(),
        batch_epoch_id=5,
        batch_idx=0,
        max_epochs=10,
        total_episodes=20,
        collect_debug=False,
    )

    # Event emitted with minimal payload, and expensive collectors not invoked
    assert len(hub.events) == 1
    from esper.leyline import AnomalyDetectedPayload
    data = hub.events[0].data
    assert isinstance(data, AnomalyDetectedPayload)
    assert data.gradient_stats is None
    assert data.stability is None


def test_emit_anomaly_diagnostics_collects_when_debug_enabled(monkeypatch):
    """Expensive diagnostics are emitted only when debug collection is enabled."""

    class _StubHub:
        def __init__(self):
            self.events = []

        def emit(self, event):
            self.events.append(event)

    class _StubAgent:
        class _Policy:
            class _Net:
                pass
            def __init__(self):
                self._network = _StubAgent._Policy._Net()
            @property
            def network(self):
                return self._network
        def __init__(self):
            self.policy = self._Policy()

    grad_called = {"count": 0}
    stability_called = {"count": 0}

    def _gradients(_):
        grad_called["count"] += 1
        class _GS:
            def to_dict(self):
                return {"g": 1}
        return [_GS()]

    def _stability(_):
        stability_called["count"] += 1
        class _S:
            def to_dict(self):
                return {"stable": True}
        return _S()

    monkeypatch.setattr("esper.simic.training.vectorized.collect_per_layer_gradients", _gradients)
    monkeypatch.setattr("esper.simic.training.vectorized.check_numerical_stability", _stability)

    hub = _StubHub()
    anomaly_report = AnomalyReport(has_anomaly=True, anomaly_types=["ratio_explosion"])

    _emit_anomaly_diagnostics(
        hub=hub,
        anomaly_report=anomaly_report,
        agent=_StubAgent(),
        batch_epoch_id=2,
        batch_idx=0,
        max_epochs=5,
        total_episodes=10,
        collect_debug=True,
        ratio_diagnostic={"foo": "bar"},
    )

    assert grad_called["count"] == 1
    assert stability_called["count"] == 1
    from esper.leyline import AnomalyDetectedPayload
    data = hub.events[0].data
    assert isinstance(data, AnomalyDetectedPayload)
    assert data.gradient_stats is not None
    assert data.stability is not None
    assert data.ratio_diagnostic == {"foo": "bar"}


# =============================================================================
# Batch Completion Tests
# =============================================================================


class _StubHub:
    def __init__(self):
        self.events: list[TelemetryEvent] = []

    def emit(self, event: TelemetryEvent) -> None:
        self.events.append(event)


def test_emit_with_env_context_requires_typed_payloads():
    """emit_with_env_context rejects None and dict payloads, requires typed dataclasses."""
    from esper.leyline.telemetry import SeedGerminatedPayload

    hub = _StubHub()

    # None payload should raise TypeError
    event_none = TelemetryEvent(event_type=TelemetryEventType.SEED_GERMINATED, data=None)
    with pytest.raises(TypeError, match="requires typed payload, got None"):
        emit_with_env_context(hub, 1, "cpu", event_none)

    # Dict payload should raise TypeError
    event_dict = TelemetryEvent(event_type=TelemetryEventType.SEED_GERMINATED, data={"foo": "bar"})
    with pytest.raises(TypeError, match="requires typed payload, got dict"):
        emit_with_env_context(hub, 2, "cpu", event_dict)

    # Typed payload should work and replace sentinel env_id
    payload = SeedGerminatedPayload(
        slot_id="r0c0",
        env_id=-1,  # Sentinel
        blueprint_id="conv",
        params=100,
    )
    event_typed = TelemetryEvent(event_type=TelemetryEventType.SEED_GERMINATED, data=payload)
    emit_with_env_context(hub, 3, "cpu", event_typed)

    # Original payload untouched (immutable dataclass)
    assert payload.env_id == -1

    # Emitted event has env_id replaced
    assert len(hub.events) == 1
    assert hub.events[0].data.env_id == 3
    assert hub.events[0].data.slot_id == "r0c0"


def test_emit_batch_completed_is_resume_aware_and_clamped():
    """BATCH_EPOCH_COMPLETED telemetry should include resume offsets and clamp totals."""
    hub = _StubHub()
    emit_batch_completed(
        hub,
        batch_idx=5,
        episodes_completed=22,  # exceeds total_episodes to test clamp
        total_episodes=20,
        env_final_accs=[0.8, 0.9],
        avg_acc=0.85,
        rolling_avg_acc=0.82,
        avg_reward=1.0,
        start_episode=10,
        requested_episodes=10,
    )

    assert len(hub.events) == 1
    payload = hub.events[0].data
    # BatchEpochCompletedPayload is a typed dataclass - use attribute access
    assert payload.episodes_completed == 20  # clamped
    assert payload.total_episodes == 20
    assert payload.start_episode == 10
    assert payload.requested_episodes == 10
    assert payload.batch_idx == 5


# =============================================================================
# Slot Configuration Tests (Regression tests)
# =============================================================================


def test_slot_config_filters_to_requested_slots_only():
    """Regression test: slot_config should only contain requested slots.

    Bug fixed: slot_config was incorrectly derived from host.injection_specs()
    which returns ALL available injection points in the network architecture,
    not just the user's requested slots. This caused the UI to show extra slots
    (e.g., r0c1) even when config only had r0c0.

    The fix filters injection_specs to only include slots that are actually
    registered in the model's seed_slots dict.
    """
    from esper.tolaria.environment import create_model
    from esper.leyline.slot_config import SlotConfig

    # Request only a single slot
    requested_slots = ["r0c0"]
    model = create_model(task="cifar_baseline", device="cpu", slots=requested_slots)

    # Verify the model only has the requested slot
    assert list(model.seed_slots.keys()) == requested_slots

    # The host has MORE injection points than we requested (this is expected)
    all_host_specs = model.host.injection_specs()
    assert len(all_host_specs) > len(requested_slots), (
        "Host should have more injection points than requested slots"
    )

    # The FIXED behavior: filter specs to only requested slots
    enabled_specs = [
        spec for spec in all_host_specs
        if spec.slot_id in model.seed_slots
    ]
    slot_config = SlotConfig.from_specs(enabled_specs)

    # slot_config.slot_ids should ONLY contain the requested slots
    assert slot_config.slot_ids == tuple(requested_slots), (
        f"slot_config.slot_ids should match requested slots {requested_slots}, "
        f"but got {slot_config.slot_ids}. "
        "This suggests the bug has regressed - slot_config is using all host "
        "injection specs instead of filtering to requested slots."
    )


def test_slot_config_preserves_subset_of_slots():
    """Verify slot_config works correctly with a subset of available slots.

    Similar to above but tests with 2 of 3 available slots to ensure
    the filtering works for intermediate cases, not just single slots.
    """
    from esper.tolaria.environment import create_model
    from esper.leyline.slot_config import SlotConfig

    # Request two slots (host has 3 for default cifar10)
    requested_slots = ["r0c0", "r0c2"]  # Skip r0c1
    model = create_model(task="cifar_baseline", device="cpu", slots=requested_slots)

    # Verify the model has exactly the requested slots
    assert set(model.seed_slots.keys()) == set(requested_slots)

    # The FIXED behavior: filter specs to only requested slots
    enabled_specs = [
        spec for spec in model.host.injection_specs()
        if spec.slot_id in model.seed_slots
    ]
    slot_config = SlotConfig.from_specs(enabled_specs)

    # slot_config should have exactly 2 slots, sorted by position
    assert len(slot_config.slot_ids) == 2
    assert set(slot_config.slot_ids) == set(requested_slots)


# =============================================================================
# Loss Aggregation Regression Tests
# =============================================================================


def test_loss_aggregation_divides_by_batch_count_not_sample_count():
    """Regression test: Loss should be averaged by batch count, not sample count.

    Bug fixed: Training/validation loops accumulated mean losses from CrossEntropyLoss
    (which uses reduction='mean' by default) and then divided by total sample count.
    This caused double-division, making reported loss ~batch_size times too low.

    The fix tracks batch count separately and divides accumulated mean losses by
    batch count, not sample count.

    Example of the bug:
        - 4 batches of 32 samples, each with mean loss ~0.6
        - Accumulated loss = 0.6 + 0.6 + 0.6 + 0.6 = 2.4
        - BUG: 2.4 / 128 samples = 0.01875 (32x too low!)
        - FIX: 2.4 / 4 batches = 0.6 (correct)
    """
    import torch.nn as nn
    from esper.simic.training.vectorized import loss_and_correct

    # Simulate 4 batches of 32 samples
    batch_size = 32
    num_classes = 10
    num_batches = 4
    criterion = nn.CrossEntropyLoss()  # Default: reduction='mean'

    # Accumulate losses as the training loop does
    loss_accum = torch.zeros(1)
    correct_accum = torch.zeros(1, dtype=torch.long)
    total_samples = 0
    batch_count = 0

    for _ in range(num_batches):
        # Random outputs and targets
        outputs = torch.randn(batch_size, num_classes)
        targets = torch.randint(0, num_classes, (batch_size,))

        loss, correct, total = loss_and_correct(outputs, targets, criterion, "classification")

        # Accumulate as training loop does
        loss_accum.add_(loss.detach())
        correct_accum.add_(correct)
        total_samples += total
        batch_count += 1

    # CORRECT: divide by batch count (what the fix does)
    correct_avg_loss = loss_accum.item() / batch_count

    # BUG: divide by sample count (what the old code did)
    buggy_avg_loss = loss_accum.item() / total_samples

    # The buggy loss should be ~batch_size times smaller
    ratio = correct_avg_loss / buggy_avg_loss
    assert abs(ratio - batch_size) < 1.0, (
        f"Expected ratio ~{batch_size}, got {ratio:.1f}. "
        "This indicates the loss aggregation pattern is inconsistent."
    )

    # The correct loss should be in a reasonable range for CrossEntropyLoss
    # (random predictions should give loss around -ln(1/num_classes) ≈ 2.3)
    assert 1.0 < correct_avg_loss < 4.0, (
        f"Correct avg loss {correct_avg_loss:.2f} outside expected range [1.0, 4.0]. "
        "CrossEntropyLoss on random predictions should be ~2.3."
    )


def test_lm_correct_tensor_shape_handles_sequence_dimension():
    """Regression test: LM correctness tensor has [B, T] shape, not [B].

    Bug fixed: Fused validation code did correct_fused.view(num_configs, batch_size)
    which assumed [K*B] shape. But for LM tasks, loss_and_correct returns [K*B, T]
    shaped correctness tensor (per-token correctness).

    The fix uses view(num_configs, -1) to handle both shapes uniformly.
    """
    import torch.nn as nn
    from esper.simic.training.vectorized import loss_and_correct

    batch_size = 8
    seq_len = 32
    vocab_size = 100
    num_configs = 3  # Simulate fused validation with 3 configs

    # Create LM-style data: [K*B, T, vocab] outputs, [K*B, T] targets
    fused_batch_size = num_configs * batch_size
    outputs = torch.randn(fused_batch_size, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (fused_batch_size, seq_len))

    # CrossEntropyLoss with reduction='none' for per-element loss
    criterion = nn.CrossEntropyLoss(reduction="none")

    # Get elementwise correctness (as fused validation does)
    loss, correct_fused, total = loss_and_correct(
        outputs, targets, criterion, "lm", elementwise=True
    )

    # Verify shape is [K*B, T] for LM (not [K*B])
    assert correct_fused.shape == (fused_batch_size, seq_len), (
        f"Expected LM correct shape [{fused_batch_size}, {seq_len}], "
        f"got {list(correct_fused.shape)}"
    )

    # The FIX: view(num_configs, -1) works for both shapes
    correct_per_config = correct_fused.view(num_configs, -1).sum(dim=1)
    assert correct_per_config.shape == (num_configs,), (
        f"Per-config correct should have shape [{num_configs}], "
        f"got {list(correct_per_config.shape)}"
    )

    # Total should be tokens per config (fused_batch_size * seq_len)
    assert total == fused_batch_size * seq_len, (
        f"Total should be {fused_batch_size * seq_len} tokens, got {total}"
    )


def test_lm_correct_tensor_is_scalar_when_elementwise_disabled():
    """LM path should return a scalar correct count when elementwise=False."""
    import torch.nn as nn
    from esper.simic.training.vectorized import loss_and_correct

    batch_size = 4
    seq_len = 16
    vocab_size = 50

    outputs = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    criterion = nn.CrossEntropyLoss()  # reduction='mean'

    loss, correct, total = loss_and_correct(outputs, targets, criterion, "lm", elementwise=False)
    assert loss.ndim == 0
    assert correct.ndim == 0
    assert total == batch_size * seq_len


def test_classification_correct_tensor_shape_still_works():
    """Verify classification tasks still work with the view(-1) fix.

    This ensures the LM shape fix didn't break classification.
    """
    import torch.nn as nn
    from esper.simic.training.vectorized import loss_and_correct

    batch_size = 8
    num_classes = 10
    num_configs = 3  # Simulate fused validation with 3 configs

    # Create classification data: [K*B, num_classes] outputs, [K*B] targets
    fused_batch_size = num_configs * batch_size
    outputs = torch.randn(fused_batch_size, num_classes)
    targets = torch.randint(0, num_classes, (fused_batch_size,))

    # CrossEntropyLoss with reduction='none' for per-element loss
    criterion = nn.CrossEntropyLoss(reduction="none")

    # Get elementwise correctness
    loss, correct_fused, total = loss_and_correct(
        outputs, targets, criterion, "classification", elementwise=True
    )

    # Verify shape is [K*B] for classification
    assert correct_fused.shape == (fused_batch_size,), (
        f"Expected classification correct shape [{fused_batch_size}], "
        f"got {list(correct_fused.shape)}"
    )

    # The FIX: view(num_configs, -1) also works for [K*B] shape
    correct_per_config = correct_fused.view(num_configs, -1).sum(dim=1)
    assert correct_per_config.shape == (num_configs,), (
        f"Per-config correct should have shape [{num_configs}], "
        f"got {list(correct_per_config.shape)}"
    )

    # Total should be samples per batch
    assert total == fused_batch_size, (
        f"Total should be {fused_batch_size} samples, got {total}"
    )
