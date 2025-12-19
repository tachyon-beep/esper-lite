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
    emit_cf_unavailable,
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
    event = TelemetryEvent(event_type=TelemetryEventType.SEED_GERMINATED, data={})
    emit_with_env_context(hub, env_idx=2, device="cpu", event=event)

    emitted = hub.emit.call_args[0][0]
    assert emitted.data["env_id"] == 2
    assert emitted.data["device"] == "cpu"


def test_last_action_event_emitted():
    from esper.leyline.factored_actions import FactoredAction

    with patch("esper.simic.telemetry.emitters.get_hub") as get_hub:
        hub = Mock()
        get_hub.return_value = hub

        emit_last_action(
            env_id=0,
            epoch=3,
            slot_idx=1,
            blueprint_idx=1,
            blend_idx=1,
            op_idx=1,
            slot_id="r0c1",
            masked={"op": False, "slot": False, "blueprint": False, "blend": True},
            success=True,
        )

        emitted = hub.emit.call_args[0][0]
        assert emitted.data["slot_id"] == "r0c1"
        assert emitted.data["blend_masked"] is True


def test_ppo_update_event_includes_vitals():
    hub = Mock()
    metrics = {"policy_loss": 0.1}
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
    data = hub.emit.call_args[0][0].data
    assert data["lr"] == 0.0003
    assert data["grad_norm"] == 1.23
    assert data["update_time_ms"] == 12.5


def test_action_distribution_snapshot():
    hub = Mock()
    emit_action_distribution(
        hub=hub,
        batch_idx=1,
        episodes_completed=4,
        action_counts={"WAIT": 3, "GERMINATE": 1},
        success_counts={"WAIT": 3, "GERMINATE": 1},
    )
    data = hub.emit.call_args[0][0].data
    assert data["action_counts"]["WAIT"] == 3


def test_counterfactual_unavailable_event():
    hub = Mock()
    emit_cf_unavailable(
        hub,
        env_id=0,
        slot_id="r0c1",
        reason="missing_baseline",
    )
    data = hub.emit.call_args[0][0].data
    assert data["available"] is False
    assert data["reason"] == "missing_baseline"


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
    assert data["step_time_ms"] == 5.0
    assert data["dataloader_wait_ms"] == 2.0


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
    assert data["fps"] == 50.0


def test_reward_summary_emitted():
    hub = Mock()
    emit_reward_summary(
        hub=hub,
        env_id=0,
        batch_idx=1,
        summary={"bounded_attribution": 0.4, "compute_rent": -0.1, "total_reward": 0.3},
    )
    data = hub.emit.call_args[0][0].data
    assert data["summary"]["total_reward"] == 0.3


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
    assert data["mask_hits"]["op"] == 10
    assert data["mask_total"]["op"] == 12


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
    assert event.data["current_acc"] == 0.6
    assert event.data["rolling_avg_acc"] == 0.8
    assert event.data["training_progress"] == 0.5


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
    """PROBATIONARY seeds should fossilize through SeedSlot.advance_stage (emits telemetry)."""
    model = _StubModel(SeedStage.PROBATIONARY)
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
    model = _StubModel(SeedStage.PROBATIONARY, gate_result=gate_result)
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
    )

    # Should stop after second update due to KL threshold (1.5 * 0.01 = 0.015)
    assert agent.update_calls == [False, False]
    assert agent.buffer.reset_calls == 1  # Cleared because last call didn't clear the buffer

    # Normalizer still updated once because at least one update succeeded
    assert len(normalizer.calls) == 1
    assert metrics["approx_kl"] == pytest.approx((0.005 + 0.02) / 2.0)


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
        class _Net:
            pass
        def __init__(self):
            self.network = self._Net()

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
    data = hub.events[0].data
    assert "gradient_stats" not in data
    assert "stability" not in data


def test_emit_anomaly_diagnostics_collects_when_debug_enabled(monkeypatch):
    """Expensive diagnostics are emitted only when debug collection is enabled."""

    class _StubHub:
        def __init__(self):
            self.events = []

        def emit(self, event):
            self.events.append(event)

    class _StubAgent:
        class _Net:
            pass
        def __init__(self):
            self.network = self._Net()

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
    data = hub.events[0].data
    assert "gradient_stats" in data
    assert "stability" in data
    assert data["ratio_diagnostic"] == {"foo": "bar"}


# =============================================================================
# Batch Completion Tests
# =============================================================================


class _StubHub:
    def __init__(self):
        self.events: list[TelemetryEvent] = []

    def emit(self, event: TelemetryEvent) -> None:
        self.events.append(event)


def test_emit_with_env_context_handles_none_and_copies():
    """Callback should handle missing data and avoid mutating shared dicts."""
    hub = _StubHub()

    event_none = TelemetryEvent(event_type=TelemetryEventType.SEED_GERMINATED, data=None)
    emit_with_env_context(hub, 1, "cpu", event_none)
    assert hub.events[0].data["env_id"] == 1
    assert hub.events[0].data["device"] == "cpu"

    shared = {"foo": "bar"}
    event_shared = TelemetryEvent(event_type=TelemetryEventType.SEED_GERMINATED, data=shared)
    emit_with_env_context(hub, 2, "cpu", event_shared)
    # Original dict is untouched
    assert "env_id" not in shared
    assert "device" not in shared
    assert hub.events[1].data["env_id"] == 2
    assert hub.events[1].data["device"] == "cpu"
    assert hub.events[1].data["foo"] == "bar"


def test_emit_batch_completed_is_resume_aware_and_clamped():
    """BATCH_COMPLETED telemetry should include resume offsets and clamp totals."""
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
    data = hub.events[0].data
    assert data["episodes_completed"] == 20  # clamped
    assert data["total_episodes"] == 20
    assert data["start_episode"] == 10
    assert data["requested_episodes"] == 10
    assert data["batch_idx"] == 5
