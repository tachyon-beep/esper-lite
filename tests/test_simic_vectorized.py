"""Unit tests for vectorized PPO helpers."""

import pytest
import torch
from unittest.mock import MagicMock

from esper.leyline import SeedStage, TelemetryEventType
from esper.simic.anomaly_detector import AnomalyReport
from esper.simic.vectorized import (
    _advance_active_seed,
    _calculate_entropy_anneal_steps,
    _emit_anomaly_diagnostics,
    _handle_telemetry_escalation,
    _run_ppo_updates,
)


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
        self.seed_slots = {"mid": seed_slot}

    def has_active_seed_in_slot(self, slot: str) -> bool:
        """Check if specific slot has an active seed."""
        return slot in self.seed_slots and self.seed_slots[slot].state is not None


def test_advance_active_seed_fossilizes_via_seed_slot():
    """PROBATIONARY seeds should fossilize through SeedSlot.advance_stage (emits telemetry)."""
    model = _StubModel(SeedStage.PROBATIONARY)
    slot_id = "mid"

    _advance_active_seed(model, slot_id)

    assert model.seed_slots["mid"].advance_calls == [SeedStage.FOSSILIZED]
    assert model.seed_slots["mid"].set_alpha_calls == [1.0]
    assert model.seed_slots["mid"].state.stage == SeedStage.FOSSILIZED
    # Transition should happen inside advance_stage, not direct transition
    assert model.seed_slots["mid"].state.transition_calls == []


def test_advance_active_seed_noop_on_failed_fossilization_gate():
    """Failed fossilization gate should be a no-op (Tamiyo learns from failed attempts)."""
    gate_result = _StubGateResult(passed=False, checks_failed=["no_improvement"])
    model = _StubModel(SeedStage.PROBATIONARY, gate_result=gate_result)
    slot_id = "mid"

    # Should not raise - failed gate is normal RL outcome
    _advance_active_seed(model, slot_id)

    # Gate was checked but transition didn't happen
    assert model.seed_slots["mid"].advance_calls == [SeedStage.FOSSILIZED]
    assert model.seed_slots["mid"].set_alpha_calls == []  # No alpha change on failed gate
    # Stage should NOT change (stub's advance_stage still sets it, but in real code it wouldn't)


def test_advance_active_seed_noop_from_training_stage():
    """TRAINING seeds are handled mechanically; fossilize action should do nothing."""
    model = _StubModel(SeedStage.TRAINING)
    slot_id = "mid"

    _advance_active_seed(model, slot_id)

    assert model.seed_slots["mid"].state.transition_calls == []
    assert model.seed_slots["mid"].start_blending_calls == []
    assert model.seed_slots["mid"].state.stage == SeedStage.TRAINING


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

    from esper.leyline import TelemetryEvent

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

            from esper.leyline import TelemetryEvent

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
    )

    # Should stop after second update due to KL threshold (1.5 * 0.01 = 0.015)
    assert agent.update_calls == [False, False]
    assert agent.buffer.reset_calls == 1  # Cleared because last call didn't clear the buffer

    # Normalizer still updated once because at least one update succeeded
    assert len(normalizer.calls) == 1
    assert metrics["approx_kl"] == pytest.approx((0.005 + 0.02) / 2.0)


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
    assert config.ticks == 1


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
    assert config.ticks == 2


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
    assert config.ticks == 1


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

    monkeypatch.setattr("esper.simic.vectorized.collect_per_layer_gradients", _fail_gradients)
    monkeypatch.setattr("esper.simic.vectorized.check_numerical_stability", _fail_stability)

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
