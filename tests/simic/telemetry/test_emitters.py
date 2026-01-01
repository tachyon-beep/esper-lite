"""Tests for simic telemetry emitters."""

import math
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from esper.leyline import TelemetryEventType
from esper.simic.telemetry.emitters import (
    VectorizedEmitter,
    compute_grad_norm_surrogate,
    emit_ppo_update_event,
)


def test_emit_ppo_update_event_propagates_group_id():
    """emit_ppo_update_event should propagate group_id to TelemetryEvent."""
    hub = MagicMock()

    emit_ppo_update_event(
        hub=hub,
        metrics={
            "policy_loss": 0.1,
            "value_loss": 0.2,
            "entropy": 1.5,
            "pre_clip_grad_norm": 2.5,
            "ppo_updates_count": 1,
        },
        episodes_completed=10,
        batch_idx=5,
        epoch=100,
        optimizer=None,
        grad_norm=1.0,
        update_time_ms=50.0,
        group_id="B",  # New parameter
    )

    # Verify hub.emit was called
    hub.emit.assert_called_once()
    event = hub.emit.call_args[0][0]

    # Verify group_id was propagated
    assert event.group_id == "B"


def test_emit_ppo_update_event_includes_value_stats():
    """emit_ppo_update_event should include value_mean/std/min/max in PPOUpdatePayload."""
    hub = MagicMock()

    emit_ppo_update_event(
        hub=hub,
        metrics={
            "policy_loss": 0.1,
            "value_loss": 0.2,
            "entropy": 1.5,
            "pre_clip_grad_norm": 3.2,
            "ppo_updates_count": 2,
            # Value function statistics
            "value_mean": 5.5,
            "value_std": 1.2,
            "value_min": 2.1,
            "value_max": 9.8,
        },
        episodes_completed=10,
        batch_idx=5,
        epoch=100,
        optimizer=None,
        grad_norm=1.0,
        update_time_ms=50.0,
    )

    hub.emit.assert_called_once()
    event = hub.emit.call_args[0][0]
    payload = event.data

    # Verify value stats were populated (not zeros)
    assert payload.value_mean == 5.5
    assert payload.value_std == 1.2
    assert payload.value_min == 2.1
    assert payload.value_max == 9.8


class TestComputeGradNormSurrogate:
    """Tests for compute_grad_norm_surrogate numerical stability."""

    def test_returns_none_for_no_gradients(self):
        """Should return None when module has no gradients."""
        model = nn.Linear(10, 5)
        # No backward pass, so no gradients
        result = compute_grad_norm_surrogate(model)
        assert result is None

    def test_computes_correct_norm_for_simple_case(self):
        """Should compute correct L2 norm for known gradients."""
        model = nn.Linear(2, 1, bias=False)
        # Manually set gradient to known values: [[3, 4]] -> norm = 5
        model.weight.grad = torch.tensor([[3.0, 4.0]])

        result = compute_grad_norm_surrogate(model)
        assert result is not None
        assert math.isclose(result, 5.0, rel_tol=1e-5)

    def test_no_overflow_with_large_gradients(self):
        """B7-PT-01: Should not overflow to inf with very large gradients.

        The old implementation used g*g which overflows for |g| > ~1.84e19.
        The new implementation using torch._foreach_norm handles this correctly.
        """
        model = nn.Linear(10, 5)

        # Set gradients to 1e20 - would cause overflow with naive g*g
        for param in model.parameters():
            param.grad = torch.full_like(param, 1e20)

        result = compute_grad_norm_surrogate(model)

        assert result is not None
        assert not math.isinf(result), "Gradient norm overflowed to inf"
        assert not math.isnan(result), "Gradient norm became NaN"
        assert result > 0, "Gradient norm should be positive"

    def test_handles_mixed_gradient_scales(self):
        """Should handle parameters with vastly different gradient scales."""
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.Linear(5, 2),
        )

        # First layer: tiny gradients
        model[0].weight.grad = torch.full_like(model[0].weight, 1e-10)
        model[0].bias.grad = torch.full_like(model[0].bias, 1e-10)

        # Second layer: huge gradients
        model[1].weight.grad = torch.full_like(model[1].weight, 1e15)
        model[1].bias.grad = torch.full_like(model[1].bias, 1e15)

        result = compute_grad_norm_surrogate(model)

        assert result is not None
        assert not math.isinf(result)
        assert not math.isnan(result)
        # Result should be dominated by the large gradients
        assert result > 1e14


class TestVectorizedEmitterRewardComponents:
    """Tests for VectorizedEmitter.on_last_action reward_components parameter."""

    @pytest.fixture
    def mock_hub(self):
        """Create a mock hub that captures emitted events."""
        hub = MagicMock()
        hub.events = []
        def capture_event(event):
            hub.events.append(event)
        hub.emit.side_effect = capture_event
        return hub

    def test_on_last_action_accepts_reward_components_dataclass(self, mock_hub):
        """on_last_action should accept RewardComponentsTelemetry directly."""
        from esper.simic.rewards.reward_telemetry import RewardComponentsTelemetry

        emitter = VectorizedEmitter(env_id=0, device="cpu", hub=mock_hub)

        rc = RewardComponentsTelemetry(
            bounded_attribution=0.5,
            compute_rent=-0.1,
            seed_stage=2,
            total_reward=0.35,
        )

        # Should not raise
        emitter.on_last_action(
            epoch=10,
            action_indices={"op": 0, "slot": 0, "blueprint": 0, "style": 0, "tempo": 0, "alpha_target": 0, "alpha_speed": 0, "alpha_curve": 0},
            slot_id="G0",
            masked={},
            success=True,
            active_alpha_algorithm="curiosity",
            total_reward=0.35,
            value_estimate=0.3,
            host_accuracy=0.75,
            reward_components=rc,
        )

        # Verify event was emitted with typed dataclass
        events = [e for e in mock_hub.events if e.event_type == TelemetryEventType.ANALYTICS_SNAPSHOT]
        assert len(events) >= 1
        payload = events[-1].data
        assert payload.reward_components is rc
        assert payload.reward_components.seed_stage == 2

    def test_on_last_action_accepts_none_reward_components(self, mock_hub):
        """on_last_action should accept None for reward_components (LOSS family)."""
        emitter = VectorizedEmitter(env_id=0, device="cpu", hub=mock_hub)

        # Should not raise when reward_components is None
        emitter.on_last_action(
            epoch=10,
            action_indices={"op": 0, "slot": 0, "blueprint": 0, "style": 0, "tempo": 0, "alpha_target": 0, "alpha_speed": 0, "alpha_curve": 0},
            slot_id="G0",
            masked={},
            success=True,
            active_alpha_algorithm=None,
            total_reward=0.0,
            value_estimate=0.0,
            host_accuracy=0.5,
            reward_components=None,
        )

        # Verify event was emitted
        events = [e for e in mock_hub.events if e.event_type == TelemetryEventType.ANALYTICS_SNAPSHOT]
        assert len(events) >= 1
        payload = events[-1].data
        assert payload.reward_components is None

    def test_on_last_action_accepts_head_telemetry(self, mock_hub):
        """on_last_action should accept and forward HeadTelemetry."""
        from esper.leyline.telemetry import HeadTelemetry

        emitter = VectorizedEmitter(env_id=0, device="cpu", hub=mock_hub)

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

        emitter.on_last_action(
            epoch=1,
            action_indices={"op": 0, "slot": 0, "blueprint": 0, "style": 0, "tempo": 0,
                           "alpha_target": 0, "alpha_speed": 0, "alpha_curve": 0},
            slot_id="r0c0",
            masked={},
            success=True,
            active_alpha_algorithm=None,
            head_telemetry=head_telem,
        )

        # Check the emitted event contains head_telemetry
        events = [e for e in mock_hub.events if e.event_type == TelemetryEventType.ANALYTICS_SNAPSHOT]
        assert len(events) == 1
        payload = events[0].data

        assert payload.head_telemetry is not None
        assert payload.head_telemetry.op_confidence == 0.85
        assert payload.head_telemetry.op_entropy == 0.3
        assert payload.head_telemetry.curve_confidence == 0.82
