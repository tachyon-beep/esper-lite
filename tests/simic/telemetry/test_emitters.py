"""Tests for simic telemetry emitters."""

import math
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from esper.karn.sanctum.schema import CounterfactualConfig, CounterfactualSnapshot
from esper.leyline import TelemetryEventType
from esper.simic.telemetry import TelemetryConfig, TelemetryLevel
from esper.simic.telemetry.emitters import (
    VectorizedEmitter,
    compute_grad_norm_surrogate,
    emit_ppo_update_event,
)
from esper.simic.training.vectorized_types import (
    ActionMaskFlags,
    ActionOutcome,
    ActionSpec,
)


def _build_action_spec(slot_id: str, indices: dict[str, int]) -> ActionSpec:
    return ActionSpec(
        slot_idx=indices["slot"],
        blueprint_idx=indices["blueprint"],
        style_idx=indices["style"],
        tempo_idx=indices["tempo"],
        alpha_target_idx=indices["alpha_target"],
        alpha_speed_idx=indices["alpha_speed"],
        alpha_curve_idx=indices["alpha_curve"],
        op_idx=indices["op"],
        target_slot=slot_id,
        slot_is_enabled=True,
    )


def _build_mask_flags(masked: dict[str, bool] | None = None) -> ActionMaskFlags:
    flags = {
        "op": False,
        "slot": False,
        "blueprint": False,
        "style": False,
        "tempo": False,
        "alpha_target": False,
        "alpha_speed": False,
        "alpha_curve": False,
    }
    if masked is not None:
        for key, value in masked.items():
            flags[key] = value
    return ActionMaskFlags(
        op_masked=flags["op"],
        slot_masked=flags["slot"],
        blueprint_masked=flags["blueprint"],
        style_masked=flags["style"],
        tempo_masked=flags["tempo"],
        alpha_target_masked=flags["alpha_target"],
        alpha_speed_masked=flags["alpha_speed"],
        alpha_curve_masked=flags["alpha_curve"],
    )


def _make_mandatory_metrics(**overrides) -> dict:
    """Create metrics dict with all mandatory fields for emit_ppo_update_event.

    This ensures tests fail loudly if required fields are missing.
    """
    base = {
        "policy_loss": 0.1,
        "value_loss": 0.2,
        "entropy": 1.5,
        "approx_kl": 0.01,
        "clip_fraction": 0.1,
        "pre_clip_grad_norm": 2.5,
        "ppo_updates_count": 1,
        # Advantage statistics (mandatory)
        "advantage_mean": 0.0,
        "advantage_std": 1.0,
        "advantage_skewness": 0.0,
        "advantage_kurtosis": 0.0,
        "advantage_positive_ratio": 0.5,
        # Ratio statistics (mandatory)
        "ratio_mean": 1.0,
        "ratio_min": 0.8,
        "ratio_max": 1.2,
        "ratio_std": 0.1,
        # Log prob extremes (mandatory)
        "log_prob_min": -5.0,
        "log_prob_max": -0.1,
        # Value statistics (mandatory)
        "value_mean": 0.0,
        "value_std": 1.0,
        "value_min": -2.0,
        "value_max": 2.0,
        # Pre-normalization advantage statistics (mandatory)
        "pre_norm_advantage_mean": 0.5,
        "pre_norm_advantage_std": 1.2,
        # Return statistics (mandatory)
        "return_mean": 0.3,
        "return_std": 0.8,
        # Value target scale (mandatory) - std used to normalize returns
        "value_target_scale": 0.8,
    }
    base.update(overrides)
    return base


class _RecordingHub:
    def __init__(self) -> None:
        self.events = []

    def emit(self, event) -> None:
        self.events.append(event)


class _StubAgent:
    optimizer = None

    def get_entropy_coef(self) -> float:
        return 0.05


class _StubEnvState:
    def __init__(self) -> None:
        self.seeds_created = 0
        self.seeds_fossilized = 0
        self.action_counts = {"WAIT": 1}
        self.successful_action_counts = {"WAIT": 1}


def test_emit_ppo_update_event_propagates_group_id():
    """emit_ppo_update_event should propagate group_id to TelemetryEvent."""
    hub = MagicMock()

    emit_ppo_update_event(
        hub=hub,
        metrics=_make_mandatory_metrics(),
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
        metrics=_make_mandatory_metrics(
            pre_clip_grad_norm=3.2,
            ppo_updates_count=2,
            # Override value function statistics for verification
            value_mean=5.5,
            value_std=1.2,
            value_min=2.1,
            value_max=9.8,
        ),
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


def test_batch_tail_event_order_is_stable() -> None:
    hub = _RecordingHub()
    telemetry_config = TelemetryConfig(level=TelemetryLevel.NORMAL)
    emitter = VectorizedEmitter(
        env_id=0,
        device="cpu",
        hub=hub,
        telemetry_config=telemetry_config,
    )
    agent = _StubAgent()

    emitter.on_ppo_update(
        metrics=_make_mandatory_metrics(),
        episodes_completed=5,
        batch_idx=0,
        epoch=5,
        agent=agent,
        ppo_grad_norm=1.0,
        ppo_update_time_ms=10.0,
        avg_acc=80.0,
        avg_reward=1.0,
        rolling_avg_acc=80.0,
    )

    emitter.on_batch_completed(
        batch_idx=0,
        episodes_completed=5,
        rolling_avg_acc=80.0,
        avg_acc=80.0,
        metrics={"entropy": 0.0, "approx_kl": 0.0, "explained_variance": 0.0},
        env_states=[_StubEnvState()],
        update_skipped=False,
        plateau_threshold=0.5,
        improvement_threshold=0.5,
        prev_rolling_avg_acc=None,
        total_episodes=5,
        start_episode=0,
        n_episodes=5,
        env_final_accs=[80.0],
        avg_reward=1.0,
        train_losses=[0.0],
        train_corrects=[1],
        train_totals=[1],
        val_losses=[0.0],
        val_corrects=[1],
        val_totals=[1],
        num_train_batches=1,
        num_test_batches=1,
        analytics=None,
        epoch=5,
    )

    event_types = [event.event_type for event in hub.events]
    assert event_types == [
        TelemetryEventType.PPO_UPDATE_COMPLETED,
        TelemetryEventType.ANALYTICS_SNAPSHOT,
        TelemetryEventType.BATCH_EPOCH_COMPLETED,
        TelemetryEventType.ANALYTICS_SNAPSHOT,
    ]
    assert hub.events[1].data.kind == "batch_stats"
    assert hub.events[3].data.kind == "action_distribution"


def test_emit_ppo_update_event_includes_lstm_health():
    """emit_ppo_update_event should include LSTM health metrics in PPOUpdatePayload (B7-DRL-04)."""
    hub = MagicMock()

    emit_ppo_update_event(
        hub=hub,
        metrics=_make_mandatory_metrics(
            # LSTM health metrics (from compute_lstm_health)
            lstm_h_l2_total=120.0,
            lstm_c_l2_total=240.0,
            lstm_h_rms=0.35,
            lstm_c_rms=1.44,
            lstm_h_env_rms_mean=0.34,
            lstm_h_env_rms_max=0.50,
            lstm_c_env_rms_mean=1.40,
            lstm_c_env_rms_max=2.10,
            lstm_h_max=2.1,
            lstm_c_max=1.9,
            lstm_has_nan=False,
            lstm_has_inf=False,
        ),
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

    # Verify LSTM health metrics were populated
    assert payload.lstm_h_l2_total == 120.0
    assert payload.lstm_c_l2_total == 240.0
    assert payload.lstm_h_rms == 0.35
    assert payload.lstm_c_rms == 1.44
    assert payload.lstm_h_env_rms_mean == 0.34
    assert payload.lstm_h_env_rms_max == 0.50
    assert payload.lstm_c_env_rms_mean == 1.40
    assert payload.lstm_c_env_rms_max == 2.10
    assert payload.lstm_h_max == 2.1
    assert payload.lstm_c_max == 1.9
    assert payload.lstm_has_nan is False
    assert payload.lstm_has_inf is False


def test_emit_ppo_update_event_lstm_health_defaults_to_none():
    """emit_ppo_update_event should use None for LSTM health when not provided (non-LSTM policy)."""
    hub = MagicMock()

    # No LSTM health metrics in the metrics dict (non-recurrent policy)
    emit_ppo_update_event(
        hub=hub,
        metrics=_make_mandatory_metrics(),
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

    # Verify LSTM health defaults to None (no LSTM)
    assert payload.lstm_h_l2_total is None
    assert payload.lstm_c_l2_total is None
    assert payload.lstm_h_rms is None
    assert payload.lstm_c_rms is None
    assert payload.lstm_h_env_rms_mean is None
    assert payload.lstm_h_env_rms_max is None
    assert payload.lstm_c_env_rms_mean is None
    assert payload.lstm_c_env_rms_max is None
    assert payload.lstm_h_max is None
    assert payload.lstm_c_max is None
    # Boolean flags default to False
    assert payload.lstm_has_nan is False
    assert payload.lstm_has_inf is False


class TestComputeGradNormSurrogate:
    """Tests for compute_grad_norm_surrogate numerical stability."""

    def test_raises_on_no_gradients(self):
        """Should raise AssertionError when module has no gradients.

        Having no gradients after backward() indicates a bug (e.g., torch.compile wrapper issue).
        The function fails loudly instead of returning a silent default.
        """
        model = nn.Linear(10, 5)
        # No backward pass, so no gradients - this is a bug condition
        with pytest.raises(AssertionError, match="No gradients found"):
            compute_grad_norm_surrogate(model)

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
        action_spec = _build_action_spec(
            "G0",
            {
                "op": 0,
                "slot": 0,
                "blueprint": 0,
                "style": 0,
                "tempo": 0,
                "alpha_target": 0,
                "alpha_speed": 0,
                "alpha_curve": 0,
            },
        )
        emitter.on_last_action(
            epoch=10,
            action_spec=action_spec,
            masked=_build_mask_flags(),
            outcome=ActionOutcome(
                action_success=True,
                reward_raw=0.35,
                reward_components=rc,
            ),
            active_alpha_algorithm="curiosity",
            value_estimate=0.3,
            host_accuracy=0.75,
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
        action_spec = _build_action_spec(
            "G0",
            {
                "op": 0,
                "slot": 0,
                "blueprint": 0,
                "style": 0,
                "tempo": 0,
                "alpha_target": 0,
                "alpha_speed": 0,
                "alpha_curve": 0,
            },
        )
        emitter.on_last_action(
            epoch=10,
            action_spec=action_spec,
            masked=_build_mask_flags(),
            outcome=ActionOutcome(
                action_success=True,
                reward_raw=0.0,
                reward_components=None,
            ),
            active_alpha_algorithm=None,
            value_estimate=0.0,
            host_accuracy=0.5,
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

        action_spec = _build_action_spec(
            "r0c0",
            {
                "op": 0,
                "slot": 0,
                "blueprint": 0,
                "style": 0,
                "tempo": 0,
                "alpha_target": 0,
                "alpha_speed": 0,
                "alpha_curve": 0,
            },
        )
        emitter.on_last_action(
            epoch=1,
            action_spec=action_spec,
            masked=_build_mask_flags(),
            outcome=ActionOutcome(action_success=True),
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


class TestCounterfactualMatrixEmission:
    """Tests for VectorizedEmitter.on_counterfactual_matrix."""

    def test_prefers_solo_on_accuracies(self):
        """Solo-on accuracies should drive individual contributions (no false interference)."""
        hub = MagicMock()
        hub.events = []
        hub.emit.side_effect = hub.events.append

        emitter = VectorizedEmitter(env_id=0, device="cpu", hub=hub)

        active_slots = ["r0c0", "r0c1", "r0c2"]
        # Accuracy when the target slot is disabled (two seeds remain active)
        baseline_accs = {"r0c0": 55.0, "r0c1": 35.0, "r0c2": 47.0}
        # Measured accuracy when only this slot is active
        solo_accs = {"r0c0": 33.0, "r0c1": 45.0, "r0c2": 42.0}
        pair_accs = {(0, 1): 47.0, (0, 2): 35.0, (1, 2): 55.0}

        emitter.on_counterfactual_matrix(
            active_slots=active_slots,
            baseline_accs=baseline_accs,
            val_acc=60.0,
            all_disabled_acc=30.0,
            pair_accs=pair_accs,
            solo_accs=solo_accs,
        )

        event = hub.events[-1]
        assert event.event_type == TelemetryEventType.COUNTERFACTUAL_MATRIX_COMPUTED
        payload = event.data

        snapshot = CounterfactualSnapshot(
            slot_ids=payload.slot_ids,
            configs=[
                CounterfactualConfig(
                    seed_mask=tuple(cfg["seed_mask"]),
                    accuracy=cfg["accuracy"],
                )
                for cfg in payload.configs
            ],
            strategy=payload.strategy,
            compute_time_ms=payload.compute_time_ms,
        )

        # Individual contributions should use measured solo-on accuracies
        assert snapshot.baseline_accuracy == pytest.approx(30.0)
        individuals = snapshot.individual_contributions()
        assert individuals["r0c0"] == pytest.approx(solo_accs["r0c0"] - 30.0)
        assert individuals["r0c1"] == pytest.approx(solo_accs["r0c1"] - 30.0)
        assert individuals["r0c2"] == pytest.approx(solo_accs["r0c2"] - 30.0)
        # Total synergy should reflect solo-on data (no spurious interference)
        assert snapshot.total_synergy() == pytest.approx(0.0)
