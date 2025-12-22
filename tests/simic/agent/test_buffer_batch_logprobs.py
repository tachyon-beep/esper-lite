"""Test buffer can accept batched log_prob tensors."""
import pytest
import torch

from esper.leyline.factored_actions import (
    NUM_ALPHA_CURVES,
    NUM_ALPHA_SPEEDS,
    NUM_ALPHA_TARGETS,
    NUM_BLUEPRINTS,
    NUM_OPS,
    NUM_STYLES,
    NUM_TEMPO,
)
from esper.simic.agent.rollout_buffer import TamiyoRolloutBuffer


def test_buffer_add_accepts_tensor_log_probs():
    """Buffer.add() should accept tensor log_probs, not just floats."""
    buffer = TamiyoRolloutBuffer(
        num_envs=4,
        max_steps_per_env=25,
        state_dim=64,
        device=torch.device("cpu"),
    )

    state = torch.randn(64)
    slot_mask = torch.ones(3, dtype=torch.bool)
    blueprint_mask = torch.ones(NUM_BLUEPRINTS, dtype=torch.bool)
    style_mask = torch.ones(NUM_STYLES, dtype=torch.bool)
    tempo_mask = torch.ones(NUM_TEMPO, dtype=torch.bool)
    alpha_target_mask = torch.ones(NUM_ALPHA_TARGETS, dtype=torch.bool)
    alpha_speed_mask = torch.ones(NUM_ALPHA_SPEEDS, dtype=torch.bool)
    alpha_curve_mask = torch.ones(NUM_ALPHA_CURVES, dtype=torch.bool)
    op_mask = torch.ones(NUM_OPS, dtype=torch.bool)
    hidden_h = torch.randn(1, 1, 128)
    hidden_c = torch.randn(1, 1, 128)

    # Should accept tensors (0-dim) for log_probs
    buffer.add(
        env_id=0,
        state=state,
        slot_action=0,
        blueprint_action=1,
        style_action=0,
        tempo_action=0,
        alpha_target_action=0,
        alpha_speed_action=0,
        alpha_curve_action=0,
        op_action=1,
        slot_log_prob=torch.tensor(-0.5),      # tensor, not float
        blueprint_log_prob=torch.tensor(-1.0),
        style_log_prob=torch.tensor(-0.3),
        tempo_log_prob=torch.tensor(-0.4),
        alpha_target_log_prob=torch.tensor(-0.4),
        alpha_speed_log_prob=torch.tensor(-0.4),
        alpha_curve_log_prob=torch.tensor(-0.4),
        op_log_prob=torch.tensor(-0.7),
        value=0.5,
        reward=1.0,
        done=False,
        slot_mask=slot_mask,
        blueprint_mask=blueprint_mask,
        style_mask=style_mask,
        tempo_mask=tempo_mask,
        alpha_target_mask=alpha_target_mask,
        alpha_speed_mask=alpha_speed_mask,
        alpha_curve_mask=alpha_curve_mask,
        op_mask=op_mask,
        hidden_h=hidden_h,
        hidden_c=hidden_c,
    )

    # Verify stored correctly
    assert buffer.slot_log_probs[0, 0].item() == -0.5
    assert buffer.blueprint_log_probs[0, 0].item() == -1.0


def test_buffer_add_still_accepts_float_log_probs():
    """Buffer.add() should still accept float log_probs for backwards compat."""
    buffer = TamiyoRolloutBuffer(
        num_envs=4,
        max_steps_per_env=25,
        state_dim=64,
        device=torch.device("cpu"),
    )

    state = torch.randn(64)
    slot_mask = torch.ones(3, dtype=torch.bool)
    blueprint_mask = torch.ones(NUM_BLUEPRINTS, dtype=torch.bool)
    style_mask = torch.ones(NUM_STYLES, dtype=torch.bool)
    tempo_mask = torch.ones(NUM_TEMPO, dtype=torch.bool)
    alpha_target_mask = torch.ones(NUM_ALPHA_TARGETS, dtype=torch.bool)
    alpha_speed_mask = torch.ones(NUM_ALPHA_SPEEDS, dtype=torch.bool)
    alpha_curve_mask = torch.ones(NUM_ALPHA_CURVES, dtype=torch.bool)
    op_mask = torch.ones(NUM_OPS, dtype=torch.bool)
    hidden_h = torch.randn(1, 1, 128)
    hidden_c = torch.randn(1, 1, 128)

    # Should still accept floats
    buffer.add(
        env_id=0,
        state=state,
        slot_action=0,
        blueprint_action=1,
        style_action=0,
        tempo_action=0,
        alpha_target_action=0,
        alpha_speed_action=0,
        alpha_curve_action=0,
        op_action=1,
        slot_log_prob=-0.5,      # float
        blueprint_log_prob=-1.0,
        style_log_prob=-0.3,
        tempo_log_prob=-0.4,
        alpha_target_log_prob=-0.4,
        alpha_speed_log_prob=-0.4,
        alpha_curve_log_prob=-0.4,
        op_log_prob=-0.7,
        value=0.5,
        reward=1.0,
        done=False,
        slot_mask=slot_mask,
        blueprint_mask=blueprint_mask,
        style_mask=style_mask,
        tempo_mask=tempo_mask,
        alpha_target_mask=alpha_target_mask,
        alpha_speed_mask=alpha_speed_mask,
        alpha_curve_mask=alpha_curve_mask,
        op_mask=op_mask,
        hidden_h=hidden_h,
        hidden_c=hidden_c,
    )

    assert buffer.slot_log_probs[0, 0].item() == -0.5


def test_buffer_add_accepts_mixed_float_and_tensor():
    """Buffer.add() should accept mix of floats and tensors."""
    buffer = TamiyoRolloutBuffer(
        num_envs=4,
        max_steps_per_env=25,
        state_dim=64,
        device=torch.device("cpu"),
    )

    state = torch.randn(64)
    slot_mask = torch.ones(3, dtype=torch.bool)
    blueprint_mask = torch.ones(NUM_BLUEPRINTS, dtype=torch.bool)
    style_mask = torch.ones(NUM_STYLES, dtype=torch.bool)
    tempo_mask = torch.ones(NUM_TEMPO, dtype=torch.bool)
    alpha_target_mask = torch.ones(NUM_ALPHA_TARGETS, dtype=torch.bool)
    alpha_speed_mask = torch.ones(NUM_ALPHA_SPEEDS, dtype=torch.bool)
    alpha_curve_mask = torch.ones(NUM_ALPHA_CURVES, dtype=torch.bool)
    op_mask = torch.ones(NUM_OPS, dtype=torch.bool)
    hidden_h = torch.randn(1, 1, 128)
    hidden_c = torch.randn(1, 1, 128)

    # Mix of tensor and float inputs
    buffer.add(
        env_id=0,
        state=state,
        slot_action=0,
        blueprint_action=1,
        style_action=0,
        tempo_action=0,
        alpha_target_action=0,
        alpha_speed_action=0,
        alpha_curve_action=0,
        op_action=1,
        slot_log_prob=torch.tensor(-0.5),      # tensor
        blueprint_log_prob=-1.0,                # float
        style_log_prob=torch.tensor(-0.3),      # tensor
        tempo_log_prob=-0.4,                    # float
        alpha_target_log_prob=-0.4,
        alpha_speed_log_prob=-0.4,
        alpha_curve_log_prob=-0.4,
        op_log_prob=-0.7,                       # float
        value=torch.tensor(0.5),                # tensor
        reward=1.0,
        done=False,
        slot_mask=slot_mask,
        blueprint_mask=blueprint_mask,
        style_mask=style_mask,
        tempo_mask=tempo_mask,
        alpha_target_mask=alpha_target_mask,
        alpha_speed_mask=alpha_speed_mask,
        alpha_curve_mask=alpha_curve_mask,
        op_mask=op_mask,
        hidden_h=hidden_h,
        hidden_c=hidden_c,
        bootstrap_value=torch.tensor(0.2),      # tensor
    )

    assert buffer.slot_log_probs[0, 0].item() == -0.5
    assert buffer.blueprint_log_probs[0, 0].item() == -1.0
    assert buffer.values[0, 0].item() == pytest.approx(0.5)
    assert buffer.bootstrap_values[0, 0].item() == pytest.approx(0.2)
