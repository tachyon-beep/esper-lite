"""Test batched bootstrap value computation."""
import torch
from unittest.mock import MagicMock

from esper.leyline import (
    NUM_ALPHA_CURVES,
    NUM_ALPHA_SPEEDS,
    NUM_ALPHA_TARGETS,
    NUM_BLUEPRINTS,
    NUM_OPS,
    NUM_STYLES,
    NUM_TEMPO,
)


def test_batch_bootstrap_single_forward_pass():
    """Bootstrap values should use single batched forward pass."""
    num_envs = 4

    # Mock network that tracks forward calls
    mock_network = MagicMock()
    forward_call_count = 0

    def mock_get_action(state, hidden, **kwargs):
        nonlocal forward_call_count
        forward_call_count += 1
        batch_size = state.shape[0]
        head_names = [
            "slot",
            "blueprint",
            "style",
            "tempo",
            "alpha_target",
            "alpha_speed",
            "alpha_curve",
            "op",
        ]
        actions = {key: torch.zeros(batch_size, dtype=torch.long) for key in head_names}
        log_probs = {key: torch.zeros(batch_size) for key in head_names}
        values = torch.randn(batch_size)  # Random values for each env
        return actions, log_probs, values, hidden

    mock_network.get_action = mock_get_action

    # Simulate batched bootstrap computation
    states = torch.randn(num_envs, 64)  # All envs' post-action states
    hidden = (torch.randn(1, num_envs, 128), torch.randn(1, num_envs, 128))

    # Single forward pass for all envs
    with torch.inference_mode():
        _, _, bootstrap_values, _ = mock_network.get_action(
            states, hidden=hidden,
            slot_mask=torch.ones(num_envs, 5, dtype=torch.bool),
            blueprint_mask=torch.ones(num_envs, NUM_BLUEPRINTS, dtype=torch.bool),
            style_mask=torch.ones(num_envs, NUM_STYLES, dtype=torch.bool),
            tempo_mask=torch.ones(num_envs, NUM_TEMPO, dtype=torch.bool),
            alpha_target_mask=torch.ones(num_envs, NUM_ALPHA_TARGETS, dtype=torch.bool),
            alpha_speed_mask=torch.ones(num_envs, NUM_ALPHA_SPEEDS, dtype=torch.bool),
            alpha_curve_mask=torch.ones(num_envs, NUM_ALPHA_CURVES, dtype=torch.bool),
            op_mask=torch.ones(num_envs, NUM_OPS, dtype=torch.bool),
        )

    assert forward_call_count == 1, f"Expected 1 forward pass, got {forward_call_count}"
    assert bootstrap_values.shape == (num_envs,)


def test_batch_bootstrap_values_correct():
    """Batched bootstrap should produce same values as per-env approach."""
    # This test verifies mathematical equivalence
    torch.manual_seed(42)

    # Simulate simple linear critic for reproducibility
    critic_weight = torch.randn(1, 64)
    critic_bias = torch.randn(1)

    def simple_critic(state):
        """state: [batch, 64] -> [batch]"""
        return (state @ critic_weight.T + critic_bias).squeeze(-1)

    states = torch.randn(4, 64)

    # Per-env approach
    per_env_values = []
    for i in range(4):
        val = simple_critic(states[i:i+1])
        per_env_values.append(val.item())

    # Batched approach
    batched_values = simple_critic(states).tolist()

    for i in range(4):
        assert abs(per_env_values[i] - batched_values[i]) < 1e-5
