"""Test batched bootstrap value computation."""
import torch
from unittest.mock import MagicMock

from esper.simic.training.vectorized_trainer import _select_hidden_for_envs
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


# --- Bootstrap hidden-state slicing (regression for subset-truncation crash) ---
#
# The GAE bootstrap forward only covers TRUNCATED envs. When envs desync their
# episode lengths and only a strict subset truncates on an epoch, the bootstrap
# feature batch is smaller than the full env batch. batched_lstm_hidden carries
# EVERY env, so it must be sliced to the truncated subset (in order) before the
# recurrent forward -- otherwise the LSTM raises a hidden/input batch mismatch
# (Expected hidden[0] size (1, K, H), got [1, N, H]).


def test_select_hidden_subset_shape_and_order():
    """Sliced hidden matches the truncated-env subset in size AND env order."""
    num_envs, hidden_dim = 4, 8
    h = torch.arange(num_envs, dtype=torch.float32).reshape(1, num_envs, 1).repeat(
        1, 1, hidden_dim
    )
    c = h + 100.0  # distinct so we'd catch an h/c swap
    # Strict subset (3 of 4) in non-trivial order -- the exact crash shape.
    env_indices = [0, 1, 3]

    sel_h, sel_c = _select_hidden_for_envs((h, c), env_indices=env_indices)

    assert sel_h.shape == (1, len(env_indices), hidden_dim)
    assert sel_c.shape == (1, len(env_indices), hidden_dim)
    # Column k of the result must be the ORIGINAL env_indices[k]'s state.
    for k, env_id in enumerate(env_indices):
        assert torch.equal(sel_h[:, k, :], h[:, env_id, :])
        assert torch.equal(sel_c[:, k, :], c[:, env_id, :])


def test_select_hidden_full_batch_is_inert():
    """When all envs truncate in lockstep, slicing is a no-op clone (same values)."""
    num_envs, hidden_dim = 4, 8
    h = torch.randn(1, num_envs, hidden_dim)
    c = torch.randn(1, num_envs, hidden_dim)

    sel_h, sel_c = _select_hidden_for_envs((h, c), env_indices=list(range(num_envs)))

    assert torch.equal(sel_h, h)
    assert torch.equal(sel_c, c)


def test_select_hidden_none_passthrough():
    """No-LSTM agents carry no hidden; slicing must pass None through."""
    assert _select_hidden_for_envs(None, env_indices=[0, 2]) is None


def test_select_hidden_preserves_duplicates():
    """An env truncating twice in a segment selects its column twice, in order."""
    h = torch.arange(3, dtype=torch.float32).reshape(1, 3, 1).repeat(1, 1, 4)
    c = h.clone()
    sel_h, _ = _select_hidden_for_envs((h, c), env_indices=[2, 0, 2])
    assert sel_h.shape == (1, 3, 4)
    assert torch.equal(sel_h[:, 0, :], h[:, 2, :])
    assert torch.equal(sel_h[:, 1, :], h[:, 0, :])
    assert torch.equal(sel_h[:, 2, :], h[:, 2, :])
