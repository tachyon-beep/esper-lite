"""Tests for FactoredRecurrentActorCritic contribution predictor head.

Phase 1.1 of Counterfactual Auxiliary Supervision implementation.
"""

import torch

from esper.leyline.slot_config import SlotConfig
from esper.tamiyo.networks.factored_lstm import FactoredRecurrentActorCritic


def test_policy_has_contribution_predictor():
    """FactoredRecurrentActorCritic has auxiliary contribution predictor head."""
    slot_config = SlotConfig.default()
    policy = FactoredRecurrentActorCritic(
        state_dim=128,
        slot_config=slot_config,
    )

    assert hasattr(policy, "contribution_predictor")
    assert policy.contribution_predictor is not None


def test_contribution_predictor_output_shape():
    """Contribution predictor outputs [batch, num_slots]."""
    slot_config = SlotConfig.default()
    num_slots = len(slot_config.slot_ids)

    policy = FactoredRecurrentActorCritic(
        state_dim=128,
        slot_config=slot_config,
    )

    batch_size = 4
    # The feature_net expects state_dim + blueprint_embed_dim (3 slots * 4 dim = 12)
    # So input is 128 (state_dim) + 12 (blueprint embeddings) = 140
    # But we need to go through full forward path to get lstm_out correctly
    # Let's test the contribution_predictor module directly with lstm_hidden_dim input

    # Get LSTM hidden dimension
    lstm_hidden_dim = policy.lstm_hidden_dim

    # Create mock LSTM output
    lstm_out = torch.randn(batch_size, lstm_hidden_dim)

    # Predict contributions
    pred_contributions = policy.contribution_predictor(lstm_out)

    assert pred_contributions.shape == (batch_size, num_slots)


def test_contribution_predictor_initialization():
    """Contribution predictor output layer initialized with gain=0.1."""
    slot_config = SlotConfig.default()
    policy = FactoredRecurrentActorCritic(
        state_dim=128,
        slot_config=slot_config,
    )

    # Get the final Linear layer from the Sequential
    contrib_last = policy.contribution_predictor[-1]

    # Verify it's a Linear layer
    assert isinstance(contrib_last, torch.nn.Linear)

    # Verify bias is initialized to zeros (centered predictions)
    assert torch.allclose(contrib_last.bias.data, torch.zeros_like(contrib_last.bias.data))

    # Verify output layer has small weights (gain=0.1 orthogonal init)
    # Orthogonal with gain=0.1 gives weights with Frobenius norm ≈ gain * sqrt(min(m, n))
    # For a 128 -> 3 layer, that's 0.1 * sqrt(3) ≈ 0.17
    # But since we're using orthogonal on a non-square matrix, just check weights are small
    weight_norm = contrib_last.weight.data.norm()
    # With gain=0.1 and a 128->3 matrix, the Frobenius norm should be around 0.1-0.5
    # (depending on exact orthogonal initialization behavior on non-square matrices)
    assert weight_norm < 1.0, f"Expected small weight norm (gain=0.1), got {weight_norm}"


def test_contribution_predictor_has_dropout():
    """Contribution predictor includes Dropout(0.1) to prevent shortcut learning."""
    slot_config = SlotConfig.default()
    policy = FactoredRecurrentActorCritic(
        state_dim=128,
        slot_config=slot_config,
    )

    # Check that there's a Dropout layer in the Sequential
    has_dropout = any(
        isinstance(layer, torch.nn.Dropout) for layer in policy.contribution_predictor
    )
    assert has_dropout, "Contribution predictor should include Dropout layer"

    # Verify dropout rate is 0.1
    dropout_layers = [
        layer
        for layer in policy.contribution_predictor
        if isinstance(layer, torch.nn.Dropout)
    ]
    assert len(dropout_layers) >= 1
    assert dropout_layers[0].p == 0.1
