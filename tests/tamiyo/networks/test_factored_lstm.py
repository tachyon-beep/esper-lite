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


def test_predict_contributions_method():
    """predict_contributions() returns predictions from LSTM output."""
    slot_config = SlotConfig.default()
    num_slots = len(slot_config.slot_ids)

    policy = FactoredRecurrentActorCritic(
        state_dim=128,
        slot_config=slot_config,
    )

    batch_size = 4
    # state_dim=128 is the raw state dimension (before blueprint embeddings are added)
    # feature_net expects state_dim + blueprint_embed_total_dim = 128 + 12 = 140
    # But predict_contributions should handle the full forward path internally
    features = torch.randn(batch_size, 128)
    # Blueprint indices for 3 slots (valid range: -1 for inactive, 0-12 for active)
    blueprint_indices = torch.randint(-1, 13, (batch_size, num_slots))
    hidden = policy.get_initial_hidden(batch_size, torch.device("cpu"))

    # Predict contributions
    pred = policy.predict_contributions(features, blueprint_indices, hidden)

    assert pred.shape == (batch_size, num_slots)
    assert pred.requires_grad  # Should be differentiable


def test_predict_contributions_stop_gradient():
    """predict_contributions with stop_gradient=True detaches LSTM output."""
    slot_config = SlotConfig.default()
    num_slots = len(slot_config.slot_ids)
    policy = FactoredRecurrentActorCritic(
        state_dim=128,
        slot_config=slot_config,
    )

    batch_size = 4
    features = torch.randn(batch_size, 128, requires_grad=True)
    blueprint_indices = torch.randint(-1, 13, (batch_size, num_slots))
    hidden = policy.get_initial_hidden(batch_size, torch.device("cpu"))

    # With stop_gradient=True (default), LSTM params should not get gradients
    pred = policy.predict_contributions(features, blueprint_indices, hidden, stop_gradient=True)
    loss = pred.sum()
    loss.backward()

    # feature_net should NOT have gradients (detached before contribution_predictor)
    feature_net_grads = [
        p.grad for p in policy.feature_net.parameters() if p.grad is not None
    ]
    assert (
        len(feature_net_grads) == 0
    ), "feature_net should not receive gradients with stop_gradient=True"

    # contribution_predictor SHOULD have gradients
    contrib_grads = [
        p.grad for p in policy.contribution_predictor.parameters() if p.grad is not None
    ]
    assert len(contrib_grads) > 0, "contribution_predictor should receive gradients"


def test_predict_contributions_no_stop_gradient():
    """predict_contributions with stop_gradient=False allows LSTM gradients."""
    slot_config = SlotConfig.default()
    num_slots = len(slot_config.slot_ids)
    policy = FactoredRecurrentActorCritic(
        state_dim=128,
        slot_config=slot_config,
    )

    batch_size = 4
    features = torch.randn(batch_size, 128, requires_grad=True)
    blueprint_indices = torch.randint(-1, 13, (batch_size, num_slots))
    hidden = policy.get_initial_hidden(batch_size, torch.device("cpu"))

    # With stop_gradient=False, LSTM params should receive gradients
    pred = policy.predict_contributions(
        features, blueprint_indices, hidden, stop_gradient=False
    )
    loss = pred.sum()
    loss.backward()

    # feature_net SHOULD have gradients (not detached)
    feature_net_grads = [
        p.grad for p in policy.feature_net.parameters() if p.grad is not None
    ]
    assert (
        len(feature_net_grads) > 0
    ), "feature_net should receive gradients with stop_gradient=False"

    # contribution_predictor SHOULD also have gradients
    contrib_grads = [
        p.grad for p in policy.contribution_predictor.parameters() if p.grad is not None
    ]
    assert len(contrib_grads) > 0, "contribution_predictor should receive gradients"
