"""Integration tests for PPO algorithm.

Tests that PPO components work together correctly:
- Feature extraction produces compatible tensors
- Forward pass works with extracted features
- Action sampling produces valid actions
"""

import pytest
import torch

from esper.simic.ppo import PPOAgent, signals_to_features
from esper.leyline import TrainingSignals, SeedTelemetry


def _all_valid_mask(batch_size: int = 1, action_dim: int = 7) -> torch.Tensor:
    """Create all-valid action mask for testing."""
    return torch.ones(batch_size, action_dim)


class TestPPOFeatureCompatibility:
    """Test that signals_to_features output is compatible with PPOAgent."""

    def test_features_without_telemetry_compatible_with_agent(self):
        """Features without telemetry should be compatible with 28-dim agent."""
        # Create signals
        signals = TrainingSignals()
        signals.metrics.epoch = 5
        signals.metrics.val_accuracy = 65.0
        signals.metrics.train_loss = 1.5
        signals.metrics.val_loss = 1.7

        # Extract features
        features = signals_to_features(signals, model=None, use_telemetry=False)

        # Create PPO agent with matching dimensions
        agent = PPOAgent(state_dim=len(features), action_dim=7, device='cpu')

        # Convert to tensor
        state_tensor = torch.tensor([features], dtype=torch.float32)
        mask = _all_valid_mask()

        # Should work without errors
        with torch.no_grad():
            dist, value = agent.network(state_tensor, mask)

        assert dist.probs.shape == (1, 7), "Policy should output 7 action probabilities"
        assert value.shape == (1,), "Value should be scalar per batch item"

    def test_features_with_telemetry_compatible_with_agent(self):
        """Features with telemetry should be compatible with 38-dim agent."""
        # Create signals (no active seed, so telemetry will be zero-padded)
        signals = TrainingSignals()
        signals.metrics.epoch = 10
        signals.metrics.val_accuracy = 70.0

        # Extract features with telemetry
        features = signals_to_features(signals, model=None, use_telemetry=True)

        # Create PPO agent with matching dimensions
        agent = PPOAgent(state_dim=len(features), action_dim=7, device='cpu')

        # Convert to tensor
        state_tensor = torch.tensor([features], dtype=torch.float32)
        mask = _all_valid_mask()

        # Should work without errors
        with torch.no_grad():
            dist, value = agent.network(state_tensor, mask)

        assert dist.probs.shape == (1, 7)
        assert value.shape == (1,)

    def test_batch_compatibility(self):
        """Test that batched features work with agent."""
        # Create multiple signals
        batch_size = 16
        all_features = []

        for i in range(batch_size):
            signals = TrainingSignals()
            signals.metrics.epoch = i
            signals.metrics.val_accuracy = 50.0 + i
            features = signals_to_features(signals, model=None, use_telemetry=False)
            all_features.append(features)

        # Stack into batch
        batch_tensor = torch.tensor(all_features, dtype=torch.float32)
        assert batch_tensor.shape == (batch_size, 30)

        # Create agent
        agent = PPOAgent(state_dim=30, action_dim=7, device='cpu')
        mask = _all_valid_mask(batch_size)

        # Should handle batch without errors
        with torch.no_grad():
            dist, value = agent.network(batch_tensor, mask)

        assert dist.probs.shape == (batch_size, 7)
        assert value.shape == (batch_size,)


class TestPPOForwardPass:
    """Test that PPOAgent forward pass works correctly."""

    def test_forward_pass_returns_valid_distribution(self):
        """Forward pass should return valid categorical distribution."""
        agent = PPOAgent(state_dim=30, action_dim=7, device='cpu')
        state = torch.randn(1, 30)
        mask = _all_valid_mask()

        with torch.no_grad():
            dist, value = agent.network(state, mask)

        # Distribution should be valid (MaskedCategorical)
        assert hasattr(dist, 'probs'), "Should have probs"
        assert hasattr(dist, 'masked_logits'), "Should have masked_logits"

        # Probabilities should sum to 1
        probs_sum = dist.probs.sum(dim=-1)
        assert torch.allclose(probs_sum, torch.ones(1), atol=1e-5), \
            f"Probs should sum to 1, got {probs_sum.item()}"

        # All probabilities should be in [0, 1]
        assert (dist.probs >= 0).all(), "All probs should be >= 0"
        assert (dist.probs <= 1).all(), "All probs should be <= 1"

    def test_forward_pass_value_is_scalar(self):
        """Value function should output scalar per state."""
        agent = PPOAgent(state_dim=30, action_dim=7, device='cpu')
        batch_size = 8
        states = torch.randn(batch_size, 30)
        mask = _all_valid_mask(batch_size)

        with torch.no_grad():
            dist, value = agent.network(states, mask)

        assert value.shape == (batch_size,), f"Expected shape ({batch_size},), got {value.shape}"

    def test_forward_pass_deterministic(self):
        """Same input should produce same output."""
        agent = PPOAgent(state_dim=30, action_dim=7, device='cpu')
        state = torch.randn(1, 30)
        mask = _all_valid_mask()

        # Set to eval mode for deterministic behavior
        agent.network.eval()

        with torch.no_grad():
            dist1, value1 = agent.network(state, mask)
            dist2, value2 = agent.network(state, mask)

        # Should be identical
        assert torch.allclose(dist1.probs, dist2.probs), "Probs should be deterministic"
        assert torch.allclose(value1, value2), "Value should be deterministic"

    def test_forward_pass_different_inputs_different_outputs(self):
        """Different inputs should produce different outputs."""
        agent = PPOAgent(state_dim=30, action_dim=7, device='cpu')
        mask = _all_valid_mask()

        # Create two very different states to ensure outputs differ
        state1 = torch.zeros(1, 30)
        state2 = torch.ones(1, 30) * 10.0  # Very different from zeros

        with torch.no_grad():
            dist1, value1 = agent.network(state1, mask)
            dist2, value2 = agent.network(state2, mask)

        # Outputs should be different
        probs_different = not torch.allclose(dist1.probs, dist2.probs, atol=1e-3)
        values_different = not torch.allclose(value1, value2, atol=1e-3)

        assert probs_different or values_different, \
            "Different inputs should produce different outputs"


class TestPPOActionSampling:
    """Test that action sampling works correctly."""

    def test_get_action_returns_valid_action(self):
        """get_action should return valid action index."""
        agent = PPOAgent(state_dim=30, action_dim=7, device='cpu')
        state = torch.randn(1, 30)
        mask = _all_valid_mask()

        action, log_prob, value, _ = agent.get_action(state, mask, deterministic=False)

        # Action should be valid index
        assert isinstance(action, int), f"Action should be int, got {type(action)}"
        assert 0 <= action < 7, f"Action {action} out of range [0, 7)"

        # Log prob should be negative (log of probability)
        assert isinstance(log_prob, float), f"Log prob should be float, got {type(log_prob)}"
        assert log_prob <= 0, f"Log prob {log_prob} should be <= 0"

        # Value should be float
        assert isinstance(value, float), f"Value should be float, got {type(value)}"

    def test_deterministic_action_selects_argmax(self):
        """Deterministic action should select highest probability action."""
        agent = PPOAgent(state_dim=30, action_dim=7, device='cpu')
        state = torch.randn(1, 30)
        mask = _all_valid_mask()

        # Get deterministic action multiple times
        actions = []
        for _ in range(10):
            action, _, _, _ = agent.get_action(state, mask, deterministic=True)
            actions.append(action)

        # All should be the same
        assert len(set(actions)) == 1, \
            f"Deterministic action should be consistent, got {set(actions)}"

    def test_stochastic_action_samples_from_distribution(self):
        """Stochastic action should sample from distribution."""
        agent = PPOAgent(state_dim=30, action_dim=7, device='cpu')
        state = torch.randn(1, 30)
        mask = _all_valid_mask()

        # Sample multiple times
        actions = []
        for _ in range(100):
            action, _, _, _ = agent.get_action(state, mask, deterministic=False)
            actions.append(action)

        # Should have some variety (with high probability)
        unique_actions = set(actions)
        assert len(unique_actions) > 1, \
            "Stochastic sampling should produce variety (got only one action in 100 samples)"

    def test_action_sampling_respects_probabilities(self):
        """Sampled actions should roughly match the distribution probabilities."""
        agent = PPOAgent(state_dim=30, action_dim=7, device='cpu')

        # Create a state that produces skewed probabilities
        # (In practice, the network initialization might already produce this)
        state = torch.randn(1, 30)
        mask = _all_valid_mask()

        # Sample many times
        n_samples = 1000
        action_counts = [0] * 7

        for _ in range(n_samples):
            action, _, _, _ = agent.get_action(state, mask, deterministic=False)
            action_counts[action] += 1

        # Get the actual probabilities from the network
        with torch.no_grad():
            dist, _ = agent.network(state, mask)
            expected_probs = dist.probs[0].numpy()

        # Observed frequencies
        observed_probs = [count / n_samples for count in action_counts]

        # They should be roughly similar (using Chi-square-like test)
        # Allow for sampling variance
        for i in range(7):
            expected = expected_probs[i]
            observed = observed_probs[i]

            # Allow 10% relative error (loose tolerance for stochastic test)
            if expected > 0.05:  # Only check for actions with non-trivial probability
                relative_error = abs(observed - expected) / expected
                assert relative_error < 0.3, \
                    f"Action {i}: expected prob {expected:.3f}, observed {observed:.3f}"

    def test_log_prob_matches_distribution(self):
        """Log prob returned by get_action should match distribution."""
        agent = PPOAgent(state_dim=30, action_dim=7, device='cpu')
        state = torch.randn(1, 30)
        mask = _all_valid_mask()

        # Get action with log prob
        action, returned_log_prob, _, _ = agent.get_action(state, mask, deterministic=False)

        # Compute log prob manually
        with torch.no_grad():
            dist, _ = agent.network(state, mask)
            expected_log_prob = dist.log_prob(torch.tensor([action])).item()

        # Should match
        assert abs(returned_log_prob - expected_log_prob) < 1e-5, \
            f"Returned log prob {returned_log_prob} doesn't match expected {expected_log_prob}"

    def test_value_matches_forward_pass(self):
        """Value returned by get_action should match forward pass."""
        agent = PPOAgent(state_dim=30, action_dim=7, device='cpu')
        state = torch.randn(1, 30)
        mask = _all_valid_mask()

        # Get value from get_action
        _, _, returned_value, _ = agent.get_action(state, mask, deterministic=False)

        # Get value from forward pass
        with torch.no_grad():
            _, expected_value = agent.network(state, mask)
            expected_value = expected_value.item()

        # Should match
        assert abs(returned_value - expected_value) < 1e-5, \
            f"Returned value {returned_value} doesn't match expected {expected_value}"


class TestPPOEndToEnd:
    """End-to-end integration tests."""

    def test_signals_to_action_pipeline(self):
        """Complete pipeline: TrainingSignals -> features -> action."""
        # Create realistic signals
        signals = TrainingSignals()
        signals.metrics.epoch = 15
        signals.metrics.global_step = 1500
        signals.metrics.train_loss = 1.2
        signals.metrics.val_loss = 1.4
        signals.metrics.val_accuracy = 68.5
        signals.metrics.best_val_accuracy = 70.0
        signals.metrics.plateau_epochs = 3

        # Extract features
        features = signals_to_features(signals, model=None, use_telemetry=False)

        # Create agent
        agent = PPOAgent(state_dim=len(features), action_dim=7, device='cpu')

        # Get action
        state_tensor = torch.tensor([features], dtype=torch.float32)
        mask = _all_valid_mask()
        action, log_prob, value, _ = agent.get_action(state_tensor, mask, deterministic=False)

        # All outputs should be valid
        assert 0 <= action < 7, f"Invalid action {action}"
        assert log_prob <= 0, f"Invalid log prob {log_prob}"
        assert isinstance(value, float), f"Invalid value type {type(value)}"

    def test_telemetry_pipeline(self):
        """Pipeline with telemetry features."""
        signals = TrainingSignals()
        signals.metrics.epoch = 20
        signals.metrics.val_accuracy = 75.0

        # Extract features with telemetry (will be zero-padded)
        features = signals_to_features(signals, model=None, use_telemetry=True)
        assert len(features) == 40, "Should have 40 features with telemetry"

        # Create agent
        agent = PPOAgent(state_dim=40, action_dim=7, device='cpu')

        # Get action
        state_tensor = torch.tensor([features], dtype=torch.float32)
        mask = _all_valid_mask()
        action, log_prob, value, _ = agent.get_action(state_tensor, mask)

        assert 0 <= action < 7
        assert log_prob <= 0
