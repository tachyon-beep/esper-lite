"""Tests for PPO feature dimension consistency.

This test suite validates that PPO network dimensions match the actual
feature vectors produced by signals_to_features(), preventing runtime
shape mismatch errors.
"""

import pytest
import torch

from esper.leyline import TrainingSignals, SeedTelemetry
from esper.simic.ppo import signals_to_features, PPOAgent


class TestPPOFeatureDimensions:
    """Test that PPO network state_dim matches signals_to_features output."""

    def test_signals_to_features_without_telemetry_is_27_dim(self):
        """Feature vector without telemetry must be exactly 27 dimensions."""
        signals = TrainingSignals()
        signals.metrics.epoch = 10
        signals.metrics.val_accuracy = 75.0

        features = signals_to_features(signals, model=None, use_telemetry=False)

        assert len(features) == 27, (
            f"Expected 27 base features, got {len(features)}. "
            "This is the base feature dimension without telemetry."
        )

    def test_signals_to_features_with_telemetry_is_37_dim(self):
        """Feature vector with telemetry must be 27 base + 10 telemetry = 37 dimensions."""
        signals = TrainingSignals()
        signals.metrics.epoch = 10
        signals.metrics.val_accuracy = 75.0

        features = signals_to_features(signals, model=None, use_telemetry=True)

        expected_dim = 27 + SeedTelemetry.feature_dim()  # 27 + 10 = 37
        assert len(features) == expected_dim, (
            f"Expected {expected_dim} features (27 base + {SeedTelemetry.feature_dim()} telemetry), "
            f"got {len(features)}. Telemetry adds {SeedTelemetry.feature_dim()} dimensions."
        )

    def test_ppo_agent_state_dim_without_telemetry_matches_features(self):
        """PPO agent created with use_telemetry=False must accept 27-dim vectors."""
        BASE_FEATURE_DIM = 27
        state_dim = BASE_FEATURE_DIM

        agent = PPOAgent(state_dim=state_dim, action_dim=7, device='cpu')

        # Create dummy 27-dim state tensor
        dummy_state = torch.randn(1, 27)

        # Forward pass should work without shape errors
        with torch.no_grad():
            dist, value = agent.network(dummy_state)

        # dist is a Categorical distribution
        assert dist.probs.shape == (1, 7), "Action probs should be (batch_size, action_dim)"
        assert value.shape == (1,), "Value should be (batch_size,)"

    def test_ppo_agent_state_dim_with_telemetry_matches_features(self):
        """PPO agent created with use_telemetry=True must accept 37-dim vectors."""
        BASE_FEATURE_DIM = 27
        state_dim = BASE_FEATURE_DIM + SeedTelemetry.feature_dim()  # 27 + 10 = 37

        agent = PPOAgent(state_dim=state_dim, action_dim=7, device='cpu')

        # Create dummy 37-dim state tensor
        dummy_state = torch.randn(1, 37)

        # Forward pass should work without shape errors
        with torch.no_grad():
            dist, value = agent.network(dummy_state)

        # dist is a Categorical distribution
        assert dist.probs.shape == (1, 7), "Action probs should be (batch_size, action_dim)"
        assert value.shape == (1,), "Value should be (batch_size,)"

    def test_ppo_agent_rejects_wrong_dimension(self):
        """PPO agent should fail with clear error when given wrong input dimension."""
        # Create agent expecting 37-dim input
        state_dim = 37
        agent = PPOAgent(state_dim=state_dim, action_dim=7, device='cpu')

        # Try to feed 54-dim input (old incorrect dimension)
        wrong_state = torch.randn(1, 54)

        with pytest.raises(RuntimeError, match="mat1 and mat2 shapes cannot be multiplied"):
            with torch.no_grad():
                agent.network(wrong_state)

    def test_telemetry_feature_dim_is_10(self):
        """Verify SeedTelemetry.feature_dim() returns 10 (not 27 legacy value)."""
        assert SeedTelemetry.feature_dim() == 10, (
            "SeedTelemetry.feature_dim() changed! This will break PPO dimension calculations. "
            "If intentional, update BASE_FEATURE_DIM calculations in training.py and vectorized.py."
        )

    def test_training_py_would_compute_correct_state_dim(self):
        """Verify the fixed dimension computation logic matches expected values."""
        # This tests the logic from training.py after the fix
        BASE_FEATURE_DIM = 27

        # Without telemetry
        use_telemetry = False
        state_dim_no_tel = BASE_FEATURE_DIM + (SeedTelemetry.feature_dim() if use_telemetry else 0)
        assert state_dim_no_tel == 27, "Without telemetry should be 27 dims"

        # With telemetry
        use_telemetry = True
        state_dim_tel = BASE_FEATURE_DIM + (SeedTelemetry.feature_dim() if use_telemetry else 0)
        assert state_dim_tel == 37, "With telemetry should be 37 dims (27 + 10)"

    def test_end_to_end_dimension_consistency(self):
        """End-to-end test: signals -> features -> agent forward pass."""
        # Setup
        signals = TrainingSignals()
        signals.metrics.epoch = 5
        signals.metrics.val_accuracy = 70.0

        # Test WITHOUT telemetry
        features_no_tel = signals_to_features(signals, model=None, use_telemetry=False)
        agent_no_tel = PPOAgent(state_dim=len(features_no_tel), action_dim=7, device='cpu')
        state_tensor_no_tel = torch.tensor([features_no_tel], dtype=torch.float32)

        with torch.no_grad():
            dist, value = agent_no_tel.network(state_tensor_no_tel)

        assert dist.probs.shape == (1, 7), "No-telemetry path should work"

        # Test WITH telemetry
        features_tel = signals_to_features(signals, model=None, use_telemetry=True)
        agent_tel = PPOAgent(state_dim=len(features_tel), action_dim=7, device='cpu')
        state_tensor_tel = torch.tensor([features_tel], dtype=torch.float32)

        with torch.no_grad():
            dist, value = agent_tel.network(state_tensor_tel)

        assert dist.probs.shape == (1, 7), "Telemetry path should work"


class TestEntropyAnnealing:
    """Test entropy coefficient annealing schedule."""

    def test_no_annealing_when_disabled(self):
        """entropy_anneal_steps=0 should use fixed entropy_coef."""
        from esper.simic.ppo import PPOAgent

        agent = PPOAgent(
            state_dim=27,
            action_dim=7,
            entropy_coef=0.05,
            entropy_anneal_steps=0,
            device='cpu'
        )
        assert agent.get_entropy_coef() == 0.05
        agent.train_steps = 100
        assert agent.get_entropy_coef() == 0.05  # Still fixed

    def test_annealing_at_start(self):
        """Step 0 should return entropy_coef_start."""
        from esper.simic.ppo import PPOAgent

        agent = PPOAgent(
            state_dim=27,
            action_dim=7,
            entropy_coef_start=0.2,
            entropy_coef_end=0.01,
            entropy_anneal_steps=100,
            device='cpu'
        )
        agent.train_steps = 0
        assert agent.get_entropy_coef() == 0.2

    def test_annealing_at_midpoint(self):
        """Midpoint should return average of start and end."""
        from esper.simic.ppo import PPOAgent

        agent = PPOAgent(
            state_dim=27,
            action_dim=7,
            entropy_coef_start=0.2,
            entropy_coef_end=0.0,
            entropy_anneal_steps=100,
            device='cpu'
        )
        agent.train_steps = 50
        assert abs(agent.get_entropy_coef() - 0.1) < 1e-6

    def test_annealing_at_end(self):
        """At anneal_steps, should return entropy_coef_end."""
        from esper.simic.ppo import PPOAgent

        agent = PPOAgent(
            state_dim=27,
            action_dim=7,
            entropy_coef_start=0.2,
            entropy_coef_end=0.01,
            entropy_anneal_steps=100,
            device='cpu'
        )
        agent.train_steps = 100
        assert abs(agent.get_entropy_coef() - 0.01) < 1e-6

    def test_annealing_clamps_beyond_schedule(self):
        """Beyond anneal_steps, should stay at entropy_coef_end."""
        from esper.simic.ppo import PPOAgent

        agent = PPOAgent(
            state_dim=27,
            action_dim=7,
            entropy_coef_start=0.2,
            entropy_coef_end=0.01,
            entropy_anneal_steps=100,
            device='cpu'
        )
        agent.train_steps = 200
        assert abs(agent.get_entropy_coef() - 0.01) < 1e-6
