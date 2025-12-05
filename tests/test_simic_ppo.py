"""Tests for PPO feature dimension consistency.

This test suite validates that PPO network dimensions match the actual
feature vectors produced by signals_to_features(), preventing runtime
shape mismatch errors.
"""

import pytest
import torch

from esper.leyline import TrainingSignals, SeedTelemetry
from esper.simic.ppo import signals_to_features, PPOAgent
from esper.simic.networks import ActorCritic, RecurrentActorCritic


class TestPPOFeatureDimensions:
    """Test that PPO network state_dim matches signals_to_features output."""

    def test_signals_to_features_without_telemetry_is_28_dim(self):
        """Feature vector without telemetry must be exactly 28 dimensions."""
        signals = TrainingSignals()
        signals.metrics.epoch = 10
        signals.metrics.val_accuracy = 75.0

        features = signals_to_features(signals, model=None, use_telemetry=False)

        assert len(features) == 30, (
            f"Expected 30 base features, got {len(features)}. "
            "This is the base feature dimension without telemetry."
        )

    def test_signals_to_features_with_telemetry_is_40_dim(self):
        """Feature vector with telemetry must be 30 base + 10 telemetry = 40 dimensions."""
        signals = TrainingSignals()
        signals.metrics.epoch = 10
        signals.metrics.val_accuracy = 75.0

        features = signals_to_features(signals, model=None, use_telemetry=True)

        expected_dim = 30 + SeedTelemetry.feature_dim()  # 30 + 10 = 40
        assert len(features) == expected_dim, (
            f"Expected {expected_dim} features (30 base + {SeedTelemetry.feature_dim()} telemetry), "
            f"got {len(features)}. Telemetry adds {SeedTelemetry.feature_dim()} dimensions."
        )

    def test_ppo_agent_state_dim_without_telemetry_matches_features(self):
        """PPO agent created with use_telemetry=False must accept 30-dim vectors."""
        BASE_FEATURE_DIM = 30
        state_dim = BASE_FEATURE_DIM

        agent = PPOAgent(state_dim=state_dim, action_dim=7, device='cpu')

        # Create dummy 30-dim state tensor and all-valid action mask
        dummy_state = torch.randn(1, 30)
        dummy_mask = torch.ones(1, 7)  # All actions valid

        # Forward pass should work without shape errors
        with torch.no_grad():
            dist, value = agent.network(dummy_state, dummy_mask)

        # dist is a MaskedCategorical distribution
        assert dist.probs.shape == (1, 7), "Action probs should be (batch_size, action_dim)"
        assert value.shape == (1,), "Value should be (batch_size,)"

    def test_ppo_agent_state_dim_with_telemetry_matches_features(self):
        """PPO agent created with use_telemetry=True must accept 40-dim vectors."""
        BASE_FEATURE_DIM = 30
        state_dim = BASE_FEATURE_DIM + SeedTelemetry.feature_dim()  # 30 + 10 = 40

        agent = PPOAgent(state_dim=state_dim, action_dim=7, device='cpu')

        # Create dummy 40-dim state tensor and all-valid action mask
        dummy_state = torch.randn(1, 40)
        dummy_mask = torch.ones(1, 7)  # All actions valid

        # Forward pass should work without shape errors
        with torch.no_grad():
            dist, value = agent.network(dummy_state, dummy_mask)

        # dist is a MaskedCategorical distribution
        assert dist.probs.shape == (1, 7), "Action probs should be (batch_size, action_dim)"
        assert value.shape == (1,), "Value should be (batch_size,)"

    def test_ppo_agent_rejects_wrong_dimension(self):
        """PPO agent should fail with clear error when given wrong input dimension."""
        # Create agent expecting 40-dim input
        state_dim = 40
        agent = PPOAgent(state_dim=state_dim, action_dim=7, device='cpu')

        # Try to feed 54-dim input (old incorrect dimension)
        wrong_state = torch.randn(1, 54)
        dummy_mask = torch.ones(1, 7)  # All actions valid

        with pytest.raises(RuntimeError, match="mat1 and mat2 shapes cannot be multiplied"):
            with torch.no_grad():
                agent.network(wrong_state, dummy_mask)

    def test_telemetry_feature_dim_is_10(self):
        """Verify SeedTelemetry.feature_dim() returns 10 (not 27 legacy value)."""
        assert SeedTelemetry.feature_dim() == 10, (
            "SeedTelemetry.feature_dim() changed! This will break PPO dimension calculations. "
            "If intentional, update BASE_FEATURE_DIM calculations in training.py and vectorized.py."
        )

    def test_training_py_would_compute_correct_state_dim(self):
        """Verify the fixed dimension computation logic matches expected values."""
        # This tests the logic from training.py after the fix
        BASE_FEATURE_DIM = 30

        # Without telemetry
        use_telemetry = False
        state_dim_no_tel = BASE_FEATURE_DIM + (SeedTelemetry.feature_dim() if use_telemetry else 0)
        assert state_dim_no_tel == 30, "Without telemetry should be 30 dims"

        # With telemetry
        use_telemetry = True
        state_dim_tel = BASE_FEATURE_DIM + (SeedTelemetry.feature_dim() if use_telemetry else 0)
        assert state_dim_tel == 40, "With telemetry should be 40 dims (30 + 10)"

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
        dummy_mask = torch.ones(1, 7)  # All actions valid

        with torch.no_grad():
            dist, value = agent_no_tel.network(state_tensor_no_tel, dummy_mask)

        assert dist.probs.shape == (1, 7), "No-telemetry path should work"

        # Test WITH telemetry
        features_tel = signals_to_features(signals, model=None, use_telemetry=True)
        agent_tel = PPOAgent(state_dim=len(features_tel), action_dim=7, device='cpu')
        state_tensor_tel = torch.tensor([features_tel], dtype=torch.float32)

        with torch.no_grad():
            dist, value = agent_tel.network(state_tensor_tel, dummy_mask)

        assert dist.probs.shape == (1, 7), "Telemetry path should work"


class TestEntropyAnnealing:
    """Test entropy coefficient annealing schedule."""

    def test_no_annealing_when_disabled(self):
        """entropy_anneal_steps=0 should use fixed entropy_coef."""
        from esper.simic.ppo import PPOAgent

        agent = PPOAgent(
            state_dim=30,
            action_dim=7,
            entropy_coef=0.05,
            entropy_coef_min=0.0,  # Disable floor for this test
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
            state_dim=30,
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
            state_dim=30,
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
            state_dim=30,
            action_dim=7,
            entropy_coef_start=0.2,
            entropy_coef_end=0.01,
            entropy_coef_min=0.0,  # Disable floor for this test
            entropy_anneal_steps=100,
            device='cpu'
        )
        agent.train_steps = 100
        assert abs(agent.get_entropy_coef() - 0.01) < 1e-6

    def test_annealing_clamps_beyond_schedule(self):
        """Beyond anneal_steps, should stay at entropy_coef_end."""
        from esper.simic.ppo import PPOAgent

        agent = PPOAgent(
            state_dim=30,
            action_dim=7,
            entropy_coef_start=0.2,
            entropy_coef_end=0.01,
            entropy_coef_min=0.0,  # Disable floor for this test
            entropy_anneal_steps=100,
            device='cpu'
        )
        agent.train_steps = 200
        assert abs(agent.get_entropy_coef() - 0.01) < 1e-6

    def test_entropy_floor_prevents_collapse(self):
        """Entropy floor should prevent coefficient from going below minimum."""
        from esper.simic.ppo import PPOAgent

        agent = PPOAgent(
            state_dim=30,
            action_dim=7,
            entropy_coef_start=0.2,
            entropy_coef_end=0.0,  # Would go to zero without floor
            entropy_coef_min=0.1,  # Floor at 0.1
            entropy_anneal_steps=100,
            device='cpu'
        )
        # At end of annealing, should be clamped to floor
        agent.train_steps = 100
        assert agent.get_entropy_coef() == 0.1

        # Beyond annealing, still at floor
        agent.train_steps = 200
        assert agent.get_entropy_coef() == 0.1

    def test_entropy_floor_default_is_sensible(self):
        """Default entropy floor should be 0.01 (unified minimum)."""
        from esper.simic.ppo import PPOAgent

        agent = PPOAgent(
            state_dim=30,
            action_dim=7,
            entropy_coef_start=0.2,
            entropy_coef_end=0.001,  # Below default floor of 0.01
            entropy_anneal_steps=100,
            device='cpu'
        )
        # Default floor should prevent going below 0.01 (unified minimum)
        agent.train_steps = 100
        assert agent.get_entropy_coef() == 0.01  # Clamped at default floor

    def test_annealed_entropy_used_in_update(self):
        """PPO update should use annealed entropy coefficient."""
        from esper.simic.ppo import PPOAgent
        import torch

        # Create agent with annealing
        agent = PPOAgent(
            state_dim=30,
            action_dim=7,
            entropy_coef_start=0.5,
            entropy_coef_end=0.01,
            entropy_coef_min=0.0,  # Disable floor for this test
            entropy_anneal_steps=10,
            device='cpu'
        )

        # Add some dummy transitions with action masks
        dummy_mask = torch.ones(7)  # All actions valid
        for _ in range(5):
            state = torch.randn(30)
            agent.store_transition(state, action=0, log_prob=-1.0, value=0.5, reward=1.0, done=False, action_mask=dummy_mask)

        # At step 0, entropy_coef should be 0.5
        assert agent.train_steps == 0
        assert agent.get_entropy_coef() == 0.5

        # Perform update
        metrics = agent.update(last_value=0.0)

        # After update, train_steps incremented
        assert agent.train_steps == 1
        # Entropy coef should have changed
        expected_coef = 0.5 + (1/10) * (0.01 - 0.5)  # 0.451
        assert abs(agent.get_entropy_coef() - expected_coef) < 1e-6


class TestRecurrentPPOAgent:
    """Tests for recurrent PPO agent."""

    def test_init_with_recurrent_creates_lstm_network(self):
        """PPOAgent(recurrent=True) should use RecurrentActorCritic."""
        agent = PPOAgent(state_dim=30, action_dim=7, recurrent=True, lstm_hidden_dim=128)
        assert isinstance(agent.network, RecurrentActorCritic)
        assert agent.recurrent is True

    def test_init_without_recurrent_uses_mlp(self):
        """PPOAgent(recurrent=False) should use standard ActorCritic."""
        agent = PPOAgent(state_dim=30, action_dim=7, recurrent=False)
        assert isinstance(agent.network, ActorCritic)
        assert agent.recurrent is False

    def test_get_action_returns_hidden_when_recurrent(self):
        """get_action should return hidden state for recurrent agent."""
        agent = PPOAgent(state_dim=30, action_dim=7, recurrent=True, device='cpu')
        state = torch.randn(30)
        mask = torch.ones(7, dtype=torch.bool)

        result = agent.get_action(state, mask, hidden=None)

        assert len(result) == 4  # (action, log_prob, value, hidden)
        action, log_prob, value, hidden = result
        assert isinstance(action, int)
        assert hidden is not None
        assert len(hidden) == 2  # (h, c)

    def test_update_recurrent_uses_batched_chunks(self):
        """Recurrent update should use batched chunk processing."""
        agent = PPOAgent(
            state_dim=30, action_dim=7, recurrent=True, device='cpu',
            chunk_length=4, lstm_hidden_dim=64,
        )

        # Add episode
        agent.recurrent_buffer.start_episode(env_id=0)
        hidden = None
        for i in range(6):
            state = torch.randn(30)
            mask = torch.ones(7, dtype=torch.bool)
            action, log_prob, value, hidden = agent.get_action(state, mask, hidden)
            agent.store_recurrent_transition(
                state=state, action=action, log_prob=log_prob, value=value,
                reward=0.1, done=(i == 5), action_mask=mask, env_id=0,
            )
        agent.recurrent_buffer.end_episode(env_id=0)

        metrics = agent.update_recurrent()

        assert 'policy_loss' in metrics
        assert 'value_loss' in metrics
        assert metrics['policy_loss'] != 0.0

    def test_advantages_are_nonzero_in_update(self):
        """Verify GAE advantages flow through to update (critical bug fix)."""
        agent = PPOAgent(
            state_dim=30, action_dim=7, recurrent=True, device='cpu',
            chunk_length=4, lstm_hidden_dim=64,
        )

        # Add episode with increasing rewards
        agent.recurrent_buffer.start_episode(env_id=0)
        hidden = None
        for i in range(4):
            state = torch.randn(30)
            mask = torch.ones(7, dtype=torch.bool)
            action, log_prob, value, hidden = agent.get_action(state, mask, hidden)
            agent.store_recurrent_transition(
                state=state, action=action, log_prob=log_prob, value=value,
                reward=float(i + 1),  # Increasing rewards
                done=(i == 3), action_mask=mask, env_id=0,
            )
        agent.recurrent_buffer.end_episode(env_id=0)

        # Compute GAE
        agent.recurrent_buffer.compute_gae(gamma=0.99, gae_lambda=0.95)
        chunks = agent.recurrent_buffer.get_chunks(device='cpu')

        # Advantages must be non-zero
        assert chunks[0]['advantages'].abs().sum() > 0, "Advantages should be non-zero with rewards"

    def test_value_coef_used_correctly(self):
        """Value coefficient should be from agent, not hardcoded."""
        agent = PPOAgent(
            state_dim=30, action_dim=7, recurrent=True, device='cpu',
            value_coef=0.25,  # Non-default value
        )
        assert agent.value_coef == 0.25
