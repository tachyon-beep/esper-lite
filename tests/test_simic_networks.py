"""Tests for RL network architectures."""

import pytest
import torch

from esper.simic.networks import ActorCritic, QNetwork, VNetwork


class TestActorCritic:
    """Tests for PPO ActorCritic network."""

    def test_forward_shapes(self):
        """Test forward pass returns correct shapes."""
        net = ActorCritic(state_dim=27, action_dim=7, hidden_dim=64)
        state = torch.randn(4, 27)  # batch of 4

        dist, value = net(state)

        assert value.shape == (4,)
        assert dist.probs.shape == (4, 7)

    def test_get_action(self):
        """Test single action sampling."""
        net = ActorCritic(state_dim=27, action_dim=7)
        state = torch.randn(1, 27)

        action, log_prob, value = net.get_action(state)

        assert isinstance(action, int)
        assert 0 <= action < 7
        assert isinstance(log_prob, float)
        assert isinstance(value, float)

    def test_get_action_deterministic(self):
        """Test deterministic action selection."""
        net = ActorCritic(state_dim=27, action_dim=7)
        state = torch.randn(1, 27)

        # Deterministic should give same action each time
        actions = [net.get_action(state, deterministic=True)[0] for _ in range(5)]
        assert len(set(actions)) == 1

    def test_get_action_batch(self):
        """Test batched action sampling."""
        net = ActorCritic(state_dim=27, action_dim=7)
        states = torch.randn(8, 27)

        actions, log_probs, values = net.get_action_batch(states)

        assert actions.shape == (8,)
        assert log_probs.shape == (8,)
        assert values.shape == (8,)

    def test_evaluate_actions(self):
        """Test action evaluation for PPO update."""
        net = ActorCritic(state_dim=27, action_dim=7)
        states = torch.randn(16, 27)
        actions = torch.randint(0, 7, (16,))

        log_probs, values, entropy = net.evaluate_actions(states, actions)

        assert log_probs.shape == (16,)
        assert values.shape == (16,)
        assert entropy.shape == (16,)


class TestQNetwork:
    """Tests for IQL Q-network."""

    def test_forward_shape(self):
        """Test Q-network outputs Q-values for all actions."""
        net = QNetwork(state_dim=27, action_dim=7)
        state = torch.randn(4, 27)

        q_values = net(state)

        assert q_values.shape == (4, 7)


class TestVNetwork:
    """Tests for IQL V-network."""

    def test_forward_shape(self):
        """Test V-network outputs scalar value."""
        net = VNetwork(state_dim=27)
        state = torch.randn(4, 27)

        values = net(state)

        assert values.shape == (4, 1)


def test_ppo_features_match_comparison_dimensions():
    """PPO and comparison should use same telemetry dimensions.

    This test prevents the critical bug where PPO uses 54-dim (27 base + 27 legacy)
    while comparison uses 37-dim (27 base + 10 seed), causing silent failures
    when models are trained in one mode and evaluated in another.
    """
    from esper.simic.ppo import signals_to_features
    from esper.simic.comparison import snapshot_to_features
    from esper.simic.episodes import TrainingSnapshot
    from esper.tamiyo import SignalTracker
    from esper.tolaria import create_model
    from esper.leyline import SeedTelemetry

    # Create mock signals and model
    tracker = SignalTracker()
    signals = tracker.update(
        epoch=1, global_step=100, train_loss=1.0, train_accuracy=50.0,
        val_loss=1.0, val_accuracy=50.0, active_seeds=[], available_slots=1
    )

    model = create_model("cpu")

    # PPO features with telemetry
    ppo_features = signals_to_features(signals, model, tracker, use_telemetry=True)

    # Comparison features with telemetry (zero seed telemetry)
    snapshot = TrainingSnapshot(
        epoch=1, global_step=100, train_loss=1.0, val_loss=1.0,
        loss_delta=0.0, train_accuracy=50.0, val_accuracy=50.0,
        accuracy_delta=0.0, plateau_epochs=0, best_val_accuracy=50.0,
        best_val_loss=1.0, loss_history_5=(1.0,)*5, accuracy_history_5=(50.0,)*5,
        has_active_seed=False, seed_stage=0, seed_epochs_in_stage=0,
        seed_alpha=0.0, seed_improvement=0.0, available_slots=1
    )

    # Create zero telemetry for comparison
    zero_telemetry = SeedTelemetry(seed_id="test")
    comparison_features = snapshot_to_features(
        snapshot, use_telemetry=True, seed_telemetry=zero_telemetry
    )

    # CRITICAL: Dimensions must match
    assert len(ppo_features) == len(comparison_features), \
        f"PPO uses {len(ppo_features)}-dim, comparison uses {len(comparison_features)}-dim"

    # Both should be 37-dim (27 base + 10 seed telemetry)
    assert len(ppo_features) == 37, f"Expected 37 dims, got {len(ppo_features)}"
