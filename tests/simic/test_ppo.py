"""Tests for PPO module."""
import pytest
import torch

from esper.simic.ppo import signals_to_features, PPOAgent
from esper.simic.features import MULTISLOT_FEATURE_SIZE


def test_ppo_agent_architecture():
    """PPOAgent should use FactoredRecurrentActorCritic and TamiyoRolloutBuffer."""
    from esper.simic.tamiyo_buffer import TamiyoRolloutBuffer
    from esper.simic.tamiyo_network import FactoredRecurrentActorCritic

    agent = PPOAgent(
        state_dim=50,
        num_envs=4,
        max_steps_per_env=25,
        device="cpu",
        compile_network=False,  # Avoid compilation overhead in test
    )

    # Direct type checks
    assert isinstance(agent.buffer, TamiyoRolloutBuffer)
    assert isinstance(agent._base_network, FactoredRecurrentActorCritic)


def test_signals_to_features_with_multislot_params():
    """Test signals_to_features accepts total_seeds and max_seeds params."""
    from esper.leyline import SeedTelemetry

    # Create minimal signals mock
    class MockMetrics:
        epoch = 10
        global_step = 100
        train_loss = 0.5
        val_loss = 0.6
        loss_delta = -0.1
        train_accuracy = 85.0
        val_accuracy = 82.0
        accuracy_delta = 0.5
        plateau_epochs = 2
        best_val_accuracy = 83.0
        best_val_loss = 0.55
        grad_norm_host = 1.0

    class MockSignals:
        metrics = MockMetrics()
        loss_history = [0.8, 0.7, 0.6, 0.5, 0.5]
        accuracy_history = [70.0, 75.0, 80.0, 82.0, 85.0]
        active_seeds = []
        available_slots = 3
        seed_stage = 0
        seed_epochs_in_stage = 0
        seed_alpha = 0.0
        seed_improvement = 0.0
        seed_counterfactual = 0.0

    features = signals_to_features(
        signals=MockSignals(),
        model=None,
        use_telemetry=False,
        slots=["mid"],
        total_seeds=1,  # NEW param
        max_seeds=3,    # NEW param
    )

    assert len(features) == MULTISLOT_FEATURE_SIZE
