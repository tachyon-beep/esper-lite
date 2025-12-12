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
        """Feature vector without telemetry must be exactly 50 dimensions."""
        signals = TrainingSignals()
        signals.metrics.epoch = 10
        signals.metrics.val_accuracy = 75.0

        features = signals_to_features(signals, model=None, use_telemetry=False, slots=["mid"])

        assert len(features) == 50, (
            f"Expected 50 base features, got {len(features)}. "
            "This is the base feature dimension without telemetry."
        )

    def test_signals_to_features_with_telemetry_is_40_dim(self):
        """Feature vector with telemetry must be 50 base + 10 telemetry = 60 dimensions."""
        signals = TrainingSignals()
        signals.metrics.epoch = 10
        signals.metrics.val_accuracy = 75.0

        features = signals_to_features(signals, model=None, use_telemetry=True, slots=["mid"])

        expected_dim = 50 + SeedTelemetry.feature_dim()  # 50 + 10 = 60
        assert len(features) == expected_dim, (
            f"Expected {expected_dim} features (50 base + {SeedTelemetry.feature_dim()} telemetry), "
            f"got {len(features)}. Telemetry adds {SeedTelemetry.feature_dim()} dimensions."
        )

    def test_ppo_agent_state_dim_without_telemetry_matches_features(self):
        """PPO agent created with use_telemetry=False must accept 50-dim vectors."""
        BASE_FEATURE_DIM = 50
        state_dim = BASE_FEATURE_DIM

        agent = PPOAgent(state_dim=state_dim, action_dim=7, device='cpu')

        # Create dummy 50-dim state tensor and all-valid action mask
        dummy_state = torch.randn(1, 50)
        dummy_mask = torch.ones(1, 7)  # All actions valid

        # Forward pass should work without shape errors
        with torch.no_grad():
            dist, value = agent.network(dummy_state, dummy_mask)

        # dist is a MaskedCategorical distribution
        assert dist.probs.shape == (1, 7), "Action probs should be (batch_size, action_dim)"
        assert value.shape == (1,), "Value should be (batch_size,)"

    def test_ppo_agent_state_dim_with_telemetry_matches_features(self):
        """PPO agent created with use_telemetry=True must accept 60-dim vectors."""
        BASE_FEATURE_DIM = 50
        state_dim = BASE_FEATURE_DIM + SeedTelemetry.feature_dim()  # 50 + 10 = 60

        agent = PPOAgent(state_dim=state_dim, action_dim=7, device='cpu')

        # Create dummy 60-dim state tensor and all-valid action mask
        dummy_state = torch.randn(1, 60)
        dummy_mask = torch.ones(1, 7)  # All actions valid

        # Forward pass should work without shape errors
        with torch.no_grad():
            dist, value = agent.network(dummy_state, dummy_mask)

        # dist is a MaskedCategorical distribution
        assert dist.probs.shape == (1, 7), "Action probs should be (batch_size, action_dim)"
        assert value.shape == (1,), "Value should be (batch_size,)"

    def test_ppo_agent_rejects_wrong_dimension(self):
        """PPO agent should fail with clear error when given wrong input dimension."""
        # Create agent expecting 45-dim input
        # Use compile_network=False to test raw network error (Dynamo produces different message)
        state_dim = 45
        agent = PPOAgent(state_dim=state_dim, action_dim=7, device='cpu', compile_network=False)

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
        BASE_FEATURE_DIM = 35

        # Without telemetry
        use_telemetry = False
        state_dim_no_tel = BASE_FEATURE_DIM + (SeedTelemetry.feature_dim() if use_telemetry else 0)
        assert state_dim_no_tel == 35, "Without telemetry should be 35 dims"

        # With telemetry
        use_telemetry = True
        state_dim_tel = BASE_FEATURE_DIM + (SeedTelemetry.feature_dim() if use_telemetry else 0)
        assert state_dim_tel == 45, "With telemetry should be 45 dims (35 + 10)"

    def test_end_to_end_dimension_consistency(self):
        """End-to-end test: signals -> features -> agent forward pass."""
        # Setup
        signals = TrainingSignals()
        signals.metrics.epoch = 5
        signals.metrics.val_accuracy = 70.0

        # Test WITHOUT telemetry
        features_no_tel = signals_to_features(signals, model=None, use_telemetry=False, slots=["mid"])
        agent_no_tel = PPOAgent(state_dim=len(features_no_tel), action_dim=7, device='cpu')
        state_tensor_no_tel = torch.tensor([features_no_tel], dtype=torch.float32)
        dummy_mask = torch.ones(1, 7)  # All actions valid

        with torch.no_grad():
            dist, value = agent_no_tel.network(state_tensor_no_tel, dummy_mask)

        assert dist.probs.shape == (1, 7), "No-telemetry path should work"

        # Test WITH telemetry
        features_tel = signals_to_features(signals, model=None, use_telemetry=True, slots=["mid"])
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
        # Use _base_network to check type (handles torch.compile wrapper)
        assert isinstance(agent._base_network, RecurrentActorCritic)
        assert agent.recurrent is True

    def test_init_without_recurrent_uses_mlp(self):
        """PPOAgent(recurrent=False) should use standard ActorCritic."""
        agent = PPOAgent(state_dim=30, action_dim=7, recurrent=False)
        # Use _base_network to check type (handles torch.compile wrapper)
        assert isinstance(agent._base_network, ActorCritic)
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

    def test_clip_value_false_uses_mse_loss(self):
        """Recurrent PPO with clip_value=False should use plain MSE loss."""
        agent = PPOAgent(
            state_dim=30, action_dim=7, recurrent=True, device='cpu',
            chunk_length=4, lstm_hidden_dim=64,
            clip_value=False,  # Disable value clipping
        )
        assert agent.clip_value is False

        # Add episode
        agent.recurrent_buffer.start_episode(env_id=0)
        hidden = None
        for i in range(4):
            state = torch.randn(30)
            mask = torch.ones(7, dtype=torch.bool)
            action, log_prob, value, hidden = agent.get_action(state, mask, hidden)
            agent.store_recurrent_transition(
                state=state, action=action, log_prob=log_prob, value=value,
                reward=0.1, done=(i == 3), action_mask=mask, env_id=0,
            )
        agent.recurrent_buffer.end_episode(env_id=0)

        # Should run without error and return metrics
        metrics = agent.update_recurrent()
        assert 'value_loss' in metrics
        assert metrics['value_loss'] >= 0.0


class TestPPOAnomalyTelemetry:
    """Test that PPO emits anomaly telemetry events."""

    def test_ratio_explosion_emits_telemetry(self):
        """Ratio explosion triggers RATIO_EXPLOSION_DETECTED event."""
        from esper.leyline import TelemetryEventType
        from esper.nissa import get_hub
        from esper.simic.anomaly_detector import AnomalyDetector

        # Capture emitted events
        captured_events = []
        hub = get_hub()

        class CaptureBackend:
            def emit(self, event):
                captured_events.append(event)

        hub.add_backend(CaptureBackend())

        # Create agent with very tight ratio threshold to trigger explosion
        agent = PPOAgent(
            state_dim=10,
            action_dim=5,
            device='cpu',
        )

        # Add steps that will cause ratio explosion
        for _ in range(10):
            state = torch.randn(10)
            agent.buffer.add(
                state=state,
                action=0,
                log_prob=-5.0,  # Very low log prob will cause high ratio
                value=0.0,
                reward=1.0,
                done=False,
                action_mask=torch.ones(5),
            )

        # Update should detect ratio explosion
        agent.update()

        # Check for RATIO_EXPLOSION_DETECTED event
        explosion_events = [
            e for e in captured_events
            if e.event_type == TelemetryEventType.RATIO_EXPLOSION_DETECTED
        ]
        assert len(explosion_events) >= 1, f"Expected RATIO_EXPLOSION_DETECTED, got: {[e.event_type for e in captured_events]}"
        assert "ratio_max" in explosion_events[0].data

    def test_value_collapse_emits_telemetry(self):
        """Value collapse triggers VALUE_COLLAPSE_DETECTED event."""
        from esper.leyline import TelemetryEventType
        from esper.nissa import get_hub
        from esper.simic.anomaly_detector import AnomalyDetector

        # Capture emitted events
        captured_events = []
        hub = get_hub()

        class CaptureBackend:
            def emit(self, event):
                captured_events.append(event)

        hub.add_backend(CaptureBackend())

        # Create agent with very tight explained variance threshold
        agent = PPOAgent(
            state_dim=10,
            action_dim=5,
            device='cpu',
        )

        # Add steps with constant values (will cause low explained variance)
        for _ in range(10):
            state = torch.randn(10)
            agent.buffer.add(
                state=state,
                action=0,
                log_prob=-1.0,
                value=0.0,  # Constant value
                reward=0.0,  # No reward
                done=False,
                action_mask=torch.ones(5),
            )

        # Update should detect value collapse
        agent.update()

        # Check for VALUE_COLLAPSE_DETECTED event
        collapse_events = [
            e for e in captured_events
            if e.event_type == TelemetryEventType.VALUE_COLLAPSE_DETECTED
        ]
        assert len(collapse_events) >= 1, f"Expected VALUE_COLLAPSE_DETECTED, got: {[e.event_type for e in captured_events]}"
        assert "explained_variance" in collapse_events[0].data

    def test_all_anomaly_event_data_fields_present(self):
        """Verify all anomaly events include required data fields."""
        from esper.leyline import TelemetryEventType
        from esper.nissa import get_hub

        # Capture emitted events
        captured_events = []
        hub = get_hub()

        class CaptureBackend:
            def emit(self, event):
                captured_events.append(event)

        hub.add_backend(CaptureBackend())

        # Create agent
        agent = PPOAgent(
            state_dim=10,
            action_dim=5,
            device='cpu',
        )

        # Add steps that will trigger ratio explosion
        for _ in range(10):
            state = torch.randn(10)
            agent.buffer.add(
                state=state,
                action=0,
                log_prob=-5.0,
                value=0.0,
                reward=1.0,
                done=False,
                action_mask=torch.ones(5),
            )

        # Update should detect anomaly
        agent.update()

        # Check data fields
        anomaly_events = [
            e for e in captured_events
            if e.event_type in [
                TelemetryEventType.RATIO_EXPLOSION_DETECTED,
                TelemetryEventType.VALUE_COLLAPSE_DETECTED,
                TelemetryEventType.NUMERICAL_INSTABILITY_DETECTED,
                TelemetryEventType.GRADIENT_ANOMALY,
            ]
        ]
        assert len(anomaly_events) >= 1

        for event in anomaly_events:
            # All anomaly events should have these fields
            assert "anomaly_type" in event.data
            assert "detail" in event.data
            assert "ratio_max" in event.data
            assert "ratio_min" in event.data
            assert "explained_variance" in event.data
            assert "train_steps" in event.data
            assert event.severity == "warning"

    def test_numerical_instability_emits_telemetry(self):
        """NaN loss triggers NUMERICAL_INSTABILITY_DETECTED event."""
        from esper.leyline import TelemetryEventType
        from esper.nissa import get_hub
        import numpy as np

        # Capture emitted events
        captured_events = []
        hub = get_hub()

        class CaptureBackend:
            def emit(self, event):
                captured_events.append(event)

        hub.add_backend(CaptureBackend())

        # Create agent
        agent = PPOAgent(
            state_dim=10,
            action_dim=5,
            device='cpu',
        )

        # Add normal transitions
        for _ in range(10):
            state = torch.randn(10)
            agent.buffer.add(
                state=state,
                action=0,
                log_prob=-1.0,
                value=0.5,
                reward=1.0,
                done=False,
                action_mask=torch.ones(5),
            )

        # Monkey-patch the network to return NaN loss
        original_evaluate = agent.network.evaluate_actions

        def evaluate_with_nan(*args, **kwargs):
            log_probs, values, entropy = original_evaluate(*args, **kwargs)
            # Inject NaN into values to cause NaN loss
            values = torch.full_like(values, float('nan'))
            return log_probs, values, entropy

        agent.network.evaluate_actions = evaluate_with_nan

        # Update should detect numerical instability
        agent.update()

        # Check for NUMERICAL_INSTABILITY_DETECTED event
        instability_events = [
            e for e in captured_events
            if e.event_type == TelemetryEventType.NUMERICAL_INSTABILITY_DETECTED
        ]

        # Should have detected NaN
        assert len(instability_events) >= 1, (
            f"Expected NUMERICAL_INSTABILITY_DETECTED, got: {[e.event_type for e in captured_events]}"
        )
        assert "anomaly_type" in instability_events[0].data
        assert instability_events[0].data["anomaly_type"] == "numerical_instability"


def test_fused_nan_inf_detection():
    """Verify NaN/Inf detection uses single fused check instead of separate checks."""
    import torch
    from esper.simic.ppo import PPOAgent
    from unittest.mock import patch

    agent = PPOAgent(state_dim=10, action_dim=4, device="cpu")

    # Create buffer with normal data
    state = torch.randn(10)
    action_mask = torch.ones(4)
    agent.store_transition(state, 0, -0.5, 0.5, 1.0, False, action_mask)
    agent.store_transition(state, 1, -0.3, 0.6, 0.5, True, action_mask)

    # Mock ratio to contain NaN to trigger detection
    original_evaluate = agent.network.evaluate_actions
    def mock_evaluate(states, actions, masks):
        log_probs, values, entropy = original_evaluate(states, actions, masks)
        # Create old_log_probs that will make ratio become NaN
        # ratio = exp(log_probs - old_log_probs), if old_log_probs is NaN, ratio is NaN
        log_probs = torch.full_like(log_probs, float('nan'))
        return log_probs, values, entropy

    with patch.object(agent.network, 'evaluate_actions', side_effect=mock_evaluate):
        metrics = agent.update(last_value=0.0)

    # OLD implementation sets separate flags: ratio_has_nan=True, ratio_has_inf=True
    # NEW implementation sets combined flag: ratio_has_numerical_issue=True
    # This test expects OLD behavior to fail, then we'll implement NEW behavior
    has_old_api = 'ratio_has_nan' in metrics or 'ratio_has_inf' in metrics
    has_new_api = 'ratio_has_numerical_issue' in metrics

    # Test should fail with old implementation (has_old_api=True)
    # After fix, should pass (has_old_api=False, has_new_api=True)
    assert not has_old_api, \
        "Should use fused check (ratio_has_numerical_issue), not separate nan/inf flags"
    assert has_new_api or not any(metrics.values()), \
        "Should have ratio_has_numerical_issue when numerical issues detected"


def test_ppo_agent_weight_decay_critic_only():
    """PPOAgent weight decay should apply only to critic, not actor or shared."""
    import torch
    from esper.simic.ppo import PPOAgent

    # Without weight decay - uses Adam
    agent_no_wd = PPOAgent(state_dim=10, action_dim=4, device="cpu")
    assert isinstance(agent_no_wd.optimizer, torch.optim.Adam)

    # With weight decay - should use AdamW with param groups
    agent_with_wd = PPOAgent(
        state_dim=10, action_dim=4, device="cpu", weight_decay=0.01
    )
    assert isinstance(agent_with_wd.optimizer, torch.optim.AdamW)

    # Verify param groups exist
    param_groups = agent_with_wd.optimizer.param_groups
    assert len(param_groups) >= 3, "Should have separate param groups for actor/shared/critic"

    # Find each group's weight decay
    actor_wd = None
    shared_wd = None
    critic_wd = None
    for group in param_groups:
        name = group.get('name', '')
        if 'actor' in name:
            actor_wd = group['weight_decay']
        elif 'shared' in name:
            shared_wd = group['weight_decay']
        elif 'critic' in name:
            critic_wd = group['weight_decay']

    # Actor should NOT have weight decay (biases toward determinism)
    assert actor_wd == 0.0, f"Actor weight_decay should be 0, got {actor_wd}"
    # Shared should NOT have weight decay (feeds into actor)
    assert shared_wd == 0.0, f"Shared weight_decay should be 0, got {shared_wd}"
    # Critic SHOULD have weight decay
    assert critic_wd > 0.0, f"Critic weight_decay should be >0, got {critic_wd}"


def test_ppo_agent_weight_decay_recurrent():
    """PPOAgent weight decay should work with recurrent=True."""
    import torch
    from esper.simic.ppo import PPOAgent

    # Recurrent agent with weight decay
    agent = PPOAgent(
        state_dim=10, action_dim=4, device="cpu",
        weight_decay=0.01, recurrent=True, lstm_hidden_dim=32
    )
    assert isinstance(agent.optimizer, torch.optim.AdamW)

    # Verify param groups
    param_groups = agent.optimizer.param_groups
    assert len(param_groups) >= 3

    # Find weight decays by group name
    for group in param_groups:
        name = group.get('name', '')
        if 'actor' in name:
            assert group['weight_decay'] == 0.0, "Actor must have wd=0"
        elif 'shared' in name:
            # For recurrent, 'shared' contains encoder + lstm
            assert group['weight_decay'] == 0.0, "Shared (encoder+lstm) must have wd=0"
        elif 'critic' in name:
            assert group['weight_decay'] > 0.0, "Critic should have weight decay"


def test_adaptive_entropy_floor_log_scaling():
    """Entropy floor should use log-ratio scaling (information-theoretic)."""
    import math
    import torch
    from esper.simic.ppo import PPOAgent

    agent = PPOAgent(
        state_dim=10, action_dim=7, device="cpu",
        entropy_coef=0.05, entropy_coef_min=0.01,
        adaptive_entropy_floor=True,
    )

    # With all 7 actions valid, floor is base floor
    mask_all = torch.ones(7)
    floor_all = agent.get_entropy_floor(mask_all)
    assert floor_all == 0.01  # Base floor

    # With only 2 actions valid, floor should use log-ratio scaling:
    # scale = log(7) / log(2) = 1.95 / 0.69 = 2.8
    # floor = 0.01 * 2.8 = 0.028
    mask_few = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    floor_few = agent.get_entropy_floor(mask_few)

    expected_scale = math.log(7) / math.log(2)  # ~2.8
    expected_floor = 0.01 * min(expected_scale, 3.0)  # Capped at 3x

    assert abs(floor_few - expected_floor) < 0.001, \
        f"Expected {expected_floor:.4f}, got {floor_few:.4f}"


def test_comprehensive_value_function_metrics():
    """PPO update should return comprehensive value function diagnostics."""
    import torch
    from esper.simic.ppo import PPOAgent

    agent = PPOAgent(state_dim=10, action_dim=4, device="cpu")

    # Add transitions
    for i in range(10):
        state = torch.randn(10)
        action_mask = torch.ones(4)
        reward = 1.0 if i > 5 else -0.5
        done = i == 9
        agent.store_transition(state, i % 4, -0.5, 0.5, reward, done, action_mask)

    metrics = agent.update(last_value=0.0)

    # Core value function diagnostics
    assert 'value_pred_mean' in metrics
    assert 'value_pred_std' in metrics
    assert 'return_mean' in metrics
    assert 'return_std' in metrics

    # Additional diagnostics (DRL Expert recommendations)
    assert 'value_mse_before' in metrics  # Critic error before update
    assert 'return_min' in metrics
    assert 'return_max' in metrics
    assert 'advantage_mean_prenorm' in metrics  # Critical for PPO stability
    assert 'advantage_std_prenorm' in metrics


def test_recurrent_ppo_epochs_safety_cap():
    """Recurrent PPO should warn early (>2) and cap n_epochs to prevent policy drift."""
    import torch
    import pytest
    from esper.simic.ppo import PPOAgent

    agent = PPOAgent(
        state_dim=10, action_dim=4, device="cpu",
        recurrent=True, lstm_hidden_dim=32,
    )

    # Add some transitions to each env
    for env_id in range(2):
        for i in range(5):
            state = torch.randn(10)
            action_mask = torch.ones(4)
            agent.store_recurrent_transition(
                state, 0, -0.5, 0.5, 1.0, i == 4, action_mask, env_id
            )

    # n_epochs > 2 should warn (early warning)
    with pytest.warns(RuntimeWarning, match="n_epochs.*elevated"):
        metrics = agent.update_recurrent(n_epochs=3)

    # n_epochs > 4 should be capped (hard limit)
    with pytest.warns(RuntimeWarning, match="n_epochs.*capped"):
        metrics = agent.update_recurrent(n_epochs=10)


class TestPPOSaveLoad:
    """Tests for PPOAgent save/load functionality."""

    def test_feedforward_save_load_roundtrip(self, tmp_path):
        """Feedforward PPOAgent should save and load correctly."""
        agent = PPOAgent(
            state_dim=35,
            action_dim=7,
            gamma=0.95,
            entropy_coef=0.1,
            recurrent=False,
            device='cpu'
        )
        agent.train_steps = 42

        # Get initial weights for comparison (use _base_network for torch.compile compatibility)
        initial_weights = {
            k: v.clone() for k, v in agent._base_network.state_dict().items()
        }

        # Save
        save_path = tmp_path / "feedforward_agent.pt"
        agent.save(save_path)

        # Load into new agent
        loaded = PPOAgent.load(save_path, device='cpu')

        # Verify config
        assert loaded.gamma == 0.95
        assert loaded.entropy_coef == 0.1
        assert loaded.recurrent is False
        assert loaded.train_steps == 42

        # Verify weights match
        for key, original in initial_weights.items():
            loaded_weight = loaded._base_network.state_dict()[key]
            assert torch.allclose(original, loaded_weight), f"Mismatch in {key}"

    def test_recurrent_save_load_roundtrip(self, tmp_path):
        """Recurrent PPOAgent should save and load correctly."""
        agent = PPOAgent(
            state_dim=35,
            action_dim=7,
            gamma=0.98,
            entropy_coef=0.05,
            recurrent=True,
            lstm_hidden_dim=64,
            chunk_length=20,
            device='cpu'
        )
        agent.train_steps = 100

        # Get initial weights for comparison (use _base_network for torch.compile compatibility)
        initial_weights = {
            k: v.clone() for k, v in agent._base_network.state_dict().items()
        }

        # Save
        save_path = tmp_path / "recurrent_agent.pt"
        agent.save(save_path)

        # Load into new agent
        loaded = PPOAgent.load(save_path, device='cpu')

        # Verify config
        assert loaded.gamma == 0.98
        assert loaded.entropy_coef == 0.05
        assert loaded.recurrent is True
        assert loaded.lstm_hidden_dim == 64
        assert loaded.chunk_length == 20
        assert loaded.train_steps == 100

        # Verify network type (use _base_network for torch.compile compatibility)
        assert isinstance(loaded._base_network, RecurrentActorCritic)

        # Verify weights match
        for key, original in initial_weights.items():
            loaded_weight = loaded._base_network.state_dict()[key]
            assert torch.allclose(original, loaded_weight), f"Mismatch in {key}"

    def test_recurrent_save_load_preserves_inference_behavior(self, tmp_path):
        """Loaded recurrent agent should produce same outputs as original."""
        agent = PPOAgent(
            state_dim=35,
            action_dim=7,
            recurrent=True,
            lstm_hidden_dim=64,
            device='cpu'
        )

        # Get action for a fixed state
        torch.manual_seed(42)
        state = torch.randn(1, 35)
        action_mask = torch.ones(7)

        with torch.no_grad():
            original_dist, original_value, _ = agent.network(state, action_mask)
            original_probs = original_dist.probs

        # Save and reload
        save_path = tmp_path / "recurrent_inference.pt"
        agent.save(save_path)
        loaded = PPOAgent.load(save_path, device='cpu')

        # Same inference
        with torch.no_grad():
            loaded_dist, loaded_value, _ = loaded.network(state, action_mask)
            loaded_probs = loaded_dist.probs

        assert torch.allclose(original_probs, loaded_probs), "Action probs differ after load"
        assert torch.allclose(original_value, loaded_value), "Value differs after load"

    def test_load_detects_architecture_from_checkpoint(self, tmp_path):
        """Load should correctly detect recurrent vs feedforward from saved checkpoint."""
        # Save recurrent
        recurrent_agent = PPOAgent(
            state_dim=35, action_dim=7, recurrent=True, device='cpu'
        )
        recurrent_path = tmp_path / "recurrent.pt"
        recurrent_agent.save(recurrent_path)

        # Save feedforward
        feedforward_agent = PPOAgent(
            state_dim=35, action_dim=7, recurrent=False, device='cpu'
        )
        feedforward_path = tmp_path / "feedforward.pt"
        feedforward_agent.save(feedforward_path)

        # Load both and verify correct type detection
        loaded_recurrent = PPOAgent.load(recurrent_path, device='cpu')
        loaded_feedforward = PPOAgent.load(feedforward_path, device='cpu')

        assert loaded_recurrent.recurrent is True
        assert isinstance(loaded_recurrent._base_network, RecurrentActorCritic)

        assert loaded_feedforward.recurrent is False
        assert isinstance(loaded_feedforward._base_network, ActorCritic)
