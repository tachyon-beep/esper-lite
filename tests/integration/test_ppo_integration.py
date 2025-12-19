"""Integration tests for PPO algorithm with factored action space.

Tests that PPO components work together correctly:
- Feature extraction produces compatible tensors
- Forward pass works with extracted features and per-head masks
- Action sampling produces valid factored actions
"""

import torch

from esper.simic.agent import PPOAgent, signals_to_features
from esper.tamiyo.policy.features import MULTISLOT_FEATURE_SIZE
from esper.leyline import TrainingSignals, SeedTelemetry
from esper.leyline.factored_actions import NUM_BLUEPRINTS, NUM_BLENDS, NUM_TEMPO, NUM_OPS


def _create_all_valid_masks(batch_size: int = 1) -> dict[str, torch.Tensor]:
    """Create all-valid per-head action masks for testing."""
    return {
        "slot": torch.ones(batch_size, 3, dtype=torch.bool),              # 3 slots
        "blueprint": torch.ones(batch_size, NUM_BLUEPRINTS, dtype=torch.bool),
        "blend": torch.ones(batch_size, NUM_BLENDS, dtype=torch.bool),
        "tempo": torch.ones(batch_size, NUM_TEMPO, dtype=torch.bool),
        "op": torch.ones(batch_size, NUM_OPS, dtype=torch.bool),
    }


class TestPPOFeatureCompatibility:
    """Test that signals_to_features output is compatible with PPOAgent."""

    def test_features_without_telemetry_compatible_with_agent(self):
        """Features without telemetry should be compatible with network."""
        # Create signals
        signals = TrainingSignals()
        signals.metrics.epoch = 5
        signals.metrics.val_accuracy = 65.0
        signals.metrics.train_loss = 1.5
        signals.metrics.val_loss = 1.7

        # Extract features
        features = signals_to_features(signals, slot_reports={}, use_telemetry=False, slots=["r0c1"])
        assert len(features) == MULTISLOT_FEATURE_SIZE, f"Expected {MULTISLOT_FEATURE_SIZE} features, got {len(features)}"

        # Create PPO agent with matching dimensions
        agent = PPOAgent(state_dim=len(features), device='cpu', compile_network=False)

        # Convert to tensor
        state_tensor = torch.tensor([features], dtype=torch.float32)
        masks = _create_all_valid_masks()

        # Should work without errors
        with torch.no_grad():
            result = agent.network.get_action(
                state_tensor,
                slot_mask=masks["slot"],
                blueprint_mask=masks["blueprint"],
                blend_mask=masks["blend"],
                op_mask=masks["op"],
            )

        # Check outputs
        assert "op" in result.actions, "Should have op action"
        assert "slot" in result.actions, "Should have slot action"
        assert result.values.shape == (1,), "Value should be scalar per batch item"

    def test_features_with_telemetry_compatible_with_agent(self):
        """Features with telemetry should be compatible with network."""
        # Create signals (no active seed, so telemetry will be zero-padded)
        signals = TrainingSignals()
        signals.metrics.epoch = 10
        signals.metrics.val_accuracy = 70.0

        # Extract features with telemetry
        features = signals_to_features(signals, slot_reports={}, use_telemetry=True, slots=["r0c1"])
        expected_dim = MULTISLOT_FEATURE_SIZE + SeedTelemetry.feature_dim() * 3
        assert len(features) == expected_dim, f"Expected {expected_dim} features, got {len(features)}"

        # Create PPO agent with matching dimensions
        agent = PPOAgent(state_dim=len(features), device='cpu', compile_network=False)

        # Convert to tensor
        state_tensor = torch.tensor([features], dtype=torch.float32)
        masks = _create_all_valid_masks()

        # Should work without errors
        with torch.no_grad():
            result = agent.network.get_action(
                state_tensor,
                slot_mask=masks["slot"],
                blueprint_mask=masks["blueprint"],
                blend_mask=masks["blend"],
                op_mask=masks["op"],
            )

        assert result.values.shape == (1,)

    def test_batch_compatibility(self):
        """Test that batched features work with agent."""
        batch_size = 16
        all_features = []

        for i in range(batch_size):
            signals = TrainingSignals()
            signals.metrics.epoch = i
            signals.metrics.val_accuracy = 50.0 + i
            features = signals_to_features(signals, slot_reports={}, use_telemetry=False, slots=["r0c1"])
            all_features.append(features)

        # Stack into batch
        batch_tensor = torch.tensor(all_features, dtype=torch.float32)
        assert batch_tensor.shape == (batch_size, MULTISLOT_FEATURE_SIZE)

        # Create agent
        agent = PPOAgent(state_dim=MULTISLOT_FEATURE_SIZE, device='cpu', compile_network=False)
        masks = _create_all_valid_masks(batch_size)

        # Should handle batch without errors
        with torch.no_grad():
            result = agent.network.get_action(
                batch_tensor,
                slot_mask=masks["slot"],
                blueprint_mask=masks["blueprint"],
                blend_mask=masks["blend"],
                tempo_mask=masks["tempo"],
                op_mask=masks["op"],
            )

        assert result.values.shape == (batch_size,)
        for key in ["slot", "blueprint", "blend", "tempo", "op"]:
            assert result.actions[key].shape == (batch_size,), f"{key} actions should have batch dim"


class TestPPOForwardPass:
    """Test that PPOAgent forward pass works correctly."""

    def test_forward_pass_returns_valid_outputs(self):
        """Forward pass should return valid factored outputs."""
        agent = PPOAgent(state_dim=MULTISLOT_FEATURE_SIZE, device='cpu', compile_network=False)
        state = torch.randn(1, MULTISLOT_FEATURE_SIZE)
        masks = _create_all_valid_masks()

        with torch.no_grad():
            output = agent.network.forward(
                state.unsqueeze(1),  # Add seq dim: [batch, seq, dim]
                slot_mask=masks["slot"].unsqueeze(1),
                blueprint_mask=masks["blueprint"].unsqueeze(1),
                blend_mask=masks["blend"].unsqueeze(1),
                tempo_mask=masks["tempo"].unsqueeze(1),
                op_mask=masks["op"].unsqueeze(1),
            )

        # Check all heads present
        assert "slot_logits" in output
        assert "blueprint_logits" in output
        assert "blend_logits" in output
        assert "tempo_logits" in output
        assert "op_logits" in output
        assert "value" in output
        assert "hidden" in output

    def test_forward_pass_value_is_scalar(self):
        """Value function should output scalar per state."""
        agent = PPOAgent(state_dim=50, device='cpu', compile_network=False)
        batch_size = 8
        states = torch.randn(batch_size, 50)
        masks = _create_all_valid_masks(batch_size)

        with torch.no_grad():
            result = agent.network.get_action(
                states,
                slot_mask=masks["slot"],
                blueprint_mask=masks["blueprint"],
                blend_mask=masks["blend"],
                tempo_mask=masks["tempo"],
                op_mask=masks["op"],
            )

        assert result.values.shape == (batch_size,), f"Expected shape ({batch_size},), got {result.values.shape}"

    def test_forward_pass_deterministic(self):
        """Same input should produce same output in deterministic mode."""
        agent = PPOAgent(state_dim=50, device='cpu', compile_network=False)
        state = torch.randn(1, 50)
        masks = _create_all_valid_masks()

        agent.network.eval()

        with torch.no_grad():
            result1 = agent.network.get_action(
                state, deterministic=True,
                slot_mask=masks["slot"],
                blueprint_mask=masks["blueprint"],
                blend_mask=masks["blend"],
                tempo_mask=masks["tempo"],
                op_mask=masks["op"],
            )
            result2 = agent.network.get_action(
                state, deterministic=True,
                slot_mask=masks["slot"],
                blueprint_mask=masks["blueprint"],
                blend_mask=masks["blend"],
                tempo_mask=masks["tempo"],
                op_mask=masks["op"],
            )

        # Actions should be identical
        for key in ["slot", "blueprint", "blend", "tempo", "op"]:
            assert torch.equal(result1.actions[key], result2.actions[key]), f"{key} actions should be deterministic"
        assert torch.allclose(result1.values, result2.values), "Values should be deterministic"


class TestPPOActionSampling:
    """Test that action sampling works correctly."""

    def test_get_action_returns_valid_factored_actions(self):
        """get_action should return valid action indices for each head."""
        agent = PPOAgent(state_dim=50, device='cpu', compile_network=False)
        state = torch.randn(1, 50)
        masks = _create_all_valid_masks()

        result = agent.network.get_action(
            state,
            slot_mask=masks["slot"],
            blueprint_mask=masks["blueprint"],
            blend_mask=masks["blend"],
            tempo_mask=masks["tempo"],
            op_mask=masks["op"],
            deterministic=False,
        )

        # Check action ranges
        assert 0 <= result.actions["slot"].item() < 3, "Slot action out of range"
        assert 0 <= result.actions["blueprint"].item() < NUM_BLUEPRINTS, "Blueprint action out of range"
        assert 0 <= result.actions["blend"].item() < NUM_BLENDS, "Blend action out of range"
        assert 0 <= result.actions["tempo"].item() < NUM_TEMPO, "Tempo action out of range"
        assert 0 <= result.actions["op"].item() < NUM_OPS, "Op action out of range"

        # Log probs should be negative
        for key in ["slot", "blueprint", "blend", "tempo", "op"]:
            assert result.log_probs[key].item() <= 0, f"{key} log prob should be <= 0"

    def test_deterministic_action_selects_argmax(self):
        """Deterministic action should select highest probability action."""
        agent = PPOAgent(state_dim=50, device='cpu', compile_network=False)
        state = torch.randn(1, 50)
        masks = _create_all_valid_masks()

        # Get deterministic action multiple times
        all_actions = {key: [] for key in ["slot", "blueprint", "blend", "tempo", "op"]}
        for _ in range(10):
            result = agent.network.get_action(
                state, deterministic=True,
                slot_mask=masks["slot"],
                blueprint_mask=masks["blueprint"],
                blend_mask=masks["blend"],
                tempo_mask=masks["tempo"],
                op_mask=masks["op"],
            )
            for key in all_actions:
                all_actions[key].append(result.actions[key].item())

        # All should be the same
        for key, action_list in all_actions.items():
            assert len(set(action_list)) == 1, f"Deterministic {key} action should be consistent"

    def test_stochastic_action_samples_from_distribution(self):
        """Stochastic action should sample from distribution."""
        agent = PPOAgent(state_dim=50, device='cpu', compile_network=False)
        state = torch.randn(1, 50)
        masks = _create_all_valid_masks()

        # Sample multiple times
        all_op_actions = []
        for _ in range(100):
            result = agent.network.get_action(
                state, deterministic=False,
                slot_mask=masks["slot"],
                blueprint_mask=masks["blueprint"],
                blend_mask=masks["blend"],
                tempo_mask=masks["tempo"],
                op_mask=masks["op"],
            )
            all_op_actions.append(result.actions["op"].item())

        # Should have some variety (with high probability)
        unique_actions = set(all_op_actions)
        assert len(unique_actions) > 1, \
            "Stochastic sampling should produce variety (got only one action in 100 samples)"


class TestPPOEndToEnd:
    """End-to-end integration tests."""

    def test_signals_to_action_pipeline(self):
        """Complete pipeline: TrainingSignals -> features -> factored actions."""
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
        features = signals_to_features(signals, slot_reports={}, use_telemetry=False, slots=["r0c1"])

        # Create agent
        agent = PPOAgent(state_dim=len(features), device='cpu', compile_network=False)

        # Get action
        state_tensor = torch.tensor([features], dtype=torch.float32)
        masks = _create_all_valid_masks()
        result = agent.network.get_action(
            state_tensor,
            slot_mask=masks["slot"],
            blueprint_mask=masks["blueprint"],
            blend_mask=masks["blend"],
            tempo_mask=masks["tempo"],
            op_mask=masks["op"],
            deterministic=False,
        )

        # All outputs should be valid
        assert 0 <= result.actions["op"].item() < NUM_OPS, "Invalid op action"
        for key in result.log_probs:
            assert result.log_probs[key].item() <= 0, f"Invalid {key} log prob"
        assert result.values.shape == (1,), "Invalid value shape"

    def test_telemetry_pipeline(self):
        """Pipeline with telemetry features."""
        signals = TrainingSignals()
        signals.metrics.epoch = 20
        signals.metrics.val_accuracy = 75.0

        # Extract features with telemetry (will be zero-padded)
        features = signals_to_features(signals, slot_reports={}, use_telemetry=True, slots=["r0c1"])
        expected_dim = MULTISLOT_FEATURE_SIZE + SeedTelemetry.feature_dim() * 3
        assert len(features) == expected_dim, f"Should have {expected_dim} features with telemetry"

        # Create agent
        agent = PPOAgent(state_dim=expected_dim, device='cpu', compile_network=False)

        # Get action
        state_tensor = torch.tensor([features], dtype=torch.float32)
        masks = _create_all_valid_masks()
        result = agent.network.get_action(
            state_tensor,
            slot_mask=masks["slot"],
            blueprint_mask=masks["blueprint"],
            blend_mask=masks["blend"],
            tempo_mask=masks["tempo"],
            op_mask=masks["op"],
        )

        assert 0 <= result.actions["op"].item() < NUM_OPS
        assert result.values.shape == (1,)

    def test_hidden_state_continuity(self):
        """LSTM hidden states should be maintained across calls."""
        agent = PPOAgent(state_dim=50, device='cpu', compile_network=False)
        state = torch.randn(1, 50)
        masks = _create_all_valid_masks()

        # First call - no hidden state
        result1 = agent.network.get_action(
            state,
            slot_mask=masks["slot"],
            blueprint_mask=masks["blueprint"],
            blend_mask=masks["blend"],
            tempo_mask=masks["tempo"],
            op_mask=masks["op"],
        )

        # Second call - pass hidden state from first call
        result2 = agent.network.get_action(
            state,
            hidden=result1.hidden,
            slot_mask=masks["slot"],
            blueprint_mask=masks["blueprint"],
            blend_mask=masks["blend"],
            tempo_mask=masks["tempo"],
            op_mask=masks["op"],
        )

        # Hidden states should be valid tensors
        assert result1.hidden is not None
        assert result2.hidden is not None
        assert len(result1.hidden) == 2, "Hidden should be (h, c) tuple"
        assert len(result2.hidden) == 2, "Hidden should be (h, c) tuple"

        # Second hidden should differ from first (LSTM updates state)
        h1, c1 = result1.hidden
        h2, c2 = result2.hidden
        assert not torch.allclose(h1, h2) or not torch.allclose(c1, c2), \
            "Hidden state should change after processing input"
