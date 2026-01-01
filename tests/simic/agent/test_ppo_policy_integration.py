"""Cross-stack tests for PPO with factored action space.

Tests that PPO components work together correctly:
- Feature extraction produces compatible tensors
- Forward pass works with extracted features and per-head masks
- Action sampling produces valid factored actions
"""

import torch

from esper.simic.agent import PPOAgent
from tests.helpers import create_all_valid_masks
from esper.tamiyo.policy.features import batch_obs_to_features
from esper.tamiyo.policy.factory import create_policy
from esper.leyline import (
    NUM_BLUEPRINTS,
    NUM_OPS,
)
from esper.leyline.slot_config import SlotConfig
from esper.leyline.signals import TrainingSignals, TrainingMetrics
from esper.simic.training.parallel_env_state import ParallelEnvState

MAX_EPOCHS = 100


def _make_mock_training_signals():
    """Create mock TrainingSignals for testing."""
    metrics = TrainingMetrics(
        epoch=10,
        global_step=1000,
        train_loss=1.5,
        val_loss=1.7,
        loss_delta=-0.02,
        train_accuracy=67.0,
        val_accuracy=65.0,
        accuracy_delta=0.5,
        plateau_epochs=2,
        best_val_accuracy=65.0,
        best_val_loss=1.6,
    )

    return TrainingSignals(
        metrics=metrics,
        loss_history=[0.6, 0.55, 0.5, 0.52, 0.5],
        accuracy_history=[65.0, 66.0, 67.0, 68.0, 70.0],
    )


def _make_mock_parallel_env_state():
    """Create minimal mock ParallelEnvState for testing."""
    class MockModel:
        pass

    class MockOptimizer:
        pass

    class MockSignalTracker:
        def reset(self):
            pass

    class MockGovernor:
        def reset(self):
            pass

    return ParallelEnvState(
        model=MockModel(),
        host_optimizer=MockOptimizer(),
        signal_tracker=MockSignalTracker(),
        governor=MockGovernor(),
        last_action_success=True,
        last_action_op=0,
    )


class TestPPOFeatureCompatibility:
    """Test that batch_obs_to_features output is compatible with PPOAgent."""

    def test_features_compatible_with_agent(self):
        """Features should be compatible with network."""
        slot_config = SlotConfig.default()
        device = torch.device("cpu")

        # Create mock inputs for batch_obs_to_features
        batch_signals = [_make_mock_training_signals()]
        batch_slot_reports = [{}]  # No active seeds
        batch_env_states = [_make_mock_parallel_env_state()]

        # Extract features (Obs V3)
        obs, blueprint_indices = batch_obs_to_features(
            batch_signals,
            batch_slot_reports,
            batch_env_states,
            slot_config,
            device,
            max_epochs=MAX_EPOCHS,
        )

        # Obs V3: 23 base + 30 per slot × 3 slots = 113 dims
        assert obs.shape == (1, 113), f"Expected (1, 113), got {obs.shape}"
        assert blueprint_indices.shape == (1, 3), f"Expected (1, 3), got {blueprint_indices.shape}"

        # Create PPO agent with matching dimensions
        policy = create_policy(
            policy_type="lstm",
            state_dim=113,
            slot_config=slot_config,
            device="cpu",
            compile_mode="off",
        )
        agent = PPOAgent(policy=policy, device="cpu")

        # Convert to 2D tensor (add sequence dim)
        state_tensor = obs.unsqueeze(1)  # [1, 1, 113]
        bp_indices = blueprint_indices.unsqueeze(1)  # [1, 1, 3]
        masks = create_all_valid_masks()

        # Should work without errors
        with torch.no_grad():
            result = agent.policy.network.get_action(
                state_tensor,
                bp_indices,
                slot_mask=masks["slot"],
                blueprint_mask=masks["blueprint"],
                style_mask=masks["style"],
                tempo_mask=masks["tempo"],
                alpha_target_mask=masks["alpha_target"],
                alpha_speed_mask=masks["alpha_speed"],
                alpha_curve_mask=masks["alpha_curve"],
                op_mask=masks["op"],
            )

        # Check outputs (network returns GetActionResult with 'actions' plural)
        assert "op" in result.actions, "Should have op action"
        assert "slot" in result.actions, "Should have slot action"
        # get_action squeezes seq dim: values are [batch] not [batch, seq]
        assert result.values.shape == (1,), "Value should be [batch]"


class TestPPOEndToEnd:
    """Test PPO agent end-to-end with real data flow."""

    def test_ppo_agent_can_sample_actions(self):
        """PPO agent should be able to sample valid actions."""
        slot_config = SlotConfig.default()
        device = torch.device("cpu")

        # Create PPO agent
        policy = create_policy(
            policy_type="lstm",
            state_dim=113,
            slot_config=slot_config,
            device="cpu",
            compile_mode="off",
        )
        agent = PPOAgent(policy=policy, device="cpu")

        # Create mock observations
        batch_signals = [_make_mock_training_signals()]
        batch_slot_reports = [{}]
        batch_env_states = [_make_mock_parallel_env_state()]

        obs, blueprint_indices = batch_obs_to_features(
            batch_signals,
            batch_slot_reports,
            batch_env_states,
            slot_config,
            device,
            max_epochs=MAX_EPOCHS,
        )

        state_tensor = obs.unsqueeze(1)
        bp_indices = blueprint_indices.unsqueeze(1)
        masks = create_all_valid_masks()

        # Sample action
        with torch.no_grad():
            result = agent.policy.get_action(
                state_tensor.squeeze(1),  # Bundle expects [batch, features]
                bp_indices.squeeze(1),
                masks,
                hidden=None,
            )

        # Validate action structure
        assert "op" in result.action
        assert "slot" in result.action
        assert "blueprint" in result.action

        # Validate action ranges
        assert 0 <= result.action["op"].item() < NUM_OPS
        assert 0 <= result.action["slot"].item() < 3
        assert 0 <= result.action["blueprint"].item() < NUM_BLUEPRINTS

    def test_ppo_buffer_integration(self):
        """PPO buffer should accept features and actions."""
        slot_config = SlotConfig.default()
        device = torch.device("cpu")

        policy = create_policy(
            policy_type="lstm",
            state_dim=113,
            slot_config=slot_config,
            device="cpu",
            compile_mode="off",
        )
        agent = PPOAgent(policy=policy, device="cpu", num_envs=1, max_steps_per_env=10)

        # Create observations
        batch_signals = [_make_mock_training_signals()]
        batch_slot_reports = [{}]
        batch_env_states = [_make_mock_parallel_env_state()]

        obs, blueprint_indices = batch_obs_to_features(
            batch_signals,
            batch_slot_reports,
            batch_env_states,
            slot_config,
            device,
            max_epochs=MAX_EPOCHS,
        )

        # Sample action
        masks = create_all_valid_masks()
        with torch.no_grad():
            result = agent.policy.get_action(
                obs,
                blueprint_indices,
                masks,
                hidden=None,
            )

        # Add to buffer (buffer.add takes individual action/log_prob fields)
        agent.buffer.add(
            env_id=0,
            state=obs[0],
            blueprint_indices=blueprint_indices[0],
            slot_action=result.action["slot"].item(),
            blueprint_action=result.action["blueprint"].item(),
            style_action=result.action["style"].item(),
            tempo_action=result.action["tempo"].item(),
            alpha_target_action=result.action["alpha_target"].item(),
            alpha_speed_action=result.action["alpha_speed"].item(),
            alpha_curve_action=result.action["alpha_curve"].item(),
            op_action=result.action["op"].item(),
            effective_op_action=result.action["op"].item(),
            slot_log_prob=result.log_prob["slot"].item(),
            blueprint_log_prob=result.log_prob["blueprint"].item(),
            style_log_prob=result.log_prob["style"].item(),
            tempo_log_prob=result.log_prob["tempo"].item(),
            alpha_target_log_prob=result.log_prob["alpha_target"].item(),
            alpha_speed_log_prob=result.log_prob["alpha_speed"].item(),
            alpha_curve_log_prob=result.log_prob["alpha_curve"].item(),
            op_log_prob=result.log_prob["op"].item(),
            value=result.value[0].item(),
            reward=1.0,
            done=False,
            hidden_h=result.hidden[0][:, 0, :],
            hidden_c=result.hidden[1][:, 0, :],
            slot_mask=masks["slot"][0],
            blueprint_mask=masks["blueprint"][0],
            style_mask=masks["style"][0],
            tempo_mask=masks["tempo"][0],
            alpha_target_mask=masks["alpha_target"][0],
            alpha_speed_mask=masks["alpha_speed"][0],
            alpha_curve_mask=masks["alpha_curve"][0],
            op_mask=masks["op"][0],
        )

        assert agent.buffer.step_counts[0] == 1, "Buffer should have 1 timestep stored for env 0"
