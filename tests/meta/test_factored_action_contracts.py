"""Contract tests for the factored action architecture.

The unified architecture always uses factored actions with per-head masks.
These tests verify the action mask computation and batched action selection.
"""

import torch
from esper.tamiyo.policy.features import MULTISLOT_FEATURE_SIZE

from esper.simic.agent import PPOAgent
from esper.tamiyo.policy.factory import create_policy
from esper.tamiyo.policy.action_masks import compute_action_masks
from esper.leyline import (
    NUM_ALPHA_CURVES,
    NUM_ALPHA_SPEEDS,
    NUM_ALPHA_TARGETS,
    NUM_BLUEPRINTS,
    NUM_OPS,
    NUM_STYLES,
    NUM_TEMPO,
    SlotConfig,
)


class TestFactoredActionMasksInVectorized:
    """Test that factored action masks work correctly in vectorized context."""

    def test_compute_action_masks_returns_dict_tensors(self):
        """compute_action_masks should return dict of boolean tensors."""
        slot_config = SlotConfig.default()
        slot_states = {"r0c0": None}  # Empty slot (canonical ID)

        masks = compute_action_masks(
            slot_states=slot_states,
            enabled_slots=["r0c0"],
            total_seeds=0,
            max_seeds=0,
            slot_config=slot_config,
            device=torch.device("cpu"),
        )

        assert isinstance(masks, dict)
        assert set(masks.keys()) == {
            "slot",
            "blueprint",
            "style",
            "tempo",
            "alpha_target",
            "alpha_speed",
            "alpha_curve",
            "op",
        }
        assert masks["slot"].shape == (slot_config.num_slots,)
        assert masks["blueprint"].shape == (NUM_BLUEPRINTS,)
        assert masks["style"].shape == (NUM_STYLES,)
        assert masks["tempo"].shape == (NUM_TEMPO,)
        assert masks["alpha_target"].shape == (NUM_ALPHA_TARGETS,)
        assert masks["alpha_speed"].shape == (NUM_ALPHA_SPEEDS,)
        assert masks["alpha_curve"].shape == (NUM_ALPHA_CURVES,)
        assert masks["op"].shape == (NUM_OPS,)
        assert masks["slot"].dtype == torch.bool

    def test_factored_masks_batch_stacking(self):
        """Test that factored masks can be batched correctly for vectorized envs."""
        n_envs = 4
        slot_config = SlotConfig.default()
        slot_ids = slot_config.slot_ids  # Canonical IDs: r0c0, r0c1, r0c2

        # Simulate masks from multiple environments
        all_masks = []
        for env_idx in range(n_envs):
            # Each env has its own slot state, but all use same enabled slots
            slot_states = {slot_id: None for slot_id in slot_ids}
            masks = compute_action_masks(
                slot_states=slot_states,
                enabled_slots=list(slot_ids),
                total_seeds=0,
                max_seeds=0,
                slot_config=slot_config,
                device=torch.device("cpu"),
            )
            all_masks.append(masks)

        # Stack into batch format: dict of [n_envs, head_dim] tensors
        batched_masks = {
            key: torch.stack([m[key] for m in all_masks])
            for key in all_masks[0].keys()
        }

        assert batched_masks["slot"].shape == (n_envs, slot_config.num_slots)
        assert batched_masks["blueprint"].shape == (n_envs, NUM_BLUEPRINTS)
        assert batched_masks["style"].shape == (n_envs, NUM_STYLES)
        assert batched_masks["tempo"].shape == (n_envs, NUM_TEMPO)
        assert batched_masks["alpha_target"].shape == (n_envs, NUM_ALPHA_TARGETS)
        assert batched_masks["alpha_speed"].shape == (n_envs, NUM_ALPHA_SPEEDS)
        assert batched_masks["alpha_curve"].shape == (n_envs, NUM_ALPHA_CURVES)
        assert batched_masks["op"].shape == (n_envs, NUM_OPS)


class TestPPOAgentFactoredInVectorized:
    """Test PPOAgent factored mode in vectorized context."""

    def test_factored_agent_batched_action_selection(self):
        """PPOAgent should handle batched action selection with per-head masks."""
        n_envs = 4
        slot_config = SlotConfig.default()
        state_dim = MULTISLOT_FEATURE_SIZE

        policy = create_policy(
            policy_type="lstm",
            state_dim=state_dim,
            slot_config=slot_config,
            device="cpu",
            compile_mode="off",
        )
        agent = PPOAgent(
            policy=policy,
            slot_config=slot_config,
            device="cpu",
        )

        # Batched states
        states = torch.randn(n_envs, state_dim)
        blueprint_indices = torch.full((n_envs, slot_config.num_slots), -1, dtype=torch.long)

        # Batched masks (dict of [n_envs, head_dim] tensors)
        masks = {
            "slot": torch.ones(n_envs, slot_config.num_slots, dtype=torch.bool),
            "blueprint": torch.ones(n_envs, NUM_BLUEPRINTS, dtype=torch.bool),
            "style": torch.ones(n_envs, NUM_STYLES, dtype=torch.bool),
            "tempo": torch.ones(n_envs, NUM_TEMPO, dtype=torch.bool),
            "alpha_target": torch.ones(n_envs, NUM_ALPHA_TARGETS, dtype=torch.bool),
            "alpha_speed": torch.ones(n_envs, NUM_ALPHA_SPEEDS, dtype=torch.bool),
            "alpha_curve": torch.ones(n_envs, NUM_ALPHA_CURVES, dtype=torch.bool),
            "op": torch.ones(n_envs, NUM_OPS, dtype=torch.bool),
        }

        # Get batched actions via network.get_action
        with torch.no_grad():
            result = agent.policy.network.get_action(
                states,
                blueprint_indices,
                slot_mask=masks["slot"],
                blueprint_mask=masks["blueprint"],
                style_mask=masks["style"],
                tempo_mask=masks["tempo"],
                alpha_target_mask=masks["alpha_target"],
                alpha_speed_mask=masks["alpha_speed"],
                alpha_curve_mask=masks["alpha_curve"],
                op_mask=masks["op"],
                deterministic=False,
            )

        # Actions should be dict of tensors
        assert isinstance(result.actions, dict)
        assert set(result.actions.keys()) == {
            "slot",
            "blueprint",
            "style",
            "tempo",
            "alpha_target",
            "alpha_speed",
            "alpha_curve",
            "op",
        }
        assert result.actions["slot"].shape == (n_envs,)
        assert result.actions["op"].shape == (n_envs,)

        # Log probs should be dict and values should be [n_envs]
        assert isinstance(result.log_probs, dict)
        assert result.values.shape == (n_envs,)

    def test_rollout_buffer_stores_factored_transitions(self):
        """TamiyoRolloutBuffer should store factored transitions from multiple envs."""
        n_envs = 4
        slot_config = SlotConfig.default()
        state_dim = MULTISLOT_FEATURE_SIZE

        policy = create_policy(
            policy_type="lstm",
            state_dim=state_dim,
            slot_config=slot_config,
            device="cpu",
            compile_mode="off",
        )
        agent = PPOAgent(
            policy=policy,
            slot_config=slot_config,
            device="cpu",
            num_envs=n_envs,
            max_steps_per_env=10,
        )
        blueprint_indices = torch.full((slot_config.num_slots,), -1, dtype=torch.long)

        # Start episodes for all envs
        for env_idx in range(n_envs):
            agent.buffer.start_episode(env_id=env_idx)

        # Store transitions from each env
        for env_idx in range(n_envs):
            state = torch.randn(state_dim)
            # Hidden state: [num_layers, hidden_dim] (batch dim squeezed in add())
            hidden_h = torch.randn(1, agent.lstm_hidden_dim)
            hidden_c = torch.randn(1, agent.lstm_hidden_dim)
            agent.buffer.add(
                env_id=env_idx,
                state=state,
                blueprint_indices=blueprint_indices,
                slot_action=env_idx % slot_config.num_slots,
                blueprint_action=env_idx % NUM_BLUEPRINTS,
                style_action=env_idx % NUM_STYLES,
                tempo_action=env_idx % NUM_TEMPO,
                alpha_target_action=env_idx % NUM_ALPHA_TARGETS,
                alpha_speed_action=env_idx % NUM_ALPHA_SPEEDS,
                alpha_curve_action=env_idx % NUM_ALPHA_CURVES,
                op_action=env_idx % NUM_OPS,
                slot_log_prob=-0.5,
                blueprint_log_prob=-0.5,
                style_log_prob=-0.5,
                tempo_log_prob=-0.5,
                alpha_target_log_prob=-0.5,
                alpha_speed_log_prob=-0.5,
                alpha_curve_log_prob=-0.5,
                op_log_prob=-0.5,
                value=0.5,
                reward=1.0,
                done=False,
                slot_mask=torch.ones(slot_config.num_slots, dtype=torch.bool),
                blueprint_mask=torch.ones(NUM_BLUEPRINTS, dtype=torch.bool),
                style_mask=torch.ones(NUM_STYLES, dtype=torch.bool),
                tempo_mask=torch.ones(NUM_TEMPO, dtype=torch.bool),
                alpha_target_mask=torch.ones(NUM_ALPHA_TARGETS, dtype=torch.bool),
                alpha_speed_mask=torch.ones(NUM_ALPHA_SPEEDS, dtype=torch.bool),
                alpha_curve_mask=torch.ones(NUM_ALPHA_CURVES, dtype=torch.bool),
                op_mask=torch.ones(NUM_OPS, dtype=torch.bool),
                hidden_h=hidden_h,
                hidden_c=hidden_c,
            )

        # Buffer should have transitions
        assert len(agent.buffer) == n_envs


class TestVectorizedFactoredDefault:
    """Test that train_ppo_vectorized uses factored mode by default."""

    def test_train_ppo_vectorized_signature(self):
        """train_ppo_vectorized should have the expected parameters."""
        from esper.simic.training.vectorized import train_ppo_vectorized
        import inspect

        sig = inspect.signature(train_ppo_vectorized)
        params = sig.parameters

        # Core parameters should exist
        assert "n_episodes" in params
        assert "n_envs" in params
        assert "max_epochs" in params
        assert "slots" in params

        # Removed parameters should NOT exist
        assert "recurrent" not in params, "recurrent parameter should be removed"
        # factored was never a parameter in train_ppo_vectorized

    def test_ppo_agent_no_factored_parameter(self):
        """PPOAgent should not have a factored parameter (unified architecture)."""
        import inspect

        sig = inspect.signature(PPOAgent.__init__)
        params = sig.parameters

        assert "factored" not in params, "factored parameter should be removed"
        assert "recurrent" not in params, "recurrent parameter should be removed"
