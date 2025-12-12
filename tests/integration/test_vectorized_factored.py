"""Integration tests for factored action mode in vectorized PPO training."""

import pytest
import torch

from esper.simic.ppo import PPOAgent
from esper.simic.action_masks import compute_action_masks, MaskSeedInfo
from esper.leyline.factored_actions import NUM_SLOTS, NUM_BLUEPRINTS, NUM_BLENDS, NUM_OPS


class TestFactoredActionMasksInVectorized:
    """Test that factored action masks work correctly in vectorized context."""

    def test_compute_action_masks_returns_dict_tensors(self):
        """compute_action_masks should return dict of boolean tensors."""
        slot_states = {"slot_0": None}  # Empty slot

        masks = compute_action_masks(
            slot_states=slot_states,
            target_slot="slot_0",
            total_seeds=0,
            max_seeds=0,
            device=torch.device("cpu"),
        )

        assert isinstance(masks, dict)
        assert set(masks.keys()) == {"slot", "blueprint", "blend", "op"}
        assert masks["slot"].shape == (NUM_SLOTS,)
        assert masks["blueprint"].shape == (NUM_BLUEPRINTS,)
        assert masks["blend"].shape == (NUM_BLENDS,)
        assert masks["op"].shape == (NUM_OPS,)
        assert masks["slot"].dtype == torch.bool

    def test_factored_masks_batch_stacking(self):
        """Test that factored masks can be batched correctly for vectorized envs."""
        n_envs = 4

        # Simulate masks from multiple environments
        all_masks = []
        for env_idx in range(n_envs):
            slot_states = {f"slot_{env_idx}": None}
            masks = compute_action_masks(
                slot_states=slot_states,
                target_slot=f"slot_{env_idx}",
                total_seeds=0,
                max_seeds=0,
                device=torch.device("cpu"),
            )
            all_masks.append(masks)

        # Stack into batch format: dict of [n_envs, head_dim] tensors
        batched_masks = {
            key: torch.stack([m[key] for m in all_masks])
            for key in all_masks[0].keys()
        }

        assert batched_masks["slot"].shape == (n_envs, NUM_SLOTS)
        assert batched_masks["blueprint"].shape == (n_envs, NUM_BLUEPRINTS)
        assert batched_masks["blend"].shape == (n_envs, NUM_BLENDS)
        assert batched_masks["op"].shape == (n_envs, NUM_OPS)


class TestPPOAgentFactoredInVectorized:
    """Test PPOAgent factored mode in vectorized context."""

    def test_factored_agent_batched_action_selection(self):
        """PPOAgent in factored mode should handle batched action selection."""
        n_envs = 4
        state_dim = 50

        agent = PPOAgent(
            state_dim=state_dim,
            factored=True,
            device="cpu",
            compile_network=False,
        )

        # Batched states
        states = torch.randn(n_envs, state_dim)

        # Batched masks (dict of [n_envs, head_dim] tensors)
        masks = {
            "slot": torch.ones(n_envs, NUM_SLOTS, dtype=torch.bool),
            "blueprint": torch.ones(n_envs, NUM_BLUEPRINTS, dtype=torch.bool),
            "blend": torch.ones(n_envs, NUM_BLENDS, dtype=torch.bool),
            "op": torch.ones(n_envs, NUM_OPS, dtype=torch.bool),
        }

        # Get batched actions
        actions, log_probs, values = agent.network.get_action_batch(
            states, masks, deterministic=False
        )

        # Actions should be dict of tensors
        assert isinstance(actions, dict)
        assert set(actions.keys()) == {"slot", "blueprint", "blend", "op"}
        assert actions["slot"].shape == (n_envs,)
        assert actions["op"].shape == (n_envs,)

        # Log probs and values should be [n_envs]
        assert log_probs.shape == (n_envs,)
        assert values.shape == (n_envs,)

    def test_factored_transition_storage_batch(self):
        """PPOAgent should store factored transitions from multiple envs."""
        n_envs = 4
        state_dim = 50

        agent = PPOAgent(
            state_dim=state_dim,
            factored=True,
            device="cpu",
            compile_network=False,
        )

        masks = {
            "slot": torch.ones(NUM_SLOTS, dtype=torch.bool),
            "blueprint": torch.ones(NUM_BLUEPRINTS, dtype=torch.bool),
            "blend": torch.ones(NUM_BLENDS, dtype=torch.bool),
            "op": torch.ones(NUM_OPS, dtype=torch.bool),
        }

        # Store transitions from multiple envs
        for env_idx in range(n_envs):
            state = torch.randn(state_dim)
            action = {
                "slot": env_idx % NUM_SLOTS,
                "blueprint": env_idx % NUM_BLUEPRINTS,
                "blend": env_idx % NUM_BLENDS,
                "op": env_idx % NUM_OPS,
            }
            agent.store_factored_transition(
                state=state,
                action=action,
                log_prob=-1.0,
                value=0.5,
                reward=1.0,
                done=(env_idx == n_envs - 1),
                action_masks=masks,
            )

        assert len(agent.factored_buffer) == n_envs

    def test_factored_update_after_batch(self):
        """PPOAgent should update from factored transitions."""
        n_transitions = 16
        state_dim = 50

        agent = PPOAgent(
            state_dim=state_dim,
            factored=True,
            device="cpu",
            compile_network=False,
            n_epochs=1,  # Fast test
        )

        masks = {
            "slot": torch.ones(NUM_SLOTS, dtype=torch.bool),
            "blueprint": torch.ones(NUM_BLUEPRINTS, dtype=torch.bool),
            "blend": torch.ones(NUM_BLENDS, dtype=torch.bool),
            "op": torch.ones(NUM_OPS, dtype=torch.bool),
        }

        # Fill buffer
        for i in range(n_transitions):
            state = torch.randn(state_dim)
            action = {
                "slot": i % NUM_SLOTS,
                "blueprint": i % NUM_BLUEPRINTS,
                "blend": i % NUM_BLENDS,
                "op": i % NUM_OPS,
            }
            agent.store_factored_transition(
                state=state,
                action=action,
                log_prob=-1.0,
                value=0.5,
                reward=1.0,
                done=(i == n_transitions - 1),
                action_masks=masks,
            )

        # Update should succeed
        metrics = agent.update_factored(last_value=0.0)

        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics
        assert len(agent.factored_buffer) == 0  # Buffer cleared


class TestVectorizedFactoredDefault:
    """Test that train_ppo_vectorized uses factored mode by default."""

    def test_train_ppo_vectorized_uses_factored_by_default(self):
        """train_ppo_vectorized should use factored action space by default."""
        from esper.simic.vectorized import train_ppo_vectorized
        import inspect

        # Factored is now the default and only mode - no factored parameter needed
        sig = inspect.signature(train_ppo_vectorized)
        params = sig.parameters

        # Verify recurrent is disabled by default (not compatible with factored)
        assert "recurrent" in params
        assert params["recurrent"].default is False
