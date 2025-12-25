"""End-to-end tests for dynamic slot configurations.

Tests verify that training works correctly with various slot configurations:
- Different slot counts (1, 5, 9, 25)
- Multiple environments with independent mask computation
- Slot saturation and recovery after culling
"""

import torch

from esper.leyline.slot_config import SlotConfig
from esper.leyline.stages import SeedStage
from esper.leyline.factored_actions import (
    LifecycleOp,
    NUM_ALPHA_CURVES,
    NUM_ALPHA_SPEEDS,
    NUM_ALPHA_TARGETS,
    NUM_BLUEPRINTS,
    NUM_OPS,
    NUM_STYLES,
    NUM_TEMPO,
)
from esper.simic.agent import PPOAgent
from esper.tamiyo.policy.factory import create_policy
from esper.tamiyo.policy.features import get_feature_size
from esper.tamiyo.policy.action_masks import compute_action_masks, MaskSeedInfo


class TestTrainingWithDifferentSlotCounts:
    """Tests that PPO training components work with various slot configurations."""

    def test_agent_with_1_slot(self):
        """PPOAgent should work with single slot configuration."""
        config = SlotConfig(slot_ids=("r0c1",))
        state_dim = get_feature_size(config)

        policy = create_policy(
            policy_type="lstm",
            state_dim=state_dim,
            slot_config=config,
            device="cpu",
            compile_mode="off",
        )
        agent = PPOAgent(
            policy=policy,
            slot_config=config,
            device="cpu",
            num_envs=1,
            max_steps_per_env=10,
        )

        assert agent.slot_config.num_slots == 1
        assert agent.policy.network.num_slots == 1

        # Verify forward pass works
        states = torch.randn(1, state_dim)
        masks = {
            "slot": torch.ones(1, 1, dtype=torch.bool),
            "blueprint": torch.ones(1, NUM_BLUEPRINTS, dtype=torch.bool),
            "style": torch.ones(1, NUM_STYLES, dtype=torch.bool),
            "tempo": torch.ones(1, NUM_TEMPO, dtype=torch.bool),
            "alpha_target": torch.ones(1, NUM_ALPHA_TARGETS, dtype=torch.bool),
            "alpha_speed": torch.ones(1, NUM_ALPHA_SPEEDS, dtype=torch.bool),
            "alpha_curve": torch.ones(1, NUM_ALPHA_CURVES, dtype=torch.bool),
            "op": torch.ones(1, NUM_OPS, dtype=torch.bool),
        }

        with torch.no_grad():
            result = agent.policy.network.get_action(
                states,
                slot_mask=masks["slot"],
                blueprint_mask=masks["blueprint"],
                style_mask=masks["style"],
                tempo_mask=masks["tempo"],
                alpha_target_mask=masks["alpha_target"],
                alpha_speed_mask=masks["alpha_speed"],
                alpha_curve_mask=masks["alpha_curve"],
                op_mask=masks["op"],
            )

        assert result.actions["slot"].item() == 0  # Only slot available
        assert result.values.shape == (1,)

    def test_agent_with_5_slots(self):
        """PPOAgent should work with 5 slot configuration."""
        config = SlotConfig.for_grid(rows=1, cols=5)
        state_dim = get_feature_size(config)

        policy = create_policy(
            policy_type="lstm",
            state_dim=state_dim,
            slot_config=config,
            device="cpu",
            compile_mode="off",
        )
        agent = PPOAgent(
            policy=policy,
            slot_config=config,
            device="cpu",
            num_envs=2,
            max_steps_per_env=10,
        )

        assert agent.slot_config.num_slots == 5
        assert agent.policy.network.num_slots == 5

        # Verify batched forward pass works
        n_envs = 2
        states = torch.randn(n_envs, state_dim)
        masks = {
            "slot": torch.ones(n_envs, 5, dtype=torch.bool),
            "blueprint": torch.ones(n_envs, NUM_BLUEPRINTS, dtype=torch.bool),
            "style": torch.ones(n_envs, NUM_STYLES, dtype=torch.bool),
            "tempo": torch.ones(n_envs, NUM_TEMPO, dtype=torch.bool),
            "alpha_target": torch.ones(n_envs, NUM_ALPHA_TARGETS, dtype=torch.bool),
            "alpha_speed": torch.ones(n_envs, NUM_ALPHA_SPEEDS, dtype=torch.bool),
            "alpha_curve": torch.ones(n_envs, NUM_ALPHA_CURVES, dtype=torch.bool),
            "op": torch.ones(n_envs, NUM_OPS, dtype=torch.bool),
        }

        with torch.no_grad():
            result = agent.policy.network.get_action(
                states,
                slot_mask=masks["slot"],
                blueprint_mask=masks["blueprint"],
                style_mask=masks["style"],
                tempo_mask=masks["tempo"],
                alpha_target_mask=masks["alpha_target"],
                alpha_speed_mask=masks["alpha_speed"],
                alpha_curve_mask=masks["alpha_curve"],
                op_mask=masks["op"],
            )

        assert result.actions["slot"].shape == (n_envs,)
        assert all(0 <= a.item() < 5 for a in result.actions["slot"])
        assert result.values.shape == (n_envs,)

    def test_agent_with_9_slots_3x3_grid(self):
        """PPOAgent should work with 9 slot (3x3 grid) configuration."""
        config = SlotConfig.for_grid(rows=3, cols=3)
        state_dim = get_feature_size(config)

        policy = create_policy(
            policy_type="lstm",
            state_dim=state_dim,
            slot_config=config,
            device="cpu",
            compile_mode="off",
        )
        agent = PPOAgent(
            policy=policy,
            slot_config=config,
            device="cpu",
            num_envs=2,
            max_steps_per_env=10,
        )

        assert agent.slot_config.num_slots == 9
        assert agent.policy.network.num_slots == 9
        assert agent.slot_config.slot_ids == (
            "r0c0", "r0c1", "r0c2",
            "r1c0", "r1c1", "r1c2",
            "r2c0", "r2c1", "r2c2",
        )

        # Verify forward pass with all 9 slots
        n_envs = 2
        states = torch.randn(n_envs, state_dim)
        masks = {
            "slot": torch.ones(n_envs, 9, dtype=torch.bool),
            "blueprint": torch.ones(n_envs, NUM_BLUEPRINTS, dtype=torch.bool),
            "style": torch.ones(n_envs, NUM_STYLES, dtype=torch.bool),
            "tempo": torch.ones(n_envs, NUM_TEMPO, dtype=torch.bool),
            "alpha_target": torch.ones(n_envs, NUM_ALPHA_TARGETS, dtype=torch.bool),
            "alpha_speed": torch.ones(n_envs, NUM_ALPHA_SPEEDS, dtype=torch.bool),
            "alpha_curve": torch.ones(n_envs, NUM_ALPHA_CURVES, dtype=torch.bool),
            "op": torch.ones(n_envs, NUM_OPS, dtype=torch.bool),
        }

        with torch.no_grad():
            result = agent.policy.network.get_action(
                states,
                slot_mask=masks["slot"],
                blueprint_mask=masks["blueprint"],
                style_mask=masks["style"],
                tempo_mask=masks["tempo"],
                alpha_target_mask=masks["alpha_target"],
                alpha_speed_mask=masks["alpha_speed"],
                alpha_curve_mask=masks["alpha_curve"],
                op_mask=masks["op"],
            )

        assert result.actions["slot"].shape == (n_envs,)
        assert all(0 <= a.item() < 9 for a in result.actions["slot"])


class TestMultiEnvMaskIndependence:
    """Tests that masks are computed independently per environment."""

    def test_different_seed_states_produce_different_masks(self):
        """Environments with different seed states should have different masks."""
        config = SlotConfig.default()  # 3 slots

        # Environment 0: All slots empty
        slot_states_env0 = {slot_id: None for slot_id in config.slot_ids}

        # Environment 1: One slot occupied with TRAINING seed
        slot_states_env1 = {
            "r0c0": MaskSeedInfo(stage=SeedStage.TRAINING, seed_age_epochs=5),
            "r0c1": None,
            "r0c2": None,
        }

        # Environment 2: All slots occupied
        slot_states_env2 = {
            "r0c0": MaskSeedInfo(stage=SeedStage.TRAINING, seed_age_epochs=3),
            "r0c1": MaskSeedInfo(stage=SeedStage.BLENDING, seed_age_epochs=2),
            "r0c2": MaskSeedInfo(stage=SeedStage.HOLDING, seed_age_epochs=1),
        }

        masks_env0 = compute_action_masks(
            slot_states_env0,
            enabled_slots=list(config.slot_ids),
            slot_config=config,
        )
        masks_env1 = compute_action_masks(
            slot_states_env1,
            enabled_slots=list(config.slot_ids),
            slot_config=config,
        )
        masks_env2 = compute_action_masks(
            slot_states_env2,
            enabled_slots=list(config.slot_ids),
            slot_config=config,
        )

        # Env 0: All slots should be valid for GERMINATE
        assert masks_env0["slot"].all(), "All slots should be valid when empty"
        assert masks_env0["op"][LifecycleOp.GERMINATE], "GERMINATE should be valid"

        # Env 1: Two slots empty, one occupied
        assert masks_env1["slot"][0]  # r0c0 occupied, but slot selection is valid
        assert masks_env1["op"][LifecycleOp.GERMINATE], "GERMINATE valid with empty slots"

        # Env 2: All slots occupied - GERMINATE should be invalid
        assert not masks_env2["op"][LifecycleOp.GERMINATE], "GERMINATE invalid when all slots full"

    def test_batched_masks_preserve_per_env_differences(self):
        """Batched mask computation should preserve per-environment differences."""
        config = SlotConfig.default()

        # Create different slot states for each environment
        slot_states_batch = [
            {slot_id: None for slot_id in config.slot_ids},  # Env 0: all empty
            {"r0c0": MaskSeedInfo(stage=SeedStage.TRAINING, seed_age_epochs=5), "r0c1": None, "r0c2": None},  # Env 1: one seed
            {"r0c0": MaskSeedInfo(stage=SeedStage.TRAINING, seed_age_epochs=5), "r0c1": MaskSeedInfo(stage=SeedStage.TRAINING, seed_age_epochs=3), "r0c2": None},  # Env 2: two seeds
            {"r0c0": MaskSeedInfo(stage=SeedStage.TRAINING, seed_age_epochs=5), "r0c1": MaskSeedInfo(stage=SeedStage.TRAINING, seed_age_epochs=3), "r0c2": MaskSeedInfo(stage=SeedStage.TRAINING, seed_age_epochs=1)},  # Env 3: all full
        ]

        all_masks = []
        for slot_states in slot_states_batch:
            masks = compute_action_masks(
                slot_states,
                enabled_slots=list(config.slot_ids),
                slot_config=config,
            )
            all_masks.append(masks)

        # Stack into batched format
        batched_op_masks = torch.stack([m["op"] for m in all_masks])

        # Verify GERMINATE validity differs across environments
        germinate_valid = batched_op_masks[:, LifecycleOp.GERMINATE]
        assert germinate_valid[0].item() is True, "Env 0 (empty) should allow GERMINATE"
        assert germinate_valid[1].item() is True, "Env 1 (1 seed) should allow GERMINATE"
        assert germinate_valid[2].item() is True, "Env 2 (2 seeds) should allow GERMINATE"
        assert germinate_valid[3].item() is False, "Env 3 (all full) should NOT allow GERMINATE"


class TestSlotSaturationAndRecovery:
    """Tests for slot saturation detection and recovery after culling."""

    def test_germinate_masked_when_all_slots_full(self):
        """GERMINATE should be masked when all slots are occupied."""
        config = SlotConfig.default()  # 3 slots

        # All slots have seeds
        slot_states = {
            "r0c0": MaskSeedInfo(stage=SeedStage.TRAINING, seed_age_epochs=10),
            "r0c1": MaskSeedInfo(stage=SeedStage.BLENDING, seed_age_epochs=5),
            "r0c2": MaskSeedInfo(stage=SeedStage.HOLDING, seed_age_epochs=2),
        }

        masks = compute_action_masks(
            slot_states,
            enabled_slots=list(config.slot_ids),
            slot_config=config,
        )

        assert not masks["op"][LifecycleOp.GERMINATE], "GERMINATE should be masked when full"
        assert masks["op"][LifecycleOp.WAIT], "WAIT should always be valid"

    def test_germinate_enabled_after_cull(self):
        """GERMINATE should be re-enabled after culling frees a slot."""
        config = SlotConfig.default()

        # Initially all slots full
        slot_states_full = {
            "r0c0": MaskSeedInfo(stage=SeedStage.TRAINING, seed_age_epochs=10),
            "r0c1": MaskSeedInfo(stage=SeedStage.TRAINING, seed_age_epochs=5),
            "r0c2": MaskSeedInfo(stage=SeedStage.TRAINING, seed_age_epochs=2),
        }

        masks_full = compute_action_masks(
            slot_states_full,
            enabled_slots=list(config.slot_ids),
            slot_config=config,
        )
        assert not masks_full["op"][LifecycleOp.GERMINATE], "GERMINATE should be masked when full"

        # After culling r0c1, it becomes empty
        slot_states_after_cull = {
            "r0c0": MaskSeedInfo(stage=SeedStage.TRAINING, seed_age_epochs=10),
            "r0c1": None,  # Culled
            "r0c2": MaskSeedInfo(stage=SeedStage.TRAINING, seed_age_epochs=2),
        }

        masks_after_cull = compute_action_masks(
            slot_states_after_cull,
            enabled_slots=list(config.slot_ids),
            slot_config=config,
        )
        assert masks_after_cull["op"][LifecycleOp.GERMINATE], "GERMINATE should be valid after cull"

    def test_slot_mask_updates_after_cull(self):
        """Slot mask should show culled slot as available for germination."""
        config = SlotConfig.default()

        # r0c1 occupied, others empty
        slot_states = {
            "r0c0": None,
            "r0c1": MaskSeedInfo(stage=SeedStage.TRAINING, seed_age_epochs=5),
            "r0c2": None,
        }

        compute_action_masks(
            slot_states,
            enabled_slots=list(config.slot_ids),
            slot_config=config,
        )

        # r0c1 has a seed, so it's valid for targeting
        # But empty slots are also valid for GERMINATE target

        # After culling r0c1
        slot_states_after = {
            "r0c0": None,
            "r0c1": None,  # Culled
            "r0c2": None,
        }

        masks_after = compute_action_masks(
            slot_states_after,
            enabled_slots=list(config.slot_ids),
            slot_config=config,
        )

        # All slots should be available
        assert masks_after["slot"].all(), "All slots should be valid after cull"


class TestLargeSlotConfigurations:
    """Tests for larger slot configurations (5x5 = 25 slots)."""

    def test_agent_with_25_slots_5x5_grid(self):
        """PPOAgent should work with 25 slot (5x5 grid) configuration."""
        config = SlotConfig.for_grid(rows=5, cols=5)
        state_dim = get_feature_size(config)

        policy = create_policy(
            policy_type="lstm",
            state_dim=state_dim,
            slot_config=config,
            device="cpu",
            compile_mode="off",
        )
        agent = PPOAgent(
            policy=policy,
            slot_config=config,
            device="cpu",
            num_envs=1,
            max_steps_per_env=10,
        )

        assert agent.slot_config.num_slots == 25
        assert agent.policy.network.num_slots == 25

        # Verify forward pass with 25 slots
        states = torch.randn(1, state_dim)
        masks = {
            "slot": torch.ones(1, 25, dtype=torch.bool),
            "blueprint": torch.ones(1, NUM_BLUEPRINTS, dtype=torch.bool),
            "style": torch.ones(1, NUM_STYLES, dtype=torch.bool),
            "tempo": torch.ones(1, NUM_TEMPO, dtype=torch.bool),
            "alpha_target": torch.ones(1, NUM_ALPHA_TARGETS, dtype=torch.bool),
            "alpha_speed": torch.ones(1, NUM_ALPHA_SPEEDS, dtype=torch.bool),
            "alpha_curve": torch.ones(1, NUM_ALPHA_CURVES, dtype=torch.bool),
            "op": torch.ones(1, NUM_OPS, dtype=torch.bool),
        }

        with torch.no_grad():
            result = agent.policy.network.get_action(
                states,
                slot_mask=masks["slot"],
                blueprint_mask=masks["blueprint"],
                style_mask=masks["style"],
                tempo_mask=masks["tempo"],
                alpha_target_mask=masks["alpha_target"],
                alpha_speed_mask=masks["alpha_speed"],
                alpha_curve_mask=masks["alpha_curve"],
                op_mask=masks["op"],
            )

        assert 0 <= result.actions["slot"].item() < 25

    def test_mask_dimensions_with_25_slots(self):
        """Action masks should have correct dimensions with 25 slots."""
        config = SlotConfig.for_grid(rows=5, cols=5)

        slot_states = {slot_id: None for slot_id in config.slot_ids}

        masks = compute_action_masks(
            slot_states,
            enabled_slots=list(config.slot_ids),
            slot_config=config,
        )

        assert masks["slot"].shape == (25,)
        assert masks["blueprint"].shape == (NUM_BLUEPRINTS,)
        assert masks["style"].shape == (NUM_STYLES,)
        assert masks["tempo"].shape == (NUM_TEMPO,)
        assert masks["alpha_target"].shape == (NUM_ALPHA_TARGETS,)
        assert masks["alpha_speed"].shape == (NUM_ALPHA_SPEEDS,)
        assert masks["alpha_curve"].shape == (NUM_ALPHA_CURVES,)
        assert masks["op"].shape == (NUM_OPS,)


class TestBufferWithDynamicSlots:
    """Tests that buffer handles dynamic slot configurations correctly."""

    def test_buffer_stores_transitions_with_5_slots(self):
        """Buffer should correctly store transitions with 5-slot masks."""
        config = SlotConfig.for_grid(rows=1, cols=5)
        state_dim = get_feature_size(config)

        policy = create_policy(
            policy_type="lstm",
            state_dim=state_dim,
            slot_config=config,
            device="cpu",
            compile_mode="off",
        )
        agent = PPOAgent(
            policy=policy,
            slot_config=config,
            device="cpu",
            num_envs=2,
            max_steps_per_env=10,
        )

        # Start episodes
        for env_idx in range(2):
            agent.buffer.start_episode(env_id=env_idx)

        # Store transitions
        for env_idx in range(2):
            state = torch.randn(state_dim)
            hidden_h = torch.randn(1, agent.lstm_hidden_dim)
            hidden_c = torch.randn(1, agent.lstm_hidden_dim)

            agent.buffer.add(
                env_id=env_idx,
                state=state,
                slot_action=env_idx % 5,
                blueprint_action=0,
                style_action=0,
                tempo_action=0,
                alpha_target_action=0,
                alpha_speed_action=0,
                alpha_curve_action=0,
                op_action=0,
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
                slot_mask=torch.ones(5, dtype=torch.bool),
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

        assert len(agent.buffer) == 2

    def test_buffer_rejects_wrong_slot_mask_dimensions(self):
        """Buffer should handle slot masks with correct dimensions for its config."""
        config = SlotConfig.for_grid(rows=1, cols=5)  # 5 slots
        state_dim = get_feature_size(config)

        policy = create_policy(
            policy_type="lstm",
            state_dim=state_dim,
            slot_config=config,
            device="cpu",
            compile_mode="off",
        )
        agent = PPOAgent(
            policy=policy,
            slot_config=config,
            device="cpu",
            num_envs=1,
            max_steps_per_env=10,
        )

        agent.buffer.start_episode(env_id=0)

        # Correct dimensions should work
        state = torch.randn(state_dim)
        hidden_h = torch.randn(1, agent.lstm_hidden_dim)
        hidden_c = torch.randn(1, agent.lstm_hidden_dim)

        agent.buffer.add(
            env_id=0,
            state=state,
            slot_action=0,
            blueprint_action=0,
            style_action=0,
            tempo_action=0,
            alpha_target_action=0,
            alpha_speed_action=0,
            alpha_curve_action=0,
            op_action=0,
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
            slot_mask=torch.ones(5, dtype=torch.bool),  # Correct: 5 slots
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

        assert len(agent.buffer) == 1
