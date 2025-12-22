"""Cross-component consistency tests for SlotConfig propagation.

Tests verify that SlotConfig is correctly propagated through the entire stack:
- PPOAgent → Network (num_slots matches)
- Features → Network (state_dim matches)
- Masks → Network (head sizes match)
"""

import pytest
import torch

from esper.leyline.slot_config import SlotConfig
from esper.leyline.factored_actions import (
    NUM_ALPHA_ALGORITHMS,
    NUM_ALPHA_CURVES,
    NUM_ALPHA_SPEEDS,
    NUM_ALPHA_TARGETS,
    NUM_OPS,
    NUM_BLUEPRINTS,
    NUM_BLENDS,
    NUM_TEMPO,
)
from esper.tamiyo.policy.features import get_feature_size, BASE_FEATURE_SIZE, SLOT_FEATURE_SIZE
from esper.simic.agent import PPOAgent
from esper.simic.agent import FactoredRecurrentActorCritic
from esper.tamiyo.policy.action_masks import compute_action_masks


class TestSlotConfigPropagation:
    """Tests that SlotConfig propagates correctly through the PPO stack."""

    def test_ppo_agent_stores_slot_config(self):
        """PPOAgent should store the provided SlotConfig."""
        config = SlotConfig.for_grid(rows=2, cols=2)  # 4 slots

        agent = PPOAgent(
            state_dim=get_feature_size(config),
            slot_config=config,
            compile_network=False,  # Skip compilation for faster tests
        )

        assert agent.slot_config == config
        assert agent.slot_config.num_slots == 4

    def test_ppo_agent_defaults_to_3_slots(self):
        """PPOAgent should default to 3-slot config when not specified."""
        agent = PPOAgent(
            state_dim=get_feature_size(SlotConfig.default()),
            compile_network=False,
        )

        assert agent.slot_config.num_slots == 3
        assert agent.slot_config.slot_ids == ("r0c0", "r0c1", "r0c2")

    def test_network_num_slots_matches_config(self):
        """Network's num_slots should match SlotConfig."""
        config = SlotConfig.for_grid(rows=2, cols=3)  # 6 slots
        state_dim = get_feature_size(config)

        network = FactoredRecurrentActorCritic(
            state_dim=state_dim,
            num_slots=config.num_slots,
        )

        assert network.num_slots == 6


class TestFeatureDimensionConsistency:
    """Tests that feature dimensions are consistent across components."""

    @pytest.mark.parametrize("num_slots", [1, 3, 5, 9, 25])
    def test_feature_size_formula(self, num_slots: int):
        """Feature size should follow the formula: BASE + num_slots * SLOT."""
        config = SlotConfig(slot_ids=tuple(f"r0c{i}" for i in range(num_slots)))

        expected = BASE_FEATURE_SIZE + num_slots * SLOT_FEATURE_SIZE
        actual = get_feature_size(config)

        assert actual == expected

    def test_network_input_matches_feature_output(self):
        """Network's state_dim should match feature extraction output."""
        config = SlotConfig.for_grid(rows=2, cols=2)  # 4 slots
        state_dim = get_feature_size(config)

        # Create network with correct state_dim
        network = FactoredRecurrentActorCritic(
            state_dim=state_dim,
            num_slots=config.num_slots,
        )

        # Create sample input matching feature output
        batch_size = 8
        seq_len = 1  # Single timestep
        states = torch.randn(batch_size, seq_len, state_dim)

        # Should not raise
        output = network(states)

        # Verify output dimensions (network returns a dict)
        # Network preserves sequence dimension: (batch, seq, dim)
        assert output["slot_logits"].shape == (batch_size, seq_len, config.num_slots)
        assert output["blueprint_logits"].shape == (batch_size, seq_len, NUM_BLUEPRINTS)
        assert output["blend_logits"].shape == (batch_size, seq_len, NUM_BLENDS)
        assert output["tempo_logits"].shape == (batch_size, seq_len, NUM_TEMPO)
        assert output["alpha_target_logits"].shape == (batch_size, seq_len, NUM_ALPHA_TARGETS)
        assert output["alpha_speed_logits"].shape == (batch_size, seq_len, NUM_ALPHA_SPEEDS)
        assert output["alpha_curve_logits"].shape == (batch_size, seq_len, NUM_ALPHA_CURVES)
        assert output["alpha_algorithm_logits"].shape == (batch_size, seq_len, NUM_ALPHA_ALGORITHMS)
        assert output["op_logits"].shape == (batch_size, seq_len, NUM_OPS)
        assert output["value"].shape == (batch_size, seq_len)


class TestMaskDimensionConsistency:
    """Tests that mask dimensions are consistent across components."""

    def test_slot_mask_matches_network_slot_head(self):
        """Slot mask dimension should match network's slot head output."""
        config = SlotConfig.for_grid(rows=2, cols=3)  # 6 slots

        # Create mask
        slot_states = {slot_id: None for slot_id in config.slot_ids}
        masks = compute_action_masks(
            slot_states,
            enabled_slots=list(config.slot_ids),
            slot_config=config,
        )

        # Create network
        network = FactoredRecurrentActorCritic(
            state_dim=get_feature_size(config),
            num_slots=config.num_slots,
        )

        # Check dimensions match
        assert masks["slot"].shape[0] == network.num_slots

    def test_op_mask_matches_network_op_head(self):
        """Op mask dimension should match NUM_OPS."""
        config = SlotConfig.default()

        slot_states = {slot_id: None for slot_id in config.slot_ids}
        masks = compute_action_masks(
            slot_states,
            enabled_slots=list(config.slot_ids),
            slot_config=config,
        )

        assert masks["op"].shape[0] == NUM_OPS

    def test_blueprint_mask_matches_network_blueprint_head(self):
        """Blueprint mask dimension should match NUM_BLUEPRINTS."""
        config = SlotConfig.default()

        slot_states = {slot_id: None for slot_id in config.slot_ids}
        masks = compute_action_masks(
            slot_states,
            enabled_slots=list(config.slot_ids),
            slot_config=config,
        )

        assert masks["blueprint"].shape[0] == NUM_BLUEPRINTS

    def test_blend_mask_matches_network_blend_head(self):
        """Blend mask dimension should match NUM_BLENDS."""
        config = SlotConfig.default()

        slot_states = {slot_id: None for slot_id in config.slot_ids}
        masks = compute_action_masks(
            slot_states,
            enabled_slots=list(config.slot_ids),
            slot_config=config,
        )

        assert masks["blend"].shape[0] == NUM_BLENDS

    def test_tempo_mask_matches_network_tempo_head(self):
        """Tempo mask dimension should match NUM_TEMPO."""
        config = SlotConfig.default()

        slot_states = {slot_id: None for slot_id in config.slot_ids}
        masks = compute_action_masks(
            slot_states,
            enabled_slots=list(config.slot_ids),
            slot_config=config,
        )

        assert masks["tempo"].shape[0] == NUM_TEMPO

    def test_alpha_target_mask_matches_network_head(self):
        """Alpha target mask dimension should match NUM_ALPHA_TARGETS."""
        config = SlotConfig.default()

        slot_states = {slot_id: None for slot_id in config.slot_ids}
        masks = compute_action_masks(
            slot_states,
            enabled_slots=list(config.slot_ids),
            slot_config=config,
        )

        assert masks["alpha_target"].shape[0] == NUM_ALPHA_TARGETS

    def test_alpha_speed_mask_matches_network_head(self):
        """Alpha speed mask dimension should match NUM_ALPHA_SPEEDS."""
        config = SlotConfig.default()

        slot_states = {slot_id: None for slot_id in config.slot_ids}
        masks = compute_action_masks(
            slot_states,
            enabled_slots=list(config.slot_ids),
            slot_config=config,
        )

        assert masks["alpha_speed"].shape[0] == NUM_ALPHA_SPEEDS

    def test_alpha_curve_mask_matches_network_head(self):
        """Alpha curve mask dimension should match NUM_ALPHA_CURVES."""
        config = SlotConfig.default()

        slot_states = {slot_id: None for slot_id in config.slot_ids}
        masks = compute_action_masks(
            slot_states,
            enabled_slots=list(config.slot_ids),
            slot_config=config,
        )

        assert masks["alpha_curve"].shape[0] == NUM_ALPHA_CURVES

    def test_alpha_algorithm_mask_matches_network_head(self):
        """Alpha algorithm mask dimension should match NUM_ALPHA_ALGORITHMS."""
        config = SlotConfig.default()

        slot_states = {slot_id: None for slot_id in config.slot_ids}
        masks = compute_action_masks(
            slot_states,
            enabled_slots=list(config.slot_ids),
            slot_config=config,
        )

        assert masks["alpha_algorithm"].shape[0] == NUM_ALPHA_ALGORITHMS


class TestMultiSlotConsistency:
    """Tests for consistency with multi-slot configurations."""

    @pytest.mark.parametrize("rows,cols", [(1, 1), (1, 3), (3, 3), (5, 5)])
    def test_grid_config_end_to_end(self, rows: int, cols: int):
        """Grid configurations should work consistently through the stack."""
        config = SlotConfig.for_grid(rows=rows, cols=cols)
        expected_slots = rows * cols
        state_dim = get_feature_size(config)

        # Verify slot count
        assert config.num_slots == expected_slots

        # Verify feature size
        assert state_dim == BASE_FEATURE_SIZE + expected_slots * SLOT_FEATURE_SIZE

        # Verify network creation
        network = FactoredRecurrentActorCritic(
            state_dim=state_dim,
            num_slots=config.num_slots,
        )
        assert network.num_slots == expected_slots

        # Verify mask dimensions
        slot_states = {slot_id: None for slot_id in config.slot_ids}
        masks = compute_action_masks(
            slot_states,
            enabled_slots=list(config.slot_ids),
            slot_config=config,
        )
        assert masks["slot"].shape[0] == expected_slots

    def test_from_specs_consistency(self):
        """SlotConfig from specs should be consistent with manual creation."""
        from esper.leyline.injection_spec import InjectionSpec

        specs = [
            InjectionSpec(slot_id="r0c0", channels=64, position=0.0, layer_range=(0, 2)),
            InjectionSpec(slot_id="r0c1", channels=128, position=0.5, layer_range=(2, 4)),
            InjectionSpec(slot_id="r0c2", channels=256, position=1.0, layer_range=(4, 6)),
        ]

        config = SlotConfig.from_specs(specs)
        state_dim = get_feature_size(config)

        # Verify config
        assert config.num_slots == 3
        assert config.slot_ids == ("r0c0", "r0c1", "r0c2")  # Sorted by position

        # Verify channels preserved
        assert config.channels_for_slot("r0c0") == 64
        assert config.channels_for_slot("r0c1") == 128
        assert config.channels_for_slot("r0c2") == 256

        # Verify network creation works
        network = FactoredRecurrentActorCritic(
            state_dim=state_dim,
            num_slots=config.num_slots,
        )
        assert network.num_slots == 3


class TestDimensionMismatchDetection:
    """Tests that dimension mismatches are detected appropriately."""

    def test_network_with_wrong_state_dim_fails(self):
        """Network should fail if state_dim doesn't match feature output."""
        config = SlotConfig.for_grid(rows=2, cols=2)  # 4 slots
        correct_state_dim = get_feature_size(config)
        wrong_state_dim = correct_state_dim + 10

        # Create network with correct dimensions
        network = FactoredRecurrentActorCritic(
            state_dim=correct_state_dim,
            num_slots=config.num_slots,
        )

        # Create input with wrong dimensions
        batch_size = 8
        seq_len = 1
        wrong_input = torch.randn(batch_size, seq_len, wrong_state_dim)

        # Should raise a runtime error due to dimension mismatch
        with pytest.raises(RuntimeError):
            network(wrong_input)

    def test_ppo_agent_network_num_slots_from_config(self):
        """PPOAgent's network should derive num_slots from slot_config."""
        config = SlotConfig.for_grid(rows=2, cols=2)  # 4 slots

        agent = PPOAgent(
            state_dim=get_feature_size(config),
            slot_config=config,
            compile_network=False,
        )

        # Agent's slot_config and network should agree on num_slots
        assert agent.slot_config.num_slots == agent.network.num_slots == 4
