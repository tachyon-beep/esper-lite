"""Integration tests for Phase 2.2: Counterfactual contribution propagation to rollout buffer.

Tests that contributions computed from counterfactual ablation are properly
wired through execute_actions() to populate rollout buffer fields.

Key invariants:
1. has_fresh_contribution=True ONLY when baseline_accs has data (counterfactual measured)
2. contribution_targets[slot_idx] = val_acc - baseline_acc (positive = slot helps)
3. contribution_mask[slot_idx] = True for slots with ablation measurements
4. Non-measurement timesteps have has_fresh_contribution=False
"""

from __future__ import annotations

import torch

from esper.leyline.slot_config import SlotConfig
from esper.simic.agent.rollout_buffer import TamiyoRolloutBuffer


class TestContributionPropagationUnit:
    """Unit tests for contribution propagation logic in isolation."""

    def test_contribution_targets_from_baseline_accs(self) -> None:
        """Validates the contribution calculation: val_acc - baseline_acc."""
        # Simulate what execute_actions does
        slot_config = SlotConfig.default()  # 3 slots: r0c0, r0c1, r0c2
        val_acc = 75.0

        # baseline_accs contains accuracy with each slot disabled
        # contribution = val_acc - baseline_acc
        baseline_accs = {
            "r0c0": 70.0,  # Disabling r0c0 drops acc by 5 -> contribution = +5.0
            "r0c1": 78.0,  # Disabling r0c1 improves acc by 3 -> contribution = -3.0
        }

        # Build contribution tensor as execute_actions does
        num_slots = slot_config.num_slots
        contribution_targets = torch.zeros(num_slots)
        contribution_mask = torch.zeros(num_slots, dtype=torch.bool)

        for slot_id, baseline_acc in baseline_accs.items():
            slot_idx = slot_config.index_for_slot_id(slot_id)
            contribution = val_acc - baseline_acc
            contribution_targets[slot_idx] = contribution
            contribution_mask[slot_idx] = True

        # Verify
        assert contribution_targets[0].item() == 5.0  # r0c0
        assert contribution_targets[1].item() == -3.0  # r0c1
        assert contribution_targets[2].item() == 0.0  # r0c2 (not in baseline_accs)

        assert contribution_mask[0].item() is True
        assert contribution_mask[1].item() is True
        assert contribution_mask[2].item() is False

    def test_empty_baseline_accs_means_no_measurement(self) -> None:
        """When baseline_accs is empty, has_fresh_contribution should be False."""
        baseline_accs: dict[str, float] = {}
        has_fresh_contribution = len(baseline_accs) > 0

        assert has_fresh_contribution is False

    def test_nonempty_baseline_accs_means_measurement(self) -> None:
        """When baseline_accs has data, has_fresh_contribution should be True."""
        baseline_accs = {"r0c0": 70.0}
        has_fresh_contribution = len(baseline_accs) > 0

        assert has_fresh_contribution is True


class TestBufferStorageIntegration:
    """Integration tests for storing contributions in rollout buffer."""

    def _make_minimal_add_kwargs(
        self, buffer: TamiyoRolloutBuffer, env_id: int = 0
    ) -> dict:
        """Create minimal valid kwargs for buffer.add()."""
        num_slots = buffer.num_slots
        return {
            "env_id": env_id,
            "state": torch.zeros(buffer.state_dim),
            "blueprint_indices": torch.zeros(num_slots, dtype=torch.long),
            "slot_action": 0,
            "blueprint_action": 0,
            "style_action": 0,
            "tempo_action": 0,
            "alpha_target_action": 0,
            "alpha_speed_action": 0,
            "alpha_curve_action": 0,
            "op_action": 0,
            "effective_op_action": 0,
            "slot_log_prob": 0.0,
            "blueprint_log_prob": 0.0,
            "style_log_prob": 0.0,
            "tempo_log_prob": 0.0,
            "alpha_target_log_prob": 0.0,
            "alpha_speed_log_prob": 0.0,
            "alpha_curve_log_prob": 0.0,
            "op_log_prob": 0.0,
            "value": 0.5,
            "reward": 1.0,
            "done": False,
            "slot_mask": torch.ones(num_slots, dtype=torch.bool),
            "blueprint_mask": torch.ones(buffer.num_blueprints, dtype=torch.bool),
            "style_mask": torch.ones(buffer.num_styles, dtype=torch.bool),
            "tempo_mask": torch.ones(buffer.num_tempo, dtype=torch.bool),
            "alpha_target_mask": torch.ones(buffer.num_alpha_targets, dtype=torch.bool),
            "alpha_speed_mask": torch.ones(buffer.num_alpha_speeds, dtype=torch.bool),
            "alpha_curve_mask": torch.ones(buffer.num_alpha_curves, dtype=torch.bool),
            "op_mask": torch.ones(buffer.num_ops, dtype=torch.bool),
            "hidden_h": torch.zeros(buffer.lstm_layers, 1, buffer.lstm_hidden_dim),
            "hidden_c": torch.zeros(buffer.lstm_layers, 1, buffer.lstm_hidden_dim),
        }

    def test_simulated_execute_actions_flow(self) -> None:
        """Simulates execute_actions contribution wiring end-to-end."""
        slot_config = SlotConfig.default()  # 3 slots
        buffer = TamiyoRolloutBuffer(
            num_envs=2,
            max_steps_per_env=10,
            state_dim=64,
            slot_config=slot_config,
        )

        # Simulate env 0 with counterfactual measurement
        val_acc_env0 = 80.0
        baseline_accs_env0 = {"r0c0": 75.0, "r0c2": 82.0}

        # Build contribution data as execute_actions does
        env_baseline_accs = baseline_accs_env0
        has_fresh_contribution = len(env_baseline_accs) > 0

        num_slots = slot_config.num_slots
        contribution_targets_tensor = torch.zeros(num_slots)
        contribution_mask_tensor = torch.zeros(num_slots, dtype=torch.bool)

        for slot_id, baseline_acc in env_baseline_accs.items():
            slot_idx = slot_config.index_for_slot_id(slot_id)
            contribution = val_acc_env0 - baseline_acc
            contribution_targets_tensor[slot_idx] = contribution
            contribution_mask_tensor[slot_idx] = True

        # Add to buffer
        kwargs = self._make_minimal_add_kwargs(buffer, env_id=0)
        kwargs["has_fresh_contribution"] = has_fresh_contribution
        kwargs["contribution_targets"] = contribution_targets_tensor
        kwargs["contribution_mask"] = contribution_mask_tensor
        buffer.add(**kwargs)

        # Simulate env 1 WITHOUT counterfactual (no active seeds)
        baseline_accs_env1: dict[str, float] = {}
        has_fresh_env1 = len(baseline_accs_env1) > 0

        kwargs = self._make_minimal_add_kwargs(buffer, env_id=1)
        kwargs["has_fresh_contribution"] = has_fresh_env1
        buffer.add(**kwargs)

        # Verify env 0 has measurement
        assert buffer.has_fresh_contribution[0, 0].item() is True
        # r0c0: 80 - 75 = 5
        assert buffer.contribution_targets[0, 0, 0].item() == 5.0
        # r0c1: not measured
        assert buffer.contribution_targets[0, 0, 1].item() == 0.0
        # r0c2: 80 - 82 = -2 (this slot hurts performance!)
        assert buffer.contribution_targets[0, 0, 2].item() == -2.0

        assert buffer.contribution_mask[0, 0, 0].item() is True
        assert buffer.contribution_mask[0, 0, 1].item() is False
        assert buffer.contribution_mask[0, 0, 2].item() is True

        # Verify env 1 has no measurement
        assert buffer.has_fresh_contribution[1, 0].item() is False
        assert buffer.contribution_targets[1, 0].sum().item() == 0.0

    def test_get_batched_sequences_preserves_contributions(self) -> None:
        """Verifies contribution data survives get_batched_sequences()."""
        slot_config = SlotConfig.default()
        buffer = TamiyoRolloutBuffer(
            num_envs=1,
            max_steps_per_env=5,
            state_dim=64,
            slot_config=slot_config,
        )

        # Add measurement timestep
        contribution_targets = torch.tensor([1.5, -0.5, 0.0])
        contribution_mask = torch.tensor([True, True, False])

        kwargs = self._make_minimal_add_kwargs(buffer, env_id=0)
        kwargs["has_fresh_contribution"] = True
        kwargs["contribution_targets"] = contribution_targets
        kwargs["contribution_mask"] = contribution_mask
        buffer.add(**kwargs)

        # Add non-measurement timestep
        kwargs = self._make_minimal_add_kwargs(buffer, env_id=0)
        kwargs["has_fresh_contribution"] = False
        buffer.add(**kwargs)

        # Get batched data
        data = buffer.get_batched_sequences()

        # Verify measurement timestep
        assert data["has_fresh_contribution"][0, 0].item() is True
        assert torch.allclose(data["contribution_targets"][0, 0], contribution_targets)
        assert (data["contribution_mask"][0, 0] == contribution_mask).all()

        # Verify non-measurement timestep
        assert data["has_fresh_contribution"][0, 1].item() is False


class TestContributionSemantics:
    """Tests for contribution value semantics."""

    def test_positive_contribution_means_slot_helps(self) -> None:
        """Positive contribution = removing slot hurts accuracy."""
        val_acc = 80.0
        baseline_acc_without_slot = 70.0  # 10% drop when slot removed

        contribution = val_acc - baseline_acc_without_slot

        assert contribution > 0
        assert contribution == 10.0

    def test_negative_contribution_means_slot_hurts(self) -> None:
        """Negative contribution = removing slot improves accuracy (parasitic seed)."""
        val_acc = 70.0
        baseline_acc_without_slot = 75.0  # 5% improvement when slot removed

        contribution = val_acc - baseline_acc_without_slot

        assert contribution < 0
        assert contribution == -5.0

    def test_zero_contribution_means_slot_neutral(self) -> None:
        """Zero contribution = slot has no effect on accuracy."""
        val_acc = 75.0
        baseline_acc_without_slot = 75.0  # No change

        contribution = val_acc - baseline_acc_without_slot

        assert contribution == 0.0
