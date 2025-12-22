"""Tests for per-head advantage computation with causal masking."""

import torch

from esper.simic.agent import compute_per_head_advantages
from esper.leyline.factored_actions import LifecycleOp


class TestPerHeadAdvantages:
    """Tests for causal advantage masking."""

    def test_wait_masks_non_op_heads(self):
        """When op=WAIT, all non-op heads should get zero advantage."""
        op_actions = torch.tensor([LifecycleOp.WAIT, LifecycleOp.WAIT])
        base_advantages = torch.tensor([1.0, 2.0])

        per_head = compute_per_head_advantages(base_advantages, op_actions)

        # op always gets advantage
        assert torch.allclose(per_head["op"], base_advantages)

        # Others should be zero when WAIT
        assert torch.allclose(per_head["slot"], torch.zeros(2))
        assert torch.allclose(per_head["blueprint"], torch.zeros(2))
        assert torch.allclose(per_head["style"], torch.zeros(2))
        assert torch.allclose(per_head["tempo"], torch.zeros(2))
        assert torch.allclose(per_head["alpha_target"], torch.zeros(2))
        assert torch.allclose(per_head["alpha_speed"], torch.zeros(2))
        assert torch.allclose(per_head["alpha_curve"], torch.zeros(2))

    def test_germinate_all_heads_active(self):
        """When op=GERMINATE, all germination heads should get advantage."""
        op_actions = torch.tensor([LifecycleOp.GERMINATE, LifecycleOp.GERMINATE])
        base_advantages = torch.tensor([1.0, 2.0])

        per_head = compute_per_head_advantages(base_advantages, op_actions)

        # All germination heads active for GERMINATE
        assert torch.allclose(per_head["op"], base_advantages)
        assert torch.allclose(per_head["slot"], base_advantages)
        assert torch.allclose(per_head["blueprint"], base_advantages)
        assert torch.allclose(per_head["style"], base_advantages)
        assert torch.allclose(per_head["tempo"], base_advantages)
        assert torch.allclose(per_head["alpha_target"], base_advantages)
        assert torch.allclose(per_head["alpha_speed"], torch.zeros(2))
        assert torch.allclose(per_head["alpha_curve"], torch.zeros(2))

    def test_fossilize_slot_active_others_masked(self):
        """When op=FOSSILIZE, slot and op active, other heads masked."""
        op_actions = torch.tensor([LifecycleOp.FOSSILIZE])
        base_advantages = torch.tensor([5.0])

        per_head = compute_per_head_advantages(base_advantages, op_actions)

        assert torch.allclose(per_head["op"], base_advantages)
        assert torch.allclose(per_head["slot"], base_advantages)
        assert torch.allclose(per_head["blueprint"], torch.zeros(1))
        assert torch.allclose(per_head["style"], torch.zeros(1))
        assert torch.allclose(per_head["tempo"], torch.zeros(1))
        assert torch.allclose(per_head["alpha_target"], torch.zeros(1))
        assert torch.allclose(per_head["alpha_speed"], torch.zeros(1))
        assert torch.allclose(per_head["alpha_curve"], torch.zeros(1))

    def test_prune_slot_active_others_masked(self):
        """When op=PRUNE, slot/op and schedule heads active."""
        op_actions = torch.tensor([LifecycleOp.PRUNE])
        base_advantages = torch.tensor([3.0])

        per_head = compute_per_head_advantages(base_advantages, op_actions)

        assert torch.allclose(per_head["op"], base_advantages)
        assert torch.allclose(per_head["slot"], base_advantages)
        assert torch.allclose(per_head["blueprint"], torch.zeros(1))
        assert torch.allclose(per_head["style"], torch.zeros(1))
        assert torch.allclose(per_head["tempo"], torch.zeros(1))
        assert torch.allclose(per_head["alpha_target"], torch.zeros(1))
        assert torch.allclose(per_head["alpha_speed"], base_advantages)
        assert torch.allclose(per_head["alpha_curve"], base_advantages)

    def test_advance_slot_active_others_masked(self):
        """When op=ADVANCE, slot and op active, other heads masked."""
        op_actions = torch.tensor([LifecycleOp.ADVANCE])
        base_advantages = torch.tensor([4.0])

        per_head = compute_per_head_advantages(base_advantages, op_actions)

        assert torch.allclose(per_head["op"], base_advantages)
        assert torch.allclose(per_head["slot"], base_advantages)
        assert torch.allclose(per_head["blueprint"], torch.zeros(1))
        assert torch.allclose(per_head["style"], torch.zeros(1))
        assert torch.allclose(per_head["tempo"], torch.zeros(1))
        assert torch.allclose(per_head["alpha_target"], torch.zeros(1))
        assert torch.allclose(per_head["alpha_speed"], torch.zeros(1))
        assert torch.allclose(per_head["alpha_curve"], torch.zeros(1))

    def test_mixed_ops_correct_masking(self):
        """Mixed op types should apply correct masking per timestep."""
        op_actions = torch.tensor([
            LifecycleOp.WAIT,
            LifecycleOp.GERMINATE,
            LifecycleOp.FOSSILIZE,
            LifecycleOp.PRUNE,
        ])
        base_advantages = torch.tensor([1.0, 2.0, 3.0, 4.0])

        per_head = compute_per_head_advantages(base_advantages, op_actions)

        # op always active
        assert torch.allclose(per_head["op"], base_advantages)

        # slot: active for GERMINATE, FOSSILIZE, PRUNE (indices 1, 2, 3)
        expected_slot = torch.tensor([0.0, 2.0, 3.0, 4.0])
        assert torch.allclose(per_head["slot"], expected_slot)

        # Germination-only heads: only active for GERMINATE (index 1)
        expected_blueprint = torch.tensor([0.0, 2.0, 0.0, 0.0])
        assert torch.allclose(per_head["blueprint"], expected_blueprint)
        assert torch.allclose(per_head["style"], expected_blueprint)
        assert torch.allclose(per_head["tempo"], expected_blueprint)
        assert torch.allclose(per_head["alpha_target"], expected_blueprint)
        assert torch.allclose(per_head["alpha_speed"], torch.tensor([0.0, 0.0, 0.0, 4.0]))
        assert torch.allclose(per_head["alpha_curve"], torch.tensor([0.0, 0.0, 0.0, 4.0]))

    def test_only_wait_episode_sparse_gradients(self):
        """Episode with only WAIT actions should only give op_head gradients.

        This tests an edge case where germination heads receive zero
        gradient signal for the entire episode.
        """
        op_actions = torch.tensor([
            LifecycleOp.WAIT,
            LifecycleOp.WAIT,
            LifecycleOp.WAIT,
            LifecycleOp.WAIT,
            LifecycleOp.WAIT,
        ])
        base_advantages = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

        per_head = compute_per_head_advantages(base_advantages, op_actions)

        # Only op should receive gradients
        assert torch.allclose(per_head["op"], base_advantages)

        # All other heads should be zero (no causal effect)
        assert torch.allclose(per_head["slot"], torch.zeros(5))
        assert torch.allclose(per_head["blueprint"], torch.zeros(5))
        assert torch.allclose(per_head["style"], torch.zeros(5))
        assert torch.allclose(per_head["tempo"], torch.zeros(5))
        assert torch.allclose(per_head["alpha_target"], torch.zeros(5))
        assert torch.allclose(per_head["alpha_speed"], torch.zeros(5))
        assert torch.allclose(per_head["alpha_curve"], torch.zeros(5))

    def test_2d_input_batched(self):
        """Should work with batched [batch, seq] input."""
        op_actions = torch.tensor([
            [LifecycleOp.WAIT, LifecycleOp.GERMINATE],
            [LifecycleOp.PRUNE, LifecycleOp.FOSSILIZE],
        ])
        base_advantages = torch.tensor([
            [1.0, 2.0],
            [3.0, 4.0],
        ])

        per_head = compute_per_head_advantages(base_advantages, op_actions)

        # op always active
        assert torch.allclose(per_head["op"], base_advantages)

        # slot: WAIT=masked, GERMINATE=active, PRUNE=active, FOSSILIZE=active
        expected_slot = torch.tensor([
            [0.0, 2.0],
            [3.0, 4.0],
        ])
        assert torch.allclose(per_head["slot"], expected_slot)

        # Germination-only heads: only GERMINATE is active
        expected_blueprint = torch.tensor([
            [0.0, 2.0],
            [0.0, 0.0],
        ])
        assert torch.allclose(per_head["blueprint"], expected_blueprint)
        assert torch.allclose(per_head["style"], expected_blueprint)
        assert torch.allclose(per_head["tempo"], expected_blueprint)
        assert torch.allclose(per_head["alpha_target"], expected_blueprint)
        assert torch.allclose(per_head["alpha_speed"], torch.tensor([[0.0, 0.0], [3.0, 0.0]]))
        assert torch.allclose(per_head["alpha_curve"], torch.tensor([[0.0, 0.0], [3.0, 0.0]]))
