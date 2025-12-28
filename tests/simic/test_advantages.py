"""Tests for per-head advantage computation with causal masking."""

import torch

from esper.leyline import HEAD_NAMES, LifecycleOp, compute_causal_masks
from esper.simic.agent import compute_per_head_advantages


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

    def test_set_alpha_target_correct_masking(self):
        """When op=SET_ALPHA_TARGET, slot/style/alpha heads get advantage (B4-CR-01).

        SET_ALPHA_TARGET is the only operation that activates:
        - slot (target selection)
        - style (blend mode selection)
        - alpha_target, alpha_speed, alpha_curve (alpha schedule control)

        blueprint and tempo are masked because SET_ALPHA_TARGET doesn't
        involve selecting a new seed architecture.
        """
        op_actions = torch.tensor([LifecycleOp.SET_ALPHA_TARGET])
        base_advantages = torch.tensor([2.0])

        per_head = compute_per_head_advantages(base_advantages, op_actions)

        # Active heads for SET_ALPHA_TARGET
        assert torch.allclose(per_head["op"], base_advantages)
        assert torch.allclose(per_head["slot"], base_advantages)
        assert torch.allclose(per_head["style"], base_advantages)
        assert torch.allclose(per_head["alpha_target"], base_advantages)
        assert torch.allclose(per_head["alpha_speed"], base_advantages)
        assert torch.allclose(per_head["alpha_curve"], base_advantages)

        # Masked heads (not causally relevant for alpha adjustment)
        assert torch.allclose(per_head["blueprint"], torch.zeros(1))
        assert torch.allclose(per_head["tempo"], torch.zeros(1))

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


class TestComputeCausalMasks:
    """Tests for compute_causal_masks (B4-DRL-01: single source of truth)."""

    def test_mask_keys_match_head_names(self):
        """Mask dictionary must contain exactly HEAD_NAMES (B4-DRL-01).

        This ensures the causal mask contract stays synchronized with
        the canonical action head list from leyline.
        """
        op_actions = torch.tensor([LifecycleOp.WAIT])
        masks = compute_causal_masks(op_actions)

        assert set(masks.keys()) == set(HEAD_NAMES), (
            f"Mask keys {set(masks.keys())} != HEAD_NAMES {set(HEAD_NAMES)}"
        )

    def test_all_ops_produce_valid_masks(self):
        """compute_causal_masks produces valid boolean tensors for all ops."""
        all_ops = torch.tensor([op.value for op in LifecycleOp])
        masks = compute_causal_masks(all_ops)

        for head, mask in masks.items():
            assert mask.dtype == torch.bool, f"{head} mask should be bool"
            assert mask.shape == all_ops.shape, f"{head} mask shape mismatch"

    def test_advantages_and_masks_use_same_keys(self):
        """Both compute_per_head_advantages and compute_causal_masks use HEAD_NAMES.

        This is the critical invariant that B4-DRL-01 establishes: both
        advantages.py and ppo.py must use identical mask definitions.
        """
        op_actions = torch.tensor([LifecycleOp.GERMINATE, LifecycleOp.PRUNE])
        base_advantages = torch.tensor([1.0, 2.0])

        masks = compute_causal_masks(op_actions)
        advantages = compute_per_head_advantages(base_advantages, op_actions)

        assert set(masks.keys()) == set(advantages.keys()), (
            "compute_causal_masks and compute_per_head_advantages must return same keys"
        )
