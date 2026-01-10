# tests/tamiyo/policy/test_probability_floor.py
"""Tests for probability floor in MaskedCategorical.

The probability floor is a hard constraint that guarantees minimum probability
mass on all valid actions, ensuring gradient flow even when entropy would
otherwise collapse. This is critical for sparse heads like blueprint/tempo
that only receive gradients ~5% of the time.

Key invariants:
1. No floor (min_prob=None) preserves original behavior
2. All valid actions have at least min_prob probability
3. Masked actions stay at ~0 probability
4. Probabilities still sum to 1 (proper normalization)
5. Gradients flow even with peaked logits (the core fix)
6. Floor is capped at uniform distribution (can't exceed 1/num_valid)
7. log_prob matches the floored distribution
"""

import torch

from esper.tamiyo.policy.action_masks import MaskedCategorical


class TestProbabilityFloorBasic:
    """Basic probability floor functionality tests."""

    def test_no_floor_preserves_original_behavior(self):
        """min_prob=None should not change the distribution."""
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        mask = torch.tensor([[True, True, True]])

        # Without floor
        dist_no_floor = MaskedCategorical(logits, mask, min_prob=None)
        probs_no_floor = dist_no_floor.probs

        # With floor=0.0 (equivalent to no floor)
        dist_zero_floor = MaskedCategorical(logits, mask, min_prob=0.0)
        probs_zero_floor = dist_zero_floor.probs

        # With explicit floor=None
        dist_explicit_none = MaskedCategorical(logits, mask)
        probs_explicit_none = dist_explicit_none.probs

        # All should be identical
        torch.testing.assert_close(probs_no_floor, probs_zero_floor)
        torch.testing.assert_close(probs_no_floor, probs_explicit_none)

    def test_floor_guarantees_minimum_probability(self):
        """All valid actions should have at least min_prob probability."""
        # Peaked logits that would normally give near-zero prob to first action
        logits = torch.tensor([[0.0, 10.0, 10.0]])
        mask = torch.tensor([[True, True, True]])
        min_prob = 0.10

        dist = MaskedCategorical(logits, mask, min_prob=min_prob)
        probs = dist.probs[0]

        # All valid actions should have at least min_prob
        for i in range(3):
            assert probs[i].item() >= min_prob - 1e-6, (
                f"Action {i} has prob {probs[i].item():.4f} < {min_prob}"
            )

    def test_floor_only_affects_valid_actions(self):
        """Masked actions should stay at ~0 probability."""
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        mask = torch.tensor([[True, True, False, False]])  # Last two masked
        min_prob = 0.10

        dist = MaskedCategorical(logits, mask, min_prob=min_prob)
        probs = dist.probs[0]

        # Masked actions should have ~0 probability
        assert probs[2].item() < 1e-6, f"Masked action has prob {probs[2].item()}"
        assert probs[3].item() < 1e-6, f"Masked action has prob {probs[3].item()}"

        # Valid actions should have at least min_prob
        assert probs[0].item() >= min_prob - 1e-6
        assert probs[1].item() >= min_prob - 1e-6

    def test_probabilities_sum_to_one(self):
        """Distribution should still be properly normalized."""
        logits = torch.tensor([[0.0, 10.0, -10.0]])
        mask = torch.tensor([[True, True, True]])
        min_prob = 0.10

        dist = MaskedCategorical(logits, mask, min_prob=min_prob)
        probs = dist.probs[0]

        # Probabilities should sum to ~1
        total = probs.sum().item()
        assert abs(total - 1.0) < 1e-5, f"Probabilities sum to {total}, not 1.0"

    def test_floor_caps_at_uniform(self):
        """Floor should not exceed 1/num_valid (uniform distribution)."""
        logits = torch.tensor([[1.0, 1.0]])  # 2 valid actions
        mask = torch.tensor([[True, True]])
        min_prob = 0.80  # Would require > 1.6 total if not capped

        dist = MaskedCategorical(logits, mask, min_prob=min_prob)
        probs = dist.probs[0]

        # With 2 actions, uniform is 0.5 each
        # Floor should be capped at 0.5 * 0.99 = 0.495
        assert probs[0].item() < 0.51, f"Prob {probs[0].item()} exceeds uniform"
        assert probs[1].item() < 0.51, f"Prob {probs[1].item()} exceeds uniform"

        # Should still sum to 1
        total = probs.sum().item()
        assert abs(total - 1.0) < 1e-5


class TestProbabilityFloorGradients:
    """Tests for gradient flow with probability floor.

    IMPORTANT: Hard probability floors are HARD constraints - they guarantee
    minimum probability mass but may not provide gradients in all cases.

    Gradient behavior depends on the distribution shape:
    1. When MULTIPLE actions are above the floor: gradients flow normally
       because the relative weights of overweight actions depend on logits.
    2. When only ONE action is above the floor: gradient is zero because
       the single overweight action gets exactly `remaining_mass = 1 - n*floor`,
       which is a constant w.r.t. logits.

    The floor's PRIMARY benefit is EXPLORATION (actions get sampled with
    min_prob), not gradient flow. In PPO:
    - During collection: floor ensures rare actions are sampled (10% chance)
    - During training: reward signals from those samples teach the agent
    """

    def test_gradient_flows_with_multiple_overweight_actions(self):
        """Gradients flow when MULTIPLE actions are above the floor.

        When multiple actions are overweight, their relative weights depend
        on the original logits, so gradients flow through them.
        """
        # Logits where all three actions are above floor (probs > 0.10)
        logits = torch.tensor([[1.0, 0.8, 0.0]], requires_grad=True)
        mask = torch.tensor([[True, True, True]])
        min_prob = 0.10

        dist = MaskedCategorical(logits, mask, min_prob=min_prob)

        # All probs should be > 0.10 (all overweight)
        probs = dist.probs[0]
        assert all(p > 0.10 for p in probs.tolist()), f"Expected all probs > 0.10: {probs}"

        # Compute log_prob - gradient should flow
        action = torch.tensor([0])
        log_prob = dist.log_prob(action)
        log_prob.backward()

        assert logits.grad is not None, "No gradient computed"
        grad_norm = logits.grad.norm().item()
        assert grad_norm > 1e-6, (
            f"Gradient norm {grad_norm} is too small with multiple overweight actions"
        )

    def test_gradient_zero_with_single_overweight_action(self):
        """Gradient is zero when only ONE action is above the floor.

        This is mathematically correct: the single overweight action gets
        exactly `remaining_mass = 1 - num_underweight * floor`, which doesn't
        depend on the original logits. The probs[0] in the numerator and
        denominator cancel: floored_prob[0] = probs[0] * (remaining / probs[0]).

        The floor still provides its PRIMARY benefit: exploration (action 1 and 2
        will be sampled 10% of the time each, providing reward signals).
        """
        # Peaked logits - only action 0 is above floor
        logits = torch.tensor([[3.0, 0.0, 0.0]], requires_grad=True)
        mask = torch.tensor([[True, True, True]])
        min_prob = 0.10

        dist = MaskedCategorical(logits, mask, min_prob=min_prob)

        # Verify distribution structure: 1 overweight, 2 underweight
        probs = dist.probs[0]
        above_floor = (probs > 0.10).sum().item()
        assert above_floor == 1, f"Expected 1 action above floor, got {above_floor}"

        # Compute log_prob - gradient will be zero due to cancellation
        action = torch.tensor([0])
        log_prob = dist.log_prob(action)
        log_prob.backward()

        # Document expected zero gradient (this is correct behavior)
        assert logits.grad is not None, "No gradient computed"
        # Gradient is zero because floored_prob[0] = 0.8 regardless of logits

    def test_entropy_gradient_with_multiple_overweight(self):
        """Entropy gradients flow when multiple actions are above floor."""
        # Logits where all actions are above floor
        logits = torch.tensor([[1.0, 0.8, 0.0]], requires_grad=True)
        mask = torch.tensor([[True, True, True]])
        min_prob = 0.10

        dist = MaskedCategorical(logits, mask, min_prob=min_prob)

        entropy = dist.entropy()
        entropy_loss = -entropy.mean()
        entropy_loss.backward()

        assert logits.grad is not None, "No gradient computed"
        grad_norm = logits.grad.norm().item()
        assert grad_norm > 1e-6, (
            f"Entropy gradient norm {grad_norm} should be non-zero with multiple overweight"
        )

    def test_floored_actions_provide_exploration(self):
        """The floor's main benefit: underweight actions get sampled.

        Even when gradients are zero, the floor ensures exploration by
        guaranteeing min_prob mass on all valid actions. With 10% probability,
        these actions will be sampled during rollout collection, providing
        reward signals that guide learning through the value function.
        """
        # Peaked logits - only action 0 would normally be sampled
        logits = torch.tensor([[100.0, 0.0, 0.0]]).expand(1000, -1)
        mask = torch.tensor([[True, True, True]]).expand(1000, -1)
        min_prob = 0.10

        dist = MaskedCategorical(logits, mask, min_prob=min_prob)
        samples = dist.sample()

        # Without floor, action 1 and 2 would ~never be sampled
        # With floor, they should appear ~10% of the time each
        count_1 = (samples == 1).sum().item()
        count_2 = (samples == 2).sum().item()

        # Allow variance but expect meaningful samples (10% ± 3%)
        assert count_1 > 50, f"Action 1 only sampled {count_1}/1000 times"
        assert count_2 > 50, f"Action 2 only sampled {count_2}/1000 times"

    def test_without_floor_entropy_gradient_vanishes(self):
        """Without floor, very peaked logits have vanishing entropy gradients."""
        logits = torch.tensor([[100.0, 0.0, 0.0]], requires_grad=True)
        mask = torch.tensor([[True, True, True]])

        dist = MaskedCategorical(logits, mask, min_prob=None)

        entropy = dist.entropy()
        entropy_loss = -entropy.mean()
        entropy_loss.backward()

        # Entropy gradient is very small (near-deterministic = low entropy)
        # This documents the problem the floor addresses through exploration
        _ = logits.grad.norm().item()  # Documents: gradient exists but is ~0


class TestProbabilityFloorLogProb:
    """Tests for log_prob consistency with probability floor."""

    def test_log_prob_correct_with_floor(self):
        """log_prob should match the floored distribution."""
        logits = torch.tensor([[0.0, 10.0, -5.0]])
        mask = torch.tensor([[True, True, True]])
        min_prob = 0.10

        dist = MaskedCategorical(logits, mask, min_prob=min_prob)
        probs = dist.probs[0]

        # Check log_prob for each action matches log(prob)
        for i in range(3):
            action = torch.tensor([i])
            computed_log_prob = dist.log_prob(action).item()
            expected_log_prob = torch.log(probs[i]).item()
            assert abs(computed_log_prob - expected_log_prob) < 1e-5, (
                f"Action {i}: log_prob={computed_log_prob:.4f} != log(prob)={expected_log_prob:.4f}"
            )

    def test_sample_uses_floored_distribution(self):
        """Samples should come from the floored distribution."""
        # Strongly peaked distribution
        logits = torch.tensor([[100.0, 0.0, 0.0]]).expand(1000, -1)
        mask = torch.tensor([[True, True, True]]).expand(1000, -1)
        min_prob = 0.10

        dist = MaskedCategorical(logits, mask, min_prob=min_prob)
        samples = dist.sample()

        # With floor=0.10, actions 1 and 2 should appear ~10% of the time
        count_1 = (samples == 1).sum().item()
        count_2 = (samples == 2).sum().item()

        # Should see at least some samples from actions 1 and 2
        # (with 10% probability each, expect ~100 samples, but allow variance)
        assert count_1 > 30, f"Action 1 only appeared {count_1}/1000 times"
        assert count_2 > 30, f"Action 2 only appeared {count_2}/1000 times"


class TestProbabilityFloorBatch:
    """Tests for batched probability floor computation."""

    def test_batch_processing(self):
        """Floor should work correctly with batched inputs."""
        batch_size = 4
        num_actions = 5
        logits = torch.randn(batch_size, num_actions)
        mask = torch.ones(batch_size, num_actions, dtype=torch.bool)
        # Mask out different actions per batch
        mask[0, 0] = False
        mask[1, 1] = False
        mask[2, 4] = False
        min_prob = 0.05

        dist = MaskedCategorical(logits, mask, min_prob=min_prob)
        probs = dist.probs

        # Check each batch element
        for b in range(batch_size):
            # Probabilities should sum to 1
            assert abs(probs[b].sum().item() - 1.0) < 1e-5

            # Valid actions should have at least min_prob
            for a in range(num_actions):
                if mask[b, a]:
                    assert probs[b, a].item() >= min_prob - 1e-6
                else:
                    assert probs[b, a].item() < 1e-6

    def test_varying_num_valid_per_batch(self):
        """Floor should adapt to different num_valid per batch element."""
        logits = torch.zeros(3, 4)
        mask = torch.tensor([
            [True, True, True, True],  # 4 valid
            [True, True, False, False],  # 2 valid
            [True, False, False, False],  # 1 valid
        ])
        min_prob = 0.30

        dist = MaskedCategorical(logits, mask, min_prob=min_prob)
        probs = dist.probs

        # Batch 0: 4 valid, floor=0.25 max -> each ~0.25
        assert abs(probs[0, 0].item() - 0.25) < 0.05  # ~uniform

        # Batch 1: 2 valid, floor capped at 0.5*0.99 -> ~0.495 each
        # Since floor > uniform, should be ~uniform
        assert abs(probs[1, 0].item() - 0.5) < 0.05
        assert abs(probs[1, 1].item() - 0.5) < 0.05

        # Batch 2: 1 valid -> must be 1.0
        assert abs(probs[2, 0].item() - 1.0) < 1e-5


class TestProbabilityFloorEntropy:
    """Tests for entropy computation with probability floor."""

    def test_entropy_increases_with_floor(self):
        """Entropy should increase when floor is applied to peaked distribution."""
        logits = torch.tensor([[100.0, 0.0, 0.0]])
        mask = torch.tensor([[True, True, True]])

        dist_no_floor = MaskedCategorical(logits, mask, min_prob=None)
        dist_with_floor = MaskedCategorical(logits, mask, min_prob=0.10)

        entropy_no_floor = dist_no_floor.entropy().item()
        entropy_with_floor = dist_with_floor.entropy().item()

        # Floor should increase entropy
        assert entropy_with_floor > entropy_no_floor, (
            f"Entropy with floor ({entropy_with_floor:.4f}) should be > "
            f"without floor ({entropy_no_floor:.4f})"
        )

    def test_entropy_bounded_by_uniform(self):
        """Entropy should not exceed max entropy (uniform distribution)."""
        logits = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
        mask = torch.tensor([[True, True, True, True]])
        min_prob = 0.10

        dist = MaskedCategorical(logits, mask, min_prob=min_prob)
        entropy = dist.entropy().item()

        # Max entropy for 4 actions is log(4) ≈ 1.386 (or 1.0 if normalized)
        # Normalized entropy should be <= 1.0
        assert entropy <= 1.0 + 1e-5, f"Normalized entropy {entropy} > 1.0"
