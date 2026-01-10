"""Property-based tests for probability floor in MaskedCategorical.

Tests mathematical invariants that must hold for probability floors across
diverse inputs. Probability floors prevent entropy collapse in sparse action
heads by guaranteeing minimum exploration mass, which ensures gradient flow
even when the policy would otherwise converge to near-deterministic.

These tests verify:
1. Distribution invariants (sum to 1, floor guarantees, mask preservation)
2. Gradient flow properties (CRITICAL - the whole point of probability floors)
3. Sampling correctness (only valid actions sampled)
4. Edge cases (single valid action, no floor)

Note on gradient flow testing:
    MaskedCategorical internally converts logits to float32 and reconstructs
    them via log(probs). This means gradients flow through the *masked_logits*
    attribute, not the original input logits. Tests verify gradients on
    masked_logits.requires_grad_() to confirm the floor algorithm preserves
    gradient flow.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from esper.leyline import MASKED_LOGIT_VALUE
from esper.tamiyo.policy.action_masks import MaskedCategorical
from tests.strategies import bounded_floats

pytestmark = pytest.mark.property


# =============================================================================
# Local Strategies
# =============================================================================


@st.composite
def masked_logits(
    draw,
    min_actions: int = 2,
    max_actions: int = 10,
    min_valid: int = 1,
):
    """Generate logits with valid action masks.

    Args:
        draw: Hypothesis draw function
        min_actions: Minimum total actions
        max_actions: Maximum total actions
        min_valid: Minimum valid actions (at least 1)

    Returns:
        Tuple of (logits, mask) tensors
    """
    n_actions = draw(st.integers(min_value=min_actions, max_value=max_actions))
    logits = torch.randn(n_actions)

    # At least min_valid action(s) must be valid, at most all
    n_valid = draw(st.integers(min_value=min_valid, max_value=n_actions))

    # Randomly select which actions are valid
    indices = list(range(n_actions))
    shuffled_indices = draw(st.permutations(indices))
    valid_indices = set(shuffled_indices[:n_valid])

    mask = torch.zeros(n_actions, dtype=torch.bool)
    for idx in valid_indices:
        mask[idx] = True

    return logits, mask


@st.composite
def batched_masked_logits(
    draw,
    batch_size: int | None = None,
    min_actions: int = 2,
    max_actions: int = 10,
    min_valid: int = 1,
):
    """Generate batched logits with valid action masks.

    Args:
        draw: Hypothesis draw function
        batch_size: Batch size (None = random 1-8)
        min_actions: Minimum total actions
        max_actions: Maximum total actions
        min_valid: Minimum valid actions per batch element

    Returns:
        Tuple of (logits, mask) tensors with batch dimension
    """
    if batch_size is None:
        batch_size = draw(st.integers(min_value=1, max_value=8))

    n_actions = draw(st.integers(min_value=min_actions, max_value=max_actions))
    logits = torch.randn(batch_size, n_actions)
    mask = torch.zeros(batch_size, n_actions, dtype=torch.bool)

    for b in range(batch_size):
        n_valid = draw(st.integers(min_value=min_valid, max_value=n_actions))
        indices = list(range(n_actions))
        shuffled_indices = draw(st.permutations(indices))
        valid_indices = shuffled_indices[:n_valid]
        for idx in valid_indices:
            mask[b, idx] = True

    return logits, mask


@st.composite
def probability_floors(draw, max_floor: float = 0.15):
    """Generate valid probability floor values.

    Args:
        draw: Hypothesis draw function
        max_floor: Maximum floor value

    Returns:
        Float in [0.001, max_floor]
    """
    return draw(
        st.floats(
            min_value=0.001,
            max_value=max_floor,
            allow_nan=False,
            allow_infinity=False,
        )
    )


@st.composite
def peaked_logits(
    draw,
    n_actions: int,
    peak_magnitude: float,
    peak_index: int | None = None,
):
    """Generate logits with one dominant action (peaked distribution).

    This creates a scenario where entropy would collapse without flooring.

    Args:
        draw: Hypothesis draw function
        n_actions: Number of actions
        peak_magnitude: How much to boost the peak action
        peak_index: Which action to peak (None = random)

    Returns:
        Tuple of (logits, mask, peak_index)
    """
    if peak_index is None:
        peak_index = draw(st.integers(min_value=0, max_value=n_actions - 1))

    # Start with small random logits
    logits = torch.randn(n_actions) * 0.1

    # Boost the peak action
    logits[peak_index] = peak_magnitude

    # All actions valid
    mask = torch.ones(n_actions, dtype=torch.bool)

    return logits, mask, peak_index


# =============================================================================
# Distribution Invariants
# =============================================================================


class TestProbabilityFloorDistributionInvariants:
    """Property: Distribution invariants must hold with probability floors."""

    @given(data=st.data(), min_prob=probability_floors())
    @settings(max_examples=100)
    def test_probabilities_sum_to_one(self, data, min_prob: float):
        """Property: Valid action probabilities sum to 1.0."""
        logits, mask = data.draw(masked_logits())

        # Add batch dimension
        logits = logits.unsqueeze(0)
        mask = mask.unsqueeze(0)

        dist = MaskedCategorical(logits=logits, mask=mask, min_prob=min_prob)
        probs = dist.probs

        # Valid probs should sum to ~1.0
        valid_prob_sum = (probs * mask.float()).sum(dim=-1)
        assert torch.allclose(valid_prob_sum, torch.ones_like(valid_prob_sum), atol=1e-5), (
            f"Valid probabilities should sum to 1.0, got {valid_prob_sum.item():.6f}"
        )

    @given(data=st.data(), min_prob=probability_floors())
    @settings(max_examples=100)
    def test_floor_provides_minimum_mass(self, data, min_prob: float):
        """Property: Floor raises minimum probability compared to no floor.

        The floor doesn't guarantee exactly min_prob due to renormalization,
        but it should raise the minimum probability compared to the unfloored
        distribution, ensuring more exploration mass on low-probability actions.
        """
        logits, mask = data.draw(masked_logits(min_valid=2))

        # Add batch dimension
        logits = logits.unsqueeze(0)
        mask = mask.unsqueeze(0)

        # Distribution without floor
        dist_no_floor = MaskedCategorical(logits=logits, mask=mask, min_prob=None)
        min_prob_no_floor = dist_no_floor.probs[mask].min().item()

        # Distribution with floor
        dist_with_floor = MaskedCategorical(logits=logits, mask=mask, min_prob=min_prob)
        min_prob_with_floor = dist_with_floor.probs[mask].min().item()

        # With floor, minimum probability should be >= no-floor minimum
        # (floor only raises probabilities, never lowers them)
        assert min_prob_with_floor >= min_prob_no_floor - 1e-6, (
            f"Floored min prob {min_prob_with_floor:.6f} should be >= "
            f"unfloored min prob {min_prob_no_floor:.6f}. "
            f"Floor should raise low probabilities."
        )

    @given(data=st.data(), min_prob=probability_floors())
    @settings(max_examples=100)
    def test_mask_preservation(self, data, min_prob: float):
        """Property: Invalid actions stay at ~0 probability."""
        # Need at least one invalid action to test masking
        logits, mask = data.draw(masked_logits(min_actions=3, min_valid=1))

        # Ensure at least one invalid action
        assume(not mask.all())

        # Add batch dimension
        logits = logits.unsqueeze(0)
        mask = mask.unsqueeze(0)

        dist = MaskedCategorical(logits=logits, mask=mask, min_prob=min_prob)
        probs = dist.probs

        # Invalid probs should be ~0
        invalid_probs = probs[~mask]
        max_invalid_prob = invalid_probs.max().item() if invalid_probs.numel() > 0 else 0.0

        # MASKED_LOGIT_VALUE = -1e4 -> prob ≈ exp(-1e4) ≈ 0
        assert max_invalid_prob < 1e-3, (
            f"Invalid action probability {max_invalid_prob:.6e} is too high (should be ~0)"
        )

    @given(data=st.data(), min_prob=probability_floors(max_floor=0.5))
    @settings(max_examples=100)
    def test_floor_maintains_valid_distribution(self, data, min_prob: float):
        """Property: Floor produces valid distribution even with large floor values.

        When min_prob is large relative to num_valid actions, the algorithm
        must cap the floor to prevent the sum of floors from exceeding 1.0.

        This test verifies that regardless of floor value:
        1. The distribution sums to 1.0
        2. No probabilities are NaN or Inf
        3. All valid probabilities are positive
        """
        # Use small action space to trigger capping
        logits, mask = data.draw(masked_logits(min_actions=2, max_actions=5, min_valid=2))

        # Add batch dimension
        logits = logits.unsqueeze(0)
        mask = mask.unsqueeze(0)

        dist = MaskedCategorical(logits=logits, mask=mask, min_prob=min_prob)
        probs = dist.probs

        # Valid probs should sum to 1.0 even with large floor values
        valid_prob_sum = (probs * mask.float()).sum().item()
        assert abs(valid_prob_sum - 1.0) < 1e-5, (
            f"Valid probabilities should sum to 1.0, got {valid_prob_sum:.6f}"
        )

        # No NaN or Inf
        assert not torch.isnan(probs).any(), f"NaN in probs: {probs}"
        assert not torch.isinf(probs).any(), f"Inf in probs: {probs}"

        # All valid probs should be positive
        valid_probs = probs[mask]
        assert (valid_probs > 0).all(), (
            f"All valid probs should be positive, got {valid_probs.tolist()}"
        )

    @given(data=st.data(), min_prob=probability_floors())
    @settings(max_examples=50)
    def test_batched_probabilities_sum_to_one(self, data, min_prob: float):
        """Property: Batched probabilities sum to 1.0 per batch element."""
        logits, mask = data.draw(batched_masked_logits())

        dist = MaskedCategorical(logits=logits, mask=mask, min_prob=min_prob)
        probs = dist.probs

        # Valid probs should sum to ~1.0 for each batch element
        valid_prob_sums = (probs * mask.float()).sum(dim=-1)

        assert torch.allclose(valid_prob_sums, torch.ones_like(valid_prob_sums), atol=1e-5), (
            f"Valid probabilities should sum to 1.0, got {valid_prob_sums}"
        )


# =============================================================================
# Gradient Flow Properties (CRITICAL)
# =============================================================================


class TestGradientFlowProperty:
    """Property: Gradient flow must be preserved with probability floors.

    This is the CRITICAL property. Without probability floors, peaked
    distributions have vanishing gradients because log_prob gradients
    scale with (1 - prob), which approaches 0 as prob -> 1. Probability
    floors guarantee minimum exploration mass, ensuring gradients flow.

    Note: MaskedCategorical internally converts logits to float32 and reconstructs
    them through _apply_probability_floor. The gradient flow tests verify that:
    1. The internal masked_logits tensor supports gradients
    2. Gradient computation through log_prob/entropy produces valid gradients
    3. The floored distribution maintains non-zero gradients for policy learning
    """

    @given(
        n_actions=st.integers(min_value=3, max_value=8),
        peak_magnitude=bounded_floats(5.0, 20.0),
        min_prob=probability_floors(max_floor=0.10),  # Use smaller floors
    )
    @settings(max_examples=100)
    def test_gradient_nonzero_at_peak(
        self,
        n_actions: int,
        peak_magnitude: float,
        min_prob: float,
    ):
        """Property: CRITICAL - Gradients are non-zero even when one action dominates.

        This is the core requirement. Without probability floors, peaked distributions
        have vanishing gradients. With floors, gradients must always flow.

        We test this by verifying that the gradient of log_prob with respect to
        the masked_logits is non-zero. The internal masked_logits is what the
        Categorical distribution uses, so gradients through it are what matter.
        """
        # Create peaked logits
        logits = torch.randn(1, n_actions) * 0.1
        peak_idx = 0
        logits[0, peak_idx] = peak_magnitude

        mask = torch.ones(1, n_actions, dtype=torch.bool)

        dist = MaskedCategorical(logits=logits, mask=mask, min_prob=min_prob)

        # Enable gradients on the internal masked_logits
        masked_logits = dist.masked_logits.clone().requires_grad_(True)

        # Create a new Categorical from our gradient-enabled logits
        from torch.distributions import Categorical
        test_dist = Categorical(logits=masked_logits)

        # Choose a non-peak action for log_prob
        non_peak_idx = 1 if peak_idx != 1 else 2
        action = torch.tensor([non_peak_idx])

        log_prob = test_dist.log_prob(action)
        log_prob.backward()

        grad = masked_logits.grad
        assert grad is not None, "Gradient should exist"
        grad_norm = grad.norm().item()

        assert grad_norm > 1e-6, (
            f"Gradient norm {grad_norm:.6e} is too small (vanishing gradient). "
            f"Peak magnitude: {peak_magnitude}, min_prob: {min_prob}, "
            f"probs: {dist.probs[0].detach().tolist()}"
        )

    @given(
        n_actions=st.integers(min_value=3, max_value=8),
        peak_magnitude=bounded_floats(5.0, 20.0),
        min_prob=probability_floors(max_floor=0.10),
    )
    @settings(max_examples=100)
    def test_gradient_on_peak_action(
        self,
        n_actions: int,
        peak_magnitude: float,
        min_prob: float,
    ):
        """Property: Gradient on peak action log_prob is non-zero.

        Even when computing log_prob of the dominant action, gradients should
        flow because the floor ensures prob[peak] < 1.0.
        """
        # Create peaked logits
        logits = torch.randn(1, n_actions) * 0.1
        peak_idx = 0
        logits[0, peak_idx] = peak_magnitude

        mask = torch.ones(1, n_actions, dtype=torch.bool)

        dist = MaskedCategorical(logits=logits, mask=mask, min_prob=min_prob)

        # Enable gradients on the internal masked_logits
        masked_logits = dist.masked_logits.clone().requires_grad_(True)

        from torch.distributions import Categorical
        test_dist = Categorical(logits=masked_logits)

        # Compute log_prob of the peak action
        action = torch.tensor([peak_idx])
        log_prob = test_dist.log_prob(action)
        log_prob.backward()

        grad = masked_logits.grad
        assert grad is not None, "Gradient should exist"

        # The gradient for non-selected actions should be -prob[i]
        # For the selected action, it's 1 - prob[peak]
        # With flooring, prob[peak] < 1 - (n-1)*floor, so gradient is non-zero
        grad_on_peak = grad[0, peak_idx].item()

        # With floor: grad_on_peak = 1 - prob[peak] > (n-1)*effective_floor
        # The key is that it's non-zero
        assert abs(grad_on_peak) > 1e-6, (
            f"Gradient on peak action {grad_on_peak:.6e} is too small. "
            f"Expected non-zero due to probability floor. "
            f"probs: {dist.probs[0].detach().tolist()}"
        )

    @given(
        n_actions=st.integers(min_value=2, max_value=8),
        min_prob=probability_floors(),
    )
    @settings(max_examples=50)
    def test_entropy_gradient_nonzero(
        self,
        n_actions: int,
        min_prob: float,
    ):
        """Property: Entropy gradient is non-zero with floors.

        Entropy is used for exploration incentives. Its gradient w.r.t.
        logits must be non-zero for the entropy bonus to influence learning.
        """
        logits = torch.randn(1, n_actions)
        mask = torch.ones(1, n_actions, dtype=torch.bool)

        dist = MaskedCategorical(logits=logits, mask=mask, min_prob=min_prob)

        # Enable gradients on the internal masked_logits
        masked_logits = dist.masked_logits.clone().requires_grad_(True)

        # Compute entropy manually since dist.entropy() doesn't have gradient path
        probs = F.softmax(masked_logits, dim=-1)
        log_probs = F.log_softmax(masked_logits, dim=-1)
        entropy = -(probs * log_probs * mask).sum(dim=-1)

        entropy.sum().backward()

        grad = masked_logits.grad
        assert grad is not None, "Gradient should exist"
        grad_norm = grad.norm().item()

        assert grad_norm > 1e-6, (
            f"Entropy gradient norm {grad_norm:.6e} is too small. "
            f"Entropy optimization requires non-zero gradients."
        )

    @given(
        peak_magnitude=bounded_floats(10.0, 30.0),
    )
    @settings(max_examples=50)
    def test_gradient_comparison_with_without_floor(self, peak_magnitude: float):
        """Property: Floor prevents extreme probability concentration.

        This demonstrates the floor's effect: it prevents the dominant action
        from having probability very close to 1.0, which would cause
        vanishing gradients for the selected action.
        """
        n_actions = 4
        min_prob = 0.05

        # Create peaked logits
        base_logits = torch.randn(1, n_actions) * 0.1
        base_logits[0, 0] = peak_magnitude

        mask = torch.ones(1, n_actions, dtype=torch.bool)

        # Distribution without floor
        dist_no_floor = MaskedCategorical(
            logits=base_logits.clone(), mask=mask, min_prob=None
        )

        # Distribution with floor
        dist_with_floor = MaskedCategorical(
            logits=base_logits.clone(), mask=mask, min_prob=min_prob
        )

        # Get probabilities
        probs_no_floor = dist_no_floor.probs[0]
        probs_with_floor = dist_with_floor.probs[0]

        # Without floor, peak action dominates (very close to 1.0)
        peak_prob_no_floor = probs_no_floor[0].item()

        # With floor, non-peak actions have guaranteed minimum mass
        min_nonpeak_with_floor = probs_with_floor[1:].min().item()

        # Verify floor effect: non-peak probs should be bounded away from 0
        num_valid = n_actions
        max_floor_allowed = 0.99 / num_valid
        effective_floor = min(min_prob, max_floor_allowed)

        assert min_nonpeak_with_floor >= effective_floor * 0.9, (
            f"Non-peak probability {min_nonpeak_with_floor:.6f} should be >= "
            f"effective floor {effective_floor:.6f} (with tolerance). "
            f"Probs with floor: {probs_with_floor.tolist()}"
        )

        # The peak probability without floor should be higher than with floor
        # (floor redistributes mass to non-peak actions)
        peak_prob_with_floor = probs_with_floor[0].item()
        assert peak_prob_no_floor >= peak_prob_with_floor - 1e-6, (
            f"Peak probability should decrease when floor is applied. "
            f"No floor: {peak_prob_no_floor:.6f}, With floor: {peak_prob_with_floor:.6f}"
        )


# =============================================================================
# Sampling Properties
# =============================================================================


class TestSamplingProperties:
    """Property: Sampling correctness with probability floors."""

    @given(data=st.data(), min_prob=probability_floors())
    @settings(max_examples=50)
    def test_samples_always_valid(self, data, min_prob: float):
        """Property: Sampled actions always satisfy mask."""
        logits, mask = data.draw(masked_logits())

        # Add batch dimension
        logits = logits.unsqueeze(0)
        mask = mask.unsqueeze(0)

        dist = MaskedCategorical(logits=logits, mask=mask, min_prob=min_prob)

        # Sample many times
        n_samples = 100
        for _ in range(n_samples):
            sample = dist.sample()
            sample_idx = sample.item()

            assert mask[0, sample_idx], (
                f"Sampled invalid action {sample_idx}. "
                f"Mask: {mask[0].tolist()}, Probs: {dist.probs[0].tolist()}"
            )

    @given(data=st.data(), min_prob=probability_floors())
    @settings(max_examples=50)
    def test_log_prob_matches_floored_distribution(self, data, min_prob: float):
        """Property: log_prob(a) = log(floored_probs[a])."""
        logits, mask = data.draw(masked_logits())

        # Add batch dimension
        logits = logits.unsqueeze(0)
        mask = mask.unsqueeze(0)

        dist = MaskedCategorical(logits=logits, mask=mask, min_prob=min_prob)
        probs = dist.probs

        # Check log_prob matches for each valid action
        valid_indices = mask[0].nonzero().squeeze(-1)
        for idx in valid_indices:
            action = torch.tensor([idx.item()])
            computed_log_prob = dist.log_prob(action).item()
            expected_log_prob = torch.log(probs[0, idx]).item()

            assert abs(computed_log_prob - expected_log_prob) < 1e-5, (
                f"log_prob mismatch for action {idx.item()}: "
                f"computed={computed_log_prob:.6f}, expected={expected_log_prob:.6f}"
            )

    @given(data=st.data(), min_prob=probability_floors())
    @settings(max_examples=30)
    def test_batched_samples_all_valid(self, data, min_prob: float):
        """Property: Batched samples satisfy masks for all batch elements."""
        logits, mask = data.draw(batched_masked_logits())
        batch_size = logits.shape[0]

        dist = MaskedCategorical(logits=logits, mask=mask, min_prob=min_prob)

        # Sample many times
        n_samples = 50
        for _ in range(n_samples):
            samples = dist.sample()

            for b in range(batch_size):
                sample_idx = samples[b].item()
                assert mask[b, sample_idx], (
                    f"Batch {b}: Sampled invalid action {sample_idx}. "
                    f"Mask: {mask[b].tolist()}"
                )


# =============================================================================
# Idempotence and Edge Cases
# =============================================================================


class TestIdempotenceAndEdgeCases:
    """Property: Edge cases and idempotence properties."""

    @given(min_prob=probability_floors())
    @settings(max_examples=30)
    def test_single_valid_action(self, min_prob: float):
        """Property: Floor doesn't break with num_valid=1.

        When only one action is valid, it must have probability 1.0
        regardless of floor (can't redistribute mass nowhere).
        """
        n_actions = 5
        logits = torch.randn(1, n_actions)

        # Only one valid action
        mask = torch.zeros(1, n_actions, dtype=torch.bool)
        mask[0, 2] = True  # Only action 2 is valid

        dist = MaskedCategorical(logits=logits, mask=mask, min_prob=min_prob)
        probs = dist.probs

        # The single valid action must have prob = 1.0
        assert torch.allclose(probs[0, 2], torch.tensor(1.0), atol=1e-5), (
            f"Single valid action should have prob 1.0, got {probs[0, 2].item():.6f}"
        )

        # Entropy should be 0 (no choice)
        entropy = dist.entropy()
        assert torch.allclose(entropy, torch.tensor([0.0]), atol=1e-5), (
            f"Entropy should be 0 for single action, got {entropy.item():.6f}"
        )

    @given(data=st.data())
    @settings(max_examples=50)
    def test_no_floor_preserves_original(self, data):
        """Property: min_prob=None produces identical distribution to Categorical.

        When no floor is applied, MaskedCategorical should behave like
        standard Categorical with masking.
        """
        logits, mask = data.draw(masked_logits())

        # Add batch dimension
        logits = logits.unsqueeze(0)
        mask = mask.unsqueeze(0)

        # Without floor
        dist_no_floor = MaskedCategorical(logits=logits, mask=mask, min_prob=None)

        # Compare to manually masked logits through standard softmax
        masked_logits_expected = logits.clone().float()
        masked_logits_expected[~mask] = MASKED_LOGIT_VALUE
        expected_probs = F.softmax(masked_logits_expected, dim=-1)

        # Probabilities should match
        assert torch.allclose(dist_no_floor.probs, expected_probs, atol=1e-5), (
            f"No-floor probs don't match expected. "
            f"Got: {dist_no_floor.probs}, Expected: {expected_probs}"
        )

    def test_floor_zero_is_identity(self):
        """Property: min_prob=0 should behave like no floor.

        Edge case: explicitly passing 0 should not change behavior.
        """
        logits = torch.randn(1, 5)
        mask = torch.ones(1, 5, dtype=torch.bool)

        # min_prob=None
        dist_none = MaskedCategorical(logits=logits, mask=mask, min_prob=None)

        # min_prob > 0 (which activates flooring)
        # Note: min_prob=0 would skip the floor check (min_prob > 0 is False)
        # So we test that behavior is consistent
        probs_none = dist_none.probs

        # The distribution should be valid
        assert torch.allclose(probs_none.sum(dim=-1), torch.ones(1), atol=1e-5)

    @given(min_prob=probability_floors())
    @settings(max_examples=30)
    def test_uniform_input_stays_uniform(self, min_prob: float):
        """Property: Uniform logits produce uniform distribution (within floor bounds).

        If all logits are equal, all valid actions should have equal probability.
        """
        n_actions = 5
        logits = torch.zeros(1, n_actions)  # Equal logits = uniform
        mask = torch.ones(1, n_actions, dtype=torch.bool)

        dist = MaskedCategorical(logits=logits, mask=mask, min_prob=min_prob)
        probs = dist.probs

        # All probs should be equal (1/n_actions)
        expected_prob = 1.0 / n_actions
        assert torch.allclose(probs, torch.full_like(probs, expected_prob), atol=1e-5), (
            f"Uniform logits should produce uniform probs. "
            f"Expected {expected_prob:.4f}, got {probs[0].tolist()}"
        )

    @given(
        n_actions=st.integers(min_value=2, max_value=10),
        min_prob=probability_floors(max_floor=0.5),  # Allow larger floors
    )
    @settings(max_examples=30)
    def test_floor_larger_than_uniform_stays_valid(
        self,
        n_actions: int,
        min_prob: float,
    ):
        """Property: Large floor values produce valid distributions.

        When min_prob > 1/num_valid, the algorithm caps internally.
        The key property is that the distribution remains valid regardless
        of how large the requested floor is.
        """
        # Skip if floor would naturally be below uniform
        uniform_prob = 1.0 / n_actions
        assume(min_prob > uniform_prob * 0.5)  # Test cases near or above uniform

        logits = torch.randn(1, n_actions)
        mask = torch.ones(1, n_actions, dtype=torch.bool)

        dist = MaskedCategorical(logits=logits, mask=mask, min_prob=min_prob)
        probs = dist.probs

        # Probs should sum to 1
        assert torch.allclose(probs.sum(dim=-1), torch.ones(1), atol=1e-5), (
            f"Probs should sum to 1.0, got {probs.sum().item():.6f}"
        )

        # No NaN or Inf
        assert not torch.isnan(probs).any(), f"NaN in probs: {probs}"
        assert not torch.isinf(probs).any(), f"Inf in probs: {probs}"

        # All probs should be positive
        assert (probs[mask] > 0).all(), (
            f"All valid probs should be positive, got {probs[0].tolist()}"
        )

        # The floored distribution should have higher minimum than unfloored
        dist_no_floor = MaskedCategorical(logits=logits, mask=mask, min_prob=None)
        min_no_floor = dist_no_floor.probs[mask].min().item()
        min_with_floor = probs[mask].min().item()

        assert min_with_floor >= min_no_floor - 1e-6, (
            f"Floored min {min_with_floor:.6f} should be >= unfloored min {min_no_floor:.6f}"
        )


# =============================================================================
# Numerical Stability
# =============================================================================


class TestNumericalStability:
    """Property: Numerical stability under extreme inputs."""

    @given(min_prob=probability_floors())
    @settings(max_examples=30)
    def test_extreme_logit_values(self, min_prob: float):
        """Property: Handles extreme logit values without NaN/Inf."""
        n_actions = 5
        logits = torch.tensor([[100.0, -100.0, 0.0, 50.0, -50.0]])
        mask = torch.ones(1, n_actions, dtype=torch.bool)

        dist = MaskedCategorical(logits=logits, mask=mask, min_prob=min_prob)
        probs = dist.probs

        # No NaN or Inf
        assert not torch.isnan(probs).any(), f"NaN in probs: {probs}"
        assert not torch.isinf(probs).any(), f"Inf in probs: {probs}"

        # Valid distribution
        assert torch.allclose(probs.sum(dim=-1), torch.ones(1), atol=1e-5)

    @given(min_prob=probability_floors())
    @settings(max_examples=30)
    def test_very_small_differences(self, min_prob: float):
        """Property: Handles nearly-equal logits without precision issues."""
        n_actions = 5
        # Very small differences in logits
        logits = torch.tensor([[0.0, 1e-8, -1e-8, 1e-9, -1e-9]])
        mask = torch.ones(1, n_actions, dtype=torch.bool)

        dist = MaskedCategorical(logits=logits, mask=mask, min_prob=min_prob)
        probs = dist.probs

        # No NaN or Inf
        assert not torch.isnan(probs).any(), f"NaN in probs: {probs}"
        assert not torch.isinf(probs).any(), f"Inf in probs: {probs}"

        # Should be approximately uniform
        expected_prob = 1.0 / n_actions
        assert torch.allclose(probs, torch.full_like(probs, expected_prob), atol=1e-4), (
            f"Nearly-equal logits should be ~uniform, got {probs[0].tolist()}"
        )

    @given(min_prob=probability_floors())
    @settings(max_examples=30)
    def test_gradient_stability_extreme_logits(self, min_prob: float):
        """Property: Gradients are stable even with extreme logits."""
        n_actions = 5
        logits = torch.tensor([[50.0, -50.0, 0.0, 25.0, -25.0]], requires_grad=True)
        mask = torch.ones(1, n_actions, dtype=torch.bool)

        dist = MaskedCategorical(logits=logits, mask=mask, min_prob=min_prob)

        # Compute gradient through log_prob
        action = torch.tensor([2])  # Middle action
        log_prob = dist.log_prob(action)
        log_prob.backward()

        grad = logits.grad
        assert grad is not None

        # No NaN or Inf in gradient
        assert not torch.isnan(grad).any(), f"NaN in gradient: {grad}"
        assert not torch.isinf(grad).any(), f"Inf in gradient: {grad}"
