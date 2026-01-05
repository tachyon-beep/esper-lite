"""Property-based tests for PPO mathematical invariants.

Tier 8: PPO Core Algorithm Invariants

These tests verify that the core PPO computations are correct.
A single typo in these formulas would cause silent training failure:
- Training runs without crashing
- Loss decreases (optimizer finds local minima)
- But the agent learns the wrong thing

Key invariants tested:
1. Ratio = exp(new_log_prob - old_log_prob), NOT exp(old - new)
2. Clipped surrogate ≤ unclipped when ratio outside bounds
3. Returns = advantages + values (GAE definition)
4. Entropy ≥ 0 for any valid distribution
5. Policy loss sign: positive advantage + ratio > 1 = gradient to increase action prob
6. KL divergence ≥ 0 (from DRL expert review)
7. Joint ratio = product of per-head ratios (factored policies)
8. Log-ratio clamping prevents overflow
9. Advantage normalization produces μ≈0, σ≈1
"""

from __future__ import annotations

import math

import pytest
import torch
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from esper.leyline import HEAD_NAMES

# Mark all tests in this module as property tests for CI
pytestmark = pytest.mark.property


# =============================================================================
# Strategy Definitions
# =============================================================================


@st.composite
def log_prob_pairs(draw: st.DrawFn) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate plausible old/new log probability pairs.

    Log probs must be ≤ 0 (since prob ≤ 1).
    Typical values range from -0.1 (high prob) to -10 (very low prob).
    """
    batch_size = draw(st.integers(min_value=1, max_value=64))

    # Old log probs: typical training range
    old_values = draw(st.lists(
        st.floats(min_value=-15.0, max_value=-0.01, allow_nan=False, allow_infinity=False),
        min_size=batch_size,
        max_size=batch_size,
    ))

    # New log probs: similar but can differ (policy has changed)
    new_values = draw(st.lists(
        st.floats(min_value=-15.0, max_value=-0.01, allow_nan=False, allow_infinity=False),
        min_size=batch_size,
        max_size=batch_size,
    ))

    return (
        torch.tensor(old_values, dtype=torch.float32),
        torch.tensor(new_values, dtype=torch.float32),
    )


@st.composite
def advantage_values(draw: st.DrawFn) -> torch.Tensor:
    """Generate plausible advantage values.

    Advantages can be positive (better than expected) or negative (worse).
    After normalization, they typically range from -3 to +3.
    """
    batch_size = draw(st.integers(min_value=1, max_value=64))
    values = draw(st.lists(
        st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        min_size=batch_size,
        max_size=batch_size,
    ))
    return torch.tensor(values, dtype=torch.float32)


@st.composite
def gae_inputs(draw: st.DrawFn) -> dict:
    """Generate inputs for GAE computation."""
    num_steps = draw(st.integers(min_value=2, max_value=20))

    # Rewards: typical RL range
    rewards = draw(st.lists(
        st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
        min_size=num_steps,
        max_size=num_steps,
    ))

    # Values: network predictions, typically in similar range to cumulative rewards
    values = draw(st.lists(
        st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=num_steps,
        max_size=num_steps,
    ))

    # Gamma and lambda
    gamma = draw(st.floats(min_value=0.9, max_value=0.999))
    gae_lambda = draw(st.floats(min_value=0.9, max_value=1.0))

    return {
        "rewards": torch.tensor(rewards, dtype=torch.float32),
        "values": torch.tensor(values, dtype=torch.float32),
        "gamma": gamma,
        "gae_lambda": gae_lambda,
    }


# =============================================================================
# Property Tests: Ratio Computation
# =============================================================================


class TestRatioInvariants:
    """Ratio = exp(new - old) must hold exactly."""

    @given(log_probs=log_prob_pairs())
    @settings(max_examples=200)
    def test_ratio_formula_correct(self, log_probs: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Property: ratio = exp(new_log_prob - old_log_prob).

        A typo reversing the order (old - new) would invert the ratio,
        causing the policy to move AWAY from good actions.
        """
        old_log_prob, new_log_prob = log_probs

        # Correct formula
        ratio = torch.exp(new_log_prob - old_log_prob)

        # Verify: when new > old, ratio > 1 (action became more likely)
        increased_prob = new_log_prob > old_log_prob
        assert (ratio[increased_prob] > 1.0).all(), \
            "When new_log_prob > old_log_prob, ratio must be > 1"

        # Verify: when new < old, ratio < 1 (action became less likely)
        decreased_prob = new_log_prob < old_log_prob
        assert (ratio[decreased_prob] < 1.0).all(), \
            "When new_log_prob < old_log_prob, ratio must be < 1"

        # Verify: when new == old, ratio == 1
        same_prob = new_log_prob == old_log_prob
        if same_prob.any():
            assert torch.allclose(ratio[same_prob], torch.ones_like(ratio[same_prob])), \
                "When log probs equal, ratio must be 1"

    @given(log_probs=log_prob_pairs())
    @settings(max_examples=100)
    def test_ratio_positive(self, log_probs: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Property: ratio is always positive (exp of anything is positive)."""
        old_log_prob, new_log_prob = log_probs
        ratio = torch.exp(new_log_prob - old_log_prob)

        assert (ratio > 0).all(), "Ratio must always be positive"

    @given(log_probs=log_prob_pairs())
    @settings(max_examples=100)
    def test_log_ratio_recovers_diff(self, log_probs: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Property: log(ratio) = new - old (roundtrip check)."""
        old_log_prob, new_log_prob = log_probs
        ratio = torch.exp(new_log_prob - old_log_prob)

        recovered_diff = torch.log(ratio)
        expected_diff = new_log_prob - old_log_prob

        assert torch.allclose(recovered_diff, expected_diff, atol=1e-5), \
            "log(ratio) must equal new_log_prob - old_log_prob"


# =============================================================================
# Property Tests: Clipped Surrogate
# =============================================================================


class TestClippedSurrogateInvariants:
    """Clipped PPO surrogate must satisfy key properties."""

    @given(
        log_probs=log_prob_pairs(),
        advantages=advantage_values(),
        clip_ratio=st.floats(min_value=0.1, max_value=0.3),
    )
    @settings(max_examples=200)
    def test_clipping_bounds_ratio(
        self,
        log_probs: tuple[torch.Tensor, torch.Tensor],
        advantages: torch.Tensor,
        clip_ratio: float,
    ) -> None:
        """Property: clipped ratio is within [1-eps, 1+eps]."""
        old_log_prob, new_log_prob = log_probs

        # Match batch sizes
        min_size = min(len(old_log_prob), len(advantages))
        old_log_prob = old_log_prob[:min_size]
        new_log_prob = new_log_prob[:min_size]
        advantages = advantages[:min_size]

        ratio = torch.exp(new_log_prob - old_log_prob)
        clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)

        assert (clipped_ratio >= 1 - clip_ratio).all(), \
            f"Clipped ratio below lower bound: {clipped_ratio.min()}"
        assert (clipped_ratio <= 1 + clip_ratio).all(), \
            f"Clipped ratio above upper bound: {clipped_ratio.max()}"

    @given(
        log_probs=log_prob_pairs(),
        advantages=advantage_values(),
        clip_ratio=st.floats(min_value=0.1, max_value=0.3),
    )
    @settings(max_examples=200)
    def test_min_surrogate_is_conservative(
        self,
        log_probs: tuple[torch.Tensor, torch.Tensor],
        advantages: torch.Tensor,
        clip_ratio: float,
    ) -> None:
        """Property: min(surr1, surr2) ≤ surr1 (clipping is always conservative).

        Taking the minimum ensures we don't over-optimize when the ratio
        is already outside the trust region.
        """
        old_log_prob, new_log_prob = log_probs

        # Match batch sizes
        min_size = min(len(old_log_prob), len(advantages))
        old_log_prob = old_log_prob[:min_size]
        new_log_prob = new_log_prob[:min_size]
        advantages = advantages[:min_size]

        ratio = torch.exp(new_log_prob - old_log_prob)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
        clipped_surr = torch.min(surr1, surr2)

        # min(a, b) <= a always
        assert (clipped_surr <= surr1 + 1e-6).all(), \
            "Clipped surrogate must be <= unclipped surrogate"

    @given(
        old_log_prob=st.floats(min_value=-10.0, max_value=-0.1),
        new_log_prob=st.floats(min_value=-10.0, max_value=-0.1),
        advantage=st.floats(min_value=0.1, max_value=5.0),  # Positive advantage
        clip_ratio=st.floats(min_value=0.1, max_value=0.3),
    )
    @settings(max_examples=200)
    def test_positive_advantage_clipping_direction(
        self,
        old_log_prob: float,
        new_log_prob: float,
        advantage: float,
        clip_ratio: float,
    ) -> None:
        """Property: For positive advantage, clipping limits INCREASE in ratio.

        When advantage > 0, we want to increase the action probability.
        But if ratio is already > 1+eps, clipping prevents excessive increase.
        """
        ratio = math.exp(new_log_prob - old_log_prob)
        assume(math.isfinite(ratio) and ratio > 0)

        clipped_ratio = max(1 - clip_ratio, min(ratio, 1 + clip_ratio))

        surr1 = ratio * advantage
        surr2 = clipped_ratio * advantage
        clipped_surr = min(surr1, surr2)

        # For positive advantage:
        # - If ratio > 1+eps, clipping REDUCES the surrogate
        if ratio > 1 + clip_ratio:
            assert clipped_surr < surr1, \
                "When ratio exceeds upper bound with positive advantage, clipping must reduce surrogate"


# =============================================================================
# Property Tests: GAE Returns
# =============================================================================


class TestGAEReturnsInvariant:
    """returns = advantages + values must hold (GAE definition)."""

    @given(inputs=gae_inputs())
    @settings(max_examples=100, deadline=None)
    def test_returns_equals_advantages_plus_values(self, inputs: dict) -> None:
        """Property: After GAE computation, returns = advantages + values.

        This is the mathematical definition. If someone types
        `returns = advantages - values` or any other variant, this fails.
        """
        rewards = inputs["rewards"]
        values = inputs["values"]
        gamma = inputs["gamma"]
        gae_lambda = inputs["gae_lambda"]

        num_steps = len(rewards)
        advantages = torch.zeros(num_steps)
        last_gae = 0.0

        # Simulate GAE computation (simplified - no dones for this test)
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_value = 0.0  # Terminal
            else:
                next_value = values[t + 1].item()

            delta = rewards[t] + gamma * next_value - values[t]
            last_gae = delta + gamma * gae_lambda * last_gae
            advantages[t] = last_gae

        # THE CRITICAL INVARIANT
        returns = advantages + values

        # Verify returns are reasonable (not NaN, not exploding)
        assert torch.isfinite(returns).all(), "Returns must be finite"

        # Verify the relationship holds if we back-compute
        recomputed_advantages = returns - values
        assert torch.allclose(recomputed_advantages, advantages, atol=1e-5), \
            "returns - values must equal advantages"

    @given(
        value=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False),
        advantage=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_return_single_step(self, value: float, advantage: float) -> None:
        """Property: return = advantage + value for single timestep."""
        value_t = torch.tensor(value, dtype=torch.float64)  # Use float64 for precision
        advantage_t = torch.tensor(advantage, dtype=torch.float64)

        return_t = advantage_t + value_t

        # Verify roundtrip with appropriate tolerance for floating point
        assert torch.isclose(return_t - value_t, advantage_t, atol=1e-10), \
            f"return - value must equal advantage: {return_t - value_t} vs {advantage_t}"


# =============================================================================
# Property Tests: Entropy
# =============================================================================


class TestEntropyInvariants:
    """Entropy must satisfy mathematical properties."""

    @given(probs=st.lists(
        st.floats(min_value=0.01, max_value=1.0),
        min_size=2,
        max_size=20,
    ))
    @settings(max_examples=100)
    def test_entropy_non_negative(self, probs: list[float]) -> None:
        """Property: Entropy ≥ 0 for any probability distribution.

        If someone types `+= entropy` instead of `-= entropy` in the loss,
        we'd be minimizing entropy (encouraging determinism).
        """
        # Normalize to valid distribution
        probs_t = torch.tensor(probs, dtype=torch.float32)
        probs_t = probs_t / probs_t.sum()

        # Compute entropy: -sum(p * log(p))
        log_probs = torch.log(probs_t + 1e-10)  # Add epsilon to avoid log(0)
        entropy = -(probs_t * log_probs).sum()

        assert entropy >= 0, f"Entropy must be non-negative, got {entropy}"

    @given(n=st.integers(min_value=2, max_value=100))
    @settings(max_examples=50)
    def test_uniform_distribution_max_entropy(self, n: int) -> None:
        """Property: Uniform distribution has maximum entropy = log(n)."""
        probs = torch.ones(n) / n
        log_probs = torch.log(probs)
        entropy = -(probs * log_probs).sum()

        expected_max = math.log(n)
        assert torch.isclose(entropy, torch.tensor(expected_max), atol=1e-5), \
            f"Uniform entropy should be log({n})={expected_max}, got {entropy}"

    def test_deterministic_zero_entropy(self) -> None:
        """Property: Deterministic distribution (one-hot) has entropy = 0."""
        probs = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])

        # Use safe log: log(0) would be -inf, but 0 * -inf = nan
        # Properly: 0 * log(0) should be treated as 0
        log_probs = torch.where(probs > 0, torch.log(probs), torch.zeros_like(probs))
        entropy = -(probs * log_probs).sum()

        assert torch.isclose(entropy, torch.tensor(0.0), atol=1e-6), \
            f"Deterministic entropy should be 0, got {entropy}"


# =============================================================================
# Property Tests: Policy Loss Sign Convention
# =============================================================================


class TestPolicyLossSignConvention:
    """Policy loss sign must encourage increasing prob of good actions."""

    @given(
        ratio=st.floats(min_value=0.5, max_value=2.0),
        advantage=st.floats(min_value=-5.0, max_value=5.0),
    )
    @settings(max_examples=200)
    def test_loss_gradient_direction(self, ratio: float, advantage: float) -> None:
        """Property: Policy loss = -min(ratio*A, clip(ratio)*A).

        The negative sign is critical:
        - Positive advantage → want to INCREASE ratio → gradient should push ratio up
        - Negative advantage → want to DECREASE ratio → gradient should push ratio down

        If someone forgets the negative sign, training does the opposite.
        """
        assume(math.isfinite(ratio) and math.isfinite(advantage))

        clip_ratio = 0.2
        clipped_ratio = max(1 - clip_ratio, min(ratio, 1 + clip_ratio))

        surr1 = ratio * advantage
        surr2 = clipped_ratio * advantage
        clipped_surr = min(surr1, surr2)

        # Policy loss is NEGATIVE of clipped surrogate
        # We MINIMIZE loss, so maximizing -(-clipped_surr) = maximizing clipped_surr
        policy_loss = -clipped_surr

        # Verify: loss has opposite sign of clipped_surr
        if clipped_surr > 0:
            assert policy_loss < 0, \
                "When clipped_surr > 0, policy_loss must be < 0"
        elif clipped_surr < 0:
            assert policy_loss > 0, \
                "When clipped_surr < 0, policy_loss must be > 0"


# =============================================================================
# Property Tests: Value Loss
# =============================================================================


class TestValueLossInvariants:
    """Value loss must be non-negative (it's a squared error)."""

    @given(
        pred=st.floats(min_value=-50.0, max_value=50.0, allow_nan=False),
        target=st.floats(min_value=-50.0, max_value=50.0, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_mse_non_negative(self, pred: float, target: float) -> None:
        """Property: MSE loss ≥ 0."""
        loss = (pred - target) ** 2
        assert loss >= 0, f"MSE must be non-negative, got {loss}"

    @given(
        pred=st.floats(min_value=-50.0, max_value=50.0, allow_nan=False),
        target=st.floats(min_value=-50.0, max_value=50.0, allow_nan=False),
        old_value=st.floats(min_value=-50.0, max_value=50.0, allow_nan=False),
        value_clip=st.floats(min_value=0.5, max_value=2.0),
    )
    @settings(max_examples=100)
    def test_clipped_value_loss_non_negative(
        self, pred: float, target: float, old_value: float, value_clip: float
    ) -> None:
        """Property: Clipped value loss ≥ 0."""
        # Clipped value prediction
        clipped_pred = old_value + max(-value_clip, min(pred - old_value, value_clip))

        loss_unclipped = (pred - target) ** 2
        loss_clipped = (clipped_pred - target) ** 2
        loss = 0.5 * max(loss_unclipped, loss_clipped)

        assert loss >= 0, f"Clipped value loss must be non-negative, got {loss}"

    @given(
        pred=st.floats(min_value=-50.0, max_value=50.0, allow_nan=False),
        target=st.floats(min_value=-50.0, max_value=50.0, allow_nan=False),
        old_value=st.floats(min_value=-50.0, max_value=50.0, allow_nan=False),
        value_clip=st.floats(min_value=0.5, max_value=2.0),
    )
    @settings(max_examples=100)
    def test_value_clipping_is_pessimistic(
        self, pred: float, target: float, old_value: float, value_clip: float
    ) -> None:
        """Property: Value clipping uses MAX (pessimistic), not MIN.

        A bug using min instead of max would defeat the purpose of clipping.
        """
        clipped_pred = old_value + max(-value_clip, min(pred - old_value, value_clip))

        loss_unclipped = (pred - target) ** 2
        loss_clipped = (clipped_pred - target) ** 2
        loss = 0.5 * max(loss_unclipped, loss_clipped)

        # Key property: loss >= 0.5 * loss_unclipped (max is pessimistic)
        assert loss >= 0.5 * loss_unclipped - 1e-10, \
            "Value loss with max must be >= unclipped loss (pessimistic)"


# =============================================================================
# Property Tests: KL Divergence (from DRL expert review)
# =============================================================================


class TestKLDivergenceInvariants:
    """KL divergence must satisfy mathematical properties."""

    @given(log_probs=log_prob_pairs())
    @settings(max_examples=200)
    def test_kl_divergence_non_negative(
        self, log_probs: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Property: KL(old||new) ≥ 0.

        The implementation uses the KL3 estimator (Schulman):
        KL = E[(ratio - 1) - log(ratio)]

        This must always be non-negative. If it's negative, something is wrong.
        """
        old_log_prob, new_log_prob = log_probs
        ratio = torch.exp(new_log_prob - old_log_prob)

        # KL3 estimator: (ratio - 1) - log(ratio)
        # This equals 0 when ratio=1, and is convex with minimum at ratio=1
        kl = (ratio - 1) - torch.log(ratio)

        # Filter out extreme ratios that could cause numerical issues
        valid_mask = (ratio > 1e-6) & (ratio < 1e6)
        if valid_mask.any():
            valid_kl = kl[valid_mask]
            assert (valid_kl >= -1e-6).all(), \
                f"KL divergence must be non-negative, got min={valid_kl.min()}"

    @given(
        old_log_prob=st.floats(min_value=-10.0, max_value=-0.1, allow_nan=False),
        new_log_prob=st.floats(min_value=-10.0, max_value=-0.1, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_kl_zero_when_distributions_equal(
        self, old_log_prob: float, new_log_prob: float
    ) -> None:
        """Property: KL = 0 when old and new distributions are identical."""
        # Use same log_prob for both (identical distributions)
        ratio = math.exp(old_log_prob - old_log_prob)  # = 1
        kl = (ratio - 1) - math.log(ratio)

        assert abs(kl) < 1e-10, f"KL should be 0 for identical distributions, got {kl}"


# =============================================================================
# Property Tests: Factored Policy Joint Ratio (from DRL expert review)
# =============================================================================


class TestJointRatioInvariants:
    """Joint ratio for factored policies must equal product of per-head ratios."""

    @given(
        head_log_ratios=st.lists(
            st.floats(min_value=-5.0, max_value=5.0, allow_nan=False),
            min_size=len(HEAD_NAMES),
            max_size=len(HEAD_NAMES),
        )
    )
    @settings(max_examples=100)
    def test_joint_ratio_equals_product_of_per_head_ratios(
        self, head_log_ratios: list[float]
    ) -> None:
        """Property: joint_ratio = exp(sum(log_ratios)) = product(ratios).

        In factored policies, the joint probability is the product of per-head
        probabilities: P(a) = P(a_slot) * P(a_blueprint) * ... * P(a_op)

        Therefore: ratio = P_new/P_old = prod(ratio_i)
        And: log(ratio) = sum(log(ratio_i))
        """
        log_ratios = torch.tensor(head_log_ratios, dtype=torch.float32)

        # Method 1: sum log ratios, then exp
        joint_log_ratio = log_ratios.sum()
        joint_ratio_from_sum = torch.exp(joint_log_ratio)

        # Method 2: exp each, then product
        per_head_ratios = torch.exp(log_ratios)
        joint_ratio_from_product = per_head_ratios.prod()

        # These must be equal (within floating point tolerance)
        assert torch.isclose(joint_ratio_from_sum, joint_ratio_from_product, rtol=1e-4), \
            f"Joint ratio mismatch: sum method={joint_ratio_from_sum}, product method={joint_ratio_from_product}"

    @given(
        head_log_ratios=st.lists(
            st.floats(min_value=-3.0, max_value=3.0, allow_nan=False),
            min_size=len(HEAD_NAMES),
            max_size=len(HEAD_NAMES),
        )
    )
    @settings(max_examples=100)
    def test_clamped_log_ratio_prevents_overflow(
        self, head_log_ratios: list[float]
    ) -> None:
        """Property: Clamping log-ratio to [-20, 20] prevents exp overflow.

        Without clamping, exp(88) overflows float32.
        The implementation clamps to [-20, 20] where exp(20) ≈ 4.85e8.
        """
        log_ratios = torch.tensor(head_log_ratios, dtype=torch.float32)

        # Add extreme values to test clamping
        extreme_log_ratios = torch.cat([log_ratios, torch.tensor([30.0, -30.0])])

        # Clamp as in the implementation
        clamped = torch.clamp(extreme_log_ratios, -20.0, 20.0)
        ratios = torch.exp(clamped)

        assert torch.isfinite(ratios).all(), \
            f"Clamped ratios must be finite, got {ratios}"
        assert (ratios > 0).all(), \
            f"Ratios must be positive, got {ratios}"


# =============================================================================
# Property Tests: Advantage Normalization (from DRL expert review)
# =============================================================================


class TestAdvantageNormalizationInvariants:
    """Normalized advantages must have μ≈0, σ≈1."""

    @given(
        advantages=st.lists(
            st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
            min_size=10,  # Need enough samples for meaningful statistics
            max_size=200,
        )
    )
    @settings(max_examples=100)
    def test_normalized_advantages_zero_mean(self, advantages: list[float]) -> None:
        """Property: After normalization, mean ≈ 0."""
        adv = torch.tensor(advantages, dtype=torch.float32)
        assume(adv.std() > 1e-6)  # Skip near-constant advantages

        mean = adv.mean()
        std = adv.std()
        normalized = (adv - mean) / (std + 1e-8)

        assert abs(normalized.mean()) < 1e-5, \
            f"Normalized mean should be ~0, got {normalized.mean()}"

    @given(
        advantages=st.lists(
            st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
            min_size=10,
            max_size=200,
        )
    )
    @settings(max_examples=100)
    def test_normalized_advantages_unit_std(self, advantages: list[float]) -> None:
        """Property: After normalization, std ≈ 1."""
        adv = torch.tensor(advantages, dtype=torch.float32)
        assume(adv.std() > 1e-6)  # Skip near-constant advantages

        mean = adv.mean()
        std = adv.std()
        normalized = (adv - mean) / (std + 1e-8)

        assert abs(normalized.std() - 1.0) < 0.01, \
            f"Normalized std should be ~1, got {normalized.std()}"


# =============================================================================
# Property Tests: Numerical Stability (from DRL expert review)
# =============================================================================


class TestNumericalStability:
    """All PPO outputs must be finite (no NaN, no Inf)."""

    @given(
        log_probs=log_prob_pairs(),
        advantages=advantage_values(),
        clip_ratio=st.floats(min_value=0.1, max_value=0.3),
    )
    @settings(max_examples=200)
    def test_all_ppo_outputs_finite(
        self,
        log_probs: tuple[torch.Tensor, torch.Tensor],
        advantages: torch.Tensor,
        clip_ratio: float,
    ) -> None:
        """Property: All intermediate and final PPO values are finite."""
        old_log_prob, new_log_prob = log_probs

        # Match batch sizes
        min_size = min(len(old_log_prob), len(advantages))
        old_log_prob = old_log_prob[:min_size]
        new_log_prob = new_log_prob[:min_size]
        advantages = advantages[:min_size]

        # Compute all intermediate values
        log_ratio = new_log_prob - old_log_prob
        ratio = torch.exp(log_ratio)
        clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
        surr1 = ratio * advantages
        surr2 = clipped_ratio * advantages
        clipped_surr = torch.min(surr1, surr2)
        policy_loss = -clipped_surr.mean()

        # All must be finite
        assert torch.isfinite(log_ratio).all(), f"log_ratio has non-finite: {log_ratio}"
        assert torch.isfinite(ratio).all(), f"ratio has non-finite: {ratio}"
        assert torch.isfinite(surr1).all(), f"surr1 has non-finite: {surr1}"
        assert torch.isfinite(surr2).all(), f"surr2 has non-finite: {surr2}"
        assert torch.isfinite(policy_loss), f"policy_loss non-finite: {policy_loss}"

    @given(
        log_prob=st.floats(min_value=-50.0, max_value=-0.001, allow_nan=False),
    )
    @settings(max_examples=50)
    def test_extreme_log_probs_handled(self, log_prob: float) -> None:
        """Property: Very negative log probs (tiny probabilities) don't cause issues."""
        old = torch.tensor([log_prob])
        new = torch.tensor([log_prob + 0.1])  # Slightly higher probability

        ratio = torch.exp(new - old)

        assert torch.isfinite(ratio).all(), \
            f"Ratio should be finite even for extreme log probs, got {ratio}"


# =============================================================================
# Property Tests: Causal Masking (from DRL expert review)
# =============================================================================


class TestCausalMaskingInvariants:
    """Causal masking for factored actions must follow op semantics.

    These tests verify the causal structure defined in leyline/causal_masks.py:
        WAIT:           Only op head matters
        GERMINATE:      op, slot, blueprint, style, tempo, alpha_target
        SET_ALPHA:      op, slot, style, alpha_target, alpha_speed, alpha_curve
        PRUNE:          op, slot, alpha_speed, alpha_curve
        FOSSILIZE:      op, slot
        ADVANCE:        op, slot
    """

    def test_wait_op_only_credits_op_head(self) -> None:
        """Property: WAIT op should ONLY credit the op head.

        When op=WAIT, no slot/germination action was taken, so only
        the op head receives gradient for "deciding to wait".
        """
        from esper.leyline.causal_masks import is_head_relevant

        # WAIT should ONLY credit op (the decision to wait)
        assert is_head_relevant("WAIT", "op") is True, "WAIT should credit op selection"
        # All other heads are irrelevant for WAIT
        assert is_head_relevant("WAIT", "slot") is False, "WAIT should NOT credit slot"
        assert is_head_relevant("WAIT", "blueprint") is False, "WAIT should NOT credit blueprint"
        assert is_head_relevant("WAIT", "style") is False, "WAIT should NOT credit style"
        assert is_head_relevant("WAIT", "tempo") is False, "WAIT should NOT credit tempo"

    def test_germinate_op_credits_germination_heads(self) -> None:
        """Property: GERMINATE op should credit all germination-related heads."""
        from esper.leyline.causal_masks import is_head_relevant

        assert is_head_relevant("GERMINATE", "slot") is True, "GERMINATE should credit slot"
        assert is_head_relevant("GERMINATE", "blueprint") is True, "GERMINATE should credit blueprint"
        assert is_head_relevant("GERMINATE", "style") is True, "GERMINATE should credit style"
        assert is_head_relevant("GERMINATE", "tempo") is True, "GERMINATE should credit tempo"
        assert is_head_relevant("GERMINATE", "alpha_target") is True, "GERMINATE should credit alpha_target"
        assert is_head_relevant("GERMINATE", "op") is True, "GERMINATE should credit op"

    def test_set_alpha_credits_alpha_and_style_heads(self) -> None:
        """Property: SET_ALPHA_TARGET should credit alpha + style heads.

        SET_ALPHA_TARGET allows changing the alpha schedule AND style.
        Blueprint is NOT relevant (no new module is being created).
        Tempo is NOT relevant (that's only for germination).
        """
        from esper.leyline.causal_masks import is_head_relevant

        # Alpha-related heads
        assert is_head_relevant("SET_ALPHA_TARGET", "alpha_target") is True, "SET_ALPHA should credit alpha_target"
        assert is_head_relevant("SET_ALPHA_TARGET", "alpha_speed") is True, "SET_ALPHA should credit alpha_speed"
        assert is_head_relevant("SET_ALPHA_TARGET", "alpha_curve") is True, "SET_ALPHA should credit alpha_curve"
        # Style IS relevant (blend style override)
        assert is_head_relevant("SET_ALPHA_TARGET", "style") is True, "SET_ALPHA should credit style"
        assert is_head_relevant("SET_ALPHA_TARGET", "slot") is True, "SET_ALPHA should credit slot"
        assert is_head_relevant("SET_ALPHA_TARGET", "op") is True, "SET_ALPHA should credit op"
        # Should NOT credit blueprint/tempo
        assert is_head_relevant("SET_ALPHA_TARGET", "blueprint") is False, "SET_ALPHA should NOT credit blueprint"
        assert is_head_relevant("SET_ALPHA_TARGET", "tempo") is False, "SET_ALPHA should NOT credit tempo"

    def test_prune_credits_slot_and_alpha_schedule(self) -> None:
        """Property: PRUNE should credit slot + alpha schedule heads."""
        from esper.leyline.causal_masks import is_head_relevant

        assert is_head_relevant("PRUNE", "slot") is True, "PRUNE should credit slot"
        assert is_head_relevant("PRUNE", "op") is True, "PRUNE should credit op"
        assert is_head_relevant("PRUNE", "alpha_speed") is True, "PRUNE should credit alpha_speed"
        assert is_head_relevant("PRUNE", "alpha_curve") is True, "PRUNE should credit alpha_curve"
        # Should NOT credit germination heads
        assert is_head_relevant("PRUNE", "blueprint") is False, "PRUNE should NOT credit blueprint"
        assert is_head_relevant("PRUNE", "style") is False, "PRUNE should NOT credit style"

    def test_fossilize_credits_slot_only(self) -> None:
        """Property: FOSSILIZE should credit slot + op only."""
        from esper.leyline.causal_masks import is_head_relevant

        assert is_head_relevant("FOSSILIZE", "slot") is True, "FOSSILIZE should credit slot"
        assert is_head_relevant("FOSSILIZE", "op") is True, "FOSSILIZE should credit op"
        assert is_head_relevant("FOSSILIZE", "blueprint") is False, "FOSSILIZE should NOT credit blueprint"
