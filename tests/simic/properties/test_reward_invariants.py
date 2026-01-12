"""Mathematical invariant tests for reward functions.

These properties MUST hold for ANY valid input - violations indicate bugs.

Tier 1: Mathematical Invariants
- Rewards are finite (no NaN/Inf)
- Rewards are bounded (within learnable PPO range)
- Components sum to total (reward composition is correct)
"""

import math
import pytest
from hypothesis import given, settings

from esper.simic.rewards import compute_contribution_reward

# Import strategies
from tests.simic.strategies import reward_inputs, reward_inputs_with_seed


@pytest.mark.property
class TestRewardFiniteness:
    """Rewards must always be finite numbers."""

    @given(inputs=reward_inputs())
    @settings(max_examples=500)
    def test_reward_is_finite(self, inputs):
        """Reward must never be NaN or Inf for any valid input."""
        reward = compute_contribution_reward(**inputs)
        assert math.isfinite(reward), f"Got non-finite reward: {reward}"

    @given(inputs=reward_inputs())
    @settings(max_examples=500)
    def test_components_are_finite(self, inputs):
        """All reward components must be finite."""
        reward, components = compute_contribution_reward(**inputs, return_components=True)

        assert math.isfinite(components.bounded_attribution), "bounded_attribution not finite"
        assert math.isfinite(components.pbrs_bonus), "pbrs_bonus not finite"
        assert math.isfinite(components.compute_rent), "compute_rent not finite"
        assert math.isfinite(components.alpha_shock), "alpha_shock not finite"
        assert math.isfinite(components.action_shaping), "action_shaping not finite"
        assert math.isfinite(components.terminal_bonus), "terminal_bonus not finite"
        assert math.isfinite(components.total_reward), "total_reward not finite"


@pytest.mark.property
class TestRewardBoundedness:
    """Rewards must stay within learnable PPO range."""

    # Terminal bonus can reach ~23 (100*0.05 + 6*3.0), attribution ~10, PBRS ~10
    # Total theoretical max ~45-50, but should not exceed reasonable bounds
    REWARD_BOUND = 50.0

    @given(inputs=reward_inputs())
    @settings(max_examples=500)
    def test_reward_is_bounded(self, inputs):
        """Reward should stay within learnable range [-50, 50]."""
        reward = compute_contribution_reward(**inputs)
        assert -self.REWARD_BOUND <= reward <= self.REWARD_BOUND, (
            f"Reward {reward} outside bounds [{-self.REWARD_BOUND}, {self.REWARD_BOUND}]"
        )


@pytest.mark.property
class TestRewardComposition:
    """Reward components must compose correctly."""

    @given(inputs=reward_inputs_with_seed())
    @settings(max_examples=300)
    def test_components_sum_to_total(self, inputs):
        """Sum of components should equal total reward (within tolerance)."""
        reward, components = compute_contribution_reward(**inputs, return_components=True)

        # Sum ALL reward components (must match total exactly)
        component_sum = (
            components.bounded_attribution
            + components.blending_warning
            + components.holding_warning
            + components.pbrs_bonus
            + components.compute_rent  # Already negative
            + components.alpha_shock
            + components.action_shaping
            + components.terminal_bonus
            + components.synergy_bonus  # B6-CR-01: was missing, caused silent failures
            # D2 capacity economics
            - components.occupancy_rent  # Subtracted: slots above threshold incur cost
            - components.fossilized_rent  # Subtracted: maintenance cost for fossilized seeds
            + components.first_germinate_bonus  # Added: breaks "do nothing" symmetry
        )

        assert abs(component_sum - reward) < 1e-6, (
            f"Component sum {component_sum} != total reward {reward}"
        )
