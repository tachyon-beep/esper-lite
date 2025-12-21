# tests/simic/properties/test_reward_antigaming.py
"""Anti-gaming property tests for reward functions.

These properties verify that emergent failure modes (discovered in production)
are prevented. Each test documents a specific exploit pattern.

Tier 3: Anti-Gaming Properties
- Ransomware pattern is penalized
- Fossilization farming is prevented
- Attribution discount applies to negative trajectories
- Ratio penalty catches dependency gaming
"""

import pytest
from hypothesis import given, settings

from esper.simic.rewards import (
    compute_contribution_reward,
    ContributionRewardConfig,
)

from tests.simic.strategies import (
    ransomware_seed_inputs,
    fossilize_inputs,
    reward_inputs_with_seed,
)


@pytest.mark.property
class TestRansomwarePattern:
    """Ransomware pattern: high contribution + negative total = exploit.

    A seed that creates dependencies (high counterfactual) but hurts overall
    performance (negative total_improvement) is gaming the system. The seed
    makes itself "necessary" without adding value - like ransomware.
    """

    @given(inputs=ransomware_seed_inputs())
    @settings(max_examples=300)
    def test_ransomware_not_rewarded(self, inputs):
        """Ransomware signature should result in low/negative reward.

        The seed_contribution is high (model "needs" the seed) but
        total_improvement is negative (model is worse WITH the seed).
        This pattern should never be rewarded.
        """
        # Skip terminal epoch cases - terminal bonus is legitimate and separate concern
        if inputs["epoch"] == inputs["max_epochs"]:
            return

        reward = compute_contribution_reward(**inputs)

        # Ransomware should NOT be profitable
        # Allow small positive due to other components, but attribution should be crushed
        assert reward <= 0.5, (
            f"Ransomware pattern got reward {reward}, expected <= 0.5. "
            f"seed_contribution={inputs['seed_contribution']}, "
            f"total_improvement={inputs['seed_info'].total_improvement}"
        )

    @given(inputs=ransomware_seed_inputs())
    @settings(max_examples=300)
    def test_attribution_discount_applied(self, inputs):
        """Attribution discount should crush rewards for negative trajectories."""
        _, components = compute_contribution_reward(**inputs, return_components=True)

        # With negative total_improvement, discount should be < 0.5
        assert components.attribution_discount < 0.5, (
            f"Negative total_improvement {inputs['seed_info'].total_improvement} "
            f"got discount {components.attribution_discount}, expected < 0.5"
        )


@pytest.mark.property
class TestFossilizationFarming:
    """Prevent fossilization farming - rushing to FOSSILIZED for bonuses."""

    @given(inputs=fossilize_inputs(valid=True))
    @settings(max_examples=200)
    def test_rapid_fossilize_discounted(self, inputs):
        """Fossilizing without sufficient HOLDING time is discounted.

        Seeds must "earn" fossilization by spending time in HOLDING.
        Rapid fossilization gets reduced bonus.
        """
        from esper.leyline import MIN_HOLDING_EPOCHS

        seed_info = inputs["seed_info"]
        epochs_in_hold = seed_info.epochs_in_stage

        _, components = compute_contribution_reward(**inputs, return_components=True)

        if epochs_in_hold < MIN_HOLDING_EPOCHS:
            config = ContributionRewardConfig()

            if inputs["seed_contribution"] and inputs["seed_contribution"] > 0:
                # Verify legitimacy discount applied - shaping should be less than max possible
                max_possible = (
                    config.fossilize_base_bonus
                    + config.fossilize_contribution_scale * inputs["seed_contribution"]
                )

                assert components.action_shaping < max_possible, (
                    f"Rapid fossilize (epoch {epochs_in_hold}/{MIN_HOLDING_EPOCHS}) should be discounted, "
                    f"got action_shaping={components.action_shaping}, max_possible={max_possible}"
                )

    @given(inputs=fossilize_inputs(valid=True))
    @settings(max_examples=200)
    def test_negative_improvement_fossilize_penalized(self, inputs):
        """Fossilizing a seed with negative total_improvement is penalized."""
        # Force negative total improvement
        seed_info = inputs["seed_info"]
        if seed_info.total_improvement >= 0:
            return  # Skip positive improvement cases

        _, components = compute_contribution_reward(**inputs, return_components=True)

        # Action shaping should be negative (penalty)
        assert components.action_shaping < 0, (
            f"Fossilizing negative-improvement seed got positive shaping: "
            f"{components.action_shaping}"
        )


@pytest.mark.property
class TestRatioPenalty:
    """Ratio penalty catches contribution >> improvement (dependency gaming)."""

    @given(inputs=reward_inputs_with_seed())
    @settings(max_examples=300)
    def test_high_ratio_penalized(self, inputs):
        """When contribution vastly exceeds improvement, apply ratio penalty.

        High contribution/improvement ratio suggests the seed created
        dependencies rather than genuine value.
        """
        seed_info = inputs["seed_info"]
        seed_contribution = inputs["seed_contribution"]

        # Need counterfactual data
        if seed_contribution is None or seed_contribution <= 1.0:
            return

        total_imp = seed_info.total_improvement
        if total_imp <= 0.1:
            return  # Covered by attribution discount

        ratio = seed_contribution / total_imp
        if ratio <= 5.0:
            return  # Not suspicious

        _, components = compute_contribution_reward(**inputs, return_components=True)

        # Should have ratio penalty applied
        assert components.ratio_penalty < 0, (
            f"Ratio {ratio:.1f} (contribution={seed_contribution}, "
            f"improvement={total_imp}) should trigger ratio_penalty, "
            f"got {components.ratio_penalty}"
        )
