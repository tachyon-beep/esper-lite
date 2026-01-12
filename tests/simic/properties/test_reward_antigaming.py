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
from hypothesis import given, settings, strategies as st

from esper.simic.rewards import (
    compute_contribution_reward,
    ContributionRewardConfig,
)
from esper.simic.rewards.contribution import FossilizedSeedDripState

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

    @given(inputs=ransomware_seed_inputs())
    @settings(max_examples=300)
    def test_ransomware_no_synergy_bonus(self, inputs):
        """B6-DRL-03: Ransomware seeds should receive zero synergy bonus.

        The anti-stacking gate blocks synergy_bonus for seeds with
        negative total_improvement, regardless of interaction_sum.
        This prevents ransomware seeds from gaming the synergy system.
        """
        _, components = compute_contribution_reward(**inputs, return_components=True)

        assert components.synergy_bonus == 0.0, (
            f"Ransomware seed received synergy_bonus={components.synergy_bonus}, "
            f"expected 0.0. attribution_discount={components.attribution_discount}, "
            f"bounded_attribution={components.bounded_attribution}"
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
        import math

        from esper.leyline import MIN_HOLDING_EPOCHS

        seed_info = inputs["seed_info"]
        epochs_in_hold = seed_info.epochs_in_stage

        _, components = compute_contribution_reward(**inputs, return_components=True)

        if epochs_in_hold < MIN_HOLDING_EPOCHS:
            config = ContributionRewardConfig()

            if inputs["seed_contribution"] and inputs["seed_contribution"] > 0:
                # Verify legitimacy discount applied - shaping should be less than max possible.
                # Max possible includes:
                # 1. fossilize_base_bonus + contribution_scale (from _contribution_fossilize_shaping)
                # 2. immediate_fossilize_bonus (from P0 fix 2026-01-11)
                # Both have legitimacy_discount applied, so undiscounted max should exceed actual.
                max_possible = (
                    config.fossilize_base_bonus
                    + config.fossilize_contribution_scale * inputs["seed_contribution"]
                    + config.fossilize_terminal_scale
                    * math.tanh(1.0 / config.fossilize_quality_ceiling)
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


@pytest.mark.property
class TestDripAntiGaming:
    """Property tests for drip reward anti-gaming guarantees."""

    @given(
        fossilize_epoch=st.integers(min_value=10, max_value=140),
        contribution_at_foss=st.floats(min_value=1.0, max_value=10.0),
    )
    @settings(max_examples=100)
    def test_epoch_normalization_prevents_early_gaming(
        self, fossilize_epoch: int, contribution_at_foss: float
    ) -> None:
        """Expected total drip is roughly equal regardless of fossilization timing.

        Without epoch normalization, early fossilization would capture more
        total drip. Normalization by remaining_epochs should equalize this.
        """
        max_epochs = 150
        config = ContributionRewardConfig(
            drip_fraction=0.7,
            max_drip_per_epoch=0.1,
            min_drip_epochs=5,
        )

        remaining = max(max_epochs - fossilize_epoch, config.min_drip_epochs)

        # Simulate drip with constant contribution
        drip_state = FossilizedSeedDripState(
            seed_id="test",
            slot_id="r0c1",
            fossilize_epoch=fossilize_epoch,
            max_epochs=max_epochs,
            drip_total=2.0,
            drip_scale=2.0 / remaining,
        )

        # Sum drip over remaining epochs with constant contribution
        total_drip = 0.0
        actual_remaining = max_epochs - fossilize_epoch
        for _ in range(actual_remaining):
            epoch_drip = drip_state.compute_epoch_drip(
                current_contribution=contribution_at_foss,
                max_drip=config.max_drip_per_epoch,
                negative_drip_ratio=0.5,
            )
            total_drip += epoch_drip

        # Expected total (without clipping): drip_total * contribution
        expected_max = 2.0 * contribution_at_foss

        # Total should be bounded regardless of timing
        assert total_drip <= expected_max + 0.1, (
            f"Total drip {total_drip} exceeds expected max {expected_max} "
            f"for fossilize_epoch={fossilize_epoch}"
        )

    @given(
        contribution_sequence=st.lists(
            st.floats(min_value=-5.0, max_value=5.0),
            min_size=10,
            max_size=50,
        )
    )
    @settings(max_examples=100)
    def test_negative_drip_for_degrading_seeds(
        self, contribution_sequence: list[float]
    ) -> None:
        """Seeds that degrade post-fossilization receive negative drip (penalty)."""
        config = ContributionRewardConfig(
            drip_fraction=0.7,
            max_drip_per_epoch=0.1,
            negative_drip_ratio=0.5,
        )

        drip_state = FossilizedSeedDripState(
            seed_id="test",
            slot_id="r0c1",
            fossilize_epoch=100,
            max_epochs=150,
            drip_total=2.0,
            drip_scale=2.0 / 50,
        )

        total_drip = 0.0
        negative_epochs = 0
        for contrib in contribution_sequence:
            epoch_drip = drip_state.compute_epoch_drip(
                current_contribution=contrib,
                max_drip=config.max_drip_per_epoch,
                negative_drip_ratio=0.5,
            )
            total_drip += epoch_drip
            if contrib < 0:
                negative_epochs += 1
                # Negative contribution should produce negative or zero drip
                assert epoch_drip <= 0, (
                    f"Negative contribution {contrib} produced positive drip {epoch_drip}"
                )

        # If mostly negative contributions, total drip should be negative
        if negative_epochs > len(contribution_sequence) * 0.7:
            assert total_drip < 0, (
                f"Mostly negative contributions ({negative_epochs}/{len(contribution_sequence)}) "
                f"should produce negative total drip, got {total_drip}"
            )

    @given(
        drip_scale=st.floats(min_value=0.01, max_value=1.0),
        contribution=st.floats(min_value=-10.0, max_value=10.0),
    )
    @settings(max_examples=100)
    def test_asymmetric_clipping_invariant(
        self, drip_scale: float, contribution: float
    ) -> None:
        """Asymmetric clipping: positive cap is 2x the negative cap."""
        drip_state = FossilizedSeedDripState(
            seed_id="test",
            slot_id="r0c1",
            fossilize_epoch=100,
            max_epochs=150,
            drip_total=2.0,
            drip_scale=drip_scale,
        )

        max_drip = 0.1
        negative_ratio = 0.5

        drip = drip_state.compute_epoch_drip(
            current_contribution=contribution,
            max_drip=max_drip,
            negative_drip_ratio=negative_ratio,
        )

        # Invariant: drip is always in [-0.05, +0.1]
        assert drip >= -max_drip * negative_ratio - 1e-9
        assert drip <= max_drip + 1e-9
