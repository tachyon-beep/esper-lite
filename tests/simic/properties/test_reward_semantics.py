"""Semantic invariant tests for reward functions.

These properties verify domain-specific rules that must always hold.

Tier 2: Semantic Invariants
- Fossilized seeds don't generate attribution rewards
- CULL inverts attribution signal
- Invalid actions are always penalized
- Terminal bonus only at terminal epoch
"""

import pytest
from hypothesis import given, settings

from esper.leyline import SeedStage
from esper.leyline.factored_actions import LifecycleOp
from esper.simic.rewards import (
    compute_contribution_reward,
    ContributionRewardConfig,
    STAGE_FOSSILIZED,
)

from tests.simic.strategies import (
    reward_inputs,
    reward_inputs_with_seed,
    fossilize_inputs,
    cull_inputs,
)


@pytest.mark.property
class TestFossilizedSeedBehavior:
    """Fossilized seeds should not generate ongoing rewards."""

    @given(inputs=reward_inputs_with_seed())
    @settings(max_examples=300)
    def test_fossilized_seed_no_attribution(self, inputs):
        """Fossilized seeds should not generate attribution rewards.

        Bug history: Without the check at line 405-407 of rewards.py,
        fossilized seeds continued generating high rewards indefinitely.
        """
        # Force seed to FOSSILIZED stage
        if inputs["seed_info"].stage != STAGE_FOSSILIZED:
            return  # Skip, handled by other tests

        reward, components = compute_contribution_reward(**inputs, return_components=True)

        assert components.bounded_attribution == 0.0, (
            f"Fossilized seed generated attribution: {components.bounded_attribution}"
        )


@pytest.mark.property
class TestCullBehavior:
    """CULL action should invert attribution signal."""

    @given(inputs=reward_inputs_with_seed())
    @settings(max_examples=300)
    def test_cull_inverts_attribution(self, inputs):
        """Culling good seed = bad, culling bad seed = good.

        Without inversion, policy learns 'CULL everything for +attribution rewards'.
        """
        # Need a seed with counterfactual (BLENDING+ stage)
        seed_info = inputs["seed_info"]
        if seed_info.stage < SeedStage.BLENDING.value:
            return  # No counterfactual, skip
        if seed_info.stage == STAGE_FOSSILIZED:
            return  # Can't cull fossilized, skip

        # Get attribution with WAIT (baseline)
        wait_inputs = {**inputs, "action": LifecycleOp.WAIT}
        _, comp_wait = compute_contribution_reward(**wait_inputs, return_components=True)

        # Get attribution with CULL
        cull_test_inputs = {**inputs, "action": LifecycleOp.CULL}
        _, comp_cull = compute_contribution_reward(**cull_test_inputs, return_components=True)

        # If WAIT gave positive attribution, CULL should give negative (and vice versa)
        if abs(comp_wait.bounded_attribution) > 0.01:
            assert comp_cull.bounded_attribution * comp_wait.bounded_attribution <= 0, (
                f"CULL attribution {comp_cull.bounded_attribution} should oppose "
                f"WAIT attribution {comp_wait.bounded_attribution}"
            )


@pytest.mark.property
class TestInvalidActionPenalties:
    """Invalid lifecycle actions must always be penalized."""

    @given(inputs=fossilize_inputs(valid=False))
    @settings(max_examples=200)
    def test_invalid_fossilize_penalized(self, inputs):
        """FOSSILIZE from non-PROBATIONARY stage should be penalized."""
        config = ContributionRewardConfig()
        reward, components = compute_contribution_reward(**inputs, return_components=True)

        # Action shaping should include the penalty
        assert components.action_shaping <= config.invalid_fossilize_penalty, (
            f"Invalid FOSSILIZE got action_shaping {components.action_shaping}, "
            f"expected <= {config.invalid_fossilize_penalty}"
        )

    @given(inputs=cull_inputs(valid=False))
    @settings(max_examples=200)
    def test_cull_fossilized_penalized(self, inputs):
        """CULL on fossilized seed should be penalized."""
        if inputs["seed_info"].stage != STAGE_FOSSILIZED:
            return  # This test specifically for fossilized

        config = ContributionRewardConfig()
        reward, components = compute_contribution_reward(**inputs, return_components=True)

        assert components.action_shaping <= config.cull_fossilized_penalty, (
            f"CULL fossilized got action_shaping {components.action_shaping}, "
            f"expected <= {config.cull_fossilized_penalty}"
        )


@pytest.mark.property
class TestTerminalBonus:
    """Terminal bonus should only apply at terminal epoch."""

    @given(inputs=reward_inputs())
    @settings(max_examples=300)
    def test_terminal_bonus_only_at_terminal(self, inputs):
        """Terminal bonus should be zero unless epoch == max_epochs."""
        reward, components = compute_contribution_reward(**inputs, return_components=True)

        is_terminal = inputs["epoch"] == inputs["max_epochs"]

        if not is_terminal:
            assert components.terminal_bonus == 0.0, (
                f"Non-terminal epoch {inputs['epoch']} got terminal_bonus: "
                f"{components.terminal_bonus}"
            )
        # Note: At terminal, bonus CAN be zero if val_acc is 0 and no fossilized seeds
