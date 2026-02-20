"""Semantic invariant tests for reward functions.

These properties verify domain-specific rules that must always hold.

Tier 2: Semantic Invariants
- Fossilized seeds don't generate attribution rewards
- PRUNE inverts attribution signal
- Invalid actions are always penalized
- Terminal bonus only at terminal epoch
"""

import pytest
from hypothesis import given, settings

from esper.leyline import LifecycleOp, SeedStage
from esper.simic.rewards import (
    compute_contribution_reward,
    ContributionRewardConfig,
    STAGE_FOSSILIZED,
)

from tests.simic.strategies import (
    reward_inputs,
    reward_inputs_with_seed,
    fossilize_inputs,
    prune_inputs,
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
    """PRUNE action should invert attribution signal."""

    @given(inputs=reward_inputs_with_seed())
    @settings(max_examples=300)
    def test_prune_inverts_attribution(self, inputs):
        """Pruning good seed = bad, pruning bad seed = good.

        Without inversion, policy learns 'PRUNE everything for +attribution rewards'.

        Note: The sign-inversion property applies to the *attribution* component
        only, not to ratio_penalty.  ratio_penalty is an anti-gaming term that
        should always be ≤ 0 regardless of action type.  When the attribution
        component is zero (e.g. zero progress) and only ratio_penalty remains,
        both WAIT and PRUNE correctly show negative bounded_attribution.
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

        # Get attribution with PRUNE
        prune_test_inputs = {**inputs, "action": LifecycleOp.PRUNE}
        _, comp_prune = compute_contribution_reward(**prune_test_inputs, return_components=True)

        # Extract attribution-only component (without ratio_penalty)
        # ratio_penalty is applied identically to both actions and should
        # not participate in the sign-inversion check.
        wait_attribution = comp_wait.bounded_attribution - comp_wait.ratio_penalty
        prune_attribution = comp_prune.bounded_attribution - comp_prune.ratio_penalty

        # If WAIT gave meaningful attribution, PRUNE should invert it
        if abs(wait_attribution) > 0.01:
            assert prune_attribution * wait_attribution <= 0, (
                f"PRUNE attribution {prune_attribution} should oppose "
                f"WAIT attribution {wait_attribution} "
                f"(ratio_penalty={comp_wait.ratio_penalty} excluded from check)"
            )


@pytest.mark.property
class TestInvalidActionPenalties:
    """Invalid lifecycle actions must always be penalized."""

    @given(inputs=fossilize_inputs(valid=False))
    @settings(max_examples=200)
    def test_invalid_fossilize_penalized(self, inputs):
        """FOSSILIZE from non-HOLDING stage should be penalized."""
        config = ContributionRewardConfig()
        reward, components = compute_contribution_reward(**inputs, return_components=True)

        # Action shaping should include the penalty
        assert components.action_shaping <= config.invalid_fossilize_penalty, (
            f"Invalid FOSSILIZE got action_shaping {components.action_shaping}, "
            f"expected <= {config.invalid_fossilize_penalty}"
        )

    @given(inputs=prune_inputs(valid=False))
    @settings(max_examples=200)
    def test_prune_fossilized_penalized(self, inputs):
        """PRUNE on fossilized seed should be penalized."""
        if inputs["seed_info"].stage != STAGE_FOSSILIZED:
            return  # This test specifically for fossilized

        config = ContributionRewardConfig()
        reward, components = compute_contribution_reward(**inputs, return_components=True)

        assert components.action_shaping <= config.prune_fossilized_penalty, (
            f"PRUNE fossilized got action_shaping {components.action_shaping}, "
            f"expected <= {config.prune_fossilized_penalty}"
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
