"""Smoke tests to verify strategies generate valid inputs.

If SeedInfo or ContributionRewardConfig fields change, these tests
will fail immediately rather than causing silent strategy breakage.
"""

from hypothesis import given, settings

from esper.simic.rewards import compute_contribution_reward

from tests.simic.strategies import (
    reward_inputs,
    reward_inputs_with_seed,
    ransomware_seed_inputs,
    fossilize_inputs,
    prune_inputs,
)


class TestStrategiesGenerateValidInputs:
    """Verify all strategies produce inputs that compute_contribution_reward accepts."""

    @given(inputs=reward_inputs())
    @settings(max_examples=50)
    def test_reward_inputs_valid(self, inputs):
        """reward_inputs() generates valid inputs."""
        # Should not raise
        compute_contribution_reward(**inputs)

    @given(inputs=reward_inputs_with_seed())
    @settings(max_examples=50)
    def test_reward_inputs_with_seed_valid(self, inputs):
        """reward_inputs_with_seed() generates valid inputs."""
        compute_contribution_reward(**inputs)

    @given(inputs=ransomware_seed_inputs())
    @settings(max_examples=50)
    def test_ransomware_inputs_valid(self, inputs):
        """ransomware_seed_inputs() generates valid inputs."""
        compute_contribution_reward(**inputs)

    @given(inputs=fossilize_inputs(valid=True))
    @settings(max_examples=50)
    def test_fossilize_inputs_valid(self, inputs):
        """fossilize_inputs() generates valid inputs."""
        compute_contribution_reward(**inputs)

    @given(inputs=prune_inputs(valid=True))
    @settings(max_examples=50)
    def test_prune_inputs_valid(self, inputs):
        """prune_inputs() generates valid inputs."""
        compute_contribution_reward(**inputs)
