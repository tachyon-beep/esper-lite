"""Integration tests for Simic-Kasmina interaction.

Tests the core integration where:
- Simic computes rewards based on seed contribution (from validation)
- Rewards reflect seed lifecycle stage and contribution
- Gradient collection works correctly during seed training
"""

import pytest
import torch

from esper.kasmina import MorphogeneticModel, CNNHost
from esper.leyline import LifecycleOp, SeedStage
from esper.simic.rewards import (
    compute_contribution_reward,
    ContributionRewardConfig,
    SeedInfo,
)
from esper.simic.telemetry import (
    SeedGradientCollector,
    collect_seed_gradients,
    GradientHealthMetrics,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def model_with_seed():
    """Create MorphogeneticModel with an active seed in TRAINING stage."""
    host = CNNHost(num_classes=10)
    model = MorphogeneticModel(host, device="cpu", slots=["r0c1"])
    model.germinate_seed("conv_light", "test_seed", slot="r0c1")
    # Advance to TRAINING stage
    slot = model.seed_slots["r0c1"]
    slot.state.stage = SeedStage.TRAINING
    return model


@pytest.fixture
def reward_config():
    """Standard reward configuration."""
    return ContributionRewardConfig()


@pytest.fixture
def seed_info_training():
    """SeedInfo for a seed in TRAINING stage."""
    return SeedInfo(
        stage=SeedStage.TRAINING.value,
        improvement_since_stage_start=0.02,
        total_improvement=0.05,
        epochs_in_stage=3,
        seed_params=10000,
        previous_stage=SeedStage.GERMINATED.value,
        previous_epochs_in_stage=1,
        seed_age_epochs=4,
    )


# =============================================================================
# Reward Computation Tests
# =============================================================================


class TestRewardComputation:
    """Tests that reward signals reflect seed contribution correctly."""

    def test_positive_contribution_yields_positive_reward(
        self, seed_info_training, reward_config
    ):
        """Positive seed contribution should yield positive reward.

        This tests the Simic->Kasmina signal: when validation shows the seed
        helps accuracy, Simic's reward function should return positive reward.
        """
        reward, components = compute_contribution_reward(
            action=LifecycleOp.WAIT,
            seed_contribution=0.02,  # 2% improvement from seed
            val_acc=75.0,
            seed_info=seed_info_training,
            epoch=5,
            max_epochs=25,
            total_params=10000,
            host_params=1000000,
            acc_at_germination=70.0,
            acc_delta=0.01,
            config=reward_config,
            return_components=True,
            num_fossilized_seeds=0,
            num_contributing_fossilized=0,
        )

        # Positive contribution should yield net positive reward
        assert reward > 0, f"Expected positive reward, got {reward}"
        assert components.seed_contribution == 0.02

    def test_negative_contribution_reflected_in_components(
        self, seed_info_training, reward_config
    ):
        """Negative seed contribution should be reflected in reward components.

        When validation shows the seed hurts accuracy, the contribution component
        should be negative. Total reward may still be positive due to other
        components (PBRS, progress bonuses, etc).
        """
        reward, components = compute_contribution_reward(
            action=LifecycleOp.WAIT,
            seed_contribution=-0.03,  # 3% degradation from seed
            val_acc=72.0,
            seed_info=seed_info_training,
            epoch=5,
            max_epochs=25,
            total_params=10000,
            host_params=1000000,
            acc_at_germination=75.0,
            acc_delta=-0.02,
            config=reward_config,
            return_components=True,
            num_fossilized_seeds=0,
            num_contributing_fossilized=0,
        )

        # The seed contribution component should reflect the negative input
        assert components.seed_contribution == -0.03
        # The bounded_attribution (which weights the contribution) should be negative
        assert components.bounded_attribution < 0, (
            f"Bounded attribution should be negative, got {components.bounded_attribution}"
        )


# =============================================================================
# Gradient Collection Tests
# =============================================================================


class TestGradientCollection:
    """Tests that gradient collection works correctly during seed training."""

    def test_seed_gradients_collected_during_training(self, model_with_seed):
        """Gradient collection should capture seed gradients after backward."""
        model = model_with_seed
        slot = model.seed_slots["r0c1"]

        # Set to BLENDING so seed is active with non-zero alpha
        slot.state.stage = SeedStage.BLENDING
        slot.state.alpha = 0.5

        # Forward pass
        x = torch.randn(2, 3, 32, 32)
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Collect gradients from seed parameters
        seed_params = list(slot.seed.parameters())
        collector = SeedGradientCollector()
        stats = collector.collect(seed_params)

        assert stats["gradient_norm"] > 0, "Seed should have non-zero gradients"
        assert stats["gradient_health"] > 0, "Gradient health should be positive"

    def test_enhanced_gradient_metrics(self, model_with_seed):
        """Enhanced gradient collection returns GradientHealthMetrics."""
        model = model_with_seed
        slot = model.seed_slots["r0c1"]
        slot.state.stage = SeedStage.BLENDING
        slot.state.alpha = 0.5

        # Forward/backward
        x = torch.randn(2, 3, 32, 32)
        output = model(x)
        # Use mean to keep gradients well below exploding threshold in a deterministic test.
        loss = output.mean()
        loss.backward()

        # Collect with enhanced metrics
        seed_params = list(slot.seed.parameters())
        metrics = collect_seed_gradients(seed_params, return_enhanced=True)

        assert isinstance(metrics, GradientHealthMetrics)
        assert metrics.gradient_norm > 0
        assert not metrics.has_vanishing
        assert not metrics.has_exploding
