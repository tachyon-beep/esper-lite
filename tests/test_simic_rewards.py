"""Tests for reward shaping functions."""

import pytest
from enum import IntEnum

from esper.leyline import MIN_CULL_AGE
from esper.simic.rewards import (
    SeedInfo,
    STAGE_BLENDING,
    STAGE_GERMINATED,
    STAGE_TRAINING,
    STAGE_PROBATIONARY,
    STAGE_FOSSILIZED,
    compute_seed_potential,
    compute_contribution_reward,
    _contribution_cull_shaping,
    _contribution_fossilize_shaping,
    ContributionRewardConfig,
)


class TestComputeSeedPotential:
    """Tests for potential-based reward shaping.

    SeedStage enum values (from leyline):
    - DORMANT=1, GERMINATED=2, TRAINING=3, BLENDING=4
    - SHADOWING=5, PROBATIONARY=6, FOSSILIZED=7
    """

    def test_no_seed_returns_zero(self):
        """Test that no active seed has zero potential."""
        obs = {'has_active_seed': 0, 'seed_stage': 0, 'seed_epochs_in_stage': 0}
        assert compute_seed_potential(obs) == 0.0

    def test_dormant_returns_zero(self):
        """Test that DORMANT stage (1) has zero potential."""
        obs = {'has_active_seed': 1, 'seed_stage': 1, 'seed_epochs_in_stage': 0}
        assert compute_seed_potential(obs) == 0.0

    def test_germinated_has_low_potential(self):
        """Test GERMINATED stage (2) has low potential."""
        obs = {'has_active_seed': 1, 'seed_stage': 2, 'seed_epochs_in_stage': 0}
        potential = compute_seed_potential(obs)
        assert potential == 1.0  # From unified STAGE_POTENTIALS

    def test_training_has_higher_potential(self):
        """Test TRAINING stage (3) has higher potential than GERMINATED (2)."""
        germ = {'has_active_seed': 1, 'seed_stage': 2, 'seed_epochs_in_stage': 0}
        train = {'has_active_seed': 1, 'seed_stage': 3, 'seed_epochs_in_stage': 0}

        assert compute_seed_potential(train) > compute_seed_potential(germ)

    def test_blending_has_high_potential(self):
        """Test BLENDING stage (4) has higher potential than earlier stages."""
        obs = {'has_active_seed': 1, 'seed_stage': 4, 'seed_epochs_in_stage': 0}
        potential = compute_seed_potential(obs)
        assert potential == 3.5  # From unified STAGE_POTENTIALS

    def test_probationary_has_highest_potential(self):
        """Test PROBATIONARY stage (6) has high potential before FOSSILIZED."""
        obs = {'has_active_seed': 1, 'seed_stage': 6, 'seed_epochs_in_stage': 0}
        potential = compute_seed_potential(obs)
        assert potential == 5.5  # From unified STAGE_POTENTIALS

    def test_fossilized_has_highest_potential(self):
        """Test FOSSILIZED stage (7) has slightly higher potential than PROBATIONARY."""
        obs = {'has_active_seed': 1, 'seed_stage': 7, 'seed_epochs_in_stage': 0}
        potential = compute_seed_potential(obs)
        assert potential == 6.0  # Small +0.5 over PROBATIONARY, not a farming target

    def test_progress_bonus_capped(self):
        """Test that progress bonus is capped at 2.0."""
        obs = {'has_active_seed': 1, 'seed_stage': 3, 'seed_epochs_in_stage': 100}
        potential = compute_seed_potential(obs)
        # Base 2.0 (TRAINING) + max 2.0 progress = 4.0
        assert potential == 4.0

    def test_stage_progression_increases_potential(self):
        """Test that potential generally increases through stages until FOSSILIZED."""
        potentials = []
        for stage in [2, 3, 4, 5, 6]:  # GERMINATED through PROBATIONARY
            obs = {'has_active_seed': 1, 'seed_stage': stage, 'seed_epochs_in_stage': 0}
            potentials.append(compute_seed_potential(obs))

        # Each stage should have higher potential than the previous
        for i in range(1, len(potentials)):
            assert potentials[i] > potentials[i-1], f"Stage {i+2} should have higher potential than stage {i+1}"


class TestPBRSStageBonus:
    """Tests for PBRS stage bonus behaviour using unified reward."""

    class _TestAction(IntEnum):
        NOOP = 0

    def test_transition_bonus_not_repeated_in_stage(self):
        """PBRS bonus for TRAINING->BLENDING transition should not repeat every epoch."""
        action = self._TestAction.NOOP

        # Step immediately after TRAINING->BLENDING transition
        seed_step1 = SeedInfo(
            stage=STAGE_BLENDING,
            improvement_since_stage_start=0.0,
            total_improvement=0.0,
            epochs_in_stage=0,
            seed_params=0,
            previous_stage=STAGE_TRAINING,
            seed_age_epochs=0,
        )

        r1 = compute_contribution_reward(
            action=action,
            seed_contribution=None,  # Proxy signal path
            val_acc=0.0,
            seed_info=seed_step1,
            epoch=0,
            max_epochs=10,
            total_params=0,
            host_params=1,
            acc_delta=0.0,
        )

        # Next epoch staying in BLENDING (no new transition)
        seed_step2 = seed_step1._replace(epochs_in_stage=1)
        r2 = compute_contribution_reward(
            action=action,
            seed_contribution=None,
            val_acc=0.0,
            seed_info=seed_step2,
            epoch=1,
            max_epochs=10,
            total_params=0,
            host_params=1,
            acc_delta=0.0,
        )

        # Transition bonus should be strictly positive and strictly larger than
        # any subsequent in-stage PBRS bonuses.
        assert r1 > 0.0
        assert 0.0 < r2 < r1


class TestCullContributionShaping:
    """Tests for CULL action shaping in contribution reward."""

    def _make_seed_info(self, stage: int, age: int, improvement: float = 0.0) -> SeedInfo:
        return SeedInfo(
            stage=stage,
            improvement_since_stage_start=improvement,
            total_improvement=improvement,
            epochs_in_stage=age,
            seed_params=1000,
            previous_stage=STAGE_GERMINATED,
            seed_age_epochs=age,
        )

    def test_cull_toxic_seed_rewarded(self):
        """Culling a toxic seed (negative contribution) should be rewarded."""
        config = ContributionRewardConfig()
        # Use age >= MIN_CULL_AGE to avoid age penalty
        seed_info = self._make_seed_info(STAGE_BLENDING, age=MIN_CULL_AGE, improvement=-1.0)

        # Toxic seed: contribution < hurting_threshold (-0.5)
        shaping = _contribution_cull_shaping(seed_info, seed_contribution=-1.0, config=config)

        # Should get positive shaping (reward for culling toxic seed)
        assert shaping > 0, f"Culling toxic seed should be rewarded: {shaping}"

    def test_cull_good_seed_penalized(self):
        """Culling a good seed (positive contribution) should be penalized."""
        config = ContributionRewardConfig()
        # Use age >= MIN_CULL_AGE to avoid age penalty interfering with test
        seed_info = self._make_seed_info(STAGE_BLENDING, age=MIN_CULL_AGE, improvement=2.0)

        # Good seed: contribution > 0
        shaping = _contribution_cull_shaping(seed_info, seed_contribution=2.0, config=config)

        # Should get negative shaping (penalty for culling good seed)
        assert shaping < 0, f"Culling good seed should be penalized: {shaping}"


class TestWaitBlendingShaping:
    """Tests for WAIT action at BLENDING stage using unified reward."""

    class _TestAction(IntEnum):
        WAIT = 0

    def _make_seed_info(self, stage: int, improvement: float, epochs: int = 1) -> SeedInfo:
        return SeedInfo(
            stage=stage,
            improvement_since_stage_start=improvement,
            total_improvement=improvement,
            epochs_in_stage=epochs,
            seed_params=0,
            previous_stage=STAGE_TRAINING,
            seed_age_epochs=epochs,
        )

    def test_wait_at_blending_with_positive_contribution(self):
        """WAIT at BLENDING with positive contribution should be positive reward."""
        seed_info = self._make_seed_info(STAGE_BLENDING, improvement=1.0)

        reward = compute_contribution_reward(
            action=self._TestAction.WAIT,
            seed_contribution=1.0,  # Positive contribution
            val_acc=70.0,
            seed_info=seed_info,
            epoch=5,
            max_epochs=25,
            acc_at_germination=65.0,  # 5% progress
        )

        # Positive contribution + PBRS should give positive reward
        assert reward > 0, f"WAIT with positive contribution should be positive: {reward}"


class TestFossilizeLegitimacyDiscount:
    """Fossilization bonus should be discounted for rapid fossilization."""

    def test_short_probation_gets_discounted(self):
        """Seeds with short PROBATIONARY get reduced fossilize bonus."""
        from esper.leyline import MIN_PROBATION_EPOCHS

        config = ContributionRewardConfig()

        # Seed with very short probation (1 epoch)
        short_probation = SeedInfo(
            stage=STAGE_PROBATIONARY,
            improvement_since_stage_start=5.0,
            total_improvement=10.0,
            epochs_in_stage=1,  # Just entered probation
            seed_age_epochs=15,
        )

        # Seed with full probation
        full_probation = SeedInfo(
            stage=STAGE_PROBATIONARY,
            improvement_since_stage_start=5.0,
            total_improvement=10.0,
            epochs_in_stage=MIN_PROBATION_EPOCHS,  # Full probation
            seed_age_epochs=20,
        )

        short_bonus = _contribution_fossilize_shaping(short_probation, 3.0, config)
        full_bonus = _contribution_fossilize_shaping(full_probation, 3.0, config)

        # Short probation should get less bonus
        assert short_bonus < full_bonus, (
            f"Short probation ({short_bonus}) should be less than full ({full_bonus})"
        )

        # Discount should be proportional
        expected_discount = 1 / MIN_PROBATION_EPOCHS
        assert short_bonus == pytest.approx(full_bonus * expected_discount, rel=0.01)

    def test_zero_probation_gets_zero_bonus(self):
        """Seeds with 0 epochs in PROBATIONARY get no bonus."""
        config = ContributionRewardConfig()
        seed = SeedInfo(
            stage=STAGE_PROBATIONARY,
            improvement_since_stage_start=5.0,
            total_improvement=10.0,
            epochs_in_stage=0,  # Just entered, no validation yet
            seed_age_epochs=15,
        )

        bonus = _contribution_fossilize_shaping(seed, 3.0, config)

        # Should get penalty, not bonus (legitimacy_discount = 0)
        assert bonus <= 0, f"Zero probation should not get positive bonus: {bonus}"


class TestContributionRewardComponents:
    """Tests for compute_contribution_reward return_components."""

    def test_return_components_returns_tuple(self):
        """Test that return_components=True returns (reward, components) tuple."""
        from esper.simic.reward_telemetry import RewardComponentsTelemetry
        from esper.leyline import SeedStage

        # Create a mock action enum
        from enum import IntEnum
        class MockAction(IntEnum):
            WAIT = 0

        seed_info = SeedInfo(
            stage=SeedStage.TRAINING.value,
            improvement_since_stage_start=1.0,
            total_improvement=1.0,
            epochs_in_stage=5,
            seed_params=1000,
            previous_stage=SeedStage.GERMINATED.value,
            seed_age_epochs=5,
        )

        result = compute_contribution_reward(
            action=MockAction.WAIT,
            seed_contribution=2.0,
            val_acc=70.0,
            seed_info=seed_info,
            epoch=10,
            max_epochs=25,
            total_params=1000,
            host_params=10000,
            acc_at_germination=65.0,
            return_components=True,
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        reward, components = result
        assert isinstance(reward, float)
        # Use isinstance instead of hasattr (per CLAUDE.md policy)
        assert isinstance(components, RewardComponentsTelemetry)

    def test_components_sum_to_total(self):
        """Test that component values sum to total_reward."""
        from esper.leyline import SeedStage

        from enum import IntEnum
        class MockAction(IntEnum):
            WAIT = 0

        seed_info = SeedInfo(
            stage=SeedStage.BLENDING.value,
            improvement_since_stage_start=2.0,
            total_improvement=3.0,
            epochs_in_stage=3,
            seed_params=5000,
            previous_stage=SeedStage.TRAINING.value,
            seed_age_epochs=8,
        )

        reward, components = compute_contribution_reward(
            action=MockAction.WAIT,
            seed_contribution=3.0,
            val_acc=72.0,
            seed_info=seed_info,
            epoch=15,
            max_epochs=25,
            total_params=5000,
            host_params=20000,
            acc_at_germination=68.0,
            return_components=True,
        )

        # Components should sum to total (within floating point tolerance)
        component_sum = (
            (components.bounded_attribution or 0.0)
            + components.compute_rent
            + components.pbrs_bonus
            + components.action_shaping
            + components.terminal_bonus
        )
        assert abs(reward - component_sum) < 0.001, f"Sum {component_sum} != total {reward}"
        assert components.total_reward == reward

    def test_components_track_context(self):
        """Test that components include action and epoch context."""
        from esper.leyline import SeedStage

        from enum import IntEnum
        class MockAction(IntEnum):
            CULL = 1

        seed_info = SeedInfo(
            stage=SeedStage.TRAINING.value,
            improvement_since_stage_start=-1.0,
            total_improvement=-1.0,
            epochs_in_stage=10,
            seed_params=1000,
            previous_stage=SeedStage.GERMINATED.value,
            seed_age_epochs=10,
        )

        reward, components = compute_contribution_reward(
            action=MockAction.CULL,
            seed_contribution=-0.5,
            val_acc=68.0,
            seed_info=seed_info,
            epoch=12,
            max_epochs=25,
            return_components=True,
        )

        assert components.action_name == "CULL"
        assert components.epoch == 12
        assert components.seed_stage == SeedStage.TRAINING.value

    def test_components_include_diagnostic_fields(self):
        """Test that components include DRL Expert recommended diagnostic fields."""
        from esper.leyline import SeedStage

        from enum import IntEnum
        class MockAction(IntEnum):
            WAIT = 0

        seed_info = SeedInfo(
            stage=SeedStage.BLENDING.value,
            improvement_since_stage_start=1.5,
            total_improvement=2.0,
            epochs_in_stage=5,
            seed_params=5000,
            previous_stage=SeedStage.TRAINING.value,
            seed_age_epochs=10,
        )

        reward, components = compute_contribution_reward(
            action=MockAction.WAIT,
            seed_contribution=2.5,
            val_acc=72.0,
            seed_info=seed_info,
            epoch=15,
            max_epochs=25,
            total_params=5000,
            host_params=20000,
            acc_at_germination=68.0,
            return_components=True,
        )

        # DRL Expert recommended fields
        assert components.val_acc == 72.0
        assert components.acc_at_germination == 68.0
        assert components.growth_ratio == 5000 / 20000  # 0.25
        assert components.progress_since_germination == 72.0 - 68.0  # 4.0


class TestProxySignalPath:
    """Tests for the proxy signal path (when seed_contribution is None)."""

    class _TestAction(IntEnum):
        WAIT = 0

    def test_positive_acc_delta_gives_reward(self):
        """Positive acc_delta should give positive bounded_attribution."""
        seed_info = SeedInfo(
            stage=STAGE_TRAINING,
            improvement_since_stage_start=1.0,
            total_improvement=1.0,
            epochs_in_stage=3,
            seed_params=1000,
            previous_stage=STAGE_GERMINATED,
            seed_age_epochs=3,
        )

        reward, components = compute_contribution_reward(
            action=self._TestAction.WAIT,
            seed_contribution=None,  # Proxy path
            val_acc=65.0,
            seed_info=seed_info,
            epoch=5,
            max_epochs=25,
            acc_delta=1.5,  # Positive delta
            return_components=True,
        )

        # Should have positive bounded_attribution from proxy signal
        assert components.bounded_attribution > 0, "Positive acc_delta should give positive attribution"
        # Should equal proxy_weight * acc_delta
        config = ContributionRewardConfig()
        expected = config.proxy_contribution_weight * 1.5
        assert components.bounded_attribution == pytest.approx(expected)

    def test_negative_acc_delta_gives_zero(self):
        """Negative acc_delta should give zero bounded_attribution (no penalty)."""
        seed_info = SeedInfo(
            stage=STAGE_TRAINING,
            improvement_since_stage_start=-0.5,
            total_improvement=-0.5,
            epochs_in_stage=3,
            seed_params=1000,
            previous_stage=STAGE_GERMINATED,
            seed_age_epochs=3,
        )

        reward, components = compute_contribution_reward(
            action=self._TestAction.WAIT,
            seed_contribution=None,  # Proxy path
            val_acc=64.0,
            seed_info=seed_info,
            epoch=5,
            max_epochs=25,
            acc_delta=-0.5,  # Negative delta
            return_components=True,
        )

        # No penalty for negative delta in proxy path
        assert components.bounded_attribution == 0.0

    def test_none_acc_delta_gives_zero(self):
        """None acc_delta should give zero bounded_attribution."""
        seed_info = SeedInfo(
            stage=STAGE_TRAINING,
            improvement_since_stage_start=0.0,
            total_improvement=0.0,
            epochs_in_stage=3,
            seed_params=1000,
            previous_stage=STAGE_GERMINATED,
            seed_age_epochs=3,
        )

        reward, components = compute_contribution_reward(
            action=self._TestAction.WAIT,
            seed_contribution=None,
            val_acc=65.0,
            seed_info=seed_info,
            epoch=5,
            max_epochs=25,
            acc_delta=None,  # No delta provided
            return_components=True,
        )

        assert components.bounded_attribution == 0.0


class TestRansomwareSeedDetection:
    """Tests for ransomware seed detection mechanisms.

    Ransomware pattern: seed has high counterfactual contribution (entangled)
    but negative total_improvement (hurt overall performance).
    """

    class _TestAction(IntEnum):
        WAIT = 0
        FOSSILIZE = 1

    def test_fossilize_penalty_for_negative_total_delta(self):
        """FOSSILIZE with negative total_improvement should be penalized."""
        config = ContributionRewardConfig()

        # Ransomware seed: high contribution but negative total delta
        ransomware_seed = SeedInfo(
            stage=STAGE_PROBATIONARY,
            improvement_since_stage_start=-0.3,
            total_improvement=-0.48,  # Hurt performance
            epochs_in_stage=3,
            seed_params=2000,
            previous_stage=STAGE_BLENDING,
            seed_age_epochs=15,
        )

        # High counterfactual contribution (entangled)
        penalty = _contribution_fossilize_shaping(ransomware_seed, 25.35, config)

        # Should get penalty, not bonus
        assert penalty < 0, f"Ransomware seed should get penalty: {penalty}"
        # Should be significant penalty (ransomware signature triggers)
        assert penalty < -0.5, f"Penalty should be significant: {penalty}"

    def test_fossilize_ransomware_signature_extra_penalty(self):
        """High contribution + negative delta should trigger extra ransomware penalty."""
        config = ContributionRewardConfig()

        # Ransomware signature: contribution > 0.1 and total_delta < -0.2
        ransomware_seed = SeedInfo(
            stage=STAGE_PROBATIONARY,
            improvement_since_stage_start=-0.5,
            total_improvement=-0.5,  # < -0.2 threshold
            epochs_in_stage=3,
            seed_age_epochs=15,
        )

        # Non-ransomware: same negative delta but low contribution
        non_ransomware_seed = SeedInfo(
            stage=STAGE_PROBATIONARY,
            improvement_since_stage_start=-0.5,
            total_improvement=-0.5,
            epochs_in_stage=3,
            seed_age_epochs=15,
        )

        ransomware_penalty = _contribution_fossilize_shaping(ransomware_seed, 15.0, config)  # High contribution
        regular_penalty = _contribution_fossilize_shaping(non_ransomware_seed, 0.05, config)  # Low contribution

        # Ransomware should get extra -0.3 penalty
        assert ransomware_penalty < regular_penalty, (
            f"Ransomware ({ransomware_penalty}) should be worse than regular ({regular_penalty})"
        )

    def test_blending_warning_escalates_over_epochs(self):
        """Blending warning should escalate as seed stays with negative trajectory."""
        from enum import IntEnum
        class MockAction(IntEnum):
            WAIT = 0

        def get_blending_warning(epochs_in_stage: int) -> float:
            seed_info = SeedInfo(
                stage=STAGE_BLENDING,
                improvement_since_stage_start=-0.3,
                total_improvement=-0.3,  # Negative trajectory
                epochs_in_stage=epochs_in_stage,
                seed_age_epochs=10 + epochs_in_stage,
            )
            _, components = compute_contribution_reward(
                action=MockAction.WAIT,
                seed_contribution=5.0,
                val_acc=60.0,
                seed_info=seed_info,
                epoch=10,
                max_epochs=25,
                acc_at_germination=60.5,
                return_components=True,
            )
            return components.blending_warning

        # Warning should escalate
        warning_epoch_1 = get_blending_warning(1)
        warning_epoch_3 = get_blending_warning(3)
        warning_epoch_6 = get_blending_warning(6)

        assert warning_epoch_1 < 0, "Should have negative warning"
        assert warning_epoch_3 < warning_epoch_1, "Warning should escalate"
        assert warning_epoch_6 < warning_epoch_3, "Warning should continue escalating"
        # Should cap at -0.4
        assert warning_epoch_6 >= -0.5, f"Warning should cap around -0.4: {warning_epoch_6}"

    def test_no_blending_warning_for_positive_trajectory(self):
        """No blending warning when total_improvement is positive."""
        from enum import IntEnum
        class MockAction(IntEnum):
            WAIT = 0

        seed_info = SeedInfo(
            stage=STAGE_BLENDING,
            improvement_since_stage_start=1.0,
            total_improvement=2.0,  # Positive trajectory
            epochs_in_stage=5,
            seed_age_epochs=15,
        )

        _, components = compute_contribution_reward(
            action=MockAction.WAIT,
            seed_contribution=5.0,
            val_acc=65.0,
            seed_info=seed_info,
            epoch=10,
            max_epochs=25,
            acc_at_germination=63.0,
            return_components=True,
        )

        assert components.blending_warning == 0.0, "No warning for positive trajectory"

    def test_attribution_discount_for_negative_total_improvement(self):
        """Attribution should be discounted when total_improvement is negative."""
        from enum import IntEnum
        class MockAction(IntEnum):
            WAIT = 0

        def get_attribution_and_discount(total_improvement: float) -> tuple[float, float]:
            seed_info = SeedInfo(
                stage=STAGE_BLENDING,
                improvement_since_stage_start=1.0,
                total_improvement=total_improvement,
                epochs_in_stage=5,
                seed_age_epochs=15,
            )
            _, components = compute_contribution_reward(
                action=MockAction.WAIT,
                seed_contribution=5.0,  # High contribution
                val_acc=65.0,
                seed_info=seed_info,
                epoch=10,
                max_epochs=25,
                acc_at_germination=60.0,  # 5% progress
                return_components=True,
            )
            return components.bounded_attribution, components.attribution_discount

        # Positive trajectory - no discount
        attr_positive, discount_positive = get_attribution_and_discount(2.0)
        assert discount_positive == 1.0, "No discount for positive trajectory"

        # Slightly negative - partial discount
        attr_slight_neg, discount_slight_neg = get_attribution_and_discount(-0.2)
        assert 0 < discount_slight_neg < 1.0, f"Should have partial discount: {discount_slight_neg}"
        assert attr_slight_neg < attr_positive, "Attribution should be reduced"

        # Very negative - heavy discount
        attr_very_neg, discount_very_neg = get_attribution_and_discount(-1.0)
        assert discount_very_neg < 0.1, f"Should have heavy discount: {discount_very_neg}"
        assert attr_very_neg < attr_slight_neg, "Attribution should be further reduced"

    def test_attribution_discount_sigmoid_values(self):
        """Verify sigmoid discount values at specific total_improvement levels."""
        import math
        from enum import IntEnum
        class MockAction(IntEnum):
            WAIT = 0

        def get_discount(total_improvement: float) -> float:
            seed_info = SeedInfo(
                stage=STAGE_BLENDING,
                improvement_since_stage_start=1.0,
                total_improvement=total_improvement,
                epochs_in_stage=5,
                seed_age_epochs=15,
            )
            _, components = compute_contribution_reward(
                action=MockAction.WAIT,
                seed_contribution=5.0,
                val_acc=65.0,
                seed_info=seed_info,
                epoch=10,
                max_epochs=25,
                acc_at_germination=60.0,
                return_components=True,
            )
            return components.attribution_discount

        # Test specific values from softened sigmoid (-5 coefficient)
        # At -0.5%, expect ~0.076
        discount_05 = get_discount(-0.5)
        assert 0.05 < discount_05 < 0.15, f"At -0.5%: {discount_05}"

        # At -0.2%, expect ~0.27
        discount_02 = get_discount(-0.2)
        assert 0.2 < discount_02 < 0.4, f"At -0.2%: {discount_02}"

        # At -1.0%, expect ~0.007
        discount_10 = get_discount(-1.0)
        assert discount_10 < 0.02, f"At -1.0%: {discount_10}"

    def test_fossilized_seeds_get_no_attribution(self):
        """FOSSILIZED seeds should not receive attribution rewards.

        After fossilization, the seed is permanent - no decision to be made.
        Continuing to reward based on counterfactual would inflate rewards
        indefinitely for envs with successful fossilized seeds.
        """
        from enum import IntEnum
        class MockAction(IntEnum):
            WAIT = 0

        # FOSSILIZED seed with high contribution
        fossilized_seed = SeedInfo(
            stage=STAGE_FOSSILIZED,
            improvement_since_stage_start=0.0,
            total_improvement=8.0,
            epochs_in_stage=10,
            seed_age_epochs=30,
        )

        _, components = compute_contribution_reward(
            action=MockAction.WAIT,
            seed_contribution=40.0,  # Very high contribution
            val_acc=75.0,
            seed_info=fossilized_seed,
            epoch=35,
            max_epochs=50,
            acc_at_germination=67.0,
            return_components=True,
        )

        # Should get zero attribution despite high seed_contribution
        assert components.bounded_attribution == 0.0, (
            f"FOSSILIZED seed should get no attribution: {components.bounded_attribution}"
        )

    def test_fossilize_negative_delta_gets_no_attribution(self):
        """FOSSILIZE with negative total_improvement should get zero attribution.

        This is the "ransomware edge case" - seed has high counterfactual
        contribution but the host actually declined. The agent shouldn't get
        credit for fossilizing such a seed.
        """
        from enum import IntEnum
        class MockAction(IntEnum):
            FOSSILIZE = 1

        # PROBATIONARY seed with high contribution but negative total improvement
        ransomware_seed = SeedInfo(
            stage=STAGE_PROBATIONARY,
            improvement_since_stage_start=-0.5,
            total_improvement=-0.02,  # Negative!
            epochs_in_stage=3,
            seed_age_epochs=15,
        )

        reward, components = compute_contribution_reward(
            action=MockAction.FOSSILIZE,
            seed_contribution=17.51,  # High counterfactual
            val_acc=60.0,
            seed_info=ransomware_seed,
            epoch=15,
            max_epochs=25,
            acc_at_germination=60.02,
            return_components=True,
        )

        # Attribution should be zero despite high seed_contribution
        assert components.bounded_attribution == 0.0, (
            f"FOSSILIZE with negative delta should get no attribution: {components.bounded_attribution}"
        )
        # Total reward should be negative (penalty only)
        assert reward < 0, f"Should be penalized for fossilizing negative-delta seed: {reward}"


class TestProbationaryIndecisionPenalty:
    """Test escalating WAIT penalty in PROBATIONARY stage."""

    def test_no_penalty_epoch_1_grace_period(self):
        """Epoch 1 in PROBATIONARY should have no WAIT penalty (grace period)."""
        from enum import IntEnum
        class MockAction(IntEnum):
            WAIT = 0

        probationary_seed = SeedInfo(
            stage=STAGE_PROBATIONARY,
            improvement_since_stage_start=1.0,
            total_improvement=2.0,
            epochs_in_stage=1,  # First epoch - grace period
            seed_age_epochs=10,
        )

        _, components = compute_contribution_reward(
            action=MockAction.WAIT,
            seed_contribution=5.0,
            val_acc=65.0,
            seed_info=probationary_seed,
            epoch=10,
            max_epochs=25,
            acc_at_germination=63.0,
            return_components=True,
        )

        assert components.probation_warning == 0.0, (
            f"Epoch 1 should have no penalty: {components.probation_warning}"
        )

    def test_penalty_starts_epoch_2(self):
        """Epoch 2 in PROBATIONARY should have WAIT penalty (-0.10)."""
        from enum import IntEnum
        class MockAction(IntEnum):
            WAIT = 0

        probationary_seed = SeedInfo(
            stage=STAGE_PROBATIONARY,
            improvement_since_stage_start=1.0,
            total_improvement=2.0,
            epochs_in_stage=2,  # Second epoch
            seed_age_epochs=11,
        )

        _, components = compute_contribution_reward(
            action=MockAction.WAIT,
            seed_contribution=5.0,
            val_acc=65.0,
            seed_info=probationary_seed,
            epoch=11,
            max_epochs=25,
            acc_at_germination=63.0,
            return_components=True,
        )

        # Epoch 2: -0.05 - (2-1)*0.05 = -0.10
        assert components.probation_warning == -0.10, (
            f"Epoch 2 should have -0.10 penalty: {components.probation_warning}"
        )

    def test_penalty_escalates_over_epochs(self):
        """WAIT penalty should escalate each epoch in PROBATIONARY."""
        from enum import IntEnum
        class MockAction(IntEnum):
            WAIT = 0

        def get_penalty(epochs_in_stage: int) -> float:
            seed = SeedInfo(
                stage=STAGE_PROBATIONARY,
                improvement_since_stage_start=1.0,
                total_improvement=2.0,
                epochs_in_stage=epochs_in_stage,
                seed_age_epochs=10 + epochs_in_stage,
            )
            _, components = compute_contribution_reward(
                action=MockAction.WAIT,
                seed_contribution=5.0,
                val_acc=65.0,
                seed_info=seed,
                epoch=10 + epochs_in_stage,
                max_epochs=25,
                acc_at_germination=63.0,
                return_components=True,
            )
            return components.probation_warning

        # Verify escalation: 0, -0.10, -0.15, -0.20, -0.25, -0.30 (capped)
        # Use pytest.approx for floating point comparison
        assert get_penalty(1) == 0.0  # Grace
        assert get_penalty(2) == pytest.approx(-0.10)
        assert get_penalty(3) == pytest.approx(-0.15)
        assert get_penalty(4) == pytest.approx(-0.20)
        assert get_penalty(5) == pytest.approx(-0.25)
        assert get_penalty(6) == pytest.approx(-0.30)  # Capped
        assert get_penalty(10) == pytest.approx(-0.30)  # Still capped

    def test_no_penalty_for_fossilize_action(self):
        """FOSSILIZE action should not receive WAIT penalty."""
        from enum import IntEnum
        class MockAction(IntEnum):
            FOSSILIZE = 1

        probationary_seed = SeedInfo(
            stage=STAGE_PROBATIONARY,
            improvement_since_stage_start=1.0,
            total_improvement=2.0,
            epochs_in_stage=5,  # Would have penalty if WAIT
            seed_age_epochs=15,
        )

        _, components = compute_contribution_reward(
            action=MockAction.FOSSILIZE,
            seed_contribution=5.0,
            val_acc=65.0,
            seed_info=probationary_seed,
            epoch=15,
            max_epochs=25,
            acc_at_germination=63.0,
            return_components=True,
        )

        assert components.probation_warning == 0.0, (
            f"FOSSILIZE should have no indecision penalty: {components.probation_warning}"
        )

    def test_no_penalty_without_counterfactual_data(self):
        """No penalty if counterfactual data not yet available."""
        from enum import IntEnum
        class MockAction(IntEnum):
            WAIT = 0

        # Seed with no improvement data (waiting for counterfactual)
        probationary_seed = SeedInfo(
            stage=STAGE_PROBATIONARY,
            improvement_since_stage_start=None,  # No data yet
            total_improvement=None,  # No data yet
            epochs_in_stage=3,  # Would have penalty if data available
            seed_age_epochs=13,
        )

        _, components = compute_contribution_reward(
            action=MockAction.WAIT,
            seed_contribution=None,  # No counterfactual
            val_acc=65.0,
            seed_info=probationary_seed,
            epoch=13,
            max_epochs=25,
            acc_at_germination=63.0,
            return_components=True,
        )

        assert components.probation_warning == 0.0, (
            f"No penalty without counterfactual data: {components.probation_warning}"
        )
