"""Tests for reward shaping functions."""

import pytest
from enum import IntEnum

from esper.leyline import MIN_CULL_AGE
from esper.leyline.factored_actions import LifecycleOp
from esper.simic.rewards import (
    SeedInfo,
    STAGE_BLENDING,
    STAGE_GERMINATED,
    STAGE_TRAINING,
    STAGE_HOLDING,
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
    - HOLDING=6, FOSSILIZED=7 (5 skipped)
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

    def test_holding_has_highest_potential(self):
        """Test HOLDING stage (6) has high potential before FOSSILIZED."""
        obs = {'has_active_seed': 1, 'seed_stage': 6, 'seed_epochs_in_stage': 0}
        potential = compute_seed_potential(obs)
        assert potential == 5.5  # From unified STAGE_POTENTIALS

    def test_fossilized_has_highest_potential(self):
        """Test FOSSILIZED stage (7) has slightly higher potential than HOLDING."""
        obs = {'has_active_seed': 1, 'seed_stage': 7, 'seed_epochs_in_stage': 0}
        potential = compute_seed_potential(obs)
        assert potential == 6.0  # Small +0.5 over HOLDING, not a farming target

    def test_progress_bonus_capped(self):
        """Test that progress bonus is capped at 2.0."""
        obs = {'has_active_seed': 1, 'seed_stage': 3, 'seed_epochs_in_stage': 100}
        potential = compute_seed_potential(obs)
        # Base 2.0 (TRAINING) + max 2.0 progress = 4.0
        assert potential == 4.0

    def test_stage_progression_increases_potential(self):
        """Test that potential generally increases through stages until FOSSILIZED."""
        potentials = []
        for stage in [2, 3, 4, 6]:  # GERMINATED through HOLDING (5 skipped)
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

    def test_cull_good_seed_inverts_attribution(self):
        """Culling a good seed should invert attribution to negative total reward."""
        seed_info = self._make_seed_info(STAGE_BLENDING, age=MIN_CULL_AGE, improvement=3.0)

        reward, components = compute_contribution_reward(
            action=LifecycleOp.CULL,
            seed_contribution=3.52,  # Good seed with +3.52% contribution
            val_acc=68.0,
            seed_info=seed_info,
            epoch=10,
            max_epochs=25,
            acc_at_germination=65.0,
            return_components=True,
        )

        # Attribution should be NEGATIVE (inverted) for culling good seed
        assert components.bounded_attribution < 0, (
            f"CULL of good seed should have negative attribution: {components.bounded_attribution}"
        )
        # Total reward should be negative
        assert reward < 0, (
            f"CULL of good seed should have negative total reward: {reward}"
        )

    def test_cull_bad_seed_inverts_attribution_to_positive(self):
        """Culling a bad seed should invert negative attribution to positive."""
        seed_info = self._make_seed_info(STAGE_BLENDING, age=MIN_CULL_AGE, improvement=-1.0)

        reward, components = compute_contribution_reward(
            action=LifecycleOp.CULL,
            seed_contribution=-2.0,  # Bad seed hurting accuracy
            val_acc=63.0,
            seed_info=seed_info,
            epoch=10,
            max_epochs=25,
            acc_at_germination=65.0,
            return_components=True,
        )

        # Attribution should be POSITIVE (inverted from negative)
        assert components.bounded_attribution > 0, (
            f"CULL of bad seed should have positive attribution: {components.bounded_attribution}"
        )
        # Total reward should be positive (good decision to remove harmful seed)
        assert reward > 0, (
            f"CULL of bad seed should have positive total reward: {reward}"
        )


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

    def test_short_holding_gets_discounted(self):
        """Seeds with short HOLDING get reduced fossilize bonus."""
        from esper.leyline import MIN_HOLDING_EPOCHS

        config = ContributionRewardConfig()

        # Seed with very short holding (1 epoch)
        short_holding = SeedInfo(
            stage=STAGE_HOLDING,
            improvement_since_stage_start=5.0,
            total_improvement=10.0,
            epochs_in_stage=1,  # Just entered holding
            seed_age_epochs=15,
        )

        # Seed with full holding
        full_holding = SeedInfo(
            stage=STAGE_HOLDING,
            improvement_since_stage_start=5.0,
            total_improvement=10.0,
            epochs_in_stage=MIN_HOLDING_EPOCHS,  # Full holding
            seed_age_epochs=20,
        )

        short_bonus = _contribution_fossilize_shaping(short_holding, 3.0, config)
        full_bonus = _contribution_fossilize_shaping(full_holding, 3.0, config)

        # Short holding should get less bonus
        assert short_bonus < full_bonus, (
            f"Short holding ({short_bonus}) should be less than full ({full_bonus})"
        )

        # Discount should be proportional
        expected_discount = 1 / MIN_HOLDING_EPOCHS
        assert short_bonus == pytest.approx(full_bonus * expected_discount, rel=0.01)

    def test_zero_holding_gets_zero_bonus(self):
        """Seeds with 0 epochs in HOLDING get no bonus."""
        config = ContributionRewardConfig()
        seed = SeedInfo(
            stage=STAGE_HOLDING,
            improvement_since_stage_start=5.0,
            total_improvement=10.0,
            epochs_in_stage=0,  # Just entered, no validation yet
            seed_age_epochs=15,
        )

        bonus = _contribution_fossilize_shaping(seed, 3.0, config)

        # Should get penalty, not bonus (legitimacy_discount = 0)
        assert bonus <= 0, f"Zero holding should not get positive bonus: {bonus}"


class TestContributionRewardComponents:
    """Tests for compute_contribution_reward return_components."""

    def test_return_components_returns_tuple(self):
        """Test that return_components=True returns (reward, components) tuple."""
        from esper.simic.rewards import RewardComponentsTelemetry
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
            total_params=25000,  # total = host + seed params
            host_params=20000,
            acc_at_germination=68.0,
            return_components=True,
        )

        # DRL Expert recommended fields
        assert components.val_acc == 72.0
        assert components.acc_at_germination == 68.0
        # growth_ratio = (total - host) / host = excess seed params as fraction
        assert components.growth_ratio == (25000 - 20000) / 20000  # 0.25
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
            stage=STAGE_HOLDING,
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
            stage=STAGE_HOLDING,
            improvement_since_stage_start=-0.5,
            total_improvement=-0.5,  # < -0.2 threshold
            epochs_in_stage=3,
            seed_age_epochs=15,
        )

        # Non-ransomware: same negative delta but low contribution
        non_ransomware_seed = SeedInfo(
            stage=STAGE_HOLDING,
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

        # Test specific values from sigmoid (steepness=3, more forgiving of variance)
        # DRL Expert review 2025-12-18: reduced from 10 to 3 to avoid penalizing
        # normal training variance (±0.1-0.3%), while still catching true regression
        #
        # At -0.5%, expect ~0.18 (still penalized but not zeroed)
        discount_05 = get_discount(-0.5)
        assert 0.10 < discount_05 < 0.25, f"At -0.5%: {discount_05}"

        # At -0.2%, expect ~0.35 (forgiving of noise)
        discount_02 = get_discount(-0.2)
        assert 0.25 < discount_02 < 0.45, f"At -0.2%: {discount_02}"

        # At -1.0%, expect ~0.05 (heavily penalized for clear regression)
        discount_10 = get_discount(-1.0)
        assert discount_10 < 0.10, f"At -1.0%: {discount_10}"

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

        # HOLDING seed with high contribution but negative total improvement
        ransomware_seed = SeedInfo(
            stage=STAGE_HOLDING,
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


class TestHoldingIndecisionPenalty:
    """Test escalating WAIT penalty in HOLDING stage."""

    def test_no_penalty_epoch_1_grace_period(self):
        """Epoch 1 in HOLDING should have no WAIT penalty (grace period)."""
        from enum import IntEnum
        class MockAction(IntEnum):
            WAIT = 0

        holding_seed = SeedInfo(
            stage=STAGE_HOLDING,
            improvement_since_stage_start=1.0,
            total_improvement=2.0,
            epochs_in_stage=1,  # First epoch - grace period
            seed_age_epochs=10,
        )

        _, components = compute_contribution_reward(
            action=MockAction.WAIT,
            seed_contribution=5.0,
            val_acc=65.0,
            seed_info=holding_seed,
            epoch=10,
            max_epochs=25,
            acc_at_germination=63.0,
            return_components=True,
        )

        assert components.holding_warning == 0.0, (
            f"Epoch 1 should have no penalty: {components.holding_warning}"
        )

    def test_penalty_starts_epoch_2(self):
        """Epoch 2 in HOLDING should have WAIT penalty (-1.0).

        DRL Expert review 2025-12-10: exponential penalty to overcome +7.5 attribution.
        """
        from enum import IntEnum
        class MockAction(IntEnum):
            WAIT = 0

        holding_seed = SeedInfo(
            stage=STAGE_HOLDING,
            improvement_since_stage_start=1.0,
            total_improvement=2.0,
            epochs_in_stage=2,  # Second epoch
            seed_age_epochs=11,
        )

        _, components = compute_contribution_reward(
            action=MockAction.WAIT,
            seed_contribution=5.0,
            val_acc=65.0,
            seed_info=holding_seed,
            epoch=11,
            max_epochs=25,
            acc_at_germination=63.0,
            return_components=True,
        )

        # Epoch 2: -1.0 * (3 ** 0) = -1.0
        assert components.holding_warning == -1.0, (
            f"Epoch 2 should have -1.0 penalty: {components.holding_warning}"
        )

    def test_penalty_escalates_over_epochs(self):
        """WAIT penalty should escalate exponentially each epoch in HOLDING.

        DRL Expert review 2025-12-10: exponential to overcome +7.5 attribution.
        Schedule: epoch 2 -> -1.0, epoch 3 -> -3.0, epoch 4 -> -9.0, capped at -10.0
        """
        from enum import IntEnum
        class MockAction(IntEnum):
            WAIT = 0

        def get_penalty(epochs_in_stage: int) -> float:
            seed = SeedInfo(
                stage=STAGE_HOLDING,
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
            return components.holding_warning

        # Verify exponential escalation: 0, -1.0, -3.0, -9.0, -10.0 (capped)
        # Formula: -1.0 * (3 ** (epochs_waiting - 1)), capped at -10.0
        assert get_penalty(1) == 0.0  # Grace period
        assert get_penalty(2) == pytest.approx(-1.0)   # 3^0 = 1
        assert get_penalty(3) == pytest.approx(-3.0)   # 3^1 = 3
        assert get_penalty(4) == pytest.approx(-9.0)   # 3^2 = 9
        assert get_penalty(5) == pytest.approx(-10.0)  # 3^3 = 27, capped at -10
        assert get_penalty(10) == pytest.approx(-10.0) # Still capped

    def test_no_penalty_for_fossilize_action(self):
        """FOSSILIZE action should not receive WAIT penalty."""
        from enum import IntEnum
        class MockAction(IntEnum):
            FOSSILIZE = 1

        holding_seed = SeedInfo(
            stage=STAGE_HOLDING,
            improvement_since_stage_start=1.0,
            total_improvement=2.0,
            epochs_in_stage=5,  # Would have penalty if WAIT
            seed_age_epochs=15,
        )

        _, components = compute_contribution_reward(
            action=MockAction.FOSSILIZE,
            seed_contribution=5.0,
            val_acc=65.0,
            seed_info=holding_seed,
            epoch=15,
            max_epochs=25,
            acc_at_germination=63.0,
            return_components=True,
        )

        assert components.holding_warning == 0.0, (
            f"FOSSILIZE should have no indecision penalty: {components.holding_warning}"
        )

    def test_no_penalty_without_counterfactual_data(self):
        """No penalty if counterfactual data not yet available."""
        from enum import IntEnum
        class MockAction(IntEnum):
            WAIT = 0

        # Seed with no improvement data (waiting for counterfactual)
        holding_seed = SeedInfo(
            stage=STAGE_HOLDING,
            improvement_since_stage_start=None,  # No data yet
            total_improvement=None,  # No data yet
            epochs_in_stage=3,  # Would have penalty if data available
            seed_age_epochs=13,
        )

        _, components = compute_contribution_reward(
            action=MockAction.WAIT,
            seed_contribution=None,  # No counterfactual
            val_acc=65.0,
            seed_info=holding_seed,
            epoch=13,
            max_epochs=25,
            acc_at_germination=63.0,
            return_components=True,
        )

        assert components.holding_warning == 0.0, (
            f"No penalty without counterfactual data: {components.holding_warning}"
        )


class TestSeedlessAttribution:
    """Test that seedless states get zero attribution."""

    def test_seedless_wait_gets_no_attribution(self):
        """WAIT with no seed should get zero attribution, even with positive acc_delta.

        Host-only learning shouldn't be credited as if a seed contributed.
        """
        from enum import IntEnum
        class MockAction(IntEnum):
            WAIT = 0

        # No seed_info means no seed exists
        _, components = compute_contribution_reward(
            action=MockAction.WAIT,
            seed_contribution=None,  # No counterfactual
            val_acc=45.0,
            seed_info=None,  # NO SEED
            epoch=3,
            max_epochs=25,
            acc_at_germination=None,
            acc_delta=10.0,  # Big accuracy improvement (host learning)
            return_components=True,
        )

        # Should get zero attribution despite positive acc_delta
        assert components.bounded_attribution == 0.0, (
            f"Seedless state should get no attribution: {components.bounded_attribution}"
        )

    def test_seedless_germinate_gets_no_attribution(self):
        """GERMINATE with no existing seed should get zero attribution."""
        from enum import IntEnum
        class MockAction(IntEnum):
            GERMINATE_NORM = 1

        _, components = compute_contribution_reward(
            action=MockAction.GERMINATE_NORM,
            seed_contribution=None,
            val_acc=50.0,
            seed_info=None,  # No seed yet (germinating creates one)
            epoch=5,
            max_epochs=25,
            acc_at_germination=None,
            acc_delta=5.0,  # Host learning
            return_components=True,
        )

        assert components.bounded_attribution == 0.0, (
            f"Seedless germinate should get no attribution: {components.bounded_attribution}"
        )

    def test_pre_blending_seed_still_gets_proxy_attribution(self):
        """Pre-blending seed (no counterfactual) should still get proxy attribution.

        This is different from seedless - a seed exists but hasn't reached
        BLENDING yet, so we use acc_delta as a proxy signal.
        """
        from enum import IntEnum
        class MockAction(IntEnum):
            WAIT = 0

        # Seed exists but in TRAINING (no counterfactual yet)
        training_seed = SeedInfo(
            stage=STAGE_TRAINING,
            improvement_since_stage_start=0.5,
            total_improvement=0.5,
            epochs_in_stage=2,
            seed_age_epochs=3,
        )

        _, components = compute_contribution_reward(
            action=MockAction.WAIT,
            seed_contribution=None,  # No counterfactual yet
            val_acc=45.0,
            seed_info=training_seed,  # SEED EXISTS
            epoch=3,
            max_epochs=25,
            acc_at_germination=40.0,
            acc_delta=2.0,  # Positive delta
            return_components=True,
        )

        # Should get proxy attribution (proxy_contribution_weight * acc_delta = 0.3 * 2.0 = 0.6)
        assert components.bounded_attribution == pytest.approx(0.6), (
            f"Pre-blending seed should get proxy attribution: {components.bounded_attribution}"
        )


class TestRatioPenalty:
    """Test contribution/improvement ratio penalty for ransomware detection.

    DRL Expert review 2025-12-10: High causal contribution with low/negative
    improvement indicates structural entanglement - the "ransomware signature".
    """

    def test_no_penalty_for_low_contribution(self):
        """Contribution < 1.0 should not trigger ratio penalty."""
        from enum import IntEnum
        class MockAction(IntEnum):
            WAIT = 0

        seed = SeedInfo(
            stage=STAGE_BLENDING,
            improvement_since_stage_start=-0.5,
            total_improvement=-0.5,
            epochs_in_stage=3,
            seed_age_epochs=8,
        )

        _, components = compute_contribution_reward(
            action=MockAction.WAIT,
            seed_contribution=0.5,  # Low contribution (below 1.0 threshold)
            val_acc=65.0,
            seed_info=seed,
            epoch=8,
            max_epochs=25,
            acc_at_germination=65.5,
            return_components=True,
        )

        assert components.ratio_penalty == 0.0, (
            f"Low contribution should have no ratio penalty: {components.ratio_penalty}"
        )

    def test_penalty_for_high_contribution_negative_improvement(self):
        """High contribution with negative improvement - ratio penalty SKIPPED to avoid stacking.

        When total_improvement is negative, the attribution_discount sigmoid zeros the
        attribution. The ratio_penalty is then skipped (attribution_discount < 0.5) to
        avoid penalty stacking where the same ransomware seed is punished multiple times.
        """
        from enum import IntEnum
        class MockAction(IntEnum):
            WAIT = 0

        # Classic ransomware signature: high contribution, negative improvement
        seed = SeedInfo(
            stage=STAGE_BLENDING,
            improvement_since_stage_start=-2.0,
            total_improvement=-2.0,
            epochs_in_stage=5,
            seed_age_epochs=10,
        )

        _, components = compute_contribution_reward(
            action=MockAction.WAIT,
            seed_contribution=37.5,  # Very high contribution (like conv_heavy)
            val_acc=62.0,
            seed_info=seed,
            epoch=10,
            max_epochs=25,
            acc_at_germination=64.0,
            return_components=True,
        )

        # With negative total_improvement, attribution_discount ≈ 0 (sigmoid zeros it).
        # ratio_penalty is skipped when attribution_discount < 0.5 to avoid stacking.
        assert components.ratio_penalty == 0.0, (
            f"Ratio penalty should be skipped when attribution_discount < 0.5: {components.ratio_penalty}"
        )
        # But the ransomware IS penalized via attribution_discount
        assert components.attribution_discount < 0.5, (
            f"Attribution discount should handle ransomware: {components.attribution_discount}"
        )

    def test_penalty_for_high_contribution_low_improvement(self):
        """High contribution with very low improvement should trigger penalty."""
        from enum import IntEnum
        class MockAction(IntEnum):
            WAIT = 0

        # Suspicious: high contribution but marginal improvement (<=0.1%)
        seed = SeedInfo(
            stage=STAGE_BLENDING,
            improvement_since_stage_start=0.05,
            total_improvement=0.05,  # Very low (below 0.1 threshold)
            epochs_in_stage=4,
            seed_age_epochs=9,
        )

        _, components = compute_contribution_reward(
            action=MockAction.WAIT,
            seed_contribution=15.0,  # High contribution
            val_acc=64.0,
            seed_info=seed,
            epoch=9,
            max_epochs=25,
            acc_at_germination=63.95,
            return_components=True,
        )

        # With 15% contribution and total_imp=0.05 (<=0.1):
        # ratio_penalty = -0.3 * min(1.0, 15/10.0) = -0.3 * 1.0 = -0.3
        assert components.ratio_penalty == pytest.approx(-0.3), (
            f"High contribution + low improvement should have penalty: {components.ratio_penalty}"
        )

    def test_penalty_scales_with_contribution_magnitude(self):
        """Ratio penalty scales with contribution when NOT masked by attribution_discount.

        The ratio penalty only fires when attribution_discount >= 0.5 (positive trajectory).
        This test uses low positive improvement (0.05) which keeps attribution_discount=1.0
        but still triggers the ratio penalty for suspiciously high contribution.
        """
        from enum import IntEnum
        class MockAction(IntEnum):
            WAIT = 0

        def get_penalty(contribution: float) -> float:
            # Use low positive improvement (<=0.1) - triggers ratio penalty zone
            # but keeps attribution_discount = 1.0 (no sigmoid discount for positive)
            seed = SeedInfo(
                stage=STAGE_BLENDING,
                improvement_since_stage_start=0.05,
                total_improvement=0.05,  # Low positive, not negative
                epochs_in_stage=3,
                seed_age_epochs=8,
            )
            _, components = compute_contribution_reward(
                action=MockAction.WAIT,
                seed_contribution=contribution,
                val_acc=64.0,
                seed_info=seed,
                epoch=8,
                max_epochs=25,
                acc_at_germination=63.95,  # Small positive progress
                return_components=True,
            )
            return components.ratio_penalty

        # At 5% contribution: -0.3 * (5/10) = -0.15
        penalty_5 = get_penalty(5.0)
        assert penalty_5 == pytest.approx(-0.15), f"5% contribution: {penalty_5}"

        # At 10%+ contribution: capped at -0.3
        penalty_10 = get_penalty(10.0)
        assert penalty_10 == pytest.approx(-0.3), f"10% contribution: {penalty_10}"

        # At 20% contribution: still capped at -0.3
        penalty_20 = get_penalty(20.0)
        assert penalty_20 == pytest.approx(-0.3), f"20% contribution: {penalty_20}"

    def test_no_penalty_for_healthy_ratio(self):
        """Healthy seeds (contribution <= 5x improvement) should have no penalty."""
        from enum import IntEnum
        class MockAction(IntEnum):
            WAIT = 0

        # Healthy seed: contribution is 2x improvement (well below 5x threshold)
        seed = SeedInfo(
            stage=STAGE_BLENDING,
            improvement_since_stage_start=3.0,
            total_improvement=3.0,
            epochs_in_stage=5,
            seed_age_epochs=10,
        )

        _, components = compute_contribution_reward(
            action=MockAction.WAIT,
            seed_contribution=6.0,  # 2x improvement - healthy ratio
            val_acc=68.0,
            seed_info=seed,
            epoch=10,
            max_epochs=25,
            acc_at_germination=65.0,
            return_components=True,
        )

        assert components.ratio_penalty == 0.0, (
            f"Healthy ratio should have no penalty: {components.ratio_penalty}"
        )

    def test_penalty_for_suspicious_high_ratio(self):
        """Ratio > 5x should trigger escalating penalty."""
        from enum import IntEnum
        class MockAction(IntEnum):
            WAIT = 0

        # Suspicious: contribution is 10x improvement
        seed = SeedInfo(
            stage=STAGE_BLENDING,
            improvement_since_stage_start=1.0,
            total_improvement=1.0,
            epochs_in_stage=4,
            seed_age_epochs=9,
        )

        _, components = compute_contribution_reward(
            action=MockAction.WAIT,
            seed_contribution=10.0,  # 10x improvement - suspicious
            val_acc=66.0,
            seed_info=seed,
            epoch=9,
            max_epochs=25,
            acc_at_germination=65.0,
            return_components=True,
        )

        # ratio = 10/1 = 10, penalty = -min(0.3, 0.1*(10-5)/5) = -min(0.3, 0.1) = -0.1
        assert components.ratio_penalty == pytest.approx(-0.1), (
            f"10x ratio should have -0.1 penalty: {components.ratio_penalty}"
        )


class TestPenaltyAntiStacking:
    """Test that penalties don't stack on ransomware seeds.

    Fix for training collapse: when attribution_discount zeros attribution,
    ratio_penalty and holding_warning should NOT add additional penalties.
    This prevents an unlearnable reward landscape.
    """

    def test_ransomware_seed_no_penalty_stacking(self):
        """Ransomware seed: attribution_discount should be sole penalty mechanism.

        When a seed has negative total_improvement (ransomware pattern), the
        attribution_discount zeros the attribution. Additional penalties via
        ratio_penalty or holding_warning create an unlearnable landscape.
        """
        from enum import IntEnum
        class MockAction(IntEnum):
            WAIT = 0

        # Classic ransomware: high contribution but negative trajectory
        ransomware_seed = SeedInfo(
            stage=STAGE_HOLDING,
            improvement_since_stage_start=-1.0,
            total_improvement=-2.0,  # Negative = ransomware
            epochs_in_stage=3,  # Would normally trigger PROB penalty
            seed_age_epochs=12,
        )

        _, components = compute_contribution_reward(
            action=MockAction.WAIT,
            seed_contribution=25.0,  # High contribution (would trigger ratio penalty)
            val_acc=62.0,
            seed_info=ransomware_seed,
            epoch=12,
            max_epochs=25,
            acc_at_germination=64.0,  # Negative progress
            return_components=True,
        )

        # Attribution discount should be near-zero (handling ransomware)
        assert components.attribution_discount < 0.01, (
            f"Attribution discount should handle ransomware: {components.attribution_discount}"
        )

        # Ratio penalty should be SKIPPED (attribution_discount < 0.5)
        assert components.ratio_penalty == 0.0, (
            f"Ratio penalty should be skipped to avoid stacking: {components.ratio_penalty}"
        )

        # HOLD penalty should be SKIPPED (bounded_attribution <= 0)
        assert components.holding_warning == 0.0, (
            f"HOLD penalty should be skipped to avoid stacking: {components.holding_warning}"
        )

        # bounded_attribution should be near-zero (not negative)
        assert abs(components.bounded_attribution) < 0.1, (
            f"Attribution should be zeroed, not heavily negative: {components.bounded_attribution}"
        )

    def test_legitimate_seed_still_gets_holding_penalty(self):
        """Legitimate seed farming should still receive HOLD penalty."""
        from enum import IntEnum
        class MockAction(IntEnum):
            WAIT = 0

        # Good seed being farmed in HOLDING
        good_seed = SeedInfo(
            stage=STAGE_HOLDING,
            improvement_since_stage_start=2.0,
            total_improvement=5.0,  # Positive trajectory
            epochs_in_stage=3,  # Should trigger PROB penalty
            seed_age_epochs=12,
        )

        _, components = compute_contribution_reward(
            action=MockAction.WAIT,
            seed_contribution=8.0,  # Good contribution
            val_acc=68.0,
            seed_info=good_seed,
            epoch=12,
            max_epochs=25,
            acc_at_germination=63.0,
            return_components=True,
        )

        # Attribution discount should be 1.0 (no discount for good seeds)
        assert components.attribution_discount == 1.0, (
            f"Good seed should have no attribution discount: {components.attribution_discount}"
        )

        # bounded_attribution should be positive
        assert components.bounded_attribution > 0, (
            f"Good seed should have positive attribution: {components.bounded_attribution}"
        )

        # HOLD penalty SHOULD fire (bounded_attribution > 0, epoch 3)
        # Epoch 3: -1.0 * (3 ** 1) = -3.0
        assert components.holding_warning == pytest.approx(-3.0), (
            f"Good seed farming should receive HOLD penalty: {components.holding_warning}"
        )


class TestRewardHackingDetection:
    """Test reward hacking detection telemetry.

    DRL Expert review 2025-12-16: Threshold raised to 5x to align with penalty
    threshold in compute_contribution_reward(). Ransomware detection added for
    the dangerous pattern of high contribution + negative total improvement.
    """

    def test_reward_hacking_suspected_emitted_on_anomalous_ratio(self):
        """Test that 6x ratio triggers with default 5x threshold."""
        from esper.simic.rewards import _check_reward_hacking
        from esper.leyline import TelemetryEventType
        from unittest.mock import Mock

        hub = Mock()

        # Seed claims 600% of total improvement (above 5x threshold)
        emitted = _check_reward_hacking(
            hub=hub,
            seed_contribution=6.0,
            total_improvement=1.0,
            # Using default threshold of 5.0
            slot_id="r0c0",
            seed_id="seed_001",
        )

        assert emitted is True
        event = hub.emit.call_args[0][0]
        assert event.event_type == TelemetryEventType.REWARD_HACKING_SUSPECTED
        assert event.data["ratio"] == 6.0
        assert event.data["slot_id"] == "r0c0"
        assert event.data["threshold"] == 5.0

    def test_no_hacking_event_for_normal_ratios(self):
        """Test that 4x ratio doesn't trigger with default 5x threshold."""
        from esper.simic.rewards import _check_reward_hacking
        from unittest.mock import Mock

        hub = Mock()

        # 4x ratio is below 5x threshold
        emitted = _check_reward_hacking(
            hub=hub,
            seed_contribution=4.0,
            total_improvement=1.0,
            # Using default threshold of 5.0
            slot_id="r0c0",
            seed_id="seed_001",
        )

        assert emitted is False
        assert not hub.emit.called

    def test_ransomware_signature_emitted(self):
        """Test ransomware detection: high contribution + negative total."""
        from esper.simic.rewards import _check_ransomware_signature
        from esper.leyline import TelemetryEventType
        from unittest.mock import Mock

        hub = Mock()

        # Seed claims high contribution while system degrades (ransomware pattern)
        emitted = _check_ransomware_signature(
            hub=hub,
            seed_contribution=2.0,  # High contribution
            total_improvement=-0.5,  # System getting worse
            slot_id="r0c0",
            seed_id="seed_001",
        )

        assert emitted is True
        event = hub.emit.call_args[0][0]
        assert event.event_type == TelemetryEventType.REWARD_HACKING_SUSPECTED
        assert event.severity == "critical"
        assert event.data["pattern"] == "ransomware_signature"
        assert event.data["seed_contribution"] == 2.0
        assert event.data["total_improvement"] == -0.5

    def test_no_ransomware_when_system_improving(self):
        """Test no ransomware alert when total improvement is positive."""
        from esper.simic.rewards import _check_ransomware_signature
        from unittest.mock import Mock

        hub = Mock()

        # High contribution but system is improving - not ransomware
        emitted = _check_ransomware_signature(
            hub=hub,
            seed_contribution=2.0,
            total_improvement=0.5,  # System improving
            slot_id="r0c0",
            seed_id="seed_001",
        )

        assert emitted is False
        assert not hub.emit.called

    def test_no_ransomware_when_contribution_low(self):
        """Test no ransomware alert when seed contribution is low."""
        from esper.simic.rewards import _check_ransomware_signature
        from unittest.mock import Mock

        hub = Mock()

        # Low contribution even though system degrading - not ransomware
        emitted = _check_ransomware_signature(
            hub=hub,
            seed_contribution=0.5,  # Low contribution
            total_improvement=-0.5,  # System getting worse
            slot_id="r0c0",
            seed_id="seed_001",
        )

        assert emitted is False
        assert not hub.emit.called


class TestFossilizeTerminalBonus:
    """Test terminal bonus for fossilized seeds.

    DRL Expert review 2025-12-10: Terminal bonus makes FOSSILIZE NPV-positive
    vs WAIT-farming in HOLDING. Addresses H4 (terminating action problem).
    """

    def test_terminal_bonus_scales_with_contributing_fossilized_count(self):
        """Terminal bonus should scale with number of CONTRIBUTING fossilized seeds.

        DRL Expert review 2025-12-11: Only contributing fossilized seeds (those with
        total_improvement >= MIN_FOSSILIZE_CONTRIBUTION) get terminal bonus. This
        prevents bad fossilizations from being NPV-positive.
        """
        from enum import IntEnum
        class MockAction(IntEnum):
            WAIT = 0

        def get_terminal_bonus(num_contributing: int, num_total: int = 0) -> tuple[float, float]:
            _, components = compute_contribution_reward(
                action=MockAction.WAIT,
                seed_contribution=None,
                val_acc=70.0,
                seed_info=None,
                epoch=25,  # Terminal epoch
                max_epochs=25,
                acc_at_germination=None,
                return_components=True,
                num_fossilized_seeds=num_total if num_total else num_contributing,
                num_contributing_fossilized=num_contributing,
            )
            return components.terminal_bonus, components.fossilize_terminal_bonus

        # Default fossilize_terminal_scale = 3.0
        # terminal_bonus = val_acc * 0.05 + num_contributing * 3.0
        # Base: 70 * 0.05 = 3.5
        total_0, fossil_0 = get_terminal_bonus(0)
        assert fossil_0 == 0.0
        assert total_0 == pytest.approx(3.5)  # Base only

        total_1, fossil_1 = get_terminal_bonus(1)
        assert fossil_1 == pytest.approx(3.0)  # 1 * 3.0
        assert total_1 == pytest.approx(6.5)  # 3.5 + 3.0

        total_5, fossil_5 = get_terminal_bonus(5)
        assert fossil_5 == pytest.approx(15.0)  # 5 * 3.0
        assert total_5 == pytest.approx(18.5)  # 3.5 + 15.0

        # Test asymmetric case: 3 total fossilized but only 1 contributing
        # Only the contributing one should get terminal bonus
        total_asym, fossil_asym = get_terminal_bonus(num_contributing=1, num_total=3)
        assert fossil_asym == pytest.approx(3.0)  # Only 1 contributing * 3.0
        assert total_asym == pytest.approx(6.5)  # 3.5 + 3.0

    def test_no_terminal_bonus_before_max_epoch(self):
        """Terminal bonus should only apply at max_epochs."""
        from enum import IntEnum
        class MockAction(IntEnum):
            WAIT = 0

        _, components = compute_contribution_reward(
            action=MockAction.WAIT,
            seed_contribution=None,
            val_acc=70.0,
            seed_info=None,
            epoch=20,  # Not terminal
            max_epochs=25,
            acc_at_germination=None,
            return_components=True,
            num_fossilized_seeds=5,
        )

        assert components.terminal_bonus == 0.0
        assert components.fossilize_terminal_bonus == 0.0

    def test_telemetry_tracks_fossilized_count(self):
        """Telemetry should track num_fossilized_seeds for debugging."""
        from enum import IntEnum
        class MockAction(IntEnum):
            WAIT = 0

        _, components = compute_contribution_reward(
            action=MockAction.WAIT,
            seed_contribution=None,
            val_acc=70.0,
            seed_info=None,
            epoch=25,
            max_epochs=25,
            acc_at_germination=None,
            return_components=True,
            num_fossilized_seeds=3,
        )

        assert components.num_fossilized_seeds == 3

    def test_terminal_bonus_config_override(self):
        """Custom config should allow adjusting fossilize_terminal_scale."""
        from enum import IntEnum
        class MockAction(IntEnum):
            WAIT = 0

        custom_config = ContributionRewardConfig(fossilize_terminal_scale=5.0)

        _, components = compute_contribution_reward(
            action=MockAction.WAIT,
            seed_contribution=None,
            val_acc=70.0,
            seed_info=None,
            epoch=25,
            max_epochs=25,
            config=custom_config,
            acc_at_germination=None,
            return_components=True,
            num_fossilized_seeds=2,
            num_contributing_fossilized=2,  # Both contributing
        )

        # 2 * 5.0 = 10.0 fossilize bonus (only contributing seeds count)
        assert components.fossilize_terminal_bonus == pytest.approx(10.0)
        # Total: 70 * 0.05 + 10.0 = 13.5
        assert components.terminal_bonus == pytest.approx(13.5)
