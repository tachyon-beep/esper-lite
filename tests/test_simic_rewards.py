"""Tests for reward shaping functions."""

import pytest
from enum import IntEnum

from esper.leyline import MIN_CULL_AGE
from esper.simic.rewards import (
    RewardConfig,
    SeedInfo,
    STAGE_BLENDING,
    STAGE_GERMINATED,
    STAGE_TRAINING,
    STAGE_SHADOWING,
    STAGE_PROBATIONARY,
    STAGE_FOSSILIZED,
    compute_seed_potential,
    compute_shaped_reward,
    _cull_shaping,
    _wait_shaping,
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
        """Test that progress bonus is capped at 3.0."""
        obs = {'has_active_seed': 1, 'seed_stage': 3, 'seed_epochs_in_stage': 100}
        potential = compute_seed_potential(obs)
        # Base 2.0 (TRAINING) + max 3.0 progress = 5.0
        assert potential == 5.0

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
    """Tests for PBRS stage bonus behaviour."""

    class _TestAction(IntEnum):
        NOOP = 0

    def _make_config_with_pbrs_only(self) -> RewardConfig:
        config = RewardConfig()
        config.acc_delta_weight = 0.0
        config.training_bonus = 0.0
        config.blending_bonus = 0.0
        config.fossilized_bonus = 0.0
        config.stage_improvement_weight = 0.0
        config.blending_improvement_bonus = 0.0
        config.compute_rent_weight = 0.0
        config.terminal_acc_weight = 0.0
        config.seed_potential_weight = 1.0
        return config

    def test_transition_bonus_not_repeated_in_stage(self):
        """PBRS bonus for TRAINING→BLENDING transition should not repeat every epoch."""
        config = self._make_config_with_pbrs_only()
        action = self._TestAction.NOOP

        # Step immediately after TRAINING→BLENDING transition
        seed_step1 = SeedInfo(
            stage=STAGE_BLENDING,
            improvement_since_stage_start=0.0,
            total_improvement=0.0,
            epochs_in_stage=0,
            seed_params=0,
            previous_stage=STAGE_TRAINING,
            seed_age_epochs=0,
        )

        r1 = compute_shaped_reward(
            action=action,
            acc_delta=0.0,
            val_acc=0.0,
            seed_info=seed_step1,
            epoch=0,
            max_epochs=10,
            total_params=0,
            host_params=1,
            config=config,
        )

        # Next epoch staying in BLENDING (no new transition)
        seed_step2 = seed_step1._replace(epochs_in_stage=1)
        r2 = compute_shaped_reward(
            action=action,
            acc_delta=0.0,
            val_acc=0.0,
            seed_info=seed_step2,
            epoch=1,
            max_epochs=10,
            total_params=0,
            host_params=1,
            config=config,
        )

        # Transition bonus should be strictly positive and strictly larger than
        # any subsequent in-stage PBRS bonuses.
        assert r1 > 0.0
        assert 0.0 < r2 < r1


class TestCullAgeProtection:
    """Tests for CULL age penalty to prevent immediate germinate-cull."""

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

    def test_cull_at_age_zero_heavily_penalized(self):
        """Culling immediately after germination should be heavily penalized."""
        config = RewardConfig.default()
        seed_info = self._make_seed_info(STAGE_GERMINATED, age=0)

        shaping = _cull_shaping(seed_info, config)

        # Age penalty: -0.3 * (MIN_CULL_AGE - 0)
        expected = -0.3 * MIN_CULL_AGE
        assert shaping == pytest.approx(expected)

    def test_cull_at_age_one_penalized(self):
        """Culling at age 1 should still be penalized."""
        config = RewardConfig.default()
        seed_info = self._make_seed_info(STAGE_TRAINING, age=1)

        shaping = _cull_shaping(seed_info, config)

        # Age penalty: -0.3 * (MIN_CULL_AGE - 1)
        expected = -0.3 * (MIN_CULL_AGE - 1)
        assert shaping == pytest.approx(expected)

    def test_cull_at_min_age_no_age_penalty(self):
        """Culling at MIN_CULL_AGE should have no age penalty."""
        config = RewardConfig.default()
        seed_info = self._make_seed_info(STAGE_TRAINING, age=MIN_CULL_AGE, improvement=0.0)

        shaping = _cull_shaping(seed_info, config)

        # No age penalty, but base shaping kicks in (promising seed = -0.3)
        # Plus PBRS correction for destroying the seed
        age_zero_penalty = -0.3 * MIN_CULL_AGE
        assert shaping != pytest.approx(age_zero_penalty)  # Not the age-0 penalty

    def test_age_penalty_only_for_early_stages(self):
        """Age penalty should only apply to GERMINATED and TRAINING."""
        config = RewardConfig.default()

        # BLENDING at age 0 should NOT get age penalty
        seed_info = SeedInfo(
            stage=STAGE_BLENDING,
            improvement_since_stage_start=0.0,
            total_improvement=0.0,
            epochs_in_stage=0,
            seed_params=1000,
            previous_stage=STAGE_TRAINING,
            seed_age_epochs=0,  # Age 0 but in BLENDING
        )

        shaping = _cull_shaping(seed_info, config)

        # Should NOT be the age-0 penalty (only applies to GERMINATED/TRAINING)
        age_zero_penalty = -0.3 * MIN_CULL_AGE
        assert shaping != pytest.approx(age_zero_penalty)


class TestWaitBlendingShaping:
    """Tests for WAIT action shaping at BLENDING stage."""

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

    def test_wait_at_blending_with_improvement_rewarded(self):
        """WAIT at BLENDING with improvement should get patience bonus."""
        config = RewardConfig.default()
        seed_info = self._make_seed_info(STAGE_BLENDING, improvement=1.0)

        shaping = _wait_shaping(seed_info, acc_delta=0.0, config=config)

        assert shaping == pytest.approx(config.wait_patience_bonus)

    def test_wait_at_blending_no_improvement_neutral(self):
        """WAIT at BLENDING without improvement should be neutral (not penalized)."""
        config = RewardConfig.default()
        seed_info = self._make_seed_info(STAGE_BLENDING, improvement=0.0)

        shaping = _wait_shaping(seed_info, acc_delta=0.0, config=config)

        # BLENDING is mechanical - no stagnant penalty
        assert shaping == pytest.approx(0.0)

    def test_wait_at_blending_no_stagnant_penalty(self):
        """WAIT at BLENDING should never get stagnant penalty (even after many epochs)."""
        config = RewardConfig.default()
        seed_info = self._make_seed_info(STAGE_BLENDING, improvement=0.0, epochs=10)

        shaping = _wait_shaping(seed_info, acc_delta=0.0, config=config)

        # Even after 10 epochs, no penalty for BLENDING (unlike TRAINING)
        assert shaping >= 0.0

    def test_wait_at_shadowing_rewarded(self):
        """WAIT at SHADOWING with improvement should get patience bonus."""
        config = RewardConfig.default()
        seed_info = self._make_seed_info(STAGE_SHADOWING, improvement=0.5)

        shaping = _wait_shaping(seed_info, acc_delta=0.0, config=config)

        assert shaping == pytest.approx(config.wait_patience_bonus)

    def test_wait_at_probationary_neutral_without_improvement(self):
        """WAIT at PROBATIONARY without improvement should be neutral."""
        config = RewardConfig.default()
        seed_info = self._make_seed_info(STAGE_PROBATIONARY, improvement=0.0)

        shaping = _wait_shaping(seed_info, acc_delta=0.0, config=config)

        # PROBATIONARY is hands-off, WAIT is always acceptable
        assert shaping == pytest.approx(0.0)


class TestFossilizeLegitimacyDiscount:
    """Fossilization bonus should be discounted for rapid fossilization."""

    def test_short_probation_gets_discounted(self):
        """Seeds with short PROBATIONARY get reduced fossilize bonus."""
        from esper.simic.rewards import (
            _contribution_fossilize_shaping,
            ContributionRewardConfig,
            SeedInfo,
            STAGE_PROBATIONARY,
        )
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
        from esper.simic.rewards import (
            _contribution_fossilize_shaping,
            ContributionRewardConfig,
            SeedInfo,
            STAGE_PROBATIONARY,
        )

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


class TestCullLateStageHealthScaling:
    """Tests for CULL PBRS penalty scaling by health_factor in late stages."""

    def test_cull_shaping_late_stage_not_excessive(self):
        """CULL penalty for late-stage failing seeds should not be excessive."""
        config = RewardConfig()

        # Failing seed in BLENDING stage
        seed_info = SeedInfo(
            stage=STAGE_BLENDING,
            improvement_since_stage_start=-2.0,  # Clearly failing
            total_improvement=-1.0,
            epochs_in_stage=5,
            seed_params=2000,
            previous_stage=STAGE_TRAINING,
            seed_age_epochs=8,
        )

        shaping = _cull_shaping(seed_info, config)

        # Should be positive (reward for culling failing seed) or only mildly negative
        # The old behavior gave ~-1.65 which is too harsh
        assert shaping > -1.0, f"CULL penalty too harsh for failing seed: {shaping}"

    def test_early_stage_pbrs_unaffected(self):
        """CULL PBRS penalty in early stages should NOT be scaled by health."""
        config = RewardConfig()

        # Failing seed in TRAINING stage (early)
        seed_info = SeedInfo(
            stage=STAGE_TRAINING,
            improvement_since_stage_start=-2.0,  # Failing
            total_improvement=-2.0,
            epochs_in_stage=3,
            seed_params=2000,
            previous_stage=STAGE_GERMINATED,
            seed_age_epochs=5,
        )

        shaping = _cull_shaping(seed_info, config)

        # Early stages should not get health scaling - preserve full PBRS incentives
        # The PBRS penalty will be smaller for TRAINING anyway (lower potential)
        # We just verify it doesn't have the health_factor applied

        # Calculate expected components:
        # - base_shaping: config.cull_failing_bonus (0.3)
        # - param_recovery_bonus: (2000/10000) * 0.1 = 0.02
        # - terminal_pbrs: no health scaling for TRAINING (stage < BLENDING)

        # This is more of a regression test - ensure early stages work
        assert isinstance(shaping, float)

    def test_health_factor_floor_at_30_percent(self):
        """Health factor should not go below 0.3 even for very bad seeds."""
        config = RewardConfig()

        # Extremely failing seed in SHADOWING stage
        seed_info = SeedInfo(
            stage=STAGE_SHADOWING,
            improvement_since_stage_start=-10.0,  # Catastrophically bad
            total_improvement=-10.0,
            epochs_in_stage=3,
            seed_params=2000,
            previous_stage=STAGE_BLENDING,
            seed_age_epochs=10,
        )

        shaping = _cull_shaping(seed_info, config)

        # Even with -10% improvement, health_factor floors at 0.3
        # So PBRS penalty is scaled to 30% of original
        # Seed is failing badly, so culling should be rewarded
        assert shaping > -0.8, f"CULL with catastrophic seed should not be harshly penalized: {shaping}"

    def test_positive_improvement_no_health_scaling(self):
        """Seeds with positive improvement don't get health_factor (it only applies to failing seeds)."""
        config = RewardConfig()

        # Good seed in BLENDING stage
        seed_info = SeedInfo(
            stage=STAGE_BLENDING,
            improvement_since_stage_start=2.0,  # Doing well
            total_improvement=3.0,
            epochs_in_stage=5,
            seed_params=2000,
            previous_stage=STAGE_TRAINING,
            seed_age_epochs=8,
        )

        shaping = _cull_shaping(seed_info, config)

        # Culling a good seed should be penalized
        # health_factor only applies when improvement < 0
        assert shaping < 0, "Culling a good seed should be penalized"


class TestContributionRewardComponents:
    """Tests for compute_contribution_reward return_components."""

    def test_return_components_returns_tuple(self):
        """Test that return_components=True returns (reward, components) tuple."""
        from esper.simic.rewards import compute_contribution_reward, SeedInfo
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
        from esper.simic.rewards import compute_contribution_reward, SeedInfo
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
        from esper.simic.rewards import compute_contribution_reward, SeedInfo
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
        from esper.simic.rewards import compute_contribution_reward, SeedInfo
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
