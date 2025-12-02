"""Unit tests for FOSSILIZE reward shaping."""

import pytest

from esper.leyline import SeedStage
from esper.simic.rewards import RewardConfig, SeedInfo, _advance_shaping


def make_seed_info(stage: SeedStage, improvement: float) -> SeedInfo:
    """Helper to construct minimal SeedInfo for shaping tests."""
    return SeedInfo(
        stage=stage.value,
        improvement_since_stage_start=improvement,
        epochs_in_stage=1,
    )


def test_advance_shaping_penalizes_training_noop():
    """FOSSILIZE in TRAINING is a no-op and should be penalized."""
    config = RewardConfig.default()
    seed_info = make_seed_info(SeedStage.TRAINING, improvement=1.0)

    shaping = _advance_shaping(seed_info, config)

    assert shaping == pytest.approx(config.advance_premature_penalty)


def test_advance_shaping_penalizes_blending_noop():
    """FOSSILIZE in BLENDING is a no-op and should be penalized."""
    config = RewardConfig.default()
    seed_info = make_seed_info(SeedStage.BLENDING, improvement=0.0)

    shaping = _advance_shaping(seed_info, config)

    assert shaping == pytest.approx(config.advance_premature_penalty)


def test_advance_shaping_penalizes_shadowing_noop():
    """FOSSILIZE in SHADOWING should not be rewarded."""
    config = RewardConfig.default()
    seed_info = make_seed_info(SeedStage.SHADOWING, improvement=1.0)

    shaping = _advance_shaping(seed_info, config)

    assert shaping == pytest.approx(config.advance_premature_penalty)


def test_advance_shaping_rewards_probationary_with_improvement():
    """FOSSILIZE in PROBATIONARY with improvement gets a large bonus.

    The bonus is: base (0.5) + 1.5 + 0.1 * improvement
    For improvement=0.5: 0.5 + 1.5 + 0.05 = 2.05
    """
    config = RewardConfig.default()
    seed_info = make_seed_info(SeedStage.PROBATIONARY, improvement=0.5)

    shaping = _advance_shaping(seed_info, config)

    # New formula: advance_good_bonus + 1.5 + 0.1 * improvement
    expected = config.advance_good_bonus + 1.5 + 0.1 * 0.5
    assert shaping == pytest.approx(expected)


def test_advance_shaping_penalizes_probationary_without_improvement():
    """FOSSILIZE in PROBATIONARY without improvement is penalized."""
    config = RewardConfig.default()
    seed_info = make_seed_info(SeedStage.PROBATIONARY, improvement=0.0)

    shaping = _advance_shaping(seed_info, config)

    assert shaping == pytest.approx(config.advance_premature_penalty)


def test_advance_shaping_penalizes_missing_seed():
    """FOSSILIZE with no active seed uses the no-seed penalty."""
    config = RewardConfig.default()

    shaping = _advance_shaping(None, config)

    assert shaping == pytest.approx(config.advance_no_seed_penalty)

