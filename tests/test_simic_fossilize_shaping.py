"""Unit tests for FOSSILIZE reward shaping."""

import pytest

from esper.leyline import SeedStage
from esper.simic.rewards import RewardConfig, SeedInfo, _advance_shaping


def make_seed_info(stage: SeedStage, improvement: float, total_improvement: float | None = None) -> SeedInfo:
    """Helper to construct minimal SeedInfo for shaping tests.

    Args:
        stage: Seed lifecycle stage
        improvement: Improvement since stage start
        total_improvement: Total improvement since germination (defaults to same as improvement)
    """
    if total_improvement is None:
        total_improvement = improvement
    return SeedInfo(
        stage=stage.value,
        improvement_since_stage_start=improvement,
        total_improvement=total_improvement,
        epochs_in_stage=1,
    )


def test_advance_shaping_penalizes_training_invalid():
    """FOSSILIZE in TRAINING is INVALID and should be heavily penalized."""
    config = RewardConfig.default()
    seed_info = make_seed_info(SeedStage.TRAINING, improvement=1.0)

    shaping = _advance_shaping(seed_info, config)

    # Invalid action penalty: -1.0 (wasted action)
    assert shaping == pytest.approx(-1.0)


def test_advance_shaping_penalizes_blending_invalid():
    """FOSSILIZE in BLENDING is INVALID and should be heavily penalized."""
    config = RewardConfig.default()
    seed_info = make_seed_info(SeedStage.BLENDING, improvement=0.0)

    shaping = _advance_shaping(seed_info, config)

    # Invalid action penalty: -1.0 (wasted action)
    assert shaping == pytest.approx(-1.0)


def test_advance_shaping_penalizes_shadowing_invalid():
    """FOSSILIZE in SHADOWING is INVALID and should be heavily penalized."""
    config = RewardConfig.default()
    seed_info = make_seed_info(SeedStage.SHADOWING, improvement=1.0)

    shaping = _advance_shaping(seed_info, config)

    # Invalid action penalty: -1.0 (wasted action)
    assert shaping == pytest.approx(-1.0)


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


def test_advance_shaping_penalizes_probationary_without_total_improvement():
    """FOSSILIZE in PROBATIONARY without total_improvement is penalized.

    The shaping now aligns with G5 gate criteria: total_improvement > 0.
    A seed that hasn't improved since germination should not be fossilized.
    Penalty is -1.0 (matching INVALID action penalty magnitude).
    """
    config = RewardConfig.default()
    # Stage improvement is positive, but total improvement (since germination) is zero
    # This simulates a seed that recovered but never exceeded its germination baseline
    seed_info = make_seed_info(SeedStage.PROBATIONARY, improvement=0.5, total_improvement=0.0)

    shaping = _advance_shaping(seed_info, config)

    # G5 gate alignment: penalty for failed fossilize matches INVALID action penalty
    assert shaping == pytest.approx(-1.0)


def test_advance_shaping_penalizes_missing_seed():
    """FOSSILIZE with no active seed uses the no-seed penalty."""
    config = RewardConfig.default()

    shaping = _advance_shaping(None, config)

    assert shaping == pytest.approx(config.advance_no_seed_penalty)

