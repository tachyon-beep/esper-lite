"""Tests for EpisodeOutcome schema."""

import pytest
from dataclasses import FrozenInstanceError

from esper.karn.store import EpisodeOutcome


def test_episode_outcome_creation():
    """Test basic EpisodeOutcome creation with all fields."""
    outcome = EpisodeOutcome(
        env_id=0,
        episode=100,
        reward_mode="shaped",
        final_accuracy=0.85,
        param_ratio=1.2,
        stability_score=0.95,
        slot_count=3,
        fossilized_count=1,
    )
    assert outcome.final_accuracy == 0.85
    assert outcome.param_ratio == 1.2
    assert outcome.stability_score == 0.95
    assert outcome.env_id == 0
    assert outcome.episode == 100
    assert outcome.reward_mode == "shaped"
    assert outcome.slot_count == 3
    assert outcome.fossilized_count == 1
    assert outcome.timestamp > 0  # Should be set by default_factory


def test_episode_outcome_frozen():
    """Test that EpisodeOutcome is immutable (frozen=True)."""
    outcome = EpisodeOutcome(
        env_id=0,
        episode=1,
        reward_mode="shaped",
        final_accuracy=0.5,
        param_ratio=1.0,
        stability_score=0.8,
    )
    with pytest.raises(FrozenInstanceError):
        outcome.final_accuracy = 0.9


def test_episode_outcome_default_values():
    """Test that optional fields have correct defaults."""
    outcome = EpisodeOutcome(
        env_id=1,
        episode=50,
        reward_mode="simplified",
        final_accuracy=0.7,
        param_ratio=1.1,
        stability_score=0.9,
    )
    assert outcome.slot_count == 0
    assert outcome.fossilized_count == 0
    # timestamp should be auto-set
    assert isinstance(outcome.timestamp, float)


def test_episode_outcome_different_reward_modes():
    """Test EpisodeOutcome with various reward modes."""
    modes = ["shaped", "simplified", "sparse", "minimal"]
    for mode in modes:
        outcome = EpisodeOutcome(
            env_id=0,
            episode=1,
            reward_mode=mode,
            final_accuracy=0.5,
            param_ratio=1.0,
            stability_score=0.8,
        )
        assert outcome.reward_mode == mode


def test_episode_outcome_slots_enabled():
    """Verify slots=True by checking __slots__ attribute."""
    # Slotted dataclasses have __slots__ defined
    assert hasattr(EpisodeOutcome, "__slots__")
