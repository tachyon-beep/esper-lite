"""Tests for EpisodeOutcome schema."""

from datetime import datetime, timezone

import pytest
from dataclasses import FrozenInstanceError

from esper.leyline import EpisodeOutcome


def test_episode_outcome_creation():
    """Test basic EpisodeOutcome creation with all fields."""
    outcome = EpisodeOutcome(
        env_id=0,
        episode_idx=100,
        final_accuracy=0.85,
        param_ratio=1.2,
        num_fossilized=1,
        num_contributing_fossilized=1,
        episode_reward=15.5,
        stability_score=0.95,
        reward_mode="shaped",
    )
    assert outcome.final_accuracy == 0.85
    assert outcome.param_ratio == 1.2
    assert outcome.stability_score == 0.95
    assert outcome.env_id == 0
    assert outcome.episode_idx == 100
    assert outcome.reward_mode == "shaped"
    assert outcome.num_fossilized == 1
    assert outcome.num_contributing_fossilized == 1
    assert outcome.episode_reward == 15.5
    assert isinstance(outcome.timestamp, datetime)


def test_episode_outcome_frozen():
    """Test that EpisodeOutcome is immutable (frozen=True)."""
    outcome = EpisodeOutcome(
        env_id=0,
        episode_idx=1,
        final_accuracy=0.5,
        param_ratio=1.0,
        num_fossilized=0,
        num_contributing_fossilized=0,
        episode_reward=0.0,
        stability_score=0.8,
        reward_mode="shaped",
    )
    with pytest.raises(FrozenInstanceError):
        outcome.final_accuracy = 0.9


def test_episode_outcome_timestamp_default():
    """Test that timestamp defaults to current UTC time."""
    before = datetime.now(timezone.utc)
    outcome = EpisodeOutcome(
        env_id=1,
        episode_idx=50,
        final_accuracy=0.7,
        param_ratio=1.1,
        num_fossilized=2,
        num_contributing_fossilized=1,
        episode_reward=10.0,
        stability_score=0.9,
        reward_mode="simplified",
    )
    after = datetime.now(timezone.utc)
    assert before <= outcome.timestamp <= after


def test_episode_outcome_different_reward_modes():
    """Test EpisodeOutcome with various reward modes."""
    modes = ["shaped", "simplified", "sparse", "minimal"]
    for mode in modes:
        outcome = EpisodeOutcome(
            env_id=0,
            episode_idx=1,
            final_accuracy=0.5,
            param_ratio=1.0,
            num_fossilized=0,
            num_contributing_fossilized=0,
            episode_reward=0.0,
            stability_score=0.8,
            reward_mode=mode,
        )
        assert outcome.reward_mode == mode


class TestDominates:
    """Tests for Pareto dominance method."""

    def _make_outcome(
        self,
        final_accuracy: float = 0.8,
        param_ratio: float = 1.0,
        stability_score: float = 0.9,
    ) -> EpisodeOutcome:
        """Helper to create test outcomes."""
        return EpisodeOutcome(
            env_id=0,
            episode_idx=1,
            final_accuracy=final_accuracy,
            param_ratio=param_ratio,
            num_fossilized=0,
            num_contributing_fossilized=0,
            episode_reward=0.0,
            stability_score=stability_score,
            reward_mode="shaped",
        )

    def test_dominates_strictly_better_all(self):
        """Test dominance when strictly better on all objectives."""
        better = self._make_outcome(
            final_accuracy=0.9,
            param_ratio=0.8,  # lower is better
            stability_score=0.95,
        )
        worse = self._make_outcome(
            final_accuracy=0.7,
            param_ratio=1.2,
            stability_score=0.8,
        )
        assert better.dominates(worse)
        assert not worse.dominates(better)

    def test_dominates_better_accuracy_equal_others(self):
        """Test dominance when better on accuracy, equal on others."""
        better = self._make_outcome(
            final_accuracy=0.9,
            param_ratio=1.0,
            stability_score=0.9,
        )
        worse = self._make_outcome(
            final_accuracy=0.8,
            param_ratio=1.0,
            stability_score=0.9,
        )
        assert better.dominates(worse)
        assert not worse.dominates(better)

    def test_dominates_better_param_ratio_equal_others(self):
        """Test dominance when better on param_ratio (lower), equal on others."""
        better = self._make_outcome(
            final_accuracy=0.8,
            param_ratio=0.9,  # lower is better
            stability_score=0.9,
        )
        worse = self._make_outcome(
            final_accuracy=0.8,
            param_ratio=1.0,
            stability_score=0.9,
        )
        assert better.dominates(worse)
        assert not worse.dominates(better)

    def test_dominates_better_stability_equal_others(self):
        """Test dominance when better on stability, equal on others."""
        better = self._make_outcome(
            final_accuracy=0.8,
            param_ratio=1.0,
            stability_score=0.95,
        )
        worse = self._make_outcome(
            final_accuracy=0.8,
            param_ratio=1.0,
            stability_score=0.85,
        )
        assert better.dominates(worse)
        assert not worse.dominates(better)

    def test_no_self_dominance(self):
        """Test that an outcome cannot dominate itself."""
        outcome = self._make_outcome()
        assert not outcome.dominates(outcome)

    def test_no_dominance_pareto_incomparable(self):
        """Test no dominance for Pareto-incomparable outcomes (trade-offs)."""
        # Better accuracy, worse param_ratio
        a = self._make_outcome(
            final_accuracy=0.9,
            param_ratio=1.2,
            stability_score=0.9,
        )
        # Worse accuracy, better param_ratio
        b = self._make_outcome(
            final_accuracy=0.7,
            param_ratio=0.8,
            stability_score=0.9,
        )
        assert not a.dominates(b)
        assert not b.dominates(a)

    def test_dominance_requires_strict_improvement(self):
        """Test that equal on all objectives means no dominance."""
        a = self._make_outcome(
            final_accuracy=0.8,
            param_ratio=1.0,
            stability_score=0.9,
        )
        b = self._make_outcome(
            final_accuracy=0.8,
            param_ratio=1.0,
            stability_score=0.9,
        )
        assert not a.dominates(b)
        assert not b.dominates(a)


class TestToDict:
    """Tests for to_dict serialization method."""

    def test_to_dict_contains_all_fields(self):
        """Test that to_dict includes all fields."""
        outcome = EpisodeOutcome(
            env_id=0,
            episode_idx=100,
            final_accuracy=0.85,
            param_ratio=1.2,
            num_fossilized=3,
            num_contributing_fossilized=2,
            episode_reward=15.5,
            stability_score=0.95,
            reward_mode="shaped",
        )
        d = outcome.to_dict()
        assert d["env_id"] == 0
        assert d["episode_idx"] == 100
        assert d["final_accuracy"] == 0.85
        assert d["param_ratio"] == 1.2
        assert d["num_fossilized"] == 3
        assert d["num_contributing_fossilized"] == 2
        assert d["episode_reward"] == 15.5
        assert d["stability_score"] == 0.95
        assert d["reward_mode"] == "shaped"
        assert "timestamp" in d

    def test_to_dict_timestamp_is_iso_format(self):
        """Test that timestamp is serialized as ISO format string."""
        outcome = EpisodeOutcome(
            env_id=0,
            episode_idx=1,
            final_accuracy=0.5,
            param_ratio=1.0,
            num_fossilized=0,
            num_contributing_fossilized=0,
            episode_reward=0.0,
            stability_score=0.8,
            reward_mode="shaped",
        )
        d = outcome.to_dict()
        # Should be a valid ISO format string
        parsed = datetime.fromisoformat(d["timestamp"])
        assert isinstance(parsed, datetime)

    def test_to_dict_json_serializable(self):
        """Test that to_dict output can be JSON serialized."""
        import json

        outcome = EpisodeOutcome(
            env_id=0,
            episode_idx=1,
            final_accuracy=0.5,
            param_ratio=1.0,
            num_fossilized=0,
            num_contributing_fossilized=0,
            episode_reward=0.0,
            stability_score=0.8,
            reward_mode="shaped",
        )
        d = outcome.to_dict()
        # Should not raise
        json_str = json.dumps(d)
        assert isinstance(json_str, str)
