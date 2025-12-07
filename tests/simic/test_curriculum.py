"""Test UCB1 blueprint curriculum."""
import math

import pytest

from esper.simic.curriculum import BlueprintCurriculum


class TestBlueprintCurriculum:
    """Verify UCB1 curriculum for blueprint selection."""

    def test_default_exploration_weight_is_sqrt2(self):
        """Default exploration weight should be sqrt(2) for [0,1] rewards."""
        curriculum = BlueprintCurriculum(
            blueprints=["norm", "lora"],
            complexity=[100, 6000],
        )
        assert curriculum.exploration_weight == pytest.approx(math.sqrt(2))

    def test_exploration_weight_scales_with_reward_range(self):
        """Exploration weight should scale with reward range."""
        curriculum = BlueprintCurriculum(
            blueprints=["norm", "lora"],
            complexity=[100, 6000],
            reward_range=(0.0, 10.0),
        )
        # sqrt(2) * 10 = 14.14...
        assert curriculum.exploration_weight == pytest.approx(math.sqrt(2) * 10)

    def test_explicit_exploration_weight_overrides_default(self):
        """Explicit exploration_weight should override the default."""
        curriculum = BlueprintCurriculum(
            blueprints=["norm", "lora"],
            complexity=[100, 6000],
            exploration_weight=5.0,
        )
        assert curriculum.exploration_weight == 5.0

    def test_initialization_phase_selects_unexplored(self):
        """Unexplored blueprints should be selected in initialization phase."""
        curriculum = BlueprintCurriculum(
            blueprints=["norm", "lora", "attention"],
            complexity=[100, 6000, 50000],
        )

        # First three selections should be the three blueprints (deterministic order)
        assert curriculum.select_blueprint() == "norm"
        curriculum.record_outcome("norm", success=True, reward=0.5)

        assert curriculum.select_blueprint() == "lora"
        curriculum.record_outcome("lora", success=True, reward=0.5)

        assert curriculum.select_blueprint() == "attention"
        curriculum.record_outcome("attention", success=True, reward=0.5)

    def test_unexplored_blueprints_have_none_score(self):
        """Unexplored blueprints should return None score."""
        curriculum = BlueprintCurriculum(
            blueprints=["norm", "lora", "attention"],
            complexity=[100, 6000, 50000],
        )

        # Use norm once
        curriculum.record_outcome("norm", success=True, reward=0.5)

        scores = curriculum.get_ucb_scores()

        # norm has a score, others are None
        assert scores["norm"] is not None
        assert scores["lora"] is None
        assert scores["attention"] is None

    def test_ucb_updates_after_success(self):
        """Successful fossilization should update UCB stats."""
        curriculum = BlueprintCurriculum(
            blueprints=["norm", "lora"],
            complexity=[100, 6000],
        )

        # Record successful fossilization
        curriculum.record_outcome("norm", success=True, reward=1.0)

        stats = curriculum.get_stats("norm")
        assert stats["trials"] == 1
        assert stats["successes"] == 1

    def test_select_blueprint_uses_ucb1_after_initialization(self):
        """After init phase, select_blueprint should use UCB1 scores."""
        curriculum = BlueprintCurriculum(
            blueprints=["norm", "lora"],
            complexity=[100, 6000],
        )

        # Complete initialization phase
        curriculum.record_outcome("norm", success=True, reward=0.5)
        curriculum.record_outcome("lora", success=True, reward=0.5)

        # Now selection uses UCB1
        scores = curriculum.get_ucb_scores()
        selected = curriculum.select_blueprint()
        assert selected == max(scores, key=lambda k: scores[k])

    def test_complexity_penalty_applied_to_reward(self):
        """Complexity penalty should be applied to recorded reward."""
        curriculum = BlueprintCurriculum(
            blueprints=["simple", "complex"],
            complexity=[100, 1000],  # complex is 10x more complex
            complexity_penalty=0.5,
        )

        # Same raw reward to both
        curriculum.record_outcome("simple", success=True, reward=1.0)
        curriculum.record_outcome("complex", success=True, reward=1.0)

        # simple: normalized_complexity = 0.1, penalty = 0.5 * 0.1 = 0.05
        # complex: normalized_complexity = 1.0, penalty = 0.5 * 1.0 = 0.5

        simple_stats = curriculum.get_stats("simple")
        complex_stats = curriculum.get_stats("complex")

        # Simple should have higher mean_reward due to lower complexity penalty
        assert simple_stats["mean_reward"] > complex_stats["mean_reward"]

    def test_high_reward_overcomes_complexity_penalty(self):
        """High mean reward should overcome complexity penalty."""
        curriculum = BlueprintCurriculum(
            blueprints=["norm", "mlp"],
            complexity=[100, 1200000],
            complexity_penalty=0.1,
        )

        # Give mlp many high rewards (normalized to 1.0)
        for _ in range(20):
            curriculum.record_outcome("mlp", success=True, reward=1.0)

        # Give norm low rewards (normalized to 0.0)
        for _ in range(20):
            curriculum.record_outcome("norm", success=False, reward=0.0)

        scores = curriculum.get_ucb_scores()

        # mlp should have higher score despite complexity
        assert scores["mlp"] > scores["norm"]

    def test_invalid_blueprint_raises_error(self):
        """Recording outcome for unknown blueprint should raise error."""
        curriculum = BlueprintCurriculum(
            blueprints=["norm", "lora"],
            complexity=[100, 6000],
        )

        with pytest.raises(ValueError, match="Unknown blueprint"):
            curriculum.record_outcome("invalid", success=True, reward=1.0)

    def test_mismatched_lengths_raises_error(self):
        """Creating curriculum with mismatched lists should raise error."""
        with pytest.raises(ValueError, match="same length"):
            BlueprintCurriculum(
                blueprints=["norm", "lora"],
                complexity=[100],  # Wrong length
            )

    def test_stats_track_successes_and_failures(self):
        """Stats should accurately track successes and failures."""
        curriculum = BlueprintCurriculum(
            blueprints=["norm"],
            complexity=[100],
            complexity_penalty=0.0,  # Disable penalty for this test
        )

        # Record mixed outcomes
        curriculum.record_outcome("norm", success=True, reward=1.0)
        curriculum.record_outcome("norm", success=False, reward=0.0)
        curriculum.record_outcome("norm", success=True, reward=0.5)

        stats = curriculum.get_stats("norm")
        assert stats["trials"] == 3
        assert stats["successes"] == 2
        assert stats["mean_reward"] == pytest.approx(0.5)

    def test_reward_normalization(self):
        """Rewards should be normalized to [0, 1] range."""
        curriculum = BlueprintCurriculum(
            blueprints=["norm"],
            complexity=[100],
            complexity_penalty=0.0,
            reward_range=(0.0, 10.0),
        )

        # Reward of 5.0 in range [0, 10] should normalize to 0.5
        curriculum.record_outcome("norm", success=True, reward=5.0)

        stats = curriculum.get_stats("norm")
        assert stats["mean_reward"] == pytest.approx(0.5)

    def test_reward_clipping(self):
        """Rewards outside range should be clipped to [0, 1]."""
        curriculum = BlueprintCurriculum(
            blueprints=["norm"],
            complexity=[100],
            complexity_penalty=0.0,
            reward_range=(0.0, 1.0),
        )

        # Reward above max
        curriculum.record_outcome("norm", success=True, reward=10.0)
        assert curriculum.get_stats("norm")["mean_reward"] == pytest.approx(1.0)

    def test_ucb1_exploration_bonus_decreases_with_trials(self):
        """UCB exploration bonus should decrease as trials increase."""
        curriculum = BlueprintCurriculum(
            blueprints=["norm", "lora"],
            complexity=[100, 6000],
            complexity_penalty=0.0,
        )

        # Initialize both
        curriculum.record_outcome("norm", success=True, reward=0.5)
        curriculum.record_outcome("lora", success=True, reward=0.5)

        score_after_1 = curriculum.get_ucb_scores()["norm"]

        # More trials for norm
        for _ in range(10):
            curriculum.record_outcome("norm", success=True, reward=0.5)

        score_after_11 = curriculum.get_ucb_scores()["norm"]

        # Exploration bonus should decrease (but mean stays same)
        # Since mean is constant, total score should decrease
        assert score_after_11 < score_after_1

    def test_negative_adjusted_rewards_are_valid(self):
        """Low rewards with high complexity can result in negative adjusted rewards."""
        curriculum = BlueprintCurriculum(
            blueprints=["complex"],
            complexity=[100],  # normalized to 1.0
            complexity_penalty=0.5,
        )

        # Zero reward with 0.5 penalty on complexity 1.0 = -0.5
        curriculum.record_outcome("complex", success=False, reward=0.0)

        stats = curriculum.get_stats("complex")
        assert stats["mean_reward"] == pytest.approx(-0.5)
