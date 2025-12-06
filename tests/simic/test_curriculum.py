"""Test UCB1 blueprint curriculum."""
import pytest

from esper.simic.curriculum import BlueprintCurriculum


class TestBlueprintCurriculum:
    """Verify UCB1 curriculum for blueprint selection."""

    def test_initial_curriculum_favors_simple(self):
        """Initial curriculum should favor simpler blueprints."""
        curriculum = BlueprintCurriculum(
            blueprints=["norm", "lora", "attention", "mlp"],
            complexity=[100, 6000, 50000, 1200000],
        )

        # Initially, simpler blueprints should have higher UCB scores
        scores = curriculum.get_ucb_scores()

        # norm (simplest) should have highest initial score
        assert scores["norm"] >= scores["mlp"]

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

    def test_ucb_exploration_bonus(self):
        """Unexplored blueprints should get exploration bonus."""
        curriculum = BlueprintCurriculum(
            blueprints=["norm", "lora", "attention"],
            complexity=[100, 6000, 50000],
        )

        # Use norm many times
        for _ in range(10):
            curriculum.record_outcome("norm", success=True, reward=0.5)

        scores = curriculum.get_ucb_scores()

        # Unexplored attention should have exploration bonus
        assert scores["attention"] > 0  # Has exploration bonus despite no trials

    def test_select_blueprint_returns_highest_score(self):
        """select_blueprint should return blueprint with highest UCB score."""
        curriculum = BlueprintCurriculum(
            blueprints=["norm", "lora", "attention"],
            complexity=[100, 6000, 50000],
        )

        # Initially should favor simple blueprints (norm has highest score)
        selected = curriculum.select_blueprint()
        scores = curriculum.get_ucb_scores()
        assert selected == max(scores, key=scores.get)

    def test_complexity_penalty_favors_simple_initially(self):
        """Complexity penalty should favor simpler blueprints when unexplored."""
        curriculum = BlueprintCurriculum(
            blueprints=["norm", "mlp"],
            complexity=[100, 1200000],  # mlp is 12000x more complex
            complexity_penalty=0.5,
        )

        scores = curriculum.get_ucb_scores()

        # norm should have much higher score due to complexity penalty
        assert scores["norm"] > scores["mlp"]

    def test_high_reward_overcomes_complexity_penalty(self):
        """High mean reward should overcome complexity penalty."""
        curriculum = BlueprintCurriculum(
            blueprints=["norm", "mlp"],
            complexity=[100, 1200000],
            complexity_penalty=0.1,
        )

        # Give mlp many high rewards
        for _ in range(20):
            curriculum.record_outcome("mlp", success=True, reward=10.0)

        # Give norm low rewards
        for _ in range(20):
            curriculum.record_outcome("norm", success=False, reward=0.1)

        scores = curriculum.get_ucb_scores()

        # mlp should now have higher score despite complexity
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
        )

        # Record mixed outcomes
        curriculum.record_outcome("norm", success=True, reward=1.0)
        curriculum.record_outcome("norm", success=False, reward=0.0)
        curriculum.record_outcome("norm", success=True, reward=0.5)

        stats = curriculum.get_stats("norm")
        assert stats["trials"] == 3
        assert stats["successes"] == 2
        assert stats["mean_reward"] == 0.5  # (1.0 + 0.0 + 0.5) / 3

    def test_exploration_weight_affects_scores(self):
        """Higher exploration weight should increase bonus for unexplored."""
        curriculum_low = BlueprintCurriculum(
            blueprints=["norm", "lora"],
            complexity=[100, 6000],
            exploration_weight=0.5,
        )

        curriculum_high = BlueprintCurriculum(
            blueprints=["norm", "lora"],
            complexity=[100, 6000],
            exploration_weight=5.0,
        )

        # Use norm once in both
        curriculum_low.record_outcome("norm", success=True, reward=1.0)
        curriculum_high.record_outcome("norm", success=True, reward=1.0)

        scores_low = curriculum_low.get_ucb_scores()
        scores_high = curriculum_high.get_ucb_scores()

        # High exploration should give more bonus to unexplored lora
        assert scores_high["lora"] > scores_low["lora"]
