"""Tests for simic.py - policy learning data structures."""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime, timezone

from esper.simic import (
    TrainingSnapshot,
    SimicAction,
    ActionTaken,
    StepOutcome,
    DecisionPoint,
    Episode,
    EpisodeCollector,
    DatasetManager,
)


class TestTrainingSnapshot:
    """Tests for TrainingSnapshot dataclass."""

    def test_default_values(self):
        """Test default values."""
        snap = TrainingSnapshot()
        assert snap.epoch == 0
        assert snap.val_accuracy == 0.0
        assert snap.has_active_seed is False

    def test_vector_size_matches(self):
        """Test that to_vector() length matches vector_size()."""
        snap = TrainingSnapshot(epoch=10, val_accuracy=75.0)
        vec = snap.to_vector()
        assert len(vec) == TrainingSnapshot.vector_size()

    def test_vector_contains_expected_values(self):
        """Test that vector contains the values we set."""
        snap = TrainingSnapshot(
            epoch=5,
            val_accuracy=80.0,
            plateau_epochs=2,
        )
        vec = snap.to_vector()
        assert vec[0] == 5.0  # epoch
        assert vec[6] == 80.0  # val_accuracy
        assert vec[8] == 2.0  # plateau_epochs

    def test_serialization_roundtrip(self):
        """Test to_dict/from_dict roundtrip."""
        original = TrainingSnapshot(
            epoch=10,
            global_step=1000,
            train_loss=0.5,
            val_loss=0.6,
            val_accuracy=75.5,
            plateau_epochs=3,
            loss_history_5=(0.9, 0.8, 0.7, 0.6, 0.5),
            accuracy_history_5=(60.0, 65.0, 70.0, 73.0, 75.5),
            has_active_seed=True,
            seed_stage=3,
            seed_alpha=0.5,
        )

        d = original.to_dict()
        restored = TrainingSnapshot.from_dict(d)

        assert restored.epoch == original.epoch
        assert restored.val_accuracy == original.val_accuracy
        assert restored.loss_history_5 == original.loss_history_5
        assert restored.has_active_seed == original.has_active_seed
        assert restored.seed_stage == original.seed_stage

    def test_json_serializable(self):
        """Test that to_dict() output is JSON serializable."""
        snap = TrainingSnapshot(epoch=5, val_accuracy=75.0)
        d = snap.to_dict()
        json_str = json.dumps(d)  # Should not raise
        assert "epoch" in json_str

    def test_inf_handling(self):
        """Test that infinity values are handled in serialization."""
        snap = TrainingSnapshot(best_val_loss=float('inf'))
        d = snap.to_dict()
        assert d["best_val_loss"] is None  # inf -> None for JSON

        restored = TrainingSnapshot.from_dict(d)
        assert restored.best_val_loss == float('inf')


class TestActionTaken:
    """Tests for ActionTaken dataclass."""

    def test_vector_size_matches(self):
        """Test that to_vector() length matches vector_size()."""
        action = ActionTaken(action=SimicAction.WAIT)
        vec = action.to_vector()
        assert len(vec) == ActionTaken.vector_size()

    def test_one_hot_encoding(self):
        """Test that action is one-hot encoded in vector."""
        for i, simic_action in enumerate(SimicAction):
            action = ActionTaken(action=simic_action)
            vec = action.to_vector()
            # First 4 elements are one-hot
            for j in range(4):
                expected = 1.0 if j == i else 0.0
                assert vec[j] == expected, f"Action {simic_action}: vec[{j}] should be {expected}"

    def test_serialization_roundtrip(self):
        """Test to_dict/from_dict roundtrip."""
        original = ActionTaken(
            action=SimicAction.GERMINATE_CONV,
            blueprint_id="conv_enhance",
            confidence=0.85,
            reason="Plateau detected",
        )

        d = original.to_dict()
        restored = ActionTaken.from_dict(d)

        assert restored.action == original.action
        assert restored.blueprint_id == original.blueprint_id
        assert restored.confidence == original.confidence
        assert restored.reason == original.reason


class TestStepOutcome:
    """Tests for StepOutcome dataclass."""

    def test_compute_reward(self):
        """Test reward computation."""
        outcome = StepOutcome(accuracy_change=1.0)
        reward = outcome.compute_reward()
        assert reward == 10.0  # accuracy_change * 10
        assert outcome.reward == 10.0

    def test_negative_reward(self):
        """Test negative reward for accuracy drop."""
        outcome = StepOutcome(accuracy_change=-0.5)
        reward = outcome.compute_reward()
        assert reward == -5.0

    def test_serialization_roundtrip(self):
        """Test to_dict/from_dict roundtrip."""
        original = StepOutcome(
            accuracy_after=76.0,
            accuracy_change=1.5,
            loss_after=0.5,
            reward=15.0,
        )

        d = original.to_dict()
        restored = StepOutcome.from_dict(d)

        assert restored.accuracy_after == original.accuracy_after
        assert restored.reward == original.reward


class TestDecisionPoint:
    """Tests for DecisionPoint dataclass."""

    def test_serialization_roundtrip(self):
        """Test to_dict/from_dict roundtrip."""
        original = DecisionPoint(
            observation=TrainingSnapshot(epoch=5, val_accuracy=70.0),
            action=ActionTaken(action=SimicAction.WAIT),
            outcome=StepOutcome(accuracy_after=71.0, accuracy_change=1.0, reward=10.0),
        )

        d = original.to_dict()
        restored = DecisionPoint.from_dict(d)

        assert restored.observation.epoch == 5
        assert restored.action.action == SimicAction.WAIT
        assert restored.outcome.reward == 10.0

    def test_timestamp_preserved(self):
        """Test that timestamp is preserved in serialization."""
        original = DecisionPoint(
            observation=TrainingSnapshot(),
            action=ActionTaken(action=SimicAction.WAIT),
            outcome=StepOutcome(),
        )

        d = original.to_dict()
        restored = DecisionPoint.from_dict(d)

        # Timestamps should be close (within 1 second)
        diff = abs((restored.timestamp - original.timestamp).total_seconds())
        assert diff < 1.0


class TestEpisode:
    """Tests for Episode dataclass."""

    def test_total_reward(self):
        """Test total_reward computation."""
        episode = Episode(episode_id="test")
        episode.decisions = [
            DecisionPoint(
                observation=TrainingSnapshot(),
                action=ActionTaken(action=SimicAction.WAIT),
                outcome=StepOutcome(reward=10.0),
            ),
            DecisionPoint(
                observation=TrainingSnapshot(),
                action=ActionTaken(action=SimicAction.WAIT),
                outcome=StepOutcome(reward=5.0),
            ),
        ]

        assert episode.total_reward() == 15.0

    def test_to_training_data(self):
        """Test to_training_data conversion."""
        episode = Episode(episode_id="test")
        episode.decisions = [
            DecisionPoint(
                observation=TrainingSnapshot(epoch=1),
                action=ActionTaken(action=SimicAction.WAIT),
                outcome=StepOutcome(reward=10.0),
            ),
        ]

        data = episode.to_training_data()
        assert len(data) == 1
        obs, act, reward = data[0]
        assert len(obs) == TrainingSnapshot.vector_size()
        assert len(act) == ActionTaken.vector_size()
        assert reward == 10.0

    def test_save_load_roundtrip(self):
        """Test save/load roundtrip."""
        episode = Episode(
            episode_id="test_save_load",
            max_epochs=20,
            final_accuracy=75.0,
            best_accuracy=76.0,
        )
        episode.decisions = [
            DecisionPoint(
                observation=TrainingSnapshot(epoch=1, val_accuracy=70.0),
                action=ActionTaken(action=SimicAction.WAIT),
                outcome=StepOutcome(reward=10.0),
            ),
        ]
        episode.ended_at = datetime.now(timezone.utc)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            episode.save(path)
            loaded = Episode.load(path)

            assert loaded.episode_id == episode.episode_id
            assert loaded.final_accuracy == episode.final_accuracy
            assert len(loaded.decisions) == 1
            assert loaded.decisions[0].observation.val_accuracy == 70.0
        finally:
            Path(path).unlink()


class TestEpisodeCollector:
    """Tests for EpisodeCollector."""

    def test_basic_collection(self):
        """Test basic collection workflow."""
        collector = EpisodeCollector()
        collector.start_episode("test_001", max_epochs=10)

        collector.record_observation(TrainingSnapshot(epoch=1))
        collector.record_action(ActionTaken(action=SimicAction.WAIT))
        collector.record_outcome(StepOutcome(accuracy_change=1.0))

        episode = collector.end_episode(
            final_accuracy=75.0,
            best_accuracy=76.0,
        )

        assert episode.episode_id == "test_001"
        assert len(episode.decisions) == 1
        assert episode.decisions[0].outcome.reward == 10.0  # computed

    def test_multiple_decisions(self):
        """Test collecting multiple decisions."""
        collector = EpisodeCollector()
        collector.start_episode("test_002", max_epochs=10)

        for i in range(5):
            collector.record_observation(TrainingSnapshot(epoch=i+1))
            collector.record_action(ActionTaken(action=SimicAction.WAIT))
            collector.record_outcome(StepOutcome(accuracy_change=0.5))

        episode = collector.end_episode(final_accuracy=75.0, best_accuracy=76.0)

        assert len(episode.decisions) == 5

    def test_error_without_start(self):
        """Test error when recording without starting episode."""
        collector = EpisodeCollector()

        with pytest.raises(RuntimeError):
            collector.record_observation(TrainingSnapshot())

    def test_error_without_observation(self):
        """Test error when recording action without observation."""
        collector = EpisodeCollector()
        collector.start_episode("test", max_epochs=10)

        with pytest.raises(RuntimeError):
            collector.record_action(ActionTaken(action=SimicAction.WAIT))


class TestDatasetManager:
    """Tests for DatasetManager."""

    def test_save_load_episode(self):
        """Test saving and loading an episode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DatasetManager(tmpdir)

            episode = Episode(
                episode_id="test_dm_001",
                final_accuracy=75.0,
                best_accuracy=76.0,
            )

            dm.save_episode(episode)
            loaded = dm.load_episode("test_dm_001")

            assert loaded.episode_id == "test_dm_001"
            assert loaded.final_accuracy == 75.0

    def test_list_episodes(self):
        """Test listing episodes in directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DatasetManager(tmpdir)

            for i in range(3):
                episode = Episode(episode_id=f"ep_{i:03d}")
                dm.save_episode(episode)

            ids = dm.list_episodes()
            assert len(ids) == 3
            assert "ep_000" in ids
            assert "ep_002" in ids

    def test_summary(self):
        """Test dataset summary statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DatasetManager(tmpdir)

            for i in range(3):
                episode = Episode(
                    episode_id=f"ep_{i:03d}",
                    final_accuracy=70.0 + i * 5,
                )
                episode.decisions = [
                    DecisionPoint(
                        observation=TrainingSnapshot(),
                        action=ActionTaken(action=SimicAction.WAIT),
                        outcome=StepOutcome(reward=10.0),
                    )
                ]
                dm.save_episode(episode)

            summary = dm.summary()
            assert summary["count"] == 3
            assert summary["total_decisions"] == 3
            assert summary["avg_final_accuracy"] == 75.0  # (70+75+80)/3

    def test_empty_summary(self):
        """Test summary on empty dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DatasetManager(tmpdir)
            summary = dm.summary()
            assert summary["count"] == 0


class TestPolicyNetwork:
    """Tests for PolicyNetwork (requires torch)."""

    @pytest.fixture
    def policy(self):
        """Create a PolicyNetwork instance."""
        from esper.simic import PolicyNetwork
        return PolicyNetwork()

    def test_predict_returns_action(self, policy):
        """Test that predict returns a SimicAction."""
        snap = TrainingSnapshot(epoch=10, val_accuracy=70.0)
        action = policy.predict(snap)
        assert isinstance(action, SimicAction)

    def test_predict_probs_sums_to_one(self, policy):
        """Test that predict_probs returns valid probabilities."""
        snap = TrainingSnapshot(epoch=10, val_accuracy=70.0)
        probs = policy.predict_probs(snap)

        assert len(probs) == len(SimicAction)  # 7 actions now
        total = sum(probs.values())
        assert abs(total - 1.0) < 0.01  # Should sum to ~1

    def test_save_load_roundtrip(self, policy):
        """Test saving and loading model weights."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        try:
            # Get prediction before save
            snap = TrainingSnapshot(epoch=10, val_accuracy=70.0)
            pred_before = policy.predict_probs(snap)

            # Save and load
            policy.save(path)

            from esper.simic import PolicyNetwork
            policy2 = PolicyNetwork()
            policy2.load(path)

            pred_after = policy2.predict_probs(snap)

            # Predictions should be identical
            for action in SimicAction:
                assert abs(pred_before[action] - pred_after[action]) < 0.01
        finally:
            Path(path).unlink()
