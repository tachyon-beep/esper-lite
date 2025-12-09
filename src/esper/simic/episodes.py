"""Episode Data Structures for Policy Learning

This module contains the data structures for collecting and managing
training episodes for Tamiyo policy learning:
- TrainingSnapshot: What Tamiyo sees (observations)
- ActionTaken: What Tamiyo does (actions)
- StepOutcome: What happens (outcomes/rewards)
- Episode: Complete trajectories for learning
- DatasetManager: For loading/saving episode datasets (offline RL)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from esper.leyline import FieldReport as LeylineFieldReport
from esper.simic.features import safe


# =============================================================================
# Observation Space
# =============================================================================

@dataclass
class TrainingSnapshot:
    """A snapshot of training state at decision time.

    This is what Tamiyo "sees" when making a decision.
    Kept flat and numeric for easy conversion to tensors.
    """

    # Timing
    epoch: int = 0
    global_step: int = 0

    # Loss metrics
    train_loss: float = 0.0
    val_loss: float = 0.0
    loss_delta: float = 0.0  # Positive = improving

    # Accuracy metrics
    train_accuracy: float = 0.0
    val_accuracy: float = 0.0
    accuracy_delta: float = 0.0

    # Trend indicators
    plateau_epochs: int = 0
    best_val_accuracy: float = 0.0
    best_val_loss: float = float('inf')

    # Recent history (last 5 epochs, padded with zeros if fewer)
    loss_history_5: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)
    accuracy_history_5: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)

    # Seed state (if any active seed)
    has_active_seed: bool = False
    seed_stage: int = 0  # SeedStage as int for tensor compatibility
    seed_epochs_in_stage: int = 0
    seed_alpha: float = 0.0
    seed_improvement: float = 0.0

    # Slot state
    available_slots: int = 1

    def to_vector(self) -> list[float]:
        """Convert to flat vector for neural network input."""
        return [
            float(self.epoch),
            float(self.global_step),
            safe(self.train_loss, default=10.0),
            safe(self.val_loss, default=10.0),
            safe(self.loss_delta, default=0.0),
            self.train_accuracy,
            self.val_accuracy,
            safe(self.accuracy_delta, default=0.0),
            float(self.plateau_epochs),
            self.best_val_accuracy,
            safe(self.best_val_loss, default=10.0),
            *[safe(v, default=10.0) for v in self.loss_history_5],
            *self.accuracy_history_5,
            float(self.has_active_seed),
            float(self.seed_stage),
            float(self.seed_epochs_in_stage),
            self.seed_alpha,
            self.seed_improvement,
            float(self.available_slots),
        ]

    @staticmethod
    def vector_size() -> int:
        """Size of the vector representation."""
        # 11 scalar fields + 5 loss_history + 5 accuracy_history + 6 seed fields
        return 27

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "loss_delta": self.loss_delta,
            "train_accuracy": self.train_accuracy,
            "val_accuracy": self.val_accuracy,
            "accuracy_delta": self.accuracy_delta,
            "plateau_epochs": self.plateau_epochs,
            "best_val_accuracy": self.best_val_accuracy,
            "best_val_loss": self.best_val_loss if self.best_val_loss != float('inf') else None,
            "loss_history_5": list(self.loss_history_5),
            "accuracy_history_5": list(self.accuracy_history_5),
            "has_active_seed": self.has_active_seed,
            "seed_stage": self.seed_stage,
            "seed_epochs_in_stage": self.seed_epochs_in_stage,
            "seed_alpha": self.seed_alpha,
            "seed_improvement": self.seed_improvement,
            "available_slots": self.available_slots,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingSnapshot":
        """Create from dictionary."""
        return cls(
            epoch=d["epoch"],
            global_step=d["global_step"],
            train_loss=d["train_loss"],
            val_loss=d["val_loss"],
            loss_delta=d["loss_delta"],
            train_accuracy=d["train_accuracy"],
            val_accuracy=d["val_accuracy"],
            accuracy_delta=d["accuracy_delta"],
            plateau_epochs=d["plateau_epochs"],
            best_val_accuracy=d["best_val_accuracy"],
            best_val_loss=d["best_val_loss"] if d["best_val_loss"] is not None else float('inf'),
            loss_history_5=tuple(d["loss_history_5"]),
            accuracy_history_5=tuple(d["accuracy_history_5"]),
            has_active_seed=d["has_active_seed"],
            seed_stage=d["seed_stage"],
            seed_epochs_in_stage=d["seed_epochs_in_stage"],
            seed_alpha=d["seed_alpha"],
            seed_improvement=d["seed_improvement"],
            available_slots=d["available_slots"],
        )


# =============================================================================
# Action Space (imported from leyline.actions)
# =============================================================================

@dataclass
class ActionTaken:
    """Record of an action taken by Tamiyo.

    Uses the topology-specific action enum from leyline.actions.
    """

    action: object
    blueprint_id: str | None = None
    target_seed_id: str | None = None
    confidence: float = 1.0
    reason: str = ""

    def to_vector(self) -> list[float]:
        """Convert to flat vector (one-hot action + confidence)."""
        action_enum = self.action.__class__
        one_hot = [0.0] * len(action_enum)
        one_hot[self.action.value] = 1.0
        return one_hot + [self.confidence]

    @staticmethod
    def vector_size(action_enum) -> int:
        return len(action_enum) + 1  # action one-hot + confidence

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "action": self.action.name,
            "blueprint_id": self.blueprint_id,
            "target_seed_id": self.target_seed_id,
            "confidence": self.confidence,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, d: dict, action_enum) -> "ActionTaken":
        """Create from dictionary."""
        if action_enum is None:
            raise ValueError("action_enum is required to deserialize ActionTaken")
        return cls(
            action=action_enum[d["action"]],
            blueprint_id=d["blueprint_id"],
            target_seed_id=d["target_seed_id"],
            confidence=d["confidence"],
            reason=d["reason"],
        )


# =============================================================================
# Outcome / Reward
# =============================================================================

@dataclass
class StepOutcome:
    """Outcome of a single decision step.

    Captures what happened after Tamiyo made a decision.
    """

    # Immediate outcome (next epoch)
    accuracy_after: float = 0.0
    accuracy_change: float = 0.0  # Positive = improved
    loss_after: float = 0.0
    loss_change: float = 0.0  # Positive = improved (loss went down)

    # Seed outcome (if action affected a seed)
    seed_still_alive: bool = True
    seed_stage_after: int = 0

    # Computed reward (populated by compute_contribution_reward from simic.rewards)
    reward: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "accuracy_after": self.accuracy_after,
            "accuracy_change": self.accuracy_change,
            "loss_after": self.loss_after,
            "loss_change": self.loss_change,
            "seed_still_alive": self.seed_still_alive,
            "seed_stage_after": self.seed_stage_after,
            "reward": self.reward,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StepOutcome":
        """Create from dictionary."""
        return cls(
            accuracy_after=d["accuracy_after"],
            accuracy_change=d["accuracy_change"],
            loss_after=d["loss_after"],
            loss_change=d["loss_change"],
            seed_still_alive=d["seed_still_alive"],
            seed_stage_after=d["seed_stage_after"],
            reward=d["reward"],
        )


# =============================================================================
# Episode (Complete Trajectory)
# =============================================================================

@dataclass
class DecisionPoint:
    """A single decision point in an episode."""

    observation: TrainingSnapshot
    action: ActionTaken
    outcome: StepOutcome
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "observation": self.observation.to_dict(),
            "action": self.action.to_dict(),
            "outcome": self.outcome.to_dict(),
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict, action_enum) -> "DecisionPoint":
        """Create from dictionary."""
        return cls(
            observation=TrainingSnapshot.from_dict(d["observation"]),
            action=ActionTaken.from_dict(d["action"], action_enum=action_enum),
            outcome=StepOutcome.from_dict(d["outcome"]),
            timestamp=datetime.fromisoformat(d["timestamp"]),
        )


@dataclass
class Episode:
    """A complete training episode for policy learning.

    Contains the full trajectory of (observation, action, outcome) tuples
    from a single training run.
    """

    episode_id: str = ""

    # Episode metadata
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: datetime | None = None

    # Training config
    max_epochs: int = 0
    initial_lr: float = 0.0
    model_name: str = ""
    dataset_name: str = ""

    # Trajectory
    decisions: list[DecisionPoint] = field(default_factory=list)

    # Episode outcome
    final_accuracy: float = 0.0
    best_accuracy: float = 0.0
    total_seeds_created: int = 0
    total_seeds_fossilized: int = 0
    total_seeds_culled: int = 0

    # Field reports for seeds (from Kasmina)
    field_reports: list[LeylineFieldReport] = field(default_factory=list)

    def total_reward(self) -> float:
        """Sum of all step rewards."""
        return sum(d.outcome.reward for d in self.decisions)

    def to_training_data(self) -> list[tuple[list[float], list[float], float]]:
        """Convert to (observation, action, reward) tuples for training."""
        return [
            (d.observation.to_vector(), d.action.to_vector(), d.outcome.reward)
            for d in self.decisions
        ]

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "episode_id": self.episode_id,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "max_epochs": self.max_epochs,
            "initial_lr": self.initial_lr,
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "decisions": [d.to_dict() for d in self.decisions],
            "final_accuracy": self.final_accuracy,
            "best_accuracy": self.best_accuracy,
            "total_seeds_created": self.total_seeds_created,
            "total_seeds_fossilized": self.total_seeds_fossilized,
            "total_seeds_culled": self.total_seeds_culled,
            # Skip field_reports for now (complex nested type)
        }

    @classmethod
    def from_dict(cls, d: dict, action_enum) -> "Episode":
        """Create from dictionary."""
        return cls(
            episode_id=d["episode_id"],
            started_at=datetime.fromisoformat(d["started_at"]),
            ended_at=datetime.fromisoformat(d["ended_at"]) if d["ended_at"] else None,
            max_epochs=d["max_epochs"],
            initial_lr=d["initial_lr"],
            model_name=d["model_name"],
            dataset_name=d["dataset_name"],
            decisions=[DecisionPoint.from_dict(dp, action_enum=action_enum) for dp in d["decisions"]],
            final_accuracy=d["final_accuracy"],
            best_accuracy=d["best_accuracy"],
            total_seeds_created=d["total_seeds_created"],
            total_seeds_fossilized=d["total_seeds_fossilized"],
            total_seeds_culled=d["total_seeds_culled"],
        )

    def save(self, path: str | Path) -> None:
        """Save episode to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path, action_enum) -> "Episode":
        """Load episode from JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f), action_enum=action_enum)


# =============================================================================
# Dataset Manager
# =============================================================================

class DatasetManager:
    """Manages a directory of episode files for training."""

    def __init__(
        self,
        data_dir: str | Path = "data/simic_episodes",
        task: str = "cifar10",
        action_enum=None,
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if action_enum is not None:
            self.action_enum = action_enum
        else:
            from esper.runtime import get_task_spec  # Local import to avoid import cycle
            self.action_enum = get_task_spec(task).action_enum

    def save_episode(self, episode: Episode) -> Path:
        """Save episode to the dataset directory."""
        filename = f"{episode.episode_id}.json"
        path = self.data_dir / filename
        episode.save(path)
        return path

    def load_episode(self, episode_id: str) -> Episode:
        """Load a specific episode by ID."""
        path = self.data_dir / f"{episode_id}.json"
        return Episode.load(path, action_enum=self.action_enum)

    def list_episodes(self) -> list[str]:
        """List all episode IDs in the dataset."""
        return [p.stem for p in self.data_dir.glob("*.json")]

    def load_all(self) -> list[Episode]:
        """Load all episodes from the dataset."""
        return [Episode.load(p, action_enum=self.action_enum) for p in self.data_dir.glob("*.json")]

    def get_training_data(self) -> list[tuple[list[float], list[float], float]]:
        """Get all training data from all episodes."""
        data = []
        for episode in self.load_all():
            data.extend(episode.to_training_data())
        return data

    def summary(self) -> dict:
        """Get summary statistics of the dataset."""
        episodes = self.load_all()
        if not episodes:
            return {"count": 0}

        return {
            "count": len(episodes),
            "total_decisions": sum(len(e.decisions) for e in episodes),
            "avg_final_accuracy": sum(e.final_accuracy for e in episodes) / len(episodes),
            "avg_total_reward": sum(e.total_reward() for e in episodes) / len(episodes),
            "total_seeds_created": sum(e.total_seeds_created for e in episodes),
            "total_seeds_fossilized": sum(e.total_seeds_fossilized for e in episodes),
        }


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Observations
    "TrainingSnapshot",
    # Actions
    "ActionTaken",
    # Outcomes
    "StepOutcome",
    # Episodes
    "DecisionPoint",
    "Episode",
    # Dataset
    "DatasetManager",
]
