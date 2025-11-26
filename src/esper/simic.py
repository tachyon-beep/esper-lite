"""Simic: Policy Learning for Tamiyo

This module handles:
- Observation/outcome data structures for policy learning
- Episode collection from training runs
- Episode persistence (save/load)
- (Future) Policy network training
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any

from esper.leyline import (
    SeedStage,
    FieldReport as LeylineFieldReport,
)


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
        # Helper to clamp inf/nan to safe values
        def safe(v: float, default: float = 0.0, max_val: float = 100.0) -> float:
            import math
            if math.isnan(v) or math.isinf(v):
                return default
            return max(-max_val, min(v, max_val))

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
# Action Space
# =============================================================================

class SimicAction(Enum):
    """Discrete actions Tamiyo can take.

    Simplified from TamiyoAction for policy learning.
    """
    WAIT = 0
    GERMINATE = 1
    ADVANCE = 2  # Advance to next stage (training, blending, fossilize)
    CULL = 3


@dataclass
class ActionTaken:
    """Record of an action taken by Tamiyo."""

    action: SimicAction
    blueprint_id: str | None = None
    target_seed_id: str | None = None
    confidence: float = 1.0
    reason: str = ""

    def to_vector(self) -> list[float]:
        """Convert to flat vector (one-hot action + confidence)."""
        one_hot = [0.0, 0.0, 0.0, 0.0]
        one_hot[self.action.value] = 1.0
        return one_hot + [self.confidence]

    @staticmethod
    def vector_size() -> int:
        return 5  # 4 actions one-hot + confidence

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
    def from_dict(cls, d: dict) -> "ActionTaken":
        """Create from dictionary."""
        return cls(
            action=SimicAction[d["action"]],
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

    # Computed reward (can be customized)
    reward: float = 0.0

    def compute_reward(self) -> float:
        """Compute reward from outcome. Simple accuracy-based for now."""
        # Reward accuracy improvement, penalize drops
        self.reward = self.accuracy_change * 10.0  # Scale to [-10, 10] roughly
        return self.reward

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
    def from_dict(cls, d: dict) -> "DecisionPoint":
        """Create from dictionary."""
        return cls(
            observation=TrainingSnapshot.from_dict(d["observation"]),
            action=ActionTaken.from_dict(d["action"]),
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
    def from_dict(cls, d: dict) -> "Episode":
        """Create from dictionary."""
        return cls(
            episode_id=d["episode_id"],
            started_at=datetime.fromisoformat(d["started_at"]),
            ended_at=datetime.fromisoformat(d["ended_at"]) if d["ended_at"] else None,
            max_epochs=d["max_epochs"],
            initial_lr=d["initial_lr"],
            model_name=d["model_name"],
            dataset_name=d["dataset_name"],
            decisions=[DecisionPoint.from_dict(dp) for dp in d["decisions"]],
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
    def load(cls, path: str | Path) -> "Episode":
        """Load episode from JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


# =============================================================================
# Episode Collector
# =============================================================================

class EpisodeCollector:
    """Collects training episodes for policy learning.

    Usage:
        collector = EpisodeCollector()
        collector.start_episode(config)

        # In training loop:
        collector.record_observation(snapshot)
        collector.record_action(action)
        collector.record_outcome(outcome)

        # After training:
        episode = collector.end_episode(final_accuracy, field_reports)
    """

    def __init__(self):
        self._current_episode: Episode | None = None
        self._pending_observation: TrainingSnapshot | None = None
        self._pending_action: ActionTaken | None = None
        self._completed_episodes: list[Episode] = []

    def start_episode(
        self,
        episode_id: str,
        max_epochs: int,
        initial_lr: float = 0.001,
        model_name: str = "HostCNN",
        dataset_name: str = "CIFAR-10",
    ) -> None:
        """Start collecting a new episode."""
        self._current_episode = Episode(
            episode_id=episode_id,
            max_epochs=max_epochs,
            initial_lr=initial_lr,
            model_name=model_name,
            dataset_name=dataset_name,
        )
        self._pending_observation = None
        self._pending_action = None

    def record_observation(self, snapshot: TrainingSnapshot) -> None:
        """Record the current training state before a decision."""
        if self._current_episode is None:
            raise RuntimeError("No episode started. Call start_episode() first.")
        self._pending_observation = snapshot

    def record_action(self, action: ActionTaken) -> None:
        """Record the action taken by Tamiyo."""
        if self._pending_observation is None:
            raise RuntimeError("No observation recorded. Call record_observation() first.")
        self._pending_action = action

    def record_outcome(self, outcome: StepOutcome) -> None:
        """Record the outcome after the action."""
        if self._pending_observation is None or self._pending_action is None:
            raise RuntimeError("Must record observation and action before outcome.")

        # Compute reward
        outcome.compute_reward()

        # Create decision point
        decision = DecisionPoint(
            observation=self._pending_observation,
            action=self._pending_action,
            outcome=outcome,
        )
        self._current_episode.decisions.append(decision)

        # Clear pending state
        self._pending_observation = None
        self._pending_action = None

    def end_episode(
        self,
        final_accuracy: float,
        best_accuracy: float,
        seeds_created: int = 0,
        seeds_fossilized: int = 0,
        seeds_culled: int = 0,
        field_reports: list[LeylineFieldReport] | None = None,
    ) -> Episode:
        """Complete the episode and return it."""
        if self._current_episode is None:
            raise RuntimeError("No episode started.")

        self._current_episode.ended_at = datetime.now(timezone.utc)
        self._current_episode.final_accuracy = final_accuracy
        self._current_episode.best_accuracy = best_accuracy
        self._current_episode.total_seeds_created = seeds_created
        self._current_episode.total_seeds_fossilized = seeds_fossilized
        self._current_episode.total_seeds_culled = seeds_culled
        if field_reports:
            self._current_episode.field_reports = field_reports

        episode = self._current_episode
        self._completed_episodes.append(episode)
        self._current_episode = None

        return episode

    @property
    def episodes(self) -> list[Episode]:
        """Get all completed episodes."""
        return self._completed_episodes.copy()

    def clear(self) -> None:
        """Clear all collected episodes."""
        self._completed_episodes.clear()


# =============================================================================
# Conversion Helpers
# =============================================================================

def snapshot_from_signals(
    signals,  # TrainingSignals from tamiyo
    seed_state=None,  # SeedState from kasmina, optional
) -> TrainingSnapshot:
    """Convert Tamiyo's TrainingSignals to a Simic TrainingSnapshot."""
    # Pad history to 5 elements
    loss_hist = signals.loss_history[-5:] if signals.loss_history else []
    loss_hist = [0.0] * (5 - len(loss_hist)) + loss_hist

    acc_hist = signals.accuracy_history[-5:] if signals.accuracy_history else []
    acc_hist = [0.0] * (5 - len(acc_hist)) + acc_hist

    snapshot = TrainingSnapshot(
        epoch=signals.epoch,
        global_step=signals.global_step,
        train_loss=signals.train_loss,
        val_loss=signals.val_loss,
        loss_delta=signals.loss_delta,
        train_accuracy=signals.train_accuracy,
        val_accuracy=signals.val_accuracy,
        accuracy_delta=signals.accuracy_delta,
        plateau_epochs=signals.plateau_epochs,
        best_val_accuracy=signals.best_val_accuracy,
        loss_history_5=tuple(loss_hist),
        accuracy_history_5=tuple(acc_hist),
        available_slots=signals.available_slots,
    )

    # Add seed state if present
    if seed_state is not None:
        snapshot.has_active_seed = True
        snapshot.seed_stage = int(seed_state.stage)
        snapshot.seed_epochs_in_stage = seed_state.epochs_in_stage
        snapshot.seed_alpha = seed_state.alpha
        snapshot.seed_improvement = seed_state.metrics.improvement_since_stage_start

    return snapshot


def action_from_decision(decision) -> ActionTaken:
    """Convert Tamiyo's TamiyoDecision to a Simic ActionTaken."""
    from esper.tamiyo import TamiyoAction

    # Map TamiyoAction to SimicAction
    action_map = {
        TamiyoAction.WAIT: SimicAction.WAIT,
        TamiyoAction.GERMINATE: SimicAction.GERMINATE,
        TamiyoAction.ADVANCE_TRAINING: SimicAction.ADVANCE,
        TamiyoAction.ADVANCE_BLENDING: SimicAction.ADVANCE,
        TamiyoAction.ADVANCE_FOSSILIZE: SimicAction.ADVANCE,
        TamiyoAction.CULL: SimicAction.CULL,
        TamiyoAction.CHANGE_BLUEPRINT: SimicAction.CULL,  # Cull + new germinate
    }

    return ActionTaken(
        action=action_map.get(decision.action, SimicAction.WAIT),
        blueprint_id=decision.blueprint_id,
        target_seed_id=decision.target_seed_id,
        confidence=decision.confidence,
        reason=decision.reason,
    )


# =============================================================================
# Dataset Manager
# =============================================================================

class DatasetManager:
    """Manages a directory of episode files for training."""

    def __init__(self, data_dir: str | Path = "data/simic_episodes"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def save_episode(self, episode: Episode) -> Path:
        """Save episode to the dataset directory."""
        filename = f"{episode.episode_id}.json"
        path = self.data_dir / filename
        episode.save(path)
        return path

    def load_episode(self, episode_id: str) -> Episode:
        """Load a specific episode by ID."""
        path = self.data_dir / f"{episode_id}.json"
        return Episode.load(path)

    def list_episodes(self) -> list[str]:
        """List all episode IDs in the dataset."""
        return [p.stem for p in self.data_dir.glob("*.json")]

    def load_all(self) -> list[Episode]:
        """Load all episodes from the dataset."""
        return [Episode.load(p) for p in self.data_dir.glob("*.json")]

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
# Policy Network (requires torch)
# =============================================================================

def _check_torch():
    """Check if torch is available."""
    try:
        import torch
        return True
    except ImportError:
        return False


class PolicyNetwork:
    """Simple MLP policy for imitation learning.

    Architecture: Input(27) → Linear(64) → ReLU → Linear(32) → ReLU → Linear(4)

    Usage:
        policy = PolicyNetwork()
        policy.train_on_episodes(episodes, epochs=100)
        action = policy.predict(snapshot)
        policy.save("policy.pt")
    """

    def __init__(self, hidden1: int = 64, hidden2: int = 32, lr: float = 0.001):
        if not _check_torch():
            raise ImportError("PyTorch required for PolicyNetwork")

        import torch
        import torch.nn as nn

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Simple MLP
        self.model = nn.Sequential(
            nn.Linear(TrainingSnapshot.vector_size(), hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, len(SimicAction)),
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Track training history
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.val_accuracies: list[float] = []

    def _prepare_data(
        self, episodes: list[Episode], val_split: float = 0.2
    ) -> tuple:
        """Prepare training and validation data from episodes."""
        import torch
        import random

        # Shuffle and split by episode
        episodes = episodes.copy()
        random.shuffle(episodes)
        split_idx = int(len(episodes) * (1 - val_split))
        train_eps = episodes[:split_idx]
        val_eps = episodes[split_idx:]

        def episodes_to_tensors(eps_list):
            X, y = [], []
            for ep in eps_list:
                for dp in ep.decisions:
                    X.append(dp.observation.to_vector())
                    y.append(dp.action.action.value)
            return (
                torch.tensor(X, dtype=torch.float32, device=self.device),
                torch.tensor(y, dtype=torch.long, device=self.device),
            )

        X_train, y_train = episodes_to_tensors(train_eps)
        X_val, y_val = episodes_to_tensors(val_eps)

        return X_train, y_train, X_val, y_val

    def train_on_episodes(
        self,
        episodes: list[Episode],
        epochs: int = 100,
        batch_size: int = 32,
        val_split: float = 0.2,
        verbose: bool = True,
    ) -> dict:
        """Train the policy on collected episodes."""
        import torch

        X_train, y_train, X_val, y_val = self._prepare_data(episodes, val_split)

        if verbose:
            print(f"Training on {len(X_train)} samples, validating on {len(X_val)}")
            print(f"Device: {self.device}")

        self.train_losses.clear()
        self.val_losses.clear()
        self.val_accuracies.clear()

        for epoch in range(epochs):
            # Training
            self.model.train()
            indices = torch.randperm(len(X_train))
            total_loss = 0.0
            n_batches = 0

            for i in range(0, len(X_train), batch_size):
                batch_idx = indices[i:i + batch_size]
                X_batch = X_train[batch_idx]
                y_batch = y_train[batch_idx]

                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_train_loss = total_loss / n_batches
            self.train_losses.append(avg_train_loss)

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = self.criterion(val_outputs, y_val).item()
                predictions = val_outputs.argmax(dim=1)
                accuracy = (predictions == y_val).float().mean().item() * 100

            self.val_losses.append(val_loss)
            self.val_accuracies.append(accuracy)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}: train_loss={avg_train_loss:.4f}, "
                      f"val_loss={val_loss:.4f}, val_acc={accuracy:.1f}%")

        return {
            "final_train_loss": self.train_losses[-1],
            "final_val_loss": self.val_losses[-1],
            "final_val_accuracy": self.val_accuracies[-1],
        }

    def predict(self, snapshot: TrainingSnapshot) -> SimicAction:
        """Predict action for a single observation."""
        import torch

        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(
                [snapshot.to_vector()],
                dtype=torch.float32,
                device=self.device
            )
            output = self.model(x)
            action_idx = output.argmax(dim=1).item()

        return SimicAction(action_idx)

    def predict_probs(self, snapshot: TrainingSnapshot) -> dict[SimicAction, float]:
        """Predict action probabilities for a single observation."""
        import torch
        import torch.nn.functional as F

        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(
                [snapshot.to_vector()],
                dtype=torch.float32,
                device=self.device
            )
            output = self.model(x)
            probs = F.softmax(output, dim=1)[0]

        return {
            SimicAction(i): probs[i].item()
            for i in range(len(SimicAction))
        }

    def evaluate(self, episodes: list[Episode]) -> dict:
        """Evaluate policy on episodes, returning accuracy and confusion matrix."""
        import torch

        X, y_true = [], []
        for ep in episodes:
            for dp in ep.decisions:
                X.append(dp.observation.to_vector())
                y_true.append(dp.action.action.value)

        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_true = torch.tensor(y_true, dtype=torch.long, device=self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            y_pred = outputs.argmax(dim=1)

        # Accuracy
        accuracy = (y_pred == y_true).float().mean().item() * 100

        # Confusion matrix (4x4)
        n_classes = len(SimicAction)
        confusion = [[0] * n_classes for _ in range(n_classes)]
        for true, pred in zip(y_true.tolist(), y_pred.tolist()):
            confusion[true][pred] += 1

        # Per-class accuracy
        class_acc = {}
        for i, action in enumerate(SimicAction):
            total = sum(confusion[i])
            correct = confusion[i][i] if total > 0 else 0
            class_acc[action.name] = (correct / total * 100) if total > 0 else 0.0

        return {
            "accuracy": accuracy,
            "confusion_matrix": confusion,
            "class_accuracy": class_acc,
            "total_samples": len(y_true),
        }

    def save(self, path: str | Path) -> None:
        """Save model weights."""
        import torch
        torch.save(self.model.state_dict(), path)

    def load(self, path: str | Path) -> None:
        """Load model weights."""
        import torch
        self.model.load_state_dict(torch.load(path, map_location=self.device))


def print_confusion_matrix(eval_result: dict) -> None:
    """Pretty print a confusion matrix."""
    cm = eval_result["confusion_matrix"]
    actions = [a.name for a in SimicAction]

    # Header
    print("\nConfusion Matrix (rows=actual, cols=predicted):")
    print(f"{'':>12}", end="")
    for name in actions:
        print(f"{name:>10}", end="")
    print()

    # Rows
    for i, name in enumerate(actions):
        print(f"{name:>12}", end="")
        for j in range(len(actions)):
            print(f"{cm[i][j]:>10}", end="")
        print()

    # Per-class accuracy
    print("\nPer-class accuracy:")
    for name, acc in eval_result["class_accuracy"].items():
        print(f"  {name}: {acc:.1f}%")


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Observations
    "TrainingSnapshot",
    # Actions
    "SimicAction",
    "ActionTaken",
    # Outcomes
    "StepOutcome",
    # Episodes
    "DecisionPoint",
    "Episode",
    # Collector
    "EpisodeCollector",
    # Dataset
    "DatasetManager",
    # Policy
    "PolicyNetwork",
    "print_confusion_matrix",
    # Helpers
    "snapshot_from_signals",
    "action_from_decision",
]
